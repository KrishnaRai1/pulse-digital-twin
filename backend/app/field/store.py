"""In-memory, read-only view of the derived field store used by the API.

Daily rows live in NumPy arrays (about 60 MB for 600k well-days). A dense (well x day) index
gives O(1) access to any well's state on any day, so a 300-well field snapshot costs a few
milliseconds.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .advisor import CyclePlanner, advise_day, decide
from .build import CONDITIONS, CYCLE_FEATURES, N_FOLDS, current_build
from .case import BBL_TO_M3, FieldCase
from .features import EW_SIGNALS, early_warning_features
from .raw import PHASES

# Phase codes used in snapshots (0-2 are stored in daily.parquet)
PH_SOAK, PH_PROD, PH_WORKOVER, PH_INJECTION, PH_STOPPED = 0, 1, 2, 3, 4
PHASE_NAMES = {PH_SOAK: "soak", PH_PROD: "production", PH_WORKOVER: "workover", PH_INJECTION: "injection", PH_STOPPED: "stopped"}

FEATURE_LABELS = {
    "lv": "Load variability", "fill": "Pump fillage", "rfi": "Rod-floating index", "motor": "Motor load", "spm": "Pump speed",
    "peak": "Peak rod load", "minl": "Min rod load", "range": "Rod load range", "lv_sd7": "Load variability spread (7 d)",
    "motor_m7": "Motor load (7 d mean)", "range_m7": "Rod load range (7 d mean)", "range_trend7": "Rod load range trend (7 d)",
    "cum_float_expo": "Cumulative floating exposure", "days_floating": "Days floating since workover",
    "cum_pound_expo": "Cumulative pound exposure", "running_days_since_workover": "Running days since workover",
    "days_since_restart": "Days since pump restart", "days_since_soak_start": "Days since soak start", "cycle": "Cycle number",
}
for short, label in (("lv", "Load variability"), ("fill", "Pump fillage"), ("rfi", "Rod-floating index")):
    for k in (3, 7, 14, 30):
        FEATURE_LABELS[f"{short}_m{k}"] = f"{label} ({k} d mean)"
    FEATURE_LABELS[f"{short}_trend7"] = f"{label} trend (7 d)"


class FieldStore:
    def __init__(self, store_dir: Path, data_dir: Path, oil_price_usd_m3: float = 420.0, steam_cost_usd_m3: float = 11.0):
        build = current_build(Path(store_dir))
        if build is None:
            raise FileNotFoundError(f"no field store build in {store_dir}")
        self.dir = build
        self.data_dir = Path(data_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.case = FieldCase.load(self.data_dir / "generation_config.json")
        # memory-mapped, read-only columns: pages are loaded on demand and can be evicted by the OS
        self.d: dict[str, np.ndarray] = {p.stem: np.load(p, mmap_mode="r") for p in sorted((self.dir / "daily").glob("*.npy"))}
        self.wells: list[str] = self.meta["wells"]
        self.code = {w: i for i, w in enumerate(self.wells)}
        self.n_wells = len(self.wells)
        self.max_day = int(self.meta["dataset"]["max_day"])
        wc = self.d["well_code"].astype(np.int64)
        self.start = np.searchsorted(wc, np.arange(self.n_wells), side="left")
        self.end = np.searchsorted(wc, np.arange(self.n_wells), side="right")
        self.idx = np.full((self.n_wells, self.max_day + 1), -1, dtype=np.int32)
        self.idx[wc, self.d["day"].astype(np.int64)] = np.arange(len(wc), dtype=np.int32)
        self.cycles = pd.read_parquet(self.dir / "cycles.parquet")
        self.wells_df = pd.read_parquet(self.dir / "wells.parquet")
        self.field_daily = pd.read_parquet(self.dir / "field_daily.parquet")
        self.failures = pd.read_parquet(self.dir / "failures.parquet")
        self.failures["well_code"] = self.failures["well_id"].map(self.code)
        # injection windows and last day per well
        self.inj = np.zeros((self.n_wells, self.max_day + 2), dtype=bool)
        for w, s, n in self.cycles[["well_id", "start_day", "inj_days"]].itertuples(index=False):
            self.inj[self.code[w], s : s + n] = True
        self.last_day = self.wells_df.set_index("well_id").reindex(self.wells)["last_day"].to_numpy().astype(int)
        ew = self.meta["early_warning"]
        self.thr_alert = float(ew["threshold_alert"])
        self.thr_watch = float(ew["threshold_watch"])
        self.ew_features: list[str] = ew["features"]
        self.folds: dict[str, int] = self.meta["folds"]
        self._ew_models: dict[int, lgb.Booster] = {}
        self._lock = threading.Lock()
        cy = lgb.Booster(model_file=str(self.dir / "models" / "cycle_response.txt"))
        self.planner = CyclePlanner(cy, CYCLE_FEATURES, self.cycles, self.case, oil_price_usd_m3, steam_cost_usd_m3,
                                    log_target=bool(self.meta["cycle_model"].get("log_target", False)))

    # ------------------------------------------------------------------ helpers
    def well_code(self, well_id: str) -> int:
        w = well_id.strip().upper()
        if w not in self.code:
            raise KeyError(well_id)
        return self.code[w]

    def clamp_day(self, day: int | None) -> int:
        return self.max_day if day is None else int(min(max(day, 0), self.max_day))

    def phase_codes(self, day: int) -> np.ndarray:
        rows = self.idx[:, day]
        ph = np.where(rows >= 0, self.d["phase"][np.maximum(rows, 0)], PH_STOPPED).astype(int)
        ph = np.where((rows < 0) & self.inj[:, day], PH_INJECTION, ph)
        return ph

    def _ew_model(self, fold: int) -> lgb.Booster:
        with self._lock:
            if fold not in self._ew_models:
                self._ew_models[fold] = lgb.Booster(model_file=str(self.dir / "models" / f"early_warning_fold{fold}.txt"))
            return self._ew_models[fold]

    def risk_level(self, risk: float | None) -> str | None:
        if risk is None or not np.isfinite(risk):
            return None
        return "alert" if risk >= self.thr_alert else ("watch" if risk >= self.thr_watch else "normal")

    def _last_failure_mode(self, code: int, day: int) -> str | None:
        f = self.failures[(self.failures["well_code"] == code) & (self.failures["failure_date_day"] <= day)]
        return None if f.empty else str(f.sort_values("failure_date_day")["failure_mode"].iloc[-1])

    # ------------------------------------------------------------------ field level
    def summary(self) -> dict:
        ds = self.meta["dataset"]
        fd = self.field_daily
        prod_days = self.d["phase"] == PH_PROD
        ew = self.meta["early_warning"]["results"]
        nc = self.meta["nowcast"]["metrics_oil_rate_bbl_d"]
        cm = self.meta["cycle_model"]["metrics"]
        bt = self.meta["spm_advisor"]
        return {
            "dataset": ds,
            "built_at": self.meta["built_at"],
            "totals": {
                "oil_m3": float(fd["cum_oil_m3"].iloc[-1]),
                "steam_m3": float(fd["cum_steam_m3"].iloc[-1]),
                "cum_sor": float(fd["cum_sor"].iloc[-1]),
                "mean_oil_bbl_d_per_producing_well": float(self.d["oil_rate_bbl_day"][prod_days].mean()),
                "median_cycle_sor": float(self.cycles["sor"].median()),
                "high_sor_cycles": int(self.cycles["high_sor_flag"].sum()),
                "wells_stopped_early": int(self.wells_df["stopped_early"].sum()),
                "failures_by_mode": self.failures["failure_mode"].value_counts().to_dict(),
            },
            "headline_metrics": {
                "early_warning_pr_auc": ew["GBM (all features)"]["pr_auc"],
                "early_warning_caught_ge3d": ew["GBM (all features)"]["failures_caught_with_ge3d_warning"],
                "nowcast_mae_prior": nc["physics prior only"]["mae"],
                "nowcast_mae_twin": nc["twin: prior + GBM residual"]["mae"],
                "cycle_model_r2": cm["GBM (out-of-fold)"]["r2"],
                "advisor_floating_days_logged_pct": bt["logged"]["days_floating_pct"],
                "advisor_floating_days_pct": bt["advisor"]["days_floating_pct"],
            },
            "max_day": self.max_day,
        }

    def snapshot(self, day: int) -> dict:
        day = self.clamp_day(day)
        rows = self.idx[:, day]
        have = rows >= 0
        r = np.maximum(rows, 0)
        ph = self.phase_codes(day)
        g = lambda c: np.where(have, self.d[c][r].astype(float), np.nan)  # noqa: E731
        oil, liq, wc = g("oil_rate_bbl_day"), g("gross_liquid_bbl_day"), g("water_cut_pct")
        spm, rfi, fill, kw, risk = g("spm"), g("rod_floating_index"), g("pump_fillage_pct"), g("motor_kw"), g("risk")
        running = have & (self.d["pump_running"][r] == 1)
        prod = ph == PH_PROD
        wells = []
        n_red = n_amber = 0
        for i, w in enumerate(self.wells):
            status, reason = self._rag(i, day, ph[i], running[i], rfi[i], fill[i], spm[i], risk[i])
            n_red += status == "red"
            n_amber += status == "amber"
            wells.append({
                "well_id": w, "phase": PHASE_NAMES[int(ph[i])], "status": status, "reason": reason,
                "cycle": int(self.d["cycle"][r[i]]) if have[i] else None,
                "oil_bbl_d": _f(oil[i]) if prod[i] else 0.0, "water_cut_pct": _f(wc[i]), "spm": _f(spm[i]) if running[i] else 0.0,
                "rfi": _f(rfi[i]) if running[i] else None, "fillage_pct": _f(fill[i]) if running[i] else None,
                "motor_kw": _f(kw[i]) if running[i] else 0.0, "risk": _f(risk[i]) if running[i] else None,
            })
        fd = self.field_daily.iloc[day]
        tot_liq = float(np.nansum(np.where(prod, liq, 0)))
        tot_oil = float(np.nansum(np.where(prod, oil, 0)))
        return {
            "day": day,
            "max_day": self.max_day,
            "kpis": {
                "total_oil_bbl_d": tot_oil,
                "total_oil_m3_d": tot_oil * BBL_TO_M3,
                "total_liquid_bbl_d": tot_liq,
                "field_water_cut_pct": (1 - tot_oil / tot_liq) * 100 if tot_liq > 0 else None,
                "total_power_kw": float(np.nansum(np.where(running, kw, 0))),
                "steam_today_m3": float(fd["steam_m3"]),
                "cum_oil_m3": float(fd["cum_oil_m3"]),
                "cum_steam_m3": float(fd["cum_steam_m3"]),
                "cum_sor": _f(fd["cum_sor"]),
                "n_producing": int(prod.sum()),
                "n_injecting": int((ph == PH_INJECTION).sum()),
                "n_soaking": int((ph == PH_SOAK).sum()),
                "n_workover": int((ph == PH_WORKOVER).sum()),
                "n_stopped": int((ph == PH_STOPPED).sum()),
                "n_red": int(n_red),
                "n_amber": int(n_amber),
                "mean_spm": _f(np.nanmean(np.where(running, spm, np.nan))) if running.any() else None,
                "failures_to_date": int((self.failures["failure_date_day"] <= day).sum()),
            },
            "thresholds": {"risk_alert": self.thr_alert, "risk_watch": self.thr_watch, "rfi": 1.0, "rfi_margin": 0.9},
            "wells": wells,
        }

    def _rag(self, code: int, day: int, ph: int, running: bool, rfi: float, fill: float, spm: float, risk: float) -> tuple[str, str]:
        if ph == PH_WORKOVER:
            mode = self._last_failure_mode(code, day)
            return "red", f"workover ({mode.replace('_', ' ')})" if mode else "workover"
        if ph == PH_STOPPED:
            return "grey", "stopped (end of record / economic limit)" if day > self.last_day[code] else "not started"
        if ph in (PH_SOAK, PH_INJECTION):
            return "blue", PHASE_NAMES[ph]
        if not running:
            return "grey", "pump off"
        if np.isfinite(risk) and risk >= self.thr_alert:
            return "red", f"failure risk {risk * 100:.0f}% (14 d)"
        if rfi > 1.0:
            return "red", f"rods floating (RFI {rfi:.2f})"
        if np.isfinite(risk) and risk >= self.thr_watch:
            return "amber", f"failure risk {risk * 100:.0f}% (watch)"
        if rfi > 0.9:
            return "amber", f"near float limit (RFI {rfi:.2f})"
        if fill < 50 and spm > self.case.spm_min + 1e-6:
            return "amber", f"fluid pound (fillage {fill:.0f}%)"
        return "green", "normal"

    def trend(self, step: int = 1) -> dict:
        fd = self.field_daily.iloc[:: max(1, step)]
        cols = ["day", "oil_bbl_d", "water_bbl_d", "steam_m3", "motor_kw", "n_producing", "n_injecting", "n_soaking",
                "n_workover", "n_stopped", "n_alert", "n_floating", "failures", "cum_oil_m3", "cum_steam_m3", "cum_sor", "mean_risk"]
        return {c: _arr(fd[c].to_numpy()) for c in cols}

    def wells_table(self) -> list[dict]:
        df = self.wells_df.copy()
        return [{k: _clean(v) for k, v in rec.items()} for rec in df.to_dict("records")]

    # ------------------------------------------------------------------ well level
    def well(self, well_id: str) -> dict:
        code = self.well_code(well_id)
        w = self.wells[code]
        row = self.wells_df[self.wells_df["well_id"] == w].iloc[0].to_dict()
        cyc = self.cycles[self.cycles["well_id"] == w]
        fails = self.failures[self.failures["well_id"] == w].sort_values("failure_date_day")
        return {
            "well_id": w,
            "summary": {k: _clean(v) for k, v in row.items()},
            "cycles": [{k: _clean(v) for k, v in rec.items()} for rec in cyc.drop(columns=["well_id"]).to_dict("records")],
            "failures": [{k: _clean(v) for k, v in rec.items()} for rec in fails.drop(columns=["well_code"]).to_dict("records")],
            "first_day": 0,
            "last_day": int(self.last_day[code]),
            "fold": int(self.folds[w]),
        }

    def well_daily(self, well_id: str, cycle: int | None = None, start: int | None = None, end: int | None = None) -> dict:
        code = self.well_code(well_id)
        sl = slice(self.start[code], self.end[code])
        m = np.ones(self.end[code] - self.start[code], dtype=bool)
        if cycle is not None:
            m &= self.d["cycle"][sl] == cycle
        if start is not None:
            m &= self.d["day"][sl] >= start
        if end is not None:
            m &= self.d["day"][sl] <= end
        cols = [
            "day", "cycle", "phase", "days_since_soak_start", "oil_rate_bbl_day", "prior_oil_rate_bbl_day", "twin_oil_bbl_d",
            "water_cut_pct", "gross_liquid_bbl_day", "reservoir_temp_c", "prior_reservoir_temp_c", "oil_viscosity_cp",
            "prior_oil_viscosity_cp", "pump_running", "spm", "motor_kw", "pump_fillage_pct", "rod_floating_index",
            "polished_rod_peak_load_kn", "polished_rod_min_load_kn", "load_variability_pct", "risk", "condition_label",
        ]
        out = {c: _arr(self.d[c][sl][m]) for c in cols}
        out["phase"] = [PHASES[int(p)] for p in self.d["phase"][sl][m]]
        out["condition_label"] = [CONDITIONS[int(c)] for c in self.d["condition_label"][sl][m]]
        cyc = self.cycles[self.cycles["well_id"] == self.wells[code]]
        out["injection_windows"] = [[int(s), int(s + n - 1), int(c)] for s, n, c in cyc[["start_day", "inj_days", "cycle"]].itertuples(index=False)]
        out["thresholds"] = {"risk_alert": self.thr_alert, "risk_watch": self.thr_watch}
        return out

    def _row(self, code: int, day: int) -> dict | None:
        i = self.idx[code, day]
        if i < 0:
            return None
        row = {c: (v[i].item() if hasattr(v[i], "item") else v[i]) for c, v in self.d.items()}
        row["phase"] = PHASES[int(row["phase"])]
        return row

    def risk_drivers(self, well_id: str, day: int, top: int = 6) -> dict:
        """Failure risk on a day with the features that pushed it up or down (TreeSHAP of the
        fold model that never saw this well)."""
        code = self.well_code(well_id)
        day = self.clamp_day(day)
        sl = slice(self.start[code], self.end[code])
        run = self.d["pump_running"][sl] == 1
        days = self.d["day"][sl][run]
        if not run.any():
            return {"well_id": self.wells[code], "day": day, "available": False}
        j = int(np.searchsorted(days, day, side="right") - 1)
        if j < 0:
            return {"well_id": self.wells[code], "day": day, "available": False}
        ph = self.d["phase"][sl]
        after_wo = np.r_[False, ph[:-1] == PH_WORKOVER][run]
        x, names = early_warning_features(
            np.full(run.sum(), code), days, self.d["cycle"][sl][run], self.d["days_since_soak_start"][sl][run],
            {col: self.d[col][sl][run] for col in EW_SIGNALS}, after_wo,
        )
        model = self._ew_model(int(self.folds[self.wells[code]]))
        contrib = model.predict(x[j : j + 1], pred_contrib=True)[0]
        prob = float(model.predict(x[j : j + 1])[0])
        items = sorted(
            [{"feature": n, "label": FEATURE_LABELS.get(n, n), "value": _f(x[j, k]), "contribution": round(float(contrib[k]), 4)} for k, n in enumerate(names)],
            key=lambda t: -abs(t["contribution"]),
        )
        return {
            "well_id": self.wells[code], "day": int(days[j]), "risk": prob, "level": self.risk_level(prob),
            "base_log_odds": float(contrib[-1]), "drivers": items[:top], "fold": int(self.folds[self.wells[code]]),
            "note": "contributions in log-odds from the fold model that never saw this well (out-of-fold)",
        }

    def alerts(self, day: int, limit: int = 25) -> dict:
        day = self.clamp_day(day)
        rows = self.idx[:, day]
        have = rows >= 0
        r = np.maximum(rows, 0)
        risk = np.where(have & (self.d["pump_running"][r] == 1), self.d["risk"][r].astype(float), np.nan)
        order = np.argsort(-np.nan_to_num(risk, nan=-1))
        out = []
        for i in order[:limit]:
            if not np.isfinite(risk[i]):
                break
            dtf = float(self.d["days_to_failure"][r[i]])
            out.append({
                "well_id": self.wells[i], "risk": float(risk[i]), "level": self.risk_level(risk[i]),
                "rfi": _f(self.d["rod_floating_index"][r[i]]), "fillage_pct": _f(self.d["pump_fillage_pct"][r[i]]),
                "load_variability_pct": _f(self.d["load_variability_pct"][r[i]]), "spm": _f(self.d["spm"][r[i]]),
                "failure_within_14d": bool(self.d["label_14d"][r[i]] == 1),
                "days_to_next_failure": None if not np.isfinite(dtf) else int(dtf),
            })
        n_alert = int(np.nansum(risk >= self.thr_alert))
        n_watch = int(np.nansum((risk >= self.thr_watch) & (risk < self.thr_alert)))
        return {"day": day, "n_alert": n_alert, "n_watch": n_watch, "thresholds": {"alert": self.thr_alert, "watch": self.thr_watch}, "wells": out,
                "note": "days_to_next_failure is hindsight from the failure log, shown only to judge the warning; it is never a model input"}

    def spm_advice(self, well_id: str, day: int | None = None) -> dict:
        code = self.well_code(well_id)
        if day is None:  # latest running production day
            sl = slice(self.start[code], self.end[code])
            ok = (self.d["pump_running"][sl] == 1) & (self.d["phase"][sl] == PH_PROD)
            day = int(self.d["day"][sl][ok][-1]) if ok.any() else int(self.last_day[code])
        day = self.clamp_day(day)
        row = self._row(code, day)
        if row is None:
            ph = PHASE_NAMES[int(self.phase_codes(day)[code])]
            return {"well_id": self.wells[code], "day": day, "applicable": False, "reason": f"no pump data ({ph})"}
        days = np.arange(day, min(day + 14, self.max_day + 1))
        ii = self.idx[code, days]
        ok = ii >= 0
        outlook = {"day": days[ok].tolist(), "prior_oil_viscosity_cp": self.d["prior_oil_viscosity_cp"][ii[ok]].astype(float)}
        risk = float(row["risk"]) if row.get("risk") is not None and np.isfinite(row["risk"]) else None
        res = advise_day(self.case, row, outlook, {"risk": risk, "level": self.risk_level(risk)})
        res.update({"well_id": self.wells[code], "day": day, "cycle": int(row["cycle"]), "risk": risk, "risk_level": self.risk_level(risk)})
        return res

    def spm_history(self, well_id: str, cycle: int | None = None) -> dict:
        """Logged vs. advised SPM for every running production day of a well (vectorised advisor),
        with the rod-floating index each would give at the dataset's latent viscosity."""
        code = self.well_code(well_id)
        sl = slice(self.start[code], self.end[code])
        m = (self.d["pump_running"][sl] == 1) & (self.d["phase"][sl] == PH_PROD)
        if cycle is not None:
            m &= self.d["cycle"][sl] == cycle
        g = {c: np.asarray(self.d[c][sl][m], dtype=float) for c in ("day", "spm", "water_cut_pct", "prior_oil_viscosity_cp", "oil_viscosity_cp", "pump_fillage_pct", "gross_liquid_bbl_day", "rod_floating_index")}
        wc = g["water_cut_pct"] / 100.0
        dec = decide(self.case, g["spm"], wc, g["prior_oil_viscosity_cp"], g["pump_fillage_pct"], g["gross_liquid_bbl_day"])
        rfi_adv = self.case.rod_floating_index(dec["rec"], g["oil_viscosity_cp"], wc)
        return {
            "well_id": self.wells[code],
            "day": _arr(g["day"].astype(int)),
            "spm_logged": _arr(g["spm"]),
            "spm_advised": _arr(dec["rec"]),
            "spm_rod_float_limit": _arr(np.minimum(dec["spm_float"], 12.0)),
            "rfi_logged": _arr(g["rod_floating_index"]),
            "rfi_advised": _arr(rfi_adv),
            "binding": [str(b) for b in dec["binding"]],
            "summary": {
                "days": int(m.sum()),
                "floating_days_logged": int((g["rod_floating_index"] > 1).sum()),
                "floating_days_advised": int((rfi_adv > 1).sum()),
                "mean_spm_logged": _f(g["spm"].mean()) if m.any() else None,
                "mean_spm_advised": _f(dec["rec"].mean()) if m.any() else None,
            },
        }

    def cycle_plan(self, well_id: str, cycle: int | None, objective: str = "oil", steam_budget_m3d: float | None = None,
                   unconstrained: bool = False, quality_max: float | None = None, temp_max_c: float | None = None) -> dict:
        code = self.well_code(well_id)
        ctx = self.planner.context(self.wells[code], cycle)
        opt = self.planner.optimize(ctx, objective, steam_budget_m3d, unconstrained, quality_max, temp_max_c)
        sweeps = self.planner.sweeps(ctx, opt["best"]["controls"])
        sl = slice(self.start[code], self.end[code])
        dw = {"cycle": self.d["cycle"][sl], "phase": self.d["phase"][sl], "oil_rate_bbl_day": self.d["oil_rate_bbl_day"][sl]}
        prof = self.planner.profile(dw, ctx, [
            {"label": "recommended", "controls": opt["best"]["controls"]},
            {"label": ctx["defaults_source"], "controls": ctx["defaults"]},
        ])
        return {"context": _jsonable(ctx), "plan": _jsonable(opt), "sweeps": sweeps, "profile": prof,
                "prices": {"oil_usd_per_m3": self.planner.oil_price, "steam_usd_per_m3": self.planner.steam_cost}}

    def cycle_whatif(self, well_id: str, cycle: int | None, scenarios: list[dict]) -> dict:
        code = self.well_code(well_id)
        ctx = self.planner.context(self.wells[code], cycle)
        controls = [{**ctx["defaults"], **{k: v for k, v in s.items() if k != "label" and v is not None}} for s in scenarios]
        ev = self.planner.evaluate(ctx, controls, contributions=True)
        sl = slice(self.start[code], self.end[code])
        dw = {"cycle": self.d["cycle"][sl], "phase": self.d["phase"][sl], "oil_rate_bbl_day": self.d["oil_rate_bbl_day"][sl]}
        prof = self.planner.profile(dw, ctx, [{"label": s.get("label") or f"scenario {i + 1}", "controls": c} for i, (s, c) in enumerate(zip(scenarios, controls))])
        for e, s in zip(ev, scenarios):
            e["label"] = s.get("label")
        return {"context": _jsonable(ctx), "results": _jsonable(ev), "sweeps": self.planner.sweeps(ctx, controls[0]), "profile": prof}

    # ------------------------------------------------------------------ models
    def models(self) -> dict:
        m = self.meta
        return {
            "early_warning": {k: v for k, v in m["early_warning"].items() if k != "features"},
            "nowcast": m["nowcast"],
            "cycle_model": m["cycle_model"],
            "spm_advisor": m["spm_advisor"],
            "folds": {"n_folds": N_FOLDS, "by": "well", "wells_per_fold": [sum(1 for f in self.folds.values() if f == k) for k in range(N_FOLDS)]},
            "lightgbm": m.get("lightgbm"),
            "built_at": m["built_at"],
            "build_seconds": m["build_seconds"],
        }


# --------------------------------------------------------------------------- JSON helpers
def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, 4) if np.isfinite(x) else None


def _arr(a: np.ndarray) -> list:
    if a.dtype.kind in "iub":
        return a.astype(int).tolist()
    a = a.astype(float)
    out = np.round(a, 4)
    return [None if not np.isfinite(v) else v for v in out.tolist()]


def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 5)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return _clean(o)


__all__ = ["FieldStore", "PHASE_NAMES", "FEATURE_LABELS"]
