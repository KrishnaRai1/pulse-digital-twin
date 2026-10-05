"""Decision support on the field dataset: SPM advisor and CSS cycle planner.

SPM advisor
-----------
Physics only (no ML), using the field-case formulas in :mod:`app.field.case`:

* **Rod-float limit**: the highest SPM whose rod-floating index stays at or below 0.9 with
  the viscosity the thermal model expects (the physics prior; the latent "true" viscosity is
  not measurable in the field) and the measured water cut.
* **Inflow match**: when the pump is not full, the well delivers less than the pump displaces;
  slowing to about 85 % fillage removes fluid pound (the cause of pump-unset failures) without
  losing liquid. When the pump is full (fillage >= 98 %) the pump is the bottleneck: speed is
  ramped up by at most 1 SPM per day (the inflow above capacity is unknown) and never past
  the rod-float limit.
* Bounds 2.0-9.6 SPM, in 0.5 SPM steps (a typical VFD set-point resolution).

The recommendation is the smaller of the two limits. Every recommendation carries the binding
constraint and the numbers behind it, so the operator can see *why* ("Reduce SPM to 5.0: rods
float above 5.3 SPM at 3,200 cP").

CSS cycle planner
-----------------
Uses the cycle-response GBM (trained by :mod:`app.field.build`) to predict oil per cycle-day
for proposed steam rate, injection days, quality, temperature, soak and production days, and
searches that grid for the best margin per cycle-day subject to the economic limit (OSR >= 0.12).
"""
from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np
import pandas as pd

from .case import BBL_TO_M3, FieldCase

RFI_TARGET = 0.9
FILL_TARGET = 0.85
PUMP_LIMITED_FILLAGE = 98.0
SPM_STEP = 0.5
RAMP_UP_SPM_PER_DAY = 1.0


# =========================================================================== SPM advisor
def decide(case: FieldCase, spm, water_cut_frac, prior_visc_cp, fillage_pct, liquid_bbl_d) -> dict[str, np.ndarray]:
    """Vectorised SPM decision. All inputs are arrays of the same length."""
    spm = np.asarray(spm, dtype=float)
    fill = np.asarray(fillage_pct, dtype=float)
    liq = np.asarray(liquid_bbl_d, dtype=float)
    spm_float_raw = case.spm_at_rfi(RFI_TARGET, prior_visc_cp, water_cut_frac)
    spm_float = np.floor(spm_float_raw / SPM_STEP) * SPM_STEP
    pump_limited = fill >= PUMP_LIMITED_FILLAGE
    spm_fill_raw = liq / (case.capacity_per_spm_bbl_d * FILL_TARGET)
    spm_fill = np.where(pump_limited, np.minimum(spm + RAMP_UP_SPM_PER_DAY, case.spm_max), np.ceil(spm_fill_raw / SPM_STEP) * SPM_STEP)
    rec = np.minimum(spm_float, spm_fill)
    rec = np.clip(rec, case.spm_min, case.spm_max)
    binding = np.where(
        spm_float < case.spm_min, "float_unavoidable",
        np.where(spm_float <= spm_fill, np.where(spm_float >= case.spm_max, "max_speed", "rod_float"),
                 np.where(pump_limited, np.where(spm_fill >= case.spm_max, "max_speed", "ramp_up"),
                          np.where(spm_fill <= case.spm_min, "min_speed", "inflow"))),
    )
    return {
        "rec": rec, "spm_float": np.clip(spm_float_raw, 0, None), "spm_fill": np.where(pump_limited, np.nan, spm_fill_raw),
        "pump_limited": pump_limited, "binding": binding, "spm": spm,
    }


def _lifted(case: FieldCase, spm_new, liq_old):
    """Liquid lifted at a new speed: never more than the pump displaces, and never more than was
    lifted before. When the pump was full the true inflow is unknown (only known to be at least
    the old rate), so speeding up is conservatively credited with no gain."""
    return np.minimum(np.asarray(liq_old, dtype=float), case.pump_capacity_bbl_d(spm_new))


def backtest(daily: pd.DataFrame, case: FieldCase) -> dict:
    """Day-by-day counterfactual on every running production day of the dataset.

    The decision uses only what is known that day (physics-prior viscosity, measured water cut,
    fillage and liquid). Its outcome is scored with the latent "true" viscosity of the dataset.
    Liquid lifted is scored conservatively (see ``_lifted``)."""
    m = (daily["pump_running"].to_numpy() == 1) & (daily["phase"].to_numpy() == "production")
    d = daily.loc[m]
    wc = d["water_cut_pct"].to_numpy() / 100.0
    spm = d["spm"].to_numpy()
    out = decide(case, spm, wc, d["prior_oil_viscosity_cp"].to_numpy(), d["pump_fillage_pct"].to_numpy(), d["gross_liquid_bbl_day"].to_numpy())
    rec = out["rec"]
    mu_true = d["oil_viscosity_cp"].to_numpy()
    rfi_log = case.rod_floating_index(spm, mu_true, wc)
    rfi_adv = case.rod_floating_index(rec, mu_true, wc)
    liq = d["gross_liquid_bbl_day"].to_numpy()
    lifted = _lifted(case, rec, liq)
    fill_log = d["pump_fillage_pct"].to_numpy()
    cap_rec = case.pump_capacity_bbl_d(rec)
    fill_adv = np.where(out["pump_limited"] & (rec <= spm), 100.0, np.minimum(100.0, 100.0 * liq / np.maximum(cap_rec, 1e-9)))
    kw = d["motor_kw"].to_numpy()
    kw_adv = kw * rec / np.maximum(spm, 1e-9)
    oil_log = d["oil_rate_bbl_day"].to_numpy()
    oil_adv = lifted * (1.0 - wc)
    n = len(d)

    def pct(x):
        return round(float(np.mean(x) * 100.0), 2)

    changes = rec - spm
    return {
        "rfi_target": RFI_TARGET,
        "fillage_target_pct": FILL_TARGET * 100,
        "n_days": int(n),
        "logged": {
            "days_floating_pct": pct(rfi_log > 1.0),
            "float_exposure_sum": round(float(np.maximum(rfi_log - 1, 0).sum()), 1),
            "days_low_fillage_pct": pct(fill_log < 50.0),
            "days_pound_above_min_speed_pct": pct((fill_log < 50.0) & (spm > case.spm_min + 1e-6)),
            "oil_bbl": round(float(oil_log.sum()), 0),
            "energy_mwh": round(float(kw.sum() * 24 / 1000.0), 1),
            "mean_spm": round(float(spm.mean()), 2),
        },
        "advisor": {
            "days_floating_pct": pct(rfi_adv > 1.0),
            "float_exposure_sum": round(float(np.maximum(rfi_adv - 1, 0).sum()), 1),
            "days_low_fillage_pct": pct(fill_adv < 50.0),
            "days_pound_above_min_speed_pct": pct((fill_adv < 50.0) & (rec > case.spm_min + 1e-6)),
            "oil_bbl": round(float(oil_adv.sum()), 0),
            "energy_mwh": round(float(kw_adv.sum() * 24 / 1000.0), 1),
            "mean_spm": round(float(rec.mean()), 2),
        },
        "share_of_days": {
            "reduce": pct(changes <= -0.25),
            "hold": pct(np.abs(changes) < 0.25),
            "increase": pct(changes >= 0.25),
            "float_unavoidable": pct(out["binding"] == "float_unavoidable"),
        },
        "assumptions": [
            "decision uses the physics-prior viscosity and measured water cut of the same day",
            "outcome is scored with the dataset's latent viscosity (what the rods would actually see)",
            "inflow equals logged liquid when the pump is not full; when full, no gain is credited for speeding up",
            "low fillage at the minimum speed is unavoidable with this pump size (2.0 SPM displaces 72 bbl/d)",
            "motor power scales in proportion to SPM (load x speed); one-day decisions, no carry-over effects",
        ],
    }


def advise_day(case: FieldCase, row: dict, outlook: dict | None = None, risk: dict | None = None) -> dict:
    """Recommendation with explanation for one well-day (``row`` holds the daily columns)."""
    running = int(row.get("pump_running", 0)) == 1 and row.get("phase") == "production"
    if not running:
        return {"applicable": False, "reason": f"pump is off ({row.get('phase')})"}
    spm = float(row["spm"])
    wc = float(row["water_cut_pct"]) / 100.0
    mu_prior = float(row["prior_oil_viscosity_cp"])
    mu_true = float(row["oil_viscosity_cp"])
    fill = float(row["pump_fillage_pct"])
    liq = float(row["gross_liquid_bbl_day"])
    kw = float(row["motor_kw"])
    dec = {k: v[0] for k, v in decide(case, [spm], [wc], [mu_prior], [fill], [liq]).items()}
    rec = float(dec["rec"])
    mu_mix = float(case.mixture_viscosity_cp(mu_prior, wc))
    v_fall = float(case.rod_fall_speed_m_s(mu_mix))
    v_rod_now = float(case.rod_speed_m_s(spm))
    rfi_now_model = v_rod_now / v_fall
    rfi_new = float(case.rod_speed_m_s(rec)) / v_fall
    cap_now = float(case.pump_capacity_bbl_d(spm))
    cap_new = float(case.pump_capacity_bbl_d(rec))
    lifted_new = float(_lifted(case, rec, liq))
    fill_new = 100.0 if (dec["pump_limited"] and rec <= spm) else min(100.0, 100.0 * liq / max(cap_new, 1e-9))

    delta = rec - spm
    action = "hold" if abs(delta) < 0.25 else ("reduce" if delta < 0 else "increase")
    headline = {
        "hold": f"Hold SPM at {spm:.1f}",
        "reduce": f"Reduce SPM to {rec:.1f}",
        "increase": f"Increase SPM to {rec:.1f}",
    }[action]

    reasons: list[dict] = []
    if rfi_now_model > 1.0:
        reasons.append({
            "severity": "high", "code": "rod_float",
            "text": (f"Rods are floating: RFI {rfi_now_model:.2f} at {spm:.1f} SPM. Oil at {mu_prior:,.0f} cP with "
                     f"{wc * 100:.0f}% water cut gives a {mu_mix:,.0f} cP fluid; rods fall at {v_fall:.2f} m/s but the "
                     f"polished rod travels at {v_rod_now:.2f} m/s (impact loading, rod-part risk)."),
        })
    elif rfi_now_model > RFI_TARGET:
        reasons.append({"severity": "medium", "code": "rod_float_margin",
                        "text": f"RFI {rfi_now_model:.2f} is inside the 10% safety margin below 1.0 (rod-fall speed {v_fall:.2f} m/s)."})
    if not dec["pump_limited"] and fill < 60.0:
        reasons.append({"severity": "high" if fill < 45 else "medium", "code": "fluid_pound",
                        "text": (f"Pump fillage {fill:.0f}%: the pump displaces {cap_now:.0f} bbl/d but the well delivers "
                                 f"about {liq:.0f} bbl/d, so the plunger hits fluid (fluid pound, pump-unset risk).")})
    if dec["pump_limited"]:
        reasons.append({"severity": "info", "code": "pump_limited",
                        "text": f"Pump is full ({fill:.0f}% fillage): liquid is limited by pump speed, not by the reservoir."})
    if dec["binding"] == "rod_float":
        reasons.append({"severity": "info", "code": "limit",
                        "text": f"Rod-float limit: RFI reaches {RFI_TARGET} at {float(dec['spm_float']):.2f} SPM; set-point rounded down to {rec:.1f}."})
    elif dec["binding"] == "inflow":
        reasons.append({"severity": "info", "code": "limit",
                        "text": f"Inflow match: {liq:.0f} bbl/d at {FILL_TARGET * 100:.0f}% fillage needs {float(dec['spm_fill']):.2f} SPM; rounded up to {rec:.1f}."})
    elif dec["binding"] == "ramp_up":
        reasons.append({"severity": "info", "code": "limit",
                        "text": (f"Pump-limited: ramp up {RAMP_UP_SPM_PER_DAY:.1f} SPM per day while the pump stays full; "
                                 f"the rod-float limit ({float(dec['spm_float']):.1f} SPM) is not reached.")})
    elif dec["binding"] == "float_unavoidable":
        reasons.append({"severity": "high", "code": "float_unavoidable",
                        "text": (f"Even the minimum speed ({case.spm_min:.1f} SPM) floats the rods at this viscosity. Run at minimum "
                                 "speed, watch rod loads, and plan the next steam cycle (or diluent) early.")})
    if risk and risk.get("risk") is not None and risk.get("level") in ("alert", "watch"):
        reasons.append({"severity": "high" if risk["level"] == "alert" else "medium", "code": "early_warning",
                        "text": f"Early-warning model: {risk['risk'] * 100:.1f}% chance of a rod/pump failure within 14 days ({risk['level']})."})

    curve_spm = np.round(np.arange(case.spm_min, case.spm_max + 1e-9, 0.1), 2)
    curve = {
        "spm": curve_spm.tolist(),
        "rfi": np.round(case.rod_speed_m_s(curve_spm) / v_fall, 4).tolist(),
        "capacity_bbl_d": np.round(case.pump_capacity_bbl_d(curve_spm), 2).tolist(),
        "lifted_bbl_d": np.round(_lifted(case, curve_spm, liq), 2).tolist(),
        "inflow_known": not bool(dec["pump_limited"]),
    }
    result = {
        "applicable": True,
        "action": action,
        "headline": headline,
        "current_spm": spm,
        "recommended_spm": rec,
        "binding_constraint": str(dec["binding"]),
        "limits": {"spm_rod_float": round(float(dec["spm_float"]), 3), "spm_inflow": None if dec["pump_limited"] else round(float(dec["spm_fill"]), 3),
                   "spm_min": case.spm_min, "spm_max": case.spm_max, "rfi_target": RFI_TARGET, "fillage_target_pct": FILL_TARGET * 100},
        "state": {
            "oil_viscosity_prior_cp": mu_prior, "oil_viscosity_latent_cp": mu_true, "water_cut_pct": wc * 100, "mixture_viscosity_cp": mu_mix,
            "rod_fall_speed_m_s": v_fall, "rod_speed_m_s": v_rod_now, "rfi_model": rfi_now_model, "rfi_logged": float(row["rod_floating_index"]),
            "fillage_pct": fill, "liquid_bbl_d": liq, "oil_bbl_d": float(row["oil_rate_bbl_day"]), "capacity_bbl_d": cap_now,
            "motor_kw": kw, "vfd_hz": float(case.vfd_hz(spm)), "pump_limited": bool(dec["pump_limited"]),
        },
        "expected": {
            "rfi": rfi_new, "fillage_pct": fill_new, "liquid_bbl_d": lifted_new, "oil_bbl_d": lifted_new * (1 - wc),
            "motor_kw": kw * rec / max(spm, 1e-9), "vfd_hz": float(case.vfd_hz(rec)),
            "note": "liquid gain is not credited when the pump is full (inflow unknown above capacity)" if dec["pump_limited"] and rec > spm else None,
        },
        "reasons": reasons,
        "curve": curve,
    }
    if outlook:
        mu_next = np.asarray(outlook["prior_oil_viscosity_cp"], dtype=float)
        lim = np.floor(case.spm_at_rfi(RFI_TARGET, mu_next, wc) / SPM_STEP) * SPM_STEP
        fill_lim = min(spm + RAMP_UP_SPM_PER_DAY, case.spm_max) if dec["pump_limited"] else math.ceil(liq / (case.capacity_per_spm_bbl_d * FILL_TARGET) / SPM_STEP) * SPM_STEP
        plan = np.clip(np.minimum(lim, fill_lim), case.spm_min, case.spm_max)
        result["outlook"] = {
            "day": list(outlook["day"]),
            "prior_oil_viscosity_cp": [round(float(v), 1) for v in mu_next],
            "spm_rod_float_limit": [round(float(v), 2) for v in case.spm_at_rfi(RFI_TARGET, mu_next, wc)],
            "spm_plan": [float(v) for v in plan],
            "note": "physics-prior viscosity for the coming days at today's water cut and inflow",
        }
    return result


# =========================================================================== CSS cycle planner
class CyclePlanner:
    """What-if and optimisation of CSS cycle controls with the cycle-response GBM."""

    def __init__(self, booster: lgb.Booster, features: list[str], cycles: pd.DataFrame, case: FieldCase, oil_price_usd_m3: float,
                 steam_cost_usd_m3: float, log_target: bool = True):
        self.booster = booster
        self.log_target = log_target
        self.features = features
        self.cycles = cycles
        self.case = case
        self.oil_price = oil_price_usd_m3
        self.steam_cost = steam_cost_usd_m3

    # ------------------------------------------------------------------ context
    def context(self, well_id: str, cycle: int | None = None) -> dict:
        wc = self.cycles[self.cycles["well_id"] == well_id].sort_values("cycle")
        if wc.empty:
            raise KeyError(well_id)
        last = int(wc["cycle"].max())
        target = last + 1 if cycle is None else int(cycle)
        if target < 1 or target > last + 1:
            raise ValueError(f"cycle must be between 1 and {last + 1}")
        prev = wc[wc["cycle"] == target - 1]
        actual = wc[wc["cycle"] == target]
        before = wc[wc["cycle"] < target]
        ctx = {
            "well_id": well_id,
            "cycle": target,
            "is_next_cycle": target == last + 1,
            "prev_opd_m3d": float(prev["opd_m3d"].iloc[0]) if len(prev) else np.nan,
            "prev_peak_oil_bbl_d": float(prev["peak_oil_bbl_d"].iloc[0]) if len(prev) else np.nan,
            "prev_sor": float(prev["sor"].iloc[0]) if len(prev) else np.nan,
            "cum_oil_before_m3": float(before["oil_m3"].sum()),
        }
        ref = actual.iloc[0] if len(actual) else (prev.iloc[0] if len(prev) else wc.iloc[-1])
        ctx["defaults"] = {
            "steam_rate_m3d": float(ref["steam_rate_m3d"]),
            "inj_days": int(ref["inj_days"]),
            "steam_quality": float(ref["steam_quality"]),
            "steam_temp_c": float(ref["steam_temp_c"]),
            "soak_days": int(ref["soak_days"]),
            "prod_days": int(ref["prod_days"]),
        }
        ctx["defaults_source"] = "actual cycle" if len(actual) else f"cycle {int(ref['cycle'])} settings"
        if len(actual):
            a = actual.iloc[0]
            ctx["actual"] = {"opd_m3d": float(a["opd_m3d"]), "oil_m3": float(a["oil_m3"]), "sor": float(a["sor"]), "opd_pred_oof_m3d": float(a["opd_pred_oof_m3d"])}
        shape_src = wc[wc["cycle"] < target]
        ctx["shape_cycle"] = int(shape_src["cycle"].iloc[-1]) if len(shape_src) else int(wc["cycle"].iloc[0])
        return ctx

    def _matrix(self, ctx: dict, controls: list[dict]) -> np.ndarray:
        rows = []
        for c in controls:
            r = {**{k: ctx[k] for k in ("prev_opd_m3d", "prev_peak_oil_bbl_d", "prev_sor", "cum_oil_before_m3")}, "cycle": ctx["cycle"], **c}
            rows.append([float(r[f]) for f in self.features])
        return np.asarray(rows, dtype=float)

    def evaluate(self, ctx: dict, controls: list[dict], contributions: bool = False) -> list[dict]:
        x = self._matrix(ctx, controls)
        raw = self.booster.predict(x)
        opd = np.exp(raw) if self.log_target else np.maximum(raw, 0.0)
        contrib = self.booster.predict(x, pred_contrib=True) if contributions else None
        out = []
        for i, c in enumerate(controls):
            days = c["inj_days"] + c["soak_days"] + c["prod_days"]
            steam = c["steam_rate_m3d"] * c["inj_days"]
            oil = float(opd[i] * days)
            res = {
                "controls": c,
                "opd_m3d": float(opd[i]),
                "oil_m3": oil,
                "oil_bbl": oil / BBL_TO_M3,
                "steam_m3": steam,
                "cycle_days": days,
                "sor": steam / max(oil, 1e-9),
                "osr": oil / steam,
                "steam_per_cycle_day_m3": steam / days,
                "net_usd_per_day": (oil * self.oil_price - steam * self.steam_cost) / days,
                "economic": oil / steam >= self.case.min_osr,
            }
            if contrib is not None:
                # contributions are in log space: report them as % effect on oil per cycle-day
                res["contributions"] = sorted(
                    [{"feature": f, "log_effect": round(float(v), 5), "effect_pct": round(float(np.expm1(v) * 100), 2)} for f, v in zip(self.features, contrib[i, :-1])],
                    key=lambda t: -abs(t["log_effect"]),
                )
                res["base_m3d"] = float(np.exp(contrib[i, -1])) if self.log_target else float(contrib[i, -1])
            out.append(res)
        return out

    def sweeps(self, ctx: dict, controls: dict) -> dict:
        grids = {
            "steam_rate_m3d": np.arange(150, 251, 10.0),
            "inj_days": np.arange(10, 21, 1.0),
            "soak_days": np.arange(2, 15, 1.0),
            "steam_quality": np.round(np.arange(0.45, 0.701, 0.05), 3),
            "steam_temp_c": np.arange(302, 331, 4.0),
            "prod_days": np.arange(120, 301, 20.0),
        }
        out = {}
        for f, grid in grids.items():
            rows = [{**controls, f: float(g)} for g in grid]
            ev = self.evaluate(ctx, rows)
            out[f] = {"x": [float(g) for g in grid], "opd_m3d": [round(e["opd_m3d"], 4) for e in ev],
                      "sor": [round(e["sor"], 3) for e in ev], "net_usd_per_day": [round(e["net_usd_per_day"], 1) for e in ev]}
        return out

    def optimize(
        self,
        ctx: dict,
        objective: str = "oil",
        steam_budget_m3d: float | None = None,
        unconstrained: bool = False,
        quality_max: float | None = None,
        temp_max_c: float | None = None,
    ) -> dict:
        """Grid search over the controls the data covers.

        Steam generators deliver a rate, shared across the field, so the default constraint is
        the steam *per cycle-day* of the reference cycle (``steam_budget_m3d``): a longer cycle may
        use more steam, a shorter one less. The planner then finds the best way to spend it
        (rate vs. days, soak, production length). Steam quality and temperature have a positive
        effect in the data, so they are set to the generator maximum given."""
        q = float(quality_max) if quality_max else self.case.steam_quality_range[1]
        t = float(temp_max_c) if temp_max_c else self.case.steam_temp_range_c[1]
        if unconstrained:
            budget = None
        else:
            dflt = ctx["defaults"]
            budget = float(steam_budget_m3d) if steam_budget_m3d else float(
                dflt["steam_rate_m3d"] * dflt["inj_days"] / (dflt["inj_days"] + dflt["soak_days"] + dflt["prod_days"]))
        grid = [
            {"steam_rate_m3d": float(r), "inj_days": int(n), "steam_quality": q, "steam_temp_c": t, "soak_days": int(sk), "prod_days": int(p)}
            for r in range(150, 251, 10)
            for n in range(10, 21)
            for sk in range(2, 15)
            for p in range(120, 301, 15)
            if budget is None or r * n / (n + sk + p) <= budget + 1e-9
        ]
        if not grid:  # budget below the smallest cycle the data covers
            grid = [{"steam_rate_m3d": 150.0, "inj_days": 10, "steam_quality": q, "steam_temp_c": t, "soak_days": int(sk), "prod_days": int(p)}
                    for sk in range(2, 15) for p in range(120, 301, 30)]
        ev = self.evaluate(ctx, grid)
        keys = {
            "margin": lambda e: e["net_usd_per_day"],
            "oil": lambda e: e["opd_m3d"],
            "osr": lambda e: e["osr"],
        }
        key = keys.get(objective, keys["oil"])
        feasible = [e for e in ev if e["economic"]]
        pool = sorted(feasible or ev, key=key, reverse=True)
        best = self.evaluate(ctx, [pool[0]["controls"]], contributions=True)[0]
        base = self.evaluate(ctx, [ctx["defaults"]], contributions=True)[0]
        bc = {c["feature"]: c["log_effect"] for c in base["contributions"]}
        why = []
        other = 0.0
        for c in best["contributions"]:
            dv = c["log_effect"] - bc.get(c["feature"], 0.0)
            if c["feature"] in base["controls"] and base["controls"][c["feature"]] != best["controls"][c["feature"]]:
                why.append({"feature": c["feature"], "effect_pct": round(float(np.expm1(dv) * 100), 2),
                            "from": base["controls"][c["feature"]], "to": best["controls"][c["feature"]]})
            else:
                other += dv
        why.sort(key=lambda w: -abs(w["effect_pct"]))
        if abs(other) > 1e-3:
            why.append({"feature": "interactions", "effect_pct": round(float(np.expm1(other) * 100), 2), "from": None, "to": None})
        # oil-vs-steam frontier: best outcome for every steam total in the grid
        frontier: dict[float, dict] = {}
        for e in ev:
            k = round(e["steam_m3"], 1)
            if k not in frontier or key(e) > key(frontier[k]):
                frontier[k] = e
        front = [{"steam_m3": k, "oil_m3": round(v["oil_m3"], 1), "opd_m3d": round(v["opd_m3d"], 4), "sor": round(v["sor"], 3)}
                 for k, v in sorted(frontier.items())]
        return {
            "objective": objective,
            "steam_budget_m3_per_cycle_day": budget,
            "quality": q,
            "temp_c": t,
            "searched": len(grid),
            "feasible": len(feasible),
            "best": best,
            "baseline": base,
            "uplift_pct": (best["opd_m3d"] / base["opd_m3d"] - 1.0) * 100.0 if base["opd_m3d"] > 0 else None,
            "why": why,
            "alternatives": pool[1:6],
            "frontier": front,
        }

    def profile(self, daily_well: dict, ctx: dict, scenarios: list[dict]) -> dict:
        """Expected cumulative oil vs. day for scenarios, using the shape of a reference cycle."""
        cyc = np.asarray(daily_well["cycle"])
        ph = np.asarray(daily_well["phase"])
        oil = np.asarray(daily_well["oil_rate_bbl_day"], dtype=float)
        m = (cyc == ctx["shape_cycle"]) & (ph == 1)
        shape = np.cumsum(oil[m]) if m.any() else np.linspace(0, 1, 100)
        shape = shape / max(shape[-1], 1e-9)
        tau = np.linspace(0, 1, len(shape))
        curves = []
        for sc in scenarios:
            ev = self.evaluate(ctx, [sc["controls"]])[0]
            c = sc["controls"]
            lead = int(c["inj_days"] + c["soak_days"])
            n = int(c["prod_days"])
            t = np.arange(lead + n + 1)
            cum = np.zeros(len(t))
            tt = np.clip((t - lead) / max(n, 1), 0, 1)
            cum[t > lead] = np.interp(tt[t > lead], tau, shape) * ev["oil_m3"]
            step = max(1, len(t) // 120)
            curves.append({"label": sc.get("label") or "", "day": t[::step].tolist(), "cum_oil_m3": np.round(cum[::step], 1).tolist(), "oil_m3": ev["oil_m3"], "sor": ev["sor"]})
        return {"shape_cycle": ctx["shape_cycle"], "curves": curves}


__all__ = ["decide", "backtest", "advise_day", "CyclePlanner", "RFI_TARGET", "FILL_TARGET"]
