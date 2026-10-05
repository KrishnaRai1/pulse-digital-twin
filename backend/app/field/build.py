"""Build the derived field store from the raw dataset.

    python -m scripts.build_field_store          # or automatically on API start-up

Outputs (``backend/field_store`` by default, git-ignored, rebuilt when the data changes):

* ``daily.parquet``        one row per well-day (soak, production, workover) with the joined
                           production + pump columns and model outputs (failure risk, nowcast)
* ``cycles.parquet``       one row per well-cycle: CSS controls, results, SOR, failures,
                           out-of-fold prediction of the cycle model
* ``wells.parquet``        one row per well: totals and a schematic map position
* ``field_daily.parquet``  field totals per day
* ``models/``              LightGBM models (early warning per fold, cycle response, nowcast)
* ``meta.json``            fingerprint of the data, thresholds and every evaluation metric

Every model is evaluated with 5-fold cross-validation **grouped by well** (the dataset guide's
rule 1); the guide's other rules are enforced by construction: ``condition_label`` and the
failure log are never features, and the latent ``reservoir_temp_c`` / ``oil_viscosity_cp``
are never inputs to a rate model.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import advisor
from .case import BBL_TO_M3, FieldCase
from .features import (
    NOWCAST_HORIZON,
    early_warning_features,
    failure_labels,
    log_ratio,
    nowcast_features,
)
from .metrics import average_precision, pr_curve, regression_metrics, warning_metrics
from .raw import FAILURE_MODES, PHASES, load_raw, read_manifest

log = logging.getLogger("pulse.field.build")

STORE_VERSION = 6
N_FOLDS = 5
FOLD_SEED = 42
CONDITIONS = ("normal", "shut_in", "workover") + FAILURE_MODES
CYCLE_FEATURES = [
    "steam_rate_m3d", "inj_days", "steam_quality", "steam_temp_c", "soak_days", "prod_days",
    "cycle", "prev_opd_m3d", "prev_peak_oil_bbl_d", "prev_sor", "cum_oil_before_m3",
]
CYCLE_CONTROLS = ["steam_rate_m3d", "inj_days", "steam_quality", "steam_temp_c", "soak_days", "prod_days"]
CYCLE_MONOTONE = {"steam_rate_m3d": 1, "inj_days": 1, "steam_quality": 1, "steam_temp_c": 1, "cycle": -1, "prev_opd_m3d": 1}

Progress = Callable[[str, float], None]


def _threads() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def well_folds(wells: list[str], n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> dict[str, int]:
    order = np.array(sorted(wells))
    np.random.default_rng(seed).shuffle(order)
    return {w: i % n_folds for i, w in enumerate(order)}


def schematic_position(i: int) -> tuple[str, int, int]:
    """300 wells laid out as 12 pads of 5 x 5 (schematic: the dataset has no coordinates)."""
    pad, k = divmod(i, 25)
    pr, pc = divmod(pad, 4)
    return f"P{pad + 1:02d}", pc * 6 + k % 5, pr * 6 + k // 5


# --------------------------------------------------------------------------- tables
def build_cycles(css: pd.DataFrame, prod: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    k = ["well_id", "cycle"]
    ctrl = css.groupby(k, sort=True).agg(
        start_day=("day", "min"),
        inj_days=("day", "size"),
        steam_rate_m3d=("steam_volume_m3", "first"),
        steam_quality=("steam_quality", "first"),
        steam_temp_c=("injection_temp_c", "first"),
        inj_pressure_kpa=("injection_pressure_kpa", "first"),
        soak_days_planned=("soak_time_days", "first"),
    )
    p = prod.copy()
    p["is_prod"] = p["phase"] == "production"
    p["oil_p"] = np.where(p["is_prod"], p["oil_rate_bbl_day"], 0.0)
    p["prior_p"] = np.where(p["is_prod"], p["prior_oil_rate_bbl_day"], 0.0)
    p["liq_p"] = np.where(p["is_prod"], p["gross_liquid_bbl_day"], 0.0)
    p["wc_p"] = np.where(p["is_prod"], p["water_cut_pct"], np.nan)
    res = p.groupby(k, sort=True).agg(
        soak_start_day=("day", "min"),
        end_day=("day", "max"),
        soak_days=("phase", lambda s: int((s == "soak").sum())),
        prod_rows=("is_prod", "sum"),
        workover_days=("phase", lambda s: int((s == "workover").sum())),
        oil_bbl=("oil_p", "sum"),
        prior_oil_bbl=("prior_p", "sum"),
        liquid_bbl=("liq_p", "sum"),
        peak_oil_bbl_d=("oil_p", "max"),
        avg_water_cut_pct=("wc_p", "mean"),
    )
    c = ctrl.join(res, how="inner").reset_index()
    c["prod_days"] = c["prod_rows"] + c["workover_days"]
    c["cycle_days"] = c["inj_days"] + c["soak_days"] + c["prod_days"]
    c["steam_m3"] = c["steam_rate_m3d"] * c["inj_days"]
    c["oil_m3"] = c["oil_bbl"] * BBL_TO_M3
    c["prior_oil_m3"] = c["prior_oil_bbl"] * BBL_TO_M3
    c["water_bbl"] = c["liquid_bbl"] - c["oil_bbl"]
    c["sor"] = c["steam_m3"] / c["oil_m3"].clip(lower=1e-6)
    c["osr"] = c["oil_m3"] / c["steam_m3"]
    c["opd_m3d"] = c["oil_m3"] / c["cycle_days"]
    c["high_sor_flag"] = c["sor"] > 10.0
    f = failures.groupby(k).agg(
        n_failures=("failure_mode", "size"),
        failure_modes=("failure_mode", lambda s: ",".join(sorted(set(s)))),
    )
    c = c.merge(f, on=k, how="left")
    c["n_failures"] = c["n_failures"].fillna(0).astype(int)
    c["failure_modes"] = c["failure_modes"].fillna("")
    c = c.sort_values(k).reset_index(drop=True)
    g = c.groupby("well_id", sort=False)
    c["prev_opd_m3d"] = g["opd_m3d"].shift(1)
    c["prev_peak_oil_bbl_d"] = g["peak_oil_bbl_d"].shift(1)
    c["prev_sor"] = g["sor"].shift(1)
    c["cum_oil_before_m3"] = g["oil_m3"].cumsum() - c["oil_m3"]
    c["last_cycle"] = c["cycle"] == g["cycle"].transform("max")
    c.drop(columns=["prod_rows"], inplace=True)
    return c


# --------------------------------------------------------------------------- models
def _lgb(params: dict, x: np.ndarray, y: np.ndarray, rounds: int, names: list[str], weight=None) -> lgb.Booster:
    base = {"verbosity": -1, "num_threads": _threads(), "seed": 7, "deterministic": True, "force_col_wise": True}
    ds = lgb.Dataset(x, label=y, weight=weight, feature_name=names, free_raw_data=True)
    return lgb.train({**base, **params}, ds, num_boost_round=rounds)


def train_early_warning(daily: pd.DataFrame, failures: pd.DataFrame, folds: dict[str, int], well_codes: dict[str, int], models_dir: Path, progress: Progress) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    run = daily["pump_running"].to_numpy() == 1
    d = daily.loc[run]
    well = d["well_code"].to_numpy()
    day = d["day"].to_numpy()
    prev_wo = daily["prev_phase"].to_numpy()[run] == "workover"
    x, names = early_warning_features(
        well, day, d["cycle"].to_numpy(), d["days_since_soak_start"].to_numpy(),
        {c: d[c].to_numpy() for c in ("load_variability_pct", "pump_fillage_pct", "rod_floating_index", "motor_load_pct", "spm", "polished_rod_peak_load_kn", "polished_rod_min_load_kn")},
        prev_wo,
    )
    fw = failures["well_id"].map(well_codes).to_numpy()
    fd = failures["failure_date_day"].to_numpy()
    y, dtf = failure_labels(well, day, fw, fd)
    fold = d["well_id"].map(folds).to_numpy()
    oof = np.zeros(len(y))
    params = {
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 300,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    }
    gain = np.zeros(len(names))
    fold_pr = []
    for k in range(N_FOLDS):
        progress(f"early warning: fold {k + 1}/{N_FOLDS}", 0.15 + 0.35 * k / N_FOLDS)
        tr, te = fold != k, fold == k
        m = _lgb(params, x[tr], y[tr], 250, names)
        oof[te] = m.predict(x[te])
        gain += m.feature_importance("gain")
        fold_pr.append(average_precision(y[te], oof[te]))
        m.save_model(str(models_dir / f"early_warning_fold{k}.txt"))
    gbm = warning_metrics(oof, y, well, day, fw, fd)
    col = {n: i for i, n in enumerate(names)}
    rules = {
        "rule: current rod-floating index": x[:, col["rfi"]],
        "rule: current load variability": x[:, col["lv"]],
        "rule: cumulative floating exposure": x[:, col["cum_float_expo"]],
    }
    results = {"GBM (all features)": gbm}
    curves = {"GBM (all features)": pr_curve(y, oof)}
    for label, s in rules.items():
        results[label] = warning_metrics(s.astype(float), y, well, day, fw, fd)
        curves[label] = pr_curve(y, s)
    imp = gain / max(gain.sum(), 1e-12)
    top = sorted(zip(names, imp), key=lambda t: -t[1])
    watch_thr = float(np.quantile(oof[y == 0], 0.95))
    meta = {
        "horizon_days": 14,
        "features": names,
        "threshold_alert": gbm["threshold"],
        "threshold_watch": watch_thr,
        "results": results,
        "pr_curves": curves,
        "fold_pr_auc": fold_pr,
        "top_features": [{"feature": n, "importance": round(float(v), 5)} for n, v in top[:15]],
        "n_rows": int(len(y)),
        "n_positive_rows": int(y.sum()),
        "evaluation": "out-of-fold scores, 5 folds grouped by well (every well scored by a model that never saw it)",
    }
    return meta, oof, y, dtf


def train_nowcast(daily: pd.DataFrame, cycles: pd.DataFrame, folds: dict[str, int], models_dir: Path, progress: Progress) -> tuple[dict, np.ndarray, np.ndarray]:
    prodmask = (daily["phase"] == "production").to_numpy()
    d = daily.loc[prodmask].merge(
        cycles[["well_id", "cycle", "steam_rate_m3d", "inj_days", "steam_quality", "steam_temp_c", "soak_days"]],
        on=["well_id", "cycle"], how="left",
    )
    # rows are already sorted by (well, day) and cycles are contiguous in time
    group = d["well_code"].to_numpy().astype(np.int64) * 100 + d["cycle"].to_numpy()
    lr = log_ratio(d["oil_rate_bbl_day"].to_numpy(), d["prior_oil_rate_bbl_day"].to_numpy())
    static = {
        "prior_oil_rate_bbl_day": d["prior_oil_rate_bbl_day"].to_numpy(),
        "prior_gross_liquid_bbl_day": d["prior_gross_liquid_bbl_day"].to_numpy(),
        "prior_reservoir_temp_c": d["prior_reservoir_temp_c"].to_numpy(),
        "prior_oil_viscosity_cp": d["prior_oil_viscosity_cp"].to_numpy(),
        "days_since_soak_start": d["days_since_soak_start"].to_numpy(),
        "cycle": d["cycle"].to_numpy(),
        "steam_rate_m3d": d["steam_rate_m3d"].to_numpy(),
        "inj_days": d["inj_days"].to_numpy(),
        "steam_quality": d["steam_quality"].to_numpy(),
        "steam_temp_c": d["steam_temp_c"].to_numpy(),
        "soak_days": d["soak_days"].to_numpy(),
    }
    x, names = nowcast_features(group, d["day"].to_numpy(), lr, d["water_cut_pct"].to_numpy(), static)
    fold = d["well_id"].map(folds).to_numpy()
    oof = np.zeros(len(lr))
    params = {"objective": "l2", "learning_rate": 0.06, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.9, "bagging_fraction": 0.7, "bagging_freq": 1}
    gain = np.zeros(len(names))
    for k in range(N_FOLDS):
        progress(f"thermal residual nowcast: fold {k + 1}/{N_FOLDS}", 0.52 + 0.28 * k / N_FOLDS)
        tr, te = fold != k, fold == k
        m = _lgb(params, x[tr], lr[tr], 220, names)
        oof[te] = m.predict(x[te])
        gain += m.feature_importance("gain")
    m.save_model(str(models_dir / "nowcast_last_fold.txt"))
    prior = d["prior_oil_rate_bbl_day"].to_numpy()
    obs = d["oil_rate_bbl_day"].to_numpy()
    twin = np.maximum((prior + 0.5) * np.exp(oof) - 0.5, 0.0)
    col = {n: i for i, n in enumerate(names)}
    persist_lr = np.nan_to_num(x[:, col["lr_m7"]].astype(float), nan=0.0)
    persist = np.maximum((prior + 0.5) * np.exp(persist_lr) - 0.5, 0.0)
    imp = gain / max(gain.sum(), 1e-12)
    meta = {
        "horizon_days": NOWCAST_HORIZON,
        "target": "log((oil_obs + 0.5) / (oil_prior + 0.5)) on production days",
        "features": names,
        "metrics_oil_rate_bbl_d": {
            "physics prior only": regression_metrics(obs, prior, floor=5.0),
            "persistence (last 7 d ratio)": regression_metrics(obs, persist, floor=5.0),
            "twin: prior + GBM residual": regression_metrics(obs, twin, floor=5.0),
        },
        "residual_r2": regression_metrics(lr, oof)["r2"],
        "top_features": [{"feature": n, "importance": round(float(v), 5)} for n, v in sorted(zip(names, imp), key=lambda t: -t[1])[:12]],
        "temperature_viscosity_note": (
            "Residuals of temperature and viscosity versus the physics prior are not predictable from "
            "controls or prior states (held-out R2 < 0), so the prior is used unchanged for those states; "
            "the oil-rate residual persists within a cycle (lag-1 autocorrelation ~0.94) and is corrected "
            "by assimilating production observed up to 7 days earlier."
        ),
        "n_rows": int(len(lr)),
        "evaluation": "out-of-fold, 5 folds grouped by well",
    }
    return meta, twin, prodmask


def _linear_design(c: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    first = c["prev_opd_m3d"].isna().to_numpy().astype(float)
    cols = {
        "intercept": np.ones(len(c)),
        "steam_rate_m3d": c["steam_rate_m3d"].to_numpy(),
        "inj_days": c["inj_days"].to_numpy(),
        "steam_quality": c["steam_quality"].to_numpy(),
        "steam_temp_c": c["steam_temp_c"].to_numpy(),
        "soak_days": c["soak_days"].to_numpy(),
        "soak_days_sq": c["soak_days"].to_numpy() ** 2,
        "prod_days": c["prod_days"].to_numpy(),
        "cycle": c["cycle"].to_numpy(),
        "first_cycle": first,
        "log_prev_opd": np.log(c["prev_opd_m3d"].fillna(1.0).to_numpy()) * (1 - first),
        "log_prev_peak": np.log(c["prev_peak_oil_bbl_d"].fillna(1.0).to_numpy()) * (1 - first),
        "log_prev_sor": np.log(c["prev_sor"].fillna(1.0).to_numpy()) * (1 - first),
        "cum_oil_before_1000m3": c["cum_oil_before_m3"].to_numpy() / 1000.0,
    }
    return np.column_stack(list(cols.values())).astype(float), list(cols)


def linear_cross_check(cycles: pd.DataFrame, folds: dict[str, int]) -> dict:
    """Transparent log-linear model of oil per cycle-day: per-unit effects of each control
    (holding the well's history fixed) and the soak optimum, as a cross-check of the GBM."""
    x, names = _linear_design(cycles)
    y = np.log(cycles["opd_m3d"].to_numpy())
    fold = cycles["well_id"].map(folds).to_numpy()
    oof = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr = fold != k
        b, *_ = np.linalg.lstsq(x[tr], y[tr], rcond=None)
        oof[~tr] = x[~tr] @ b
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    coef = dict(zip(names, b))
    steps = {"steam_rate_m3d": 10.0, "inj_days": 1.0, "steam_quality": 0.05, "steam_temp_c": 10.0, "prod_days": 30.0, "cycle": 1.0}
    effects = [{"control": k, "step": v, "effect_pct": round((np.exp(coef[k] * v) - 1) * 100, 2)} for k, v in steps.items()]
    soak_opt = -coef["soak_days"] / (2 * coef["soak_days_sq"]) if coef["soak_days_sq"] < 0 else None
    return {
        "r2_oof": regression_metrics(cycles["opd_m3d"].to_numpy(), np.exp(oof), floor=0.1)["r2"],
        "coefficients": {k: float(v) for k, v in coef.items()},
        "effects": effects,
        "soak_optimum_days": float(soak_opt) if soak_opt is not None else None,
    }


def train_cycle_model(cycles: pd.DataFrame, folds: dict[str, int], models_dir: Path) -> tuple[dict, np.ndarray]:
    """GBM on log(oil per cycle-day) with monotone constraints from physics: more steam (rate,
    days, quality, temperature) never lowers oil; later cycles never raise it."""
    x = cycles[CYCLE_FEATURES].to_numpy(dtype=float)
    y = cycles["opd_m3d"].to_numpy()
    ly = np.log(y)
    fold = cycles["well_id"].map(folds).to_numpy()
    params = {
        "objective": "l2", "learning_rate": 0.03, "num_leaves": 7, "min_data_in_leaf": 40, "feature_fraction": 0.9,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
        "monotone_constraints": [CYCLE_MONOTONE.get(f, 0) for f in CYCLE_FEATURES],
        "monotone_constraints_method": "advanced",
    }
    oof = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        m = _lgb(params, x[tr], ly[tr], 500, CYCLE_FEATURES)
        oof[te] = np.exp(m.predict(x[te]))
    full = _lgb(params, x, ly, 500, CYCLE_FEATURES)
    full.save_model(str(models_dir / "cycle_response.txt"))
    prev = cycles["prev_opd_m3d"].to_numpy()
    by_cycle = cycles.groupby("cycle")["opd_m3d"].transform("mean").to_numpy()
    has_prev = np.isfinite(prev)
    contrib = full.predict(x, pred_contrib=True)
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    meta = {
        "target": "log of oil per cycle-day (m3/d); cycle oil / (injection + soak + production days)",
        "log_target": True,
        "features": CYCLE_FEATURES,
        "controls": CYCLE_CONTROLS,
        "monotone_constraints": CYCLE_MONOTONE,
        "metrics": {
            "GBM (out-of-fold)": regression_metrics(y, oof, floor=0.1),
            "previous cycle repeated": regression_metrics(y[has_prev], prev[has_prev], floor=0.1),
            "field mean for cycle number": regression_metrics(y, by_cycle, floor=0.1),
        },
        "metrics_cycles_2plus": {
            "GBM (out-of-fold)": regression_metrics(y[has_prev], oof[has_prev], floor=0.1),
            "previous cycle repeated": regression_metrics(y[has_prev], prev[has_prev], floor=0.1),
        },
        "importance": sorted(
            [{"feature": f, "mean_abs_effect_pct": round(float((np.exp(v) - 1) * 100), 2)} for f, v in zip(CYCLE_FEATURES, mean_abs)],
            key=lambda t: -t["mean_abs_effect_pct"],
        ),
        "linear_cross_check": linear_cross_check(cycles, folds),
        "n_cycles": int(len(y)),
        "evaluation": "out-of-fold, 5 folds grouped by well; deployed model refit on all wells",
    }
    meta["partial_dependence"] = partial_dependence(full, cycles)
    return meta, oof


def partial_dependence(model: lgb.Booster, cycles: pd.DataFrame) -> dict:
    """Field-average response of oil per cycle-day to each control (other features as observed)."""
    base = cycles[CYCLE_FEATURES].to_numpy(dtype=float)
    grids = {
        "steam_rate_m3d": np.arange(150, 251, 10.0),
        "inj_days": np.arange(10, 21, 1.0),
        "steam_quality": np.round(np.arange(0.45, 0.701, 0.025), 3),
        "steam_temp_c": np.arange(302, 331, 4.0),
        "soak_days": np.arange(2, 15, 1.0),
        "prod_days": np.arange(120, 301, 15.0),
    }
    out = {}
    for f, grid in grids.items():
        j = CYCLE_FEATURES.index(f)
        vals = []
        for g in grid:
            xx = base.copy()
            xx[:, j] = g
            vals.append(float(np.exp(model.predict(xx)).mean()))
        out[f] = {"x": [float(v) for v in grid], "opd_m3d": [round(v, 4) for v in vals]}
    return out


# --------------------------------------------------------------------------- field aggregates
def build_field_daily(daily: pd.DataFrame, cycles: pd.DataFrame, failures: pd.DataFrame, n_wells: int, thr_alert: float, last_day_by_well: np.ndarray) -> pd.DataFrame:
    max_day = int(daily["day"].max())
    days = np.arange(max_day + 1)
    out = pd.DataFrame({"day": days})
    # injection windows from the cycle table
    inj = np.zeros(max_day + 2)
    steam = np.zeros(max_day + 2)
    for s, n, r in cycles[["start_day", "inj_days", "steam_rate_m3d"]].itertuples(index=False):
        inj[s : s + n] += 1
        steam[s : s + n] += r
    out["n_injecting"] = inj[: max_day + 1].astype(int)
    out["steam_m3"] = steam[: max_day + 1]
    run = daily["pump_running"] == 1
    g = daily.assign(
        oil=np.where(daily["phase"] == "production", daily["oil_rate_bbl_day"], 0.0),
        liq=np.where(daily["phase"] == "production", daily["gross_liquid_bbl_day"], 0.0),
        kw=daily["motor_kw"],
        prod=(daily["phase"] == "production").astype(int),
        soak=(daily["phase"] == "soak").astype(int),
        wo=(daily["phase"] == "workover").astype(int),
        alert=(run & (daily["risk"] >= thr_alert)).astype(int),
        floating=(run & (daily["rod_floating_index"] > 1.0)).astype(int),
        risk_run=np.where(run, daily["risk"], np.nan),
    ).groupby("day")
    agg = g.agg(
        oil_bbl_d=("oil", "sum"), liquid_bbl_d=("liq", "sum"), motor_kw=("kw", "sum"),
        n_producing=("prod", "sum"), n_soaking=("soak", "sum"), n_workover=("wo", "sum"),
        n_alert=("alert", "sum"), n_floating=("floating", "sum"), mean_risk=("risk_run", "mean"),
    )
    out = out.merge(agg, left_on="day", right_index=True, how="left").fillna({c: 0 for c in agg.columns if c != "mean_risk"})
    fails = failures.groupby("failure_date_day").size()
    out["failures"] = out["day"].map(fails).fillna(0).astype(int)
    out["n_stopped"] = [int((last_day_by_well < dday).sum()) for dday in days]
    out["water_bbl_d"] = out["liquid_bbl_d"] - out["oil_bbl_d"]
    out["cum_oil_m3"] = (out["oil_bbl_d"] * BBL_TO_M3).cumsum()
    out["cum_steam_m3"] = out["steam_m3"].cumsum()
    out["cum_sor"] = out["cum_steam_m3"] / out["cum_oil_m3"].replace(0, np.nan)
    return out


def build_wells(cycles: pd.DataFrame, daily: pd.DataFrame, failures: pd.DataFrame, wells: list[str]) -> pd.DataFrame:
    g = cycles.groupby("well_id")
    w = pd.DataFrame({
        "n_cycles": g["cycle"].max(),
        "last_day": g["end_day"].max(),
        "cum_oil_m3": g["oil_m3"].sum(),
        "cum_steam_m3": g["steam_m3"].sum(),
        "last_cycle_sor": g["sor"].last(),
        "best_cycle_opd_m3d": g["opd_m3d"].max(),
        "high_sor_cycles": g["high_sor_flag"].sum(),
    })
    w["cum_sor"] = w["cum_steam_m3"] / w["cum_oil_m3"]
    prodrows = daily[daily["phase"] == "production"].groupby("well_id")
    w["mean_oil_bbl_d"] = prodrows["oil_rate_bbl_day"].mean()
    runrows = daily[daily["pump_running"] == 1].groupby("well_id")
    w["pct_days_floating"] = runrows["rod_floating_index"].apply(lambda s: float((s > 1.0).mean() * 100.0))
    w["mean_spm"] = runrows["spm"].mean()
    fc = failures.groupby(["well_id", "failure_mode"]).size().unstack(fill_value=0)
    for m in FAILURE_MODES:
        w[f"failures_{m}"] = fc[m] if m in fc else 0
    w = w.fillna({f"failures_{m}": 0 for m in FAILURE_MODES})
    w["n_failures"] = sum(w[f"failures_{m}"] for m in FAILURE_MODES).astype(int)
    w = w.reindex(wells)
    w.index.name = "well_id"
    w = w.reset_index()
    pos = [schematic_position(i) for i in range(len(wells))]
    w["pad"] = [p[0] for p in pos]
    w["grid_x"] = [p[1] for p in pos]
    w["grid_y"] = [p[2] for p in pos]
    w["stopped_early"] = w["n_cycles"] < int(cycles["cycle"].max())
    return w


# --------------------------------------------------------------------------- main entry
def build_store(data_dir: Path, store_dir: Path, progress: Progress | None = None) -> dict:
    t0 = time.time()
    progress = progress or (lambda msg, frac: log.info("[%3.0f%%] %s", frac * 100, msg))
    root = Path(store_dir)
    root.mkdir(parents=True, exist_ok=True)
    # each build goes to its own folder; CURRENT is switched only after a complete build, so a
    # running API (which memory-maps the previous build) is never disturbed, on any OS
    tmp_dir = root / time.strftime("build-%Y%m%d-%H%M%S", time.gmtime())
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    (tmp_dir / "models").mkdir(parents=True, exist_ok=True)
    tmp_models = tmp_dir / "models"

    progress("loading raw files", 0.0)
    raw = load_raw(data_dir)
    case = FieldCase.load(Path(data_dir) / "generation_config.json")
    wells = sorted(raw.prod["well_id"].unique())
    well_codes = {w: i for i, w in enumerate(wells)}
    folds = well_folds(wells)

    progress("joining production and pump tables", 0.05)
    daily = raw.prod.merge(raw.srp.drop(columns=["phase"]), on=["well_id", "cycle", "day"], how="inner", validate="one_to_one")
    if len(daily) != len(raw.prod):
        raise ValueError("production_history and srp_vfd_data keys do not match one to one")
    daily = daily.sort_values(["well_id", "day"]).reset_index(drop=True)
    daily["well_code"] = daily["well_id"].map(well_codes).astype(np.int16)
    daily["prev_phase"] = daily.groupby("well_id")["phase"].shift(1)
    daily["motor_kw"] = case.motor_kw(daily["motor_load_pct"].to_numpy())

    progress("building cycle table", 0.1)
    cycles = build_cycles(raw.css, raw.prod, raw.failures)

    ew_meta, risk, label, dtf = train_early_warning(daily, raw.failures, folds, well_codes, tmp_models, progress)
    run = daily["pump_running"].to_numpy() == 1
    daily["risk"] = np.nan
    daily.loc[run, "risk"] = risk
    daily["label_14d"] = 0
    daily.loc[run, "label_14d"] = label
    daily["days_to_failure"] = np.nan
    daily.loc[run, "days_to_failure"] = dtf

    nc_meta, twin, prodmask = train_nowcast(daily, cycles, folds, tmp_models, progress)
    daily["twin_oil_bbl_d"] = np.nan
    daily.loc[prodmask, "twin_oil_bbl_d"] = twin

    progress("CSS cycle response model", 0.82)
    cy_meta, cy_oof = train_cycle_model(cycles, folds, tmp_models)
    cycles["opd_pred_oof_m3d"] = cy_oof
    cycles["fold"] = cycles["well_id"].map(folds)

    progress("SPM advisor back-test", 0.88)
    adv_meta = advisor.backtest(daily, case)

    progress("field aggregates", 0.92)
    wells_df = build_wells(cycles, daily, raw.failures, wells)
    last_day = wells_df["last_day"].to_numpy()
    field_daily = build_field_daily(daily, cycles, raw.failures, len(wells), ew_meta["threshold_alert"], last_day)
    wells_df["fold"] = wells_df["well_id"].map(folds)

    progress("writing store", 0.96)
    keep = [
        "well_code", "cycle", "day", "phase", "days_since_soak_start", "oil_rate_bbl_day", "water_cut_pct",
        "gross_liquid_bbl_day", "reservoir_temp_c", "oil_viscosity_cp", "prior_oil_rate_bbl_day",
        "prior_gross_liquid_bbl_day", "prior_reservoir_temp_c", "prior_oil_viscosity_cp", "pump_running", "spm",
        "motor_load_pct", "motor_kw", "pump_fillage_pct", "rod_floating_index", "polished_rod_peak_load_kn",
        "polished_rod_min_load_kn", "load_variability_pct", "condition_label", "risk", "label_14d", "days_to_failure",
        "twin_oil_bbl_d",
    ]
    out = daily[keep].copy()
    out["phase"] = out["phase"].map({p: i for i, p in enumerate(PHASES)}).astype(np.int8)
    out["condition_label"] = out["condition_label"].map({c: i for i, c in enumerate(CONDITIONS)}).fillna(0).astype(np.int8)
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    out["cycle"] = out["cycle"].astype(np.int8)
    out["day"] = out["day"].astype(np.int16)
    out["days_since_soak_start"] = out["days_since_soak_start"].astype(np.int16)
    out["pump_running"] = out["pump_running"].astype(np.int8)
    out["label_14d"] = out["label_14d"].astype(np.int8)
    out.to_parquet(tmp_dir / "daily.parquet", compression="zstd", index=False)  # for notebooks / export
    # one uncompressed .npy per column: the API memory-maps these, so resident memory stays small
    (tmp_dir / "daily").mkdir()
    for c in out.columns:
        np.save(tmp_dir / "daily" / f"{c}.npy", np.ascontiguousarray(out[c].to_numpy()))
    cycles.to_parquet(tmp_dir / "cycles.parquet", compression="zstd", index=False)
    wells_df.to_parquet(tmp_dir / "wells.parquet", compression="zstd", index=False)
    field_daily.to_parquet(tmp_dir / "field_daily.parquet", compression="zstd", index=False)
    raw.failures.to_parquet(tmp_dir / "failures.parquet", index=False)

    meta = {
        "store_version": STORE_VERSION,
        "fingerprint": raw.fingerprint,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - t0, 1),
        "dataset": {
            **{k: v for k, v in read_manifest(data_dir).get("dataset", {}).items()},
            "n_wells": len(wells),
            "n_cycles": int(len(cycles)),
            "n_rows": int(len(daily)),
            "n_running_days": int(run.sum()),
            "n_failures": int(len(raw.failures)),
            "max_day": int(daily["day"].max()),
        },
        "wells": wells,
        "folds": folds,
        "phases": list(PHASES),
        "conditions": list(CONDITIONS),
        "early_warning": ew_meta,
        "nowcast": nc_meta,
        "cycle_model": cy_meta,
        "spm_advisor": adv_meta,
        "lightgbm": lgb.__version__,
    }
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=float))

    switch_current(root, tmp_dir.name)
    progress(f"done in {time.time() - t0:.0f} s", 1.0)
    return meta


def current_build(root: Path) -> Path | None:
    """Folder of the active build (``<root>/CURRENT`` holds its name)."""
    ptr = Path(root) / "CURRENT"
    if not ptr.is_file():
        return None
    d = Path(root) / ptr.read_text().strip()
    return d if (d / "meta.json").is_file() else None


def switch_current(root: Path, name: str) -> None:
    tmp = Path(root) / "CURRENT.tmp"
    tmp.write_text(name)
    os.replace(tmp, Path(root) / "CURRENT")
    for d in Path(root).glob("build-*"):  # best effort: a folder still mapped by a process stays
        if d.name != name:
            shutil.rmtree(d, ignore_errors=True)


__all__ = ["build_store", "current_build", "switch_current", "build_cycles", "well_folds", "schematic_position", "STORE_VERSION", "CYCLE_FEATURES", "CYCLE_CONTROLS", "CONDITIONS"]
