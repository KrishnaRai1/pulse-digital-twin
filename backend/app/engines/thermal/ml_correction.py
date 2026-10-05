"""LightGBM residual correction of the physics baseline.

The physics model gives ``q_phys``. The ML model learns ``y = ln(q_obs / q_phys)`` from
cycle history and multiplies the baseline by ``exp(y)`` (a mobility correction; the
implied effective viscosity is ``mu_phys / exp(y)``).

Safety: the correction is clipped to a factor of 0.5x-2x so the ML layer can never
override the physics by more than a factor of two, and models are stored as LightGBM
text files (no pickle) so loading an artifact cannot execute code.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .cycle import PHASE_PRODUCTION, CycleResult, CycleSpec, simulate_cycle
from .radial import RadialGrid, Reservoir
from .synthetic_truth import observed_rate, truth_log_factor, well_hidden_factor

log = logging.getLogger("pulse.ml")

FEATURES = [
    "steam_m3",
    "quality",
    "inj_rate_m3d",
    "inj_days",
    "soak_days",
    "cycle_no",
    "t_prod_days",
    "r_heated_m",
    "t_avg_c",
    "t_wellbore_c",
    "log_mu_eff",
    "log_q_phys",
    "thickness_m",
    "perm_md",
    "depth_m",
]
MODEL_FILE = "thermal_residual_lgbm.txt"
META_FILE = "thermal_residual_meta.json"
MAX_LOG_CORRECTION = float(np.log(2.0))


def build_features(res: Reservoir, spec: CycleSpec, r: CycleResult) -> pd.DataFrame:
    """Feature rows for the production days of one simulated cycle."""
    mask = (r.phase == PHASE_PRODUCTION) & (r.q_oil_m3d > 0)
    soak_end = spec.inj_days + spec.soak_days
    t_prod = r.t_days[mask] - soak_end
    n = int(mask.sum())
    data = {
        "steam_m3": np.full(n, spec.steam_m3),
        "quality": np.full(n, spec.quality),
        "inj_rate_m3d": np.full(n, spec.inj_rate_m3d),
        "inj_days": np.full(n, spec.inj_days),
        "soak_days": np.full(n, spec.soak_days),
        "cycle_no": np.full(n, float(spec.cycle_no)),
        "t_prod_days": t_prod,
        "r_heated_m": r.r_heated_m[mask],
        "t_avg_c": r.t_avg_c[mask],
        "t_wellbore_c": r.t_wellbore_c[mask],
        "log_mu_eff": np.log(r.mu_eff_cp[mask]),
        "log_q_phys": np.log(np.maximum(r.q_oil_m3d[mask], 1e-6)),
        "thickness_m": np.full(n, res.thickness_m),
        "perm_md": np.full(n, res.perm_md),
        "depth_m": np.full(n, res.depth_m),
    }
    return pd.DataFrame(data, columns=FEATURES)


@dataclass
class MLCorrector:
    booster: lgb.Booster
    meta: dict

    # ------------------------------------------------------------------ inference
    def log_factor(self, feats: pd.DataFrame) -> np.ndarray:
        if len(feats) == 0:
            return np.zeros(0)
        raw = self.booster.predict(feats[FEATURES], num_threads=1)
        return np.clip(raw, -MAX_LOG_CORRECTION, MAX_LOG_CORRECTION)

    def contributions(self, feats: pd.DataFrame) -> list[dict]:
        """Per-feature SHAP contributions (LightGBM ``pred_contrib``) for one row."""
        row = feats[FEATURES].tail(1)
        contrib = self.booster.predict(row, pred_contrib=True, num_threads=1)[0]
        items = [
            {"feature": f, "contribution": float(c), "value": float(row.iloc[0][f])}
            for f, c in zip(FEATURES, contrib[:-1])
        ]
        items.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return items

    # ------------------------------------------------------------------ persistence
    def save(self, models_dir: Path) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(models_dir / MODEL_FILE))
        (models_dir / META_FILE).write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, models_dir: Path) -> MLCorrector | None:
        model_path = models_dir / MODEL_FILE
        if not model_path.exists():
            return None
        meta_path = models_dir / META_FILE
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(lgb.Booster(model_file=str(model_path)), meta)


def fit_residual_model(
    feats: pd.DataFrame, y: np.ndarray, groups: np.ndarray, q_phys: np.ndarray, q_obs: np.ndarray, seed: int = 7, source: str = "synthetic"
) -> MLCorrector:
    """Fit LightGBM on ln(q_obs/q_phys); hold out 20 % of cycles (groups) for metrics."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    val_groups = set(uniq[: max(1, int(0.2 * len(uniq)))])
    is_val = np.array([g in val_groups for g in groups])
    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 2,
        "verbosity": -1,
    }
    dtrain = lgb.Dataset(feats[~is_val][FEATURES], y[~is_val])
    dval = lgb.Dataset(feats[is_val][FEATURES], y[is_val], reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred = np.clip(booster.predict(feats[is_val][FEATURES], num_threads=1), -MAX_LOG_CORRECTION, MAX_LOG_CORRECTION)
    qp, qo = q_phys[is_val], q_obs[is_val]
    keep = qo > 0.5
    mape_phys = float(np.mean(np.abs(qp[keep] - qo[keep]) / qo[keep]))
    mape_ml = float(np.mean(np.abs(qp[keep] * np.exp(pred[keep]) - qo[keep]) / qo[keep]))
    ss_res = float(np.sum((y[is_val] - pred) ** 2))
    ss_tot = float(np.sum((y[is_val] - y[is_val].mean()) ** 2))
    imp = booster.feature_importance(importance_type="gain")
    meta = {
        "source": source,
        "n_rows": int(len(feats)),
        "n_cycles": int(len(uniq)),
        "best_iteration": int(booster.best_iteration),
        "holdout_mape_physics": mape_phys,
        "holdout_mape_corrected": mape_ml,
        "holdout_r2_log_factor": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "max_correction_factor": 2.0,
        "feature_importance": {f: float(v / max(imp.sum(), 1e-9)) for f, v in zip(FEATURES, imp)},
    }
    return MLCorrector(booster, meta)


def train_synthetic(models_dir: Path, n_cycles: int = 260, seed: int = 7) -> MLCorrector:
    """Generate synthetic cycle histories, apply the hidden truth, and fit the corrector."""
    rng = np.random.default_rng(seed)
    frames, ys, groups, qps, qos = [], [], [], [], []
    for gid in range(n_cycles):
        res = Reservoir(
            thickness_m=float(rng.uniform(14, 30)),
            perm_md=float(rng.uniform(50, 180)),
            t_res_c=float(rng.uniform(46, 60)),
            depth_m=float(rng.uniform(820, 1100)),
        )
        spec = CycleSpec(
            cycle_no=int(rng.integers(1, 5)),
            steam_m3=float(rng.uniform(2000, 9500)),
            quality=float(rng.uniform(0.55, 0.85)),
            inj_rate_m3d=float(rng.uniform(120, 320)),
            inj_pressure_mpa=float(rng.uniform(1.4, 2.4)),
            soak_days=float(rng.uniform(3, 14)),
            prod_days=float(rng.uniform(150, 220)),
        )
        r = simulate_cycle(res, spec, RadialGrid(res, n=60))
        f = build_features(res, spec, r)
        if len(f) < 20:
            continue
        lf = truth_log_factor(
            steam_m3=f["steam_m3"], quality=f["quality"], inj_days=f["inj_days"], cycle_no=f["cycle_no"], t_prod_days=f["t_prod_days"]
        )
        hidden = well_hidden_factor(f"syn-{gid % 40}")
        q_phys = np.exp(f["log_q_phys"].to_numpy())
        q_obs = observed_rate(q_phys, lf, hidden, rng)
        frames.append(f)
        ys.append(np.log(q_obs / q_phys))
        groups.append(np.full(len(f), gid))
        qps.append(q_phys)
        qos.append(q_obs)
    feats = pd.concat(frames, ignore_index=True)
    model = fit_residual_model(
        feats, np.concatenate(ys), np.concatenate(groups), np.concatenate(qps), np.concatenate(qos), seed=seed
    )
    model.save(models_dir)
    log.info("trained thermal residual model: %s", {k: v for k, v in model.meta.items() if k != "feature_importance"})
    return model
