"""Receding-horizon optimiser (model-predictive control) for the SRP pumping speed.

At every call the controller

1. takes the viscosity / inflow forecast of the thermal twin over the next ``H`` days,
2. builds the feasible SPM set for each day from the physics safety layer
   (:mod:`limits`) - float, fatigue, structure, drive,
3. solves the finite-horizon optimal control problem *exactly* by dynamic programming
   over a 0.5-SPM grid, maximising oil revenue minus energy, wear-risk and
   movement costs subject to a rate-of-change limit,
4. returns the whole plan; only the first action is applied (receding horizon).

Dynamic programming is used instead of a gradient solver because the objective has
kinks (``min`` of inflow and pump capacity) and hard constraints; DP is exact on the
grid, deterministic and easy to audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..srp import loads
from ..srp.rods import SrpDesign
from .limits import SpmLimits, spm_limits

GRID_STEP = 0.5
MAX_RATE_UP = 1.5  # SPM per day; decreases (towards safety) are never rate limited
UNSAFE_PENALTY_USD_DAY = 5000.0


@dataclass(frozen=True)
class Economics:
    oil_usd_m3: float = 420.0
    power_usd_kwh: float = 0.09
    pound_cost_usd_day: float = 250.0
    float_risk_cost_usd_day: float = 600.0
    fatigue_cost_usd_day: float = 300.0
    move_cost_usd_per_spm: float = 4.0


@dataclass
class Forecast:
    mu_tub_pa_s: np.ndarray
    q_liquid_m3d: np.ndarray
    q_oil_m3d: np.ndarray
    net_lift_m: float

    @property
    def horizon(self) -> int:
        return len(self.mu_tub_pa_s)


@dataclass
class MpcResult:
    plan_spm: np.ndarray
    objective_usd: float
    hold_objective_usd: float
    stage: list[dict]
    hold_stage: list[dict]
    candidates: list[dict]
    envelope: dict
    limits_now: SpmLimits
    infeasible_days: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def recommended(self) -> float:
        return float(self.plan_spm[0])


def stage_terms(design: SrpDesign, econ: Economics, fc: Forecast, k: int, n: float, lim: SpmLimits | None = None) -> dict:
    """Economic and physical terms for running at ``n`` SPM on day ``k``."""
    mu = float(fc.mu_tub_pa_s[k])
    ql = max(float(fc.q_liquid_m3d[k]), 1e-6)
    f_load = design.fluid_load_n(fc.net_lift_m)
    cap = max(design.pump_capacity_m3d(n), 1e-6)
    frac = min(1.0, cap / ql)
    fill = min(1.0, ql / cap)
    oil = float(fc.q_oil_m3d[k]) * frac
    kw = loads.electrical_power_kw(loads.polished_rod_power_kw(n, mu, design, f_load))
    idx = loads.float_index(n, mu, design)
    pprl, mprl = loads.polished_rod_loads(n, mu, design, f_load)
    util = loads.goodman_utilisation(pprl, mprl, design)["utilisation"]
    revenue = oil * econ.oil_usd_m3
    energy = kw * 24.0 * econ.power_usd_kwh
    pound = econ.pound_cost_usd_day * (max(0.0, 0.85 - fill) / 0.85) ** 2
    float_pen = econ.float_risk_cost_usd_day * (max(0.0, idx - 0.70) / 0.15) ** 2
    fat_pen = econ.fatigue_cost_usd_day * (max(0.0, util - 0.70) / 0.20) ** 2
    unsafe = 0.0
    if lim is not None and n > lim.spm_max + 1e-9 and lim.feasible:
        unsafe = UNSAFE_PENALTY_USD_DAY
    reward = revenue - energy - pound - float_pen - fat_pen - unsafe
    return {
        "spm": float(n),
        "oil_m3d": oil,
        "liquid_capacity_m3d": cap,
        "fillage": fill,
        "kw": kw,
        "float_index": idx,
        "fatigue_utilisation": util,
        "structure_utilisation": loads.structure_utilisation(pprl, design),
        "pprl_kn": pprl / 1000.0,
        "mprl_kn": mprl / 1000.0,
        "revenue_usd": revenue,
        "energy_usd": energy,
        "pound_penalty_usd": pound,
        "float_penalty_usd": float_pen,
        "fatigue_penalty_usd": fat_pen,
        "unsafe_penalty_usd": unsafe,
        "reward_usd": reward,
    }


def solve(
    design: SrpDesign,
    econ: Economics,
    fc: Forecast,
    spm_now: float,
    max_rate_up: float = MAX_RATE_UP,
) -> MpcResult:
    horizon = fc.horizon
    grid = np.arange(design.min_spm, design.max_spm + 1e-9, GRID_STEP)
    ng = len(grid)
    f_load = design.fluid_load_n(fc.net_lift_m)
    lims = [spm_limits(design, float(fc.mu_tub_pa_s[k]), f_load) for k in range(horizon)]

    reward = np.full((horizon, ng), -np.inf)
    for k in range(horizon):
        feas = grid <= lims[k].spm_max + 1e-9
        if not feas.any():  # nothing is safe: keep the slowest speed and flag it
            feas = np.zeros(ng, dtype=bool)
            feas[0] = True
        for i in np.flatnonzero(feas):
            reward[k, i] = stage_terms(design, econ, fc, k, float(grid[i]))["reward_usd"]

    def move_cost(a: float, b: float) -> float:
        return econ.move_cost_usd_per_spm * abs(b - a)

    def allowed(prev: float, nxt: float) -> bool:
        return nxt <= prev + 1e-9 or (nxt - prev) <= max_rate_up + 1e-9

    # backward recursion: value[k, i] = best total reward from day k on, given n_k = grid[i]
    value = np.full((horizon + 1, ng), 0.0)
    best_next = np.zeros((horizon, ng), dtype=int)
    for k in range(horizon - 1, -1, -1):
        for i in range(ng):
            if not np.isfinite(reward[k, i]):
                value[k, i] = -np.inf
                continue
            if k == horizon - 1:
                value[k, i] = reward[k, i]
                continue
            best, arg = -np.inf, 0
            for j in range(ng):
                if not np.isfinite(reward[k + 1, j]) or not allowed(grid[i], grid[j]):
                    continue
                v = value[k + 1, j] - move_cost(grid[i], grid[j])
                if v > best:
                    best, arg = v, j
            if not np.isfinite(best):  # no allowed successor: relax the rate limit
                for j in range(ng):
                    if np.isfinite(reward[k + 1, j]):
                        v = value[k + 1, j] - move_cost(grid[i], grid[j])
                        if v > best:
                            best, arg = v, j
            value[k, i] = reward[k, i] + best
            best_next[k, i] = arg

    # first action from the (possibly off-grid) current speed
    first_best, first_i = -np.inf, 0
    for i in range(ng):
        if not np.isfinite(reward[0, i]):
            continue
        if not allowed(spm_now, grid[i]):
            continue
        v = value[0, i] - move_cost(spm_now, grid[i])
        if v > first_best:
            first_best, first_i = v, i
    if not np.isfinite(first_best):
        first_i = int(np.argmax(np.where(np.isfinite(reward[0]), value[0], -np.inf)))
        first_best = value[0, first_i]

    idx_path = [first_i]
    for k in range(horizon - 1):
        idx_path.append(int(best_next[k, idx_path[-1]]))
    plan = grid[idx_path]

    stage = [stage_terms(design, econ, fc, k, float(plan[k]), lims[k]) for k in range(horizon)]
    hold = [stage_terms(design, econ, fc, k, float(spm_now), lims[k]) for k in range(horizon)]
    objective = float(sum(s["reward_usd"] for s in stage)) - sum(
        move_cost(a, b) for a, b in zip(np.concatenate([[spm_now], plan[:-1]]), plan)
    )
    hold_obj = float(sum(s["reward_usd"] for s in hold))

    # candidates around the current speed and the recommendation (explainability)
    cand_n = sorted({round(float(x) * 2) / 2 for x in [spm_now - 2, spm_now - 1, spm_now, spm_now + 1, plan[0]]})
    cand_n = [c for c in cand_n if design.min_spm <= c <= design.max_spm]
    candidates = []
    for c in cand_n:
        t = stage_terms(design, econ, fc, 0, c, lims[0])
        t["violations"] = _violations(c, lims[0])
        candidates.append(t)

    fine = np.arange(design.min_spm, design.max_spm + 1e-9, 0.25)
    env_terms = [stage_terms(design, econ, fc, 0, float(n)) for n in fine]
    envelope = {
        "spm": fine.tolist(),
        "float_index": [t["float_index"] for t in env_terms],
        "fatigue_utilisation": [t["fatigue_utilisation"] for t in env_terms],
        "fillage": [t["fillage"] for t in env_terms],
        "oil_m3d": [t["oil_m3d"] for t in env_terms],
        "kw": [t["kw"] for t in env_terms],
        "reward_usd": [t["reward_usd"] for t in env_terms],
        "float_margin": loads.FLOAT_MARGIN,
    }
    infeasible_days = sum(1 for lm in lims if not lm.feasible)
    notes = []
    if not lims[0].feasible:
        notes.extend(lims[0].notes)
    return MpcResult(
        plan_spm=plan,
        objective_usd=objective,
        hold_objective_usd=hold_obj,
        stage=stage,
        hold_stage=hold,
        candidates=candidates,
        envelope=envelope,
        limits_now=lims[0],
        infeasible_days=infeasible_days,
        notes=notes,
    )


def _violations(n: float, lim: SpmLimits) -> list[str]:
    out = []
    if n > lim.spm_float + 1e-9:
        out.append("rod float")
    if n > lim.spm_fatigue + 1e-9:
        out.append("rod fatigue")
    if n > lim.spm_structure + 1e-9:
        out.append("unit structure")
    if n > lim.spm_drive + 1e-9:
        out.append("drive limit")
    return out
