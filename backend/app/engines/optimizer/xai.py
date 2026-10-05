"""Explainable-AI layer for the optimiser.

Everything here is derived from the numbers the optimiser actually used - no canned
prose. The output has three parts the UI renders:

* ``drivers``       ranked, quantified reasons for the recommendation
* ``contributions`` $/day decomposition of "recommended vs. keep current"
* ``narrative``     one operator-readable sentence assembled from the top drivers
"""
from __future__ import annotations

import numpy as np

from .mpc import MpcResult


def _pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b if b else 0.0


def _uplift_per_day(res: MpcResult) -> float:
    """Average economic gain per day versus holding speed.

    The optimiser's internal objective also carries a large *constraint* penalty for leaving the
    safety envelope. That is a mathematical device, not money, so it is removed here: the figure
    shown to operators only contains oil revenue, energy and the modelled wear/risk costs."""
    n = max(len(res.plan_spm), 1)
    plan_sum = sum(t["reward_usd"] for t in res.stage)
    moves = plan_sum - res.objective_usd
    plan_val = sum(t["reward_usd"] + t["unsafe_penalty_usd"] for t in res.stage) - moves
    hold_val = sum(t["reward_usd"] + t["unsafe_penalty_usd"] for t in res.hold_stage)
    return (plan_val - hold_val) / n


def explain(
    res: MpcResult,
    spm_now: float,
    mu_now_cp: float,
    mu_24h_ago_cp: float,
    mu_7d_cp: float,
    ml_top: list[dict] | None = None,
) -> dict:
    rec = res.recommended
    lim = res.limits_now
    cur = res.hold_stage[0]
    new = res.stage[0]
    drivers: list[dict] = []

    visc_delta = _pct(mu_now_cp, mu_24h_ago_cp)
    drivers.append(
        {
            "factor": "Tubing viscosity trend",
            "detail": f"{mu_now_cp:,.0f} cP now, {visc_delta:+.0f}% vs 24 h ago, {_pct(mu_7d_cp, mu_now_cp):+.0f}% expected in 7 days",
            "effect": "lowers SPM ceiling" if visc_delta > 1 else "neutral",
            "weight": min(1.0, abs(visc_delta) / 25.0),
        }
    )
    drivers.append(
        {
            "factor": f"Binding limit: {lim.binding.replace('_', ' ')}",
            "detail": (
                f"ceiling {lim.spm_max:.2f} SPM (float {lim.spm_float:.2f}, fatigue {lim.spm_fatigue:.2f}, "
                f"structure {lim.spm_structure:.2f}, drive {lim.spm_drive:.1f}); current {spm_now:.1f} SPM"
            ),
            "effect": "current speed exceeds ceiling" if spm_now > lim.spm_max else "current speed inside envelope",
            "weight": 1.0 if spm_now > lim.spm_max else 0.35,
        }
    )
    drivers.append(
        {
            "factor": "Inflow vs pump capacity",
            "detail": (
                f"pump fillage {cur['fillage'] * 100:.0f}% at {spm_now:.1f} SPM -> {new['fillage'] * 100:.0f}% at {rec:.1f} SPM "
                f"(capacity {new['liquid_capacity_m3d']:.0f} m3/d)"
            ),
            "effect": "avoid fluid pound" if new["fillage"] > cur["fillage"] else "keeps production",
            "weight": float(np.clip(abs(new["fillage"] - cur["fillage"]) * 2.0, 0.1, 1.0)),
        }
    )
    drivers.append(
        {
            "factor": "Rod loading",
            "detail": (
                f"float index {cur['float_index']:.2f} -> {new['float_index']:.2f} (limit 0.85); fatigue "
                f"{cur['fatigue_utilisation'] * 100:.0f}% -> {new['fatigue_utilisation'] * 100:.0f}% of modified-Goodman allowable"
            ),
            "effect": "reduces impact loading" if new["float_index"] < cur["float_index"] else "unchanged",
            "weight": float(np.clip(abs(new["float_index"] - cur["float_index"]) * 2.0, 0.1, 1.0)),
        }
    )
    drivers.append(
        {
            "factor": "Energy",
            "detail": f"{cur['kw']:.1f} kW -> {new['kw']:.1f} kW ({(new['kw'] - cur['kw']) * 24:+.0f} kWh/day)",
            "effect": "saves energy" if new["kw"] < cur["kw"] else "uses more energy",
            "weight": float(np.clip(abs(new["kw"] - cur["kw"]) / 10.0, 0.05, 0.8)),
        }
    )
    drivers.sort(key=lambda d: d["weight"], reverse=True)

    contributions = [
        {"name": "Oil revenue", "usd_per_day": new["revenue_usd"] - cur["revenue_usd"]},
        {"name": "Energy cost", "usd_per_day": -(new["energy_usd"] - cur["energy_usd"])},
        {
            "name": "Pump-off wear",
            "usd_per_day": -(new["pound_penalty_usd"] - cur["pound_penalty_usd"]),
        },
        {
            "name": "Rod-float risk (modelled)",
            "usd_per_day": -(new["float_penalty_usd"] - cur["float_penalty_usd"]),
        },
        {
            "name": "Rod fatigue",
            "usd_per_day": -(new["fatigue_penalty_usd"] - cur["fatigue_penalty_usd"]),
        },
    ]

    if abs(rec - spm_now) < 0.25:
        action = f"Hold {spm_now:.1f} SPM"
    elif rec < spm_now:
        action = f"Reduce SPM from {spm_now:.1f} to {rec:.1f}"
    else:
        action = f"Increase SPM from {spm_now:.1f} to {rec:.1f}"

    reasons = []
    if spm_now > lim.spm_max + 1e-6:
        reasons.append(
            f"the current speed is above the {lim.binding.replace('_', ' ')} ceiling of {lim.spm_max:.2f} SPM "
            f"(tubing viscosity {mu_now_cp:,.0f} cP, {visc_delta:+.0f}% in 24 h)"
        )
        if new["float_index"] < cur["float_index"]:
            reasons.append("slowing the pump lets the rods fall freely and prevents rod impact loading")
    elif rec < spm_now - 0.25:
        if new["fillage"] > cur["fillage"] + 0.05:
            reasons.append(f"inflow only fills {cur['fillage'] * 100:.0f}% of the pump at the current speed, causing fluid pound")
        reasons.append(f"it saves {abs(new['kw'] - cur['kw']):.1f} kW without losing oil")
    elif rec > spm_now + 0.25:
        reasons.append(
            f"inflow ({cur['liquid_capacity_m3d'] * cur['fillage']:.0f} m3/d) exceeds the pump capacity at {spm_now:.1f} SPM and the safety envelope allows more"
        )
    else:
        reasons.append("the current speed already balances inflow, rod loading and energy")
    if not lim.feasible:
        reasons.append("no safe pumping speed exists at this viscosity: re-steam the well")
    narrative = f"{action}: " + "; ".join(reasons) + "."
    narrative = narrative[0].upper() + narrative[1:]

    return {
        "action": action,
        "narrative": narrative,
        "drivers": drivers,
        "contributions": contributions,
        "uplift_usd_per_day": _uplift_per_day(res),
        "ml_correction": ml_top or [],
    }
