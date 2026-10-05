"""CSS cycle planner: choose the steam volume for the next cycle.

The planner sweeps candidate steam volumes through the twin (physics + ML correction)
and ranks them on cumulative oil, steam-oil ratio (SOR) and marginal economics. It is a
heavier computation than a single forecast, so the API runs it as a background job.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PlanEconomics:
    oil_usd_m3: float
    steam_usd_m3: float


def plan_steam_volume(
    simulate: Callable[[float], dict],
    volumes: list[float],
    econ: PlanEconomics,
) -> dict:
    """``simulate(volume) -> {cum_oil_m3, peak_rate_m3d, curve_t, curve_q}``."""
    rows = []
    for v in volumes:
        s = simulate(float(v))
        profit = econ.oil_usd_m3 * s["cum_oil_m3"] - econ.steam_usd_m3 * v
        rows.append(
            {
                "steam_m3": float(v),
                "cum_oil_m3": float(s["cum_oil_m3"]),
                "peak_rate_m3d": float(s["peak_rate_m3d"]),
                "sor": float(v / s["cum_oil_m3"]) if s["cum_oil_m3"] > 0 else None,
                "profit_usd": float(profit),
            }
        )
    for i, r in enumerate(rows):
        if i == 0:
            r["marginal_oil_per_1000m3"] = None
        else:
            dv = r["steam_m3"] - rows[i - 1]["steam_m3"]
            r["marginal_oil_per_1000m3"] = (r["cum_oil_m3"] - rows[i - 1]["cum_oil_m3"]) / dv * 1000.0
    best = max(rows, key=lambda r: r["profit_usd"])
    lowest_sor = min((r for r in rows if r["sor"] is not None), key=lambda r: r["sor"])
    return {
        "rows": rows,
        "best_profit_volume_m3": best["steam_m3"],
        "best_profit_usd": best["profit_usd"],
        "lowest_sor_volume_m3": lowest_sor["steam_m3"],
        "note": (
            "Profit = oil revenue - steam cost over the planned production window; marginal oil per extra 1000 m3 "
            "of steam shows the diminishing return that sets the optimum."
        ),
    }
