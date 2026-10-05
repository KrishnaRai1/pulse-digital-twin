"""Polished-rod loads, rod-float criterion, fatigue (modified Goodman) and power.

All quantities SI unless noted. The dynamic factor follows the API RP 11L / Coberly
simplification ``alpha = S N^2 / 70,500`` (S in inches, N in strokes/min). Viscous drag
on the rod string is the Couette shear in the tubing annulus scaled by a coupling and
guide multiplier ``KC`` (couplings, centralisers and paraffin dominate real drag).
"""
from __future__ import annotations

import math

from .rods import G, SrpDesign

KC = 3.0  # multiplier on ideal Couette drag for couplings / guides
FLOAT_MARGIN = 0.85  # float index must stay below this (15 % margin to true floating)
FRICTION_FRACTION = 0.03  # mechanical friction as fraction of buoyant rod weight


def dynamic_factor(stroke_m: float, spm: float) -> float:
    return stroke_m * 39.3701 * spm**2 / 70500.0


def drag_coefficient(mu_pa_s: float, design: SrpDesign) -> float:
    """Viscous drag per unit rod velocity (N per m/s), summed over taper sections."""
    r_t = design.tubing_id_mm * 1e-3 / 2.0
    total = 0.0
    for sec in design.rods:
        r_r = sec.dia_mm * 1e-3 / 2.0
        total += 2.0 * math.pi * mu_pa_s * sec.length_m / math.log(r_t / r_r)
    return KC * total


def peak_rod_velocity(stroke_m: float, spm: float) -> float:
    return math.pi * stroke_m * spm / 60.0


def float_index(spm: float, mu_pa_s: float, design: SrpDesign, stroke_m: float | None = None) -> float:
    """``1 - MPRL/W_b``: >= 1 means the rods cannot fall as fast as the polished rod."""
    s = stroke_m or design.stroke_m
    wb = design.weight_buoyant_n
    f_d = drag_coefficient(mu_pa_s, design) * peak_rod_velocity(s, spm)
    return (f_d + dynamic_factor(s, spm) * wb) / wb


def polished_rod_loads(
    spm: float, mu_pa_s: float, design: SrpDesign, fluid_load_n: float, stroke_m: float | None = None
) -> tuple[float, float]:
    """(PPRL, MPRL) in N."""
    s = stroke_m or design.stroke_m
    wb = design.weight_buoyant_n
    a = dynamic_factor(s, spm)
    f_d = drag_coefficient(mu_pa_s, design) * peak_rod_velocity(s, spm)
    pprl = wb * (1.0 + a) + fluid_load_n + f_d
    mprl = wb * (1.0 - a) - f_d
    return pprl, mprl


def goodman_utilisation(pprl_n: float, mprl_n: float, design: SrpDesign) -> dict[str, float]:
    """Modified-Goodman utilisation at the top (most stressed) rod section.

    ``Sa = (T/4 + 0.5625 S_min) * SF``; utilisation = ``S_max / Sa`` (must stay below 1).
    """
    area = design.top_area_m2
    s_max = pprl_n / area / 1e6
    s_min = max(mprl_n, 0.0) / area / 1e6
    allowable = (design.rod_tensile_mpa / 4.0 + 0.5625 * s_min) * design.service_factor
    return {"s_max_mpa": s_max, "s_min_mpa": s_min, "allowable_mpa": allowable, "utilisation": s_max / allowable}


def polished_rod_power_kw(spm: float, mu_pa_s: float, design: SrpDesign, fluid_load_n: float) -> float:
    """Work per stroke x strokes per second (kW)."""
    s = design.stroke_m
    v_mean = s * spm / 30.0  # average speed over a full stroke cycle (2 S per period)
    w_fluid = fluid_load_n * s * 0.92
    w_visc = 2.0 * drag_coefficient(mu_pa_s, design) * v_mean * s * 0.5
    w_fric = 2.0 * FRICTION_FRACTION * design.weight_buoyant_n * s
    return (w_fluid + w_visc + w_fric) * spm / 60.0 / 1000.0


def electrical_power_kw(pr_power_kw: float) -> float:
    """Motor + gearbox + VFD: ~60 % efficient plus 0.5 kW of no-load losses."""
    return pr_power_kw / 0.60 + 0.5


def vfd_hz_for_spm(spm: float, design: SrpDesign) -> float:
    return 60.0 * spm / design.max_spm


def structure_utilisation(pprl_n: float, design: SrpDesign) -> float:
    return pprl_n / (design.unit_rating_kn * 1000.0)


__all__ = [
    "G",
    "FLOAT_MARGIN",
    "dynamic_factor",
    "drag_coefficient",
    "peak_rod_velocity",
    "float_index",
    "polished_rod_loads",
    "goodman_utilisation",
    "polished_rod_power_kw",
    "electrical_power_kw",
    "vfd_hz_for_spm",
    "structure_utilisation",
]
