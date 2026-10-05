"""Steam thermodynamics and wellbore heat loss.

Correlations are engineering approximations (typically within 1-3 % of the IAPWS
steam tables over 0.1-20 MPa), sufficient for a reduced-order digital twin.
"""
from __future__ import annotations

import math

import numpy as np

WATER_CP_J_KG_K = 4300.0  # mean liquid-water heat capacity over 50-250 C
WATER_CWE_KG_M3 = 1000.0  # steam volumes are reported as cold-water equivalent
T_CRIT_C = 373.95
LV_100C_J_KG = 2.257e6


def t_sat_c(p_mpa: float) -> float:
    """Saturation temperature (C) from the Antoine equation (valid 99-374 C)."""
    p_mmhg = max(p_mpa, 0.11) * 7500.62
    t = 1810.94 / (8.14019 - math.log10(p_mmhg)) - 244.485
    return float(min(t, T_CRIT_C - 1.0))


def latent_heat_j_kg(t_c: float) -> float:
    """Latent heat of vaporisation by the Watson correlation."""
    t_c = min(t_c, T_CRIT_C - 0.5)
    return float(LV_100C_J_KG * ((T_CRIT_C - t_c) / (T_CRIT_C - 100.0)) ** 0.38)


def wellbore_heat_loss_w(
    depth_m: float, t_steam_c: float, t_surface_c: float, geo_gradient_c_per_m: float, u_w_m2k: float, tubing_od_m: float = 0.0889
) -> float:
    """Heat lost along the injection string (W), overall coefficient ``u_w`` (W/m2/K).

    Uses the mean temperature difference between the steam and the undisturbed
    geotherm, a simplification of Ramey (1962) that ignores the time-dependent
    earth resistance (conservative: it slightly over-predicts the loss).
    """
    t_geo_mean = t_surface_c + 0.5 * geo_gradient_c_per_m * depth_m
    return max(0.0, u_w_m2k * math.pi * tubing_od_m * depth_m * (t_steam_c - t_geo_mean))


def sandface_steam(
    *,
    rate_m3d: float,
    quality_surface: float,
    p_bh_mpa: float,
    depth_m: float,
    t_res_c: float,
    t_surface_c: float = 30.0,
    geo_gradient_c_per_m: float = 0.026,
    u_w_m2k: float = 3.0,
) -> dict[str, float]:
    """Energy delivered to the sandface for a CWE injection rate.

    Returns the injection temperature, bottom-hole quality, heat rate (W) into the
    reservoir above reservoir temperature, and the fraction of surface enthalpy lost.
    """
    m_dot = rate_m3d * WATER_CWE_KG_M3 / 86400.0
    t_s = t_sat_c(p_bh_mpa)
    lv = latent_heat_j_kg(t_s)
    h_surface = quality_surface * lv + WATER_CP_J_KG_K * (t_s - t_res_c)
    q_loss = wellbore_heat_loss_w(depth_m, t_s, t_surface_c, geo_gradient_c_per_m, u_w_m2k)
    h_bh = max(h_surface - q_loss / max(m_dot, 1e-9), 0.0)
    x_bh = float(np.clip((h_bh - WATER_CP_J_KG_K * (t_s - t_res_c)) / lv, 0.0, 1.0))
    q_res_w = m_dot * h_bh
    return {
        "t_steam_c": t_s,
        "quality_bottomhole": x_bh,
        "heat_rate_w": q_res_w,
        "wellbore_loss_fraction": float(1.0 - h_bh / h_surface) if h_surface > 0 else 1.0,
        "latent_heat_j_kg": lv,
    }


def tubing_temperature_profile(
    *,
    depth_m: float,
    t_sandface_c: float,
    t_surface_c: float,
    geo_gradient_c_per_m: float,
    liquid_rate_m3d: float,
    liquid_density_kg_m3: float = 940.0,
    cp_j_kg_k: float = 2100.0,
    tubing_id_m: float = 0.0759,
    u_t_w_m2k: float = 1.5,
    n: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Flowing fluid temperature vs depth (Ramey-type exponential relaxation).

    ``T(z) = T_geo(z) + (T_sf - T_geo(D)) * exp(-(D - z) / A)`` with the relaxation
    length ``A = m cp / (pi d U)``: slow wells lose their heat quickly, which is the
    mechanism that makes heavy oil reach the pump cold late in a CSS cycle.
    """
    z = np.linspace(0.0, depth_m, n)
    t_geo = t_surface_c + geo_gradient_c_per_m * z
    m_dot = max(liquid_rate_m3d, 0.1) * liquid_density_kg_m3 / 86400.0
    relax = m_dot * cp_j_kg_k / (math.pi * tubing_id_m * u_t_w_m2k)
    t = t_geo + (t_sandface_c - t_geo[-1]) * np.exp(-(depth_m - z) / relax)
    return z, t
