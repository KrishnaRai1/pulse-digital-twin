"""Radial heat conduction around a wellbore + Marx-Langenheim heat balance.

Two coupled pieces of physics, both standard in thermal-recovery engineering:

1. **Marx-Langenheim (1959)** energy balance during injection: how much of the
   injected enthalpy stays in the pay zone (the rest leaks by vertical conduction
   into the over/under-burden) and therefore how large the heated radius becomes.
2. **Radial conduction with a vertical-loss sink** during soak and production: a
   fully implicit finite-volume solution of

       M_s dT/dt = (1/r) d/dr (K r dT/dr) - lambda(t) M_s (T - T_res)

   on a logarithmic grid. ``lambda(t)`` is the semi-infinite-medium loss rate to
   the burden, scaled by a calibration multiplier ``loss_mult`` that absorbs
   unmodelled sinks (shale streaks, water influx, steam override). The ML layer
   corrects what is still left over.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erfc, erfcx

SECONDS_PER_DAY = 86400.0
MD_TO_M2 = 9.869233e-16


@dataclass(frozen=True)
class Reservoir:
    thickness_m: float = 20.0
    t_res_c: float = 52.0
    perm_md: float = 90.0
    kro: float = 0.35
    m_s: float = 2.4e6  # J/m3/K volumetric heat capacity of pay (rock + fluids)
    k_res: float = 2.0  # W/m/K radial conductivity
    k_ob: float = 1.8  # W/m/K over/under-burden
    m_ob: float = 2.3e6  # J/m3/K
    loss_mult: float = 3.0  # calibration multiplier on vertical heat loss
    r_w: float = 0.1
    r_e: float = 200.0
    depth_m: float = 950.0
    geo_gradient_c_per_m: float = 0.026
    t_surface_c: float = 30.0
    # pressure support: pump drawdown + steam-induced repressurisation that decays
    dp_pump_mpa: float = 0.8
    dp_rep_mpa_per_1000m3: float = 0.30
    tau_p_days: float = 80.0
    depletion_per_cycle: float = 0.93


class RadialGrid:
    """Logarithmic finite-volume grid from the wellbore wall to the outer boundary."""

    def __init__(self, res: Reservoir, n: int = 90):
        self.res = res
        self.n = n
        self.r = res.r_w * (res.r_e / res.r_w) ** (np.arange(n) / (n - 1))
        faces = np.empty(n + 1)
        faces[0] = res.r_w
        faces[1:-1] = np.sqrt(self.r[:-1] * self.r[1:])
        faces[-1] = res.r_e
        self.faces = faces
        self.area = np.pi * (faces[1:] ** 2 - faces[:-1] ** 2)  # m2 per ring
        self.volume = self.area * res.thickness_m
        self.cap = res.m_s * self.volume  # J/K per ring
        self.g = 2.0 * np.pi * res.k_res * res.thickness_m / np.log(self.r[1:] / self.r[:-1])  # W/K

    def energy_j(self, temp: np.ndarray) -> float:
        return float(np.sum(self.cap * (temp - self.res.t_res_c)))

    def _area_weights(self, r_max: float) -> np.ndarray:
        """Fraction of each ring inside ``r_max`` (continuous in r_max, so results do not jump
        when the boundary crosses a grid node)."""
        lo, hi = self.faces[:-1] ** 2, self.faces[1:] ** 2
        return np.clip((r_max**2 - lo) / (hi - lo), 0.0, 1.0)

    def _log_weights(self, r_max: float) -> np.ndarray:
        lo, hi = np.log(self.faces[:-1]), np.log(self.faces[1:])
        return np.clip((np.log(r_max) - lo) / (hi - lo), 0.0, 1.0) * (hi - lo)

    def volume_avg_temp(self, temp: np.ndarray, r_max: float) -> float:
        w = self._area_weights(max(r_max, self.r[2])) * self.volume
        return float(np.sum(temp * w) / np.sum(w))

    def flow_avg(self, values: np.ndarray, r_max: float) -> float:
        """Mean of ``values`` with weight d(ln r): the quantity that governs radial Darcy flow
        resistance (integral of mu dr / r), continuous in ``r_max``."""
        w = self._log_weights(max(r_max, self.r[2]))
        return float(np.sum(values * w) / np.sum(w))

    def implicit_step(self, temp: np.ndarray, dt_s: float, lam: float) -> np.ndarray:
        """One backward-Euler step with sink ``lam`` (1/s) towards the reservoir temperature."""
        n = self.n
        c_over_dt = self.cap / dt_s
        diag = c_over_dt + lam * self.cap
        diag[:-1] += self.g
        diag[1:] += self.g
        upper = -self.g.copy()
        lower = -self.g.copy()
        rhs = c_over_dt * temp + lam * self.cap * self.res.t_res_c
        # outer boundary: Dirichlet at reservoir temperature
        diag[-1] = 1.0
        lower[-1] = 0.0
        rhs[-1] = self.res.t_res_c
        ab = np.zeros((3, n))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        return solve_banded((1, 1), ab, rhs)


# --------------------------------------------------------------------------- Marx-Langenheim
def ml_heat_efficiency(t_s: float, res: Reservoir) -> float:
    """Fraction of injected heat retained in the pay zone after ``t_s`` seconds of injection."""
    km = (res.loss_mult**2) * res.k_ob * res.m_ob
    t_d = 4.0 * km * t_s / (res.thickness_m * res.m_s) ** 2
    if t_d < 1e-9:
        return 1.0
    s = np.sqrt(t_d)
    g = erfcx(s) + 2.0 * s / np.sqrt(np.pi) - 1.0
    return float(np.clip(g / t_d, 0.0, 1.0))


def ml_heated_area_m2(heat_rate_w: float, t_s: float, dt_steam_k: float, res: Reservoir) -> float:
    """Heated area (m2) of a plateau at steam temperature: A_s = E_h * Q_i * t / (M_s h dT)."""
    if dt_steam_k <= 0:
        return 0.0
    e_h = ml_heat_efficiency(t_s, res)
    return e_h * heat_rate_w * t_s / (res.m_s * res.thickness_m * dt_steam_k)


def vertical_loss_rate(t_since_heat_s: float, res: Reservoir, t0_days: float = 5.0) -> float:
    """lambda(t) [1/s]: volumetric heat-loss rate coefficient towards the burden."""
    alpha_ob = res.k_ob / res.m_ob
    t = t_since_heat_s + t0_days * SECONDS_PER_DAY
    flux_per_k = 2.0 * res.k_ob / np.sqrt(np.pi * alpha_ob * t)  # W/m2/K, both faces
    return float(res.loss_mult * flux_per_k / (res.thickness_m * res.m_s))


def heated_profile(
    grid: RadialGrid, t_prev: np.ndarray, t_steam_c: float, q_res_j: float
) -> tuple[np.ndarray, float]:
    """Temperature profile after injection that stores exactly ``q_res_j`` more energy.

    A smoothed step (erfc front) of amplitude ``t_steam - t_res`` is expanded until the
    stored energy matches; residual heat from earlier cycles is preserved (max of both).
    Returns the profile and the heated radius (radius where T is halfway to steam T).
    """
    res = grid.res
    dt_full = t_steam_c - res.t_res_c
    e_prev = grid.energy_j(t_prev)

    def build(r_h: float) -> np.ndarray:
        w = max(0.12 * r_h, 0.5)
        shape = 0.5 * erfc((grid.r - r_h) / w)
        return np.maximum(t_prev, res.t_res_c + dt_full * shape)

    if q_res_j <= 0 or dt_full <= 0:
        return t_prev.copy(), res.r_w
    lo, hi = res.r_w, 0.9 * res.r_e
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if grid.energy_j(build(mid)) - e_prev < q_res_j:
            lo = mid
        else:
            hi = mid
    r_h = 0.5 * (lo + hi)
    return build(r_h), r_h
