"""Physics baseline for one Cyclic Steam Stimulation (CSS) cycle and for a well's history.

A cycle has three phases: steam **injection**, **soak** (shut-in) and **production**.
``simulate_history`` chains cycles so that residual heat of earlier cycles is carried
into the next one, exactly as the radial temperature field would be in the reservoir.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc

from . import steam
from .radial import (
    MD_TO_M2,
    SECONDS_PER_DAY,
    RadialGrid,
    Reservoir,
    heated_profile,
    ml_heat_efficiency,
    ml_heated_area_m2,
    vertical_loss_rate,
)
from .viscosity import Walther

PHASE_INJECTION, PHASE_SOAK, PHASE_PRODUCTION = 0, 1, 2
PHASE_NAMES = {0: "INJECTION", 1: "SOAK", 2: "PRODUCTION"}


@dataclass(frozen=True)
class CycleSpec:
    cycle_no: int
    steam_m3: float
    quality: float = 0.75
    inj_rate_m3d: float = 200.0
    inj_pressure_mpa: float = 1.8
    soak_days: float = 7.0
    prod_days: float = 180.0

    @property
    def inj_days(self) -> float:
        return self.steam_m3 / self.inj_rate_m3d


@dataclass
class CycleResult:
    spec: CycleSpec
    t_days: np.ndarray  # days since start of injection (daily grid)
    phase: np.ndarray
    r_heated_m: np.ndarray
    t_avg_c: np.ndarray  # volume-averaged temperature of the drainage zone
    t_wellbore_c: np.ndarray  # temperature at the sandface
    mu_eff_cp: np.ndarray  # dry-oil effective viscosity of the drainage zone
    q_oil_m3d: np.ndarray  # physics-only oil rate
    water_cut: np.ndarray
    cum_oil_m3: np.ndarray
    profiles: np.ndarray  # (n_steps, n_nodes) radial temperature history
    r_nodes: np.ndarray
    end_profile: np.ndarray
    steam_info: dict = field(default_factory=dict)
    r_drain_m: float = 0.0

    @property
    def total_days(self) -> float:
        return float(self.t_days[-1])

    @property
    def liquid_rate_m3d(self) -> np.ndarray:
        return self.q_oil_m3d / np.clip(1.0 - self.water_cut, 0.05, 1.0)

    @property
    def sor(self) -> float:
        cum = float(self.cum_oil_m3[-1])
        return self.spec.steam_m3 / cum if cum > 1e-6 else float("inf")


def default_walther() -> Walther:
    """Two lab points for a 13-15 API waxy heavy crude (synthetic calibration)."""
    return Walther.from_two_points(50.0, 1500.0, 100.0, 45.0)


def water_cut_curve(t_prod_days: np.ndarray, steam_m3: float) -> np.ndarray:
    """Condensate-dominated water cut that decays towards connate production."""
    wc0 = float(np.clip(0.50 + 0.03 * steam_m3 / 1000.0, 0.5, 0.80))
    wc_inf = 0.30
    return wc_inf + (wc0 - wc_inf) * np.exp(-t_prod_days / 40.0)


def cool_profile(grid: RadialGrid, temp: np.ndarray, days: float, t_since_heat_days: float) -> np.ndarray:
    """Advance the radial field ``days`` with no production (used for gaps between cycles)."""
    t = temp
    n = int(np.ceil(days))
    for i in range(n):
        dt = min(1.0, days - i)
        lam = vertical_loss_rate((t_since_heat_days + i + 0.5 * dt) * SECONDS_PER_DAY, grid.res)
        t = grid.implicit_step(t, dt * SECONDS_PER_DAY, lam)
    return t


def simulate_cycle(
    res: Reservoir,
    spec: CycleSpec,
    grid: RadialGrid | None = None,
    t_start_profile: np.ndarray | None = None,
    walther: Walther | None = None,
) -> CycleResult:
    grid = grid or RadialGrid(res)
    walther = walther or default_walther()
    t0 = grid.res.t_res_c * np.ones(grid.n) if t_start_profile is None else t_start_profile.copy()

    info = steam.sandface_steam(
        rate_m3d=spec.inj_rate_m3d,
        quality_surface=spec.quality,
        p_bh_mpa=spec.inj_pressure_mpa,
        depth_m=res.depth_m,
        t_res_c=res.t_res_c,
        t_surface_c=res.t_surface_c,
        geo_gradient_c_per_m=res.geo_gradient_c_per_m,
    )
    t_s = info["t_steam_c"]
    dt_steam = t_s - res.t_res_c
    q_rate = info["heat_rate_w"]

    n_inj = int(np.ceil(spec.inj_days))
    n_soak = int(np.ceil(spec.soak_days))
    n_prod = int(np.ceil(spec.prod_days))
    n_steps = n_inj + n_soak + n_prod + 1  # include t=0

    t_days = np.zeros(n_steps)
    phase = np.zeros(n_steps, dtype=int)
    r_h = np.zeros(n_steps)
    t_avg = np.zeros(n_steps)
    t_wb = np.zeros(n_steps)
    mu_eff = np.zeros(n_steps)
    q_oil = np.zeros(n_steps)
    wc = np.zeros(n_steps)
    profiles = np.zeros((n_steps, grid.n))

    # ---- injection: Marx-Langenheim energy balance ----------------------------------
    profiles[0] = t0
    t_wb[0] = t0[0]
    t_avg[0] = grid.volume_avg_temp(t0, 2.0)
    mu_eff[0] = float(walther.mu_cp(t_avg[0]))
    inj_end_s = spec.inj_days * SECONDS_PER_DAY
    for i in range(1, n_inj + 1):
        t_d = min(float(i), spec.inj_days)
        a_s = ml_heated_area_m2(q_rate, t_d * SECONDS_PER_DAY, dt_steam, res)
        r_now = float(np.sqrt(a_s / np.pi)) + res.r_w
        t_days[i] = t_d
        phase[i] = PHASE_INJECTION
        r_h[i] = r_now
        # plateau at steam temperature inside the heated radius, smoothed for display
        w = max(0.12 * r_now, 0.5)
        shape = 0.5 * erfc((grid.r - r_now) / w)
        profiles[i] = np.maximum(t0, res.t_res_c + dt_steam * shape)
        t_wb[i] = profiles[i][0]
        t_avg[i] = grid.volume_avg_temp(profiles[i], r_now)
        mu_eff[i] = float(walther.mu_cp(t_avg[i]))
    q_res_j = ml_heat_efficiency(inj_end_s, res) * q_rate * inj_end_s
    temp, r_h_end = heated_profile(grid, t0, t_s, q_res_j)
    profiles[n_inj] = temp
    r_h[n_inj] = r_h_end
    t_wb[n_inj] = temp[0]
    t_avg[n_inj] = grid.volume_avg_temp(temp, r_h_end)
    mu_eff[n_inj] = float(walther.mu_cp(t_avg[n_inj]))
    t_days[n_inj] = spec.inj_days
    r_drain = max(r_h_end, 2.0)

    # ---- soak + production: implicit radial conduction with vertical loss -------------
    day_clock = spec.inj_days
    t_since_heat = 0.0
    ln_ratio = np.log(r_drain / res.r_w)
    depl = res.depletion_per_cycle ** max(spec.cycle_no - 1, 0)
    k_m2 = res.perm_md * MD_TO_M2 * res.kro
    idx = n_inj
    soak_end = spec.inj_days + spec.soak_days
    for j in range(1, n_soak + n_prod + 1):
        idx += 1
        dt = 1.0
        lam = vertical_loss_rate((t_since_heat + 0.5 * dt) * SECONDS_PER_DAY, res)
        temp = grid.implicit_step(temp, dt * SECONDS_PER_DAY, lam)
        t_since_heat += dt
        day_clock += dt
        t_days[idx] = day_clock
        r_h[idx] = r_h_end
        t_wb[idx] = temp[0]
        t_avg[idx] = grid.volume_avg_temp(temp, r_drain)
        mu_eff[idx] = grid.flow_avg(walther.mu_cp(temp), r_drain)
        profiles[idx] = temp
        if j <= n_soak:
            phase[idx] = PHASE_SOAK
        else:
            phase[idx] = PHASE_PRODUCTION
            t_p = day_clock - soak_end
            dp = (
                res.dp_pump_mpa
                + res.dp_rep_mpa_per_1000m3 * spec.steam_m3 / 1000.0 * np.exp(-t_p / res.tau_p_days)
            ) * 1e6
            mu_pa_s = mu_eff[idx] * 1e-3
            q_m3s = 2.0 * np.pi * k_m2 * res.thickness_m * dp / (mu_pa_s * ln_ratio)
            q_oil[idx] = q_m3s * SECONDS_PER_DAY * depl
            wc[idx] = float(water_cut_curve(np.array([t_p]), spec.steam_m3)[0])

    cum = np.cumsum(np.concatenate([[0.0], 0.5 * (q_oil[1:] + q_oil[:-1]) * np.diff(t_days)]))
    return CycleResult(
        spec=spec,
        t_days=t_days,
        phase=phase,
        r_heated_m=r_h,
        t_avg_c=t_avg,
        t_wellbore_c=t_wb,
        mu_eff_cp=mu_eff,
        q_oil_m3d=q_oil,
        water_cut=wc,
        cum_oil_m3=cum,
        profiles=profiles,
        r_nodes=grid.r,
        end_profile=temp,
        steam_info=info,
        r_drain_m=r_drain,
    )


@dataclass
class HistoryResult:
    results: list[CycleResult]
    start_days: list[float]  # start of each cycle in days since the first cycle start


def simulate_history(
    res: Reservoir,
    cycles: list[CycleSpec],
    starts_days: list[float],
    grid: RadialGrid | None = None,
    walther: Walther | None = None,
) -> HistoryResult:
    """Chain cycles in time order; residual heat carries over (with conduction across gaps)."""
    grid = grid or RadialGrid(res)
    profile = None
    prev_end = None
    since_heat = 0.0
    results: list[CycleResult] = []
    for spec, start in zip(cycles, starts_days):
        if profile is not None and prev_end is not None:
            gap = max(0.0, start - prev_end)
            profile = cool_profile(grid, profile, gap, since_heat)
        r = simulate_cycle(res, spec, grid, profile, walther)
        results.append(r)
        profile = r.end_profile
        prev_end = start + r.total_days
        since_heat = r.total_days - spec.inj_days
    return HistoryResult(results=results, start_days=list(starts_days))
