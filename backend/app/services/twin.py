"""Digital-twin facade: one place that turns a well's cycle history into its live state.

Pipeline per well:  cycle history (DB) -> physics (radial conduction + Marx-Langenheim)
-> ML residual correction (LightGBM) -> state at any timestamp, forecasts, what-if runs.
Results are cached per (well, cycle-history) and invalidated when history is edited.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass

import numpy as np
from sqlalchemy.engine import Engine

from ..domain.wells import DAY, WellConfig, row_to_spec
from ..engines.optimizer.mpc import Forecast
from ..engines.thermal import steam
from ..engines.thermal.cycle import (
    PHASE_INJECTION,
    PHASE_NAMES,
    PHASE_PRODUCTION,
    PHASE_SOAK,
    CycleResult,
    CycleSpec,
    default_walther,
    simulate_cycle,
    simulate_history,
)
from ..engines.thermal.ml_correction import MLCorrector, build_features
from ..engines.thermal.radial import RadialGrid
from ..engines.thermal.viscosity import emulsion_factor
from . import storage

WALTHER = default_walther()


@dataclass
class WellSeries:
    well_id: str
    spec: CycleSpec
    start_ts: int
    result: CycleResult
    log_factor: np.ndarray  # ML correction per step (0 outside production)
    q_corr: np.ndarray
    mu_corr: np.ndarray
    q_liq_corr: np.ndarray
    cum_corr: np.ndarray
    t_abs: np.ndarray
    ml_used: bool
    feats: object = None

    @property
    def total_days(self) -> float:
        return float(self.result.t_days[-1])


def _corrected(ml: MLCorrector | None, res_cfg, spec: CycleSpec, r: CycleResult):
    lf = np.zeros(len(r.t_days))
    feats = None
    if ml is not None:
        feats = build_features(res_cfg, spec, r)
        mask = (r.phase == PHASE_PRODUCTION) & (r.q_oil_m3d > 0)
        lf[mask] = ml.log_factor(feats)
    q = r.q_oil_m3d * np.exp(lf)
    mu = r.mu_eff_cp * np.exp(-lf)
    q_liq = q / np.clip(1.0 - r.water_cut, 0.05, 1.0)
    cum = np.cumsum(np.concatenate([[0.0], 0.5 * (q[1:] + q[:-1]) * np.diff(r.t_days)]))
    return lf, q, mu, q_liq, cum, feats


class Twin:
    def __init__(self, engine: Engine, fleet: dict[str, WellConfig], ml: MLCorrector | None):
        self.engine = engine
        self.fleet = fleet
        self.ml = ml
        self._cache: dict[str, tuple[str, WellSeries]] = {}
        self._grids: dict[str, RadialGrid] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ history
    def set_ml(self, ml: MLCorrector | None) -> None:
        with self._lock:
            self.ml = ml
            self._cache.clear()

    def invalidate(self, well_id: str | None = None) -> None:
        with self._lock:
            if well_id:
                self._cache.pop(well_id, None)
            else:
                self._cache.clear()

    def _grid(self, cfg: WellConfig) -> RadialGrid:
        if cfg.id not in self._grids:
            self._grids[cfg.id] = RadialGrid(cfg.reservoir)
        return self._grids[cfg.id]

    def cycles(self, well_id: str) -> list[dict]:
        return storage.get_cycles(self.engine, well_id)

    def series(self, well_id: str) -> WellSeries:
        cfg = self.fleet[well_id]
        rows = self.cycles(well_id)
        if not rows:
            raise LookupError(f"no cycle history for {well_id}")
        sig = hashlib.sha1(json.dumps(rows, sort_keys=True, default=str).encode(), usedforsecurity=False).hexdigest() + ("ml" if self.ml else "")
        with self._lock:
            hit = self._cache.get(well_id)
            if hit and hit[0] == sig:
                return hit[1]
            first = rows[0]["start_ts"]
            specs = [row_to_spec(r) for r in rows]
            starts = [(r["start_ts"] - first) / DAY for r in rows]
            hist = simulate_history(cfg.reservoir, specs, starts, self._grid(cfg), WALTHER)
            last, spec = hist.results[-1], specs[-1]
            lf, q, mu, ql, cum, feats = _corrected(self.ml, cfg.reservoir, spec, last)
            start_ts = int(rows[-1]["start_ts"])
            series = WellSeries(
                well_id=well_id,
                spec=spec,
                start_ts=start_ts,
                result=last,
                log_factor=lf,
                q_corr=q,
                mu_corr=mu,
                q_liq_corr=ql,
                cum_corr=cum,
                t_abs=start_ts + last.t_days * DAY,
                ml_used=self.ml is not None,
                feats=feats,
            )
            self._cache[well_id] = (sig, series)
            return series

    # ------------------------------------------------------------------ state
    def tubing(self, cfg: WellConfig, t_sandface_c: float, q_liquid_m3d: float, water_cut: float) -> dict:
        res = cfg.reservoir
        z, temp = steam.tubing_temperature_profile(
            depth_m=cfg.design.pump_depth_m,
            t_sandface_c=t_sandface_c,
            t_surface_c=res.t_surface_c,
            geo_gradient_c_per_m=res.geo_gradient_c_per_m,
            liquid_rate_m3d=max(q_liquid_m3d, 0.5),
        )
        mu_cp = WALTHER.mu_cp(temp) * float(emulsion_factor(water_cut))
        # viscous drag integrates along the rods -> arithmetic mean over depth
        return {
            "t_wellhead_c": float(temp[0]),
            "t_pump_c": float(temp[-1]),
            "mu_tub_cp": float(np.mean(mu_cp)),
            "mu_tub_pa_s": float(np.mean(mu_cp)) * 1e-3,
            "z": z,
            "temp": temp,
        }

    def state_at(self, well_id: str, ts: float) -> dict:
        cfg = self.fleet[well_id]
        s = self.series(well_id)
        r = s.result
        day = (ts - s.start_ts) / DAY
        ended = day > s.total_days
        d = float(np.clip(day, 0.0, s.total_days))

        def at(a):
            return float(np.interp(d, r.t_days, a))

        idx = int(np.clip(np.searchsorted(r.t_days, d, side="left"), 0, len(r.t_days) - 1))
        phase = int(r.phase[idx])
        if d <= 0:
            phase = PHASE_INJECTION
        soak_end = s.spec.inj_days + s.spec.soak_days
        phase_name = "CYCLE_END" if ended else PHASE_NAMES[phase]
        t_avg = at(r.t_avg_c)
        t_wb = at(r.t_wellbore_c)
        q_phys = 0.0 if (ended or phase != PHASE_PRODUCTION) else at(r.q_oil_m3d)
        q_corr = 0.0 if (ended or phase != PHASE_PRODUCTION) else at(s.q_corr)
        wc = at(r.water_cut) if phase == PHASE_PRODUCTION and not ended else 0.0
        q_liq = q_corr / max(1.0 - wc, 0.05) if q_corr > 0 else 0.0
        res = cfg.reservoir
        t_steam = float(r.steam_info.get("t_steam_c", 200.0))
        health = float(np.clip((t_avg - res.t_res_c) / max(t_steam - res.t_res_c, 1.0), 0.0, 1.0))
        tub = self.tubing(cfg, t_wb, q_liq if q_liq > 0 else 5.0, wc)
        inj_done = min(day, s.spec.inj_days)
        return {
            "well_id": well_id,
            "ts": int(ts),
            "phase": phase_name,
            "cycle_no": s.spec.cycle_no,
            "day_in_cycle": float(day),
            "prod_day": float(max(0.0, d - soak_end)) if phase == PHASE_PRODUCTION else 0.0,
            "t_avg_c": t_avg,
            "t_wellbore_c": t_wb,
            "r_heated_m": at(r.r_heated_m),
            "mu_eff_phys_cp": at(r.mu_eff_cp),
            "mu_eff_cp": at(s.mu_corr),
            "q_oil_phys_m3d": q_phys,
            "q_oil_m3d": q_corr,
            "water_cut": wc,
            "q_liquid_m3d": q_liq,
            "cum_oil_m3": at(s.cum_corr),
            "thermal_health": health,
            "steam_injected_m3": float(max(0.0, inj_done) * s.spec.inj_rate_m3d),
            "steam_total_m3": s.spec.steam_m3,
            "ml_factor": float(np.exp(at(s.log_factor))) if phase == PHASE_PRODUCTION else 1.0,
            "t_wellhead_c": tub["t_wellhead_c"],
            "mu_tub_cp": tub["mu_tub_cp"],
            "mu_tub_pa_s": tub["mu_tub_pa_s"],
            "t_steam_c": t_steam,
        }

    # ------------------------------------------------------------------ profiles
    def profile(self, well_id: str, ts: float) -> dict:
        cfg = self.fleet[well_id]
        s = self.series(well_id)
        r = s.result
        res = cfg.reservoir
        day = float(np.clip((ts - s.start_ts) / DAY, 0.0, s.total_days))
        idx = int(np.clip(np.searchsorted(r.t_days, day, side="left"), 0, len(r.t_days) - 1))
        temp_r = r.profiles[idx]
        mu_r = WALTHER.mu_cp(temp_r)
        st = self.state_at(well_id, ts)

        # wellbore: geotherm, flowing temperature and pressure vs depth
        depth = cfg.design.pump_depth_m
        z = np.linspace(0, depth, 50)
        t_geo = res.t_surface_c + res.geo_gradient_c_per_m * z
        _, t_flow = steam.tubing_temperature_profile(
            depth_m=depth,
            t_sandface_c=st["t_wellbore_c"],
            t_surface_c=res.t_surface_c,
            geo_gradient_c_per_m=res.geo_gradient_c_per_m,
            liquid_rate_m3d=max(st["q_liquid_m3d"], 5.0),
            n=50,
        )
        if st["phase"] == "PRODUCTION":
            whp = 250.0 + 6.0 * max(st["q_liquid_m3d"], 1.0) ** 0.8
            grad = 9.81 * cfg.design.fluid_density / 1000.0  # kPa/m of produced fluid column
            fluid_level = depth * (1.0 - cfg.net_lift_fraction)
            p = np.where(z < fluid_level, whp, whp + (z - fluid_level) * grad)
        else:  # shut-in / injecting: hydrostatic water column + surface pressure
            whp = 1500.0 if st["phase"] == "INJECTION" else 900.0
            p = whp + z * 9.81
        # cross-section: radial profile with gravity-override skew (top of pay hotter)
        nz = 15
        r_show = r.r_nodes[r.r_nodes <= max(1.6 * st["r_heated_m"], 12.0)]
        t_show = temp_r[: len(r_show)]
        skew = np.linspace(1.0, 0.72, nz)  # illustrative steam-override factor
        z_rel = np.linspace(0.0, res.thickness_m, nz)
        grid_t = np.array([res.t_res_c + (t_show - res.t_res_c) * k for k in skew])
        return {
            "well_id": well_id,
            "ts": int(ts),
            "day_in_cycle": day,
            "wellbore": {
                "depth_m": z.tolist(),
                "t_geotherm_c": t_geo.tolist(),
                "t_flowing_c": t_flow.tolist(),
                "pressure_kpa": p.tolist(),
                "pump_depth_m": depth,
                "pay_top_m": res.depth_m - res.thickness_m / 2,
                "pay_bottom_m": res.depth_m + res.thickness_m / 2,
                "fluid_level_m": depth * (1.0 - cfg.net_lift_fraction),
            },
            "radial": {
                "r_m": r.r_nodes.tolist(),
                "t_c": temp_r.tolist(),
                "mu_cp": mu_r.tolist(),
                "r_heated_m": st["r_heated_m"],
                "r_drain_m": r.r_drain_m,
            },
            "cross_section": {
                "r_m": r_show.tolist(),
                "z_m": z_rel.tolist(),
                "t_c": grid_t.tolist(),
                "note": "Radial field from the physics solver; vertical skew is an illustrative steam-override factor.",
            },
            "state": {k: v for k, v in st.items() if k not in ()},
        }

    def viscosity_curves(self, well_id: str, ts: float) -> dict:
        s = self.series(well_id)
        r = s.result
        step = max(1, len(r.t_days) // 240)
        sl = slice(None, None, step)
        return {
            "t_days": r.t_days[sl].tolist(),
            "phase": r.phase[sl].tolist(),
            "mu_eff_phys_cp": r.mu_eff_cp[sl].tolist(),
            "mu_eff_cp": s.mu_corr[sl].tolist(),
            "t_avg_c": r.t_avg_c[sl].tolist(),
            "q_oil_phys_m3d": r.q_oil_m3d[sl].tolist(),
            "q_oil_m3d": s.q_corr[sl].tolist(),
            "now_day": float((ts - s.start_ts) / DAY),
            "inj_days": s.spec.inj_days,
            "soak_days": s.spec.soak_days,
            "prod_days": s.spec.prod_days,
            "ml_used": s.ml_used,
        }

    # ------------------------------------------------------------------ forecast for the optimiser
    def forecast(self, well_id: str, ts: float, horizon_days: int = 14) -> tuple[Forecast, dict]:
        cfg = self.fleet[well_id]
        mus, qls, qos = [], [], []
        for k in range(horizon_days):
            st = self.state_at(well_id, ts + k * DAY)
            mus.append(st["mu_tub_pa_s"])
            qls.append(max(st["q_liquid_m3d"], 0.0))
            qos.append(max(st["q_oil_m3d"], 0.0))
        fc = Forecast(
            mu_tub_pa_s=np.array(mus),
            q_liquid_m3d=np.array(qls),
            q_oil_m3d=np.array(qos),
            net_lift_m=cfg.design.pump_depth_m * cfg.net_lift_fraction,
        )
        now = self.state_at(well_id, ts)
        past = self.state_at(well_id, ts - DAY)
        ahead = self.state_at(well_id, ts + 7 * DAY)
        extras = {
            "mu_now_cp": now["mu_tub_cp"],
            "mu_24h_ago_cp": past["mu_tub_cp"],
            "mu_7d_cp": ahead["mu_tub_cp"],
            "state": now,
        }
        return fc, extras

    def ml_top_features(self, well_id: str) -> list[dict]:
        s = self.series(well_id)
        if self.ml is None or s.feats is None or len(s.feats) == 0:
            return []
        return self.ml.contributions(s.feats)[:4]

    # ------------------------------------------------------------------ what-if
    def _profile_after(self, well_id: str, upto_cycle_no: int, gap_days: float) -> np.ndarray | None:
        """Radial field at the start of a cycle: history up to ``upto_cycle_no`` then a gap."""
        cfg = self.fleet[well_id]
        rows = [r for r in self.cycles(well_id) if r["cycle_no"] <= upto_cycle_no]
        if not rows:
            return None
        first = rows[0]["start_ts"]
        specs = [row_to_spec(r) for r in rows]
        starts = [(r["start_ts"] - first) / DAY for r in rows]
        hist = simulate_history(cfg.reservoir, specs, starts, self._grid(cfg), WALTHER)
        from ..engines.thermal.cycle import cool_profile

        end = hist.results[-1]
        return cool_profile(self._grid(cfg), end.end_profile, gap_days, end.total_days - end.spec.inj_days)

    def simulate_proposed(
        self,
        well_id: str,
        spec_overrides: dict,
        base: str = "next",
        gap_days: float = 20.0,
    ) -> dict:
        cfg = self.fleet[well_id]
        rows = self.cycles(well_id)
        last = rows[-1]
        if base == "current":
            template = row_to_spec(last)
            start_profile = self._profile_after(well_id, last["cycle_no"] - 1, gap_days) if len(rows) > 1 else None
            cno = last["cycle_no"]
        else:
            template = row_to_spec(last)
            start_profile = self._profile_after(well_id, last["cycle_no"], gap_days)
            cno = last["cycle_no"] + 1
        spec = CycleSpec(
            cycle_no=cno,
            steam_m3=float(spec_overrides.get("steam_m3", template.steam_m3)),
            quality=float(spec_overrides.get("quality", template.quality)),
            inj_rate_m3d=float(spec_overrides.get("inj_rate_m3d", template.inj_rate_m3d)),
            inj_pressure_mpa=float(spec_overrides.get("inj_pressure_mpa", template.inj_pressure_mpa)),
            soak_days=float(spec_overrides.get("soak_days", template.soak_days)),
            prod_days=float(spec_overrides.get("prod_days", template.prod_days)),
        )
        r = simulate_cycle(cfg.reservoir, spec, self._grid(cfg), start_profile, WALTHER)
        lf, q, mu, ql, cum, _ = _corrected(self.ml, cfg.reservoir, spec, r)
        soak_end = spec.inj_days + spec.soak_days
        prod = r.phase == PHASE_PRODUCTION
        peak = float(np.max(q[prod])) if prod.any() else 0.0
        econ_limit_day = None
        below = np.flatnonzero(prod & (q < 2.0))
        if below.size:
            econ_limit_day = float(r.t_days[below[0]] - soak_end)
        step = max(1, len(r.t_days) // 200)
        return {
            "spec": {
                "cycle_no": spec.cycle_no,
                "steam_m3": spec.steam_m3,
                "quality": spec.quality,
                "inj_rate_m3d": spec.inj_rate_m3d,
                "inj_pressure_mpa": spec.inj_pressure_mpa,
                "soak_days": spec.soak_days,
                "prod_days": spec.prod_days,
                "inj_days": spec.inj_days,
            },
            "t_days": r.t_days[::step].tolist(),
            "phase": r.phase[::step].tolist(),
            "q_oil_m3d": q[::step].tolist(),
            "q_oil_phys_m3d": r.q_oil_m3d[::step].tolist(),
            "cum_oil_m3": cum[::step].tolist(),
            "cum_oil_phys_m3": r.cum_oil_m3[::step].tolist(),
            "mu_eff_cp": mu[::step].tolist(),
            "t_avg_c": r.t_avg_c[::step].tolist(),
            "summary": {
                "cum_oil_m3": float(cum[-1]),
                "cum_oil_phys_m3": float(r.cum_oil_m3[-1]),
                "peak_rate_m3d": peak,
                "sor": float(spec.steam_m3 / cum[-1]) if cum[-1] > 0 else None,
                "r_heated_m": float(r.r_heated_m[int(np.ceil(spec.inj_days))]),
                "economic_limit_day": econ_limit_day,
                "steam_bh_quality": float(r.steam_info.get("quality_bottomhole", 0.0)),
                "wellbore_loss_pct": float(100.0 * r.steam_info.get("wellbore_loss_fraction", 0.0)),
            },
            "ml_used": self.ml is not None,
        }

    def whatif(self, well_id: str, scenarios: list[dict], base: str = "next") -> dict:
        out = []
        for sc in scenarios:
            sim = self.simulate_proposed(well_id, sc, base=base)
            sim["label"] = sc.get("label") or f"{sc.get('steam_m3', sim['spec']['steam_m3']):,.0f} m³"
            out.append(sim)
        return {"well_id": well_id, "base": base, "scenarios": out, "ml_used": self.ml is not None}

    def sweep_for_planner(self, well_id: str, spec_overrides: dict):
        def run(volume: float) -> dict:
            sim = self.simulate_proposed(well_id, {**spec_overrides, "steam_m3": volume})
            return {"cum_oil_m3": sim["summary"]["cum_oil_m3"], "peak_rate_m3d": sim["summary"]["peak_rate_m3d"]}

        return run


__all__ = ["Twin", "WellSeries", "WALTHER", "PHASE_INJECTION", "PHASE_SOAK", "PHASE_PRODUCTION"]
