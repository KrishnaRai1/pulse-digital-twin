"""Field-case physics shared by the dataset pipeline and the SPM advisor.

Every constant comes from ``generation_config.json`` (the parameters the Baghewala dataset
v1.4 was generated with), except the hot-water viscosity used in the oil/water mixture,
which the config does not list. It was recovered from the data (see ``docs/DATASET.md``):
with ``mu_w = 0.31 cP`` the formulas below reproduce the dataset's ``rod_floating_index``
with a median error of 0.4 % (r = 0.9998) and its pump-capacity ceiling exactly.

Formulas
--------
* Oil viscosity          Walther law through the config's two anchors (equals the config table).
* Produced-fluid mixture  log-linear blend  ln mu_mix = (1 - wc) ln mu_oil + wc ln mu_w.
* Rod-fall speed         v_fall = min(v_max, v0 * (mu_ref / mu_mix) ** n).
* Rod speed              v_rod = 2 * stroke * SPM / 60  (mean polished-rod speed).
* Rod-floating index     RFI = v_rod / v_fall; above 1 the rods cannot fall as fast as the
                          polished rod moves down (impact loading, rod-part failures).
* Pump capacity          Q = 0.1166 * D_plunger[in]^2 * stroke[in] * SPM * vol_eff  (bbl/d).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..engines.thermal.viscosity import Walther

BBL_TO_M3 = 0.158987
IN_TO_M = 0.0254
MU_WATER_CP = 0.31  # recovered from the data; not part of generation_config.json


@dataclass(frozen=True)
class FieldCase:
    version: str = "1.4"
    case: str = "baghewala"
    seed: int = 42
    noise_level: float = 0.5
    min_osr: float = 0.12
    n_wells: int = 300
    n_cycles: int = 10
    # pump and drive
    rod_diameter_in: float = 0.875
    plunger_diameter_in: float = 2.0
    stroke_length_in: float = 96.0
    pump_vol_eff: float = 0.8
    spm_min: float = 2.0
    spm_max: float = 9.6
    hz_per_spm: float = 6.25
    motor_rated_kw: float = 18.5
    # rod-float criterion
    rod_float_v0_m_s: float = 0.2
    rod_float_mu_ref_cp: float = 10000.0
    rod_float_exponent: float = 0.25
    rod_float_v_max_m_s: float = 1.2
    # reservoir and fluids
    reservoir_depth_m: float = 1150.0
    pay_thickness_m: float = 20.0
    initial_reservoir_temp_c: float = 47.0
    oil_density_kg_m3: float = 955.0
    visc_anchor: tuple[float, float, float, float] = (11500.0, 50.0, 20.0, 200.0)
    steam_rate_range: tuple[float, float] = (150.0, 250.0)
    steam_quality_range: tuple[float, float] = (0.45, 0.70)
    steam_temp_range_c: tuple[float, float] = (302.0, 330.0)
    injection_days_range: tuple[int, int] = (10, 20)
    soak_days_range: tuple[int, int] = (2, 14)
    production_days_range: tuple[int, int] = (120, 300)
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Path | None) -> FieldCase:
        if path is None or not Path(path).exists():
            return cls()
        cfg = json.loads(Path(path).read_text())
        p = cfg.get("params", {})
        kw: dict = {"raw": cfg}
        for k in ("version", "case", "seed", "noise_level", "min_osr", "n_wells", "n_cycles"):
            if k in cfg:
                kw[k] = cfg[k]
        for f in cls.__dataclass_fields__:
            if f in p:
                v = p[f]
                kw[f] = tuple(v) if isinstance(v, list) else v
        if "steam_quality_range" in p:
            kw["steam_quality_range"] = tuple(p["steam_quality_range"])
        if "visc_anchor" in p:
            kw["visc_anchor"] = tuple(float(x) for x in p["visc_anchor"])
        return cls(**kw)

    # ------------------------------------------------------------------ fluids
    @property
    def walther(self) -> Walther:
        mu1, t1, mu2, t2 = self.visc_anchor
        return Walther.from_two_points(t1, mu1, t2, mu2)

    def oil_viscosity_cp(self, temp_c):
        return self.walther.mu_cp(temp_c)

    @staticmethod
    def mixture_viscosity_cp(mu_oil_cp, water_cut_frac):
        wc = np.clip(np.nan_to_num(np.asarray(water_cut_frac, dtype=float), nan=0.0), 0.0, 0.99)
        mu = np.maximum(np.asarray(mu_oil_cp, dtype=float), 0.3)
        return np.exp((1.0 - wc) * np.log(mu) + wc * np.log(MU_WATER_CP))

    # ------------------------------------------------------------------ rods
    @property
    def stroke_m(self) -> float:
        return self.stroke_length_in * IN_TO_M

    def rod_speed_m_s(self, spm):
        return 2.0 * self.stroke_m * np.asarray(spm, dtype=float) / 60.0

    def rod_fall_speed_m_s(self, mu_mix_cp):
        mu = np.maximum(np.asarray(mu_mix_cp, dtype=float), 1e-3)
        return np.minimum(self.rod_float_v_max_m_s, self.rod_float_v0_m_s * (self.rod_float_mu_ref_cp / mu) ** self.rod_float_exponent)

    def rod_floating_index(self, spm, mu_oil_cp, water_cut_frac):
        mu_mix = self.mixture_viscosity_cp(mu_oil_cp, water_cut_frac)
        return self.rod_speed_m_s(spm) / self.rod_fall_speed_m_s(mu_mix)

    def spm_at_rfi(self, rfi_target: float, mu_oil_cp, water_cut_frac):
        """Pump speed at which the rod-floating index equals ``rfi_target``."""
        v_fall = self.rod_fall_speed_m_s(self.mixture_viscosity_cp(mu_oil_cp, water_cut_frac))
        return rfi_target * v_fall * 60.0 / (2.0 * self.stroke_m)

    # ------------------------------------------------------------------ pump / drive
    @property
    def capacity_per_spm_bbl_d(self) -> float:
        return 0.1166 * self.plunger_diameter_in**2 * self.stroke_length_in * self.pump_vol_eff

    def pump_capacity_bbl_d(self, spm):
        return self.capacity_per_spm_bbl_d * np.asarray(spm, dtype=float)

    def vfd_hz(self, spm):
        return np.asarray(spm, dtype=float) * self.hz_per_spm

    def motor_kw(self, motor_load_pct):
        return np.asarray(motor_load_pct, dtype=float) / 100.0 * self.motor_rated_kw

    def public(self) -> dict:
        """Parameters shown in the UI (no raw config blob)."""
        return {
            "version": self.version,
            "case": self.case,
            "seed": self.seed,
            "noise_level": self.noise_level,
            "min_osr": self.min_osr,
            "pump": {
                "rod_diameter_in": self.rod_diameter_in,
                "plunger_diameter_in": self.plunger_diameter_in,
                "stroke_length_in": self.stroke_length_in,
                "pump_vol_eff": self.pump_vol_eff,
                "spm_range": [self.spm_min, self.spm_max],
                "hz_per_spm": self.hz_per_spm,
                "motor_rated_kw": self.motor_rated_kw,
                "capacity_bbl_d_per_spm": round(self.capacity_per_spm_bbl_d, 3),
            },
            "rod_float": {
                "v0_m_s": self.rod_float_v0_m_s,
                "mu_ref_cp": self.rod_float_mu_ref_cp,
                "exponent": self.rod_float_exponent,
                "v_max_m_s": self.rod_float_v_max_m_s,
                "mu_water_cp_recovered": MU_WATER_CP,
            },
            "reservoir": {
                "depth_m": self.reservoir_depth_m,
                "pay_thickness_m": self.pay_thickness_m,
                "initial_temp_c": self.initial_reservoir_temp_c,
                "oil_density_kg_m3": self.oil_density_kg_m3,
                "viscosity_anchor": {"mu1_cp": self.visc_anchor[0], "t1_c": self.visc_anchor[1], "mu2_cp": self.visc_anchor[2], "t2_c": self.visc_anchor[3]},
            },
            "controls": {
                "steam_rate_m3_d": list(self.steam_rate_range),
                "steam_quality": list(self.steam_quality_range),
                "steam_temp_c": list(self.steam_temp_range_c),
                "injection_days": list(self.injection_days_range),
                "soak_days": list(self.soak_days_range),
                "production_days": list(self.production_days_range),
            },
        }


__all__ = ["FieldCase", "BBL_TO_M3", "MU_WATER_CP"]
