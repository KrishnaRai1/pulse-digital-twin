"""Sucker-rod-pump (SRP) design data and rod-string mechanics."""
from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81
STEEL_E_PA = 2.07e11
STEEL_RHO = 7850.0
SOUND_SPEED = math.sqrt(STEEL_E_PA / STEEL_RHO)  # ~5135 m/s


@dataclass(frozen=True)
class RodSection:
    length_m: float
    dia_mm: float

    @property
    def area_m2(self) -> float:
        return math.pi * (self.dia_mm * 1e-3) ** 2 / 4.0

    @property
    def mass_kg(self) -> float:
        return STEEL_RHO * self.area_m2 * self.length_m


@dataclass(frozen=True)
class SrpDesign:
    rods: tuple[RodSection, ...]
    pump_depth_m: float
    plunger_mm: float
    stroke_m: float
    tubing_id_mm: float = 75.9
    max_spm: float = 12.0  # SPM at 60 Hz VFD output
    min_spm: float = 1.0
    unit_rating_kn: float = 95.0  # pumping-unit structure rating
    motor_kw_rated: float = 30.0
    rod_tensile_mpa: float = 620.0  # API grade C
    service_factor: float = 0.90  # modified-Goodman service factor (non-corrosive)
    fluid_density: float = 940.0

    @property
    def total_length_m(self) -> float:
        return sum(r.length_m for r in self.rods)

    @property
    def weight_air_n(self) -> float:
        return sum(r.mass_kg for r in self.rods) * G

    @property
    def weight_buoyant_n(self) -> float:
        return self.weight_air_n * (1.0 - self.fluid_density / STEEL_RHO)

    @property
    def plunger_area_m2(self) -> float:
        return math.pi * (self.plunger_mm * 1e-3) ** 2 / 4.0

    @property
    def top_area_m2(self) -> float:
        return self.rods[0].area_m2

    def fluid_load_n(self, net_lift_m: float) -> float:
        return self.plunger_area_m2 * self.fluid_density * G * net_lift_m

    def pump_capacity_m3d(self, spm: float, efficiency: float = 0.90) -> float:
        """Displacement of the pump at ``spm`` (m3/d) with 8 % stroke loss to rod stretch."""
        return self.plunger_area_m2 * self.stroke_m * 0.92 * spm * 1440.0 * efficiency
