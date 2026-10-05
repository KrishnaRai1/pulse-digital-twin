"""Heavy-oil viscosity model.

* Temperature dependence: ASTM D341 / Walther, calibrated from two lab points.
* Water-oil emulsification (steam condensate is produced with the oil): Richardson
  exponential below the phase-inversion point, relaxing towards a water-continuous
  emulsion above it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MIN_T_C = 5.0
_MAX_T_C = 350.0


@dataclass(frozen=True)
class Walther:
    """log10(log10(mu + 0.7)) = A - B * log10(T[K])."""

    a: float
    b: float

    @classmethod
    def from_two_points(cls, t1_c: float, mu1_cp: float, t2_c: float, mu2_cp: float) -> Walther:
        z1 = np.log10(np.log10(mu1_cp + 0.7))
        z2 = np.log10(np.log10(mu2_cp + 0.7))
        l1 = np.log10(t1_c + 273.15)
        l2 = np.log10(t2_c + 273.15)
        b = (z1 - z2) / (l2 - l1)
        a = z1 + b * l1
        return cls(float(a), float(b))

    def mu_cp(self, t_c):
        t = np.clip(np.asarray(t_c, dtype=float), _MIN_T_C, _MAX_T_C)
        z = self.a - self.b * np.log10(t + 273.15)
        mu = 10.0 ** (10.0**z) - 0.7
        return np.maximum(mu, 0.3)


def emulsion_factor(water_cut) -> np.ndarray:
    """Viscosity multiplier of an oil/water emulsion relative to dry oil."""
    phi = np.clip(np.asarray(water_cut, dtype=float), 0.0, 0.98)
    rising = np.exp(2.5 * phi)  # oil-continuous
    peak = np.exp(2.5 * 0.55)
    # linear-in-log decay from the inversion peak (phi=0.55) to 1.5x at phi=0.80
    frac = np.clip((phi - 0.55) / 0.25, 0.0, 1.0)
    inverted = np.exp(np.log(peak) * (1 - frac) + np.log(1.5) * frac)
    out = np.where(phi <= 0.55, rising, inverted)
    return np.where(phi > 0.8, 1.2 + (0.98 - phi) * 1.5, out)
