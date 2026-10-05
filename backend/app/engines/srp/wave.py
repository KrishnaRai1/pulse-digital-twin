"""Damped 1-D wave equation for the rod string (Gibbs method).

The rod string obeys ``u_tt = a^2 u_xx - c u_t`` with ``T = -EA u_x`` (``u`` positive up,
``x`` positive downwards). For a periodic stroke the solution is a Fourier series; for
every harmonic each rod taper section is a 2x2 transfer matrix relating (displacement,
tension) at its top to those at its bottom:

    [ u ]      [  cosh(g l)            -sinh(g l)/(EA g) ] [ u ]
    [ T ]_bot =[ -EA g sinh(g l)        cosh(g l)        ] [ T ]_top,

    g = sqrt(-w^2 + i c w) / a,       c = 2 zeta w1,      w1 = pi a / (2 L)

so that ``zeta`` is a damping ratio referred to the fundamental rod-string resonance.
Viscous oil raises ``zeta``, which is how the thermal-viscosity engine couples into the
diagnostics. ``surface_to_downhole`` is the industry-standard way of turning a measured
surface dynamometer card into a downhole pump card.
"""
from __future__ import annotations

import numpy as np

from .rods import SOUND_SPEED, STEEL_E_PA, G, SrpDesign

MAX_HARMONICS = 24


def damping_ratio_from_viscosity(mu_pa_s: float) -> float:
    """Empirical map from tubing-average viscosity to the Gibbs damping ratio."""
    return float(np.clip(0.08 + 0.12 * np.log10(1.0 + mu_pa_s / 0.05), 0.05, 0.45))


def _transfer(design: SrpDesign, n_h: int, period_s: float, zeta: float) -> np.ndarray:
    """Transfer matrices, shape (n_h, 2, 2), for harmonics 1..n_h."""
    n = np.arange(1, n_h + 1)
    omega = 2.0 * np.pi * n / period_s
    length = design.total_length_m
    w1 = np.pi * SOUND_SPEED / (2.0 * length)
    c = 2.0 * zeta * w1
    gamma = np.sqrt(-(omega**2) + 1j * c * omega + 0j) / SOUND_SPEED
    m = np.zeros((n_h, 2, 2), dtype=complex)
    m[:, 0, 0] = m[:, 1, 1] = 1.0
    for sec in design.rods:  # top section first
        ea = STEEL_E_PA * sec.area_m2
        gl = gamma * sec.length_m
        ch, sh = np.cosh(gl), np.sinh(gl)
        s = np.zeros((n_h, 2, 2), dtype=complex)
        s[:, 0, 0] = ch
        s[:, 0, 1] = -sh / (ea * gamma)
        s[:, 1, 0] = -ea * gamma * sh
        s[:, 1, 1] = ch
        m = s @ m
    return m


def _harmonics(x: np.ndarray, n_h: int) -> np.ndarray:
    """Complex amplitudes of harmonics 1..n_h (x(t) = mean + sum Re(X_n e^{i n w t}))."""
    coeff = np.fft.rfft(x) * (2.0 / len(x))
    return coeff[1 : n_h + 1]


def _synth(amp: np.ndarray, mean: float, n_pts: int) -> np.ndarray:
    full = np.zeros(n_pts // 2 + 1, dtype=complex)
    full[1 : len(amp) + 1] = amp * (n_pts / 2.0)
    return np.fft.irfft(full, n_pts) + mean


def _n_harm(n_pts: int) -> int:
    return min(MAX_HARMONICS, n_pts // 2 - 1)


def _taper(n_h: int) -> np.ndarray:
    n = np.arange(1, n_h + 1)
    return np.sinc(n / (n_h + 1.0))  # Lanczos sigma factor limits Gibbs ringing


def surface_to_downhole(
    pos_surface: np.ndarray, load_surface: np.ndarray, spm: float, design: SrpDesign, zeta: float, taper: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Downhole (pump) displacement and load from a surface card sampled uniformly in time."""
    n_pts = len(pos_surface)
    n_h = _n_harm(n_pts)
    period = 60.0 / spm
    m = _transfer(design, n_h, period, zeta)
    tp = _taper(n_h) if taper else np.ones(n_h)
    u0 = _harmonics(pos_surface, n_h) * tp
    f0 = _harmonics(load_surface, n_h) * tp
    ul = m[:, 0, 0] * u0 + m[:, 0, 1] * f0
    fl = m[:, 1, 0] * u0 + m[:, 1, 1] * f0
    pos_pump = _synth(ul, float(np.mean(pos_surface)), n_pts)
    load_pump = _synth(fl, float(np.mean(load_surface)) - design.weight_buoyant_n, n_pts)
    return pos_pump, load_pump


def downhole_to_surface(
    pos_pump: np.ndarray, load_pump: np.ndarray, spm: float, design: SrpDesign, zeta: float, taper: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`surface_to_downhole` (det of every transfer matrix is 1)."""
    n_pts = len(pos_pump)
    n_h = _n_harm(n_pts)
    m = _transfer(design, n_h, 60.0 / spm, zeta)
    tp = _taper(n_h) if taper else np.ones(n_h)
    ul = _harmonics(pos_pump, n_h) * tp
    fl = _harmonics(load_pump, n_h) * tp
    u0 = m[:, 1, 1] * ul - m[:, 0, 1] * fl
    f0 = -m[:, 1, 0] * ul + m[:, 0, 0] * fl
    pos_s = _synth(u0, float(np.mean(pos_pump)), n_pts)
    load_s = _synth(f0, float(np.mean(load_pump)) + design.weight_buoyant_n, n_pts)
    return pos_s, load_s


def surface_from_pump_load(
    pos_surface: np.ndarray, load_pump: np.ndarray, spm: float, design: SrpDesign, zeta: float, taper: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward model: prescribed polished-rod motion + valve-controlled pump load.

    Returns ``(surface_load, pump_position, pump_load)``. Solves the bottom boundary
    condition for the surface tension of every harmonic:
    ``F0 = (F_L - M21 u0) / M22``.
    """
    n_pts = len(pos_surface)
    n_h = _n_harm(n_pts)
    m = _transfer(design, n_h, 60.0 / spm, zeta)
    tp = _taper(n_h) if taper else np.ones(n_h)
    u0 = _harmonics(pos_surface, n_h)
    fl = _harmonics(load_pump, n_h) * tp
    f0 = (fl - m[:, 1, 0] * u0) / m[:, 1, 1]
    ul = m[:, 0, 0] * u0 + m[:, 0, 1] * f0
    mean_pump_load = float(np.mean(load_pump))
    load_s = _synth(f0, mean_pump_load + design.weight_buoyant_n, n_pts)
    pos_pump = _synth(ul, float(np.mean(pos_surface)), n_pts)
    load_p = _synth(fl, mean_pump_load, n_pts)
    return load_s, pos_pump, load_p


__all__ = [
    "G",
    "damping_ratio_from_viscosity",
    "surface_to_downhole",
    "downhole_to_surface",
    "surface_from_pump_load",
]
