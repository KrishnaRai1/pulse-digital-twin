"""Synthetic "field truth" used to demo and train the ML residual corrector.

Real reservoirs deviate from any reduced-order physics model in systematic ways:
steam override at large volumes, wellbore heat loss on long injections, early-time gas
drive, late-time water encroachment, cumulative depletion, quality effects... This
module encodes such deviations as a hidden log-factor so the simulator can emit
"observed" oil rates that differ from the physics baseline in a *learnable* way.

It is only ever used for the built-in demo data. In a real deployment the residuals
come from field history: build ``build_features`` rows for each historical cycle, use
``ln(q_observed / q_physics)`` as the target and call ``fit_residual_model`` (see docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import zlib

import numpy as np


def well_hidden_factor(well_id: str, sigma: float = 0.12) -> float:
    """Per-well unmodelled heterogeneity (log-normal, deterministic per id)."""
    rng = np.random.default_rng(zlib.crc32(well_id.encode()))
    return float(rng.normal(0.0, sigma))


def truth_log_factor(
    *,
    steam_m3,
    quality,
    inj_days,
    cycle_no,
    t_prod_days,
) -> np.ndarray:
    steam_m3 = np.asarray(steam_m3, dtype=float)
    over = np.clip((steam_m3 - 6500.0) / 3000.0, 0.0, 1.5)  # steam override at high volumes
    g = -0.25 * over
    g = g - 0.15 * np.clip((np.asarray(inj_days, dtype=float) - 30.0) / 30.0, 0.0, 1.5)
    tp = np.asarray(t_prod_days, dtype=float)
    g = g + 0.12 * (1.0 - np.exp(-tp / 30.0)) - 0.15 * (tp / 200.0)  # early gas drive, late water
    g = g - 0.06 * (np.asarray(cycle_no, dtype=float) - 1.0)
    g = g + 0.10 * np.tanh((np.asarray(quality, dtype=float) - 0.70) / 0.10)
    return g


def observed_rate(q_phys, log_factor, well_factor, rng: np.random.Generator, noise: float = 0.05):
    q_phys = np.asarray(q_phys, dtype=float)
    eps = rng.normal(0.0, noise, size=q_phys.shape)
    return q_phys * np.exp(log_factor + well_factor + eps)
