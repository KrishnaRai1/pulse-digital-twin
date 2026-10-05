"""Dynamometer card generation, alignment and featurisation.

Surface cards are produced by the wave-equation forward model: prescribed harmonic
polished-rod motion plus a valve-controlled pump load. Fault archetypes:

* ``NORMAL``               full pump, crisp load transfer
* ``FLUID_POUND``          partial fill: plunger strikes the fluid column late in the downstroke
* ``GAS_INTERFERENCE``     slow, rounded load transfer and reduced net load
* ``ROD_FLOATING``         rods cannot fall as fast as the polished rod (viscous drag): the
                           downstroke load collapses towards zero, followed by an impact spike
* ``PUMP_UNSETTING_RISK``  severe float: compressive loads, double impact and a load-leak
                           signature that precedes hold-down unseating

These are *synthetic archetypes* meant to bootstrap the classifier. Replace/augment them
with labelled field cards (``python -m scripts.train_models --cards labelled_cards.csv``, see docs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rods import RodSection, SrpDesign
from .wave import damping_ratio_from_viscosity, surface_from_pump_load, surface_to_downhole

N_POINTS = 128
CLASSES = ["NORMAL", "ROD_FLOATING", "PUMP_UNSETTING_RISK", "FLUID_POUND", "GAS_INTERFERENCE"]
CLASS_LABELS = {
    "NORMAL": "Normal",
    "ROD_FLOATING": "Rod Floating Detected",
    "PUMP_UNSETTING_RISK": "Pump Unsetting Risk",
    "FLUID_POUND": "Fluid Pound",
    "GAS_INTERFERENCE": "Gas Interference",
}
FAULT_SEVERITY = {  # traffic-light severity of each class
    "NORMAL": "green",
    "ROD_FLOATING": "red",
    "PUMP_UNSETTING_RISK": "red",
    "FLUID_POUND": "amber",
    "GAS_INTERFERENCE": "amber",
}


@dataclass
class Card:
    position: np.ndarray  # surface polished-rod position (m)
    load: np.ndarray  # surface polished-rod load (N)
    dh_position: np.ndarray  # downhole pump displacement (m)
    dh_load: np.ndarray  # downhole pump load (N)
    spm: float
    label: str = "NORMAL"


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _window(theta, a, b, edge=0.25):
    return _smoothstep((theta - a) / edge) * (1.0 - _smoothstep((theta - b) / edge))


def _ring(theta, start, amp, decay, freq):
    """Decaying oscillation starting at ``start`` (wraps around the cycle)."""
    d = np.mod(theta - start, 2 * np.pi)
    return amp * np.exp(-d / decay) * np.cos(freq * d) * (d < 2.2)


def stroke_position(stroke_m: float, n: int = N_POINTS) -> tuple[np.ndarray, np.ndarray]:
    theta = 2.0 * np.pi * np.arange(n) / n
    return theta, 0.5 * stroke_m * (1.0 - np.cos(theta))


def generate_card(
    design: SrpDesign,
    spm: float,
    mu_pa_s: float,
    net_lift_m: float,
    label: str = "NORMAL",
    severity: float = 0.7,
    fillage: float = 1.0,
    rng: np.random.Generator | None = None,
    noise: float = 0.004,
    n: int = N_POINTS,
) -> Card:
    rng = rng or np.random.default_rng(0)
    theta, pos = stroke_position(design.stroke_m, n)
    zeta = damping_ratio_from_viscosity(mu_pa_s)
    wb = design.weight_buoyant_n
    f_fl = design.fluid_load_n(net_lift_m)

    th_r, w_r = 0.22 + rng.uniform(0, 0.14), 0.08 + rng.uniform(0, 0.07)
    th_f, w_f = np.pi + 0.12 + rng.uniform(0, 0.12), 0.08 + rng.uniform(0, 0.07)
    if label == "FLUID_POUND":
        f = float(np.clip(fillage, 0.08, 0.97))
        th_f = min(2 * np.pi - np.arccos(1.0 - 2.0 * f), 2 * np.pi - 0.35)
        w_f = 0.05 + rng.uniform(0, 0.02)
    elif label == "GAS_INTERFERENCE":
        th_r, w_r = rng.uniform(0.32, 0.85), rng.uniform(0.16, 0.40)
        th_f, w_f = np.pi + rng.uniform(0.32, 0.85), rng.uniform(0.16, 0.40)
        f_fl *= rng.uniform(0.78, 0.97)
    elif label == "PUMP_UNSETTING_RISK":
        f_fl *= rng.uniform(0.86, 0.95)

    f_pump = f_fl * (_sig((theta - th_r) / w_r) - _sig((theta - th_f) / w_f))
    load, _, _ = surface_from_pump_load(pos, f_pump, spm, design, zeta)

    if label == "ROD_FLOATING":
        s = float(np.clip(severity, 0.03, 1.0))
        m = _window(theta, np.pi + 0.10, 2 * np.pi - 0.60) * s
        load = load * (1 - m) + 0.08 * wb * m
        load = load + _ring(theta, 2 * np.pi - 0.55, (0.06 + 0.65 * s) * wb, 0.28, 8.5)
    elif label == "PUMP_UNSETTING_RISK":
        s = float(np.clip(severity, 0.03, 1.0))
        m = _window(theta, np.pi + 0.05, 2 * np.pi - 0.35) * (0.5 + 0.5 * s)
        load = load * (1 - m) + (-(0.02 + 0.18 * s) * wb) * m
        load = load * (1.0 - (0.02 + 0.12 * s) * _window(theta, 0.4, np.pi - 0.1))
        load = load + _ring(theta, 2 * np.pi - 0.40, (0.15 + 1.35 * s) * wb, 0.30, 9.5)
        load = load + _ring(theta, np.pi - 0.05, (0.05 + 0.5 * s) * wb, 0.22, 11.0)

    load = load + rng.normal(0.0, noise * wb, size=n)
    pos_n = pos + rng.normal(0.0, noise * 0.2 * design.stroke_m, size=n)
    dh_pos, dh_load = surface_to_downhole(pos_n, load, spm, design, zeta)
    return Card(pos_n, load, dh_pos, dh_load, spm, label)


def baseline_card(design: SrpDesign, spm: float, mu_pa_s: float, net_lift_m: float) -> Card:
    """Model-expected healthy card (deterministic, noise-free) for the same conditions."""
    return generate_card(design, spm, mu_pa_s, net_lift_m, "NORMAL", rng=np.random.default_rng(7), noise=0.0)


# --------------------------------------------------------------------------- pre-processing
def resample_periodic(x: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    """Resample a periodic signal (one stroke cycle) to ``n`` uniform samples."""
    x = np.asarray(x, dtype=float)
    if len(x) == n:
        return x
    src = np.arange(len(x)) / len(x)
    dst = np.arange(n) / n
    return np.interp(dst, src, x, period=1.0)


def align_to_bottom(pos: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Rotate arrays so index 0 is the bottom of the stroke (minimum polished-rod position)."""
    k = int(np.argmin(pos))
    return tuple(np.roll(a, -k) for a in (pos, *others))


def analyze_surface_card(
    pos: np.ndarray, load: np.ndarray, spm: float, design: SrpDesign, mu_pa_s: float = 0.2
) -> Card:
    """Measured surface card -> aligned 128-point card with wave-equation downhole card."""
    pos = resample_periodic(pos)
    load = resample_periodic(load)
    pos, load = align_to_bottom(pos, load)
    dh_pos, dh_load = surface_to_downhole(pos, load, spm, design, damping_ratio_from_viscosity(mu_pa_s))
    return Card(pos, load, dh_pos, dh_load, spm, "UNKNOWN")


def card_tensor(card: Card, design: SrpDesign) -> np.ndarray:
    """(4, 128) network input: normalised surface/downhole position and load."""
    wb = design.weight_buoyant_n

    def norm_pos(p):
        rng_ = float(np.ptp(p))
        return (p - p.min()) / rng_ if rng_ > 1e-9 else np.zeros_like(p)

    return np.stack(
        [
            norm_pos(card.position),
            card.load / wb,
            norm_pos(card.dh_position),
            card.dh_load / wb,
        ]
    ).astype(np.float32)


def random_design(rng: np.random.Generator) -> SrpDesign:
    total = rng.uniform(700, 1300)
    fr = np.array([rng.uniform(0.25, 0.38), rng.uniform(0.30, 0.38)])
    fr = np.append(fr, 1 - fr.sum())
    dias = rng.choice([[25.4, 22.2, 19.05], [22.2, 19.05, 15.9], [25.4, 22.2, 19.05]])
    return SrpDesign(
        rods=tuple(RodSection(float(total * f), float(d)) for f, d in zip(fr, dias)),
        pump_depth_m=float(total),
        plunger_mm=float(rng.choice([57.0, 70.0, 83.0])),
        stroke_m=float(rng.uniform(1.8, 3.2)),
    )
