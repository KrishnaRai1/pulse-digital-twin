"""Traffic-light (Red / Amber / Green) well status.

Status is derived from three independent evidence sources so that the operator can see
*why* a well is flagged:

* CNN diagnosis of the latest dynamometer card
* physics limits (how close the set-point is to the rod-float ceiling)
* thermal health of the heated zone (decay of the CSS cycle)
"""
from __future__ import annotations

from ..engines.optimizer.limits import SpmLimits
from ..engines.srp.cards import CLASS_LABELS, FAULT_SEVERITY

THERMAL_AMBER = 0.45  # fraction of the steam temperature rise still stored in the pay
THERMAL_RED = 0.22
CONFIDENCE_MIN = 0.70

_RANK = {"green": 0, "amber": 1, "red": 2}


def well_status(
    *,
    phase: str,
    label: str | None,
    probability: float,
    thermal_health: float,
    spm: float,
    limits: SpmLimits | None,
    float_index: float | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    level = "green"

    def raise_to(new: str, why: str) -> None:
        nonlocal level
        reasons.append(why)
        if _RANK[new] > _RANK[level]:
            level = new

    if phase == "CYCLE_END":
        raise_to("amber", "Planned production window finished: schedule re-steaming")
        return level, reasons
    if phase != "PRODUCTION":
        return level, reasons  # injecting / soaking wells are not pumping

    if label and probability >= CONFIDENCE_MIN and label != "NORMAL":
        sev = FAULT_SEVERITY.get(label, "amber")
        raise_to(sev, f"{CLASS_LABELS.get(label, label)} ({probability * 100:.0f}% CNN confidence)")
    if limits is not None:
        if not limits.feasible:
            raise_to("red", "No safe pumping speed at current viscosity: re-steam required")
        elif spm > limits.spm_max + 1e-6:
            raise_to("red", f"Set-point {spm:.1f} SPM above the {limits.binding.replace('_', ' ')} ceiling ({limits.spm_max:.2f})")
        elif float_index is not None and float_index > 0.70:
            raise_to("amber", f"Approaching rod-float ceiling (index {float_index:.2f} of 0.85)")
    if thermal_health < THERMAL_RED:
        raise_to("red", f"Heated zone nearly depleted ({thermal_health * 100:.0f}% of thermal capacity left)")
    elif thermal_health < THERMAL_AMBER:
        raise_to("amber", f"Thermal decay: {thermal_health * 100:.0f}% of thermal capacity left")
    return level, reasons
