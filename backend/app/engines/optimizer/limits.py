"""Physics-informed safety layer: hard limits on the pumping speed (SPM).

Every set-point, whoever proposes it (MPC, operator, replay of a legacy value), is checked
against these limits before it can be recommended, approved or written to the SCADA
adapter. The ML/optimisation layers can only choose *inside* this envelope.

Limits (all monotone in SPM, so each is found by bisection):

* **VFD/drive range**      0 .. ``design.max_spm``
* **Rod float**            float index <= 0.85 (rods must fall at least as fast as the
                           polished rod, with a 15 % margin)
* **Rod fatigue**          API modified-Goodman utilisation <= 0.90 at the top taper
* **Unit structure**       PPRL <= 95 % of the pumping-unit rating
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..srp import loads
from ..srp.rods import SrpDesign

FATIGUE_LIMIT = 0.90
STRUCTURE_LIMIT = 0.95


def _max_ok(is_ok: Callable[[float], bool], lo: float, hi: float) -> float:
    """Largest x in [lo, hi] with is_ok(x) for a predicate that is True then False."""
    if is_ok(hi):
        return hi
    if not is_ok(lo):
        return 0.0  # infeasible even at the lower bound
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if is_ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class SpmLimits:
    spm_drive: float
    spm_float: float
    spm_fatigue: float
    spm_structure: float
    spm_max: float = 0.0
    binding: str = "drive"
    feasible: bool = True
    spm_min: float = 1.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "drive": self.spm_drive,
            "rod_float": self.spm_float,
            "rod_fatigue": self.spm_fatigue,
            "unit_structure": self.spm_structure,
            "max_allowed": self.spm_max,
            "binding": self.binding,
            "feasible": self.feasible,
            "min": self.spm_min,
        }


def spm_limits(design: SrpDesign, mu_pa_s: float, fluid_load_n: float) -> SpmLimits:
    hi = design.max_spm
    lo = 0.05

    def float_ok(n: float) -> bool:
        return loads.float_index(n, mu_pa_s, design) <= loads.FLOAT_MARGIN

    def fatigue_ok(n: float) -> bool:
        pprl, mprl = loads.polished_rod_loads(n, mu_pa_s, design, fluid_load_n)
        return loads.goodman_utilisation(pprl, mprl, design)["utilisation"] <= FATIGUE_LIMIT

    def structure_ok(n: float) -> bool:
        pprl, _ = loads.polished_rod_loads(n, mu_pa_s, design, fluid_load_n)
        return loads.structure_utilisation(pprl, design) <= STRUCTURE_LIMIT

    lim = SpmLimits(
        spm_drive=hi,
        spm_float=_max_ok(float_ok, lo, hi),
        spm_fatigue=_max_ok(fatigue_ok, lo, hi),
        spm_structure=_max_ok(structure_ok, lo, hi),
        spm_min=design.min_spm,
    )
    candidates = {
        "drive": lim.spm_drive,
        "rod_float": lim.spm_float,
        "rod_fatigue": lim.spm_fatigue,
        "unit_structure": lim.spm_structure,
    }
    lim.binding = min(candidates, key=candidates.get)
    lim.spm_max = candidates[lim.binding]
    lim.feasible = lim.spm_max >= lim.spm_min
    if not lim.feasible:
        lim.notes.append(
            f"{lim.binding.replace('_', ' ')} limit ({lim.spm_max:.2f} SPM) is below the minimum pumping speed "
            f"({lim.spm_min:.1f} SPM): re-heat (re-steam) the well or stop pumping."
        )
    return lim


def check_setpoint(spm: float, design: SrpDesign, mu_pa_s: float, fluid_load_n: float) -> list[str]:
    """Violations of the hard limits for a candidate set-point (empty list = safe)."""
    lim = spm_limits(design, mu_pa_s, fluid_load_n)
    problems = []
    if spm > lim.spm_drive + 1e-6:
        problems.append(f"{spm:.2f} SPM exceeds the drive maximum ({lim.spm_drive:.1f}).")
    if spm > lim.spm_float + 1e-6:
        problems.append(f"{spm:.2f} SPM exceeds the rod-float ceiling ({lim.spm_float:.2f}).")
    if spm > lim.spm_fatigue + 1e-6:
        problems.append(f"{spm:.2f} SPM exceeds the rod-fatigue ceiling ({lim.spm_fatigue:.2f}).")
    if spm > lim.spm_structure + 1e-6:
        problems.append(f"{spm:.2f} SPM exceeds the unit structure ceiling ({lim.spm_structure:.2f}).")
    if spm < 0:
        problems.append("negative speed")
    return problems
