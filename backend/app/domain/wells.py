"""Well fleet definition (synthetic, for demo and testing).

Well names follow the ``BGW-nn`` pattern of the Baghewala CSS field. All numbers here are
**illustrative synthetic parameters**, not field data. Replace ``build_fleet`` with a loader
for your completion database to point the twin at real wells; nothing else depends on it.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from ..engines.srp.rods import RodSection, SrpDesign
from ..engines.thermal.cycle import CycleSpec
from ..engines.thermal.radial import Reservoir

DAY = 86400


@dataclass(frozen=True)
class WellConfig:
    id: str
    name: str
    pad: str
    x_m: float  # schematic pad coordinates
    y_m: float
    reservoir: Reservoir
    design: SrpDesign
    phase0: str  # phase at simulation start: injection | soak | production
    day0: float  # days elapsed inside that phase at simulation start
    cycle_no: int  # number of the current cycle
    steam_m3: float  # current-cycle steam volume
    scenario: str = "healthy"
    gas_prone: bool = False
    net_lift_fraction: float = 0.78
    notes: str = ""


def _design(depth: float, plunger: float, stroke: float) -> SrpDesign:
    total = depth - 40.0
    return SrpDesign(
        rods=(
            RodSection(round(total * 0.30, 1), 25.4),
            RodSection(round(total * 0.35, 1), 22.2),
            RodSection(round(total * 0.35, 1), 19.05),
        ),
        pump_depth_m=total,
        plunger_mm=plunger,
        stroke_m=stroke,
        min_spm=0.5,
    )


def _res(seed_id: str, depth: float, perm: float, thick: float) -> Reservoir:
    rng = np.random.default_rng(zlib.crc32(seed_id.encode()))
    return Reservoir(
        thickness_m=thick,
        t_res_c=float(30.0 + 0.026 * depth * 0.95 + rng.uniform(-1.5, 1.5)),
        perm_md=perm,
        depth_m=depth,
    )


# id, pad, x, y, depth, perm, thickness, plunger, stroke, phase0, day0, cycle_no, steam, scenario, gas
_FLEET = [
    ("BGW-01", "A", 120, 110, 900, 95, 20, 70, 2.7, "production", 32, 3, 5000, "healthy", False),
    ("BGW-02", "A", 300, 90, 930, 85, 18, 57, 2.5, "injection", 11, 2, 4500, "injecting", False),
    ("BGW-03", "A", 210, 240, 960, 90, 22, 57, 3.0, "production", 148, 3, 5200, "cooling_float", False),
    ("BGW-04", "A", 390, 250, 910, 100, 21, 70, 2.7, "production", 58, 2, 4800, "healthy", False),
    ("BGW-05", "B", 640, 120, 1000, 80, 19, 57, 2.5, "soak", 4, 2, 4300, "soaking", False),
    ("BGW-06", "B", 820, 90, 1020, 92, 23, 57, 3.0, "production", 168, 4, 5400, "unset_risk", False),
    ("BGW-07", "B", 700, 250, 980, 88, 20, 57, 2.7, "production", 88, 3, 5000, "fluid_pound", False),
    ("BGW-08", "B", 880, 260, 940, 70, 17, 57, 2.5, "production", 44, 2, 4400, "healthy", False),
    ("BGW-09", "C", 1150, 110, 1040, 96, 22, 57, 3.0, "production", 124, 3, 5300, "thermal_decay", False),
    ("BGW-10", "C", 1330, 100, 890, 105, 21, 70, 2.7, "production", 26, 2, 4900, "healthy", False),
    ("BGW-11", "C", 1230, 250, 970, 84, 19, 57, 2.5, "production", 70, 2, 4600, "gas_interference", True),
    ("BGW-12", "C", 1410, 255, 1010, 91, 21, 57, 3.0, "production", 96, 3, 5100, "healthy", False),
]


def build_fleet() -> list[WellConfig]:
    fleet = []
    for wid, pad, x, y, depth, perm, thick, plg, stroke, ph, d0, cno, steam, scen, gas in _FLEET:
        fleet.append(
            WellConfig(
                id=wid,
                name=wid,
                pad=pad,
                x_m=x,
                y_m=y,
                reservoir=_res(wid, depth, perm, thick),
                design=_design(depth, plg, stroke),
                phase0=ph,
                day0=d0,
                cycle_no=cno,
                steam_m3=steam,
                scenario=scen,
                gas_prone=gas,
            )
        )
    return fleet


def seed_cycles(cfg: WellConfig, now_ts: int, seed: int = 42) -> list[dict]:
    """Cycle history rows (oldest first) ending with the current cycle in progress."""
    rng = np.random.default_rng(zlib.crc32(cfg.id.encode()) ^ seed)
    cur_spec = CycleSpec(
        cycle_no=cfg.cycle_no,
        steam_m3=cfg.steam_m3,
        quality=float(rng.uniform(0.68, 0.80)),
        inj_rate_m3d=float(rng.uniform(170, 230)),
        inj_pressure_mpa=float(rng.uniform(1.6, 2.0)),
        soak_days=7.0,
        prod_days=180.0,
    )
    if cfg.phase0 == "injection":
        elapsed = cfg.day0
    elif cfg.phase0 == "soak":
        elapsed = cur_spec.inj_days + cfg.day0
    else:
        elapsed = cur_spec.inj_days + cur_spec.soak_days + cfg.day0
    start = now_ts - int(elapsed * DAY)
    rows = [_row(cfg.id, cur_spec, start, "seed")]
    next_start = start
    for cno in range(cfg.cycle_no - 1, 0, -1):
        spec = CycleSpec(
            cycle_no=cno,
            steam_m3=float(rng.uniform(3800, 5200)),
            quality=float(rng.uniform(0.66, 0.80)),
            inj_rate_m3d=float(rng.uniform(170, 230)),
            inj_pressure_mpa=float(rng.uniform(1.6, 2.0)),
            soak_days=float(rng.uniform(5, 9)),
            prod_days=180.0,
        )
        length = spec.inj_days + spec.soak_days + spec.prod_days
        gap = float(rng.uniform(12, 30))
        st = next_start - int((length + gap) * DAY)
        rows.append(_row(cfg.id, spec, st, "seed"))
        next_start = st
    rows.reverse()
    return rows


def _row(well_id: str, spec: CycleSpec, start_ts: int, source: str) -> dict:
    return {
        "well_id": well_id,
        "cycle_no": spec.cycle_no,
        "start_ts": int(start_ts),
        "steam_m3": float(spec.steam_m3),
        "quality": float(spec.quality),
        "inj_rate_m3d": float(spec.inj_rate_m3d),
        "inj_pressure_mpa": float(spec.inj_pressure_mpa),
        "soak_days": float(spec.soak_days),
        "prod_days": float(spec.prod_days),
        "source": source,
    }


def row_to_spec(row: dict) -> CycleSpec:
    return CycleSpec(
        cycle_no=int(row["cycle_no"]),
        steam_m3=float(row["steam_m3"]),
        quality=float(row["quality"]),
        inj_rate_m3d=float(row["inj_rate_m3d"]),
        inj_pressure_mpa=float(row["inj_pressure_mpa"]),
        soak_days=float(row["soak_days"]),
        prod_days=float(row["prod_days"]),
    )


__all__ = ["WellConfig", "build_fleet", "seed_cycles", "row_to_spec", "DAY"]
