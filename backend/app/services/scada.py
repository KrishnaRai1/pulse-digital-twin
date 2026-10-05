"""SCADA / PLC write path.

The optimiser never talks to hardware directly. Approved set-points go through a
``ScadaAdapter``. Only ``SimulatedScada`` is implemented here (it changes the set-point
the field simulator uses). A production deployment implements the same three methods
against OPC UA / Modbus / the RTU vendor API and is selected in ``main.py``; the
advisory -> approval -> safety-check -> adapter flow stays identical, which is the
phased path from *advisory* mode to *closed-loop* control.
"""
from __future__ import annotations

from typing import Protocol


class ScadaAdapter(Protocol):
    def read_setpoint(self, well_id: str) -> float: ...

    def write_setpoint(self, well_id: str, spm: float, actor: str) -> bool: ...


class SimulatedScada:
    def __init__(self, runtime) -> None:
        self._rt = runtime

    def read_setpoint(self, well_id: str) -> float:
        return float(self._rt.setpoints[well_id])

    def write_setpoint(self, well_id: str, spm: float, actor: str) -> bool:
        self._rt.set_setpoint(well_id, float(spm))
        return True
