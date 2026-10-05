"""Dependency container shared by every router (built once in the app lifespan)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request
from sqlalchemy.engine import Engine

from ..config import Settings
from ..engines.srp.cnn_infer import DynoClassifier
from ..engines.thermal.ml_correction import MLCorrector
from ..field.service import FieldDataService
from ..services.diagnostics import DiagnosticsService
from ..services.jobs import JobManager
from ..services.optimizer_service import OptimizerService
from ..services.scada import ScadaAdapter
from ..services.simulator import FieldRuntime
from ..services.stream import StreamHub
from ..services.twin import Twin


@dataclass
class Container:
    settings: Settings
    engine: Engine
    twin: Twin
    runtime: FieldRuntime
    classifier: DynoClassifier | None
    ml: MLCorrector | None
    optimizer: OptimizerService
    diagnostics: DiagnosticsService
    jobs: JobManager
    hub: StreamHub
    scada: ScadaAdapter
    field: FieldDataService
    started_ts: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


def get_container(request: Request) -> Container:
    return request.app.state.container


def well_or_404(c: Container, well_id: str) -> str:
    wid = well_id.upper()
    if wid not in c.twin.fleet:
        raise HTTPException(status_code=404, detail=f"unknown well '{well_id[:16]}'")
    return wid
