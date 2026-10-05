"""PULSE application factory.

``uvicorn app.main:app`` starts the API. On start-up the service

1. opens the database (SQLite by default, PostgreSQL/TimescaleDB via ``PULSE_DATABASE_URL``),
2. loads (or, on first run, trains) the CNN and the LightGBM residual model,
3. loads the field-dataset store (300-well Baghewala data), building it in the background on
   first run or when the data in ``data/field`` changed,
4. builds the digital-twin fleet, warm-starts the field runtime and starts the simulator loop
   that stands in for SCADA/historian data (disable with ``PULSE_SIM_ENABLED=false``).
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import db
from .api import api_router, ws_router
from .api.deps import Container
from .config import Settings, get_settings
from .domain.wells import build_fleet
from .engines.srp.cnn_infer import DynoClassifier
from .engines.thermal import ml_correction
from .engines.thermal.ml_correction import MLCorrector
from .field.service import FieldDataService
from .security import SecurityHeaders
from .services.diagnostics import DiagnosticsService
from .services.jobs import JobManager
from .services.optimizer_service import OptimizerError, OptimizerService
from .services.scada import SimulatedScada
from .services.simulator import FieldRuntime
from .services.stream import StreamHub
from .services.twin import Twin

log = logging.getLogger("pulse")
VERSION = "1.0.0"


def load_or_train_models(settings: Settings) -> tuple[DynoClassifier | None, MLCorrector | None]:
    md = Path(settings.models_dir)
    clf, ml = DynoClassifier.load(md), MLCorrector.load(md)
    if settings.auto_train:
        if clf is None:
            try:
                from .engines.srp import cnn as cnn_mod  # noqa: PLC0415 (PyTorch is a training-only dependency)
            except ImportError:
                log.error("CNN weights missing and PyTorch is not installed: run `python -m scripts.train_models` "
                          "in an environment with requirements-train.txt")
            else:
                log.warning("CNN weights not found in %s: training on synthetic cards (~1 min)", md)
                cnn_mod.train(md)
                clf = DynoClassifier.load(md)
        if ml is None:
            log.warning("thermal residual model not found in %s: training on synthetic cycles", md)
            ml_correction.train_synthetic(md)
            ml = MLCorrector.load(md)
    if clf is None:
        log.error("CNN unavailable: dynamometer classification is disabled")
    if ml is None:
        log.warning("ML residual model unavailable: the twin runs on physics only")
    return clf, ml


def build_container(settings: Settings) -> Container:
    if settings.is_sqlite:
        Path(settings.database_url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
    engine = db.make_engine(settings.database_url)
    if settings.reset_demo_data:
        db.metadata.drop_all(engine)
    db.init_db(engine)
    clf, ml = load_or_train_models(settings)
    fleet = build_fleet()
    twin = Twin(engine, {w.id: w for w in fleet}, ml)
    hub = StreamHub()
    runtime = FieldRuntime(settings, engine, twin, clf, hub, fleet)
    scada = SimulatedScada(runtime)
    optimizer = OptimizerService(settings, engine, twin, scada, hub)
    optimizer.runtime = runtime  # type: ignore[attr-defined]
    runtime.advisory_hook = lambda well_id, ts: optimizer.create_advisory(well_id, ts)
    return Container(
        settings=settings,
        engine=engine,
        twin=twin,
        runtime=runtime,
        classifier=clf,
        ml=ml,
        optimizer=optimizer,
        diagnostics=DiagnosticsService(engine, twin, runtime, clf, hub),
        jobs=JobManager(engine, hub),
        hub=hub,
        scada=scada,
        field=FieldDataService(settings),
        started_ts=int(time.time()),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = await asyncio.to_thread(build_container, settings)
        app.state.container = container
        container.hub.bind_loop(asyncio.get_running_loop())
        container.field.start()  # loads or builds the field-dataset store in the background
        await asyncio.to_thread(container.runtime.warm_start)
        task = asyncio.create_task(container.runtime.run()) if settings.sim_enabled else None
        log.info("PULSE %s ready (%s mode, %d wells, simulator %s)", VERSION, settings.control_mode, len(container.twin.fleet), "on" if task else "off")
        try:
            yield
        finally:
            if task:
                task.cancel()
            container.jobs.shutdown()
            container.engine.dispose()

    app = FastAPI(
        title="PULSE: Predictive Unified Lift & Steam Engine",
        version=VERSION,
        description="Digital twin and advisory optimiser for cyclic steam stimulation (CSS) wells with sucker-rod pumps.",
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    app.add_middleware(SecurityHeaders)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_origin_regex=settings.cors_origin_regex,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
        max_age=600,
    )

    @app.exception_handler(OptimizerError)
    async def _optimizer_error(_: Request, exc: OptimizerError):
        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        # Report where and why, never echo the submitted values back (they may be huge or non-finite JSON).
        detail = [{"loc": list(e.get("loc", ())), "msg": str(e.get("msg", "")), "type": e.get("type", "")} for e in exc.errors()[:20]]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.exception_handler(LookupError)
    async def _lookup_error(_: Request, exc: LookupError):
        return JSONResponse(status_code=404, content={"detail": "resource not found"})

    app.include_router(api_router)
    app.include_router(ws_router)

    dist = Path(settings.frontend_dist) if settings.frontend_dist else None
    if dist and (dist / "index.html").is_file():
        root = dist.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith(("api/", "ws/")):
                return JSONResponse(status_code=404, content={"detail": "not found"})
            target = (root / path).resolve()
            if path and target.is_file() and root in target.parents:
                return FileResponse(target)
            return FileResponse(root / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def index():
            return {"service": "PULSE", "version": VERSION, "docs": "/docs", "api": "/api/v1"}

    return app


app = create_app()
