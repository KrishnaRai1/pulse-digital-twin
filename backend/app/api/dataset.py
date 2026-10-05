"""Field dataset (Baghewala synthetic v1.4) endpoints: field history, well history, models,
SPM advisor, CSS cycle planner, early warning and data validation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import Field

from ..field.service import FieldUnavailable
from ..schemas import Strict
from ..security import limit_compute, limit_jobs, require_api_key
from .deps import Container, get_container

router = APIRouter(prefix="/dataset", tags=["field dataset"])

DOWNLOADABLE = (
    "generation_config.json", "validation_report.csv", "early_warning_report.json", "rod_failure_log.csv",
    "MANIFEST.json", "validation_plots.jpg", "validation_plots.png",
)


def _store(c: Container):
    try:
        return c.field.require()
    except FieldUnavailable as exc:
        raise HTTPException(status_code=503, detail=exc.status) from None


def _well(fn, *args):
    try:
        return fn(*args)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown well") from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------- status & field
@router.get("/status")
def status(c: Container = Depends(get_container)) -> dict:
    return c.field.status()


@router.get("/summary")
def summary(c: Container = Depends(get_container)) -> dict:
    return _store(c).summary()


@router.get("/field")
def field_snapshot(day: int | None = Query(None, ge=0, le=100_000), c: Container = Depends(get_container)) -> dict:
    """State of all wells on one day: RAG status, rates, pump state, failure risk, field KPIs."""
    s = _store(c)
    return s.snapshot(s.clamp_day(day))


@router.get("/field/trend")
def field_trend(step: int = Query(1, ge=1, le=30), c: Container = Depends(get_container)) -> dict:
    return _store(c).trend(step)


@router.get("/wells")
def wells(c: Container = Depends(get_container)) -> list[dict]:
    return _store(c).wells_table()


# ---------------------------------------------------------------- well level
@router.get("/wells/{well_id}")
def well(well_id: str, c: Container = Depends(get_container)) -> dict:
    return _well(_store(c).well, well_id)


@router.get("/wells/{well_id}/daily")
def well_daily(
    well_id: str,
    cycle: int | None = Query(None, ge=1, le=100),
    start: int | None = Query(None, ge=0),
    end: int | None = Query(None, ge=0),
    c: Container = Depends(get_container),
) -> dict:
    return _well(_store(c).well_daily, well_id, cycle, start, end)


@router.get("/wells/{well_id}/risk")
def well_risk(well_id: str, day: int | None = Query(None, ge=0), c: Container = Depends(get_container)) -> dict:
    s = _store(c)
    return _well(s.risk_drivers, well_id, s.clamp_day(day))


@router.get("/wells/{well_id}/spm", dependencies=[Depends(limit_compute)])
def spm_advice(well_id: str, day: int | None = Query(None, ge=0), c: Container = Depends(get_container)) -> dict:
    """SPM recommendation for one well-day with the binding constraint and the reasons."""
    return _well(_store(c).spm_advice, well_id, day)


@router.get("/wells/{well_id}/spm-history")
def spm_history(well_id: str, cycle: int | None = Query(None, ge=1, le=100), c: Container = Depends(get_container)) -> dict:
    """Logged vs. advised SPM over a well's production days (the advisor replayed day by day)."""
    return _well(_store(c).spm_history, well_id, cycle)


@router.get("/wells/{well_id}/cycle-plan", dependencies=[Depends(limit_compute)])
async def cycle_plan(
    well_id: str,
    cycle: int | None = Query(None, ge=1, le=100, description="cycle to (re)plan; default: the next cycle"),
    objective: Literal["oil", "margin", "osr"] = "oil",
    steam_budget_m3d: float | None = Query(None, ge=1, le=60, description="cap on steam per cycle-day (m3/d); default: the reference cycle's"),
    unconstrained: bool = Query(False, description="ignore the steam budget"),
    quality_max: float | None = Query(None, ge=0.45, le=0.70),
    temp_max_c: float | None = Query(None, ge=302, le=330),
    c: Container = Depends(get_container),
) -> dict:
    """Best next-cycle (or re-planned) CSS settings within a steam budget, with the reasons
    (TreeSHAP differences versus the reference settings) and an oil-vs-steam frontier."""
    s = _store(c)
    return await run_in_threadpool(_well, s.cycle_plan, well_id, cycle, objective, steam_budget_m3d, unconstrained, quality_max, temp_max_c)


class CycleScenario(Strict):
    label: str | None = Field(default=None, max_length=40)
    steam_rate_m3d: float | None = Field(default=None, ge=150, le=250)
    inj_days: int | None = Field(default=None, ge=10, le=20)
    steam_quality: float | None = Field(default=None, ge=0.45, le=0.70)
    steam_temp_c: float | None = Field(default=None, ge=302, le=330)
    soak_days: int | None = Field(default=None, ge=2, le=14)
    prod_days: int | None = Field(default=None, ge=120, le=300)


class CycleWhatIf(Strict):
    cycle: int | None = Field(default=None, ge=1, le=100)
    scenarios: list[CycleScenario] = Field(min_length=1, max_length=6)


@router.post("/wells/{well_id}/cycle-whatif", dependencies=[Depends(limit_compute)])
def cycle_whatif(well_id: str, body: CycleWhatIf, c: Container = Depends(get_container)) -> dict:
    """Read-only what-if: predicted oil per cycle-day, SOR and margin for up to six control sets.

    Ranges are those covered by the data (the model is not trusted outside them)."""
    scen = [s.model_dump() for s in body.scenarios]
    return _well(_store(c).cycle_whatif, well_id, body.cycle, scen)


# ---------------------------------------------------------------- models & early warning
@router.get("/models")
def models(c: Container = Depends(get_container)) -> dict:
    return _store(c).models()


@router.get("/early-warning")
def early_warning(c: Container = Depends(get_container)) -> dict:
    s = _store(c)
    m = s.models()["early_warning"]
    base = Path(c.settings.field_data_dir) / "early_warning_report.json"
    m["baseline_report"] = json.loads(base.read_text()) if base.is_file() else None
    return m


@router.get("/early-warning/alerts")
def alerts(day: int | None = Query(None, ge=0), limit: int = Query(25, ge=1, le=300), c: Container = Depends(get_container)) -> dict:
    s = _store(c)
    return s.alerts(s.clamp_day(day), limit)


# ---------------------------------------------------------------- data validation & files
@router.get("/validation")
def validation(c: Container = Depends(get_container)) -> dict:
    d = Path(c.settings.field_data_dir)
    rows = []
    vr = d / "validation_report.csv"
    if vr.is_file():
        with vr.open(newline="") as f:
            rows = list(csv.DictReader(f))
    manifest = json.loads((d / "MANIFEST.json").read_text()) if (d / "MANIFEST.json").is_file() else {}
    store = c.field.store
    case = store.case.public() if store else None
    files = [{"name": p.name, "bytes": p.stat().st_size, "downloadable": p.name in DOWNLOADABLE} for p in sorted(d.iterdir()) if p.is_file()] if d.is_dir() else []
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("status", "")] = counts.get(r.get("status", ""), 0) + 1
    return {
        "checks": rows,
        "status_counts": counts,
        "manifest": manifest,
        "field_case": case,
        "files": files,
        "plot": next((n for n in ("validation_plots.png", "validation_plots.jpg") if (d / n).is_file()), None),
        "rules": [
            "Train/test splits are by well, never by row (5-fold cross-validation grouped by well).",
            "condition_label and the failure log are labels only, never model inputs.",
            "reservoir_temp_c and oil_viscosity_cp (latent states) are never used to predict production rate.",
        ],
    }


@router.get("/files/{name}")
def download(name: str, c: Container = Depends(get_container)):
    d = Path(c.settings.field_data_dir)
    if name not in DOWNLOADABLE or not (d / name).is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(d / name, filename=name)


@router.post("/rebuild", status_code=202, dependencies=[Depends(require_api_key), Depends(limit_jobs)])
def rebuild(c: Container = Depends(get_container)):
    """Rebuild the derived store and retrain the dataset models from ``data/field`` (background)."""
    if not c.field.raw_present():
        raise HTTPException(status_code=409, detail="dataset files are missing")
    started = c.field.rebuild()
    return JSONResponse(status_code=202 if started else 409, content=c.field.status())
