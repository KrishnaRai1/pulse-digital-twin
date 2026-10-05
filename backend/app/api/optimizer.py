"""Optimiser & explainable-AI endpoints: recommendation, advisory workflow, audit trail, cycle planner."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ..schemas import CyclePlanRequest, DecisionIn
from ..security import limit_compute, limit_jobs, require_api_key
from ..services import storage
from ..services.optimizer_service import OptimizerError
from .deps import Container, get_container, well_or_404

router = APIRouter(tags=["optimizer"])


@router.get("/wells/{well_id}/recommendation", dependencies=[Depends(limit_compute)])
def recommendation(well_id: str, c: Container = Depends(get_container)) -> dict:
    """Current MPC recommendation with the plan, the safety envelope and the XAI explanation.

    Read-only: nothing is stored or applied."""
    wid = well_or_404(c, well_id)
    return c.optimizer.recommend(wid, c.runtime.sim_ts)


@router.post("/wells/{well_id}/advisories", status_code=201, dependencies=[Depends(require_api_key), Depends(limit_compute)])
def create_advisory(well_id: str, force: bool = Query(False), c: Container = Depends(get_container)) -> dict:
    wid = well_or_404(c, well_id)
    adv = c.optimizer.create_advisory(wid, c.runtime.sim_ts, actor="operator-request", force=force)
    if adv is None:
        raise HTTPException(status_code=409, detail="no adjustment is recommended for this well right now")
    return adv


@router.get("/advisories")
def list_advisories(
    well_id: str | None = Query(None, max_length=16),
    status: str | None = Query(None, pattern="^(pending|applied|rejected|expired)$"),
    limit: int = Query(50, ge=1, le=200),
    c: Container = Depends(get_container),
) -> list[dict]:
    return storage.list_advisories(c.engine, well_id=well_id.upper() if well_id else None, status=status, limit=limit)


@router.post("/advisories/{adv_id}/approve", dependencies=[Depends(require_api_key)])
def approve(adv_id: int, body: DecisionIn | None = None, c: Container = Depends(get_container)) -> dict:
    body = body or DecisionIn()
    return c.optimizer.decide(adv_id, True, body.actor, body.note)


@router.post("/advisories/{adv_id}/reject", dependencies=[Depends(require_api_key)])
def reject(adv_id: int, body: DecisionIn | None = None, c: Container = Depends(get_container)) -> dict:
    body = body or DecisionIn()
    return c.optimizer.decide(adv_id, False, body.actor, body.note)


@router.get("/audit")
def audit(limit: int = Query(50, ge=1, le=500), c: Container = Depends(get_container)) -> list[dict]:
    return storage.list_audit(c.engine, limit=limit)


@router.post("/wells/{well_id}/cycle-plan", status_code=202, dependencies=[Depends(require_api_key), Depends(limit_jobs)])
def cycle_plan(well_id: str, body: CyclePlanRequest | None = None, c: Container = Depends(get_container)) -> dict:
    """Queue a steam-volume sweep (net value vs volume). Poll ``/jobs/{id}`` for the result."""
    wid = well_or_404(c, well_id)
    body = body or CyclePlanRequest()
    volumes = body.volumes or [2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    overrides = body.overrides.overrides() if body.overrides else {}
    job_id = c.jobs.submit("cycle_plan", lambda: c.optimizer.plan_cycle(wid, volumes, overrides))
    return {"job_id": job_id, "status": "queued"}


__all__ = ["router", "OptimizerError"]
