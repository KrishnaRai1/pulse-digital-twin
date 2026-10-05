"""SRP diagnostic console endpoints: on-demand classification, latest card vs baseline, history."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ..schemas import ClassifyRequest
from ..security import limit_classify
from ..services import storage
from ..services.diagnostics import DiagnosticsUnavailable
from .deps import Container, get_container, well_or_404

router = APIRouter(tags=["diagnostics"])


@router.post("/diagnostics/classify", dependencies=[Depends(limit_classify)])
def classify(body: ClassifyRequest, c: Container = Depends(get_container)) -> dict:
    """Classify one surface dynamometer card (wave equation -> downhole card -> CNN).

    The response includes ``latency_ms_total`` and ``within_budget`` (< 200 ms requirement)."""
    wid = body.well_id.upper() if body.well_id else None
    try:
        return c.diagnostics.classify_arrays(
            wid, body.spm, body.position, body.load, body.position_unit, body.load_unit, body.viscosity_cp
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown well") from None
    except DiagnosticsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/wells/{well_id}/card")
def latest_card(well_id: str, c: Container = Depends(get_container)) -> dict:
    """Most recent card (surface + downhole) with the healthy-baseline card for overlay."""
    wid = well_or_404(c, well_id)
    return {"well_id": wid, "card": c.runtime.latest_card(wid), "baseline": c.runtime.baseline_for(wid)}


@router.get("/wells/{well_id}/cards")
def card_history(
    well_id: str,
    limit: int = Query(30, ge=1, le=200),
    source: str | None = Query(None, pattern="^(live|upload)$"),
    c: Container = Depends(get_container),
) -> list[dict]:
    wid = well_or_404(c, well_id)
    return storage.card_history(c.engine, wid, limit=limit, source=source)
