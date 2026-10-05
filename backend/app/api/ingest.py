"""Data ingestion (CSV / Excel) and background-job status."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..security import limit_ingest, require_api_key
from ..services import storage
from ..services.etl import UNIT_LOAD_N, UNIT_POS_M, IngestError, clean_production, parse_cards, read_table
from .deps import Container, get_container

router = APIRouter(tags=["ingest"])

CHUNK = 1024 * 1024


async def _read_limited(file: UploadFile, max_mb: int) -> bytes:
    limit = max_mb * 1024 * 1024
    buf = bytearray()
    while chunk := await file.read(CHUNK):
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(status_code=413, detail=f"file too large (max {max_mb} MB)")
    if not buf:
        raise HTTPException(status_code=422, detail="empty file")
    return bytes(buf)


@router.post("/ingest/production", dependencies=[Depends(require_api_key), Depends(limit_ingest)])
async def ingest_production(
    file: UploadFile = File(...),
    align_to_now: bool = Form(False),
    c: Container = Depends(get_container),
) -> dict:
    """Upload a production / sensor log. The file is validated, cleaned (range checks, Hampel
    outlier rejection, short-gap interpolation) and written to the time-series store.
    The full cleaning report is returned so nothing is changed silently.

    ``align_to_now`` shifts the file's timestamps so its last sample lands on the current
    (simulated) time: handy for replaying the bundled sample files in the demo."""
    data = await _read_limited(file, c.settings.max_upload_mb)
    known = set(c.twin.fleet)

    def work() -> dict:
        df = read_table(file.filename or "", data, c.settings.max_upload_rows)
        rows, report = clean_production(df, known)
        if align_to_now:
            shift = int(c.runtime.sim_ts) - max(r["ts"] for r in rows)
            for r in rows:
                r["ts"] += shift
            report["time_shift_seconds"] = shift
            report["time_range"] = [report["time_range"][0] + shift, report["time_range"][1] + shift]
        n = storage.insert_telemetry(c.engine, rows)
        storage.audit(c.engine, "upload", "production_ingested", None, {"rows": n, "wells": list(report["wells"])[:20]})
        report["rows_written"] = n
        return report

    try:
        report = await run_in_threadpool(work)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    c.hub.publish({"type": "ingest", "kind": "production", "rows": report["rows_written"], "wells": sorted(report["wells"])})
    return report


@router.post("/ingest/dynamometer", status_code=202, dependencies=[Depends(require_api_key), Depends(limit_ingest)])
async def ingest_dynamometer(
    file: UploadFile = File(...),
    position_unit: str = Form("m"),
    load_unit: str = Form("kN"),
    c: Container = Depends(get_container),
) -> dict:
    """Upload dynamometer cards. Cards are validated immediately; classification and the replay
    to connected dashboards run as a background job (poll ``/jobs/{id}``)."""
    if position_unit not in UNIT_POS_M or load_unit not in UNIT_LOAD_N:
        raise HTTPException(status_code=422, detail="unsupported unit")
    if c.classifier is None:
        raise HTTPException(status_code=503, detail="CNN model is not loaded")
    data = await _read_limited(file, c.settings.max_upload_mb)
    known = set(c.twin.fleet)

    def parse() -> tuple[list[dict], dict]:
        df = read_table(file.filename or "", data, c.settings.max_upload_rows)
        return parse_cards(df, known, position_unit, load_unit, lambda w: float(c.runtime.setpoints.get(w) or 4.0), int(time.time()))

    try:
        cards, report = await run_in_threadpool(parse)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    job_id = c.jobs.submit("dyno_upload", lambda: c.diagnostics.analyze_uploaded(cards))
    storage.audit(c.engine, "upload", "dynamometer_ingested", None, {"cards": len(cards)})
    return {"job_id": job_id, "status": "queued", "ingest": report}


@router.get("/jobs/{job_id}")
def job_status(job_id: str, c: Container = Depends(get_container)) -> dict:
    if not job_id.isalnum() or len(job_id) > 32:
        raise HTTPException(status_code=404, detail="job not found")
    job = c.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job
