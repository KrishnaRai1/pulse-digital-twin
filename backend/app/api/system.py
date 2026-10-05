"""Health, system information, configuration echo and downloadable sample files."""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import text

from ..engines.optimizer.mpc import MAX_RATE_UP
from ..engines.srp.loads import FLOAT_MARGIN
from .deps import Container, get_container

router = APIRouter(tags=["system"])


@router.get("/health")
def health(c: Container = Depends(get_container)) -> dict:
    db_ok = True
    try:
        with c.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    ok = db_ok and c.classifier is not None
    return {
        "status": "ok" if ok else "degraded",
        "database": "up" if db_ok else "down",
        "cnn_loaded": c.classifier is not None,
        "ml_loaded": c.ml is not None,
        "uptime_s": int(time.time()) - c.started_ts,
        "field_dataset": c.field.state,
    }


@router.get("/system/info")
def info(c: Container = Depends(get_container)) -> dict:
    s = c.settings
    cnn = dict(c.classifier.meta) if c.classifier else None
    ml = dict(c.ml.meta) if c.ml else None
    return {
        "control_mode": s.control_mode,
        "auth_required_for_writes": bool(s.api_key),
        "simulator": {
            "enabled": s.sim_enabled,
            "tick_seconds": s.sim_tick_seconds,
            "sim_minutes_per_tick": s.sim_minutes_per_tick,
            "sim_time": int(c.runtime.sim_ts),
            "ticks": c.runtime.tick_n,
        },
        "database": {"dialect": c.engine.dialect.name, "telemetry_retention_hours": s.telemetry_retention_hours},
        "stream_clients": c.hub.n_clients,
        "models": {
            "cnn": cnn,
            "thermal_ml": ml,
            "inference_budget_ms": s.inference_budget_ms,
        },
        "safety": {
            "hard_limits": ["drive maximum", f"rod float index <= {FLOAT_MARGIN}", "rod fatigue (modified Goodman) <= 0.90", "unit structure rating <= 0.95"],
            "mpc_horizon_days": 14,
            "max_speed_increase_spm_per_day": MAX_RATE_UP,
            "every_setpoint_revalidated_at_write": True,
        },
        "economics": {
            "oil_usd_per_m3": s.oil_price_usd_per_m3,
            "power_usd_per_kwh": s.power_price_usd_per_kwh,
            "steam_usd_per_m3": s.steam_cost_usd_per_m3,
        },
        "data_provenance": "synthetic demonstration data; connect a SCADA/historian feed for field use",
    }


def _sample_dir(c: Container) -> Path:
    return Path(c.settings.samples_dir)


@router.get("/samples")
def list_samples(c: Container = Depends(get_container)) -> list[dict]:
    d = _sample_dir(c)
    if not d.is_dir():
        return []
    return [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(d.iterdir()) if p.is_file() and p.suffix in (".csv", ".xlsx")]


@router.get("/samples/{name}")
def download_sample(name: str, c: Container = Depends(get_container)):
    # only names that exist in the samples directory listing are served (no path traversal)
    allowed = {p.name: p for p in _sample_dir(c).glob("*") if p.is_file() and p.suffix in (".csv", ".xlsx")}
    if name not in allowed:
        raise HTTPException(status_code=404, detail="sample not found")
    return FileResponse(allowed[name], filename=name)
