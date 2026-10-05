"""Thermal-viscosity engine endpoints: state prediction and steam-cycle what-if."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ..schemas import ThermalPredictRequest, WhatIfRequest
from ..security import limit_compute
from .deps import Container, get_container, well_or_404

router = APIRouter(prefix="/thermal", tags=["thermal"], dependencies=[Depends(limit_compute)])


@router.post("/predict")
def predict(body: ThermalPredictRequest, c: Container = Depends(get_container)) -> dict:
    """Reservoir/wellbore temperature, viscosity and rate at a timestamp (default: now).

    Returns the physics-only values next to the ML-corrected ones so the size of the
    data-driven correction is always visible."""
    wid = well_or_404(c, body.well_id)
    ts = body.ts if body.ts is not None else c.runtime.sim_ts
    t0 = time.perf_counter()
    st = c.twin.state_at(wid, ts)
    return {
        "state": st,
        "physics_only": {"mu_eff_cp": st["mu_eff_phys_cp"], "q_oil_m3d": st["q_oil_phys_m3d"]},
        "ml_correction": {"applied": c.ml is not None, "factor": st["ml_factor"], "top_features": c.twin.ml_top_features(wid)},
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


@router.post("/whatif")
def whatif(body: WhatIfRequest, c: Container = Depends(get_container)) -> dict:
    """Simulate alternative steam volumes / qualities / soak times for the next (or current) cycle."""
    wid = well_or_404(c, body.well_id)
    t0 = time.perf_counter()
    out = c.twin.whatif(wid, [_scn(s) for s in body.scenarios], base=body.base)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return out


def _scn(s) -> dict:
    d = s.overrides()
    if s.label:
        d["label"] = s.label
    return d
