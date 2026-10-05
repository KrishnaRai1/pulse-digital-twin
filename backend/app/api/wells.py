"""Well digital-twin endpoints: detail, telemetry, wellbore / radial profiles, cycle history, set-points."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Query

from ..domain.wells import DAY
from ..engines.optimizer.limits import check_setpoint, spm_limits
from ..schemas import CycleIn, SetpointIn
from ..security import limit_compute, require_api_key
from ..services import storage
from .deps import Container, get_container, well_or_404

router = APIRouter(tags=["wells"])


def _config(c: Container, wid: str) -> dict:
    cfg = c.twin.fleet[wid]
    d, r = cfg.design, cfg.reservoir
    return {
        "design": {
            "pump_depth_m": d.pump_depth_m,
            "plunger_mm": d.plunger_mm,
            "stroke_m": d.stroke_m,
            "min_spm": d.min_spm,
            "max_spm": d.max_spm,
            "unit_rating_kn": d.unit_rating_kn,
            "motor_kw_rated": d.motor_kw_rated,
            "rods": [{"length_m": s.length_m, "dia_mm": s.dia_mm} for s in d.rods],
            "buoyant_rod_weight_kn": d.weight_buoyant_n / 1000.0,
            "pump_capacity_m3d_at_1spm": d.pump_capacity_m3d(1.0),
        },
        "reservoir": {
            "thickness_m": r.thickness_m,
            "t_res_c": r.t_res_c,
            "perm_md": r.perm_md,
            "depth_m": r.depth_m,
            "drainage_radius_m": r.r_e,
        },
        "scenario_note": cfg.notes,
    }


@router.get("/wells")
def list_wells(c: Container = Depends(get_container)) -> list[dict]:
    return c.runtime.overview()["wells"]


@router.get("/wells/{well_id}")
def well_detail(well_id: str, c: Container = Depends(get_container)) -> dict:
    wid = well_or_404(c, well_id)
    summary = c.runtime.latest.get(wid)
    cfg = c.twin.fleet[wid]
    state = c.twin.state_at(wid, c.runtime.sim_ts)
    f_load = cfg.design.fluid_load_n(cfg.design.pump_depth_m * cfg.net_lift_fraction)
    lim = spm_limits(cfg.design, state["mu_tub_pa_s"], f_load)
    return {
        "summary": summary,
        "state": state,
        "config": _config(c, wid),
        "limits": lim.as_dict(),
        "setpoint_spm": c.scada.read_setpoint(wid),
        "cycles": c.twin.cycles(wid),
        "ml_top_features": c.twin.ml_top_features(wid),
        "sim_ts": int(c.runtime.sim_ts),
    }


@router.get("/wells/{well_id}/telemetry")
def telemetry(
    well_id: str,
    hours: float = Query(24, gt=0, le=72),
    max_points: int = Query(600, ge=10, le=2000),
    c: Container = Depends(get_container),
) -> dict:
    wid = well_or_404(c, well_id)
    now = int(c.runtime.sim_ts)
    rows = storage.query_telemetry(c.engine, wid, int(now - hours * 3600), None, max_points)
    cols = ["ts", *storage.TELEMETRY_FIELDS]
    return {"well_id": wid, "columns": cols, "series": {k: [r.get(k) for r in rows] for k in cols}, "now": now}


@router.get("/wells/{well_id}/profile")
def profile(well_id: str, c: Container = Depends(get_container), _: None = Depends(limit_compute)) -> dict:
    wid = well_or_404(c, well_id)
    return c.twin.profile(wid, c.runtime.sim_ts)


@router.get("/wells/{well_id}/viscosity")
def viscosity(well_id: str, c: Container = Depends(get_container), _: None = Depends(limit_compute)) -> dict:
    wid = well_or_404(c, well_id)
    return c.twin.viscosity_curves(wid, c.runtime.sim_ts)


# --------------------------------------------------------------------------- cycle history
def _next_start(rows: list[dict], sim_ts: float) -> int:
    last = rows[-1]
    end = last["start_ts"] + int((last["steam_m3"] / last["inj_rate_m3d"] + last["soak_days"] + last["prod_days"]) * DAY)
    return int(min(end, sim_ts))


def _validate_order(rows: list[dict], row: dict, sim_ts: float) -> None:
    others = [r for r in rows if r["cycle_no"] != row["cycle_no"]]
    prev = max((r for r in others if r["cycle_no"] < row["cycle_no"]), key=lambda r: r["cycle_no"], default=None)
    nxt = min((r for r in others if r["cycle_no"] > row["cycle_no"]), key=lambda r: r["cycle_no"], default=None)
    if prev and row["start_ts"] < prev["start_ts"] + int(prev["steam_m3"] / prev["inj_rate_m3d"] * DAY):
        raise HTTPException(status_code=422, detail="cycle starts before the previous cycle's steam injection has finished")
    inj_end = row["start_ts"] + int(row["steam_m3"] / row["inj_rate_m3d"] * DAY)
    if nxt and inj_end > nxt["start_ts"]:
        raise HTTPException(status_code=422, detail="steam injection would overrun the start of the next cycle")
    if row["start_ts"] > sim_ts + 3600:
        raise HTTPException(status_code=422, detail="cycle start is in the future; record only cycles that have started")


def _after_edit(c: Container, wid: str, actor: str, action: str, detail: dict) -> None:
    c.twin.invalidate(wid)
    storage.audit(c.engine, actor, action, wid, detail)
    c.runtime.tick(advance=False)  # refresh the live view immediately


@router.get("/wells/{well_id}/cycles")
def get_cycles(well_id: str, c: Container = Depends(get_container)) -> list[dict]:
    return c.twin.cycles(well_or_404(c, well_id))


@router.post("/wells/{well_id}/cycles", status_code=201, dependencies=[Depends(require_api_key)])
def add_cycle(well_id: str, body: CycleIn, c: Container = Depends(get_container)) -> dict:
    """Append a new steam cycle (a re-steam that has started) to the well's history."""
    wid = well_or_404(c, well_id)
    rows = c.twin.cycles(wid)
    row = {
        "well_id": wid,
        "cycle_no": rows[-1]["cycle_no"] + 1 if rows else 1,
        "start_ts": body.start_ts if body.start_ts is not None else (_next_start(rows, c.runtime.sim_ts) if rows else int(c.runtime.sim_ts)),
        **body.model_dump(exclude={"start_ts"}),
        "source": "operator",
    }
    _validate_order(rows, row, c.runtime.sim_ts)
    storage.upsert_cycle(c.engine, row)
    _after_edit(c, wid, "operator", "cycle_added", {"cycle_no": row["cycle_no"], "steam_m3": row["steam_m3"]})
    return row


@router.put("/wells/{well_id}/cycles/{cycle_no}", dependencies=[Depends(require_api_key)])
def update_cycle(well_id: str, cycle_no: int, body: CycleIn, c: Container = Depends(get_container)) -> dict:
    """Override the recorded parameters of a historical cycle; the twin is recomputed from the edit."""
    wid = well_or_404(c, well_id)
    rows = c.twin.cycles(wid)
    old = next((r for r in rows if r["cycle_no"] == cycle_no), None)
    if old is None:
        raise HTTPException(status_code=404, detail="cycle not found")
    row = {**old, **body.model_dump(exclude={"start_ts"}), "start_ts": body.start_ts if body.start_ts is not None else old["start_ts"], "source": "operator"}
    _validate_order(rows, row, c.runtime.sim_ts)
    storage.upsert_cycle(c.engine, row)
    _after_edit(c, wid, "operator", "cycle_edited", {"cycle_no": cycle_no, "before": {k: old[k] for k in ("steam_m3", "quality", "soak_days")}})
    return row


@router.delete("/wells/{well_id}/cycles/{cycle_no}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_cycle(well_id: str, cycle_no: int, c: Container = Depends(get_container)) -> None:
    wid = well_or_404(c, well_id)
    rows = c.twin.cycles(wid)
    if len(rows) <= 1:
        raise HTTPException(status_code=409, detail="a well needs at least one cycle")
    if cycle_no != rows[-1]["cycle_no"]:
        raise HTTPException(status_code=409, detail="only the most recent cycle can be deleted")
    storage.delete_cycle(c.engine, wid, cycle_no)
    _after_edit(c, wid, "operator", "cycle_deleted", {"cycle_no": cycle_no})


# --------------------------------------------------------------------------- manual set-point
@router.post("/wells/{well_id}/setpoint", dependencies=[Depends(require_api_key)])
def set_setpoint(well_id: str, body: SetpointIn, c: Container = Depends(get_container)) -> dict:
    """Manual operator override. Still passes through the hard physics limits."""
    wid = well_or_404(c, well_id)
    cfg = c.twin.fleet[wid]
    state = c.twin.state_at(wid, c.runtime.sim_ts)
    if body.spm > 0:
        if state["phase"] != "PRODUCTION":
            raise HTTPException(status_code=409, detail=f"well is in {state['phase'].lower()}: it is not pumping")
        f_load = cfg.design.fluid_load_n(cfg.design.pump_depth_m * cfg.net_lift_fraction)
        problems = check_setpoint(body.spm, cfg.design, state["mu_tub_pa_s"], f_load)
        if problems:
            storage.audit(c.engine, body.actor, "manual_setpoint_blocked", wid, {"spm": body.spm, "problems": problems})
            raise HTTPException(status_code=409, detail="Blocked by the physics safety layer: " + " ".join(problems))
        if body.spm < cfg.design.min_spm:
            raise HTTPException(status_code=422, detail=f"minimum stable speed is {cfg.design.min_spm} SPM (use 0 to stop)")
    prev = c.scada.read_setpoint(wid)
    c.scada.write_setpoint(wid, body.spm, body.actor)
    storage.audit(c.engine, body.actor, "manual_setpoint", wid, {"from": prev, "to": body.spm})
    c.runtime.tick(advance=False)
    return {"well_id": wid, "previous_spm": prev, "setpoint_spm": c.scada.read_setpoint(wid), "ts": int(time.time())}
