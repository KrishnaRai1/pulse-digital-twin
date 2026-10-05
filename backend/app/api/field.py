"""Field-level views: overview (traffic-light table + KPIs), trends, alerts."""
from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, Depends, Query

from ..services import storage
from .deps import Container, get_container

router = APIRouter(tags=["field"])

BIN_S = 600  # trend bin width (10 min)


@router.get("/field/overview")
def overview(c: Container = Depends(get_container)) -> dict:
    return c.runtime.overview()


@router.get("/field/trend")
def trend(hours: int = Query(24, ge=1, le=72), c: Container = Depends(get_container)) -> dict:
    """Field totals over time (oil, liquid, power) plus the number of wells per status."""
    now = int(c.runtime.sim_ts)
    since = now - hours * 3600
    per_bin: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for wid in c.twin.fleet:
        for r in storage.query_telemetry(c.engine, wid, since, now, max_points=2000):
            b = (r["ts"] // BIN_S) * BIN_S
            for k in ("oil_rate_m3d", "liquid_rate_m3d", "motor_kw"):
                if r.get(k) is not None:
                    per_bin[b][wid][k].append(r[k])
    ts, oil, liq, kw = [], [], [], []
    for b in sorted(per_bin):
        wells = per_bin[b]

        def total(key: str, wells=wells) -> float:
            return sum(sum(v[key]) / len(v[key]) for v in wells.values() if v[key])

        ts.append(b)
        oil.append(total("oil_rate_m3d"))
        liq.append(total("liquid_rate_m3d"))
        kw.append(total("motor_kw"))
    return {"ts": ts, "oil_m3d": oil, "liquid_m3d": liq, "power_kw": kw, "bin_seconds": BIN_S}


@router.get("/alerts")
def alerts(limit: int = Query(50, ge=1, le=500), well_id: str | None = Query(None, max_length=16), c: Container = Depends(get_container)) -> dict:
    active = [{"well_id": w, "kind": k} for w, kinds in c.runtime.active_alerts.items() for k in sorted(kinds)]
    return {"active": active, "history": storage.list_alerts(c.engine, limit=limit, well_id=well_id.upper() if well_id else None)}
