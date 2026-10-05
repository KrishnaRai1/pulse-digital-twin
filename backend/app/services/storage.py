"""Persistence helpers (thin wrappers around SQLAlchemy Core statements)."""
from __future__ import annotations

import json
import time
from typing import Any

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.engine import Engine

from .. import db


def _rows(result) -> list[dict[str, Any]]:
    return [dict(r._mapping) for r in result]


# --------------------------------------------------------------------------- cycles
def get_cycles(engine: Engine, well_id: str) -> list[dict]:
    with engine.connect() as c:
        res = c.execute(select(db.cycles).where(db.cycles.c.well_id == well_id).order_by(db.cycles.c.cycle_no))
        return _rows(res)


def upsert_cycle(engine: Engine, row: dict) -> None:
    db.upsert_rows(engine, db.cycles, [row], ["well_id", "cycle_no"])


def delete_cycle(engine: Engine, well_id: str, cycle_no: int) -> int:
    with engine.begin() as c:
        res = c.execute(delete(db.cycles).where((db.cycles.c.well_id == well_id) & (db.cycles.c.cycle_no == cycle_no)))
        return res.rowcount


def count_cycles(engine: Engine) -> int:
    with engine.connect() as c:
        return int(c.execute(select(func.count()).select_from(db.cycles)).scalar() or 0)


# --------------------------------------------------------------------------- telemetry
TELEMETRY_FIELDS = [
    "spm",
    "stroke_m",
    "vfd_hz",
    "motor_load_pct",
    "motor_kw",
    "pprl_kn",
    "mprl_kn",
    "whp_kpa",
    "tubing_temp_c",
    "oil_rate_m3d",
    "liquid_rate_m3d",
    "water_cut",
    "visc_cp",
    "fillage_pct",
]


def insert_telemetry(engine: Engine, rows: list[dict]) -> int:
    return db.upsert_rows(engine, db.telemetry, rows, ["well_id", "ts"])


def query_telemetry(engine: Engine, well_id: str, since_ts: int, until_ts: int | None = None, max_points: int = 600) -> list[dict]:
    t = db.telemetry
    stmt = select(t).where((t.c.well_id == well_id) & (t.c.ts >= since_ts))
    if until_ts is not None:
        stmt = stmt.where(t.c.ts <= until_ts)
    stmt = stmt.order_by(t.c.ts)
    with engine.connect() as c:
        rows = _rows(c.execute(stmt))
    if len(rows) > max_points:  # decimate evenly, keeping first and last
        step = len(rows) / max_points
        idx = sorted({int(i * step) for i in range(max_points)} | {len(rows) - 1})
        rows = [rows[i] for i in idx]
    return rows


def latest_telemetry_ts(engine: Engine, well_id: str) -> int | None:
    with engine.connect() as c:
        return c.execute(select(func.max(db.telemetry.c.ts)).where(db.telemetry.c.well_id == well_id)).scalar()


def latest_telemetry_row(engine: Engine, well_id: str) -> dict | None:
    t = db.telemetry
    with engine.connect() as c:
        r = c.execute(select(t).where(t.c.well_id == well_id).order_by(t.c.ts.desc()).limit(1)).first()
    return dict(r._mapping) if r else None


def daily_mean_oil_rate(engine: Engine, well_id: str, since_ts: int, until_ts: int) -> list[tuple[int, float]]:
    """(day_index, mean oil rate) from telemetry, used to retrain the ML residual model."""
    t = db.telemetry
    with engine.connect() as c:
        res = c.execute(
            select(t.c.ts, t.c.oil_rate_m3d).where((t.c.well_id == well_id) & (t.c.ts >= since_ts) & (t.c.ts <= until_ts)).order_by(t.c.ts)
        ).all()
    buckets: dict[int, list[float]] = {}
    for ts, q in res:
        if q is not None:
            buckets.setdefault(int((ts - since_ts) // 86400), []).append(float(q))
    return [(d, sum(v) / len(v)) for d, v in sorted(buckets.items())]


# --------------------------------------------------------------------------- dyno cards
def insert_card(engine: Engine, row: dict) -> int:
    with engine.begin() as c:
        res = c.execute(insert(db.dyno_cards).values(**row))
        return int(res.inserted_primary_key[0])


def insert_cards(engine: Engine, rows: list[dict]) -> None:
    if not rows:
        return
    with engine.begin() as c:
        c.execute(insert(db.dyno_cards), rows)


def card_history(engine: Engine, well_id: str, limit: int = 30, source: str | None = None) -> list[dict]:
    t = db.dyno_cards
    stmt = select(t).where(t.c.well_id == well_id)
    if source:
        stmt = stmt.where(t.c.source == source)
    stmt = stmt.order_by(t.c.ts.desc(), t.c.id.desc()).limit(limit)
    with engine.connect() as c:
        return _rows(c.execute(stmt))


# --------------------------------------------------------------------------- alerts
def insert_alert(engine: Engine, row: dict) -> int:
    with engine.begin() as c:
        return int(c.execute(insert(db.alerts).values(**row)).inserted_primary_key[0])


def list_alerts(engine: Engine, limit: int = 50, well_id: str | None = None) -> list[dict]:
    stmt = select(db.alerts)
    if well_id:
        stmt = stmt.where(db.alerts.c.well_id == well_id)
    stmt = stmt.order_by(db.alerts.c.ts.desc(), db.alerts.c.id.desc()).limit(limit)
    with engine.connect() as c:
        return _rows(c.execute(stmt))


# --------------------------------------------------------------------------- advisories / audit
def insert_advisory(engine: Engine, row: dict) -> int:
    with engine.begin() as c:
        return int(c.execute(insert(db.advisories).values(**row)).inserted_primary_key[0])


def get_advisory(engine: Engine, adv_id: int) -> dict | None:
    with engine.connect() as c:
        r = c.execute(select(db.advisories).where(db.advisories.c.id == adv_id)).first()
    return dict(r._mapping) if r else None


def list_advisories(engine: Engine, well_id: str | None = None, status: str | None = None, limit: int = 50) -> list[dict]:
    stmt = select(db.advisories)
    if well_id:
        stmt = stmt.where(db.advisories.c.well_id == well_id)
    if status:
        stmt = stmt.where(db.advisories.c.status == status)
    stmt = stmt.order_by(db.advisories.c.ts.desc(), db.advisories.c.id.desc()).limit(limit)
    with engine.connect() as c:
        return _rows(c.execute(stmt))


def update_advisory(engine: Engine, adv_id: int, **values: Any) -> None:
    with engine.begin() as c:
        c.execute(update(db.advisories).where(db.advisories.c.id == adv_id).values(**values))


def expire_pending(engine: Engine, well_id: str, except_id: int | None = None) -> None:
    stmt = update(db.advisories).where((db.advisories.c.well_id == well_id) & (db.advisories.c.status == "pending"))
    if except_id is not None:
        stmt = stmt.where(db.advisories.c.id != except_id)
    with engine.begin() as c:
        c.execute(stmt.values(status="expired"))


def audit(engine: Engine, actor: str, action: str, well_id: str | None = None, detail: dict | None = None) -> None:
    with engine.begin() as c:
        c.execute(insert(db.audit_log).values(ts=int(time.time()), actor=actor, action=action, well_id=well_id, detail=detail or {}))


def list_audit(engine: Engine, limit: int = 50) -> list[dict]:
    with engine.connect() as c:
        return _rows(c.execute(select(db.audit_log).order_by(db.audit_log.c.id.desc()).limit(limit)))


# --------------------------------------------------------------------------- jobs
def save_job(engine: Engine, row: dict) -> None:
    with engine.begin() as c:
        exists = c.execute(select(db.jobs.c.id).where(db.jobs.c.id == row["id"])).first()
        if exists:
            c.execute(update(db.jobs).where(db.jobs.c.id == row["id"]).values(**{k: v for k, v in row.items() if k != "id"}))
        else:
            c.execute(insert(db.jobs).values(**row))


def get_job(engine: Engine, job_id: str) -> dict | None:
    with engine.connect() as c:
        r = c.execute(select(db.jobs).where(db.jobs.c.id == job_id)).first()
    return dict(r._mapping) if r else None


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))
