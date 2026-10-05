"""Database layer (SQLAlchemy Core).

SQLite is the zero-setup default. The same schema runs on PostgreSQL; when the
dialect is PostgreSQL the ``telemetry`` table is additionally converted into a
TimescaleDB hypertable on a best-effort basis (the standard ``timescale/timescaledb``
image ships the extension). All statements are parameterised.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    event,
    select,
    text,
)
from sqlalchemy.engine import Engine

log = logging.getLogger("pulse.db")

metadata = MetaData()

telemetry = Table(
    "telemetry",
    metadata,
    Column("well_id", String(16), primary_key=True),
    Column("ts", BigInteger, primary_key=True),  # epoch seconds (UTC)
    Column("spm", Float),
    Column("stroke_m", Float),
    Column("vfd_hz", Float),
    Column("motor_load_pct", Float),
    Column("motor_kw", Float),
    Column("pprl_kn", Float),
    Column("mprl_kn", Float),
    Column("whp_kpa", Float),
    Column("tubing_temp_c", Float),
    Column("oil_rate_m3d", Float),
    Column("liquid_rate_m3d", Float),
    Column("water_cut", Float),
    Column("visc_cp", Float),
    Column("fillage_pct", Float),
)
Index("ix_telemetry_ts", telemetry.c.ts)

cycles = Table(
    "cycles",
    metadata,
    Column("well_id", String(16), primary_key=True),
    Column("cycle_no", Integer, primary_key=True),
    Column("start_ts", BigInteger, nullable=False),  # start of steam injection
    Column("steam_m3", Float, nullable=False),  # cold-water-equivalent volume
    Column("quality", Float, nullable=False),  # surface steam quality 0..1
    Column("inj_rate_m3d", Float, nullable=False),
    Column("inj_pressure_mpa", Float, nullable=False),  # bottom-hole injection pressure
    Column("soak_days", Float, nullable=False),
    Column("prod_days", Float, nullable=False),
    Column("source", String(16), nullable=False, default="operator"),
)

dyno_cards = Table(
    "dyno_cards",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("well_id", String(16), nullable=False),
    Column("ts", BigInteger, nullable=False),
    Column("source", String(16), nullable=False),  # live | upload
    Column("spm", Float),
    Column("position", JSON, nullable=False),
    Column("load", JSON, nullable=False),
    Column("label", String(32)),
    Column("probability", Float),
    Column("probabilities", JSON),
)
Index("ix_dyno_well_ts", dyno_cards.c.well_id, dyno_cards.c.ts)

alerts = Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", BigInteger, nullable=False),
    Column("well_id", String(16), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("severity", String(8), nullable=False),  # red | amber
    Column("message", Text, nullable=False),
    Column("probability", Float),
)
Index("ix_alerts_ts", alerts.c.ts)

advisories = Table(
    "advisories",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", BigInteger, nullable=False),
    Column("well_id", String(16), nullable=False),
    Column("current_spm", Float, nullable=False),
    Column("recommended_spm", Float, nullable=False),
    Column("status", String(12), nullable=False),  # pending|approved|rejected|applied|expired
    Column("payload", JSON, nullable=False),
    Column("decided_ts", BigInteger),
    Column("decided_by", String(64)),
    Column("note", Text),
)
Index("ix_adv_well", advisories.c.well_id, advisories.c.ts)

audit_log = Table(
    "audit_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", BigInteger, nullable=False),
    Column("actor", String(64), nullable=False),
    Column("action", String(48), nullable=False),
    Column("well_id", String(16)),
    Column("detail", JSON),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("kind", String(48), nullable=False),
    Column("status", String(12), nullable=False),  # queued|running|done|failed
    Column("created_ts", BigInteger, nullable=False),
    Column("finished_ts", BigInteger),
    Column("result", JSON),
    Column("error", Text),
)


def make_engine(url: str) -> Engine:
    kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)
    if engine.dialect.name == "postgresql":  # pragma: no cover - needs a live server
        try:
            with engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                conn.execute(
                    text(
                        "SELECT create_hypertable('telemetry', 'ts', "
                        "chunk_time_interval => 86400, if_not_exists => TRUE, migrate_data => TRUE)"
                    )
                )
            log.info("telemetry converted to TimescaleDB hypertable")
        except Exception as exc:  # plain PostgreSQL is fine, just without hypertables
            log.warning("TimescaleDB extension unavailable, using plain tables: %s", exc)


def _dialect_insert(engine: Engine):
    if engine.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:
        from sqlalchemy.dialects.postgresql import insert
    return insert


def upsert_rows(engine: Engine, table: Table, rows: Iterable[dict[str, Any]], keys: list[str]) -> int:
    """Insert rows, updating on primary-key conflict. Works on SQLite and PostgreSQL."""
    rows = list(rows)
    if not rows:
        return 0
    insert = _dialect_insert(engine)
    # every row must share the same column set for executemany
    columns = sorted({c for r in rows for c in r})
    rows = [{c: r.get(c) for c in columns} for r in rows]
    stmt = insert(table)
    update_cols = {c: stmt.excluded[c] for c in columns if c not in keys}
    stmt = stmt.on_conflict_do_update(index_elements=keys, set_=update_cols) if update_cols else stmt.on_conflict_do_nothing()
    # chunk to stay under SQLite's bound-parameter limit
    step = max(1, 30000 // max(1, len(columns)))
    with engine.begin() as conn:
        for i in range(0, len(rows), step):
            conn.execute(stmt, rows[i : i + step])
    return len(rows)


def prune_telemetry(engine: Engine, older_than_ts: int) -> None:
    with engine.begin() as conn:
        conn.execute(delete(telemetry).where(telemetry.c.ts < older_than_ts))


def now_ts() -> int:
    return int(time.time())


__all__ = [
    "metadata",
    "telemetry",
    "cycles",
    "dyno_cards",
    "alerts",
    "advisories",
    "audit_log",
    "jobs",
    "make_engine",
    "init_db",
    "upsert_rows",
    "prune_telemetry",
    "now_ts",
    "select",
]
