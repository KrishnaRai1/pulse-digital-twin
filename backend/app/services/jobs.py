"""Background job queue.

Heavy computations (steam-volume sweeps, bulk card classification, ML retraining) must not
block the API event loop. ``JobManager`` runs them on a small thread pool and persists
status/results in the database so a client can poll ``GET /api/v1/jobs/{id}``.

The interface (``submit`` / ``get``) is deliberately tiny: swapping in Celery + Redis for
horizontal scale means re-implementing these two methods, nothing else changes.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from sqlalchemy.engine import Engine

from . import storage
from .stream import StreamHub

log = logging.getLogger("pulse.jobs")


def _np_default(o: Any):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def to_jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=_np_default))


class JobManager:
    def __init__(self, engine: Engine, hub: StreamHub | None = None, workers: int = 2):
        self.engine = engine
        self.hub = hub
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pulse-job")

    def submit(self, kind: str, fn: Callable[[], Any]) -> str:
        job_id = uuid.uuid4().hex[:16]
        storage.save_job(self.engine, {"id": job_id, "kind": kind, "status": "queued", "created_ts": int(time.time())})
        self._pool.submit(self._run, job_id, kind, fn)
        return job_id

    def _run(self, job_id: str, kind: str, fn: Callable[[], Any]) -> None:
        storage.save_job(self.engine, {"id": job_id, "status": "running"})
        try:
            result = to_jsonable(fn())
            storage.save_job(self.engine, {"id": job_id, "status": "done", "result": result, "finished_ts": int(time.time())})
            status = "done"
        except Exception as exc:  # report a safe message; details go to the log
            log.exception("job %s (%s) failed", job_id, kind)
            storage.save_job(
                self.engine,
                {"id": job_id, "status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "finished_ts": int(time.time())},
            )
            status = "failed"
        if self.hub:
            self.hub.publish({"type": "job", "id": job_id, "kind": kind, "status": status})

    def get(self, job_id: str) -> dict | None:
        return storage.get_job(self.engine, job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
