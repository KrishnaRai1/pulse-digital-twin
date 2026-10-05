"""Lifecycle of the field store inside the API process.

On start-up the service loads the derived store if it matches the raw data; otherwise it
builds it in a background thread (about 1.5 minutes) while the rest of the API is already
serving. Dataset endpoints answer ``503`` with build progress until the store is ready.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from ..config import Settings
from .build import STORE_VERSION, build_store, current_build
from .raw import DatasetError, find_table, fingerprint
from .store import FieldStore

log = logging.getLogger("pulse.field")


class FieldUnavailable(RuntimeError):
    def __init__(self, status: dict):
        super().__init__(status.get("message") or status.get("state"))
        self.status = status


class FieldDataService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_dir = Path(settings.field_data_dir)
        self.store_dir = Path(settings.field_store_dir)
        self.store: FieldStore | None = None
        self.state = "idle"
        self.message = ""
        self.progress = {"step": "", "fraction": 0.0}
        self.started_at: float | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        return {
            "state": self.state,
            "message": self.message,
            "progress": self.progress,
            "elapsed_s": round(time.time() - self.started_at, 1) if self.started_at and self.state == "building" else None,
            "data_dir_present": self.raw_present(),
            "ready": self.store is not None,
        }

    def raw_present(self) -> bool:
        return all(find_table(self.data_dir, n) for n in ("css_cycle_log", "production_history", "srp_vfd_data")) and (self.data_dir / "rod_failure_log.csv").is_file()

    def require(self) -> FieldStore:
        if self.store is None:
            raise FieldUnavailable(self.status())
        return self.store

    # ------------------------------------------------------------------ start / build
    def _store_matches(self) -> bool:
        build = current_build(self.store_dir)
        if build is None:
            return False
        meta = build / "meta.json"
        try:
            m = json.loads(meta.read_text())
        except json.JSONDecodeError:
            return False
        return m.get("store_version") == STORE_VERSION and m.get("fingerprint") == fingerprint(self.data_dir)

    def start(self) -> None:
        """Non-blocking: load or build in a background thread."""
        if not self.raw_present():
            self.state = "missing"
            self.message = f"field dataset not found in {self.data_dir} (see docs/DATASET.md to import it)"
            log.warning(self.message)
            return
        if self._store_matches():
            self._spawn(self._load)
        elif self.settings.field_auto_build:
            self._spawn(self._build_then_load)
        else:
            self.state = "stale"
            self.message = "field store missing or out of date: run `python -m scripts.build_field_store`"
            log.warning(self.message)

    def rebuild(self) -> bool:
        with self._lock:
            if self.state == "building":
                return False
        self._spawn(self._build_then_load)
        return True

    def _spawn(self, fn) -> None:
        self._thread = threading.Thread(target=fn, name="pulse-field", daemon=True)
        self._thread.start()

    def _load(self) -> None:
        self.state = "loading"
        self.message = "loading field store"
        try:
            t0 = time.time()
            store = FieldStore(self.store_dir, self.data_dir, self.settings.oil_price_usd_per_m3, self.settings.steam_cost_usd_per_m3)
            self.store = store
            self.state = "ready"
            self.message = f"loaded in {time.time() - t0:.1f} s"
            log.info("field store ready: %d wells, %d days", store.n_wells, store.max_day + 1)
        except Exception as exc:  # report, keep the rest of the API alive
            log.exception("could not load the field store")
            self.state = "error"
            self.message = f"{type(exc).__name__}: {str(exc)[:300]}"

    def _build_then_load(self) -> None:
        with self._lock:
            self.state = "building"
            self.started_at = time.time()
        self.message = "building field store and training models (about 1-3 minutes)"

        def progress(step: str, frac: float) -> None:
            self.progress = {"step": step, "fraction": round(frac, 3)}
            log.info("field store: [%3.0f%%] %s", frac * 100, step)

        try:
            build_store(self.data_dir, self.store_dir, progress)
        except DatasetError as exc:
            self.state, self.message = "error", f"dataset error: {exc}"
            log.error(self.message)
            return
        except Exception as exc:
            log.exception("field store build failed")
            self.state, self.message = "error", f"{type(exc).__name__}: {str(exc)[:300]}"
            return
        self._load()

    def wait(self, timeout: float = 600.0) -> bool:
        """Block until the background work finishes (used by tests and scripts)."""
        if self._thread:
            self._thread.join(timeout)
        return self.store is not None


__all__ = ["FieldDataService", "FieldUnavailable"]
