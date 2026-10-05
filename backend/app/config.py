"""Runtime configuration.

Every setting can be overridden with an environment variable prefixed ``PULSE_``
(for example ``PULSE_DATABASE_URL``) or via a ``.env`` file. Nothing secret is
committed; see ``.env.example`` in the repository root.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PULSE_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"

    # --- storage -----------------------------------------------------------------
    # SQLite by default (zero-setup). Any SQLAlchemy URL works, e.g.
    # postgresql+psycopg://user:pass@timescaledb:5432/pulse
    database_url: str = f"sqlite:///{BACKEND_DIR / 'data' / 'pulse.db'}"
    models_dir: Path = BACKEND_DIR / "models"
    samples_dir: Path = REPO_DIR / "data" / "samples"
    # Field dataset (Baghewala synthetic v1.4): raw files and the derived store built from them
    field_data_dir: Path = REPO_DIR / "data" / "field"
    field_store_dir: Path = BACKEND_DIR / "field_store"
    field_auto_build: bool = True  # build the store in the background on start-up when missing/stale
    telemetry_retention_hours: int = 72
    # Built React app to serve from the API process (optional; nginx serves it in Docker).
    frontend_dist: Path | None = None

    # --- security ----------------------------------------------------------------
    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    # Optional regular expression for extra allowed origins, e.g. Vercel preview deployments:
    # https://pulse-[a-z0-9-]+\.vercel\.app  (keep it as narrow as possible)
    cors_origin_regex: str | None = None
    # When set, every mutating request (POST/PUT/DELETE) must carry X-API-Key.
    api_key: str | None = None
    max_upload_mb: int = 10
    max_upload_rows: int = 200_000

    # --- control -----------------------------------------------------------------
    # advisory: operator must approve each set-point (default, safe).
    # closed_loop: approved-by-policy set-points are written to the SCADA adapter
    #              automatically, still passing through the physics safety layer.
    control_mode: Literal["advisory", "closed_loop"] = "advisory"

    # --- simulator (stands in for SCADA/historian in demos) ----------------------
    sim_enabled: bool = True
    sim_tick_seconds: float = 2.0
    sim_minutes_per_tick: float = 3.0
    sim_seed: int = 42
    sim_history_hours: int = 24
    reset_demo_data: bool = False  # wipe the database and re-seed on start

    # --- models ------------------------------------------------------------------
    auto_train: bool = True  # train CNN + LightGBM on first start if artifacts are missing
    inference_budget_ms: float = 200.0

    # --- economics used by the optimizer (illustrative defaults) -----------------
    oil_price_usd_per_m3: float = 420.0
    power_price_usd_per_kwh: float = 0.09
    steam_cost_usd_per_m3: float = 11.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()
