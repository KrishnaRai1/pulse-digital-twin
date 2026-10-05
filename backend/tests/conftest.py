"""Shared fixtures. Models are trained once per test session (reduced size, ~1 min)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import REPO_DIR, Settings
from app.engines.srp import cnn
from app.engines.thermal import ml_correction
from app.field.build import build_store
from app.field.raw import SIDE_FILES
from app.main import create_app
from app.security import ALL_LIMITERS


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory) -> Path:
    # allow re-using pre-trained artifacts (fast local runs): PULSE_TEST_MODELS=/path/to/models
    pre = os.environ.get("PULSE_TEST_MODELS")
    md = tmp_path_factory.mktemp("models")
    if pre and (Path(pre) / cnn.NUMPY_FILE).exists():
        for f in Path(pre).iterdir():
            shutil.copy(f, md / f.name)
        return md
    cnn.train(md, n_per_class=250, epochs=12)
    ml_correction.train_synthetic(md, n_cycles=90)
    return md


def make_settings(tmp: Path, models_dir: Path, **kw) -> Settings:
    base = dict(
        database_url=f"sqlite:///{tmp}/test.db",
        models_dir=models_dir,
        samples_dir=REPO_DIR / "data" / "samples",
        sim_enabled=False,  # tests drive ticks explicitly
        auto_train=False,
        env="test",
        # the live-twin tests do not need the 300-well dataset: point at an empty folder
        field_data_dir=tmp / "no-field-data",
        field_store_dir=tmp / "field_store",
        field_auto_build=False,
    )
    base.update(kw)
    return Settings(**base)


@pytest.fixture(scope="module")
def client(models_dir, tmp_path_factory):
    app = create_app(make_settings(tmp_path_factory.mktemp("db"), models_dir))
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_limiters():
    for lim in ALL_LIMITERS:
        lim.reset()
    yield


# --------------------------------------------------------------------------- field dataset
FIELD_DIR = REPO_DIR / "data" / "field"
SUBSET_EVERY = 8  # every 8th well -> 38 wells, ~76k well-days: realistic but fast to build


@pytest.fixture(scope="session")
def field_subset_dir(tmp_path_factory) -> Path:
    """A real subset of the committed dataset (whole wells, all columns)."""
    if not (FIELD_DIR / "production_history.parquet").exists():
        pytest.skip("field dataset not present in data/field")
    out = tmp_path_factory.mktemp("field_data")
    prod = pd.read_parquet(FIELD_DIR / "production_history.parquet")
    wells = sorted(prod["well_id"].astype(str).unique())[::SUBSET_EVERY]
    keep = set(wells)
    for name in ("production_history", "srp_vfd_data", "css_cycle_log"):
        df = prod if name == "production_history" else pd.read_parquet(FIELD_DIR / f"{name}.parquet")
        df[df["well_id"].astype(str).isin(keep)].to_parquet(out / f"{name}.parquet", index=False)
    fl = pd.read_csv(FIELD_DIR / "rod_failure_log.csv")
    fl[fl["well_id"].isin(keep)].to_csv(out / "rod_failure_log.csv", index=False)
    for f in SIDE_FILES:
        if (FIELD_DIR / f).exists():
            shutil.copy(FIELD_DIR / f, out / f)
    return out


@pytest.fixture(scope="session")
def field_store_dir(field_subset_dir, tmp_path_factory) -> Path:
    store = tmp_path_factory.mktemp("field_store")
    build_store(field_subset_dir, store, progress=lambda *_: None)
    return store


@pytest.fixture(scope="module")
def field_client(models_dir, field_subset_dir, field_store_dir, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fdb")
    settings = make_settings(tmp, models_dir, field_data_dir=field_subset_dir, field_store_dir=field_store_dir)
    app = create_app(settings)
    with TestClient(app) as c:
        assert c.app.state.container.field.wait(120), c.app.state.container.field.status()
        yield c
