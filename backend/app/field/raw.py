"""Raw field-dataset files: schema, loading and validation.

The data directory (``data/field`` by default) holds the dataset as delivered, with the
three large tables stored as lossless, zstd-compressed Parquet so they fit GitHub's file-size
limit. The loader also accepts the original ``.csv`` (or ``.csv.gz``) for any table, so a new
dataset version, or real field data in the same schema, can simply be dropped in.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["well_id", "cycle", "day"]

SCHEMA: dict[str, list[str]] = {
    "css_cycle_log": [
        "well_id", "cycle", "day", "phase", "steam_volume_m3", "steam_quality",
        "injection_pressure_kpa", "injection_temp_c", "soak_time_days",
    ],
    "production_history": [
        "well_id", "cycle", "day", "phase", "days_since_soak_start", "oil_rate_bbl_day", "water_cut_pct",
        "gross_liquid_bbl_day", "reservoir_temp_c", "oil_viscosity_cp", "prior_oil_rate_bbl_day",
        "prior_gross_liquid_bbl_day", "prior_reservoir_temp_c", "prior_oil_viscosity_cp",
    ],
    "srp_vfd_data": [
        "well_id", "cycle", "day", "phase", "pump_running", "spm", "stroke_length_in", "vfd_frequency_hz",
        "motor_load_pct", "pump_fillage_pct", "rod_floating_index", "polished_rod_peak_load_kn",
        "polished_rod_min_load_kn", "load_variability_pct", "condition_label",
    ],
    "rod_failure_log": ["well_id", "cycle", "failure_date_day", "failure_mode", "downtime_days"],
}
TABLES = ("css_cycle_log", "production_history", "srp_vfd_data")
SIDE_FILES = ("generation_config.json", "validation_report.csv", "early_warning_report.json", "validation_plots.png", "validation_plots.jpg")
STRING_COLS = ("well_id", "phase", "condition_label", "failure_mode")
PHASES = ("soak", "production", "workover")
FAILURE_MODES = ("rod_part", "pump_unset", "worn_barrel")


class DatasetError(RuntimeError):
    """The dataset is missing or does not match the expected schema."""


@dataclass
class RawDataset:
    css: pd.DataFrame
    prod: pd.DataFrame
    srp: pd.DataFrame
    failures: pd.DataFrame
    data_dir: Path
    fingerprint: str


def find_table(data_dir: Path, name: str) -> Path | None:
    for ext in (".parquet", ".csv.gz", ".csv"):
        p = data_dir / f"{name}{ext}"
        if p.is_file():
            return p
    return None


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=columns)
    else:
        df = pd.read_csv(path, usecols=columns)
    for c in STRING_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def validate_columns(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in SCHEMA[name] if c not in df.columns]
    if missing:
        raise DatasetError(f"{name}: missing columns {missing}")


def validate_values(css: pd.DataFrame, prod: pd.DataFrame, srp: pd.DataFrame, failures: pd.DataFrame) -> list[str]:
    """Structural checks the pipeline relies on. Returns a list of problems (empty = OK)."""
    issues: list[str] = []
    if prod.duplicated(KEYS).any():
        issues.append("production_history: duplicate (well_id, cycle, day) rows")
    if srp.duplicated(KEYS).any():
        issues.append("srp_vfd_data: duplicate (well_id, cycle, day) rows")
    if len(prod) != len(srp):
        issues.append(f"production_history ({len(prod)}) and srp_vfd_data ({len(srp)}) do not have the same rows")
    bad_phase = set(prod["phase"].unique()) - set(PHASES)
    if bad_phase:
        issues.append(f"production_history: unknown phases {sorted(bad_phase)}")
    bad_mode = set(failures["failure_mode"].unique()) - set(FAILURE_MODES)
    if bad_mode:
        issues.append(f"rod_failure_log: unknown failure modes {sorted(bad_mode)}")
    for name, df, cols in (
        ("production_history", prod, ["oil_rate_bbl_day", "gross_liquid_bbl_day", "prior_oil_rate_bbl_day", "oil_viscosity_cp"]),
        ("srp_vfd_data", srp, ["spm", "motor_load_pct", "pump_fillage_pct", "rod_floating_index"]),
    ):
        for c in cols:
            v = df[c].to_numpy(dtype=float)
            if not np.isfinite(v).all():
                issues.append(f"{name}.{c}: non-finite values")
            elif (v < 0).any():
                issues.append(f"{name}.{c}: negative values")
    if (css["steam_volume_m3"] <= 0).any():
        issues.append("css_cycle_log.steam_volume_m3: non-positive values")
    return issues


def load_raw(data_dir: Path) -> RawDataset:
    data_dir = Path(data_dir)
    paths = {name: find_table(data_dir, name) for name in TABLES}
    missing = [n for n, p in paths.items() if p is None]
    if missing:
        raise DatasetError(f"dataset files not found in {data_dir}: {', '.join(missing)} (.parquet, .csv or .csv.gz)")
    fail_path = data_dir / "rod_failure_log.csv"
    if not fail_path.is_file():
        raise DatasetError(f"rod_failure_log.csv not found in {data_dir}")
    css = read_table(paths["css_cycle_log"])
    prod = read_table(paths["production_history"])
    srp = read_table(paths["srp_vfd_data"])
    failures = read_table(fail_path)
    for name, df in (("css_cycle_log", css), ("production_history", prod), ("srp_vfd_data", srp), ("rod_failure_log", failures)):
        validate_columns(df, name)
    issues = validate_values(css, prod, srp, failures)
    if issues:
        raise DatasetError("; ".join(issues))
    return RawDataset(css, prod, srp, failures, data_dir, fingerprint(data_dir))


def fingerprint(data_dir: Path) -> str:
    """Cheap identity of the data files (name, size, mtime-independent content hash of the manifest
    when present, else sizes). Used to rebuild the derived store when the data changes."""
    data_dir = Path(data_dir)
    h = hashlib.sha256()
    manifest = data_dir / "MANIFEST.json"
    if manifest.is_file():
        h.update(manifest.read_bytes())
    for name in TABLES + ("rod_failure_log",):
        for ext in (".parquet", ".csv.gz", ".csv"):
            p = data_dir / f"{name}{ext}"
            if p.is_file():
                h.update(f"{p.name}:{p.stat().st_size}".encode())
    return h.hexdigest()[:16]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def read_manifest(data_dir: Path) -> dict:
    p = Path(data_dir) / "MANIFEST.json"
    return json.loads(p.read_text()) if p.is_file() else {}


__all__ = [
    "SCHEMA", "TABLES", "SIDE_FILES", "KEYS", "PHASES", "FAILURE_MODES", "DatasetError", "RawDataset",
    "load_raw", "find_table", "read_table", "validate_columns", "validate_values", "fingerprint", "sha256_file", "read_manifest",
]
