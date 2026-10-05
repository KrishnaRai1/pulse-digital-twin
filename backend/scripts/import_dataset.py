"""Import the PULSE field dataset (CSV files as delivered) into ``data/field``.

    python -m scripts.import_dataset --src ~/Downloads/synthetic_data
    python -m scripts.import_dataset --src dir1 --src dir2 --out ../data/field

The three large tables are converted to zstd-compressed Parquet. The conversion is checked to
be lossless (every value is compared after a round trip) and the SHA-256 of each source CSV is
recorded in ``MANIFEST.json``. The small files are copied unchanged. Files are matched by name
suffix, so ``0c55da9a-css_cycle_log.csv`` is recognised as ``css_cycle_log.csv``.

After importing, rebuild the derived store and models with ``python -m scripts.build_field_store``
(the API also rebuilds automatically on its next start when the data changed).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import REPO_DIR
from app.field.raw import SIDE_FILES, TABLES, DatasetError, sha256_file, validate_columns

CATEGORICAL = ("well_id", "phase", "condition_label")


def _find(srcs: list[Path], name: str) -> Path | None:
    hits: list[Path] = []
    for s in srcs:
        if s.is_file() and s.name.endswith(name):
            hits.append(s)
        elif s.is_dir():
            hits += [p for p in s.iterdir() if p.is_file() and p.name.endswith(name)]
    return sorted(hits)[0] if hits else None


def _equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        if a[c].dtype.kind in "fiub" and b[c].dtype.kind in "fiub":
            if not np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True):
                return False
        elif not (a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all():
            return False
    return True


def convert(src: Path, dst: Path, name: str) -> dict:
    t0 = time.time()
    df = pd.read_csv(src)
    validate_columns(df, name)
    out = df.copy()
    for c in CATEGORICAL:
        if c in out.columns:
            out[c] = out[c].astype("category")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, compression="zstd", compression_level=19, index=False)
    back = pd.read_parquet(dst)
    if not _equal(df, back):
        raise DatasetError(f"{name}: Parquet round trip is not lossless")
    return {
        "file": dst.name,
        "source": src.name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "source_sha256": sha256_file(src),
        "source_bytes": src.stat().st_size,
        "parquet_sha256": sha256_file(dst),
        "parquet_bytes": dst.stat().st_size,
        "lossless_roundtrip": True,
        "seconds": round(time.time() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, action="append", required=True, help="folder or file (repeatable)")
    ap.add_argument("--out", type=Path, default=REPO_DIR / "data" / "field")
    args = ap.parse_args(argv)
    srcs = [p.expanduser().resolve() for p in args.src]
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    old = json.loads((out / "MANIFEST.json").read_text()) if (out / "MANIFEST.json").is_file() else {}
    manifest: dict = {"imported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tables": {}, "files": {}}
    for name in TABLES:
        src = _find(srcs, f"{name}.csv")
        if src is None:
            if (out / f"{name}.parquet").is_file():  # already imported: keep it
                manifest["tables"][name] = old.get("tables", {}).get(name, {"file": f"{name}.parquet"})
                print(f"{name}: no CSV given, keeping the existing {name}.parquet")
                continue
            print(f"error: {name}.csv not found in {', '.join(map(str, srcs))}", file=sys.stderr)
            return 2
        info = convert(src, out / f"{name}.parquet", name)
        manifest["tables"][name] = info
        print(f"{name}: {info['rows']:,} rows, {info['source_bytes'] / 1e6:.1f} MB CSV -> {info['parquet_bytes'] / 1e6:.1f} MB Parquet (lossless)")

    fail = _find(srcs, "rod_failure_log.csv")
    if fail is None:
        print("error: rod_failure_log.csv not found", file=sys.stderr)
        return 2
    validate_columns(pd.read_csv(fail), "rod_failure_log")
    for fname in ("rod_failure_log.csv",) + SIDE_FILES:
        src = fail if fname == "rod_failure_log.csv" else _find(srcs, fname)
        if src is None:
            if fname in old.get("files", {}) and (out / fname).is_file():
                manifest["files"][fname] = old["files"][fname]
            elif not fname.startswith("validation_plots"):
                print(f"note: optional file {fname} not found, skipped")
            continue
        if src.resolve() != (out / fname).resolve():  # importing in place (CSVs already in data/field)
            shutil.copyfile(src, out / fname)
        manifest["files"][fname] = {"source": src.name, "sha256": sha256_file(out / fname), "bytes": (out / fname).stat().st_size}

    cfg_path = out / "generation_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        manifest["dataset"] = {k: cfg.get(k) for k in ("version", "case", "seed", "noise_level", "n_wells", "n_cycles", "min_osr")}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out / 'MANIFEST.json'}")
    print("next: python -m scripts.build_field_store")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
