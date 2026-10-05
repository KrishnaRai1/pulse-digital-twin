"""Build the derived field store (tables, models, evaluation) from ``data/field``.

    python -m scripts.build_field_store
    python -m scripts.build_field_store --data ../data/field --out field_store

Takes a few minutes on a laptop CPU. The API does the same automatically in the background
when the store is missing or the data changed (``PULSE_FIELD_AUTO_BUILD=true``).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.config import get_settings
from app.field.build import build_store


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=s.field_data_dir)
    ap.add_argument("--out", type=Path, default=s.field_store_dir)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    meta = build_store(args.data, args.out)
    ew = meta["early_warning"]["results"]
    print(json.dumps({
        "build_seconds": meta["build_seconds"],
        "early_warning_pr_auc": {k: round(v["pr_auc"], 3) for k, v in ew.items()},
        "nowcast_mae_bbl_d": {k: round(v["mae"], 2) for k, v in meta["nowcast"]["metrics_oil_rate_bbl_d"].items()},
        "cycle_model_r2": {k: round(v["r2"], 3) for k, v in meta["cycle_model"]["metrics"].items()},
        "spm_advisor": {k: meta["spm_advisor"][k] for k in ("logged", "advisor")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
