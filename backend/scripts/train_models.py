"""Train / retrain the model artifacts.

    python -m scripts.train_models                 # CNN + thermal residual model (synthetic data)
    python -m scripts.train_models --only cnn
    python -m scripts.train_models --cards field_cards.csv   # add labelled field cards to the CNN

Labelled card CSV (one card per row; arrays are JSON lists):
    well_id,spm,position,load,label
    BGW-03,4.2,"[0.1,0.2,...]","[38.2,39.0,...]",FLUID_POUND
Position in metres, load in kN, label one of NORMAL, ROD_FLOATING, PUMP_UNSETTING_RISK,
FLUID_POUND, GAS_INTERFERENCE.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import get_settings
from app.domain.wells import build_fleet
from app.engines.srp import cnn
from app.engines.srp.cards import CLASSES, analyze_surface_card, card_tensor
from app.engines.thermal import ml_correction
from app.services.etl import _json_array


def field_cards(path: Path) -> tuple[np.ndarray, np.ndarray]:
    fleet = {w.id: w for w in build_fleet()}
    df = pd.read_csv(path)
    xs, ys = [], []
    for i, r in enumerate(df.to_dict("records")):
        wid, label = str(r["well_id"]).strip().upper(), str(r["label"]).strip().upper()
        if wid not in fleet or label not in CLASSES:
            raise SystemExit(f"row {i + 2}: unknown well or label ({wid!r}, {label!r})")
        design = fleet[wid].design
        card = analyze_surface_card(_json_array(r["position"]), _json_array(r["load"]) * 1000.0, float(r["spm"]), design)
        xs.append(card_tensor(card, design))
        ys.append(CLASSES.index(label))
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["cnn", "thermal"], help="train just one model")
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--cards", type=Path, help="labelled field cards to add to the CNN training set")
    ap.add_argument("--epochs", type=int, default=22)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    md = args.models_dir or Path(get_settings().models_dir)

    if args.only in (None, "cnn"):
        extra = field_cards(args.cards) if args.cards else None
        meta = cnn.train(md, epochs=args.epochs, extra=extra)
        print("CNN:", json.dumps({k: meta[k] for k in ("val_accuracy", "parameters", "latency_ms_p50", "latency_ms_p95")}, indent=2))
    if args.only in (None, "thermal"):
        model = ml_correction.train_synthetic(md)
        print("Thermal residual:", json.dumps({k: v for k, v in model.meta.items() if k != "feature_importance"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
