"""Regenerate the demo upload files in ``data/samples``.

    python -m scripts.generate_samples

* ``sample_production_log.csv``  6 h of 3-minute sensor data for three wells, deliberately
  "dirty": missing values, out-of-range spikes, duplicated rows, a water-cut column in percent,
  so the ETL cleaning report has something to show.
* ``sample_dyno_cards.csv``      surface dynamometer cards (JSON-array format, kN and metres)
  covering all five diagnostic classes.

All values are synthetic (produced by the same physics engines as the live demo).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import REPO_DIR, Settings
from app.db import init_db, make_engine
from app.domain.wells import build_fleet, seed_cycles
from app.engines.srp.cards import generate_card
from app.services import storage
from app.services.simulator import FieldRuntime
from app.services.stream import StreamHub
from app.services.twin import Twin

NOW = 1_790_000_000  # fixed reference time so the files are reproducible
OUT = REPO_DIR / "data" / "samples"


def build_runtime(tmp: Path) -> FieldRuntime:
    settings = Settings(database_url=f"sqlite:///{tmp}/s.db", sim_enabled=False, auto_train=False, env="test")
    engine = make_engine(settings.database_url)
    init_db(engine)
    fleet = build_fleet()
    for cfg in fleet:
        for row in seed_cycles(cfg, NOW, settings.sim_seed):
            storage.upsert_cycle(engine, row)
    twin = Twin(engine, {w.id: w for w in fleet}, None)
    rt = FieldRuntime(settings, engine, twin, None, StreamHub(), fleet)
    for cfg in fleet:
        _, aux = rt.build_sample(cfg, NOW, 4.0)
        rt.setpoints[cfg.id] = rt.initial_setpoint(cfg, aux["state"], aux["ql_true"])
    rt.sim_ts = NOW
    return rt


def production_log(rt: FieldRuntime) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for wid in ("BGW-01", "BGW-04", "BGW-10"):
        cfg = rt.fleet[wid]
        for k in range(120):  # 6 h at 3 min
            ts = NOW - (120 - k) * 180
            row, _ = rt.build_sample(cfg, ts, rt.setpoints[wid])
            row.pop("ts")
            row["timestamp"] = pd.Timestamp(ts, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            rows.append(row)
    df = pd.DataFrame(rows)
    df["water_cut"] = (df["water_cut"] * 100).round(2)  # percent, on purpose
    df = df.round(3)
    # inject realistic data-quality problems
    idx = rng.choice(len(df), 40, replace=False)
    df.loc[idx[:20], "oil_rate_m3d"] = np.nan
    df.loc[idx[20:32], "visc_cp"] = np.nan
    df.loc[idx[32:36], "motor_kw"] = 9999.0  # sensor spike
    df.loc[idx[36:], "whp_kpa"] = -99.0  # sentinel value
    df = pd.concat([df, df.sample(6, random_state=1)]).sort_values(["well_id", "timestamp"])  # duplicates
    cols = ["timestamp", "well_id", "spm", "stroke_m", "vfd_hz", "motor_load_pct", "motor_kw", "pprl_kn", "mprl_kn", "whp_kpa", "tubing_temp_c", "oil_rate_m3d", "liquid_rate_m3d", "water_cut", "visc_cp", "fillage_pct"]
    return df[cols]


def dyno_cards(rt: FieldRuntime) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    plan = [("BGW-01", "NORMAL"), ("BGW-03", "ROD_FLOATING"), ("BGW-06", "PUMP_UNSETTING_RISK"), ("BGW-07", "FLUID_POUND"), ("BGW-11", "GAS_INTERFERENCE")]
    rows = []
    for wid, label in plan:
        cfg = rt.fleet[wid]
        _, aux = rt.build_sample(cfg, NOW, rt.setpoints[wid])
        spm = rt.setpoints[wid]
        for k in range(3):
            gen, kw, _ = (label, {}, label)
            if label == "FLUID_POUND":
                kw = {"fillage": float(rng.uniform(0.35, 0.6))}
            elif label == "ROD_FLOATING":
                kw = {"severity": float(rng.uniform(0.5, 0.9))}
            elif label == "PUMP_UNSETTING_RISK":
                kw = {"severity": float(rng.uniform(0.6, 1.0))}
            card = generate_card(cfg.design, spm, aux["mu"], aux["net_lift"], gen, rng=rng, noise=0.006, **kw)
            rows.append(
                {
                    "well_id": wid,
                    "timestamp": pd.Timestamp(NOW + k * 60 + len(rows), unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "spm": spm,
                    "position": json.dumps([round(float(v), 4) for v in card.position]),
                    "load": json.dumps([round(float(v) / 1000.0, 3) for v in card.load]),
                    "expected_label": label,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        rt = build_runtime(Path(d))
        production_log(rt).to_csv(OUT / "sample_production_log.csv", index=False)
        dyno_cards(rt).to_csv(OUT / "sample_dyno_cards.csv", index=False)
    print("wrote", *sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
