"""Diagnostics service: classify a card (single or batch) with the wave-equation + CNN pipeline."""
from __future__ import annotations

import time

import numpy as np
from sqlalchemy.engine import Engine

from ..domain.wells import WellConfig
from ..engines.srp.cards import analyze_surface_card
from ..engines.srp.cnn_infer import DynoClassifier
from ..engines.srp.rods import SrpDesign
from . import storage
from .etl import UNIT_LOAD_N, UNIT_POS_M
from .simulator import card_payload
from .stream import StreamHub
from .twin import Twin


class DiagnosticsUnavailable(RuntimeError):
    pass


class DiagnosticsService:
    def __init__(self, engine: Engine, twin: Twin, runtime, classifier: DynoClassifier | None, hub: StreamHub):
        self.engine = engine
        self.twin = twin
        self.runtime = runtime
        self.classifier = classifier
        self.hub = hub

    def _design_and_mu(self, well_id: str | None, viscosity_cp: float | None) -> tuple[SrpDesign, float, WellConfig | None]:
        cfg = self.twin.fleet.get(well_id) if well_id else None
        if well_id and cfg is None:
            raise KeyError(well_id)
        if cfg is None:
            cfg = next(iter(self.twin.fleet.values()))
            named = None
        else:
            named = cfg
        if viscosity_cp:
            mu = viscosity_cp * 1e-3
        elif named is not None:
            mu = self.twin.state_at(named.id, self.runtime.sim_ts)["mu_tub_pa_s"]
        else:
            mu = 0.2
        return cfg.design, float(mu), named

    def classify_arrays(
        self,
        well_id: str | None,
        spm: float,
        position: list[float] | np.ndarray,
        load: list[float] | np.ndarray,
        position_unit: str = "m",
        load_unit: str = "kN",
        viscosity_cp: float | None = None,
    ) -> dict:
        if self.classifier is None:
            raise DiagnosticsUnavailable("CNN model is not loaded (train it with `python -m scripts.train_models`)")
        t0 = time.perf_counter()
        design, mu, named = self._design_and_mu(well_id, viscosity_cp)
        pos = np.asarray(position, dtype=float) * UNIT_POS_M[position_unit]
        ld = np.asarray(load, dtype=float) * UNIT_LOAD_N[load_unit]
        card = analyze_surface_card(pos, ld, spm, design, mu)
        res = self.classifier.classify([card], [design])[0]
        total_ms = (time.perf_counter() - t0) * 1000.0
        payload = card_payload(card, res, well_id or "-", int(self.runtime.sim_ts), "api", design.weight_buoyant_n)
        payload["latency_ms_total"] = round(total_ms, 3)
        payload["within_budget"] = total_ms <= 200.0
        payload["viscosity_cp_used"] = round(mu * 1000.0, 1)
        payload["well_known"] = named is not None
        return payload

    def analyze_uploaded(self, cards: list[dict], replay_hz: float = 4.0) -> dict:
        """Job body: classify all cards, persist them, replay them to dashboards one by one."""
        if self.classifier is None:
            raise DiagnosticsUnavailable("CNN model is not loaded")
        analysed, rows = [], []
        for c in cards:
            design, mu, _ = self._design_and_mu(c["well_id"], None)
            card = analyze_surface_card(c["position"], c["load"], c["spm"], design, mu)
            res = self.classifier.classify([card], [design])[0]
            payload = card_payload(card, res, c["well_id"], int(c["ts"]), "upload", design.weight_buoyant_n)
            analysed.append(payload)
            rows.append(
                {
                    "well_id": c["well_id"],
                    "ts": int(c["ts"]),
                    "source": "upload",
                    "spm": float(c["spm"]),
                    "position": payload["position"],
                    "load": [round(v * 1000.0, 1) for v in payload["load"]],
                    "label": res["label"],
                    "probability": res["probability"],
                    "probabilities": res["probabilities"],
                }
            )
        storage.insert_cards(self.engine, rows)
        for p in analysed[-200:]:  # replay the tail so long files do not tie up the socket
            self.hub.publish({"type": "dyno", **p})
            time.sleep(1.0 / max(replay_hz, 0.5))
        counts: dict[str, int] = {}
        for p in analysed:
            counts[p["label"]] = counts.get(p["label"], 0) + 1
        return {"cards": len(analysed), "label_counts": counts, "wells": sorted({p["well_id"] for p in analysed})}
