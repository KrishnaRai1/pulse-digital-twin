"""Field runtime: the SCADA / historian stand-in used for demos, tests and development.

It advances a simulated clock, evaluates the digital twin for each well, produces one
telemetry sample and one dynamometer card per pumping well per tick, runs the CNN on the
cards, derives the traffic-light status and raises alerts. Faults are **not scripted**:
rod floating, pump unsetting and fluid pound emerge from the physics when the current
SPM set-point violates what the (cooling, thickening) fluid allows, so approving the
optimiser's advisory genuinely clears them.

In production this module is replaced by a real SCADA/historian connector that feeds the
same ``telemetry`` / ``dyno_cards`` tables; everything downstream is unchanged.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable

import numpy as np

from ..config import Settings
from ..domain.wells import WellConfig, seed_cycles
from ..engines.optimizer.limits import spm_limits
from ..engines.srp import loads
from ..engines.srp.cards import Card, baseline_card, generate_card
from ..engines.srp.cnn_infer import DynoClassifier
from ..engines.thermal.synthetic_truth import truth_log_factor, well_hidden_factor
from . import storage
from .status import well_status
from .stream import StreamHub
from .twin import Twin

log = logging.getLogger("pulse.runtime")
BBL_PER_M3 = 6.2898
CARD_BUFFER = 60


def _seed(well_id: str, ts: int) -> int:
    return (zlib.crc32(well_id.encode()) * 1_000_003 + int(ts)) % (2**32)


def choose_fault(cfg: WellConfig, spm: float, mu_pa_s: float, q_liq: float) -> tuple[str, dict, str]:
    """(generator label, generator kwargs, ground-truth label) from the physical condition."""
    d = cfg.design
    idx = loads.float_index(spm, mu_pa_s, d)
    cap = d.pump_capacity_m3d(spm)
    fill = min(1.0, q_liq / cap) if cap > 0 else 1.0
    if idx >= 1.15:
        return "PUMP_UNSETTING_RISK", {"severity": float(np.clip(0.4 + (idx - 1.15) / 0.5 * 0.6, 0.4, 1.0))}, "PUMP_UNSETTING_RISK"
    if idx >= 0.85:
        return "ROD_FLOATING", {"severity": float(np.clip(0.3 + (idx - 0.85) / 0.30 * 0.7, 0.3, 1.0))}, "ROD_FLOATING"
    if fill < 0.88:
        return "FLUID_POUND", {"fillage": float(fill)}, "FLUID_POUND"
    if cfg.gas_prone:
        return "GAS_INTERFERENCE", {}, "GAS_INTERFERENCE"
    if idx > 0.70:  # approaching float: sub-threshold distortion, still a healthy card
        return "ROD_FLOATING", {"severity": float((idx - 0.70) / 0.15 * 0.28)}, "NORMAL"
    return "NORMAL", {}, "NORMAL"


def solve_spm_for_index(target: float, mu_pa_s: float, design, hi: float) -> float:
    lo = 0.05
    if loads.float_index(hi, mu_pa_s, design) < target:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if loads.float_index(mid, mu_pa_s, design) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class FieldRuntime:
    def __init__(
        self,
        settings: Settings,
        engine,
        twin: Twin,
        classifier: DynoClassifier | None,
        hub: StreamHub,
        fleet: list[WellConfig],
    ):
        self.settings = settings
        self.engine = engine
        self.twin = twin
        self.classifier = classifier
        self.hub = hub
        self.fleet_list = fleet
        self.fleet = {w.id: w for w in fleet}
        self.setpoints: dict[str, float] = {}
        self.latest: dict[str, dict] = {}
        self.cards: dict[str, deque] = {w.id: deque(maxlen=CARD_BUFFER) for w in fleet}
        self.sim_ts: float = float(time.time())
        self.tick_n = 0
        self.active_alerts: dict[str, set[str]] = {w.id: set() for w in fleet}
        self._streak: dict[tuple[str, str], int] = {}
        self._lock = threading.RLock()
        self.advisory_hook: Callable[[str, float], object] | None = None

    # ------------------------------------------------------------------ set-points
    def set_setpoint(self, well_id: str, spm: float) -> None:
        d = self.fleet[well_id].design
        with self._lock:
            self.setpoints[well_id] = float(np.clip(spm, 0.0, d.max_spm))

    def initial_setpoint(self, cfg: WellConfig, st: dict, ql_true: float | None = None) -> float:
        d = cfg.design
        mu = st["mu_tub_pa_s"]
        ql = max(ql_true if ql_true else st["q_liquid_m3d"], 1e-3)
        per = d.pump_capacity_m3d(1.0)
        need = ql / (per * 0.95)
        f_load = d.fluid_load_n(d.pump_depth_m * cfg.net_lift_fraction)
        lim = spm_limits(d, mu, f_load)
        safe_hi = max(d.min_spm, min(d.max_spm, lim.spm_max * 0.9))
        scen = cfg.scenario
        if scen == "cooling_float":
            n = solve_spm_for_index(1.00, mu, d, d.max_spm)
        elif scen == "unset_risk":
            n = solve_spm_for_index(1.35, mu, d, d.max_spm)
        elif scen == "near_float":
            n = solve_spm_for_index(0.78, mu, d, d.max_spm)
            if ql / max(d.pump_capacity_m3d(n), 1e-9) < 0.9:  # pump would run partly empty: keep it full instead
                n = need
        elif scen == "fluid_pound":
            n = float(np.clip(need * 2.2, d.min_spm, lim.spm_float * 0.95))
        elif scen == "gas_interference":
            n = float(np.clip(need * 1.05, d.min_spm, safe_hi))
        elif st["phase"] != "PRODUCTION":
            n = 4.0
        else:
            n = float(np.clip(need, d.min_spm, safe_hi))
        return round(float(np.clip(n, d.min_spm, d.max_spm)), 1)

    # ------------------------------------------------------------------ one telemetry sample
    def build_sample(self, cfg: WellConfig, ts: float, spm: float) -> tuple[dict, dict]:
        rng = np.random.default_rng(_seed(cfg.id, int(ts)))
        st = self.twin.state_at(cfg.id, ts)
        d = cfg.design
        net_lift = d.pump_depth_m * cfg.net_lift_fraction
        f_load = d.fluid_load_n(net_lift)
        producing = st["phase"] == "PRODUCTION" and spm > 0
        row: dict = {"well_id": cfg.id, "ts": int(ts)}
        aux: dict = {"state": st, "producing": producing, "net_lift": net_lift, "f_load": f_load}
        mu = st["mu_tub_pa_s"]
        if producing:
            series = self.twin.series(cfg.id)
            lf = truth_log_factor(
                steam_m3=series.spec.steam_m3,
                quality=series.spec.quality,
                inj_days=series.spec.inj_days,
                cycle_no=series.spec.cycle_no,
                t_prod_days=st["prod_day"],
            )
            q_true = float(st["q_oil_phys_m3d"] * np.exp(lf + well_hidden_factor(cfg.id) + rng.normal(0, 0.05)))
            wc = st["water_cut"]
            ql_true = q_true / max(1.0 - wc, 0.05)
            cap = d.pump_capacity_m3d(spm)
            served = min(1.0, cap / ql_true) if ql_true > 0 else 1.0
            fill = min(1.0, ql_true / cap) if cap > 0 else 1.0
            pprl, mprl = loads.polished_rod_loads(spm, mu, d, f_load)
            kw = loads.electrical_power_kw(loads.polished_rod_power_kw(spm, mu, d, f_load))
            s_meas = spm * (1 + rng.normal(0, 0.004))
            row.update(
                spm=s_meas,
                stroke_m=d.stroke_m * (1 + rng.normal(0, 0.003)),
                vfd_hz=loads.vfd_hz_for_spm(spm, d) * (1 + rng.normal(0, 0.002)),
                motor_load_pct=float(np.clip(kw / d.motor_kw_rated * 100 * (1 + rng.normal(0, 0.02)), 0, 130)),
                motor_kw=kw * (1 + rng.normal(0, 0.02)),
                pprl_kn=pprl / 1000 * (1 + rng.normal(0, 0.01)),
                mprl_kn=max(mprl, 0.02 * d.weight_buoyant_n) / 1000 * (1 + rng.normal(0, 0.02)),
                whp_kpa=250.0 + 6.0 * max(ql_true * served, 1.0) ** 0.8 + rng.normal(0, 6),
                tubing_temp_c=st["t_wellhead_c"] + rng.normal(0, 0.4),
                oil_rate_m3d=q_true * served,
                liquid_rate_m3d=ql_true * served,
                water_cut=wc,
                visc_cp=st["mu_tub_cp"],
                fillage_pct=fill * 100.0,
            )
            aux.update(ql_true=ql_true, fill=fill, mu=mu, spm=spm, kw=kw)
        else:
            inj = st["phase"] == "INJECTION"
            row.update(
                spm=0.0,
                stroke_m=0.0,
                vfd_hz=0.0,
                motor_load_pct=1.0,
                motor_kw=0.4,
                pprl_kn=0.0,
                mprl_kn=0.0,
                whp_kpa=(1600.0 + rng.normal(0, 25)) if inj else (900.0 - min(st["day_in_cycle"], 60) * 3 + rng.normal(0, 8)),
                tubing_temp_c=(st["t_steam_c"] * 0.97) if inj else st["t_wellbore_c"] * 0.85,
                oil_rate_m3d=0.0,
                liquid_rate_m3d=0.0,
                water_cut=0.0,
                visc_cp=st["mu_tub_cp"],
                fillage_pct=0.0,
            )
            aux.update(ql_true=0.0, fill=0.0, mu=mu, spm=0.0, kw=0.4)
        return row, aux

    def make_card(self, cfg: WellConfig, ts: float, spm: float, aux: dict) -> Card:
        rng = np.random.default_rng(_seed(cfg.id, int(ts)) ^ 0xA5A5)
        gen_label, kw, truth = choose_fault(cfg, spm, aux["mu"], aux["ql_true"])
        card = generate_card(cfg.design, spm, aux["mu"], aux["net_lift"], gen_label, rng=rng, noise=0.005, **kw)
        card.label = truth
        return card

    # ------------------------------------------------------------------ lifecycle
    def warm_start(self) -> None:
        now = int(time.time())
        step = int(self.settings.sim_minutes_per_tick * 60)
        for cfg in self.fleet_list:
            if not storage.get_cycles(self.engine, cfg.id):
                for row in seed_cycles(cfg, now, self.settings.sim_seed):
                    storage.upsert_cycle(self.engine, row)
        last_ts = [storage.latest_telemetry_ts(self.engine, c.id) for c in self.fleet_list]
        last = max([t for t in last_ts if t is not None], default=None)
        self.sim_ts = float(max(now, (last + step) if last is not None else now))
        n_hist = int(self.settings.sim_history_hours * 3600 / step)
        for cfg in self.fleet_list:
            _, aux0 = self.build_sample(cfg, self.sim_ts, 4.0)
            st = aux0["state"]
            prev = storage.latest_telemetry_row(self.engine, cfg.id)
            if prev is not None and prev.get("spm"):
                spm = float(round(prev["spm"], 1))
            else:
                spm = self.initial_setpoint(cfg, st, aux0["ql_true"])
            self.setpoints[cfg.id] = spm
            if last is None:  # first run: back-fill the trailing history window
                rows = [self.build_sample(cfg, self.sim_ts - (n_hist - k) * step, spm)[0] for k in range(n_hist)]
                storage.insert_telemetry(self.engine, rows)
        self.tick(advance=False)
        if self.advisory_hook:
            for wid, s in list(self.latest.items()):
                if s["status"] != "green" and s["producing"]:
                    try:
                        self.advisory_hook(wid, self.sim_ts)
                    except Exception:
                        log.exception("advisory generation failed for %s", wid)

    def tick(self, advance: bool = True) -> dict:
        with self._lock:
            step = self.settings.sim_minutes_per_tick * 60
            if advance:
                self.sim_ts += step
                self.tick_n += 1
            ts = int(self.sim_ts)
            rows, auxes, cards = [], {}, {}
            for cfg in self.fleet_list:
                spm = self.setpoints[cfg.id]
                row, aux = self.build_sample(cfg, ts, spm)
                rows.append(row)
                auxes[cfg.id] = aux
                if aux["producing"]:
                    cards[cfg.id] = self.make_card(cfg, ts, spm, aux)
            cls: dict[str, dict] = {}
            if cards and self.classifier is not None:
                ids = list(cards)
                out = self.classifier.classify([cards[i] for i in ids], [self.fleet[i].design for i in ids])
                cls = dict(zip(ids, out))
            events: list[dict] = []
            for cfg in self.fleet_list:
                self._update_well(cfg, ts, auxes[cfg.id], cards.get(cfg.id), cls.get(cfg.id), events)
            if advance:
                storage.insert_telemetry(self.engine, rows)
                if self.tick_n % 200 == 0:
                    from .. import db

                    db.prune_telemetry(self.engine, ts - self.settings.telemetry_retention_hours * 3600)
                if self.advisory_hook and self.tick_n % 6 == 0:
                    for wid, s in list(self.latest.items()):
                        if s["producing"] and s["status"] != "green":
                            try:
                                self.advisory_hook(wid, ts)
                            except Exception:
                                log.exception("advisory generation failed for %s", wid)
            tick_msg = {"type": "tick", "ts": ts, "wells": [self._compact(w) for w in self.latest.values()], "kpis": self.kpis()}
            events.insert(0, tick_msg)
            for e in events:
                self.hub.publish(e)
            return {"ts": ts, "events": len(events)}

    def _update_well(self, cfg: WellConfig, ts: int, aux: dict, card: Card | None, cls: dict | None, events: list[dict]) -> None:
        st = aux["state"]
        d = cfg.design
        spm = self.setpoints[cfg.id]
        lim = None
        idx = None
        if aux["producing"]:
            lim = spm_limits(d, aux["mu"], aux["f_load"])
            idx = loads.float_index(spm, aux["mu"], d)
        label = cls["label"] if cls else None
        prob = cls["probability"] if cls else 0.0
        status, reasons = well_status(
            phase=st["phase"],
            label=label,
            probability=prob,
            thermal_health=st["thermal_health"],
            spm=spm if aux["producing"] else 0.0,
            limits=lim,
            float_index=idx,
        )
        sor = (st["steam_injected_m3"] / st["cum_oil_m3"]) if st["cum_oil_m3"] > 1.0 else None
        prev = self.latest.get(cfg.id, {})
        summary = {
            "id": cfg.id,
            "name": cfg.name,
            "pad": cfg.pad,
            "x_m": cfg.x_m,
            "y_m": cfg.y_m,
            "ts": ts,
            "phase": st["phase"],
            "cycle_no": st["cycle_no"],
            "day_in_cycle": st["day_in_cycle"],
            "prod_day": st["prod_day"],
            "producing": aux["producing"],
            "status": status,
            "reasons": reasons,
            "spm": spm if aux["producing"] else 0.0,
            "setpoint_spm": spm,
            "oil_rate_m3d": self._rate(cfg.id, aux, "oil"),
            "liquid_rate_m3d": self._rate(cfg.id, aux, "liq"),
            "water_cut": st["water_cut"],
            "visc_cp": st["mu_tub_cp"],
            "mu_eff_cp": st["mu_eff_cp"],
            "t_avg_c": st["t_avg_c"],
            "t_wellbore_c": st["t_wellbore_c"],
            "thermal_health": st["thermal_health"],
            "kw": aux["kw"],
            "cum_oil_m3": st["cum_oil_m3"],
            "steam_injected_m3": st["steam_injected_m3"],
            "sor": sor,
            "float_index": idx,
            "spm_ceiling": lim.spm_max if lim else None,
            "diag": {"label": label, "probability": prob} if cls else None,
        }
        self.latest[cfg.id] = summary
        if card is not None and cls is not None:
            payload = _card_payload(card, cls, cfg.id, ts, "live", d.weight_buoyant_n)
            self.cards[cfg.id].append(payload)
            events.append({"type": "dyno", **payload})
            if cls["label"] != "NORMAL" and cls["probability"] >= 0.8 and self.tick_n % 6 == 0:
                storage.insert_card(
                    self.engine,
                    {
                        "well_id": cfg.id,
                        "ts": ts,
                        "source": "live",
                        "spm": spm,
                        "position": [round(float(v), 4) for v in card.position],
                        "load": [round(float(v), 1) for v in card.load],
                        "label": cls["label"],
                        "probability": cls["probability"],
                        "probabilities": cls["probabilities"],
                    },
                )
        self._alerts(cfg, label, prob, ts, events, prev)

    def _rate(self, well_id: str, aux: dict, kind: str) -> float:
        if not aux["producing"]:
            return 0.0
        cap = self.fleet[well_id].design.pump_capacity_m3d(aux["spm"])
        liq = aux["ql_true"] * min(1.0, cap / max(aux["ql_true"], 1e-9))
        if kind == "liq":
            return float(liq)
        return float(liq * (1.0 - aux["state"]["water_cut"]))

    def _alerts(self, cfg: WellConfig, label: str | None, prob: float, ts: int, events: list[dict], prev: dict) -> None:
        from ..engines.srp.cards import CLASS_LABELS, FAULT_SEVERITY

        active = self.active_alerts[cfg.id]
        kind = label if label and label != "NORMAL" and prob >= 0.8 else None
        for k in list(active):
            if k != kind:
                self._streak.pop((cfg.id, k), None)
                active.discard(k)
                events.append({"type": "alert_cleared", "well_id": cfg.id, "kind": k, "ts": ts})
        if kind:
            n = self._streak.get((cfg.id, kind), 0) + 1
            self._streak[(cfg.id, kind)] = n
            if n >= 2 and kind not in active:
                active.add(kind)
                msg = f"{CLASS_LABELS[kind]} on {cfg.name} ({prob * 100:.0f}% confidence)"
                sev = FAULT_SEVERITY[kind]
                alert_id = storage.insert_alert(
                    self.engine, {"ts": ts, "well_id": cfg.id, "kind": kind, "severity": sev, "message": msg, "probability": prob}
                )
                events.append({"type": "alert", "id": alert_id, "well_id": cfg.id, "kind": kind, "severity": sev, "message": msg, "probability": prob, "ts": ts})

    # ------------------------------------------------------------------ views
    def _compact(self, s: dict) -> dict:
        return {k: s[k] for k in ("id", "status", "phase", "spm", "oil_rate_m3d", "visc_cp", "kw", "producing")} | {
            "diag": s["diag"]["label"] if s["diag"] else None
        }

    def kpis(self) -> dict:
        wells = list(self.latest.values())
        oil = sum(w["oil_rate_m3d"] for w in wells)
        liq = sum(w["liquid_rate_m3d"] for w in wells)
        kw = sum(w["kw"] for w in wells)
        steam = sum(w["steam_injected_m3"] for w in wells if w["cum_oil_m3"] > 1.0)
        cum = sum(w["cum_oil_m3"] for w in wells if w["cum_oil_m3"] > 1.0)
        counts = {"green": 0, "amber": 0, "red": 0}
        for w in wells:
            counts[w["status"]] += 1
        active = sum(len(v) for v in self.active_alerts.values())
        return {
            "total_oil_m3d": oil,
            "total_oil_bbld": oil * BBL_PER_M3,
            "total_liquid_m3d": liq,
            "avg_sor": (steam / cum) if cum > 1.0 else None,
            "total_power_kw": kw,
            "wells": len(wells),
            "pumping_wells": sum(1 for w in wells if w["producing"]),
            "status_counts": counts,
            "active_alerts": active,
            "sim_ts": int(self.sim_ts),
        }

    def overview(self) -> dict:
        with self._lock:
            order = {"red": 0, "amber": 1, "green": 2}
            wells = sorted(self.latest.values(), key=lambda w: (order[w["status"]], w["id"]))
            return {"kpis": self.kpis(), "wells": wells, "control_mode": self.settings.control_mode}

    def latest_card(self, well_id: str) -> dict | None:
        with self._lock:
            return self.cards[well_id][-1] if self.cards[well_id] else None

    def baseline_for(self, well_id: str) -> dict | None:
        s = self.latest.get(well_id)
        if not s or not s["producing"]:
            return None
        cfg = self.fleet[well_id]
        st = self.twin.state_at(well_id, self.sim_ts)
        b = baseline_card(cfg.design, self.setpoints[well_id], st["mu_tub_pa_s"], cfg.design.pump_depth_m * cfg.net_lift_fraction)
        return {
            "position": _r(b.position, 4),
            "load": _r(b.load / 1000.0, 3),
            "dh_position": _r(b.dh_position, 4),
            "dh_load": _r(b.dh_load / 1000.0, 3),
        }

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.settings.sim_tick_seconds)
            try:
                await asyncio.to_thread(self.tick)
            except Exception:
                log.exception("simulator tick failed")


def _r(a, n: int) -> list[float]:
    return [round(float(v), n) for v in a]


def _card_payload(card: Card, cls: dict, well_id: str, ts: int, source: str, wb: float) -> dict:
    return {
        "well_id": well_id,
        "ts": ts,
        "source": source,
        "spm": round(float(card.spm), 2),
        "position": _r(card.position, 4),
        "load": _r(card.load / 1000.0, 3),
        "dh_position": _r(card.dh_position, 4),
        "dh_load": _r(card.dh_load / 1000.0, 3),
        "label": cls["label"],
        "probability": round(cls["probability"], 4),
        "probabilities": {k: round(v, 4) for k, v in cls["probabilities"].items()},
        "latency_ms": round(cls["latency_ms"], 3),
        "buoyant_rod_weight_kn": round(wb / 1000.0, 3),
        "truth": card.label,
    }


card_payload = _card_payload
