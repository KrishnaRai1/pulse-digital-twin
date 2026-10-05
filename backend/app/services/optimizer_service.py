"""Optimiser service: recommendations, advisory lifecycle and the safety gate.

Flow:  twin forecast -> MPC (inside the physics envelope) -> XAI -> *advisory* row.
An advisory is applied to the SCADA adapter only after an operator approves it
(advisory mode) or automatically by policy (closed-loop mode) - and in both cases the
set-point is re-validated against the hard physical limits at the moment of writing.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..config import Settings
from ..engines.optimizer import mpc, xai
from ..engines.optimizer.cycle_planner import PlanEconomics, plan_steam_volume
from ..engines.optimizer.limits import check_setpoint, spm_limits
from . import storage
from .scada import ScadaAdapter
from .stream import StreamHub
from .twin import Twin


class OptimizerError(Exception):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


class OptimizerService:
    def __init__(self, settings: Settings, engine, twin: Twin, scada: ScadaAdapter, hub: StreamHub):
        self.settings = settings
        self.engine = engine
        self.twin = twin
        self.scada = scada
        self.hub = hub

    def economics(self) -> mpc.Economics:
        return mpc.Economics(oil_usd_m3=self.settings.oil_price_usd_per_m3, power_usd_kwh=self.settings.power_price_usd_per_kwh)

    # ------------------------------------------------------------------ recommendation
    def recommend(self, well_id: str, ts: float, spm_now: float | None = None) -> dict[str, Any]:
        cfg = self.twin.fleet[well_id]
        if spm_now is None:
            spm_now = self.scada.read_setpoint(well_id)
        state = self.twin.state_at(well_id, ts)
        if state["phase"] != "PRODUCTION":
            return {
                "well_id": well_id,
                "ts": int(ts),
                "applicable": False,
                "reason": f"Well is in {state['phase'].replace('_', ' ').lower()}: no pumping-speed optimisation.",
                "mode": self.settings.control_mode,
            }
        fc, extras = self.twin.forecast(well_id, ts, horizon_days=14)
        design = cfg.design
        res = mpc.solve(design, self.economics(), fc, spm_now)
        rec = round(res.recommended * 2) / 2
        f_load = design.fluid_load_n(fc.net_lift_m)
        violations = check_setpoint(rec, design, float(fc.mu_tub_pa_s[0]), f_load)
        if violations:  # defence in depth: clamp to the envelope, never trust the solver alone
            lim = spm_limits(design, float(fc.mu_tub_pa_s[0]), f_load)
            rec = max(design.min_spm, min(rec, lim.spm_max))
            violations = check_setpoint(rec, design, float(fc.mu_tub_pa_s[0]), f_load) if lim.feasible else violations
        res.plan_spm[0] = rec
        explanation = xai.explain(
            res,
            spm_now,
            extras["mu_now_cp"],
            extras["mu_24h_ago_cp"],
            extras["mu_7d_cp"],
            self.twin.ml_top_features(well_id),
        )
        plan = [
            {
                "day": k,
                "spm": float(res.plan_spm[k]),
                "oil_m3d": s["oil_m3d"],
                "kw": s["kw"],
                "float_index": s["float_index"],
                "fillage": s["fillage"],
                "mu_tub_cp": float(fc.mu_tub_pa_s[k] * 1000.0),
                "q_liquid_m3d": float(fc.q_liquid_m3d[k]),
            }
            for k, s in enumerate(res.stage)
        ]
        action = "hold" if abs(rec - spm_now) < 0.25 else "adjust"
        return {
            "well_id": well_id,
            "ts": int(ts),
            "applicable": True,
            "mode": self.settings.control_mode,
            "current_spm": float(spm_now),
            "recommended_spm": float(rec),
            "action": action,
            "needs_resteam": not res.limits_now.feasible,
            "limits": res.limits_now.as_dict(),
            "plan": plan,
            "candidates": res.candidates,
            "envelope": res.envelope,
            "xai": explanation,
            "safety": {"passes": not violations, "violations": violations},
            "notes": res.notes,
            "objective_usd": res.objective_usd,
            "hold_objective_usd": res.hold_objective_usd,
        }

    # ------------------------------------------------------------------ advisories
    def create_advisory(self, well_id: str, ts: float, actor: str = "pulse-mpc", force: bool = False) -> dict | None:
        rec = self.recommend(well_id, ts)
        if not rec.get("applicable"):
            return None
        if rec["action"] == "hold" and not force:
            return None
        pending = storage.list_advisories(self.engine, well_id=well_id, status="pending", limit=1)
        if pending and abs(pending[0]["recommended_spm"] - rec["recommended_spm"]) < 0.25 and not force:
            return pending[0]
        row = {
            "ts": int(time.time()),
            "well_id": well_id,
            "current_spm": rec["current_spm"],
            "recommended_spm": rec["recommended_spm"],
            "status": "pending",
            "payload": _advisory_payload(rec),
        }
        adv_id = storage.insert_advisory(self.engine, row)
        storage.expire_pending(self.engine, well_id, except_id=adv_id)
        storage.audit(self.engine, actor, "advisory_created", well_id, {"id": adv_id, "recommended_spm": rec["recommended_spm"]})
        adv = storage.get_advisory(self.engine, adv_id)
        self.hub.publish({"type": "advisory", "well_id": well_id, "advisory": _public(adv)})
        if self.settings.control_mode == "closed_loop" and rec["safety"]["passes"]:
            return self.decide(adv_id, True, actor="closed-loop-policy", note="automatic (closed-loop mode)", ts=ts)
        return adv

    def decide(self, adv_id: int, approve: bool, actor: str, note: str | None = None, ts: float | None = None) -> dict:
        adv = storage.get_advisory(self.engine, adv_id)
        if adv is None:
            raise OptimizerError("advisory not found", 404)
        if adv["status"] != "pending":
            raise OptimizerError(f"advisory is already {adv['status']}", 409)
        well_id = adv["well_id"]
        cfg = self.twin.fleet[well_id]
        if not approve:
            storage.update_advisory(self.engine, adv_id, status="rejected", decided_ts=int(time.time()), decided_by=actor, note=note)
            storage.audit(self.engine, actor, "advisory_rejected", well_id, {"id": adv_id, "note": note})
        else:
            now = ts if ts is not None else self._now(well_id)
            state = self.twin.state_at(well_id, now)
            f_load = cfg.design.fluid_load_n(cfg.design.pump_depth_m * cfg.net_lift_fraction)
            problems = check_setpoint(adv["recommended_spm"], cfg.design, state["mu_tub_pa_s"], f_load)
            if problems:  # conditions changed since the advisory was drafted
                storage.update_advisory(self.engine, adv_id, status="expired", decided_ts=int(time.time()), decided_by=actor, note="; ".join(problems))
                storage.audit(self.engine, actor, "advisory_blocked_by_safety_layer", well_id, {"id": adv_id, "problems": problems})
                raise OptimizerError("Blocked by the physics safety layer: " + " ".join(problems), 409)
            self.scada.write_setpoint(well_id, adv["recommended_spm"], actor)
            storage.update_advisory(self.engine, adv_id, status="applied", decided_ts=int(time.time()), decided_by=actor, note=note)
            storage.audit(self.engine, actor, "setpoint_applied", well_id, {"id": adv_id, "from": adv["current_spm"], "to": adv["recommended_spm"]})
        out = storage.get_advisory(self.engine, adv_id)
        self.hub.publish({"type": "advisory", "well_id": well_id, "advisory": _public(out)})
        return _public(out)

    def _now(self, well_id: str) -> float:
        rt = getattr(self, "runtime", None)
        return float(rt.sim_ts) if rt is not None else time.time()

    # ------------------------------------------------------------------ cycle planner (job body)
    def plan_cycle(self, well_id: str, volumes: list[float], overrides: dict) -> dict:
        econ = PlanEconomics(self.settings.oil_price_usd_per_m3, self.settings.steam_cost_usd_per_m3)
        result = plan_steam_volume(self.twin.sweep_for_planner(well_id, overrides), volumes, econ)
        result["well_id"] = well_id
        result["ml_used"] = self.twin.ml is not None
        return result


def _advisory_payload(rec: dict) -> dict:
    return {
        "narrative": rec["xai"]["narrative"],
        "drivers": rec["xai"]["drivers"],
        "contributions": rec["xai"]["contributions"],
        "limits": rec["limits"],
        "safety": rec["safety"],
        "needs_resteam": rec["needs_resteam"],
        "uplift_usd_per_day": rec["xai"]["uplift_usd_per_day"],
    }


def _public(adv: dict | None) -> dict | None:
    if adv is None:
        return None
    out = dict(adv)
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in out.items()}
