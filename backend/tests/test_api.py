"""End-to-end API tests (FastAPI TestClient, real engines, temporary SQLite database)."""
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.domain.wells import build_fleet
from app.engines.srp.cards import generate_card
from app.main import create_app
from app.security import limit_ingest

from .conftest import make_settings

API = "/api/v1"
FLEET = {w.id: w for w in build_fleet()}


@pytest.fixture(scope="module")
def container(client):
    return client.app.state.container


def _card_body(wid="BGW-01", label="NORMAL", **kw):
    cfg = FLEET[wid]
    card = generate_card(cfg.design, 2.5, 0.06, cfg.design.pump_depth_m * cfg.net_lift_fraction, label, rng=np.random.default_rng(3), **kw)
    return {"well_id": wid, "spm": 2.5, "position": card.position.tolist(), "load": (card.load / 1000).tolist(), "load_unit": "kN"}


# --------------------------------------------------------------------------- system + field
def test_health_and_info(client):
    h = client.get(f"{API}/health").json()
    assert h["status"] == "ok" and h["cnn_loaded"]
    info = client.get(f"{API}/system/info").json()
    assert info["control_mode"] == "advisory"
    assert info["models"]["cnn"]["classes"]
    assert info["safety"]["every_setpoint_revalidated_at_write"] is True


def test_field_overview_has_kpis_and_traffic_lights(client):
    ov = client.get(f"{API}/field/overview").json()
    assert len(ov["wells"]) == 12
    assert {w["status"] for w in ov["wells"]} <= {"green", "amber", "red"}
    k = ov["kpis"]
    assert k["total_oil_m3d"] > 0 and k["total_power_kw"] > 0 and k["avg_sor"] > 0
    assert sum(k["status_counts"].values()) == 12
    # red first: the table is sorted by urgency
    order = {"red": 0, "amber": 1, "green": 2}
    ranks = [order[w["status"]] for w in ov["wells"]]
    assert ranks == sorted(ranks)


def test_field_trend_and_alerts(client):
    tr = client.get(f"{API}/field/trend?hours=6").json()
    assert len(tr["ts"]) == len(tr["oil_m3d"]) > 10
    assert "active" in client.get(f"{API}/alerts").json()


def test_security_headers_and_no_store(client):
    r = client.get(f"{API}/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in r.headers["content-security-policy"]


# --------------------------------------------------------------------------- well twin
def test_well_detail_telemetry_and_profiles(client):
    d = client.get(f"{API}/wells/bgw-04").json()  # case-insensitive
    assert d["summary"]["id"] == "BGW-04" and d["cycles"] and d["limits"]["max_allowed"] > 0
    t = client.get(f"{API}/wells/BGW-04/telemetry?hours=24&max_points=200").json()
    assert 10 < len(t["series"]["ts"]) <= 201
    assert set(t["series"]) >= {"spm", "motor_load_pct", "visc_cp", "oil_rate_m3d"}
    p = client.get(f"{API}/wells/BGW-04/profile").json()
    wb, rad = p["wellbore"], p["radial"]
    assert len(wb["depth_m"]) == len(wb["t_flowing_c"]) == len(wb["pressure_kpa"])
    assert len(rad["r_m"]) == len(rad["t_c"]) == len(rad["mu_cp"])
    assert rad["mu_cp"][0] < rad["mu_cp"][-1], "hot near the wellbore = thinner oil"
    v = client.get(f"{API}/wells/BGW-04/viscosity").json()
    assert len(v["t_days"]) == len(v["mu_eff_cp"])


def test_unknown_well_and_bad_params(client):
    assert client.get(f"{API}/wells/NOPE-1").status_code == 404
    assert client.get(f"{API}/wells/BGW-01/telemetry?hours=9999").status_code == 422
    assert client.get(f"{API}/wells/BGW-01/cards?source=hack").status_code == 422


def test_thermal_predict_and_whatif(client):
    p = client.post(f"{API}/thermal/predict", json={"well_id": "BGW-04"}).json()
    assert p["state"]["phase"] in {"PRODUCTION", "SOAK", "INJECTION", "CYCLE_END"}
    assert "physics_only" in p and "ml_correction" in p
    w = client.post(
        f"{API}/thermal/whatif",
        json={"well_id": "BGW-04", "scenarios": [{"steam_m3": 3000}, {"steam_m3": 6000}, {"steam_m3": 9000, "label": "big"}]},
    ).json()
    cum = [s["summary"]["cum_oil_m3"] for s in w["scenarios"]]
    assert cum[0] < cum[1] < cum[2]
    assert w["scenarios"][2]["label"] == "big"
    assert all(len(s["t_days"]) == len(s["q_oil_m3d"]) for s in w["scenarios"])


def test_whatif_validation(client):
    bad = client.post(f"{API}/thermal/whatif", json={"well_id": "BGW-04", "scenarios": [{"steam_m3": 5}]})
    assert bad.status_code == 422
    extra = client.post(f"{API}/thermal/whatif", json={"well_id": "BGW-04", "scenarios": [{"steam_m3": 3000, "evil": 1}]})
    assert extra.status_code == 422
    assert client.post(f"{API}/thermal/whatif", json={"well_id": "BGW-04", "scenarios": []}).status_code == 422


# --------------------------------------------------------------------------- diagnostics
def test_classify_healthy_and_faulty_cards_within_latency_budget(client):
    r = client.post(f"{API}/diagnostics/classify", json=_card_body()).json()
    assert r["label"] == "NORMAL" and r["within_budget"] and r["latency_ms_total"] < 200
    assert len(r["dh_position"]) == 128
    f = client.post(f"{API}/diagnostics/classify", json=_card_body("BGW-07", "FLUID_POUND", fillage=0.4)).json()
    assert f["label"] == "FLUID_POUND" and f["probability"] > 0.5
    assert abs(sum(f["probabilities"].values()) - 1) < 1e-2


def test_classify_accepts_points_and_other_units(client):
    b = _card_body()
    pts = [[p * 39.3701, ld * 224.809] for p, ld in zip(b["position"], b["load"])]  # inches, lbf
    r = client.post(f"{API}/diagnostics/classify", json={"well_id": "BGW-01", "spm": 2.5, "points": pts, "position_unit": "in", "load_unit": "lbf"})
    assert r.status_code == 200 and r.json()["label"] == "NORMAL"


@pytest.mark.parametrize(
    "patch",
    [
        {"spm": 0},
        {"spm": 99},
        {"load": [1.0] * 40},  # length mismatch with position
        {"well_id": "NOPE"},
        {"unexpected": 1},
    ],
)
def test_classify_rejects_bad_input(client, patch):
    body = {**_card_body(), **patch}
    r = client.post(f"{API}/diagnostics/classify", json=body)
    assert r.status_code in (404, 422)


def test_classify_rejects_non_finite_numbers(client):
    raw = '{"well_id": "BGW-01", "spm": 2.5, "position": [' + ",".join(["NaN"] * 64) + '], "load": [' + ",".join(["1.0"] * 64) + "]}"
    r = client.post(f"{API}/diagnostics/classify", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_latest_card_comes_with_baseline_overlay(client):
    d = client.get(f"{API}/wells/BGW-01/card").json()
    assert len(d["card"]["position"]) == 128 and len(d["baseline"]["load"]) == 128
    assert client.get(f"{API}/wells/BGW-02/card").json()["baseline"] is None  # injecting well


# --------------------------------------------------------------------------- optimiser
def test_recommendation_is_safe_and_explained(client):
    for wid in ("BGW-01", "BGW-03", "BGW-06", "BGW-09"):
        r = client.get(f"{API}/wells/{wid}/recommendation").json()
        assert r["applicable"]
        assert r["recommended_spm"] <= r["limits"]["max_allowed"] + 1e-6 or not r["limits"]["feasible"]
        assert r["xai"]["narrative"] and r["xai"]["drivers"]
        assert len(r["plan"]) == 14
    non = client.get(f"{API}/wells/BGW-02/recommendation").json()
    assert non["applicable"] is False


def test_advisory_approval_applies_setpoint_and_clears_a_float_fault(client, container):
    rt = container.runtime
    pend = [a for a in client.get(f"{API}/advisories?status=pending").json() if a["well_id"] == "BGW-03"]
    assert pend, "the optimiser should have drafted an advisory for the floating well"
    adv = pend[0]
    assert adv["recommended_spm"] < adv["current_spm"]
    assert adv["payload"]["narrative"]
    before = client.get(f"{API}/wells/BGW-03").json()["summary"]["diag"]["label"]
    assert before in ("ROD_FLOATING", "PUMP_UNSETTING_RISK")
    out = client.post(f"{API}/advisories/{adv['id']}/approve", json={"actor": "test-operator", "note": "ok"})
    assert out.status_code == 200 and out.json()["status"] == "applied"
    assert rt.setpoints["BGW-03"] == pytest.approx(adv["recommended_spm"])
    for _ in range(3):
        rt.tick()
    after = client.get(f"{API}/wells/BGW-03").json()["summary"]
    assert after["diag"]["label"] != "ROD_FLOATING", "slower pumping must let the rods fall again"
    assert after["status"] != "red"
    # cannot approve twice
    assert client.post(f"{API}/advisories/{adv['id']}/approve").status_code == 409
    audit = client.get(f"{API}/audit").json()
    assert any(a["action"] == "setpoint_applied" and a["actor"] == "test-operator" for a in audit)


def test_reject_leaves_setpoint_unchanged(client, container):
    pend = [a for a in client.get(f"{API}/advisories?status=pending").json() if a["well_id"] != "BGW-03"]
    assert pend
    adv = pend[0]
    before = container.runtime.setpoints[adv["well_id"]]
    out = client.post(f"{API}/advisories/{adv['id']}/reject", json={"actor": "test-operator", "note": "no"})
    assert out.json()["status"] == "rejected"
    assert container.runtime.setpoints[adv["well_id"]] == before
    assert client.post(f"{API}/advisories/999999/approve").status_code == 404


def test_manual_setpoint_is_blocked_by_the_physics_safety_layer(client, container):
    wid = "BGW-01"
    lim = client.get(f"{API}/wells/{wid}").json()["limits"]
    cur = container.runtime.setpoints[wid]
    r = client.post(f"{API}/wells/{wid}/setpoint", json={"spm": lim["max_allowed"] + 2.5})
    assert r.status_code == 409 and "safety" in r.json()["detail"].lower()
    assert container.runtime.setpoints[wid] == cur
    ok = client.post(f"{API}/wells/{wid}/setpoint", json={"spm": min(cur, lim["max_allowed"] * 0.9), "actor": "op"})
    assert ok.status_code == 200
    assert client.post(f"{API}/wells/BGW-02/setpoint", json={"spm": 3}).status_code == 409  # not pumping


def test_cycle_plan_runs_as_a_background_job(client):
    r = client.post(f"{API}/wells/BGW-04/cycle-plan", json={"volumes": [3000, 5000, 7000]})
    assert r.status_code == 202
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done"
    rows = job["result"]["rows"]
    assert [x["steam_m3"] for x in rows] == [3000, 5000, 7000]
    assert job["result"]["best_profit_volume_m3"] in (3000, 5000, 7000)


def _wait_job(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"{API}/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.2)
    raise AssertionError("job timed out")


# --------------------------------------------------------------------------- cycle history editing
def test_cycle_history_edit_recomputes_the_twin(client):
    wid = "BGW-01"
    cycles = client.get(f"{API}/wells/{wid}/cycles").json()
    last = cycles[-1]
    before = client.post(f"{API}/thermal/predict", json={"well_id": wid}).json()["state"]
    body = {k: last[k] for k in ("steam_m3", "quality", "inj_rate_m3d", "inj_pressure_mpa", "soak_days", "prod_days")}
    body["steam_m3"] = last["steam_m3"] * 0.6
    r = client.put(f"{API}/wells/{wid}/cycles/{last['cycle_no']}", json=body)
    assert r.status_code == 200
    after = client.post(f"{API}/thermal/predict", json={"well_id": wid}).json()["state"]
    assert after["steam_total_m3"] == pytest.approx(body["steam_m3"])
    assert after["t_avg_c"] != pytest.approx(before["t_avg_c"])
    # restore
    body["steam_m3"] = last["steam_m3"]
    assert client.put(f"{API}/wells/{wid}/cycles/{last['cycle_no']}", json=body).status_code == 200


def test_cycle_history_validation(client, container):
    wid = "BGW-01"
    last = client.get(f"{API}/wells/{wid}/cycles").json()[-1]
    body = {"steam_m3": 4000, "start_ts": int(container.runtime.sim_ts) + 10 * 86400}
    assert client.post(f"{API}/wells/{wid}/cycles", json=body).status_code == 422  # future start
    body = {"steam_m3": 4000, "start_ts": last["start_ts"] + 3600}
    assert client.post(f"{API}/wells/{wid}/cycles", json=body).status_code == 422  # overlaps the injection
    assert client.put(f"{API}/wells/{wid}/cycles/999", json={"steam_m3": 4000}).status_code == 404
    if len(client.get(f"{API}/wells/{wid}/cycles").json()) > 1:
        first = client.get(f"{API}/wells/{wid}/cycles").json()[0]["cycle_no"]
        assert client.delete(f"{API}/wells/{wid}/cycles/{first}").status_code == 409  # only the newest


# --------------------------------------------------------------------------- ingestion
def test_samples_are_listed_and_downloadable(client):
    names = [s["name"] for s in client.get(f"{API}/samples").json()]
    assert "sample_production_log.csv" in names and "sample_dyno_cards.csv" in names
    assert client.get(f"{API}/samples/sample_production_log.csv").status_code == 200
    assert client.get(f"{API}/samples/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get(f"{API}/samples/not_there.csv").status_code == 404


def test_production_upload_is_cleaned_and_reported(client):
    data = client.get(f"{API}/samples/sample_production_log.csv").content
    r = client.post(f"{API}/ingest/production", files={"file": ("log.csv", data, "text/csv")}, data={"align_to_now": "true"})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["rows_written"] > 300 and set(rep["wells"]) == {"BGW-01", "BGW-04", "BGW-10"}
    assert rep["dropped_duplicates"] == 6
    assert rep["columns"]["motor_kw"]["out_of_range"] == 4
    assert rep["columns"]["whp_kpa"]["out_of_range"] > 0
    assert rep["columns"]["oil_rate_m3d"]["filled"] > 0
    assert any("percent" in w for w in rep["warnings"])
    assert rep["time_shift_seconds"] != 0


def test_upload_guards(client):
    assert client.post(f"{API}/ingest/production", files={"file": ("x.exe", b"abc", "application/octet-stream")}).status_code == 422
    assert client.post(f"{API}/ingest/production", files={"file": ("x.csv", b"", "text/csv")}).status_code == 422
    big = b"a,b\n" + b"1,2\n" * (3 * 1024 * 1024)  # > 10 MB
    assert client.post(f"{API}/ingest/production", files={"file": ("x.csv", big, "text/csv")}).status_code == 413
    no_cols = b"foo,bar\n1,2\n"
    assert client.post(f"{API}/ingest/production", files={"file": ("x.csv", no_cols, "text/csv")}).status_code == 422
    r = client.post(f"{API}/ingest/dynamometer", files={"file": ("c.csv", b"a,b\n1,2\n", "text/csv")}, data={"position_unit": "parsec"})
    assert r.status_code == 422


def test_dynamometer_upload_classifies_all_five_classes(client):
    data = client.get(f"{API}/samples/sample_dyno_cards.csv").content
    r = client.post(f"{API}/ingest/dynamometer", files={"file": ("cards.csv", data, "text/csv")}, data={"position_unit": "m", "load_unit": "kN"})
    assert r.status_code == 202, r.text
    assert r.json()["ingest"]["cards_ok"] == 15
    job = _wait_job(client, r.json()["job_id"], timeout=120)
    assert job["status"] == "done", job
    counts = job["result"]["label_counts"]
    assert job["result"]["cards"] == 15
    assert sum(counts.values()) == 15
    assert len([k for k, v in counts.items() if v]) >= 4, counts
    hist = client.get(f"{API}/wells/BGW-07/cards?source=upload").json()
    assert hist and hist[0]["label"] and len(hist[0]["position"]) == 128


def test_job_status_input_validation(client):
    assert client.get(f"{API}/jobs/nonexistent").status_code == 404
    assert client.get(f"{API}/jobs/..%2F..").status_code == 404


def test_ingest_rate_limit(client):
    for _ in range(30):
        client.post(f"{API}/ingest/production", files={"file": ("x.exe", b"abc", "application/octet-stream")})
    assert client.post(f"{API}/ingest/production", files={"file": ("x.exe", b"abc", "application/octet-stream")}).status_code == 429
    limit_ingest.reset()


# --------------------------------------------------------------------------- websocket
def test_websocket_streams_ticks_and_subscribed_cards(client, container):
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"action": "subscribe", "wells": ["bgw-01", "BOGUS"]})
        assert ws.receive_json() == {"type": "subscribed", "wells": ["BGW-01"]}
        first = ws.receive_json()
        assert first["type"] == "dyno" and first["well_id"] == "BGW-01"
        container.runtime.tick()
        seen = set()
        for _ in range(40):
            m = ws.receive_json()
            seen.add(m["type"])
            if m["type"] == "dyno":
                assert m["well_id"] == "BGW-01", "cards are only sent for subscribed wells"
            if {"tick", "dyno"} <= seen:
                break
        assert {"tick", "dyno"} <= seen
        ws.send_json({"action": "ping"})
        while ws.receive_json()["type"] != "pong":
            pass


def test_websocket_rejects_foreign_origin(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/stream", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()


# --------------------------------------------------------------------------- auth + modes
def test_api_key_protects_mutating_routes_only(models_dir, tmp_path):
    app = create_app(make_settings(tmp_path, models_dir, api_key="s3cret-key"))
    with TestClient(app) as c:
        assert c.get(f"{API}/field/overview").status_code == 200  # reads stay open
        body = {"spm": 1.5}
        assert c.post(f"{API}/wells/BGW-01/setpoint", json=body).status_code == 401
        assert c.post(f"{API}/wells/BGW-01/setpoint", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post(f"{API}/wells/BGW-01/setpoint", json=body, headers={"X-API-Key": "s3cret-key"}).status_code == 200
        assert c.post(f"{API}/advisories/1/approve").status_code == 401
        assert c.post(f"{API}/ingest/production", files={"file": ("x.csv", b"a\n1\n", "text/csv")}).status_code == 401
        assert c.get(f"{API}/system/info").json()["auth_required_for_writes"] is True


def test_closed_loop_mode_applies_safe_advisories_automatically(models_dir, tmp_path):
    app = create_app(make_settings(tmp_path, models_dir, control_mode="closed_loop"))
    with TestClient(app) as c:
        pending = c.get(f"{API}/advisories?status=pending").json()
        applied = c.get(f"{API}/advisories?status=applied").json()
        assert not pending and applied, "in closed-loop mode nothing waits for approval"
        assert all(a["decided_by"] == "closed-loop-policy" for a in applied)
        audit = c.get(f"{API}/audit").json()
        assert any(a["action"] == "setpoint_applied" for a in audit)


def test_data_survives_restart(models_dir, tmp_path):
    s = make_settings(tmp_path, models_dir)
    with TestClient(create_app(s)) as c:
        n1 = len(c.get(f"{API}/wells/BGW-01/telemetry?hours=24&max_points=2000").json()["series"]["ts"])
        cyc1 = c.get(f"{API}/wells/BGW-01/cycles").json()
    with TestClient(create_app(s)) as c:
        cyc2 = c.get(f"{API}/wells/BGW-01/cycles").json()
        n2 = len(c.get(f"{API}/wells/BGW-01/telemetry?hours=24&max_points=2000").json()["series"]["ts"])
    assert cyc1 == cyc2 and n2 >= n1 - 1
