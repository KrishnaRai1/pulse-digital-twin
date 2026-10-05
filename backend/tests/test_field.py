"""Field dataset: physics recovery, leakage-safe features, metrics, models, store and API.

Runs on a real subset of the committed Baghewala dataset (every 8th well)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.field import advisor
from app.field.build import current_build, well_folds
from app.field.case import FieldCase
from app.field.features import cumsum_reset, early_warning_features, failure_labels, group_start_index, rolling_mean
from app.field.metrics import average_precision, roc_auc, warning_metrics
from app.field.raw import load_raw
from app.field.store import FieldStore

API = "/api/v1/dataset"


@pytest.fixture(scope="module")
def raw(field_subset_dir):
    return load_raw(field_subset_dir)


@pytest.fixture(scope="module")
def daily(raw):
    d = raw.prod.merge(raw.srp.drop(columns=["phase"]), on=["well_id", "cycle", "day"])
    return d.sort_values(["well_id", "day"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def case(field_subset_dir):
    return FieldCase.load(field_subset_dir / "generation_config.json")


@pytest.fixture(scope="module")
def store(field_subset_dir, field_store_dir):
    return FieldStore(field_store_dir, field_subset_dir)


# --------------------------------------------------------------------------- physics recovered from the data
def test_rod_floating_index_formula_reproduces_dataset(daily, case):
    run = daily[daily["pump_running"] == 1]
    rfi = case.rod_floating_index(run["spm"], run["oil_viscosity_cp"], run["water_cut_pct"] / 100)
    rel = np.abs(rfi / run["rod_floating_index"] - 1)
    assert np.median(rel) < 0.01
    assert np.corrcoef(rfi, run["rod_floating_index"])[0, 1] > 0.999


def test_pump_capacity_bounds_lifted_liquid(daily, case):
    run = daily[daily["pump_running"] == 1]
    cap = case.pump_capacity_bbl_d(run["spm"])
    assert (run["gross_liquid_bbl_day"] <= cap * 1.0005).all()


def test_viscosity_law_matches_config_table(case):
    tbl = case.raw["params"]["visc_table"]
    for t, mu in zip(tbl["temps_c"], tbl["mu_cp"]):
        if t <= 350:
            assert case.oil_viscosity_cp(t) == pytest.approx(mu, rel=1e-3)


def test_spm_at_rfi_inverts_rfi(case):
    mu, wc = np.array([50.0, 2000.0, 12000.0]), np.array([0.6, 0.3, 0.1])
    spm = case.spm_at_rfi(0.9, mu, wc)
    assert np.allclose(case.rod_floating_index(spm, mu, wc), 0.9)


# --------------------------------------------------------------------------- features & metrics
def test_rolling_mean_matches_pandas():
    rng = np.random.default_rng(0)
    g = np.repeat([0, 1, 2], [5, 9, 4])
    v = rng.normal(size=len(g))
    ours = rolling_mean(v, group_start_index(g), 3)
    ref = pd.Series(v).groupby(g).rolling(3, min_periods=1).mean().to_numpy()
    assert np.allclose(ours, ref)


def test_cumsum_reset():
    v = np.array([1, 2, 3, 4, 5], dtype=float)
    reset = np.array([True, False, True, False, False])
    assert cumsum_reset(v, reset).tolist() == [1, 3, 3, 7, 12]


def test_failure_labels_horizon():
    well = np.array([0] * 20 + [1] * 5)
    day = np.r_[np.arange(20), np.arange(5)]
    y, dtf = failure_labels(well, day, np.array([0]), np.array([15]))
    assert y[:2].sum() == 0 and y[2:16].all() and y[16:20].sum() == 0  # days 2..15 are within 14 days
    assert y[20:].sum() == 0 and np.isnan(dtf[20:]).all()  # another well is never labelled


def test_early_warning_features_do_not_look_ahead(daily):
    w = daily[(daily["well_id"] == daily["well_id"].iloc[0]) & (daily["pump_running"] == 1)]
    cols = {c: w[c].to_numpy(dtype=float) for c in ("load_variability_pct", "pump_fillage_pct", "rod_floating_index", "motor_load_pct", "spm", "polished_rod_peak_load_kn", "polished_rod_min_load_kn")}
    args = (np.zeros(len(w), int), w["day"].to_numpy(), w["cycle"].to_numpy(), w["days_since_soak_start"].to_numpy())
    x1, _ = early_warning_features(*args, cols, np.zeros(len(w), bool))
    k = len(w) // 2
    changed = {c: v.copy() for c, v in cols.items()}
    for v in changed.values():
        v[k + 1 :] *= 3.0  # rewrite the future
    x2, _ = early_warning_features(*args, changed, np.zeros(len(w), bool))
    assert np.allclose(x1[: k + 1], x2[: k + 1], equal_nan=True)


def test_auc_and_average_precision_known_values():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert roc_auc(y, s) == pytest.approx(0.75)
    assert average_precision(y, s) == pytest.approx(0.8333333, rel=1e-6)
    assert roc_auc(y, -s) == pytest.approx(0.25)


def test_warning_metrics_flag_rate():
    rng = np.random.default_rng(1)
    n = 5000
    well, day = np.zeros(n, int), np.arange(n)
    y, _ = failure_labels(well, day, np.array([0, 0]), np.array([1000, 3000]))
    s = rng.random(n) + y * 2.0  # a perfect-ish score
    m = warning_metrics(s, y, well, day, np.array([0, 0]), np.array([1000, 3000]))
    assert m["flag_rate_on_healthy_days"] == pytest.approx(0.02, abs=0.002)
    assert m["failures_caught_with_ge3d_warning"] == 1.0 and m["n_failures"] == 2


def test_folds_split_by_well():
    wells = [f"W{i}" for i in range(23)]
    f = well_folds(wells)
    assert set(f) == set(wells) and set(f.values()) == set(range(5))
    assert f == well_folds(list(reversed(wells)))  # deterministic


# --------------------------------------------------------------------------- built store
def test_store_meta_and_models(field_store_dir):
    meta = json.loads((current_build(field_store_dir) / "meta.json").read_text())
    ew = meta["early_warning"]["results"]
    assert ew["GBM (all features)"]["pr_auc"] > ew["rule: cumulative floating exposure"]["pr_auc"]
    assert ew["GBM (all features)"]["roc_auc"] > 0.85
    nc = meta["nowcast"]["metrics_oil_rate_bbl_d"]
    assert nc["twin: prior + GBM residual"]["mae"] < nc["physics prior only"]["mae"]
    assert meta["cycle_model"]["metrics"]["GBM (out-of-fold)"]["r2"] > 0.4
    bt = meta["spm_advisor"]
    assert bt["advisor"]["days_floating_pct"] < bt["logged"]["days_floating_pct"]


def test_every_running_day_scored_out_of_fold(store):
    run = store.d["pump_running"] == 1
    assert np.isfinite(store.d["risk"][run]).all()
    assert np.isnan(store.d["risk"][~run]).all()


def test_snapshot_consistency(store):
    snap = store.snapshot(900)
    k = snap["kpis"]
    assert k["n_producing"] + k["n_injecting"] + k["n_soaking"] + k["n_workover"] + k["n_stopped"] == store.n_wells
    assert {w["status"] for w in snap["wells"]} <= {"green", "amber", "red", "blue", "grey"}
    oil = sum(w["oil_bbl_d"] for w in snap["wells"])
    assert oil == pytest.approx(k["total_oil_bbl_d"], rel=1e-4)


def test_spm_advice_is_bounded_and_explained(store):
    run = (store.d["pump_running"] == 1) & (store.d["rod_floating_index"] > 1.3)
    i = int(np.nonzero(run)[0][0])
    well, day = store.wells[int(store.d["well_code"][i])], int(store.d["day"][i])
    a = store.spm_advice(well, day)
    assert a["applicable"] and 2.0 <= a["recommended_spm"] <= 9.6
    assert (a["recommended_spm"] / 0.5) == pytest.approx(round(a["recommended_spm"] / 0.5))
    assert a["recommended_spm"] < a["current_spm"] and a["action"] == "reduce"
    assert any(r["code"] == "rod_float" for r in a["reasons"])
    assert a["expected"]["rfi"] <= 0.9 + 1e-9 or a["binding_constraint"] == "float_unavoidable"


def test_decide_vectorised_matches_constraints(case):
    out = advisor.decide(case, [6.0, 3.0, 4.0], [0.4, 0.2, 0.5], [3000.0, 20.0, 200.0], [60.0, 99.0, 30.0], [150.0, 100.0, 40.0])
    rfi = case.rod_floating_index(out["rec"], [3000.0, 20.0, 200.0], [0.4, 0.2, 0.5])
    assert (rfi <= 0.9 + 1e-9).all()
    assert out["rec"][1] == pytest.approx(4.0)  # pump-limited: ramp +1 SPM/day


def test_cycle_planner_respects_budget_and_monotonicity(store):
    well = store.wells[0]
    plan = store.cycle_plan(well, None)["plan"]
    best = plan["best"]
    assert best["steam_per_cycle_day_m3"] <= plan["steam_budget_m3_per_cycle_day"] + 1e-6
    res = store.cycle_whatif(well, None, [{"steam_rate_m3d": r} for r in (150, 200, 250)])["results"]
    opd = [r["opd_m3d"] for r in res]
    assert opd[0] <= opd[1] <= opd[2]  # monotone constraint: more steam never lowers oil


def test_risk_drivers_reproduce_stored_out_of_fold_risk(store):
    run = np.nonzero(store.d["pump_running"] == 1)[0]
    i = int(run[len(run) // 3])
    well, day = store.wells[int(store.d["well_code"][i])], int(store.d["day"][i])
    r = store.risk_drivers(well, day)
    assert r["risk"] == pytest.approx(float(store.d["risk"][i]), rel=1e-4, abs=1e-6)
    assert len(r["drivers"]) > 0


# --------------------------------------------------------------------------- API
def test_dataset_status_and_summary(field_client):
    assert field_client.get(f"{API}/status").json()["state"] == "ready"
    s = field_client.get(f"{API}/summary").json()
    assert s["dataset"]["n_wells"] == 38 and s["totals"]["oil_m3"] > 0


def test_dataset_field_well_and_trend(field_client):
    f = field_client.get(f"{API}/field", params={"day": 1200}).json()
    assert f["day"] == 1200 and len(f["wells"]) == 38
    big = field_client.get(f"{API}/field", params={"day": 99999}).json()
    assert big["day"] == big["max_day"]
    w = f["wells"][0]["well_id"]
    d = field_client.get(f"{API}/wells/{w.lower()}/daily", params={"cycle": 2}).json()
    assert set(d["cycle"]) == {2} and len(d["day"]) == len(d["oil_rate_bbl_day"])
    t = field_client.get(f"{API}/field/trend", params={"step": 7}).json()
    assert len(t["day"]) == len(t["oil_bbl_d"])
    assert field_client.get(f"{API}/wells/WELL-999").status_code == 404


def test_dataset_models_alerts_and_validation(field_client):
    ew = field_client.get(f"{API}/early-warning").json()
    assert "GBM (all features)" in ew["results"] and ew["baseline_report"]["horizon_days"] == 14
    a = field_client.get(f"{API}/early-warning/alerts", params={"day": 1500, "limit": 5}).json()
    assert len(a["wells"]) <= 5
    v = field_client.get(f"{API}/validation").json()
    assert v["status_counts"].get("FAIL", 0) == 0 and len(v["checks"]) > 40
    assert field_client.get(f"{API}/files/generation_config.json").status_code == 200
    assert field_client.get(f"{API}/files/..%2F..%2Fapp%2Fconfig.py").status_code == 404
    assert field_client.get(f"{API}/files/production_history.parquet").status_code == 404


def test_dataset_whatif_validation_and_plan(field_client):
    w = field_client.get(f"{API}/wells").json()[0]["well_id"]
    bad = field_client.post(f"{API}/wells/{w}/cycle-whatif", json={"scenarios": [{"steam_rate_m3d": 900}]})
    assert bad.status_code == 422
    ok = field_client.post(f"{API}/wells/{w}/cycle-whatif", json={"scenarios": [{"label": "more steam", "steam_rate_m3d": 250}]})
    assert ok.status_code == 200 and ok.json()["results"][0]["label"] == "more steam"
    plan = field_client.get(f"{API}/wells/{w}/cycle-plan", params={"objective": "margin"}).json()
    assert plan["plan"]["best"]["economic"] is True
    spm = field_client.get(f"{API}/wells/{w}/spm").json()
    assert "headline" in spm or spm["applicable"] is False


def test_dataset_unavailable_returns_503(client):
    r = client.get(f"{API}/field")
    assert r.status_code == 503 and r.json()["detail"]["state"] == "missing"
