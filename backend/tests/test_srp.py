"""Rod-string wave equation, load model and the CNN classifier."""
import time

import numpy as np
import pytest

from app.domain.wells import build_fleet
from app.engines.srp import cnn, cnn_infer
from app.engines.srp.cards import (
    CLASSES,
    analyze_surface_card,
    baseline_card,
    card_tensor,
    generate_card,
)
from app.engines.srp.loads import FLOAT_MARGIN, float_index, goodman_utilisation, polished_rod_loads
from app.engines.srp.wave import (
    damping_ratio_from_viscosity,
    downhole_to_surface,
    surface_from_pump_load,
    surface_to_downhole,
)

FLEET = {w.id: w for w in build_fleet()}
CFG = FLEET["BGW-01"]
D = CFG.design
NET_LIFT = D.pump_depth_m * CFG.net_lift_fraction


def test_wave_transfer_is_exactly_invertible_without_taper():
    """Every harmonic's 2x2 transfer matrix has determinant 1, so the round trip is exact."""
    card = generate_card(D, 4.0, 0.05, NET_LIFT, "NORMAL", noise=0.0)
    zeta = damping_ratio_from_viscosity(0.05)
    pos_d, load_d = surface_to_downhole(card.position, card.load, 4.0, D, zeta, taper=False)
    pos_s, load_s = downhole_to_surface(pos_d, load_d, 4.0, D, zeta, taper=False)
    assert np.max(np.abs(load_s - card.load)) < 1e-6 * np.ptp(card.load)
    assert np.max(np.abs(pos_s - card.position)) < 1e-6 * np.ptp(card.position)


def test_wave_round_trip_with_high_harmonic_taper_is_close():
    """The taper (noise suppression) only removes the highest harmonics: < 6 % of the load range."""
    card = generate_card(D, 4.0, 0.05, NET_LIFT, "NORMAL", noise=0.0)
    zeta = damping_ratio_from_viscosity(0.05)
    pos_d, load_d = surface_to_downhole(card.position, card.load, 4.0, D, zeta)
    _, load_s = downhole_to_surface(pos_d, load_d, 4.0, D, zeta)
    assert np.max(np.abs(load_s - card.load)) < 0.06 * np.ptp(card.load)


def test_forward_model_is_consistent_with_inverse():
    card = generate_card(D, 4.0, 0.05, NET_LIFT, "NORMAL", noise=0.0)
    zeta = damping_ratio_from_viscosity(0.05)
    load_s, pos_p, load_p = surface_from_pump_load(card.position, card.dh_load, 4.0, D, zeta)
    assert np.max(np.abs(load_s - card.load)) < 0.05 * np.ptp(card.load)


def test_float_index_grows_with_speed_and_viscosity():
    base = float_index(3.0, 0.05, D)
    assert float_index(6.0, 0.05, D) > base
    assert float_index(3.0, 0.5, D) > base
    assert float_index(1.0, 0.01, D) < FLOAT_MARGIN


def test_polished_rod_loads_and_fatigue_are_bounded_when_healthy():
    f_load = D.fluid_load_n(NET_LIFT)
    pprl, mprl = polished_rod_loads(2.0, 0.05, D, f_load)
    assert pprl > mprl > 0
    assert pprl < D.unit_rating_kn * 1000.0
    assert goodman_utilisation(pprl, mprl, D)


@pytest.mark.parametrize("label", CLASSES)
def test_cards_are_valid_for_every_class(label):
    kw = {"fillage": 0.5} if label == "FLUID_POUND" else {"severity": 0.9} if "ROD" in label or "UNSET" in label else {}
    card = generate_card(D, 3.0, 0.1, NET_LIFT, label, rng=np.random.default_rng(1), **kw)
    assert card.position.shape == card.load.shape == (128,)
    assert np.isfinite(card.load).all() and np.isfinite(card.dh_load).all()
    assert card_tensor(card, D).shape == (4, 128)


def test_healthy_baseline_is_deterministic():
    a, b = baseline_card(D, 3.0, 0.1, NET_LIFT), baseline_card(D, 3.0, 0.1, NET_LIFT)
    assert np.array_equal(a.load, b.load)


@pytest.fixture(scope="module")
def classifier(models_dir):
    clf = cnn_infer.DynoClassifier.load(models_dir)
    assert clf is not None
    return clf


def test_cnn_accuracy_on_fresh_synthetic_cards(classifier):
    rng = np.random.default_rng(2024)
    correct = total = 0
    for _ in range(12):
        cfg = FLEET[rng.choice(list(FLEET))]
        for label in CLASSES:
            kw = {"fillage": float(rng.uniform(0.2, 0.8))} if label == "FLUID_POUND" else {"severity": float(rng.uniform(0.5, 1.0))} if "ROD" in label or "UNSET" in label else {}
            card = generate_card(cfg.design, float(rng.uniform(1.5, 5)), float(rng.uniform(0.03, 0.3)), cfg.design.pump_depth_m * 0.78, label, rng=rng, noise=0.008, **kw)
            pred = classifier.classify([card], [cfg.design])[0]["label"]
            correct += pred == label
            total += 1
    assert correct / total >= 0.85, f"accuracy {correct / total:.2f} on {total} fresh cards"


def test_inference_meets_latency_budget(classifier):
    card = generate_card(D, 3.0, 0.1, NET_LIFT, "NORMAL")
    classifier.classify([card], [D])  # warm-up
    t0 = time.perf_counter()
    for _ in range(20):
        # the full pipeline: measured surface card -> wave equation -> CNN
        c = analyze_surface_card(card.position, card.load, 3.0, D, 0.1)
        out = classifier.classify([c], [D])
    per_call_ms = (time.perf_counter() - t0) / 20 * 1000
    assert per_call_ms < 200.0
    assert abs(sum(out[0]["probabilities"].values()) - 1.0) < 1e-3


def test_classifier_reports_class_probabilities(classifier):
    card = generate_card(D, 3.0, 0.1, NET_LIFT, "FLUID_POUND", fillage=0.4)
    res = classifier.classify([card], [D])[0]
    assert set(res["probabilities"]) == set(CLASSES)
    assert res["label"] == max(res["probabilities"], key=res["probabilities"].get)


def test_numpy_inference_matches_pytorch(models_dir):
    """The served NumPy forward pass (batch-norm folded) equals the trained PyTorch network."""
    import torch

    model = cnn.DynoCNN()
    model.load_state_dict(torch.load(models_dir / cnn_infer.MODEL_FILE, map_location="cpu", weights_only=True))
    model.eval()
    x, _ = cnn.make_dataset(20, seed=5)
    with torch.inference_mode():
        ref = torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()
    ours = cnn_infer.DynoClassifier.load(models_dir).predict_proba(x)
    assert np.abs(ref - ours).max() < 1e-4
