"""PyTorch 1-D CNN that classifies dynamometer cards (training only).

Inference runs without PyTorch: after training, the weights are exported with batch-norm
folded in (``dyno_cnn.npz``) and served by :mod:`app.engines.srp.cnn_infer`.

Input  : (4, 128) = surface position, surface load, downhole position, downhole load,
         positions normalised to 0..1 and loads divided by the buoyant rod weight.
Output : probabilities over ``CLASSES``.

Circular padding is used because a card is one periodic stroke cycle. The network is
tiny (~30 k parameters), so single-card inference on CPU takes a couple of milliseconds,
well inside the 200 ms budget for continuous state monitoring.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .cards import CLASSES, N_POINTS, card_tensor, generate_card, random_design
from .cnn_infer import META_FILE, MODEL_FILE, NUMPY_FILE, export_state_dict, forward

log = logging.getLogger("pulse.cnn")


class DynoCNN(nn.Module):
    def __init__(self, in_ch: int = 4, n_classes: int = len(CLASSES)):
        super().__init__()

        def block(i, o, k):
            return nn.Sequential(
                nn.Conv1d(i, o, k, padding=k // 2, padding_mode="circular"),
                nn.BatchNorm1d(o),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(in_ch, 24, 7), nn.MaxPool1d(2), block(24, 48, 5), nn.MaxPool1d(2), block(48, 64, 3)
        )
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(h)


# --------------------------------------------------------------------------- data
def _sample(label: str, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    design = random_design(rng)
    spm = float(rng.uniform(2.5, 9.0))
    net_lift = float(rng.uniform(0.6, 0.95) * design.pump_depth_m)
    mu = float(10 ** rng.uniform(np.log10(0.01), np.log10(3.0)))
    kw: dict = {}
    gen_label = label
    if label == "FLUID_POUND":
        kw["fillage"] = float(rng.uniform(0.15, 0.90))
    elif label == "ROD_FLOATING":
        kw["severity"] = float(rng.uniform(0.30, 1.0))
    elif label == "PUMP_UNSETTING_RISK":
        kw["severity"] = float(rng.uniform(0.30, 1.0))
    elif label == "NORMAL" and rng.random() < 0.4:
        # "almost faulty" healthy cards: sub-threshold distortion. A fault is *declared*
        # only above severity 0.30 (float) / below 90 % fill (pound), so the boundary
        # between classes is fuzzy exactly where it is in the field.
        if rng.random() < 0.5:
            gen_label, kw["severity"] = "ROD_FLOATING", float(rng.uniform(0.0, 0.29))
        else:
            gen_label, kw["fillage"] = "FLUID_POUND", float(rng.uniform(0.90, 0.99))
    card = generate_card(
        design, spm, mu, net_lift, gen_label, rng=rng, noise=float(rng.uniform(0.002, 0.025)), **kw
    )
    card.label = label
    x = card_tensor(card, design)
    shift = int(rng.integers(-3, 4))  # timing jitter
    return np.roll(x, shift, axis=1), label


def make_dataset(n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for ci, label in enumerate(CLASSES):
        for _ in range(n_per_class):
            x, _ = _sample(label, rng)
            xs.append(x)
            ys.append(ci)
    x = np.stack(xs)
    y = np.array(ys, dtype=np.int64)
    order = rng.permutation(len(y))
    return x[order], y[order]


# --------------------------------------------------------------------------- training
def train(
    models_dir: Path,
    n_per_class: int = 600,
    epochs: int = 22,
    seed: int = 11,
    extra: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Train the classifier on synthetic archetypes, optionally adding labelled field cards.

    ``extra`` = (x[N,4,128], y[N]) built with ``card_tensor``; field cards are up-weighted x5
    in the training split. Validation metrics are always measured on held-out synthetic cards."""
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    x, y = make_dataset(n_per_class, seed)
    n_val = int(0.2 * len(y))
    xv, yv, xt, yt = x[:n_val], y[:n_val], x[n_val:], y[n_val:]
    if extra is not None and len(extra[1]):
        ex, ey = extra
        xt = np.concatenate([xt] + [ex.astype(np.float32)] * 5)
        yt = np.concatenate([yt] + [ey.astype(np.int64)] * 5)
    xt_t, yt_t = torch.from_numpy(xt), torch.from_numpy(yt)
    model = DynoCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(yt) / 64))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(yt), 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            loss = loss_fn(model(xt_t[idx]), yt_t[idx])
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    with torch.inference_mode():
        pred = model(torch.from_numpy(xv)).argmax(1).numpy()
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(yv, pred):
        conf[t, p] += 1
    acc = float((pred == yv).mean())
    recall = {c: float(conf[i, i] / max(conf[i].sum(), 1)) for i, c in enumerate(CLASSES)}

    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / MODEL_FILE)
    export_state_dict({k: v.detach().numpy() for k, v in model.state_dict().items()}, models_dir / NUMPY_FILE)
    with np.load(models_dir / NUMPY_FILE) as f:
        params = {k: f[k] for k in f.files}
    np_pred = forward(params, xv).argmax(1)
    agree = float((np_pred == pred).mean())
    if agree < 0.995:
        raise RuntimeError(f"NumPy export disagrees with PyTorch on {100 * (1 - agree):.1f}% of validation cards")
    lat = _measure_latency(params, xv[:1])
    meta = {
        "classes": CLASSES,
        "val_accuracy": acc,
        "per_class_recall": recall,
        "confusion_matrix": conf.tolist(),
        "n_train": int(len(yt)),
        "n_val": int(len(yv)),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "latency_ms_p50": lat[0],
        "latency_ms_p95": lat[1],
        "latency_runtime": "numpy (batch-norm folded)",
        "numpy_torch_agreement": agree,
        "data": "synthetic wave-equation cards (see engines/srp/cards.py)" + (f" + {len(extra[1])} labelled field cards" if extra is not None else ""),
    }
    (models_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    log.info("trained dyno CNN: acc=%.3f latency p50=%.2f ms", acc, lat[0])
    return meta


def _measure_latency(params: dict, x1: np.ndarray, reps: int = 100) -> tuple[float, float]:
    times = []
    for _ in range(10):
        forward(params, x1)
    for _ in range(reps):
        t0 = time.perf_counter()
        forward(params, x1)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


__all__ = ["DynoCNN", "train", "make_dataset", "N_POINTS"]
