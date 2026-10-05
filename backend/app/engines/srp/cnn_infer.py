"""NumPy inference for the dynamometer-card CNN (no PyTorch needed at run time).

Training happens in :mod:`app.engines.srp.cnn` (PyTorch). After training the weights are
exported to ``dyno_cnn.npz`` with batch-norm folded into the convolutions, and this module
runs the forward pass with plain NumPy: about 1 ms per card on a CPU and roughly 400 MB less
memory than loading PyTorch, which is what lets the API run on a small container.

Architecture (must match ``DynoCNN``): three blocks of circular-padded Conv1d + (folded) BN +
ReLU with max-pooling after the first two, then global mean and max pooling, then a linear head.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cards import CLASSES, Card, card_tensor
from .rods import SrpDesign

log = logging.getLogger("pulse.cnn")
MODEL_FILE = "dyno_cnn.pt"
NUMPY_FILE = "dyno_cnn.npz"
META_FILE = "dyno_cnn_meta.json"
N_BLOCKS = 3
POOL_AFTER = (True, True, False)


def _conv1d_circular(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x (B, Cin, L), w (Cout, Cin, K) -> (B, Cout, L) with circular 'same' padding."""
    k = w.shape[-1]
    p = k // 2
    xp = np.concatenate([x[..., -p:], x, x[..., :p]], axis=-1) if p else x
    win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=-1)  # (B, Cin, L, K)
    return np.einsum("bclk,ock->bol", win, w, optimize=True) + b[None, :, None]


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Logits for a batch of cards x (B, 4, 128)."""
    h = np.asarray(x, dtype=np.float32)
    for i in range(N_BLOCKS):
        h = _conv1d_circular(h, params[f"w{i}"], params[f"b{i}"])
        np.maximum(h, 0.0, out=h)
        if POOL_AFTER[i]:
            bsz, c, length = h.shape
            h = h[..., : length - length % 2].reshape(bsz, c, length // 2, 2).max(axis=-1)
    feat = np.concatenate([h.mean(axis=-1), h.max(axis=-1)], axis=1)
    return feat @ params["head_w"].T + params["head_b"]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def export_state_dict(state: dict, path: Path) -> None:
    """Fold BatchNorm into the convolutions and save NumPy weights (called after training)."""
    out: dict[str, np.ndarray] = {}
    conv_idx = (0, 2, 4)  # positions of the conv blocks inside ``features`` (pools sit at 1 and 3)
    for i, j in enumerate(conv_idx):
        w = state[f"features.{j}.0.weight"].astype(np.float64)
        b = state[f"features.{j}.0.bias"].astype(np.float64)
        gamma = state[f"features.{j}.1.weight"].astype(np.float64)
        beta = state[f"features.{j}.1.bias"].astype(np.float64)
        mean = state[f"features.{j}.1.running_mean"].astype(np.float64)
        var = state[f"features.{j}.1.running_var"].astype(np.float64)
        scale = gamma / np.sqrt(var + 1e-5)
        out[f"w{i}"] = (w * scale[:, None, None]).astype(np.float32)
        out[f"b{i}"] = ((b - mean) * scale + beta).astype(np.float32)
    out["head_w"] = state["head.1.weight"].astype(np.float32)
    out["head_b"] = state["head.1.bias"].astype(np.float32)
    np.savez(path, **out)


def _convert_torch_checkpoint(models_dir: Path) -> bool:
    """One-off upgrade of an older ``dyno_cnn.pt`` to the NumPy format (needs PyTorch)."""
    try:
        import torch  # noqa: PLC0415  (optional dependency)
    except ImportError:
        return False
    state = torch.load(models_dir / MODEL_FILE, map_location="cpu", weights_only=True)
    export_state_dict({k: v.numpy() for k, v in state.items()}, models_dir / NUMPY_FILE)
    log.info("converted %s to %s", MODEL_FILE, NUMPY_FILE)
    return True


@dataclass
class DynoClassifier:
    params: dict[str, np.ndarray]
    meta: dict

    @classmethod
    def load(cls, models_dir: Path) -> DynoClassifier | None:
        models_dir = Path(models_dir)
        npz = models_dir / NUMPY_FILE
        if not npz.exists() and (models_dir / MODEL_FILE).exists():
            _convert_torch_checkpoint(models_dir)
        if not npz.exists():
            return None
        with np.load(npz, allow_pickle=False) as f:  # plain arrays only: loading cannot execute code
            params = {k: f[k] for k in f.files}
        meta_path = models_dir / META_FILE
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta["runtime"] = "numpy"
        return cls(params, meta)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """x: (B, 4, 128) -> (B, n_classes) probabilities."""
        return softmax(forward(self.params, x))

    def classify(self, cards: list[Card], designs: list[SrpDesign]) -> list[dict]:
        t0 = time.perf_counter()
        x = np.stack([card_tensor(c, d) for c, d in zip(cards, designs)])
        probs = self.predict_proba(x)
        ms = (time.perf_counter() - t0) * 1000.0
        out = []
        for p in probs:
            k = int(np.argmax(p))
            out.append(
                {
                    "label": CLASSES[k],
                    "probability": float(p[k]),
                    "probabilities": {c: float(v) for c, v in zip(CLASSES, p)},
                    "latency_ms": ms / len(probs),
                }
            )
        return out


__all__ = ["DynoClassifier", "forward", "softmax", "export_state_dict", "MODEL_FILE", "NUMPY_FILE", "META_FILE"]
