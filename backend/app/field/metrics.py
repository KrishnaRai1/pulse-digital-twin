"""Model-evaluation metrics (NumPy only, no scikit-learn dependency)."""
from __future__ import annotations

import numpy as np


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    # average ranks for ties
    _, first, counts = np.unique(sorted_s, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """PR-AUC as average precision (step-wise, same definition as scikit-learn)."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(s, dtype=float), kind="mergesort")
    s_sorted = np.asarray(s, dtype=float)[order]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    # evaluate only at the last index of each distinct score
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tp, fp = tp[last], fp[last]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def pr_curve(y: np.ndarray, s: np.ndarray, n_points: int = 60) -> list[dict]:
    """Down-sampled precision-recall curve for plotting."""
    y = np.asarray(y).astype(bool)
    order = np.argsort(-np.asarray(s, dtype=float), kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    n_pos = max(int(y.sum()), 1)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    idx = np.unique(np.searchsorted(recall, np.linspace(0.0, 1.0, n_points), side="left").clip(0, len(y) - 1))
    return [{"recall": round(float(recall[i]), 4), "precision": round(float(precision[i]), 4)} for i in idx]


def threshold_at_flag_rate(scores_healthy: np.ndarray, flag_rate: float) -> float:
    """Score threshold that flags ``flag_rate`` of healthy days."""
    return float(np.quantile(np.asarray(scores_healthy, dtype=float), 1.0 - flag_rate))


def warning_metrics(
    score: np.ndarray,
    label: np.ndarray,
    well: np.ndarray,
    day: np.ndarray,
    fail_well: np.ndarray,
    fail_day: np.ndarray,
    flag_rate: float = 0.02,
    horizon: int = 14,
    min_lead: int = 3,
) -> dict:
    """Event-level early-warning metrics, defined like the dataset's baseline report.

    * threshold flags ``flag_rate`` of healthy days (label 0),
    * a failure is *caught* when the score crosses the threshold on a running day 3 to
      ``horizon`` days before it, and the lead time is measured from the first such flag,
    * false-alert days are flagged healthy days per running well-year.
    """
    score = np.asarray(score, dtype=float)
    label = np.asarray(label).astype(bool)
    thr = threshold_at_flag_rate(score[~label], flag_rate)
    flagged = score >= thr
    caught, leads = 0, []
    n_events = 0
    # rows are sorted by (well, day): locate each event's look-back window by binary search
    key = well.astype(np.int64) * 100_000 + day.astype(np.int64)
    for w, d in zip(fail_well, fail_day):
        lo = np.searchsorted(key, int(w) * 100_000 + int(d) - horizon, side="left")
        hi = np.searchsorted(key, int(w) * 100_000 + int(d), side="right")
        if hi <= lo:
            continue
        n_events += 1
        days = day[lo:hi]
        f = flagged[lo:hi]
        early = f & ((d - days) >= min_lead)
        if early.any():
            caught += 1
            leads.append(int(d - days[f].min()))
    running_well_years = len(score) / 365.0
    return {
        "roc_auc": roc_auc(label, score),
        "pr_auc": average_precision(label, score),
        "threshold": thr,
        "flag_rate_on_healthy_days": float(flagged[~label].mean()),
        "failures_caught_with_ge3d_warning": caught / max(n_events, 1),
        "median_lead_days": float(np.median(leads)) if leads else 0.0,
        "n_failures": int(n_events),
        "false_alert_days_per_running_well_year": float((flagged & ~label).sum() / max(running_well_years, 1e-9)),
        "positive_rate": float(label.mean()),
    }


def regression_metrics(y: np.ndarray, p: np.ndarray, floor: float = 1.0) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    err = p - y
    big = np.abs(y) >= floor
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "mape_pct": float(np.mean(np.abs(err[big]) / np.abs(y[big])) * 100.0) if big.any() else float("nan"),
        "r2": float(1.0 - np.sum(err**2) / max(np.sum((y - y.mean()) ** 2), 1e-12)),
        "n": int(len(y)),
    }


__all__ = ["roc_auc", "average_precision", "pr_curve", "threshold_at_flag_rate", "warning_metrics", "regression_metrics"]
