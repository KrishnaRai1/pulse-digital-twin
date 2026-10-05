"""Vectorised, leakage-safe feature engineering for the field dataset.

All functions operate on arrays sorted by group (well or well-cycle) and then by day, and only
look *backwards* in time: a feature on day ``t`` uses rows with day <= t (early warning) or
day <= t - horizon (nowcast). The same functions build the training matrix for the whole field
and the on-demand matrix for one well, so both are identical by construction.
"""
from __future__ import annotations

import numpy as np

EW_SIGNALS = {
    "load_variability_pct": "lv",
    "pump_fillage_pct": "fill",
    "rod_floating_index": "rfi",
    "motor_load_pct": "motor",
    "spm": "spm",
    "polished_rod_peak_load_kn": "peak",
    "polished_rod_min_load_kn": "minl",
}


# --------------------------------------------------------------------------- primitives
def group_start_index(groups: np.ndarray) -> np.ndarray:
    """For rows sorted by ``groups``: index of the first row of each row's group."""
    groups = np.asarray(groups)
    n = len(groups)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    new = np.r_[True, groups[1:] != groups[:-1]]
    starts = np.nonzero(new)[0]
    return np.repeat(starts, np.diff(np.r_[starts, n]))


def rolling_mean(values: np.ndarray, start: np.ndarray, k: int, min_periods: int = 1) -> np.ndarray:
    """Trailing mean over the last ``k`` rows (current row included) within each group."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    cs = np.r_[0.0, np.cumsum(np.where(ok, v, 0.0))]
    cc = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(v))
    lo = np.maximum(i - k + 1, start)
    s = cs[i + 1] - cs[lo]
    c = cc[i + 1] - cc[lo]
    out = np.full(len(v), np.nan)
    m = c >= min_periods
    out[m] = s[m] / c[m]
    return out


def rolling_std(values: np.ndarray, start: np.ndarray, k: int, min_periods: int = 3) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    m1 = rolling_mean(v, start, k, min_periods)
    m2 = rolling_mean(v * v, start, k, min_periods)
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))


def shift_rows(values: np.ndarray, start: np.ndarray, k: int) -> np.ndarray:
    """Value ``k`` rows earlier in the same group (NaN when it does not exist)."""
    v = np.asarray(values, dtype=float)
    i = np.arange(len(v))
    j = i - k
    out = np.full(len(v), np.nan)
    m = j >= start
    out[m] = v[j[m]]
    return out


def cumsum_reset(values: np.ndarray, reset: np.ndarray) -> np.ndarray:
    """Cumulative sum that restarts at rows where ``reset`` is True (the row itself counts)."""
    v = np.nan_to_num(np.asarray(values, dtype=float))
    cs = np.cumsum(v)
    seg = np.cumsum(np.asarray(reset, dtype=bool))  # segment id
    first = np.nonzero(np.r_[True, seg[1:] != seg[:-1]])[0]
    base = np.r_[0.0, cs][first]  # cumsum before the segment start
    seg_index = np.cumsum(np.r_[True, seg[1:] != seg[:-1]]) - 1
    return cs - base[seg_index]


# --------------------------------------------------------------------------- early warning
def early_warning_features(
    well: np.ndarray,
    day: np.ndarray,
    cycle: np.ndarray,
    days_since_soak: np.ndarray,
    signals: dict[str, np.ndarray],
    after_workover: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix for running days (rows sorted by well, day).

    ``signals`` maps the dataset column names in ``EW_SIGNALS`` to arrays; ``after_workover``
    marks the first running day after a workover (damage resets there in the field)."""
    start = group_start_index(well)
    cols: dict[str, np.ndarray] = {}
    s = {short: np.asarray(signals[col], dtype=float) for col, short in EW_SIGNALS.items()}
    s["range"] = s["peak"] - s["minl"]
    for name in ("lv", "fill", "rfi", "motor", "spm", "peak", "minl", "range"):
        cols[name] = s[name]
    for name in ("lv", "fill", "rfi"):
        for k in (3, 7, 14, 30):
            cols[f"{name}_m{k}"] = rolling_mean(s[name], start, k)
        cols[f"{name}_trend7"] = cols[f"{name}_m7"] - shift_rows(cols[f"{name}_m7"], start, 7)
    cols["lv_sd7"] = rolling_std(s["lv"], start, 7)
    cols["motor_m7"] = rolling_mean(s["motor"], start, 7)
    cols["range_m7"] = rolling_mean(s["range"], start, 7)
    cols["range_trend7"] = cols["range_m7"] - shift_rows(cols["range_m7"], start, 7)

    # exposure since the last workover (or since the well's first running day)
    new_well = np.r_[True, well[1:] != well[:-1]]
    reset = new_well | np.asarray(after_workover, dtype=bool)
    cols["cum_float_expo"] = cumsum_reset(np.maximum(s["rfi"] - 1.0, 0.0), reset)
    cols["days_floating"] = cumsum_reset((s["rfi"] > 1.0).astype(float), reset)
    cols["cum_pound_expo"] = cumsum_reset(np.maximum(0.5 - s["fill"] / 100.0, 0.0), reset)
    cols["running_days_since_workover"] = cumsum_reset(np.ones(len(day)), reset)
    # days since the pump last (re)started: a gap in calendar days means it was off
    gap = np.r_[True, np.diff(day) > 1] | new_well
    cols["days_since_restart"] = cumsum_reset(np.ones(len(day)), gap)
    cols["days_since_soak_start"] = np.asarray(days_since_soak, dtype=float)
    cols["cycle"] = np.asarray(cycle, dtype=float)
    names = list(cols)
    return np.column_stack([cols[n] for n in names]).astype(np.float32), names


def failure_labels(well: np.ndarray, day: np.ndarray, fail_well: np.ndarray, fail_day: np.ndarray, horizon: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Label 1 when the same well fails within ``horizon`` days (failure day included).

    Returns (label, days_to_next_failure) for rows sorted by (well, day)."""
    key = well.astype(np.int64) * 100_000 + day.astype(np.int64)
    fkey = np.sort(fail_well.astype(np.int64) * 100_000 + fail_day.astype(np.int64))
    j = np.searchsorted(fkey, key, side="left")
    nxt = np.where(j < len(fkey), fkey[np.minimum(j, len(fkey) - 1)], -1)
    same = (nxt // 100_000) == well.astype(np.int64)
    dtf = np.where(same & (nxt >= 0), nxt - key, np.nan)
    return ((dtf >= 0) & (dtf < horizon)).astype(np.int8), dtf


# --------------------------------------------------------------------------- nowcast
NOWCAST_HORIZON = 7


def log_ratio(obs: np.ndarray, prior: np.ndarray) -> np.ndarray:
    return np.log((np.asarray(obs, dtype=float) + 0.5) / (np.asarray(prior, dtype=float) + 0.5))


def nowcast_features(
    group: np.ndarray,
    day: np.ndarray,
    lr: np.ndarray,
    water_cut: np.ndarray,
    target_static: dict[str, np.ndarray],
    horizon: int = NOWCAST_HORIZON,
) -> tuple[np.ndarray, list[str]]:
    """Features to predict the observed/physics oil ratio on day t from data up to t - horizon.

    Rows: production days sorted by (well-cycle group, day). ``target_static`` holds values
    known in advance for day t (physics prior, cycle controls...)."""
    start = group_start_index(group)
    asof = {
        "lr_m3": rolling_mean(lr, start, 3),
        "lr_m7": rolling_mean(lr, start, 7),
        "lr_m14": rolling_mean(lr, start, 14),
        "lr_ctd": rolling_mean(lr, start, 100_000),
        "wc_m7": rolling_mean(water_cut, start, 7),
        "n_obs": np.arange(len(day)) - start + 1.0,
    }
    key = group.astype(np.int64) * 100_000 + day.astype(np.int64)
    j = np.searchsorted(key, key - horizon, side="right") - 1
    valid = j >= start
    cols: dict[str, np.ndarray] = {}
    for name, arr in asof.items():
        out = np.full(len(day), np.nan)
        out[valid] = arr[j[valid]]
        cols[name] = out
    age = np.full(len(day), np.nan)
    age[valid] = day[valid] - day[j[valid]]
    cols["info_age_days"] = age
    for name, arr in target_static.items():
        cols[name] = np.asarray(arr, dtype=float)
    names = list(cols)
    return np.column_stack([cols[n] for n in names]).astype(np.float32), names


__all__ = [
    "EW_SIGNALS", "NOWCAST_HORIZON", "group_start_index", "rolling_mean", "rolling_std", "shift_rows",
    "cumsum_reset", "early_warning_features", "failure_labels", "log_ratio", "nowcast_features",
]
