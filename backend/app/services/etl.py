"""ETL: parse, validate and clean production logs and dynamometer cards (CSV / Excel).

Cleaning is deliberately conservative and fully reported back to the caller, so nothing is
silently altered:

1. column names normalised, aliases mapped to the canonical schema
2. timestamps parsed (ISO strings, epoch seconds or milliseconds); bad rows dropped
3. duplicates on (well, timestamp) collapsed (last wins)
4. physical-range check -> out-of-range values become missing
5. robust statistical outlier rejection (Hampel filter: median +/- 5 scaled MAD)
6. short gaps (<= 6 samples) linearly interpolated *inside* the series; longer gaps stay NULL
7. cards: unit conversion, resampling to a fixed 128-point periodic grid, alignment to the
   bottom of the stroke (see ``engines.srp.cards``) so the CNN always sees the same layout
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


class IngestError(ValueError):
    """Raised for malformed uploads; the message is safe to show to the user."""


MAX_COLS = 80
MAX_UNCOMPRESSED_XLSX = 200 * 1024 * 1024
MAX_CARDS_PER_UPLOAD = 500

PRODUCTION_ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp", "time", "datetime", "date", "ts", "date_time"],
    "well_id": ["well_id", "well", "wellname", "well_name", "uwi", "wellid"],
    "spm": ["spm", "strokes_per_min", "strokes_per_minute", "stroke_rate"],
    "stroke_m": ["stroke_m", "stroke_length_m", "stroke_length", "stroke_len"],
    "vfd_hz": ["vfd_hz", "vfd_frequency", "freq_hz", "frequency_hz", "vfd_freq"],
    "motor_load_pct": ["motor_load_pct", "motor_load", "motor_load_percent"],
    "motor_kw": ["motor_kw", "power_kw", "kw", "motor_power_kw"],
    "pprl_kn": ["pprl_kn", "pprl", "peak_load_kn"],
    "mprl_kn": ["mprl_kn", "mprl", "min_load_kn"],
    "whp_kpa": ["whp_kpa", "wellhead_pressure_kpa", "whp", "tubing_pressure_kpa"],
    "tubing_temp_c": ["tubing_temp_c", "wellhead_temp_c", "tubing_temp", "wht_c"],
    "oil_rate_m3d": ["oil_rate_m3d", "oil_m3d", "oil_rate", "qo_m3d", "oil"],
    "liquid_rate_m3d": ["liquid_rate_m3d", "fluid_rate_m3d", "liquid_rate", "total_fluid_m3d"],
    "water_cut": ["water_cut", "wc", "watercut", "bsw"],
    "visc_cp": ["visc_cp", "viscosity_cp", "viscosity"],
    "fillage_pct": ["fillage_pct", "pump_fillage", "fillage"],
}
MEASURES = [k for k in PRODUCTION_ALIASES if k not in ("timestamp", "well_id")]
PHYSICAL_RANGE = {
    "spm": (0, 20),
    "stroke_m": (0, 10),
    "vfd_hz": (0, 80),
    "motor_load_pct": (0, 200),
    "motor_kw": (0, 500),
    "pprl_kn": (0, 500),
    "mprl_kn": (-50, 500),
    "whp_kpa": (0, 20000),
    "tubing_temp_c": (-10, 400),
    "oil_rate_m3d": (0, 2000),
    "liquid_rate_m3d": (0, 5000),
    "water_cut": (0, 1),
    "visc_cp": (0, 1e7),
    "fillage_pct": (0, 100),
}

UNIT_POS_M = {"m": 1.0, "in": 0.0254, "mm": 0.001, "ft": 0.3048}
UNIT_LOAD_N = {"N": 1.0, "kN": 1000.0, "lbf": 4.4482216, "klbf": 4448.2216}


# --------------------------------------------------------------------------- reading
def _check_zip(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist()) > MAX_UNCOMPRESSED_XLSX:
                raise IngestError("Excel file expands to an unreasonable size")
    except zipfile.BadZipFile as exc:
        raise IngestError("not a valid .xlsx file") from exc


def read_table(filename: str, data: bytes, max_rows: int) -> pd.DataFrame:
    ext = Path(filename or "").suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(data), nrows=max_rows + 1, encoding_errors="replace")
            if df.shape[1] == 1 and ";" in str(df.columns[0]):
                df = pd.read_csv(io.BytesIO(data), sep=";", nrows=max_rows + 1, encoding_errors="replace")
        elif ext == ".xlsx":
            _check_zip(data)
            df = pd.read_excel(io.BytesIO(data), engine="openpyxl", nrows=max_rows + 1)
        else:
            raise IngestError("unsupported file type: upload a .csv or .xlsx file")
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"could not parse file: {type(exc).__name__}") from exc
    if df.empty:
        raise IngestError("file contains no rows")
    if df.shape[1] > MAX_COLS:
        raise IngestError(f"too many columns (max {MAX_COLS})")
    if len(df) > max_rows:
        raise IngestError(f"too many rows (max {max_rows:,})")
    df.columns = [re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower().replace("/", "")).strip("_") for c in df.columns]
    return df


def _apply_aliases(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    rename = {}
    for canon, names in aliases.items():
        for n in names:
            if n in df.columns and canon not in rename.values():
                rename[n] = canon
                break
    return df.rename(columns=rename)


# --------------------------------------------------------------------------- helpers
def to_epoch(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    is_num = num.notna()
    n = num[is_num].astype(float)
    out[is_num] = np.where(n > 1e11, n / 1000.0, n)
    rest = s[~is_num]
    if len(rest):
        dt = pd.to_datetime(rest, errors="coerce", utc=True)
        out[~is_num] = (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    return out


def hampel_mask(x: pd.Series, window: int = 11, k: float = 5.0) -> pd.Series:
    med = x.rolling(window, center=True, min_periods=3).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=3).median()
    scale = np.maximum(1.4826 * mad, 0.02 * med.abs() + 1e-9)
    return ((x - med).abs() > k * scale).fillna(False)


# --------------------------------------------------------------------------- production logs
def clean_production(df: pd.DataFrame, known_wells: set[str]) -> tuple[list[dict], dict]:
    df = _apply_aliases(df, PRODUCTION_ALIASES)
    missing = [c for c in ("timestamp", "well_id") if c not in df.columns]
    if missing:
        raise IngestError(f"required column(s) missing: {', '.join(missing)}")
    measures = [c for c in MEASURES if c in df.columns]
    if not measures:
        raise IngestError("no recognised measurement columns (e.g. spm, oil_rate_m3d, visc_cp)")
    report: dict = {"rows_in": int(len(df)), "measures": measures, "warnings": []}
    df = df[["timestamp", "well_id", *measures]].copy()
    df["ts"] = to_epoch(df["timestamp"])
    bad_ts = int(df["ts"].isna().sum())
    df = df.dropna(subset=["ts"])
    df["well_id"] = df["well_id"].astype(str).str.strip().str.upper()
    unknown = df.loc[~df["well_id"].isin(known_wells), "well_id"].unique().tolist()
    df = df[df["well_id"].isin(known_wells)]
    before = len(df)
    df = df.sort_values(["well_id", "ts"]).drop_duplicates(["well_id", "ts"], keep="last")
    report.update(dropped_bad_timestamp=bad_ts, dropped_unknown_well=len(unknown), unknown_wells=unknown[:10], dropped_duplicates=before - len(df))
    if unknown:
        report["warnings"].append(f"ignored unknown well id(s): {', '.join(unknown[:5])}")
    col_stats: dict[str, dict] = {c: {"missing_before": 0, "out_of_range": 0, "statistical_outliers": 0, "filled": 0} for c in measures}
    out_frames = []
    for _well, g in df.groupby("well_id", sort=True):
        g = g.copy()
        for c in measures:
            g[c] = pd.to_numeric(g[c], errors="coerce")
            st = col_stats[c]
            st["missing_before"] += int(g[c].isna().sum())
            if c == "water_cut" and g[c].notna().any() and float(g[c].max()) > 1.5:
                g[c] = g[c] / 100.0
                report["warnings"].append("water_cut looked like percent and was divided by 100")
            lo, hi = PHYSICAL_RANGE[c]
            bad = (g[c] < lo) | (g[c] > hi)
            st["out_of_range"] += int(bad.sum())
            g.loc[bad, c] = np.nan
            if g[c].notna().sum() >= 5:
                outl = hampel_mask(g[c])
                st["statistical_outliers"] += int(outl.sum())
                g.loc[outl, c] = np.nan
            n_nan = int(g[c].isna().sum())
            g[c] = g[c].interpolate(limit=6, limit_area="inside")
            st["filled"] += n_nan - int(g[c].isna().sum())
        out_frames.append(g)
    if not out_frames:
        raise IngestError("no usable rows after validation")
    clean = pd.concat(out_frames)
    for c in measures:
        s = clean[c]
        col_stats[c].update(
            missing_after=int(s.isna().sum()),
            mean=None if s.notna().sum() == 0 else float(s.mean()),
            std=None if s.notna().sum() < 2 else float(s.std()),
            min=None if s.notna().sum() == 0 else float(s.min()),
            max=None if s.notna().sum() == 0 else float(s.max()),
        )
    diffs = clean.groupby("well_id")["ts"].diff().dropna()
    report.update(
        rows_out=int(len(clean)),
        wells=clean.groupby("well_id").size().to_dict(),
        columns=col_stats,
        time_range=[int(clean["ts"].min()), int(clean["ts"].max())],
        median_interval_s=float(diffs.median()) if len(diffs) else None,
    )
    rows = []
    for rec in clean[["well_id", "ts", *measures]].to_dict("records"):
        row = {"well_id": rec["well_id"], "ts": int(rec["ts"])}
        for c in measures:
            v = rec[c]
            row[c] = None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        rows.append(row)
    return rows, report


# --------------------------------------------------------------------------- dynamometer cards
_CARD_ALIASES = {
    "well_id": ["well_id", "well", "wellname", "well_name", "uwi", "wellid"],
    "timestamp": ["timestamp", "time", "datetime", "date", "ts"],
    "card_id": ["card_id", "card", "cardid", "stroke_id"],
    "position": ["position", "pos", "displacement", "polished_rod_position", "x"],
    "load": ["load", "polished_rod_load", "force", "y"],
    "spm": ["spm", "strokes_per_min", "strokes_per_minute"],
}


def _json_array(cell) -> np.ndarray:
    if isinstance(cell, (list, tuple)):
        arr = cell
    else:
        text = str(cell).strip()
        if len(text) > 200_000:
            raise IngestError("array cell too long")
        arr = json.loads(text)
    a = np.asarray(arr, dtype=float)
    if a.ndim != 1:
        raise IngestError("card arrays must be one-dimensional")
    return a


def parse_cards(
    df: pd.DataFrame,
    known_wells: set[str],
    position_unit: str,
    load_unit: str,
    default_spm: Callable[[str], float],
    fallback_ts: int,
) -> tuple[list[dict], dict]:
    df = _apply_aliases(df, _CARD_ALIASES)
    for c in ("well_id", "position", "load"):
        if c not in df.columns:
            raise IngestError(f"required column missing: {c}")
    df["well_id"] = df["well_id"].astype(str).str.strip().str.upper()
    raw: list[tuple[str, int | None, float | None, np.ndarray, np.ndarray, str]] = []
    first = df["position"].dropna().iloc[0] if df["position"].notna().any() else ""
    array_format = isinstance(first, str) and first.strip().startswith("[")
    try:
        if array_format:
            for i, r in enumerate(df.to_dict("records")):
                ts = to_epoch(pd.Series([r.get("timestamp")]))[0] if "timestamp" in df.columns else np.nan
                spm = pd.to_numeric(r.get("spm"), errors="coerce") if "spm" in df.columns else np.nan
                raw.append((r["well_id"], None if np.isnan(ts) else int(ts), None if np.isnan(spm) else float(spm), _json_array(r["position"]), _json_array(r["load"]), f"row {i + 1}"))
        else:
            key = "card_id" if "card_id" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
            if key is None:
                raise IngestError("long-format card files need a card_id (or timestamp) column to separate strokes")
            for (well, cid), g in df.groupby(["well_id", key], sort=False):
                ts = None
                if "timestamp" in g.columns:
                    t0 = to_epoch(g["timestamp"]).iloc[0]
                    ts = None if np.isnan(t0) else int(t0)
                spm = float(pd.to_numeric(g["spm"], errors="coerce").median()) if "spm" in g.columns else np.nan
                raw.append(
                    (well, ts, None if np.isnan(spm) else spm, pd.to_numeric(g["position"], errors="coerce").to_numpy(float), pd.to_numeric(g["load"], errors="coerce").to_numpy(float), f"{well}/{cid}")
                )
    except IngestError:
        raise
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise IngestError(f"could not read card arrays: {type(exc).__name__}") from exc
    if len(raw) > MAX_CARDS_PER_UPLOAD:
        raise IngestError(f"too many cards in one upload (max {MAX_CARDS_PER_UPLOAD})")

    cards, rejected = [], []
    kp, kl = UNIT_POS_M[position_unit], UNIT_LOAD_N[load_unit]
    for idx, (well, ts, spm, pos, load, label) in enumerate(raw):
        if well not in known_wells:
            rejected.append({"card": label, "reason": "unknown well"})
            continue
        ok = np.isfinite(pos) & np.isfinite(load)
        if ok.sum() < 32 or len(pos) > 4096:
            rejected.append({"card": label, "reason": "needs 32-4096 valid samples"})
            continue
        pos, load = pos[ok] * kp, load[ok] * kl
        outl = hampel_mask(pd.Series(load), window=7, k=6.0).to_numpy()
        n_out = int(outl.sum())
        if n_out:
            s = pd.Series(np.where(outl, np.nan, load)).interpolate(limit_direction="both")
            load = s.to_numpy()
        stroke = float(np.ptp(pos))
        if not (0.05 <= stroke <= 12.0):
            rejected.append({"card": label, "reason": f"implausible stroke length {stroke:.2f} m (check the position unit)"})
            continue
        if load.max() > 1e6 or load.min() < -2e5:
            rejected.append({"card": label, "reason": "load outside physical range (check the load unit)"})
            continue
        cards.append(
            {
                "well_id": well,
                "ts": ts if ts is not None else fallback_ts + idx,
                "spm": spm if spm and 0.2 < spm <= 20 else default_spm(well),
                "position": pos,
                "load": load,
                "outliers_fixed": n_out,
            }
        )
    if not cards:
        raise IngestError("no valid cards found: " + (rejected[0]["reason"] if rejected else "empty file"))
    cards.sort(key=lambda c: (c["ts"], c["well_id"]))
    report = {
        "format": "array" if array_format else "long",
        "cards_in": len(raw),
        "cards_ok": len(cards),
        "cards_rejected": len(rejected),
        "rejected": rejected[:20],
        "outliers_fixed": int(sum(c["outliers_fixed"] for c in cards)),
        "wells": sorted({c["well_id"] for c in cards}),
    }
    return cards, report
