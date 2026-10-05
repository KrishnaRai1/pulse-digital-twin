"""ETL: parsing, validation, cleaning and unit handling."""
import io
import json

import numpy as np
import pandas as pd
import pytest

from app.services import etl
from app.services.etl import IngestError, clean_production, parse_cards, read_table

WELLS = {"BGW-01", "BGW-04"}


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def prod_df(n=60):
    ts = pd.date_range("2026-01-01", periods=n, freq="3min", tz="UTC")
    return pd.DataFrame(
        {
            "Timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Well": "bgw-01",
            "SPM": np.full(n, 2.0),
            "Oil Rate (m3/d)": np.linspace(10, 9, n),
            "Water Cut": np.full(n, 40.0),  # percent
            "Visc CP": np.full(n, 80.0),
        }
    )


def test_reads_csv_and_normalises_headers():
    df = read_table("x.csv", csv_bytes(prod_df()), 1000)
    assert "oil_rate_m3d" in df.columns and "timestamp" in df.columns


def test_rejects_unsupported_extension_and_empty_and_too_long():
    with pytest.raises(IngestError):
        read_table("x.exe", b"abc", 10)
    with pytest.raises(IngestError):
        read_table("x.csv", b"a,b\n", 10)
    with pytest.raises(IngestError, match="too many rows"):
        read_table("x.csv", csv_bytes(prod_df(50)), 10)


def test_rejects_corrupt_xlsx():
    with pytest.raises(IngestError):
        read_table("x.xlsx", b"not a zip", 10)


def test_xlsx_round_trip():
    buf = io.BytesIO()
    prod_df().to_excel(buf, index=False)
    df = read_table("x.xlsx", buf.getvalue(), 1000)
    rows, rep = clean_production(df, WELLS)
    assert rep["rows_out"] == 60 and rows[0]["well_id"] == "BGW-01"


def test_cleaning_handles_percent_gaps_outliers_duplicates_and_unknown_wells():
    df = prod_df()
    df.loc[5, "Oil Rate (m3/d)"] = np.nan  # gap -> interpolated
    df.loc[10, "Oil Rate (m3/d)"] = 9999  # out of range -> dropped then interpolated
    df.loc[20, "SPM"] = 15.0  # in range but wildly off -> Hampel outlier
    df = pd.concat([df, df.iloc[[3]], df.iloc[[4]].assign(Well="ZZZ-99")])
    rows, rep = clean_production(read_table("x.csv", csv_bytes(df), 1000), WELLS)
    assert rep["dropped_duplicates"] == 1
    assert rep["unknown_wells"] == ["ZZZ-99"]
    assert rep["columns"]["oil_rate_m3d"]["out_of_range"] == 1
    assert rep["columns"]["spm"]["statistical_outliers"] >= 1
    assert rep["columns"]["oil_rate_m3d"]["missing_after"] == 0
    assert any("percent" in w for w in rep["warnings"])
    assert all(r["water_cut"] == pytest.approx(0.4) for r in rows)
    assert all(r["spm"] < 5 for r in rows)


def test_epoch_seconds_and_milliseconds_are_understood():
    s = pd.Series([1_790_000_000, 1_790_000_000_000, "2026-09-21T08:13:20Z"])
    out = etl.to_epoch(s)
    assert out[0] == out[1] and abs(out[2] - 1_790_000_000) < 86400 * 40


def test_missing_required_columns_is_a_clear_error():
    with pytest.raises(IngestError, match="required column"):
        clean_production(pd.DataFrame({"spm": [1, 2, 3]}), WELLS)


# --------------------------------------------------------------------------- cards
def array_cards(n=2, unit_scale=1.0):
    th = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    pos = 1.4 * (1 - np.cos(th)) / 2
    rows = []
    for k in range(n):
        load = 30000 + 12000 * np.sin(th - 0.5) + 200 * k
        rows.append(
            {
                "well_id": "BGW-01",
                "timestamp": f"2026-01-01T00:0{k}:00Z",
                "spm": 3.0,
                "position": json.dumps((pos / unit_scale).round(4).tolist()),
                "load": json.dumps((load / 1000.0).round(3).tolist()),
            }
        )
    return pd.DataFrame(rows)


def test_parse_array_format_cards_with_units():
    cards, rep = parse_cards(read_table("c.csv", csv_bytes(array_cards()), 1000), WELLS, "m", "kN", lambda w: 4.0, 0)
    assert rep["format"] == "array" and len(cards) == 2
    assert cards[0]["load"].max() > 30000  # kN -> N
    assert cards[0]["spm"] == 3.0


def test_position_unit_conversion_and_implausible_stroke_rejected():
    inches = array_cards(unit_scale=0.0254)  # position expressed in inches
    cards, _ = parse_cards(read_table("c.csv", csv_bytes(inches), 1000), WELLS, "in", "kN", lambda w: 4.0, 0)
    assert np.ptp(cards[0]["position"]) == pytest.approx(1.4, rel=0.01)
    with pytest.raises(IngestError, match="no valid cards"):  # inches read as metres -> 55 m stroke
        parse_cards(read_table("c.csv", csv_bytes(inches), 1000), WELLS, "m", "kN", lambda w: 4.0, 0)


def test_parse_long_format_cards():
    a = array_cards(1).iloc[0]
    pos, load = json.loads(a["position"]), json.loads(a["load"])
    long = pd.DataFrame({"well_id": "BGW-04", "card_id": 1, "position": pos, "load": load, "spm": 3.0})
    cards, rep = parse_cards(read_table("c.csv", csv_bytes(long), 1000), WELLS, "m", "kN", lambda w: 4.0, 100)
    assert rep["format"] == "long" and cards[0]["well_id"] == "BGW-04" and cards[0]["ts"] == 100


def test_unknown_well_cards_are_reported_not_fatal():
    df = pd.concat([array_cards(1), array_cards(1).assign(well_id="NOPE")])
    cards, rep = parse_cards(read_table("c.csv", csv_bytes(df), 1000), WELLS, "m", "kN", lambda w: 4.0, 0)
    assert len(cards) == 1 and rep["cards_rejected"] == 1


def test_malformed_arrays_are_a_clean_error():
    df = array_cards(1)
    df.loc[0, "position"] = "[1,2,"
    with pytest.raises(IngestError):
        parse_cards(read_table("c.csv", csv_bytes(df), 1000), WELLS, "m", "kN", lambda w: 4.0, 0)
