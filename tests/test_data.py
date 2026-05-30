"""Offline tests for src.data — loader + validator behavior."""
from __future__ import annotations

import pandas as pd
import pytest

from src import data


def _make_csv(tmp_path, dates, values):
    p = tmp_path / "series.csv"
    df = pd.DataFrame({"date": dates, "value": values})
    df.to_csv(p, index=False)
    return p


def _monthly_dates(n, start="2018-01-01"):
    return [str(d.date()) for d in pd.date_range(start=start, periods=n, freq="MS")]


def test_load_series_basic(tmp_path):
    dates = _monthly_dates(70)
    values = [10.0 + i * 0.1 for i in range(70)]
    p = _make_csv(tmp_path, dates, values)
    df = data.load_series(p)
    assert len(df) == 70
    assert df["date"].iloc[0].day == 1
    assert df["value"].iloc[-1] == pytest.approx(10.0 + 69 * 0.1)


def test_load_snaps_mid_month_to_first_of_month(tmp_path):
    p = _make_csv(tmp_path, ["2024-01-15", "2024-02-20"], [1.0, 2.0])
    df = data.load_series(p)
    assert list(df["date"].dt.day.unique()) == [1]


def test_load_rejects_wrong_columns(tmp_path):
    p = tmp_path / "wrong.csv"
    pd.DataFrame({"d": ["2024-01-01"], "v": [1.0]}).to_csv(p, index=False)
    with pytest.raises(data.SeriesValidationError):
        data.load_series(p)


def test_validate_too_short(tmp_path):
    p = _make_csv(tmp_path, _monthly_dates(30), [1.0] * 30)
    df = data.load_series(p)
    with pytest.raises(data.SeriesValidationError, match="need >="):
        data.validate_series(df, min_obs=60)


def test_validate_detects_gap(tmp_path):
    # 65 months but with a hole at month 30
    dates = _monthly_dates(65)
    values = list(range(65))
    p = _make_csv(tmp_path, dates, values)
    df = data.load_series(p)
    df = df.drop(index=30).reset_index(drop=True)
    with pytest.raises(data.SeriesValidationError, match="gap month"):
        data.validate_series(df, min_obs=60)


def test_validate_passes_on_clean_series(tmp_path):
    dates = _monthly_dates(70, start="2020-01-01")
    p = _make_csv(tmp_path, dates, [10.0 + i for i in range(70)])
    df = data.load_series(p)
    # Spoof "today" so 2025-10 is fresh enough.
    today = pd.Timestamp("2025-12-01")
    data.validate_series(df, min_obs=60, reference_today=today)


def test_validate_rejects_stale_series(tmp_path):
    # Last point in 2020, "today" in 2026 → > 12 months stale.
    dates = _monthly_dates(70, start="2014-01-01")
    p = _make_csv(tmp_path, dates, [10.0] * 70)
    df = data.load_series(p)
    today = pd.Timestamp("2026-01-01")
    with pytest.raises(data.SeriesValidationError, match="months old"):
        data.validate_series(df, min_obs=60, reference_today=today)


def test_to_api_payload_shape(tmp_path):
    p = _make_csv(tmp_path, ["2024-01-01", "2024-02-01"], [12.3, 13.4])
    df = data.load_series(p)
    payload = data.to_api_payload(df)
    assert payload == {"2024-01-01": 12.3, "2024-02-01": 13.4}


def test_stats_summary(tmp_path):
    dates = _monthly_dates(60, start="2020-01-01")
    p = _make_csv(tmp_path, dates, list(range(60)))
    df = data.load_series(p)
    today = pd.Timestamp("2025-06-01")
    s = data.stats(df, reference_today=today)
    assert s.n == 60
    assert s.start == pd.Timestamp("2020-01-01")
    assert s.end == pd.Timestamp("2024-12-01")
    assert s.min == 0.0 and s.max == 59.0
    assert s.months_since_latest == 6
