"""Tests for src.adaptive — pressures, band shift, end-to-end shock.

All tests are offline (no live API). The end-to-end test uses
`simulated_pressure` to bypass /alerts, plus the cached TTF artifact for
bands + drivers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import adaptive
from src.signals import HorizonBand, parse_drivers, parse_forecast_bands

REPO_ROOT = Path(__file__).resolve().parents[1]
TTF_CACHE = (
    REPO_ROOT / "cache"
    / "3d08b704b6962bbdeaac04be6b46a9235c4da1920b24f4d32abb27e7001739e4"
)


# ---------- band shifting --------------------------------------------------

def _band(date, q10, q50, q90):
    qs = {0.1: q10, 0.5: q50, 0.9: q90}
    return HorizonBand(date=date, q10=q10, q50=q50, q90=q90, point=q50, quantiles=qs)


def test_zero_pressure_does_not_shift():
    bands = [_band("h1", 30, 40, 50)]
    out = adaptive.shift_bands(bands, 0.0, sensitivity=0.7)
    assert out[0].q10 == 30 and out[0].q50 == 40 and out[0].q90 == 50


def test_positive_pressure_shifts_up():
    bands = [_band("h1", 30, 40, 50)]
    # delta=0.20, sens=0.5 -> shift = 0.10 -> *1.10
    out = adaptive.shift_bands(bands, 0.20, sensitivity=0.5)
    assert out[0].q10 == pytest.approx(33.0)
    assert out[0].q50 == pytest.approx(44.0)
    assert out[0].q90 == pytest.approx(55.0)


def test_negative_pressure_shifts_down():
    bands = [_band("h1", 30, 40, 50)]
    out = adaptive.shift_bands(bands, -0.20, sensitivity=0.5)
    assert out[0].q10 == pytest.approx(27.0)
    assert out[0].q50 == pytest.approx(36.0)
    assert out[0].q90 == pytest.approx(45.0)


def test_shift_is_clipped():
    bands = [_band("h1", 30, 40, 50)]
    # delta * sens = 1.0 * 0.7 = 0.70 -> clipped to 0.30
    out = adaptive.shift_bands(bands, 1.0, sensitivity=0.7,
                                clip=(-0.30, 0.30))
    assert out[0].q50 == pytest.approx(40.0 * 1.30)


def test_quantiles_dict_also_shifts():
    qs = {0.05: 25, 0.5: 40, 0.95: 55}
    band = HorizonBand(date="h1", q10=qs[0.05], q50=qs[0.5], q90=qs[0.95],
                       point=qs[0.5], quantiles=qs)
    out = adaptive.shift_bands([band], 0.1, sensitivity=1.0)
    assert out[0].quantiles[0.05] == pytest.approx(25 * 1.1)
    assert out[0].quantiles[0.95] == pytest.approx(55 * 1.1)


# ---------- alert pressure -------------------------------------------------

def test_empty_alerts_returns_zero_pressure():
    out = adaptive.shock_pressure_from_alerts([])
    assert out.pressure == 0.0
    assert out.n_contributors == 0


def test_alert_pressure_signed_and_weighted():
    alerts = [
        {"name": "A", "pct_change": 20.0},  # rank 0, full weight
        {"name": "B", "pct_change": -10.0},
    ]
    # decay=0.85, weights = [1, 0.85], sum=1.85
    # pressure = (0.20*1 + (-0.10)*0.85) / 1.85
    expected = (0.20 + (-0.10) * 0.85) / (1 + 0.85)
    out = adaptive.shock_pressure_from_alerts(alerts, top_n=10, decay=0.85)
    assert out.pressure == pytest.approx(expected)
    assert out.n_contributors == 2


def test_alert_pressure_handles_missing_pct():
    alerts = [{"name": "A", "pct_change": None}, {"name": "B"}]
    out = adaptive.shock_pressure_from_alerts(alerts)
    assert out.pressure == 0.0


# ---------- end-to-end with simulated pressure -----------------------------

requires_cache = pytest.mark.skipif(
    not (TTF_CACHE / "forecast.json").exists()
    or not (TTF_CACHE / "external_signals.json").exists(),
    reason="cached TTF artifacts missing",
)


@requires_cache
def test_simulated_positive_shock_raises_hedge():
    forecast = json.loads((TTF_CACHE / "forecast.json").read_text(encoding="utf-8"))
    signals = json.loads((TTF_CACHE / "external_signals.json").read_text(encoding="utf-8"))
    bands = parse_forecast_bands(forecast)
    cached_drivers = parse_drivers(signals)
    scenario = adaptive.SCENARIOS["hormuz"]
    # +0.30 pressure (large supply shock) -> bands shift ~ +21% (clipped via sens 0.7).
    result = adaptive.run_shock(
        scenario,
        bands=bands,
        spot=45.79,
        cached_drivers=cached_drivers,
        simulated_pressure=+0.30,
    )
    # Shock pressure positive; with baseline ~ 0 or slightly positive,
    # delta should be positive and shift_pct positive.
    assert result.applied_shift_pct > 0.05
    # Quarter avg hedge must rise.
    assert (result.shocked_summary.avg_hedge_now
            > result.baseline_summary.avg_hedge_now)
    # No live alerts were called.
    assert result.alerts_surfaced == []


@requires_cache
def test_simulated_negative_shock_lowers_hedge():
    forecast = json.loads((TTF_CACHE / "forecast.json").read_text(encoding="utf-8"))
    signals = json.loads((TTF_CACHE / "external_signals.json").read_text(encoding="utf-8"))
    bands = parse_forecast_bands(forecast)
    cached_drivers = parse_drivers(signals)
    result = adaptive.run_shock(
        adaptive.SCENARIOS["ukraine"],
        bands=bands,
        spot=45.79,
        cached_drivers=cached_drivers,
        simulated_pressure=-0.25,
    )
    assert result.applied_shift_pct < 0
    assert (result.shocked_summary.avg_hedge_now
            < result.baseline_summary.avg_hedge_now)


@requires_cache
def test_simulated_strong_positive_flips_at_least_one_month_to_buy_more():
    """A big enough shock should push at least one near-term month's hedge ratio
    above the naive 50% baseline (the visible 'flip' for the demo)."""
    forecast = json.loads((TTF_CACHE / "forecast.json").read_text(encoding="utf-8"))
    signals = json.loads((TTF_CACHE / "external_signals.json").read_text(encoding="utf-8"))
    bands = parse_forecast_bands(forecast)
    cached_drivers = parse_drivers(signals)
    result = adaptive.run_shock(
        adaptive.SCENARIOS["hormuz"],
        bands=bands,
        spot=45.79,
        cached_drivers=cached_drivers,
        simulated_pressure=+0.50,  # very strong shock (clipped, but produces +21% shift)
    )
    # Baseline near-quarter hedge is around 41-42%; under strong shock,
    # the quarter average should cross 50% baseline OR at least be much higher.
    assert result.shocked_summary.avg_hedge_now > 0.50


@requires_cache
def test_format_result_renders_all_sections():
    forecast = json.loads((TTF_CACHE / "forecast.json").read_text(encoding="utf-8"))
    signals = json.loads((TTF_CACHE / "external_signals.json").read_text(encoding="utf-8"))
    bands = parse_forecast_bands(forecast)
    cached_drivers = parse_drivers(signals)
    result = adaptive.run_shock(
        adaptive.SCENARIOS["hormuz"],
        bands=bands, spot=45.79,
        cached_drivers=cached_drivers,
        simulated_pressure=+0.20,
    )
    text = adaptive.format_result(result)
    assert "ADAPTIVE SHOCK" in text
    assert "pressures" in text
    assert "per-month diff" in text
    assert "hormuz" in text
