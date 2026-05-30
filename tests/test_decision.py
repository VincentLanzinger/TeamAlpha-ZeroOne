"""Tests for src.decision — formula, tiers, summaries, and a real-artifact run."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import decision
from src.signals import HorizonBand, parse_forecast_bands

REPO_ROOT = Path(__file__).resolve().parents[1]
TTF_CACHE = (
    REPO_ROOT / "cache"
    / "3d08b704b6962bbdeaac04be6b46a9235c4da1920b24f4d32abb27e7001739e4"
)

# ---------- determinism + bounds -------------------------------------------

def test_hedge_is_deterministic():
    a, parts_a = decision.compute_hedge(spot=50, q10=45, q50=48, q90=55)
    b, parts_b = decision.compute_hedge(spot=50, q10=45, q50=48, q90=55)
    assert a == b
    assert parts_a == parts_b


def test_hedge_always_in_unit_interval():
    # Sweep across extreme inputs.
    for spot in (10, 50, 100):
        for q50 in (5, 50, 200):
            for q10, q90 in [(1, 500), (40, 60), (49, 51)]:
                h, _ = decision.compute_hedge(spot, q10, q50, q90)
                assert 0.0 <= h <= 1.0


def test_zero_spot_returns_baseline():
    h, parts = decision.compute_hedge(spot=0, q10=1, q50=1, q90=1)
    assert h == decision.BASELINE


# ---------- direction semantics --------------------------------------------

def test_spot_at_median_with_symmetric_band_is_near_baseline():
    # spot = q50, symmetric band → drift=0, slight insurance from upside tail.
    h, parts = decision.compute_hedge(spot=50, q10=45, q50=50, q90=55)
    assert parts["drift"] == pytest.approx(0.0)
    # q90 = spot * 1.1 → upside_tail = 0.10 → insurance ~ tanh(0.3) * 0.3 = ~0.087
    assert parts["insurance_term"] > 0
    assert decision.BASELINE < h < decision.BASELINE + decision.INSURANCE_WEIGHT


def test_median_below_spot_lowers_hedge():
    # Prices expected to fall (q50 < spot, q90 also <= spot — no insurance).
    h, parts = decision.compute_hedge(spot=50, q10=30, q50=40, q90=48)
    assert parts["drift"] < 0
    assert parts["insurance_term"] == 0.0       # q90 < spot
    assert h < decision.BASELINE


def test_median_above_spot_raises_hedge():
    # Prices expected to rise.
    h, parts = decision.compute_hedge(spot=50, q10=52, q50=58, q90=64)
    assert parts["drift"] > 0
    assert h > decision.BASELINE


def test_wide_upside_tail_increases_hedge_even_when_drift_negative():
    # Median below spot (drift < 0) but very wide right tail (q90 >> spot).
    # Drift pushes hedge down; insurance pushes back up but should not exceed baseline.
    h, parts = decision.compute_hedge(spot=50, q10=20, q50=45, q90=80)
    assert parts["drift"] < 0
    assert parts["insurance_term"] > 0
    # We don't assert h vs baseline — just that insurance contributes meaningfully.
    assert parts["insurance_term"] > 0.10


# ---------- tiers ----------------------------------------------------------

def test_tier_thresholds():
    assert decision.tier_for(0.10) == "ACT"
    assert decision.tier_for(0.24) == "ACT"
    assert decision.tier_for(0.25) == "RECOMMEND"
    assert decision.tier_for(0.40) == "RECOMMEND"
    assert decision.tier_for(0.50) == "ABSTAIN"
    assert decision.tier_for(1.20) == "ABSTAIN"


# ---------- per-month + summary -------------------------------------------

def _band(date, q10, q50, q90):
    return HorizonBand(date=date, q10=q10, q50=q50, q90=q90, point=q50)


def test_decide_builds_one_row_per_band():
    bands = [_band("2026-06-01", 30, 40, 50), _band("2026-07-01", 32, 41, 52)]
    rows = decision.decide(bands, spot=45)
    assert len(rows) == 2
    assert rows[0].date == "2026-06-01"
    assert all(r.spot == 45 for r in rows)


def test_summary_picks_worst_tier_in_window():
    bands = [
        _band("h1", 38, 40, 42),   # narrow → ACT
        _band("h2", 30, 40, 50),   # wider  → RECOMMEND
        _band("h3", 20, 40, 60),   # very wide → ABSTAIN
    ]
    rows = decision.decide(bands, spot=40)
    summary = decision.summarise_next_quarter(rows, quarter_months=3)
    assert summary.weakest_tier == "ABSTAIN"
    assert summary.quarter_months == ("h1", "h2", "h3")


# ---------- integration against the cached TTF artifact -------------------

requires_cache = pytest.mark.skipif(
    not (TTF_CACHE / "forecast.json").exists(),
    reason="cached TTF forecast missing; run hour_one_gate.py first",
)


@requires_cache
def test_real_ttf_decision_rows_and_tiers():
    forecast = json.loads((TTF_CACHE / "forecast.json").read_text(encoding="utf-8"))
    bands = parse_forecast_bands(forecast)
    rows = decision.decide(bands, spot=45.79)

    # 6 horizon months (h+1..h+6).
    assert len(rows) == 6
    # All hedge ratios in [0, 1].
    assert all(0.0 <= r.hedge_ratio <= 1.0 for r in rows)
    # Across the next quarter (h+1..h+3), tiers should be ACT or RECOMMEND
    # (h=3 has the tightest band ~20.5%, h=1 and h=2 sit in RECOMMEND range).
    near_tiers = {r.tier for r in rows[:3]}
    assert near_tiers.issubset({"ACT", "RECOMMEND"})
    # h+6 is very wide (~70%) → ABSTAIN.
    assert rows[5].tier == "ABSTAIN"
    # The model expects prices to fall (median below spot every month),
    # so the hedge should sit BELOW the 50% baseline in the next quarter.
    summary = decision.summarise_next_quarter(rows, quarter_months=3)
    assert summary.avg_hedge_now < summary.avg_baseline
