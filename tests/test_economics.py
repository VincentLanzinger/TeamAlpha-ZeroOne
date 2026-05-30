"""Tests for src.economics — cost-of-waiting + backtest trust grounding."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import decision, economics
from src.signals import HorizonBand, parse_forecast_bands

REPO_ROOT = Path(__file__).resolve().parents[1]
TTF_CACHE = (
    REPO_ROOT / "cache"
    / "3d08b704b6962bbdeaac04be6b46a9235c4da1920b24f4d32abb27e7001739e4"
)


# ---------- pure economics --------------------------------------------------

def _band(date, q10, q50, q90, quantiles=None):
    return HorizonBand(
        date=date, q10=q10, q50=q50, q90=q90, point=q50,
        quantiles=quantiles or {0.1: q10, 0.5: q50, 0.9: q90},
    )


def test_cost_of_waiting_rule_vs_baseline_ev_matches_formula():
    bands = [_band("h1", 30, 40, 50)]
    rows = decision.decide(bands, spot=45)
    cost = economics.cost_of_waiting(rows, bands)
    c = cost[0]
    h = rows[0].hedge_ratio
    expected_rule_ev = h * 45 + (1 - h) * 40
    expected_base_ev = 0.5 * 45 + 0.5 * 40
    assert c.rule_cost_ev == pytest.approx(expected_rule_ev)
    assert c.baseline_cost_ev == pytest.approx(expected_base_ev)


def test_full_hedge_regret_zero_when_q90_below_spot():
    # If every future scenario is below spot, the wait-strategy regret is 0;
    # the hedge-now strategy bears all regret.
    bands = [_band("h1", 30, 35, 40)]   # all quantiles below spot=50
    rows = decision.decide(bands, spot=50)
    cost = economics.cost_of_waiting(rows, bands)
    c = cost[0]
    # all_wait regret should be exactly 0 (no upside above spot).
    assert c.all_wait_expected_regret == pytest.approx(0.0)
    # all_now regret > 0 (we paid spot but could have waited and paid less).
    assert c.all_now_expected_regret > 0.0


def test_full_wait_regret_zero_when_q10_above_spot():
    # All future scenarios above spot; hedging now is the strictly best move,
    # so all-wait carries all the regret, all-now zero.
    bands = [_band("h1", 60, 70, 80)]
    rows = decision.decide(bands, spot=50)
    cost = economics.cost_of_waiting(rows, bands)
    c = cost[0]
    assert c.all_now_expected_regret == pytest.approx(0.0)
    assert c.all_wait_expected_regret > 0.0


def test_expected_regret_uses_all_quantiles_when_available():
    # Build a band with explicit fine-grained quantiles.
    qs = {round(0.05 * i, 2): 30 + 4 * i for i in range(1, 20)}  # 34..104
    # Override q10/q50/q90 to match the lookup levels.
    band = HorizonBand(
        date="h1", q10=qs[0.1], q50=qs[0.5], q90=qs[0.9],
        point=qs[0.5], quantiles=qs,
    )
    rows = decision.decide([band], spot=70)
    cost = economics.cost_of_waiting(rows, [band])
    # E[regret of all-now]  =  mean over p of max(0, spot - p)
    # E[regret of all-wait] =  mean over p of max(0, p - spot)
    expected_now = sum(max(0, 70 - p) for p in qs.values()) / len(qs)
    expected_wait = sum(max(0, p - 70) for p in qs.values()) / len(qs)
    assert cost[0].all_now_expected_regret == pytest.approx(expected_now)
    assert cost[0].all_wait_expected_regret == pytest.approx(expected_wait)


def test_quarter_summary_avg_metric_correct():
    bands = [
        _band("h1", 30, 40, 50),
        _band("h2", 32, 41, 52),
        _band("h3", 34, 42, 54),
    ]
    rows = decision.decide(bands, spot=45)
    cost = economics.cost_of_waiting(rows, bands)
    s = economics.quarter_summary(cost, months=3)
    expected = sum(r.rule_cost_ev for r in cost) / 3
    assert s["rule_cost_ev"] == pytest.approx(expected)


# ---------- trust grounding -------------------------------------------------

def test_no_backtest_means_full_trust():
    t = economics.compute_trust(None)
    assert t.trust == 1.0
    assert t.threshold_shift == 0.0
    assert t.effective_act_max == decision.ACT_MAX_WIDTH
    assert t.effective_recommend_max == decision.RECOMMEND_MAX_WIDTH


def test_perfect_mape_means_full_trust():
    bt = {"6m": {"metrics": {"MAPE": 0.0, "MASE": 1.0, "RMSE": 0.0}}}
    t = economics.compute_trust(bt)
    assert t.trust_mape == 1.0
    assert t.mase_penalty == 1.0
    assert t.trust == 1.0
    assert t.threshold_shift == 0.0


def test_high_mape_lowers_trust_and_shifts_thresholds():
    bt = {"6m": {"metrics": {"MAPE": 15.0, "MASE": 1.0, "RMSE": 0.0}}}  # 15%
    t = economics.compute_trust(bt)
    # trust_mape = 1 - 0.15 / 0.30 = 0.5
    assert t.trust_mape == pytest.approx(0.5)
    assert t.mase_penalty == 1.0
    assert t.trust == pytest.approx(0.5)
    # shift = (1 - 0.5) * 0.20 = 0.10
    assert t.threshold_shift == pytest.approx(0.10)
    assert t.effective_act_max == pytest.approx(decision.ACT_MAX_WIDTH - 0.10)
    assert t.effective_recommend_max == pytest.approx(decision.RECOMMEND_MAX_WIDTH - 0.10)


def test_high_mase_triggers_red_flag_penalty():
    bt = {"6m": {"metrics": {"MAPE": 10.0, "MASE": 50.0, "RMSE": 0.0}}}
    t = economics.compute_trust(bt)
    assert t.mase_penalty == 0.5
    # trust_mape = 1 - 0.10/0.30 = 2/3 ~ 0.667
    # trust = 0.667 * 0.5 = 0.333
    assert t.trust == pytest.approx(2/3 * 0.5)


def test_mape_zero_trust_caps_at_zero():
    bt = {"6m": {"metrics": {"MAPE": 100.0, "MASE": 1.0, "RMSE": 0.0}}}
    t = economics.compute_trust(bt)
    assert t.trust_mape == 0.0
    assert t.trust == 0.0


def test_apply_grounding_marks_changed_months():
    # Wide band, originally ABSTAIN, stays ABSTAIN even when trust = 1.0
    bands = [
        _band("h1", 38, 40, 42),   # width 0.10 -> ACT
        _band("h2", 32, 40, 48),   # width 0.40 -> RECOMMEND
        _band("h3", 20, 40, 60),   # width 1.00 -> ABSTAIN
    ]
    rows = decision.decide(bands, spot=40)
    # Force a shift big enough to push h1 and h2 to a worse tier.
    bt = {"6m": {"metrics": {"MAPE": 15.0, "MASE": 1.0}}}  # shift = 0.10
    t = economics.compute_trust(bt)
    grounded = economics.apply_grounding(rows, t)
    # h1 had width 0.10; ACT_MAX is now 0.15 → still ACT
    assert grounded[0].grounded_tier == "ACT"
    # h2 had width 0.40; RECOMMEND_MAX is now 0.40 → ABSTAIN (boundary)
    assert grounded[1].grounded_tier == "ABSTAIN"
    assert grounded[1].changed is True
    # h3 was already ABSTAIN.
    assert grounded[2].changed is False


# ---------- integration against the cached TTF artifact -------------------

requires_cache = pytest.mark.skipif(
    not (TTF_CACHE / "forecast.json").exists()
    or not (TTF_CACHE / "backtest_metrics.json").exists(),
    reason="cached TTF artifacts missing",
)


@requires_cache
def test_real_ttf_grounding_widens_abstain():
    forecast = json.loads((TTF_CACHE / "forecast.json").read_text(encoding="utf-8"))
    bands = parse_forecast_bands(forecast)
    rows = decision.decide(bands, spot=45.79)
    bt = json.loads((TTF_CACHE / "backtest_metrics.json").read_text(encoding="utf-8"))
    t = economics.compute_trust(bt)
    # MAPE 14.65% AND MASE > 10 (the API reports ~93) → both signals fire.
    assert t.mase_penalty == 0.5
    assert t.trust < 0.5
    assert t.threshold_shift > 0.05
    grounded = economics.apply_grounding(rows, t)
    # At least one month should change tier due to grounding.
    assert sum(1 for g in grounded if g.changed) >= 1


@requires_cache
def test_real_ttf_quantiles_carry_19_levels():
    forecast = json.loads((TTF_CACHE / "forecast.json").read_text(encoding="utf-8"))
    bands = parse_forecast_bands(forecast)
    assert bands[0].quantiles
    # API publishes 0.05..0.95 step 0.05 = 19 levels
    assert len(bands[0].quantiles) == 19


@requires_cache
def test_real_ttf_cost_of_waiting_runs_clean():
    forecast = json.loads((TTF_CACHE / "forecast.json").read_text(encoding="utf-8"))
    bands = parse_forecast_bands(forecast)
    rows = decision.decide(bands, spot=45.79)
    cost = economics.cost_of_waiting(rows, bands)
    # All regret values non-negative.
    for c in cost:
        for attr in (
            "rule_expected_regret", "baseline_expected_regret",
            "all_now_expected_regret", "all_wait_expected_regret",
        ):
            assert getattr(c, attr) >= 0.0
