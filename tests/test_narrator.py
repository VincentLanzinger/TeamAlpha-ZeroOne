"""Tests for src.narrator — template fallback, LLM hook, fact integrity."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src import curation, decision, economics, narrator
from src.signals import HorizonBand

REPO_ROOT = Path(__file__).resolve().parents[1]
TTF_CACHE = (
    REPO_ROOT / "cache"
    / "3d08b704b6962bbdeaac04be6b46a9235c4da1920b24f4d32abb27e7001739e4"
)


# ---------- helpers --------------------------------------------------------

def _toy_bundle() -> narrator.DecisionBundle:
    band = HorizonBand(
        date="2026-06-01", q10=30, q50=40, q90=50, point=40,
        quantiles={0.1: 30, 0.5: 40, 0.9: 50},
    )
    rows = decision.decide([band], spot=45)
    cost_rows = economics.cost_of_waiting(rows, [band])
    trust = economics.compute_trust({
        "6m": {"metrics": {"MAPE": 15.0, "MASE": 50.0, "RMSE": 5.0}}
    })
    return narrator.DecisionBundle(
        ticker="TTF",
        display_name="Dutch TTF Natural Gas",
        unit="EUR / MWh",
        spot=45.0,
        bands=[band],
        hedge_rows=rows,
        summary=decision.summarise_next_quarter(rows, quarter_months=1),
        curated=[],
        backtest={"6m": {"metrics": {"MAPE": 15.0, "MASE": 50.0}}},
        trust=trust,
        cost_rows=cost_rows,
        cost_summary=economics.quarter_summary(cost_rows, months=1),
    )


# ---------- facts builder --------------------------------------------------

def test_build_facts_carries_all_key_numbers():
    bundle = _toy_bundle()
    facts = narrator.build_facts(bundle)
    # Spot, hedge%, baseline, drift, tail must appear (formatted from the bundle).
    assert "TTF" in facts and "Dutch TTF Natural Gas" in facts
    assert "45.00" in facts
    assert "MAPE: 15.0%" in facts
    assert "MASE: 50.00" in facts
    assert "trust factor" in facts


def test_build_facts_handles_no_backtest():
    bundle = _toy_bundle()
    no_bt = narrator.DecisionBundle(**{**bundle.__dict__, "trust": None, "backtest": None})
    facts = narrator.build_facts(no_bt)
    assert "(no backtest)" in facts


# ---------- template fallback ---------------------------------------------

def test_template_path_without_key_returns_two_strings(monkeypatch):
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    bundle = _toy_bundle()
    out = narrator.narrate(bundle)
    assert isinstance(out.memo, str) and len(out.memo) > 100
    assert isinstance(out.counter_case, str) and len(out.counter_case) > 50
    assert out.used_llm is False
    assert out.model == "template"


def test_template_memo_mentions_recommendation_and_caveat():
    bundle = _toy_bundle()
    memo = narrator._template_memo(bundle)
    # Must mention the recommendation and the backtest trust caveat.
    assert "hedge" in memo.lower()
    assert "MAPE" in memo
    assert "MASE" in memo
    assert "trust factor" in memo or "trust" in memo.lower()
    # Must not propose alternative numerical recommendations.
    assert "should hedge" not in memo.lower()
    assert "recommend instead" not in memo.lower()


def test_template_counter_case_frames_opposite_call():
    bundle = _toy_bundle()
    counter = narrator._template_counter_case(bundle)
    # The bundle has a "less hedge" recommendation; counter must argue "higher".
    assert "higher" in counter.lower() or "more" in counter.lower()


def test_template_is_deterministic():
    bundle = _toy_bundle()
    a = narrator._template_memo(bundle)
    b = narrator._template_memo(bundle)
    assert a == b


# ---------- LLM hook -------------------------------------------------------

def test_use_llm_without_key_falls_back(monkeypatch):
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    bundle = _toy_bundle()
    out = narrator.narrate(bundle, use_llm=True)
    assert out.used_llm is False
    assert "no FEATHERLESS_API_KEY" in " ".join(out.notes)


def test_use_llm_calls_featherless_when_key_present(monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", "sk-test-key")
    bundle = _toy_bundle()
    calls: list[tuple] = []

    def fake_call(messages, *, model, api_key, temperature, max_tokens, timeout_s=30.0):
        calls.append((model, api_key, messages[0]["role"], messages[1]["role"]))
        if "counter-case" in messages[1]["content"].lower():
            return "FAKE COUNTER CASE."
        return "FAKE MEMO."

    with patch.object(narrator, "_call_featherless", side_effect=fake_call):
        out = narrator.narrate(bundle, use_llm=True)
    assert out.used_llm is True
    assert out.memo == "FAKE MEMO."
    assert out.counter_case == "FAKE COUNTER CASE."
    assert len(calls) == 2


def test_llm_network_failure_falls_back_to_template(monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", "sk-test-key")
    import urllib.error
    bundle = _toy_bundle()

    def boom(*a, **kw):
        raise urllib.error.URLError("simulated network down")

    with patch.object(narrator, "_call_featherless", side_effect=boom):
        out = narrator.narrate(bundle, use_llm=True)
    assert out.used_llm is False
    assert out.model == "template"
    assert any("failed" in n for n in out.notes)


def test_has_llm_reflects_env(monkeypatch):
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    assert narrator.has_llm() is False
    monkeypatch.setenv("FEATHERLESS_API_KEY", "sk-x")
    assert narrator.has_llm() is True
    monkeypatch.setenv("FEATHERLESS_API_KEY", "   ")  # whitespace = no key
    assert narrator.has_llm() is False


# ---------- integration with the real cached TTF ---------------------------

requires_cache = pytest.mark.skipif(
    not (TTF_CACHE / "forecast.json").exists()
    or not (TTF_CACHE / "external_signals.json").exists(),
    reason="cached TTF artifacts missing",
)


@requires_cache
def test_build_bundle_from_real_cache_carries_full_state():
    bundle = narrator.build_bundle_from_cache(TTF_CACHE)
    assert bundle.ticker == "TTF"
    assert bundle.spot == pytest.approx(45.79)
    assert len(bundle.bands) == 6
    assert len(bundle.hedge_rows) == 6
    assert bundle.cost_summary
    assert bundle.trust is not None


@requires_cache
def test_real_bundle_template_memo_mentions_ttf_and_trust():
    bundle = narrator.build_bundle_from_cache(TTF_CACHE)
    memo = narrator._template_memo(bundle)
    assert "TTF" in memo
    assert "MAPE" in memo and "MASE" in memo
    # Memo length sane.
    assert 200 < len(memo) < 2500


@requires_cache
def test_real_bundle_facts_block_includes_all_six_months():
    bundle = narrator.build_bundle_from_cache(TTF_CACHE)
    facts = narrator.build_facts(bundle)
    for r in bundle.hedge_rows:
        assert r.date in facts
