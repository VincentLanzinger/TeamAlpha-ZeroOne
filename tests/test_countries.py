"""Tests for src.countries — registry sanity + cache-hit invariants."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import countries, sybilion_client as sc

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_country_specs_have_required_fields():
    for code, spec in countries.COUNTRIES.items():
        assert spec.code == code
        assert spec.name
        assert spec.flag
        assert isinstance(spec.region_id, int) and spec.region_id > 0
        assert spec.one_liner
        assert spec.rationale
        assert spec.amplify_tokens, f"{code} has empty amplify_tokens"
        assert spec.forecast_title and 20 <= len(spec.forecast_title) <= 511
        assert spec.forecast_description
        assert spec.forecast_keywords
        assert len(spec.forecast_keywords) <= 20


def test_at_least_one_country_marks_cached_in_repo():
    cached = countries.cached_country_codes()
    assert cached, "no country flagged cached_in_repo — demo won't load instantly"
    assert "EU" in cached, "EU should be the cached-default"


def test_get_country_raises_on_unknown_code():
    with pytest.raises(KeyError):
        countries.get_country("ZZ")


def test_eu_country_body_matches_committed_cache_hash():
    """EU body must hash to the committed cache dir name so the demo loads
    instantly when EU is picked."""
    spec = countries.COUNTRIES["EU"]
    # Build the body the way the cockpit will (same params as the
    # cached hour-one-gate submission).
    from src import config, data
    df = data.load_series(REPO_ROOT / config.TICKER_REGISTRY["TTF"].csv_path)
    body = sc.build_forecast_body(
        timeseries=data.to_api_payload(df),
        title=spec.forecast_title,
        description=spec.forecast_description,
        keywords=spec.forecast_keywords,
        soft_horizon=config.SOFT_HORIZON,
        recency_factor=0.7,
        backtest=True,
        strictly_positive=True,
        regions=[spec.region_id],
        categories=[25, 46],
    )
    key = sc.cache_key(body)
    committed_dir = REPO_ROOT / "cache" / "3d08b704b6962bbdeaac04be6b46a9235c4da1920b24f4d32abb27e7001739e4"
    # Either the hash matches (EU country loads the cached forecast instantly),
    # or the committed cache dir simply doesn't exist (clean checkout).
    if committed_dir.exists():
        # If we built the body to match, the hash should be exactly that dir.
        # If not, the EU spec needs to be re-aligned with the cache — the test
        # surfaces the divergence loudly.
        assert key == committed_dir.name, (
            f"EU country body no longer matches committed cache. "
            f"Expected {committed_dir.name[:12]}..., got {key[:12]}...  "
            f"Either update countries.py EU spec to match the committed body "
            f"in cache/<old_hash>/_request_body.json, or re-cache."
        )


def test_country_amplify_tokens_overlap_with_defaults():
    # Every country should at minimum carry the Tier-1/2 defaults (brent,
    # vix, geopolitical, carbon, etc.) — otherwise a token sweep at the
    # global level would miss country-specific bodies.
    common = {"brent", "carbon", "vix", "geopolitical", "global risk"}
    for code, spec in countries.COUNTRIES.items():
        missing = common - set(spec.amplify_tokens.keys())
        assert not missing, f"{code} missing amplification tokens: {missing}"
