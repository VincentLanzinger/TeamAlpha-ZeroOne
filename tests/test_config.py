"""Smoke tests for src.config — verify the ticker registry is well-formed.

These run without network access, so they're safe in CI / pre-commit.
"""
from src import config


def test_active_ticker_resolves():
    spec = config.active_ticker()
    assert spec.symbol == config.TICKER
    assert spec.csv_path.startswith("data/")


def test_all_tickers_have_valid_metadata():
    # Sybilion's TimeseriesMetadata requires title >= 20 chars and <= 511.
    for sym, spec in config.TICKER_REGISTRY.items():
        assert 20 <= len(spec.metadata_title) <= 511, sym
        assert len(spec.metadata_description) <= 2048, sym
        assert len(spec.keywords) <= 20, sym
        for kw in spec.keywords:
            assert 1 <= len(kw) <= 255, (sym, kw)


def test_horizon_within_supported_range():
    # API supports 1..12 months
    assert 1 <= config.SOFT_HORIZON <= 12


def test_min_observations_matches_horizon_tier():
    # API tiers: 1–3 mo: 40 obs, 4–6 mo: 60, 7–12 mo: 120
    assert config.min_observations() in (40, 60, 120)
    if config.SOFT_HORIZON <= 3:
        assert config.min_observations() == 40
    elif config.SOFT_HORIZON <= 6:
        assert config.min_observations() == 60
    else:
        assert config.min_observations() == 120
