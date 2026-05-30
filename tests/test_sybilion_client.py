"""Tests for src.sybilion_client.

Two layers:
  - Pure / offline tests (always run): cache key stability, body builders.
  - Live smoke tests (skipped if no SYBILION_API_TOKEN): check_account hits /me.
"""
from __future__ import annotations

import os

import pytest

from src import sybilion_client as sc


# ---------- offline tests ----------

def test_cache_key_is_stable_under_key_order():
    a = {"frequency": "monthly", "pipeline_version": "v1", "timeseries": {"2024-01-01": 1.0}}
    b = {"timeseries": {"2024-01-01": 1.0}, "pipeline_version": "v1", "frequency": "monthly"}
    assert sc.cache_key(a) == sc.cache_key(b)


def test_cache_key_changes_when_body_changes():
    base = {"frequency": "monthly", "soft_horizon": 6, "timeseries": {"2024-01-01": 1.0}}
    other = {"frequency": "monthly", "soft_horizon": 3, "timeseries": {"2024-01-01": 1.0}}
    assert sc.cache_key(base) != sc.cache_key(other)


def test_build_forecast_body_minimal_shape():
    body = sc.build_forecast_body(
        timeseries={"2024-01-01": 10.5, "2024-02-01": 11.0},
        title="TTF natural gas monthly front-month settlement price benchmark",
        description="Test description",
        keywords=["TTF", "gas", "energy"],
        soft_horizon=6,
        regions=[3],
        categories=[25, 46],
    )
    assert body["pipeline_version"] == "v1"
    assert body["frequency"] == "monthly"
    assert body["soft_horizon"] == 6
    assert body["timeseries_metadata"]["title"].startswith("TTF natural gas")
    assert body["timeseries_metadata"]["keywords"] == ["TTF", "gas", "energy"]
    assert body["filters"]["regions"] == [3]
    assert body["filters"]["categories"] == [25, 46]
    assert body["timeseries"]["2024-01-01"] == 10.5


def test_build_drivers_body_uses_version_not_pipeline_version():
    body = sc.build_drivers_body(
        title="TTF natural gas monthly front-month settlement price benchmark",
        description=None,
        keywords=["TTF", "gas"],
        regions=[3],
    )
    # RecommendRequestV1 uses `version`, NOT `pipeline_version`.
    assert body["version"] == "v1"
    assert "pipeline_version" not in body
    assert body["timeseries_metadata"]["title"].startswith("TTF natural gas")
    assert body["filters"]["regions"] == [3]


def test_canonical_json_sorts_keys():
    s = sc.canonical_body_json({"b": 2, "a": 1, "nested": {"y": 1, "x": 0}})
    assert s == '{"a":1,"b":2,"nested":{"x":0,"y":1}}'


def test_has_token_returns_bool():
    # Should never raise even with no env set.
    assert isinstance(sc.has_token(), bool)


# ---------- live smoke tests ----------

# Only run when a token is actually configured. The smoke test costs nothing
# (/me and /usage are not billed) but it hits the live API.

requires_token = pytest.mark.skipif(
    not sc.has_token(),
    reason="no SYBILION_API_TOKEN in env; live smoke skipped (offline tests still run)",
)


@requires_token
def test_check_account_live():
    info = sc.check_account()
    assert info.user_id
    assert info.balance_eur_cents >= 0
    assert info.available_eur_cents >= 0
    assert info.api_usage_tier >= 0


@requires_token
def test_list_regions_live_returns_items():
    items = sc.list_regions()
    assert isinstance(items, list)
    # Catalog must be non-empty for filters to be useful at all.
    assert len(items) > 0
    assert all("id" in r and "name" in r for r in items)
