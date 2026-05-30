"""Tests for src.signals + src.curation.

Uses the real cached TTF artifact (committed at cache/3d08b704.../) as a
fixture so the parser + curation logic are exercised against actual API
output, not a hand-built mock.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import curation
from src.signals import HORIZONS, Driver, parse_drivers, rank_by_horizon

REPO_ROOT = Path(__file__).resolve().parents[1]
TTF_CACHE = (
    REPO_ROOT
    / "cache"
    / "3d08b704b6962bbdeaac04be6b46a9235c4da1920b24f4d32abb27e7001739e4"
)


# ---------- pure unit tests (no fixture) -----------------------------------

def _make_driver(name, imp_per_h, dir_per_h=None, correlation=None):
    return Driver(
        uuid="t",
        name=name,
        category=name.split(" - ", 1)[0].strip() if " - " in name else name,
        scope=name.split(" - ", 1)[1].strip() if " - " in name else "",
        importance_overall=imp_per_h,
        direction_overall=dir_per_h or 0.0,
        importance_by_h={h: imp_per_h for h in HORIZONS},
        direction_by_h=({h: dir_per_h for h in HORIZONS} if dir_per_h is not None else {}),
        correlation=correlation,
    )


def test_curate_drops_population():
    pop = _make_driver("Population - Nigeria", 93.0)
    enr = _make_driver("Energy - Belgium", 98.0)
    out = curation.curate([pop, enr])
    drops = [c for c in out if c.decision == curation.DROPPED]
    kept  = [c for c in out if c.decision == curation.KEPT]
    assert {c.driver.name for c in drops} == {"Population - Nigeria"}
    assert {c.driver.name for c in kept} == {"Energy - Belgium"}


def test_curate_demotes_implausible_scope():
    # Market Indices is whitelisted, but Slovenia is implausible for TTF.
    slo = _make_driver("Market Indices - Slovenia", 50.0)
    out = curation.curate([slo])
    assert len(out) == 1
    c = out[0]
    assert c.decision == curation.DEMOTED
    assert c.adjusted_importance_overall == pytest.approx(50.0 * curation.DEMOTE_FACTOR_DEFAULT)


def test_curate_keeps_world_scope():
    world = _make_driver("Commodities - World", 80.0)
    out = curation.curate([world])
    c = out[0]
    assert c.decision == curation.KEPT
    assert c.adjusted_importance_overall == pytest.approx(80.0)


def test_curated_ranking_promotes_real_signal_over_noise():
    drivers = [
        _make_driver("Population - Nigeria", 93.0),  # raw rank 1 — should drop
        _make_driver("Population - Mali", 86.0),
        _make_driver("Energy - Belgium", 60.0),
        _make_driver("Commodities - World", 50.0),
        _make_driver("Market Indices - Slovenia", 40.0),  # demoted to 12
    ]
    out = curation.curate(drivers)
    kept = curation.kept_only(out)
    assert kept[0].driver.name == "Energy - Belgium"
    assert kept[1].driver.name == "Commodities - World"
    assert kept[2].driver.name == "Market Indices - Slovenia"  # demoted but kept


def test_per_horizon_rank_uses_horizon_value():
    d1 = _make_driver("Energy - Belgium", 100.0)
    # Heterogeneous per-horizon importance.
    d2 = Driver(
        uuid="x", name="Commodities - World", category="Commodities", scope="World",
        importance_overall=200.0 / 6,
        importance_by_h={1: 200.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0},
    )
    decisions = curation.curate([d1, d2])
    h1 = curation.rank_curated_by_horizon(decisions, 1)
    assert h1[0].driver.name == "Commodities - World"  # 200 at h=1 beats Energy's 100
    h6 = curation.rank_curated_by_horizon(decisions, 6)
    assert h6[0].driver.name == "Energy - Belgium"   # 100 at h=6 beats Commodities' 0


def test_report_renders_all_sections():
    drivers = [
        _make_driver("Population - Nigeria", 93.0),
        _make_driver("Energy - Belgium", 98.0),
        _make_driver("Market Indices - Slovenia", 50.0),
    ]
    txt = curation.kept_vs_dropped_report(curation.curate(drivers))
    assert "KEPT" in txt
    assert "DEMOTED" in txt
    assert "DROPPED" in txt
    assert "Population - Nigeria" in txt
    assert "Energy - Belgium" in txt


# ---------- integration against the real cached TTF artifact ---------------

requires_cache = pytest.mark.skipif(
    not (TTF_CACHE / "external_signals.json").exists(),
    reason="cached TTF artifact missing; run scripts/hour_one_gate.py first",
)


@requires_cache
def test_parse_drivers_real_artifact_returns_71():
    artifact = json.loads((TTF_CACHE / "external_signals.json").read_text(encoding="utf-8"))
    drivers = parse_drivers(artifact)
    assert len(drivers) == 71
    # Ranking by overall importance — top should be a credible Energy-region driver.
    top = drivers[0]
    assert top.category in {"Energy", "Population", "Global risk", "Commodities", "Market Indices"}
    # Real artifact validation: at least 20 non-zero importance drivers.
    non_zero = [d for d in drivers if d.importance_overall > 0]
    assert len(non_zero) >= 20


@requires_cache
def test_curation_real_artifact_drops_all_population_drivers():
    artifact = json.loads((TTF_CACHE / "external_signals.json").read_text(encoding="utf-8"))
    drivers = parse_drivers(artifact)
    decisions = curation.curate(drivers)
    kept = curation.kept_only(decisions)
    # No 'Population' category should survive curation.
    assert not any(c.driver.category == "Population" for c in kept)
    # Energy - Belgium should remain in the top 5 after curation.
    top5 = [c.driver.name for c in kept[:5]]
    assert "Energy - Belgium" in top5
