"""Driver-curation layer — the differentiator.

Raw rankings from Sybilion mix economically credible drivers (Energy,
Commodities, Exchange Rates...) with spurious proxies that score high by
coincidence (most famously: 'Population - <random country>' time series that
trend monotonically and latch onto any other trending target).

This module turns raw drivers into a CURATED ranking by:
  1. WHITELIST FILTER — drop entirely any category not in the project
     whitelist. Default whitelist matches the user's brief.
  2. SCOPE PLAUSIBILITY — among whitelisted drivers, demote (multiply
     importance by `demote_factor`) those whose country/region scope is
     implausible for the active ticker (e.g. Market Indices - Slovenia is
     unlikely to drive TTF gas).
  3. Re-rank by the adjusted importance.

Output: a list of CurationDecision objects (kept, demoted, dropped) with
their adjusted per-horizon importance, plus a human-readable report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.signals import HORIZONS, Driver

# Project-wide default. Categories OUTSIDE this set are dropped.
DEFAULT_WHITELIST: frozenset[str] = frozenset({
    "Energy",
    "Commodities",
    "Exchange Rates",
    "Global risk",
    "Equities",
    "Market Indices",
    "Industry",
})

# Per-ticker default plausibility scopes. "World" is always plausible.
# Scopes outside this list inside a whitelisted category are demoted, not
# dropped — the signal is conserved at lower weight.
PLAUSIBLE_SCOPES_TTF: frozenset[str] = frozenset({
    "World", "Europe", "European Union",
    "Germany", "Netherlands", "Belgium", "France", "Italy", "Spain",
    "Norway", "United Kingdom", "Denmark", "Austria", "Poland",
    "United States", "Russia", "Russian Federation",
    "Qatar", "Australia", "China",
})

PLAUSIBLE_SCOPES_ALUMINUM: frozenset[str] = frozenset({
    "World", "China", "United States", "Russia", "Russian Federation",
    "Australia", "India", "Canada", "Norway", "United Arab Emirates",
    "Iceland", "Bahrain", "Germany", "European Union",
})

DEMOTE_FACTOR_DEFAULT: float = 0.3


# Decision tags
KEPT = "kept"
DEMOTED = "demoted"
DROPPED = "dropped"


@dataclass(frozen=True)
class CurationDecision:
    driver: Driver
    decision: str               # "kept" | "demoted" | "dropped"
    reason: str
    adjusted_importance_by_h: dict[int, float]
    adjusted_importance_overall: float

    @property
    def name(self) -> str: return self.driver.name

    @property
    def category(self) -> str: return self.driver.category

    @property
    def scope(self) -> str: return self.driver.scope


def curate(
    drivers: Iterable[Driver],
    *,
    whitelist: Iterable[str] = DEFAULT_WHITELIST,
    plausible_scopes: Iterable[str] = PLAUSIBLE_SCOPES_TTF,
    demote_factor: float = DEMOTE_FACTOR_DEFAULT,
) -> list[CurationDecision]:
    """Apply whitelist + scope plausibility, return decisions sorted by
    adjusted_importance_overall desc.
    """
    wl = frozenset(whitelist)
    sc = frozenset(plausible_scopes)
    out: list[CurationDecision] = []
    for d in drivers:
        if d.category not in wl:
            out.append(
                CurationDecision(
                    driver=d,
                    decision=DROPPED,
                    reason=f"category '{d.category}' not in whitelist",
                    adjusted_importance_by_h={h: 0.0 for h in HORIZONS},
                    adjusted_importance_overall=0.0,
                )
            )
            continue
        if d.scope and d.scope not in sc:
            factor = demote_factor
            decision = DEMOTED
            reason = (
                f"scope '{d.scope}' implausible for active ticker — "
                f"demoted by x{factor}"
            )
        else:
            factor = 1.0
            decision = KEPT
            reason = "in whitelist + plausible scope"
        adj_h = {h: d.importance_by_h.get(h, 0.0) * factor for h in HORIZONS}
        adj_overall = d.importance_overall * factor
        out.append(
            CurationDecision(
                driver=d,
                decision=decision,
                reason=reason,
                adjusted_importance_by_h=adj_h,
                adjusted_importance_overall=adj_overall,
            )
        )
    out.sort(key=lambda x: x.adjusted_importance_overall, reverse=True)
    return out


def kept_only(decisions: Iterable[CurationDecision]) -> list[CurationDecision]:
    return [c for c in decisions if c.decision in (KEPT, DEMOTED)]


def rank_curated_by_horizon(
    decisions: Iterable[CurationDecision], h: int
) -> list[CurationDecision]:
    return sorted(
        (c for c in decisions if c.decision != DROPPED),
        key=lambda c: c.adjusted_importance_by_h.get(h, 0.0),
        reverse=True,
    )


# ---------- report -----------------------------------------------------------

def kept_vs_dropped_report(
    decisions: list[CurationDecision], *, top_n: int = 25
) -> str:
    kept = [c for c in decisions if c.decision == KEPT]
    dem = [c for c in decisions if c.decision == DEMOTED]
    drp = [c for c in decisions if c.decision == DROPPED]
    lines: list[str] = []
    lines.append("=== curation summary ===")
    lines.append(f"total drivers     : {len(decisions)}")
    lines.append(f"kept              : {len(kept)}")
    lines.append(f"demoted           : {len(dem)}")
    lines.append(f"dropped           : {len(drp)}")
    lines.append("")
    # Top KEPT
    lines.append(f"-- top {min(top_n, len(kept))} KEPT (whitelist + plausible scope) --")
    lines.append(f"{'#':>3}  {'adj_imp':>8}  {'raw_imp':>8}  {'dir':>4}  {'corr':>7}  category            name")
    for i, c in enumerate(kept[:top_n], start=1):
        corr = f"{c.driver.correlation:+.3f}" if c.driver.correlation is not None else "   ?   "
        lines.append(
            f"{i:>3}  {c.adjusted_importance_overall:>8.2f}  "
            f"{c.driver.importance_overall:>8.2f}  "
            f"{c.driver.direction_sign():>4}  {corr:>7}  "
            f"{c.driver.category:<18}  {c.driver.name}"
        )
    lines.append("")
    # Top DEMOTED
    if dem:
        lines.append(f"-- top {min(10, len(dem))} DEMOTED (whitelist but implausible scope) --")
        lines.append(f"{'#':>3}  {'adj_imp':>8}  {'raw_imp':>8}  {'dir':>4}  scope")
        for i, c in enumerate(dem[:10], start=1):
            lines.append(
                f"{i:>3}  {c.adjusted_importance_overall:>8.2f}  "
                f"{c.driver.importance_overall:>8.2f}  "
                f"{c.driver.direction_sign():>4}  {c.driver.category} - {c.driver.scope}"
            )
        lines.append("")
    # Top DROPPED (highest raw importance — these are the most-spurious noise)
    if drp:
        drp_by_raw = sorted(drp, key=lambda c: c.driver.importance_overall, reverse=True)
        lines.append(f"-- top {min(10, len(drp_by_raw))} DROPPED (off-whitelist) -- the noise we cut --")
        lines.append(f"{'#':>3}  {'raw_imp':>8}  {'dir':>4}  category            name")
        for i, c in enumerate(drp_by_raw[:10], start=1):
            lines.append(
                f"{i:>3}  {c.driver.importance_overall:>8.2f}  "
                f"{c.driver.direction_sign():>4}  "
                f"{c.driver.category:<18}  {c.driver.name}"
            )
    return "\n".join(lines)


# ---------- CLI --------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    """python -m src.curation [cache_dir]

    Loads external_signals.json from the cached gate run (or a given dir)
    and prints the curated-vs-raw side-by-side.
    """
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description=_cli.__doc__.splitlines()[0])
    p.add_argument(
        "cache_dir", nargs="?", default=None,
        help="path to a cache/<hash>/ dir (default: newest in cache/)",
    )
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    if args.cache_dir:
        cdir = Path(args.cache_dir)
    else:
        cands = [d for d in (repo_root / "cache").iterdir() if d.is_dir() and (d / "external_signals.json").exists()]
        if not cands:
            print("No cached external_signals.json found under cache/.")
            return 2
        cdir = max(cands, key=lambda d: d.stat().st_mtime)

    print(f"--- loading {cdir} ---")
    artifact = json.loads((cdir / "external_signals.json").read_text(encoding="utf-8"))
    from src.signals import parse_drivers
    drivers = parse_drivers(artifact)
    decisions = curate(drivers)

    # Side-by-side: raw top N vs curated top N.
    print()
    print(f"-- raw top {args.top} --")
    print(f"{'#':>3}  {'raw_imp':>8}  category            name")
    for i, d in enumerate(drivers[: args.top], start=1):
        print(f"{i:>3}  {d.importance_overall:>8.2f}  {d.category:<18}  {d.name}")
    print()
    kept = kept_only(decisions)
    print(f"-- curated top {args.top} (after whitelist + scope demotion) --")
    print(f"{'#':>3}  {'adj_imp':>8}  {'raw_imp':>8}  {'tag':>8}  category            name")
    for i, c in enumerate(kept[: args.top], start=1):
        print(
            f"{i:>3}  {c.adjusted_importance_overall:>8.2f}  "
            f"{c.driver.importance_overall:>8.2f}  {c.decision:>8}  "
            f"{c.driver.category:<18}  {c.driver.name}"
        )
    print()
    print(kept_vs_dropped_report(decisions, top_n=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
