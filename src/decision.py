"""Decision engine — turn (q10, q50, q90, spot) into a hedge ratio.

The hedge_ratio is the fraction of next-quarter allowances we should forward-
buy *now* (vs wait). The complement is the wait fraction.

Decision components
-------------------

1) DRIFT term (EV-flavored)
   drift = (q50 - spot) / spot
     > 0 : median is above spot => prices expected to rise => hedge MORE now
     < 0 : median is below spot => prices expected to fall => hedge LESS now
   contribution = tanh(drift * drift_scale) * drift_weight
     bounded in [-drift_weight, +drift_weight]

2) UPSIDE-TAIL INSURANCE (Kelly-flavored)
   upside_tail = max(0, (q90 - spot) / spot)
     If q90 sits above spot there is a real bad-case scenario; we pay an
     insurance premium proportional to its magnitude (tanh-saturated).
   contribution = tanh(upside_tail * insurance_scale) * insurance_weight
     bounded in [0, +insurance_weight]

3) BASELINE
   = 0.50 (matches the naive "always hedge 50%" rule we compare against)

   hedge_ratio = clip(baseline + drift_term + insurance, 0, 1)

Confidence tiers (by band_width = (q90 - q10) / q50)
----------------------------------------------------
   ACT       : band_width < 0.25  (tight band - high confidence)
   RECOMMEND : 0.25 <= band_width < 0.50
   ABSTAIN   : band_width >= 0.50  (very wide - too uncertain to commit)

The decision is fully deterministic: no rng, no clock. The same forecast.json
+ spot always produces the same recommendation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.signals import HorizonBand

# -- tier thresholds ---------------------------------------------------------
ACT_MAX_WIDTH: float = 0.25
RECOMMEND_MAX_WIDTH: float = 0.50

# -- formula coefficients ----------------------------------------------------
BASELINE: float = 0.50
DRIFT_SCALE: float = 4.0       # tanh saturation point ~ 25% drift
DRIFT_WEIGHT: float = 0.30     # max drift contribution
INSURANCE_SCALE: float = 3.0   # tanh saturation point ~ 30% upside tail
INSURANCE_WEIGHT: float = 0.30  # max insurance premium


@dataclass(frozen=True)
class HedgeRow:
    """Per-horizon-month decision."""
    date: str
    spot: float
    q10: float
    q50: float
    q90: float
    band_width: float        # (q90 - q10) / q50
    drift_pct: float         # (q50 - spot) / spot
    upside_tail_pct: float   # max(0, (q90 - spot) / spot)
    drift_term: float        # contribution to hedge_ratio
    insurance_term: float    # contribution to hedge_ratio
    hedge_ratio: float       # final, clipped to [0, 1]
    baseline_ratio: float    # naive 50%
    tier: str                # ACT | RECOMMEND | ABSTAIN
    rationale: str

    @property
    def delta_vs_baseline(self) -> float:
        return self.hedge_ratio - self.baseline_ratio


@dataclass(frozen=True)
class HedgeSummary:
    """Roll-up across the actionable horizon (next quarter by default)."""
    quarter_months: tuple[str, ...]
    avg_hedge_now: float
    avg_baseline: float
    weakest_tier: str        # worst tier in the window — gates the recommendation

    def text(self) -> str:
        lines = [
            f"buy now : {self.avg_hedge_now * 100:>5.1f}%  "
            f"(vs naive {self.avg_baseline * 100:.0f}%)",
            f"wait    : {(1 - self.avg_hedge_now) * 100:>5.1f}%  "
            f"(vs naive {(1 - self.avg_baseline) * 100:.0f}%)",
            f"window  : {', '.join(self.quarter_months)}",
            f"tier    : {self.weakest_tier}  (worst tier in window)",
        ]
        return "\n".join(lines)


# -- formula -----------------------------------------------------------------

def tier_for(band_width: float) -> str:
    if band_width < ACT_MAX_WIDTH:
        return "ACT"
    if band_width < RECOMMEND_MAX_WIDTH:
        return "RECOMMEND"
    return "ABSTAIN"


def compute_hedge(
    spot: float,
    q10: float,
    q50: float,
    q90: float,
    *,
    baseline: float = BASELINE,
    drift_scale: float = DRIFT_SCALE,
    drift_weight: float = DRIFT_WEIGHT,
    insurance_scale: float = INSURANCE_SCALE,
    insurance_weight: float = INSURANCE_WEIGHT,
) -> tuple[float, dict[str, float]]:
    """Return (hedge_ratio, parts_dict).

    `parts_dict` exposes each component so the reasoning surface in Phase 5
    can show the decomposition.
    """
    if spot <= 0 or q50 <= 0:
        # Degenerate input — return baseline rather than blow up.
        return baseline, {
            "drift": 0.0, "drift_term": 0.0,
            "upside_tail": 0.0, "insurance_term": 0.0,
            "band_width": 0.0, "baseline": baseline,
            "pre_clip": baseline,
        }
    drift = (q50 - spot) / spot
    upside_tail = max(0.0, (q90 - spot) / spot)
    band_width = max(0.0, (q90 - q10) / q50)

    drift_term = math.tanh(drift * drift_scale) * drift_weight
    insurance_term = math.tanh(upside_tail * insurance_scale) * insurance_weight

    pre_clip = baseline + drift_term + insurance_term
    hedge = max(0.0, min(1.0, pre_clip))
    return hedge, {
        "drift": drift,
        "drift_term": drift_term,
        "upside_tail": upside_tail,
        "insurance_term": insurance_term,
        "band_width": band_width,
        "baseline": baseline,
        "pre_clip": pre_clip,
    }


def _rationale(parts: dict[str, float], tier: str) -> str:
    drift = parts["drift"]
    upside = parts["upside_tail"]
    bits: list[str] = []
    if drift < -0.02:
        bits.append(f"median {drift*100:+.1f}% vs spot (decline)")
    elif drift > 0.02:
        bits.append(f"median {drift*100:+.1f}% vs spot (rise)")
    else:
        bits.append("median ~ spot")
    if upside > 0.01:
        bits.append(f"upside tail q90 = +{upside*100:.1f}% above spot")
    else:
        bits.append("q90 <= spot (no upside tail)")
    bits.append(f"{tier}")
    return "; ".join(bits)


def decide(
    bands: list[HorizonBand],
    spot: float,
    *,
    baseline: float = BASELINE,
) -> list[HedgeRow]:
    """Decide per-horizon-month given the forecast bands + current spot."""
    rows: list[HedgeRow] = []
    for b in bands:
        hedge, parts = compute_hedge(spot, b.q10, b.q50, b.q90, baseline=baseline)
        t = tier_for(parts["band_width"])
        rows.append(
            HedgeRow(
                date=b.date,
                spot=spot,
                q10=b.q10,
                q50=b.q50,
                q90=b.q90,
                band_width=parts["band_width"],
                drift_pct=parts["drift"],
                upside_tail_pct=parts["upside_tail"],
                drift_term=parts["drift_term"],
                insurance_term=parts["insurance_term"],
                hedge_ratio=hedge,
                baseline_ratio=baseline,
                tier=t,
                rationale=_rationale(parts, t),
            )
        )
    return rows


def summarise_next_quarter(
    rows: list[HedgeRow], *, quarter_months: int = 3
) -> HedgeSummary:
    """Roll the per-month decisions up to a single buy-X-now / wait-Y figure
    across the next `quarter_months` horizons.
    """
    if not rows:
        return HedgeSummary(quarter_months=(), avg_hedge_now=0.0,
                            avg_baseline=0.5, weakest_tier="ABSTAIN")
    next_q = rows[:quarter_months]
    avg = sum(r.hedge_ratio for r in next_q) / len(next_q)
    bl = sum(r.baseline_ratio for r in next_q) / len(next_q)
    # Tier severity: ABSTAIN > RECOMMEND > ACT (worst wins).
    severity = {"ACT": 0, "RECOMMEND": 1, "ABSTAIN": 2}
    weakest = max(next_q, key=lambda r: severity.get(r.tier, 0)).tier
    return HedgeSummary(
        quarter_months=tuple(r.date for r in next_q),
        avg_hedge_now=avg,
        avg_baseline=bl,
        weakest_tier=weakest,
    )


# -- pretty printer ----------------------------------------------------------

def format_table(rows: list[HedgeRow]) -> str:
    lines: list[str] = []
    lines.append(
        f"{'date':<12}  {'q10':>7} {'q50':>7} {'q90':>7}  "
        f"{'width%':>6}  {'drift%':>7}  {'tail%':>6}  "
        f"{'hedge':>6}  {'baseln':>6}  {'tier':<9}  rationale"
    )
    for r in rows:
        lines.append(
            f"{r.date:<12}  "
            f"{r.q10:>7.2f} {r.q50:>7.2f} {r.q90:>7.2f}  "
            f"{r.band_width*100:>5.1f}%  "
            f"{r.drift_pct*100:>+6.1f}%  "
            f"{r.upside_tail_pct*100:>5.1f}%  "
            f"{r.hedge_ratio*100:>5.1f}%  "
            f"{r.baseline_ratio*100:>5.1f}%  "
            f"{r.tier:<9}  {r.rationale}"
        )
    return "\n".join(lines)


# -- CLI ---------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    """python -m src.decision [cache_dir]

    Loads the newest cached forecast and the current spot from the active
    ticker's CSV. Prints per-horizon decisions + next-quarter summary.
    """
    import argparse
    import json
    from pathlib import Path

    from src import config, data
    from src.signals import parse_forecast_bands

    p = argparse.ArgumentParser(description=_cli.__doc__.splitlines()[0])
    p.add_argument("cache_dir", nargs="?", default=None)
    p.add_argument("--quarter", type=int, default=3,
                   help="months in the 'next quarter' rollup (default 3)")
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    if args.cache_dir:
        cdir = Path(args.cache_dir)
    else:
        cands = [d for d in (repo_root / "cache").iterdir()
                 if d.is_dir() and (d / "forecast.json").exists()]
        if not cands:
            print("No cached forecast.json under cache/.")
            return 2
        cdir = max(cands, key=lambda d: d.stat().st_mtime)

    spec = config.active_ticker()
    df = data.load_series(repo_root / spec.csv_path)
    spot = data.current_spot(df)
    forecast = json.loads((cdir / "forecast.json").read_text(encoding="utf-8"))
    bands = parse_forecast_bands(forecast)

    print(f"--- decision for TICKER={spec.symbol} ({spec.display_name}) ---")
    print(f"cache  = {cdir.name[:12]}...")
    print(f"spot   = {spot:.2f} {spec.unit}")
    print()
    rows = decide(bands, spot)
    print(format_table(rows))
    print()
    summary = summarise_next_quarter(rows, quarter_months=args.quarter)
    print("=== next-quarter recommendation ===")
    print(summary.text())
    delta = (summary.avg_hedge_now - summary.avg_baseline) * 100
    direction = "less" if delta < 0 else "more"
    print(f"delta vs baseline: {delta:+.1f} pp ({abs(delta):.1f} pp {direction} hedge)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
