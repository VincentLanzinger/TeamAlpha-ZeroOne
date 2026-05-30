"""Adaptive layer — re-decide live under a shock.

WHY THIS IS THE SUNDAY CAPABILITY
---------------------------------
A forecast is async (minutes) and billable. We can't re-forecast on stage when
the news breaks. But /alerts is SYNCHRONOUS and returns a ranked list of
macroeconomic events with signed pct_change. So a shock loop becomes:

  1. Baseline pressure = signed weighted average of CACHED drivers' directional
     signal (free; from external_signals.json).
  2. Shocked pressure = signed weighted average of LIVE /alerts pct_change
     under shock-specific metadata (1 sync API call).
  3. Pressure delta (shocked - baseline) drives a multiplicative shift of all
     band quantiles; we then re-run the decision engine on the shifted bands.

The demo flow:
  click "Trigger Hormuz shock"
   -> POST /alerts with hormuz metadata (~1s, billable)
   -> aggregate pct_change of top alerts
   -> shift cached bands
   -> re-decide
   -> diff old -> new

This is intentionally lossy (cached bands assume the original training-time
view of the world) but it's the only way to react in seconds on stage. The
trade-off is documented in the demo notes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from src import config, curation, decision, sybilion_client as sc
from src.signals import Driver, HorizonBand, parse_drivers


# -- scenarios --------------------------------------------------------------

@dataclass(frozen=True)
class ShockScenario:
    name: str
    description: str
    keywords: tuple[str, ...]            # appended to ticker base keywords for the /alerts call
    regions: tuple[int, ...] = ()        # empty by default — see note below
    categories: tuple[int, ...] = ()
    sensitivity: float = 0.70            # delta_pressure * sensitivity -> multiplicative shift
    alert_limit: int = 25
    # NOTE on filters: /alerts is keyword-driven semantic search. Empirically
    # adding category/region filters reduces the result count sharply (often
    # to zero) without obvious quality gain. We default to no filters so the
    # live shock has signal; scenarios can opt in if they need scope.


SCENARIOS: dict[str, ShockScenario] = {
    "hormuz": ShockScenario(
        name="hormuz",
        description="Strait of Hormuz closure: oil + LNG tanker supply shock",
        keywords=(
            "Strait of Hormuz", "tanker", "oil supply shock",
            "LNG supply disruption", "Middle East tensions",
            "energy crisis", "shipping route closure",
        ),
        sensitivity=0.70,
    ),
    "ukraine": ShockScenario(
        name="ukraine",
        description="Ukraine corridor disruption: European gas crisis",
        keywords=(
            "Ukraine pipeline", "gas crisis", "European gas storage",
            "Russian gas", "Nord Stream", "geopolitical",
            "winter gas demand",
        ),
        sensitivity=0.70,
    ),
    "lng_outage": ShockScenario(
        name="lng_outage",
        description="Major LNG terminal outage: regasification capacity loss",
        keywords=(
            "LNG terminal outage", "Sabine Pass", "Freeport LNG",
            "regasification capacity", "shipping disruption",
            "European gas import",
        ),
        sensitivity=0.70,
    ),
}


# -- pressure computation ---------------------------------------------------

@dataclass(frozen=True)
class PressureBreakdown:
    pressure: float                              # signed in [-1, +1] roughly
    n_contributors: int
    top_contributors: list[tuple[str, float]]    # (name, signed contribution)
    source: str                                  # "cached_signals" | "live_alerts"


def baseline_pressure_from_signals(
    drivers: Iterable[Driver],
    *,
    curated_only: bool = True,
) -> PressureBreakdown:
    """Aggregate from cached external_signals.json drivers.

    Uses curation.curate() to drop population/etc; then weighted-mean signed
    direction across kept drivers, importance-weighted.
    """
    if curated_only:
        decs = curation.curate(list(drivers))
        kept = [d for d in decs if d.decision != curation.DROPPED]
        if not kept:
            return PressureBreakdown(0.0, 0, [], source="cached_signals")
        total_imp = sum(c.adjusted_importance_overall for c in kept) or 1e-9
        contribs = [
            (
                c.driver.name,
                c.driver.direction_overall * c.adjusted_importance_overall / total_imp,
            )
            for c in kept
        ]
    else:
        ds = list(drivers)
        if not ds:
            return PressureBreakdown(0.0, 0, [], source="cached_signals")
        total_imp = sum(d.importance_overall for d in ds) or 1e-9
        contribs = [
            (d.name, d.direction_overall * d.importance_overall / total_imp)
            for d in ds
        ]
    pressure = sum(v for _, v in contribs)
    top = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)[:5]
    return PressureBreakdown(
        pressure=pressure, n_contributors=len(contribs),
        top_contributors=top, source="cached_signals",
    )


def shock_pressure_from_alerts(
    alerts: list[dict[str, Any]],
    *,
    top_n: int = 10,
    decay: float = 0.85,
) -> PressureBreakdown:
    """Aggregate from live /alerts response.

    Each alert has a signed `pct_change` (% move). We normalise to a fraction,
    weight by rank (exponential decay = top alerts weigh more), and return the
    weighted mean. The result is a single signed scalar in roughly [-0.5, +0.5].
    """
    if not alerts:
        return PressureBreakdown(0.0, 0, [], source="live_alerts")
    head = alerts[:top_n]
    weights = [decay ** i for i in range(len(head))]
    total_w = sum(weights) or 1e-9
    contribs: list[tuple[str, float]] = []
    pressure = 0.0
    for a, w in zip(head, weights):
        name = str(a.get("name") or "?")
        pct = a.get("pct_change")
        try:
            frac = float(pct) / 100.0
        except (TypeError, ValueError):
            frac = 0.0
        contribution = frac * w / total_w
        contribs.append((name, contribution))
        pressure += contribution
    top = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)[:5]
    return PressureBreakdown(
        pressure=pressure, n_contributors=len(head),
        top_contributors=top, source="live_alerts",
    )


# -- band shifting ----------------------------------------------------------

def shift_bands(
    bands: list[HorizonBand],
    delta_pressure: float,
    *,
    sensitivity: float = 0.70,
    clip: tuple[float, float] = (-0.30, 0.30),
) -> list[HorizonBand]:
    """Apply a multiplicative shift to every quantile of every horizon.

    shift_pct = clip(delta_pressure * sensitivity, [-0.30, +0.30])
    new_q = old_q * (1 + shift_pct)
    """
    raw = delta_pressure * sensitivity
    shift_pct = max(clip[0], min(clip[1], raw))
    scale = 1.0 + shift_pct
    out: list[HorizonBand] = []
    for b in bands:
        new_qs = {q: v * scale for q, v in b.quantiles.items()}
        out.append(HorizonBand(
            date=b.date,
            q10=b.q10 * scale,
            q50=b.q50 * scale,
            q90=b.q90 * scale,
            point=b.point * scale,
            quantiles=new_qs,
        ))
    return out


def shift_pct(delta_pressure: float, sensitivity: float = 0.70,
              clip: tuple[float, float] = (-0.30, 0.30)) -> float:
    return max(clip[0], min(clip[1], delta_pressure * sensitivity))


# -- diff structure ---------------------------------------------------------

@dataclass(frozen=True)
class MonthDiff:
    date: str
    old_drift_pct: float
    new_drift_pct: float
    old_hedge: float
    new_hedge: float
    old_tier: str
    new_tier: str

    @property
    def hedge_delta_pp(self) -> float:
        return (self.new_hedge - self.old_hedge) * 100

    @property
    def tier_changed(self) -> bool:
        return self.old_tier != self.new_tier


@dataclass(frozen=True)
class AdaptiveResult:
    scenario: ShockScenario
    spot: float
    baseline_pressure: PressureBreakdown
    shocked_pressure: PressureBreakdown
    delta_pressure: float
    applied_shift_pct: float
    alerts_surfaced: list[dict[str, Any]]
    baseline_rows: list[decision.HedgeRow]
    shocked_rows: list[decision.HedgeRow]
    baseline_summary: decision.HedgeSummary
    shocked_summary: decision.HedgeSummary
    diff: list[MonthDiff]

    def headline(self) -> str:
        d = (self.shocked_summary.avg_hedge_now
             - self.baseline_summary.avg_hedge_now) * 100
        word = "MORE" if d > 0 else ("LESS" if d < 0 else "SAME")
        return (
            f"[{self.scenario.name.upper()}] next-quarter hedge "
            f"{self.baseline_summary.avg_hedge_now * 100:.1f}% -> "
            f"{self.shocked_summary.avg_hedge_now * 100:.1f}% "
            f"({d:+.1f} pp, {word} hedge)"
        )


# -- orchestrator -----------------------------------------------------------

def run_shock(
    scenario: ShockScenario,
    *,
    bands: list[HorizonBand],
    spot: float,
    cached_drivers: list[Driver],
    simulated_pressure: float | None = None,
    token: str | None = None,
) -> AdaptiveResult:
    """End-to-end shock: pull live /alerts under shock metadata, recompute pressure,
    shift cached bands, re-decide.

    If `simulated_pressure` is given, the live call is SKIPPED — we just apply
    the given pressure as if it were the shocked-alerts result. Use this for
    determinism in tests and for a stage safety net.
    """
    baseline = baseline_pressure_from_signals(cached_drivers)

    if simulated_pressure is not None:
        shocked = PressureBreakdown(
            pressure=float(simulated_pressure),
            n_contributors=0,
            top_contributors=[("(simulated)", float(simulated_pressure))],
            source="simulated",
        )
        alerts: list[dict[str, Any]] = []
    else:
        spec = config.active_ticker()
        # Compose shock metadata: ticker keywords + scenario keywords.
        combined_keywords = tuple(spec.keywords) + scenario.keywords
        # Title needs >= 20 chars; combine ticker title with the scenario name.
        shock_title = (
            f"{spec.metadata_title} -- shock scenario: {scenario.description}"
        )[:511]
        alerts = sc.get_alerts(
            title=shock_title,
            description=(
                f"{spec.metadata_description}\n\nShock context: {scenario.description}"
            )[:2048],
            keywords=list(combined_keywords)[:20],
            context_enriched=True,
            regions=list(scenario.regions) or None,
            categories=list(scenario.categories) or None,
            limit=scenario.alert_limit,
            token=token,
        )
        shocked = shock_pressure_from_alerts(alerts, top_n=10)

    delta_pressure = shocked.pressure - baseline.pressure
    applied_shift = shift_pct(delta_pressure, sensitivity=scenario.sensitivity)
    shocked_bands = shift_bands(bands, delta_pressure, sensitivity=scenario.sensitivity)

    base_rows = decision.decide(bands, spot)
    new_rows = decision.decide(shocked_bands, spot)
    base_summary = decision.summarise_next_quarter(base_rows, quarter_months=3)
    new_summary = decision.summarise_next_quarter(new_rows, quarter_months=3)

    diff = [
        MonthDiff(
            date=b.date,
            old_drift_pct=b.drift_pct,
            new_drift_pct=n.drift_pct,
            old_hedge=b.hedge_ratio,
            new_hedge=n.hedge_ratio,
            old_tier=b.tier,
            new_tier=n.tier,
        )
        for b, n in zip(base_rows, new_rows, strict=True)
    ]
    return AdaptiveResult(
        scenario=scenario,
        spot=spot,
        baseline_pressure=baseline,
        shocked_pressure=shocked,
        delta_pressure=delta_pressure,
        applied_shift_pct=applied_shift,
        alerts_surfaced=alerts,
        baseline_rows=base_rows,
        shocked_rows=new_rows,
        baseline_summary=base_summary,
        shocked_summary=new_summary,
        diff=diff,
    )


# -- pretty printer ---------------------------------------------------------

def format_result(r: AdaptiveResult) -> str:
    lines: list[str] = []
    lines.append("=== ADAPTIVE SHOCK ===")
    lines.append(f"scenario      : {r.scenario.name}")
    lines.append(f"description   : {r.scenario.description}")
    lines.append(f"keywords      : {', '.join(r.scenario.keywords)}")
    lines.append(f"sensitivity   : {r.scenario.sensitivity}")
    lines.append("")
    lines.append("--- pressures ---")
    lines.append(
        f"baseline (cached drivers) : {r.baseline_pressure.pressure:+.4f}  "
        f"(n={r.baseline_pressure.n_contributors})"
    )
    for nm, v in r.baseline_pressure.top_contributors[:3]:
        lines.append(f"    {v:+.4f}   {nm}")
    lines.append(
        f"shocked  ({r.shocked_pressure.source})  : "
        f"{r.shocked_pressure.pressure:+.4f}  (n={r.shocked_pressure.n_contributors})"
    )
    for nm, v in r.shocked_pressure.top_contributors[:5]:
        lines.append(f"    {v:+.4f}   {nm}")
    lines.append(
        f"delta                     : {r.delta_pressure:+.4f}   "
        f"applied band shift = {r.applied_shift_pct*100:+.2f}%"
    )
    if r.alerts_surfaced:
        lines.append("")
        lines.append(f"--- top {min(8, len(r.alerts_surfaced))} live alerts surfaced ---")
        for a in r.alerts_surfaced[:8]:
            pct = a.get("pct_change")
            trending = "*" if a.get("trending") else " "
            name = a.get("name") or "?"
            try:
                pct_s = f"{float(pct):+6.2f}%"
            except (TypeError, ValueError):
                pct_s = "    n/a"
            n_news = len(a.get("news") or [])
            lines.append(f"  {trending} {pct_s}  {name}  ({n_news} news items)")
    lines.append("")
    lines.append("--- per-month diff ---")
    lines.append(
        f"{'date':<12}  {'drift% old->new':<18}  "
        f"{'hedge% old->new':<18}  {'tier old->new':<22}  Delta"
    )
    for m in r.diff:
        d_arrow = "->"
        tier_change = "**" if m.tier_changed else "  "
        lines.append(
            f"{m.date:<12}  "
            f"{m.old_drift_pct*100:>+6.1f}% {d_arrow} {m.new_drift_pct*100:>+6.1f}%   "
            f"{m.old_hedge*100:>5.1f}% {d_arrow} {m.new_hedge*100:>5.1f}%        "
            f"{m.old_tier:<10} {d_arrow} {m.new_tier:<10}{tier_change}"
            f"  {m.hedge_delta_pp:+.1f} pp"
        )
    lines.append("")
    lines.append(r.headline())
    if r.baseline_summary.weakest_tier != r.shocked_summary.weakest_tier:
        lines.append(
            f"window tier: {r.baseline_summary.weakest_tier} -> "
            f"{r.shocked_summary.weakest_tier}"
        )
    return "\n".join(lines)


# -- CLI --------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    """python -m src.adaptive --scenario {hormuz|ukraine|lng_outage}

    Loads the newest cached forecast + cached drivers; pulls live /alerts under
    the chosen scenario; prints the full before/after diff.
    """
    import argparse
    import json
    from pathlib import Path

    from src import data
    from src.signals import parse_forecast_bands

    p = argparse.ArgumentParser(description=_cli.__doc__.splitlines()[0])
    p.add_argument("--scenario", choices=list(SCENARIOS), default="hormuz")
    p.add_argument("--cache-dir", default=None,
                   help="explicit cache/<hash>/ dir (default: newest)")
    p.add_argument("--simulated-pressure", type=float, default=None,
                   help="bypass live /alerts; apply this pressure directly")
    p.add_argument("--sensitivity", type=float, default=None,
                   help="override scenario sensitivity")
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
    signals = json.loads((cdir / "external_signals.json").read_text(encoding="utf-8"))
    cached_drivers = parse_drivers(signals)

    scenario = SCENARIOS[args.scenario]
    if args.sensitivity is not None:
        scenario = ShockScenario(
            **{**scenario.__dict__, "sensitivity": args.sensitivity}
        )

    result = run_shock(
        scenario,
        bands=bands,
        spot=spot,
        cached_drivers=cached_drivers,
        simulated_pressure=args.simulated_pressure,
    )
    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
