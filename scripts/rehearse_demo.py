"""Deterministic dry-run of the stage demo.

Runs the full path twice (gate -> cached forecast -> decision -> curation ->
economics -> adaptive shock simulated -> adaptive shock live -> narrator)
and prints a one-line PASS/FAIL per step so a non-developer can run it
before going on stage.

Use:
    python scripts/rehearse_demo.py [--skip-live] [--skip-llm]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src import (  # noqa: E402
    adaptive, config, curation, data, decision, economics, narrator,
    sybilion_client as sc,
)
from src.signals import parse_drivers, parse_forecast_bands  # noqa: E402


GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def step(label: str, fn):
    """Run fn(), print PASS/FAIL with timing, return its result (or None on fail)."""
    t0 = time.time()
    try:
        result = fn()
        dt = (time.time() - t0) * 1000
        print(f"  {GREEN}PASS{RESET}  {label:<55}  {DIM}{dt:>6.0f} ms{RESET}")
        return result
    except Exception as e:
        dt = (time.time() - t0) * 1000
        print(f"  {RED}FAIL{RESET}  {label:<55}  {DIM}{dt:>6.0f} ms{RESET}")
        print(f"        -> {type(e).__name__}: {e}")
        if os.environ.get("REHEARSE_VERBOSE"):
            traceback.print_exc()
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--skip-live", action="store_true",
                   help="skip the live /alerts shock (still runs simulated)")
    p.add_argument("--skip-llm", action="store_true",
                   help="skip the Featherless live narration (still runs template)")
    args = p.parse_args(argv)

    print("=" * 70)
    print(" Hedge Decision Agent -- demo rehearsal")
    print("=" * 70)
    print()

    # --- Step 1: env ----------------------------------------------------------
    print("Environment")
    sybilion_ok = step("SYBILION_API_TOKEN present", lambda: (
        sc.has_token() or (_ for _ in ()).throw(RuntimeError("missing"))
    ))
    featherless_ok = step(
        "FEATHERLESS_API_KEY present (optional)",
        lambda: narrator.has_llm() or (_ for _ in ()).throw(RuntimeError("missing")),
    ) if not args.skip_llm else None

    # --- Step 2: cache ---------------------------------------------------------
    print("\nCached forecast")
    cands = [d for d in (REPO / "cache").iterdir()
             if d.is_dir() and (d / "forecast.json").exists()]
    cdir = step("locate cache dir", lambda: max(cands, key=lambda d: d.stat().st_mtime)
                if cands else (_ for _ in ()).throw(RuntimeError("no cache")))
    if cdir is None:
        return 2
    print(f"        cache = {cdir.name[:24]}...")

    # --- Step 3: data ---------------------------------------------------------
    print("\nData + decision")
    spec = config.active_ticker()
    df = step("load + validate TTF CSV",
              lambda: data.load_series(REPO / spec.csv_path))
    if df is None:
        return 2
    spot = step("compute spot", lambda: data.current_spot(df))
    bands = step("parse forecast bands",
                  lambda: parse_forecast_bands(
                      json.loads((cdir / "forecast.json").read_text(encoding="utf-8"))
                  ))
    cached_drivers = step("parse external signals",
                          lambda: parse_drivers(
                              json.loads((cdir / "external_signals.json").read_text(encoding="utf-8"))
                          ))

    # --- Step 4: curation + decision + economics ------------------------------
    decisions = step("curate drivers",
                      lambda: curation.curate(cached_drivers))
    rows = step("decide per month", lambda: decision.decide(bands, spot))
    summary = step("summarise next quarter",
                    lambda: decision.summarise_next_quarter(rows, quarter_months=3))
    cost = step("cost-of-waiting",
                lambda: economics.cost_of_waiting(rows, bands))
    trust = step("backtest grounding",
                  lambda: economics.compute_trust(
                      json.loads((cdir / "backtest_metrics.json").read_text(encoding="utf-8"))
                  ))

    print()
    print(f"  Recommendation: buy {summary.avg_hedge_now*100:.1f}% now, "
          f"wait {(1-summary.avg_hedge_now)*100:.1f}% -- "
          f"tier {summary.weakest_tier}, trust {trust.trust:.2f}")

    # --- Step 5: shock x2 -----------------------------------------------------
    print("\nAdaptive shock (twice)")
    sim = step(
        "simulated +0.20 (hormuz)",
        lambda: adaptive.run_shock(
            adaptive.SCENARIOS["hormuz"], bands=bands, spot=spot,
            cached_drivers=cached_drivers, simulated_pressure=+0.20,
        ),
    )
    if sim is not None:
        delta = (sim.shocked_summary.avg_hedge_now - sim.baseline_summary.avg_hedge_now) * 100
        print(f"        {sim.baseline_summary.avg_hedge_now*100:.1f}% -> "
              f"{sim.shocked_summary.avg_hedge_now*100:.1f}% ({delta:+.1f} pp)")

    if not args.skip_live and sybilion_ok:
        live = step(
            "live /alerts (ukraine, billable)",
            lambda: adaptive.run_shock(
                adaptive.SCENARIOS["ukraine"], bands=bands, spot=spot,
                cached_drivers=cached_drivers,
            ),
        )
        if live is not None:
            delta = (live.shocked_summary.avg_hedge_now - live.baseline_summary.avg_hedge_now) * 100
            print(f"        {live.baseline_summary.avg_hedge_now*100:.1f}% -> "
                  f"{live.shocked_summary.avg_hedge_now*100:.1f}% ({delta:+.1f} pp), "
                  f"{len(live.alerts_surfaced)} live alerts")

    # --- Step 6: narrator -----------------------------------------------------
    print("\nNarrator")
    bundle = step("build bundle from cache",
                  lambda: narrator.build_bundle_from_cache(cdir))
    template_out = step(
        "narrate (template, deterministic)",
        lambda: narrator.narrate(bundle, use_llm=False),
    )
    if template_out is not None:
        print(f"        memo: {len(template_out.memo)} chars  "
              f"counter: {len(template_out.counter_case)} chars")
    if not args.skip_llm and featherless_ok:
        llm_out = step(
            "narrate (Featherless live)",
            lambda: narrator.narrate(bundle, use_llm=True),
        )
        if llm_out is not None:
            print(f"        model: {llm_out.model}  "
                  f"used_llm: {llm_out.used_llm}")

    # --- Step 7: balance gate --------------------------------------------------
    print("\nBalance gate")
    if sybilion_ok:
        info = step("check_account", lambda: sc.check_account())
        if info is not None:
            print(f"        available: €{info.available_eur:.2f}  "
                  f"tier: {info.api_usage_tier}  "
                  f"recent: {info.recent_usage_events} events, "
                  f"€{info.recent_spend_eur_cents/100:.2f}")

    print()
    print("=" * 70)
    print(" Rehearsal complete. If every step says PASS, you're ready for stage.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
