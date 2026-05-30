"""Phase 2 — HOUR-ONE GATE.

Submit the first forecast for the active TICKER (default TTF), parse bands +
external_signals.json, and print a clear KEEP / SWITCH verdict.

Run:
    python scripts/hour_one_gate.py                # uses cache if available
    python scripts/hour_one_gate.py --force        # skip cache (re-bill!)

The script refuses to network if no token is in env, and prints all the
loaded series stats so you can sanity-check the data before paying for a
submit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running as `python scripts/hour_one_gate.py` (no src on sys.path).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config, data, curation
from src import sybilion_client as sc
from src.signals import Driver, parse_drivers, parse_forecast_bands  # noqa: F401


# -- verdict ----------------------------------------------------------------

def verdict(
    drivers: list[Driver],
    *,
    importance_threshold: float = 5.0,
    min_credible: int = 5,
    min_top_importance: float = 20.0,
) -> tuple[str, str]:
    """Return ('KEEP'|'SWITCH', reason).

    Importance scale (observed): per-driver values 0..~100. Many drivers are
    flat-zero; credible TTF drivers like 'Commodities - World' score 30–98.
    Threshold calibrated against that scale.
    """
    if not drivers:
        return "SWITCH", "No drivers returned at all."
    credible = [d for d in drivers if d.importance_overall >= importance_threshold]
    n_cred = len(credible)
    top_imp = drivers[0].importance_overall
    n_distinct_cats = len({d.category for d in credible if d.category})
    if n_cred >= min_credible and top_imp >= min_top_importance:
        return (
            "KEEP",
            f"{n_cred} drivers with importance >= {importance_threshold:.1f}, "
            f"top = {top_imp:.2f}, spans {n_distinct_cats} category bucket(s).",
        )
    return (
        "SWITCH",
        f"Thin: only {n_cred} drivers with importance >= {importance_threshold:.1f} "
        f"(need {min_credible}) and top importance = {top_imp:.2f} "
        f"(need {min_top_importance:.1f}). Try changing TICKER in src/config.py.",
    )


# -- main -------------------------------------------------------------------

def run(force: bool = False, top_n: int = 15) -> int:
    spec = config.active_ticker()
    csv_path = REPO_ROOT / spec.csv_path

    print(f"=== TICKER = {spec.symbol} ({spec.display_name}) ===")
    print(f"csv = {csv_path}\n")

    if not csv_path.exists():
        print(
            f"!! CSV missing: place a 'date,value' monthly file with "
            f">={config.min_observations()} rows at {csv_path} and retry.",
            file=sys.stderr,
        )
        return 2

    df = data.load_series(csv_path)
    data.validate_series(df, min_obs=config.min_observations())
    s = data.stats(df)
    print("--- series stats ---")
    print(s.summary(unit=spec.unit))
    print()

    body = sc.build_forecast_body(
        timeseries=data.to_api_payload(df),
        title=spec.metadata_title,
        description=spec.metadata_description,
        keywords=spec.keywords,
        soft_horizon=config.SOFT_HORIZON,
        recency_factor=spec.recency_factor_override or config.RECENCY_FACTOR,
        backtest=True,
        strictly_positive=config.STRICTLY_POSITIVE,
        regions=list(spec.forecast_regions) or None,
        categories=list(spec.forecast_categories) or None,
    )
    key = sc.cache_key(body)
    cdir = sc.cache_dir_for(body)
    print(f"--- forecast request ---")
    print(f"cache_key  = {key[:12]}...")
    print(f"cache_dir  = {cdir}")
    cached = cdir.exists() and (cdir / "forecast.json").exists()
    print(f"cache_hit  = {cached}{'  (no spend)' if cached else '  (will bill)'}")
    if not cached and not sc.has_token():
        print(
            "\n!! No SYBILION_API_TOKEN in env — refusing to submit live. "
            "Paste the token into .env, then re-run this script.",
            file=sys.stderr,
        )
        return 3

    print("--- submit ---")
    result = sc.submit_and_wait_forecast(body, skip_cache=force)
    print(f"job_id     = {result.get('_job_id', '(cache hit)')}")
    print(f"artifacts  = {sorted(k for k in result if not k.startswith('_'))}\n")

    forecast = result.get("forecast.json", {})
    signals = result.get("external_signals.json", {})

    # --- bands ----
    bands = parse_forecast_bands(forecast)
    print("--- forecast bands (per horizon month) ---")
    print(f"{'date':<12}  {'q10':>8}  {'q50':>8}  {'q90':>8}  {'width%':>8}")
    spot = data.current_spot(df)
    print(f"{'(spot)':<12}  {' '*8}  {spot:>8.2f}")
    for b in bands:
        print(
            f"{b.date:<12}  {b.q10:>8.2f}  {b.q50:>8.2f}  "
            f"{b.q90:>8.2f}  {b.width_pct * 100:>7.1f}%"
        )
    print()

    # --- raw drivers ----
    drivers = parse_drivers(signals)
    n_nonzero = sum(1 for d in drivers if d.importance_overall > 0)
    print(
        f"--- top {min(top_n, len(drivers))} of {len(drivers)} raw drivers "
        f"({n_nonzero} with importance > 0) ---"
    )
    print(f"{'#':>3}  {'imp':>8}  {'dir':>4}  {'dir_val':>8}  {'corr':>7}  {'category':<18}  name")
    for i, d in enumerate(drivers[:top_n], start=1):
        corr = f"{d.correlation:+.3f}" if d.correlation is not None else "   ?   "
        print(
            f"{i:>3}  {d.importance_overall:>8.3f}  "
            f"{d.direction_sign():>4}  {d.direction_overall:>+8.3f}  {corr:>7}  "
            f"{d.category:<18}  {d.name}"
        )
    print()

    # --- curated drivers (Phase 3) ----
    decisions = curation.curate(drivers)
    kept = curation.kept_only(decisions)
    print(
        f"--- top {min(top_n, len(kept))} of {len(kept)} CURATED drivers "
        f"(whitelist + scope plausibility) ---"
    )
    print(f"{'#':>3}  {'adj_imp':>8}  {'raw_imp':>8}  {'tag':>8}  {'category':<18}  name")
    for i, c in enumerate(kept[:top_n], start=1):
        print(
            f"{i:>3}  {c.adjusted_importance_overall:>8.2f}  "
            f"{c.driver.importance_overall:>8.2f}  {c.decision:>8}  "
            f"{c.driver.category:<18}  {c.driver.name}"
        )
    print()

    # --- backtest summary (if present) ----
    bt = result.get("backtest_metrics.json")
    if bt:
        print("--- backtest metrics (rolling windows) ---")
        metrics = bt.get("data") or bt
        print(json.dumps(metrics, indent=2)[:1500])
        print()

    # --- verdict ----
    v, reason = verdict(drivers)
    print(f"=== HOUR-ONE VERDICT: {v} ===")
    print(reason)
    if v == "SWITCH":
        alt = "ALUMINUM" if spec.symbol == "TTF" else "TTF"
        print(
            f"\n-> Try flipping `TICKER = \"{alt}\"` in src/config.py (or refine "
            f"filters/keywords in TICKER_REGISTRY) and re-run.",
        )
    return 0 if v == "KEEP" else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--force",
        action="store_true",
        help="skip cache and re-submit (re-bill)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=15,
        help="print top N drivers (default 15)",
    )
    args = p.parse_args(argv)
    return run(force=args.force, top_n=args.top)


if __name__ == "__main__":
    raise SystemExit(main())
