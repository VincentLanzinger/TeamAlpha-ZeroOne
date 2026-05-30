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

from src import config, data
from src import sybilion_client as sc  # noqa: E402


# -- parsers / pretty printers ----------------------------------------------

def parse_forecast_bands(forecast_json: dict[str, Any]) -> list[dict[str, float]]:
    """Return per-horizon-month rows with q0.10 / q0.50 / q0.90 (and point if available)."""
    series = (
        forecast_json.get("data", {}).get("forecast_series")
        or forecast_json.get("forecast_series")
        or {}
    )
    rows: list[dict[str, float]] = []
    for date, row in sorted(series.items()):
        q = row.get("quantile_forecast", {})
        point = row.get("forecast")
        # API publishes 19 levels 0.05..0.95; we focus on 0.10/0.50/0.90.
        rows.append(
            {
                "date": date,
                "q10": float(q.get("0.1") or q.get("0.10") or float("nan")),
                "q50": float(q.get("0.5") or q.get("0.50") or float("nan")),
                "q90": float(q.get("0.9") or q.get("0.90") or float("nan")),
                "point": float(point) if point is not None else float("nan"),
            }
        )
    return rows


def _imp(d: dict[str, Any]) -> float:
    """Importance with three nesting fallbacks: overall.mean → mean → flat."""
    imp = d.get("importance")
    if isinstance(imp, dict):
        ov = imp.get("overall")
        if isinstance(ov, dict) and "mean" in ov:
            return float(ov["mean"])
        if "mean" in imp:
            return float(imp["mean"])
        return 0.0
    return float(imp or 0.0)


def _dir(d: dict[str, Any]) -> str:
    direction = d.get("direction")
    if isinstance(direction, dict):
        return str(direction.get("overall") or direction.get("mean") or "?")
    if direction is None:
        return "?"
    return str(direction)


def _corr(d: dict[str, Any]) -> float | None:
    corr = d.get("correlation")
    if isinstance(corr, dict):
        ov = corr.get("overall")
        if isinstance(ov, dict) and "mean" in ov:
            return float(ov["mean"])
        return float(corr.get("mean")) if "mean" in corr else None
    return float(corr) if corr is not None else None


def _name(d: dict[str, Any]) -> str:
    return str(d.get("driver_name") or d.get("name") or d.get("hash_id") or "?")


def _category(d: dict[str, Any]) -> str:
    return str(d.get("category") or d.get("category_name") or "?")


def parse_drivers(external_signals: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a list of driver rows ranked by importance.overall.mean desc."""
    raw = (
        external_signals.get("data", {}).get("drivers")
        or external_signals.get("drivers")
        or external_signals.get("data", {}).get("external_signals")
        or external_signals.get("external_signals")
        or []
    )
    rows = [
        {
            "name": _name(d),
            "category": _category(d),
            "importance": _imp(d),
            "direction": _dir(d),
            "correlation": _corr(d),
        }
        for d in raw
    ]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows


# -- verdict ----------------------------------------------------------------

def verdict(
    drivers: list[dict[str, Any]],
    *,
    importance_threshold: float = 0.05,
    min_credible: int = 5,
    min_top_importance: float = 0.10,
) -> tuple[str, str]:
    """Return ('KEEP'|'SWITCH', reason)."""
    if not drivers:
        return "SWITCH", "No drivers returned at all."
    credible = [d for d in drivers if d["importance"] >= importance_threshold]
    n_cred = len(credible)
    top_imp = drivers[0]["importance"]
    n_distinct_cats = len({d["category"] for d in credible if d["category"] != "?"})
    if n_cred >= min_credible and top_imp >= min_top_importance:
        return (
            "KEEP",
            f"{n_cred} drivers with importance >= {importance_threshold:.2f}, "
            f"top = {top_imp:.3f}, spans {n_distinct_cats} category bucket(s).",
        )
    return (
        "SWITCH",
        f"Thin: only {n_cred} drivers with importance >= {importance_threshold:.2f} "
        f"(need {min_credible}) and top importance = {top_imp:.3f} "
        f"(need {min_top_importance:.2f}). Try changing TICKER in src/config.py.",
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
    print(f"cache_key  = {key[:12]}…")
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
    rows = parse_forecast_bands(forecast)
    print("--- forecast bands (per horizon month) ---")
    print(f"{'date':<12}  {'q10':>8}  {'q50':>8}  {'q90':>8}  {'width%':>8}")
    spot = data.current_spot(df)
    print(f"{'(spot)':<12}  {' '*8}  {spot:>8.2f}")
    for r in rows:
        width_pct = (r["q90"] - r["q10"]) / r["q50"] * 100 if r["q50"] else float("nan")
        print(
            f"{r['date']:<12}  {r['q10']:>8.2f}  {r['q50']:>8.2f}  "
            f"{r['q90']:>8.2f}  {width_pct:>7.1f}%"
        )
    print()

    # --- drivers ----
    drivers = parse_drivers(signals)
    print(f"--- top {min(top_n, len(drivers))} drivers (of {len(drivers)}) — ranked by importance.overall.mean ---")
    print(f"{'#':>3}  {'imp':>7}  {'dir':>5}  {'corr':>7}  category  name")
    for i, d in enumerate(drivers[:top_n], start=1):
        corr = f"{d['correlation']:+.2f}" if d["correlation"] is not None else "   ?  "
        print(
            f"{i:>3}  {d['importance']:>7.3f}  {d['direction']:>5}  {corr:>7}  "
            f"{d['category']:<14}  {d['name']}"
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
            f"\n→ Try flipping `TICKER = \"{alt}\"` in src/config.py (or refine "
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
