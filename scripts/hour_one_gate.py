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


HORIZON_KEYS = ("horizon_1", "horizon_2", "horizon_3", "horizon_4", "horizon_5", "horizon_6")


def _sum_inner(d: Any) -> float:
    """An importance/direction horizon entry is {<lag-key str>: float, ...}.
    Aggregate by SUM across lag keys (captures the total contribution of the
    driver at that horizon, regardless of how it splits across lags)."""
    if isinstance(d, dict):
        return float(sum(float(v) for v in d.values()))
    return float(d) if d is not None else 0.0


def _imp(driver: dict[str, Any]) -> float:
    """Driver-level importance = MEAN over horizons of (SUM over inner lag keys).

    The API artifact has no `importance.overall.mean`; we compute it.
    """
    imp = driver.get("importance") or {}
    if not isinstance(imp, dict):
        return float(imp or 0.0)
    # Some drivers have `overall.mean` injected; prefer that when present.
    overall = imp.get("overall")
    if isinstance(overall, dict) and "mean" in overall:
        return float(overall["mean"])
    per_h = [_sum_inner(imp.get(h)) for h in HORIZON_KEYS if h in imp]
    return float(sum(per_h) / len(per_h)) if per_h else 0.0


def _dir(driver: dict[str, Any]) -> float:
    """Driver-level direction = signed mean (positive = up, negative = down).

    Prefer `direction.overall.mean`; otherwise mean over horizons of summed
    inner values.
    """
    direction = driver.get("direction") or {}
    if not isinstance(direction, dict):
        try:
            return float(direction)
        except (TypeError, ValueError):
            return 0.0
    overall = direction.get("overall")
    if isinstance(overall, dict) and "mean" in overall:
        return float(overall["mean"])
    # horizon_0 captures the "now" direction for some drivers.
    if "horizon_0" in direction:
        return float(_sum_inner(direction["horizon_0"]))
    per_h = [_sum_inner(direction.get(h)) for h in HORIZON_KEYS if h in direction]
    return float(sum(per_h) / len(per_h)) if per_h else 0.0


def _corr(driver: dict[str, Any]) -> float | None:
    corr = driver.get("pearson_correlation") or driver.get("correlation") or {}
    if isinstance(corr, dict):
        ov = corr.get("overall")
        if isinstance(ov, dict) and "mean" in ov:
            return float(ov["mean"])
        if "mean" in corr:
            return float(corr["mean"])
        return None
    try:
        return float(corr)
    except (TypeError, ValueError):
        return None


def _name(driver: dict[str, Any]) -> str:
    return str(driver.get("driver_name") or driver.get("name") or driver.get("hash_id") or "?")


def _category(driver: dict[str, Any]) -> str:
    """Drivers carry no explicit category; derive from the 'Cat - Scope' name prefix.

    Examples: 'Commodities - World' -> 'Commodities'; 'Population - Nicaragua' -> 'Population'.
    """
    name = _name(driver)
    if " - " in name:
        return name.split(" - ", 1)[0].strip()
    return name


def parse_drivers(external_signals: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a list of driver rows ranked by computed importance desc.

    Accepts the actual artifact shape: `data` is a dict keyed by UUID, each value
    is `{driver_name, importance: {horizon_1..6: {lag: value}}, direction: {...},
    pearson_correlation: {...}}`. Falls back to legacy `drivers` list shape if seen.
    """
    raw = external_signals.get("data") or external_signals.get("drivers") or []
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = [(d.get("hash_id", ""), d) for d in raw]
    else:
        items = []
    rows = [
        {
            "uuid": uid,
            "name": _name(d),
            "category": _category(d),
            "importance": _imp(d),
            "direction": _dir(d),
            "correlation": _corr(d),
        }
        for uid, d in items
    ]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows


def _dir_arrow(value: float) -> str:
    if value > 0.01:
        return "+"
    if value < -0.01:
        return "-"
    return "0"


# -- verdict ----------------------------------------------------------------

def verdict(
    drivers: list[dict[str, Any]],
    *,
    importance_threshold: float = 5.0,
    min_credible: int = 5,
    min_top_importance: float = 20.0,
) -> tuple[str, str]:
    """Return ('KEEP'|'SWITCH', reason).

    Importance scale (observed): per-driver values are 0 to ~70. Many drivers are
    flat-zero (no contribution); credible TTF drivers like 'Commodities - World'
    score in the 30–70 range. Thresholds calibrated to that scale.
    """
    if not drivers:
        return "SWITCH", "No drivers returned at all."
    credible = [d for d in drivers if d["importance"] >= importance_threshold]
    n_cred = len(credible)
    top_imp = drivers[0]["importance"]
    n_distinct_cats = len({d["category"] for d in credible if d["category"] != "?"})
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
    n_nonzero = sum(1 for d in drivers if d["importance"] > 0)
    print(
        f"--- top {min(top_n, len(drivers))} of {len(drivers)} drivers "
        f"({n_nonzero} with importance > 0) ---"
    )
    print(f"{'#':>3}  {'imp':>8}  {'dir':>4}  {'dir_val':>8}  {'corr':>7}  {'category':<18}  name")
    for i, d in enumerate(drivers[:top_n], start=1):
        corr = f"{d['correlation']:+.3f}" if d["correlation"] is not None else "   ?   "
        print(
            f"{i:>3}  {d['importance']:>8.3f}  "
            f"{_dir_arrow(d['direction']):>4}  {d['direction']:>+8.3f}  {corr:>7}  "
            f"{d['category']:<18}  {d['name']}"
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
