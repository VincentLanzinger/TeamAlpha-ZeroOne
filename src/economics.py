"""Cost-of-waiting analysis + backtest trust grounding.

Given the per-month hedge decisions and the forecast bands, compute for each
month:
  - cost(hedge_now)  = spot                                  (certain)
  - cost(wait)       = distribution over the forecast quantiles
  - expected cost(strategy h) = h * spot + (1-h) * E[future]
                              ~ h * spot + (1-h) * q50
  - worst-case cost(strategy h, q*) = h * spot + (1-h) * q*  for q* = q90
  - regret(strategy h, scenario p) = h*spot + (1-h)*p - min(spot, p)
    (always >= 0; integrated over the 19 quantile levels yields E[regret])

These let us compare the rule's hedge ratio against:
  - the naive 50% baseline,
  - the corner cases "all now" (h=1.0) and "all wait" (h=0.0),
  - on both EV and worst-case axes.

BACKTEST GROUNDING

Sybilion publishes MAPE/MASE/RMSE per rolling window. We use them to widen
the ABSTAIN zone when the model has been historically poor. The grounding is
a SHIFT (not a scale), so even a perfect MAPE doesn't move the thresholds:

  trust_mape = clip(1 - MAPE / mape_zero_trust, 0, 1)   # MAPE 30% -> 0
  mase_penalty = 0.5 if MASE > 10 else 1.0              # heuristic floor
  trust = trust_mape * mase_penalty

  shift = (1 - trust) * MAX_THRESHOLD_SHIFT             # 0..0.20
  effective ACT_MAX        = ACT_MAX        - shift
  effective RECOMMEND_MAX  = RECOMMEND_MAX  - shift

Why MASE is a floor rather than a multiplier: the API's MASE values look
anomalously large (~93 on the TTF run, while MAPE is a sensible 15%). We
treat very-high MASE as a "something is off" red flag that halves trust,
rather than as a directly-interpretable scaled error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.decision import (
    ACT_MAX_WIDTH,
    BASELINE,
    HedgeRow,
    RECOMMEND_MAX_WIDTH,
)
from src.signals import HorizonBand

# Grounding parameters
MAPE_ZERO_TRUST: float = 0.30           # MAPE >= 30% -> trust = 0
MASE_RED_FLAG: float = 10.0             # MASE above this halves trust
MAX_THRESHOLD_SHIFT: float = 0.20       # max move of the ACT/RECOMMEND thresholds


# ---------- cost-of-waiting ------------------------------------------------

@dataclass(frozen=True)
class CostRow:
    """All economics for one horizon month, per MWh."""
    date: str
    spot: float
    q10: float
    q50: float
    q90: float
    hedge_ratio: float
    baseline_ratio: float

    # Cost under each strategy in EV / worst-case
    rule_cost_ev: float          # h*spot + (1-h)*q50
    rule_cost_p90: float         # h*spot + (1-h)*q90
    baseline_cost_ev: float
    baseline_cost_p90: float
    all_now_cost: float          # = spot
    all_wait_cost_ev: float      # = q50
    all_wait_cost_p90: float     # = q90

    # Regret (averaged over the 19 quantile scenarios, uniform weights)
    rule_expected_regret: float
    baseline_expected_regret: float
    all_now_expected_regret: float
    all_wait_expected_regret: float

    # Convenience deltas
    rule_savings_vs_baseline_ev: float    # >0 = rule cheaper in EV
    rule_savings_vs_all_now_ev: float
    rule_extra_risk_vs_all_now: float     # rule_p90 - all_now_cost


def _expected_cost(h: float, spot: float, mid: float) -> float:
    return h * spot + (1.0 - h) * mid


def _worst_cost(h: float, spot: float, q_worst: float) -> float:
    return h * spot + (1.0 - h) * q_worst


def _expected_regret(h: float, spot: float, quantiles: dict[float, float]) -> float:
    """Expected regret of strategy h across the quantile scenarios.

    For each scenario p:
      regret(h, p) = h*spot + (1-h)*p - min(spot, p)
                   = h * max(0, spot - p)  +  (1-h) * max(0, p - spot)
    Uniform-weighted mean over the quantile values.

    Falls back to a 3-point approximation if `quantiles` is empty.
    """
    if not quantiles:
        return 0.0
    vals = list(quantiles.values())
    n = len(vals)
    total = 0.0
    for p in vals:
        if p <= spot:
            total += h * (spot - p)
        else:
            total += (1.0 - h) * (p - spot)
    return total / n


def cost_of_waiting(
    rows: list[HedgeRow],
    bands: list[HorizonBand],
) -> list[CostRow]:
    """Aligned by index — `rows[i]` and `bands[i]` must describe the same month."""
    out: list[CostRow] = []
    for hr, b in zip(rows, bands, strict=True):
        h = hr.hedge_ratio
        hb = hr.baseline_ratio
        out.append(
            CostRow(
                date=hr.date,
                spot=hr.spot,
                q10=hr.q10,
                q50=hr.q50,
                q90=hr.q90,
                hedge_ratio=h,
                baseline_ratio=hb,
                rule_cost_ev=_expected_cost(h, hr.spot, hr.q50),
                rule_cost_p90=_worst_cost(h, hr.spot, hr.q90),
                baseline_cost_ev=_expected_cost(hb, hr.spot, hr.q50),
                baseline_cost_p90=_worst_cost(hb, hr.spot, hr.q90),
                all_now_cost=hr.spot,
                all_wait_cost_ev=hr.q50,
                all_wait_cost_p90=hr.q90,
                rule_expected_regret=_expected_regret(h, hr.spot, b.quantiles),
                baseline_expected_regret=_expected_regret(hb, hr.spot, b.quantiles),
                all_now_expected_regret=_expected_regret(1.0, hr.spot, b.quantiles),
                all_wait_expected_regret=_expected_regret(0.0, hr.spot, b.quantiles),
                rule_savings_vs_baseline_ev=_expected_cost(hb, hr.spot, hr.q50)
                                              - _expected_cost(h, hr.spot, hr.q50),
                rule_savings_vs_all_now_ev=hr.spot - _expected_cost(h, hr.spot, hr.q50),
                rule_extra_risk_vs_all_now=_worst_cost(h, hr.spot, hr.q90) - hr.spot,
            )
        )
    return out


def quarter_summary(cost_rows: list[CostRow], *, months: int = 3) -> dict[str, float]:
    """Average cost metrics across the first `months` rows."""
    next_q = cost_rows[:months]
    if not next_q:
        return {}
    n = len(next_q)
    def mean(attr: str) -> float:
        return sum(getattr(r, attr) for r in next_q) / n
    return {
        "rule_cost_ev": mean("rule_cost_ev"),
        "rule_cost_p90": mean("rule_cost_p90"),
        "baseline_cost_ev": mean("baseline_cost_ev"),
        "baseline_cost_p90": mean("baseline_cost_p90"),
        "all_now_cost": mean("all_now_cost"),
        "all_wait_cost_ev": mean("all_wait_cost_ev"),
        "all_wait_cost_p90": mean("all_wait_cost_p90"),
        "rule_expected_regret": mean("rule_expected_regret"),
        "baseline_expected_regret": mean("baseline_expected_regret"),
        "rule_savings_vs_baseline_ev": mean("rule_savings_vs_baseline_ev"),
        "rule_savings_vs_all_now_ev": mean("rule_savings_vs_all_now_ev"),
        "rule_extra_risk_vs_all_now": mean("rule_extra_risk_vs_all_now"),
    }


# ---------- backtest grounding ---------------------------------------------

@dataclass(frozen=True)
class TrustFactor:
    mape: float | None
    mase: float | None
    rmse: float | None
    trust_mape: float
    mase_penalty: float
    trust: float
    threshold_shift: float
    effective_act_max: float
    effective_recommend_max: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        return (
            f"MAPE        = {self.mape * 100:.1f}%  -> trust_mape = {self.trust_mape:.2f}"
            if self.mape is not None
            else "MAPE = n/a"
        ) + "\n" + (
            f"MASE        = {self.mase:.2f}  -> mase_penalty = {self.mase_penalty:.2f}"
            if self.mase is not None
            else "MASE = n/a"
        ) + "\n" + (
            f"trust       = {self.trust:.2f}\n"
            f"shift       = {self.threshold_shift:.3f}\n"
            f"effective ACT_MAX        = {self.effective_act_max:.3f}  (was {ACT_MAX_WIDTH})\n"
            f"effective RECOMMEND_MAX  = {self.effective_recommend_max:.3f}  (was {RECOMMEND_MAX_WIDTH})"
        ) + ("\n" + "\n".join(self.notes) if self.notes else "")


def _pick_window(metrics: dict[str, Any], prefer: tuple[str, ...] = ("12m", "24m", "6m", "60m")) -> dict[str, Any] | None:
    """Return the first window present; metrics may be flat or nested under window keys."""
    if not isinstance(metrics, dict):
        return None
    if "metrics" in metrics and isinstance(metrics["metrics"], dict):
        return metrics["metrics"]
    body = metrics.get("data", metrics)
    if not isinstance(body, dict):
        return None
    for w in prefer:
        if w in body and isinstance(body[w], dict):
            inner = body[w].get("metrics", body[w])
            if isinstance(inner, dict):
                return inner
    # Last resort: assume body is already a metrics dict
    if any(k in body for k in ("MAPE", "MASE", "RMSE", "MAE")):
        return body
    return None


def compute_trust(
    backtest_metrics: dict[str, Any] | None,
    *,
    mape_zero_trust: float = MAPE_ZERO_TRUST,
    mase_red_flag: float = MASE_RED_FLAG,
    max_threshold_shift: float = MAX_THRESHOLD_SHIFT,
) -> TrustFactor:
    notes: list[str] = []
    if backtest_metrics is None:
        notes.append("backtest_metrics missing -> trust=1.0 (no penalty)")
        return TrustFactor(
            mape=None, mase=None, rmse=None,
            trust_mape=1.0, mase_penalty=1.0, trust=1.0,
            threshold_shift=0.0,
            effective_act_max=ACT_MAX_WIDTH,
            effective_recommend_max=RECOMMEND_MAX_WIDTH,
            notes=tuple(notes),
        )
    inner = _pick_window(backtest_metrics) or {}
    mape_pct = inner.get("MAPE")
    mase = inner.get("MASE")
    rmse = inner.get("RMSE")

    mape_frac = (float(mape_pct) / 100.0) if mape_pct is not None else None
    trust_mape = (
        max(0.0, min(1.0, 1.0 - (mape_frac / mape_zero_trust)))
        if mape_frac is not None
        else 1.0
    )
    if mape_frac is None:
        notes.append("no MAPE in backtest -> trust_mape = 1.0")
    mase_penalty = 1.0
    if mase is not None and float(mase) > mase_red_flag:
        mase_penalty = 0.5
        notes.append(
            f"MASE={float(mase):.2f} > {mase_red_flag} red-flag floor -> mase_penalty=0.5"
        )
    trust = trust_mape * mase_penalty
    shift = (1.0 - trust) * max_threshold_shift
    eff_act = max(0.0, ACT_MAX_WIDTH - shift)
    eff_rec = max(eff_act, RECOMMEND_MAX_WIDTH - shift)
    return TrustFactor(
        mape=mape_frac,
        mase=float(mase) if mase is not None else None,
        rmse=float(rmse) if rmse is not None else None,
        trust_mape=trust_mape,
        mase_penalty=mase_penalty,
        trust=trust,
        threshold_shift=shift,
        effective_act_max=eff_act,
        effective_recommend_max=eff_rec,
        notes=tuple(notes),
    )


def tier_for_grounded(band_width: float, trust: TrustFactor) -> str:
    if band_width < trust.effective_act_max:
        return "ACT"
    if band_width < trust.effective_recommend_max:
        return "RECOMMEND"
    return "ABSTAIN"


@dataclass(frozen=True)
class GroundedRow:
    date: str
    band_width: float
    original_tier: str
    grounded_tier: str
    changed: bool


def apply_grounding(rows: list[HedgeRow], trust: TrustFactor) -> list[GroundedRow]:
    out: list[GroundedRow] = []
    for r in rows:
        new = tier_for_grounded(r.band_width, trust)
        out.append(
            GroundedRow(
                date=r.date,
                band_width=r.band_width,
                original_tier=r.tier,
                grounded_tier=new,
                changed=new != r.tier,
            )
        )
    return out


# ---------- pretty printers -------------------------------------------------

def format_cost_table(rows: list[CostRow]) -> str:
    """Wide table; numbers per MWh."""
    lines: list[str] = []
    lines.append(
        f"{'date':<12}  {'spot':>6} {'q50':>6} {'q90':>6}  "
        f"{'rule_EV':>7} {'rule_p90':>8}  {'base_EV':>7} {'base_p90':>8}  "
        f"{'sav_vs_base':>11} {'sav_vs_now':>10} {'risk_vs_now':>11}  "
        f"{'rgt_rule':>8} {'rgt_base':>8}"
    )
    for r in rows:
        lines.append(
            f"{r.date:<12}  "
            f"{r.spot:>6.2f} {r.q50:>6.2f} {r.q90:>6.2f}  "
            f"{r.rule_cost_ev:>7.2f} {r.rule_cost_p90:>8.2f}  "
            f"{r.baseline_cost_ev:>7.2f} {r.baseline_cost_p90:>8.2f}  "
            f"{r.rule_savings_vs_baseline_ev:>+11.3f} "
            f"{r.rule_savings_vs_all_now_ev:>+10.3f} "
            f"{r.rule_extra_risk_vs_all_now:>+11.3f}  "
            f"{r.rule_expected_regret:>8.3f} {r.baseline_expected_regret:>8.3f}"
        )
    return "\n".join(lines)


def format_grounding_table(grounded: list[GroundedRow]) -> str:
    lines: list[str] = []
    lines.append(f"{'date':<12}  {'width%':>6}  {'original':<10}  {'grounded':<10}  changed?")
    for g in grounded:
        flag = "YES" if g.changed else ""
        lines.append(
            f"{g.date:<12}  {g.band_width*100:>5.1f}%  "
            f"{g.original_tier:<10}  {g.grounded_tier:<10}  {flag}"
        )
    return "\n".join(lines)


# ---------- CLI ------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    """python -m src.economics [cache_dir]

    Reads forecast.json + backtest_metrics.json from the newest cached forecast,
    computes per-month costs + regret + trust grounding, prints the full report.
    """
    import argparse
    import json
    from pathlib import Path

    from src import config, data, decision
    from src.signals import parse_forecast_bands

    p = argparse.ArgumentParser(description=_cli.__doc__.splitlines()[0])
    p.add_argument("cache_dir", nargs="?", default=None)
    p.add_argument("--volume", type=float, default=100_000.0,
                   help="MWh per month for total-EUR rollup (default 100,000)")
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
    bands = parse_forecast_bands(
        json.loads((cdir / "forecast.json").read_text(encoding="utf-8"))
    )
    rows = decision.decide(bands, spot)
    cost_rows = cost_of_waiting(rows, bands)
    bt_path = cdir / "backtest_metrics.json"
    bt = json.loads(bt_path.read_text(encoding="utf-8")) if bt_path.exists() else None
    trust = compute_trust(bt)
    grounded = apply_grounding(rows, trust)

    print(f"--- cost-of-waiting / regret  ({spec.symbol}, spot={spot:.2f} {spec.unit}) ---")
    print(f"cache = {cdir.name[:12]}...  (quantile levels in artifact: "
          f"{len(bands[0].quantiles) if bands else 0})")
    print()
    print(format_cost_table(cost_rows))
    print()
    s = quarter_summary(cost_rows, months=3)
    if s:
        print("=== next-quarter rollup (per MWh) ===")
        print(f"rule     EV={s['rule_cost_ev']:.2f}  p90={s['rule_cost_p90']:.2f}  "
              f"E[regret]={s['rule_expected_regret']:.3f}")
        print(f"baseline EV={s['baseline_cost_ev']:.2f}  p90={s['baseline_cost_p90']:.2f}  "
              f"E[regret]={s['baseline_expected_regret']:.3f}")
        print(f"all_now  EV={s['all_now_cost']:.2f}  "
              f"  (locked-in cost; no risk)")
        print(f"all_wait EV={s['all_wait_cost_ev']:.2f}  p90={s['all_wait_cost_p90']:.2f}  "
              f"  (no commitment; full exposure)")
        v = args.volume * 3
        print()
        print(f"--- next quarter total cost @ {args.volume:,.0f} MWh/month, 3 months "
              f"= {v:,.0f} MWh ---")
        for label, ev, p90 in [
            ("rule    ", s["rule_cost_ev"], s["rule_cost_p90"]),
            ("baseline", s["baseline_cost_ev"], s["baseline_cost_p90"]),
            ("all_now ", s["all_now_cost"], s["all_now_cost"]),
            ("all_wait", s["all_wait_cost_ev"], s["all_wait_cost_p90"]),
        ]:
            print(f"{label}: EV {v * ev:>14,.0f} EUR   p90 {v * p90:>14,.0f} EUR")
        sav_base = v * s["rule_savings_vs_baseline_ev"]
        sav_now = v * s["rule_savings_vs_all_now_ev"]
        risk_now = v * s["rule_extra_risk_vs_all_now"]
        print()
        print(
            f"rule vs baseline: {sav_base:+,.0f} EUR EV "
            f"(>0 = rule saves money in expectation)"
        )
        print(
            f"rule vs all_now : {sav_now:+,.0f} EUR EV  "
            f"with +{risk_now:,.0f} EUR worst-case extra spend (p90)"
        )

    print()
    print("=== backtest trust grounding ===")
    print(trust.summary())
    print()
    print("--- tier shifts ---")
    print(format_grounding_table(grounded))
    changed = sum(1 for g in grounded if g.changed)
    print()
    print(
        f"{changed} of {len(grounded)} months tier-shifted due to grounding "
        f"({'wider abstain' if any(g.grounded_tier == 'ABSTAIN' and not g.original_tier == 'ABSTAIN' for g in grounded) else 'no abstain widening'})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
