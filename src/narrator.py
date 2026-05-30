"""LLM explanation layer — STRICT verbalizer of pre-computed decisions.

Hard role contract (enforced via system prompt + tests):
  * The LLM verbalizes the given numbers into plain English.
  * It NEVER computes new metrics or proposes a different hedge ratio.
  * If a question goes beyond the provided data, it says so explicitly.

Two outputs:
  1. `memo`         — a short procurement-board memo (~180 words).
  2. `counter_case` — a devil's-advocate counter-case (~120 words) framed as
                      "what would have to be true for the opposite call".

Default provider: Featherless.ai (OpenAI-compatible chat completions).
  Auth: Authorization: Bearer $FEATHERLESS_API_KEY
  Base: https://api.featherless.ai/v1/chat/completions
  Model: from env FEATHERLESS_MODEL, default a recent Llama-3.x-Instruct.

Fallback: if no key is configured, we generate `memo` and `counter_case` from
a deterministic template using the same bundle of facts. No LLM call, no
network — useful for tests, demos without internet, and as a baseline to
compare against the LLM output.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from src import curation, decision, economics
from src.adaptive import AdaptiveResult
from src.signals import Driver, HorizonBand


# -- inputs ------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionBundle:
    """Everything the narrator needs. Built upstream by the cockpit / CLI."""
    ticker: str
    display_name: str
    unit: str
    spot: float
    bands: list[HorizonBand]
    hedge_rows: list[decision.HedgeRow]
    summary: decision.HedgeSummary
    curated: list[curation.CurationDecision]
    backtest: dict[str, Any] | None
    trust: economics.TrustFactor | None
    cost_rows: list[economics.CostRow]
    cost_summary: dict[str, float]
    adaptive: AdaptiveResult | None = None


# -- output ------------------------------------------------------------------

@dataclass(frozen=True)
class Narration:
    memo: str
    counter_case: str
    used_llm: bool
    model: str          # "template" or the Featherless model id
    facts: str          # the structured-fact list we passed to the LLM
    notes: tuple[str, ...] = field(default_factory=tuple)


# -- featherless config ------------------------------------------------------

FEATHERLESS_DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
FEATHERLESS_URL = "https://api.featherless.ai/v1/chat/completions"
FEATHERLESS_UA = "hedge-decision-agent/0.1 (+python-urllib)"  # Cloudflare blocks default Python-urllib UA
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_TEMPERATURE = 0.3


def _featherless_key() -> str | None:
    v = os.environ.get("FEATHERLESS_API_KEY")
    return v.strip() if v and v.strip() else None


def _featherless_model() -> str:
    v = os.environ.get("FEATHERLESS_MODEL")
    return v.strip() if v and v.strip() else FEATHERLESS_DEFAULT_MODEL


def has_llm() -> bool:
    return _featherless_key() is not None


# -- fact-list builder -------------------------------------------------------

def _format_curated_top(curated: list[curation.CurationDecision], n: int) -> str:
    if not curated:
        return "(no curated drivers)"
    kept = [c for c in curated if c.decision != curation.DROPPED][:n]
    lines: list[str] = []
    for c in kept:
        sign = "+" if c.driver.direction_overall > 0 else "-"
        corr = (
            f"corr {c.driver.correlation:+.2f}"
            if c.driver.correlation is not None
            else "corr n/a"
        )
        lines.append(
            f"    - {c.driver.name}  (imp {c.adjusted_importance_overall:.1f}, "
            f"dir {sign}, {corr})"
        )
    return "\n".join(lines)


def _format_dropped_summary(curated: list[curation.CurationDecision]) -> str:
    drops = [c for c in curated if c.decision == curation.DROPPED]
    by_cat: dict[str, int] = {}
    for c in drops:
        by_cat[c.driver.category] = by_cat.get(c.driver.category, 0) + 1
    if not by_cat:
        return "  (nothing dropped)"
    parts = sorted(by_cat.items(), key=lambda kv: -kv[1])
    return "  " + "; ".join(f"{n} {cat}" for cat, n in parts)


def build_facts(bundle: DecisionBundle) -> str:
    """Render the bundle as a compact, plain-English fact list. This is what
    the LLM sees — no extra latitude.
    """
    s = bundle.summary
    delta_pp = (s.avg_hedge_now - s.avg_baseline) * 100
    direction_word = "LESS" if delta_pp < 0 else ("MORE" if delta_pp > 0 else "SAME")

    per_month = []
    for r in bundle.hedge_rows:
        per_month.append(
            f"    {r.date}: band width {r.band_width*100:.1f}%, "
            f"drift {r.drift_pct*100:+.1f}%, "
            f"upside tail {r.upside_tail_pct*100:.1f}%, "
            f"hedge {r.hedge_ratio*100:.1f}%, tier {r.tier}"
        )

    cost = bundle.cost_summary
    trust_block = ""
    if bundle.trust is not None:
        t = bundle.trust
        mape_str = f"{t.mape*100:.1f}%" if t.mape is not None else "n/a"
        mase_str = f"{t.mase:.2f}" if t.mase is not None else "n/a"
        trust_block = (
            f"  - MAPE: {mape_str}   MASE: {mase_str}\n"
            f"  - trust factor: {t.trust:.2f} (1.0 = full trust, 0 = ignore model)\n"
            f"  - effective ACT_MAX width: {t.effective_act_max:.3f} "
            f"(was {decision.ACT_MAX_WIDTH})\n"
            f"  - effective RECOMMEND_MAX width: {t.effective_recommend_max:.3f} "
            f"(was {decision.RECOMMEND_MAX_WIDTH})"
        )

    adaptive_block = ""
    if bundle.adaptive is not None:
        a = bundle.adaptive
        d_pp = (a.shocked_summary.avg_hedge_now - a.baseline_summary.avg_hedge_now) * 100
        adaptive_block = (
            f"\nADAPTIVE shock ({a.scenario.name} — {a.scenario.description}):\n"
            f"  - pressure: baseline {a.baseline_pressure.pressure:+.3f}, "
            f"shocked {a.shocked_pressure.pressure:+.3f}, "
            f"delta {a.delta_pressure:+.3f}\n"
            f"  - applied band shift: {a.applied_shift_pct*100:+.2f}%\n"
            f"  - next-quarter hedge: "
            f"{a.baseline_summary.avg_hedge_now*100:.1f}% -> "
            f"{a.shocked_summary.avg_hedge_now*100:.1f}% "
            f"({d_pp:+.1f} pp)\n"
            f"  - top live alerts:\n"
            + "\n".join(
                f"      {al.get('pct_change'):+.2f}%  {al.get('name')}"
                for al in (a.alerts_surfaced or [])[:5]
                if al.get("pct_change") is not None
            )
        )

    return f"""TICKER: {bundle.ticker} ({bundle.display_name})
SPOT: {bundle.spot:.2f} {bundle.unit}

DECISION (next quarter, {", ".join(s.quarter_months)}):
  - buy now: {s.avg_hedge_now*100:.1f}%   (vs naive baseline {s.avg_baseline*100:.0f}%)
  - delta vs baseline: {delta_pp:+.1f} percentage points ({direction_word} hedge)
  - worst tier in window: {s.weakest_tier}

PER-MONTH:
{chr(10).join(per_month)}

CURATED DRIVERS (top 5, kept after whitelist + scope demotion):
{_format_curated_top(bundle.curated, 5)}

DROPPED (categories not in whitelist):
{_format_dropped_summary(bundle.curated)}

BACKTEST GROUNDING:
{trust_block or "  (no backtest)"}

COST OF WAITING (per MWh, quarter avg):
  - rule EV {cost.get('rule_cost_ev', 0):.2f}, p90 {cost.get('rule_cost_p90', 0):.2f}
  - baseline EV {cost.get('baseline_cost_ev', 0):.2f}, p90 {cost.get('baseline_cost_p90', 0):.2f}
  - rule saves {cost.get('rule_savings_vs_baseline_ev', 0):+.3f} EUR/MWh vs baseline in EV
  - rule expected regret {cost.get('rule_expected_regret', 0):.3f} EUR/MWh
    (baseline {cost.get('baseline_expected_regret', 0):.3f})
{adaptive_block}
"""


# -- prompts -----------------------------------------------------------------

SYSTEM_PROMPT = """You are a procurement risk analyst writing for a board of \
directors at an EU industrial emitter. You receive STRUCTURED FACTS about a \
hedging recommendation that an automated decision engine has already produced. \
Your job is ONLY to verbalize those facts in clear, sober English.

HARD RULES (do not break):
1. Use ONLY the numbers in the FACTS block. Do NOT invent figures or proposals \
not supported by them.
2. Do NOT recommend a different hedge ratio. The decision is already made.
3. If asked anything beyond the FACTS, reply literally "not in the dataset".
4. No marketing language. No emojis. No bullet symbols beyond plain "-".
5. Treat the backtest trust caveat (MASE / MAPE / trust factor) as a \
first-class signal: any memo that omits it is incomplete.
"""

MEMO_USER_PROMPT = """Write a SHORT board memo (150-200 words) explaining the \
recommendation to a procurement committee. Structure:
- One opening sentence with the recommendation and the spot price.
- Two or three sentences on the per-month band and drift picture (why the \
rule lands where it does).
- One sentence naming the top 2-3 curated drivers that support the view.
- One sentence on the backtest trust caveat (MAPE / MASE / trust factor).
- One closing sentence on the cost-of-waiting trade-off (savings vs p90 risk).

End. Do not add a heading. Plain paragraphs.

FACTS:
{facts}
"""

COUNTER_USER_PROMPT = """Write a SHORT (~120 words) devil's-advocate \
counter-case to the recommendation above. Frame it as: "What would have to be \
true for the OPPOSITE call (i.e. a HIGHER hedge if the rule says LESS, or a \
LOWER hedge if the rule says MORE) to be the better one?"

Use ONLY the provided FACTS. Argue from:
- a driver whose direction might be wrong or under-weighted,
- the upside-tail quantile (q90) being more likely than the model implies,
- or a known backtest weakness (MASE).

End. Plain paragraphs.

FACTS:
{facts}
"""


# -- featherless call --------------------------------------------------------

def _call_featherless(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 600,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        FEATHERLESS_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": FEATHERLESS_UA,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


# -- template fallback ------------------------------------------------------

def _template_memo(bundle: DecisionBundle) -> str:
    s = bundle.summary
    delta_pp = (s.avg_hedge_now - s.avg_baseline) * 100
    direction_word = "less than" if delta_pp < 0 else ("more than" if delta_pp > 0 else "the same as")
    avg_drift = sum(r.drift_pct for r in bundle.hedge_rows) / max(1, len(bundle.hedge_rows))
    avg_tail = sum(r.upside_tail_pct for r in bundle.hedge_rows) / max(1, len(bundle.hedge_rows))
    kept = [c for c in bundle.curated if c.decision != curation.DROPPED][:3]
    driver_names = "; ".join(c.driver.name for c in kept) if kept else "no curated drivers"
    cost = bundle.cost_summary
    sav = cost.get("rule_savings_vs_baseline_ev", 0)
    p90_risk = (
        cost.get("rule_cost_p90", 0) - cost.get("rule_cost_ev", 0)
    )
    rgt = cost.get("rule_expected_regret", 0)
    rgt_base = cost.get("baseline_expected_regret", 0)
    trust_line = ""
    if bundle.trust is not None:
        t = bundle.trust
        mape_str = f"{t.mape*100:.1f}%" if t.mape is not None else "n/a"
        mase_str = f"{t.mase:.2f}" if t.mase is not None else "n/a"
        trust_line = (
            f"Trust caveat: backtest MAPE {mape_str} with MASE {mase_str} drops "
            f"the model's trust factor to {t.trust:.2f} and shrinks the "
            f"effective ACT zone to {t.effective_act_max:.2f} band-width; "
            f"treat the recommendation as directional, not a precise commitment. "
        )

    return (
        f"Procurement hedge recommendation - {bundle.display_name}. "
        f"Spot is {bundle.spot:.2f} {bundle.unit}; the decision engine recommends "
        f"buying {s.avg_hedge_now*100:.1f}% of next-quarter exposure now, "
        f"{abs(delta_pp):.1f} percentage points {direction_word} the 50% naive baseline. "
        f"Across the {len(bundle.hedge_rows)}-month forecast window the median "
        f"sits {avg_drift*100:+.1f}% versus spot, which is the central reason "
        f"the rule leans toward waiting; the q90 upside tail averages "
        f"{avg_tail*100:.1f}% above spot, which is the insurance against "
        f"adverse outcomes built into the hedge ratio. The worst tier in the "
        f"window is {s.weakest_tier}, driven mainly by the highest-uncertainty "
        f"month at the end of the horizon. The supporting drivers, after "
        f"curation removed spurious proxies, are: {driver_names}. "
        f"{trust_line}"
        f"On cost, the rule saves {sav:+.3f} EUR/MWh in expectation versus "
        f"baseline and runs an expected regret of {rgt:.2f} EUR/MWh "
        f"(baseline {rgt_base:.2f}); the p90 cost remains roughly "
        f"{p90_risk:+.2f} EUR/MWh above EV, which is the procurement risk "
        f"the committee accepts in exchange for the EV win."
    )


def _template_counter_case(bundle: DecisionBundle) -> str:
    s = bundle.summary
    delta_pp = (s.avg_hedge_now - s.avg_baseline) * 100
    less_or_more = "less" if delta_pp < 0 else "more"
    opposite = "higher" if less_or_more == "less" else "lower"
    kept = [c for c in bundle.curated if c.decision != curation.DROPPED]
    top = kept[0] if kept else None
    top_name = top.driver.name if top else "no top driver"
    top_dir = (
        f"{top.driver.direction_overall:+.2f}"
        if top
        else "n/a"
    )
    worst_month = max(bundle.hedge_rows, key=lambda r: r.upside_tail_pct)
    mase = bundle.trust.mase if bundle.trust and bundle.trust.mase is not None else None
    mase_line = (
        f"MASE of {mase:.1f} is anomalously high; if the API normalisation "
        f"makes that metric meaningless then the trust grounding may be "
        f"over-correcting and the rule is more reliable than the caveat "
        f"suggests, "
        if mase
        else ""
    )
    return (
        f"What would have to be true for a {opposite} hedge to be the better "
        f"call? Three angles, all from the supplied facts. "
        f"First, the top curated driver ({top_name}, direction {top_dir}) is "
        f"the strongest single voice in the system; if its sign flips - for "
        f"example under a supply shock the cached forecast did not see - the "
        f"median drift becomes positive and the rule mechanically shifts "
        f"toward more-hedge. Second, the {worst_month.date} upside tail "
        f"({worst_month.upside_tail_pct*100:.1f}% above spot) treats q90 as "
        f"a 10% scenario; if real-world tail risk is fatter (geopolitical or "
        f"weather), the regret of waiting is materially larger than the cost "
        f"summary implies. Third, {mase_line}so the model may actually be "
        f"trustworthy and the wait signal is real."
    )


# -- main entry point -------------------------------------------------------

def narrate(
    bundle: DecisionBundle,
    *,
    model: str | None = None,
    api_key: str | None = None,
    use_llm: bool | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Narration:
    """Generate (memo, counter_case) for the given decision.

    Behaviour:
      - If `use_llm` is False, always uses the template (deterministic).
      - If `use_llm` is True, requires a key and uses Featherless.
      - If `use_llm` is None (default): uses Featherless when a key is
        available, otherwise falls back to the template.
    """
    facts = build_facts(bundle)
    key = api_key or _featherless_key()
    notes: list[str] = []

    want_llm = bool(use_llm) if use_llm is not None else (key is not None)
    if want_llm and not key:
        notes.append("use_llm requested but no FEATHERLESS_API_KEY in env")
        want_llm = False

    if not want_llm:
        memo = _template_memo(bundle)
        counter = _template_counter_case(bundle)
        if not notes:
            notes.append("template (no LLM call)")
        return Narration(
            memo=memo, counter_case=counter,
            used_llm=False, model="template",
            facts=facts, notes=tuple(notes),
        )

    mdl = model or _featherless_model()
    memo_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": MEMO_USER_PROMPT.format(facts=facts)},
    ]
    counter_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": COUNTER_USER_PROMPT.format(facts=facts)},
    ]
    try:
        memo = _call_featherless(memo_msgs, model=mdl, api_key=key,
                                  temperature=temperature, max_tokens=400)
        counter = _call_featherless(counter_msgs, model=mdl, api_key=key,
                                     temperature=temperature, max_tokens=350)
        notes.append(f"Featherless model={mdl}")
        return Narration(
            memo=memo, counter_case=counter,
            used_llm=True, model=mdl,
            facts=facts, notes=tuple(notes),
        )
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, KeyError) as exc:
        notes.append(f"Featherless call failed ({type(exc).__name__}: {exc}); "
                     f"falling back to template")
        return Narration(
            memo=_template_memo(bundle),
            counter_case=_template_counter_case(bundle),
            used_llm=False, model="template",
            facts=facts, notes=tuple(notes),
        )


# -- bundle builder ---------------------------------------------------------

def build_bundle_from_cache(cache_dir: Any, *, adaptive: AdaptiveResult | None = None) -> DecisionBundle:
    """Convenience: load a cached forecast + run decision/curation/economics."""
    from pathlib import Path

    from src import config, data
    from src.signals import parse_drivers, parse_forecast_bands

    cdir = Path(cache_dir)
    spec = config.active_ticker()
    repo_root = Path(__file__).resolve().parents[1]
    df = data.load_series(repo_root / spec.csv_path)
    spot = data.current_spot(df)

    forecast = json.loads((cdir / "forecast.json").read_text(encoding="utf-8"))
    bands = parse_forecast_bands(forecast)
    signals = json.loads((cdir / "external_signals.json").read_text(encoding="utf-8"))
    drivers = parse_drivers(signals)
    curated = curation.curate(drivers)

    hedge_rows = decision.decide(bands, spot)
    summary = decision.summarise_next_quarter(hedge_rows, quarter_months=3)

    bt_path = cdir / "backtest_metrics.json"
    backtest = json.loads(bt_path.read_text(encoding="utf-8")) if bt_path.exists() else None
    trust = economics.compute_trust(backtest)

    cost_rows = economics.cost_of_waiting(hedge_rows, bands)
    cost_summary = economics.quarter_summary(cost_rows, months=3)

    return DecisionBundle(
        ticker=spec.symbol,
        display_name=spec.display_name,
        unit=spec.unit,
        spot=spot,
        bands=bands,
        hedge_rows=hedge_rows,
        summary=summary,
        curated=curated,
        backtest=backtest,
        trust=trust,
        cost_rows=cost_rows,
        cost_summary=cost_summary,
        adaptive=adaptive,
    )


# -- CLI --------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    """python -m src.narrator [--llm] [--no-llm] [cache_dir]

    Loads the newest cached TTF forecast, builds the bundle, runs the narrator
    once, prints memo + counter-case.
    """
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description=_cli.__doc__.splitlines()[0])
    p.add_argument("cache_dir", nargs="?", default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--llm", action="store_true",
                   help="force LLM (Featherless); requires FEATHERLESS_API_KEY")
    g.add_argument("--no-llm", action="store_true",
                   help="force the template fallback (no network)")
    p.add_argument("--show-facts", action="store_true",
                   help="also print the facts block fed to the LLM")
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

    bundle = build_bundle_from_cache(cdir)
    use_llm: Optional[bool] = None
    if args.llm:
        use_llm = True
    elif args.no_llm:
        use_llm = False

    out = narrate(bundle, use_llm=use_llm)
    print("=" * 70)
    print(f"MEMO  (model={out.model}, used_llm={out.used_llm})")
    print("=" * 70)
    print(out.memo)
    print()
    print("=" * 70)
    print("COUNTER-CASE")
    print("=" * 70)
    print(out.counter_case)
    if out.notes:
        print()
        for n in out.notes:
            print(f"# {n}")
    if args.show_facts:
        print()
        print("=" * 70)
        print("FACTS (fed to LLM verbatim)")
        print("=" * 70)
        print(out.facts)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
