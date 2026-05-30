"""Hedge Decision Cockpit — Streamlit demo, designed for non-specialist users.

Run with:
    streamlit run app.py

UX principles
-------------
1. LEAD WITH THE DECISION in plain English ("Buy 42% now, wait on 58%").
2. EXPLAIN THE WHY — drift, signals, accuracy caveat — in human language.
3. PROGRESSIVE DISCLOSURE — hide quantile / MAPE / importance behind toggles.
4. CONVERT TO EUR — every per-MWh number gets a volume-aware EUR rollup.
5. TOOLTIPS FOR JARGON — every technical term has an explainer.

Underlying logic is unchanged from Phases 0-7; only the rendering changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import adaptive, config, curation, decision, narrator
from src.signals import HorizonBand, parse_drivers
from src.sybilion_client import has_token

REPO = Path(__file__).resolve().parent

# Plain-English tier labels + colours
TIER_LABEL = {
    "ACT":       "High confidence",
    "RECOMMEND": "Moderate confidence",
    "ABSTAIN":   "Too uncertain",
}
TIER_DOT = {
    "ACT":       "🟢",
    "RECOMMEND": "🟡",
    "ABSTAIN":   "🔴",
}
TIER_COLOR = {
    "ACT":       "#1f9d55",
    "RECOMMEND": "#d69e2e",
    "ABSTAIN":   "#c53030",
}


# -- caching -----------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _latest_cache_dir() -> str:
    cands = [
        d for d in (REPO / "cache").iterdir()
        if d.is_dir() and (d / "forecast.json").exists()
    ]
    if not cands:
        raise FileNotFoundError(
            "No cached forecast under cache/. Run scripts/hour_one_gate.py first."
        )
    return str(max(cands, key=lambda d: d.stat().st_mtime))


@st.cache_data(show_spinner=False)
def _load_bundle(cache_dir: str) -> narrator.DecisionBundle:
    return narrator.build_bundle_from_cache(Path(cache_dir))


@st.cache_data(show_spinner=False)
def _load_cached_drivers(cache_dir: str):
    return parse_drivers(json.loads(
        (Path(cache_dir) / "external_signals.json").read_text(encoding="utf-8")
    ))


# -- helpers -----------------------------------------------------------------

def fmt_eur(x: float) -> str:
    """Compact EUR formatter: 12_870_000 -> '€12.87M'."""
    if abs(x) >= 1_000_000:
        return f"€{x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"€{x / 1_000:.0f}k"
    return f"€{x:.0f}"


def fmt_pct(x: float, signed: bool = False) -> str:
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def tier_badge_md(tier: str) -> str:
    return f"{TIER_DOT.get(tier, '⚪')} **{TIER_LABEL.get(tier, tier)}**"


def hedge_bar_html(hedge: float, baseline: float = 0.50) -> str:
    """A simple proportional bar showing the hedge ratio with a baseline marker."""
    pct = max(0.0, min(1.0, hedge))
    color = "#1f9d55" if pct < 0.45 else ("#d69e2e" if pct < 0.65 else "#c53030")
    baseline_pos = int(baseline * 100)
    return f"""
<div style='background:#222; border-radius:8px; padding:4px;
            position:relative; height:36px; margin:8px 0;'>
  <div style='background:{color}; height:28px; width:{pct*100:.1f}%;
              border-radius:4px;'></div>
  <div style='position:absolute; top:0; left:{baseline_pos}%;
              border-left:2px dashed #888; height:36px;'></div>
  <div style='position:absolute; top:38px; left:{baseline_pos}%;
              font-size:11px; color:#aaa; transform:translateX(-50%);'>
       naive {baseline*100:.0f}%</div>
</div>
"""


def confidence_phrase(trust: float | None) -> str:
    if trust is None:
        return "no track record on file"
    if trust >= 0.75:
        return "the model has been historically reliable"
    if trust >= 0.4:
        return "the model has a mixed track record"
    return "the model's track record is weak — treat as directional"


def driver_phrase(direction: float, name: str) -> str:
    arrow = "↑" if direction > 0.01 else ("↓" if direction < -0.01 else "→")
    plain = name.replace(" - ", " in ")
    return f"{arrow} {plain}"


# -- charts ------------------------------------------------------------------

def forecast_chart(bundle: narrator.DecisionBundle,
                    shocked_bands: list[HorizonBand] | None,
                    show_band_labels: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    dates = [b.date for b in bundle.bands]
    q10 = [b.q10 for b in bundle.bands]
    q50 = [b.q50 for b in bundle.bands]
    q90 = [b.q90 for b in bundle.bands]

    ax.fill_between(dates, q10, q90, alpha=0.20, color="#4178c0",
                    label="Where prices land 80% of the time")
    ax.plot(dates, q50, color="#274171", lw=2.5,
            label="Most-likely price")
    if shocked_bands is not None:
        sq10 = [b.q10 for b in shocked_bands]
        sq50 = [b.q50 for b in shocked_bands]
        sq90 = [b.q90 for b in shocked_bands]
        ax.fill_between(dates, sq10, sq90, alpha=0.18, color="#c0392b",
                        label="After the shock")
        ax.plot(dates, sq50, color="#7a1f15", lw=2.0, linestyle="--",
                label="Most-likely (shocked)")

    ax.axhline(bundle.spot, color="#dddddd", lw=1, linestyle=":",
                label=f"Today €{bundle.spot:.2f}")
    ax.set_ylabel(f"Price (€/{bundle.unit.split('/')[-1].strip()})")
    ax.set_xlabel("Month")
    ax.set_title("Next 6 months — forecast vs today's price")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.20)
    fig.tight_layout()
    return fig


# -- per-month table (plain-English) -----------------------------------------

def per_month_friendly_df(
    bundle: narrator.DecisionBundle,
    shock: adaptive.AdaptiveResult | None,
) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(bundle.hedge_rows):
        # Pretty month
        month = pd.Timestamp(r.date).strftime("%b %Y")
        expected = f"€{r.q50:.2f}  ({r.drift_pct*100:+.0f}% vs today)"
        confidence = (
            "●●●●○" if r.tier == "ACT" else
            "●●●○○" if r.tier == "RECOMMEND" else
            "●○○○○"
        )
        row = {
            "Month": month,
            "Expected price": expected,
            "Confidence": f"{confidence}  {TIER_LABEL[r.tier]}",
            "Buy now": f"{r.hedge_ratio*100:.0f}%",
        }
        if shock is not None and i < len(shock.shocked_rows):
            sr = shock.shocked_rows[i]
            delta_pp = (sr.hedge_ratio - r.hedge_ratio) * 100
            row.update({
                "After shock": f"{sr.hedge_ratio*100:.0f}%",
                "Δ": f"{delta_pp:+.0f} pp",
            })
        rows.append(row)
    return pd.DataFrame(rows)


# -- cost translation -------------------------------------------------------

def eur_rollup(bundle: narrator.DecisionBundle, monthly_mwh: float, months: int = 3) -> dict:
    s = bundle.cost_summary
    v = monthly_mwh * months
    return {
        "volume_mwh": v,
        "rule_ev":         v * s.get("rule_cost_ev", 0),
        "rule_p90":        v * s.get("rule_cost_p90", 0),
        "baseline_ev":     v * s.get("baseline_cost_ev", 0),
        "baseline_p90":    v * s.get("baseline_cost_p90", 0),
        "all_now":         v * s.get("all_now_cost", 0),
        "all_wait_ev":     v * s.get("all_wait_cost_ev", 0),
        "all_wait_p90":    v * s.get("all_wait_cost_p90", 0),
        "savings_vs_base": v * s.get("rule_savings_vs_baseline_ev", 0),
        "savings_vs_now":  v * s.get("rule_savings_vs_all_now_ev", 0),
        "risk_vs_now":     v * s.get("rule_extra_risk_vs_all_now", 0),
    }


# ----------------------------------------------------------------------------
#   PAGE
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Hedge Decision Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject some CSS for cleaner cards
st.markdown("""
<style>
.hedge-card { background:#1a1f2e; border:1px solid #2c3344;
              border-radius:10px; padding:18px 22px; margin:8px 0; }
.muted { color:#9aa3b0; font-size:13px; }
.eur-big { font-size:28px; font-weight:700; }
.pill   { display:inline-block; padding:2px 10px; border-radius:12px;
          font-size:12px; font-weight:600; color:white; }
</style>
""", unsafe_allow_html=True)

try:
    cache_dir = _latest_cache_dir()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

bundle = _load_bundle(cache_dir)
cached_drivers = _load_cached_drivers(cache_dir)

# ============================== SIDEBAR =====================================

with st.sidebar:
    st.title("Controls")
    st.caption(
        f"Asset: **{bundle.display_name}**  \n"
        f"Source: cached forecast `{Path(cache_dir).name[:8]}…`"
    )
    st.caption(
        f"Sybilion: {'connected' if has_token() else '**missing token**'}  \n"
        f"LLM narrator: {'Featherless' if narrator.has_llm() else 'template fallback'}"
    )

    st.divider()
    st.subheader("Your volume")
    monthly_mwh = st.number_input(
        "Gas you buy per month (MWh)",
        min_value=1_000, max_value=10_000_000, value=100_000, step=10_000,
        help="Used to translate per-MWh figures into EUR totals across the next quarter."
    )

    st.divider()
    st.subheader("Simulate a market event")
    scenario_name = st.selectbox(
        "Scenario",
        list(adaptive.SCENARIOS),
        format_func=lambda n: adaptive.SCENARIOS[n].description,
        help="Pick a hypothetical news event. The system pulls fresh market signals "
             "under this scenario and recomputes the recommendation."
    )
    shock_mode = st.radio(
        "How to react",
        ["Live: use today's real news + market data",
         "Simulated: assume a controlled shock magnitude"],
        index=1,
        help="Live pulls the Sybilion /alerts endpoint (sync, billable). Simulated "
             "bypasses the API and applies a clean pressure value — useful on stage."
    )
    simulated_pressure = None
    if shock_mode.startswith("Simulated"):
        simulated_pressure = st.slider(
            "Event severity (signed)",
            min_value=-0.30, max_value=+0.30, value=+0.20, step=0.05,
            help="Positive = bullish (prices go up, hedge MORE). "
                 "Negative = bearish (prices fall, hedge LESS)."
        )
    sensitivity = st.slider(
        "How much to react to the event",
        min_value=0.10, max_value=1.50, value=0.70, step=0.10,
        help="Multiplier on the pressure when shifting the forecast bands."
    )
    c1, c2 = st.columns(2)
    with c1:
        trigger = st.button("⚡ Run scenario", type="primary", use_container_width=True)
    with c2:
        clear = st.button("Reset", use_container_width=True)

    st.divider()
    st.subheader("Narrative")
    narrate_mode = st.radio(
        "Generated by",
        ["Template (no LLM, instant)", "Featherless LLM (1-2s, more natural)"],
        index=0,
    )
    gen_narration = st.button("📝 Generate memo", use_container_width=True)

    st.divider()
    show_advanced = st.toggle(
        "Show advanced details",
        value=False,
        help="Show raw quantile bands, importance scores, full backtest metrics."
    )

# ============================== STATE =======================================

if clear:
    st.session_state.pop("shock", None)
    st.session_state.pop("narration", None)
if trigger:
    spec = adaptive.SCENARIOS[scenario_name]
    spec = adaptive.ShockScenario(**{**spec.__dict__, "sensitivity": sensitivity})
    if shock_mode.startswith("Live") and not has_token():
        st.error("Live mode needs SYBILION_API_TOKEN in .env. Switch to Simulated.")
    else:
        with st.spinner(f"Running '{scenario_name}'…"):
            try:
                result = adaptive.run_shock(
                    spec,
                    bands=bundle.bands, spot=bundle.spot,
                    cached_drivers=cached_drivers,
                    simulated_pressure=simulated_pressure,
                )
                st.session_state["shock"] = result
                st.session_state.pop("narration", None)
            except Exception as e:
                st.error(f"Scenario failed: {type(e).__name__}: {e}")

shock: adaptive.AdaptiveResult | None = st.session_state.get("shock")
active_summary = shock.shocked_summary if shock else bundle.summary
active_rows = shock.shocked_rows if shock else bundle.hedge_rows

# ============================== HEADER ======================================

st.title("Hedge Decision Agent")
st.markdown(
    f"<div class='muted'>Helps a procurement team decide what share of next "
    f"quarter's <strong>{bundle.display_name.lower()}</strong> to buy now "
    f"versus wait for. Built on the Sybilion forecasting API; the decision "
    f"engine + driver curation + backtest grounding are ours.</div>",
    unsafe_allow_html=True,
)
st.markdown("")

# ============================== HERO: THE RECOMMENDATION ====================

hedge = active_summary.avg_hedge_now
baseline = active_summary.avg_baseline
delta_pp = (hedge - baseline) * 100
tier = active_summary.weakest_tier

c_left, c_right = st.columns([2, 1])

with c_left:
    st.markdown("<div class='hedge-card'>", unsafe_allow_html=True)
    st.markdown("### The recommendation")
    if shock:
        st.markdown(
            f"<div style='font-size:22px; line-height:1.4'>Under the "
            f"<em>{shock.scenario.name}</em> scenario, "
            f"<strong>buy {hedge*100:.0f}% now</strong>, "
            f"wait on {(1-hedge)*100:.0f}%.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='font-size:22px; line-height:1.4'>"
            f"<strong>Buy {hedge*100:.0f}% now</strong>, "
            f"wait on {(1-hedge)*100:.0f}% — and watch the market.</div>",
            unsafe_allow_html=True,
        )
    st.markdown(hedge_bar_html(hedge, baseline=baseline), unsafe_allow_html=True)
    direction_word = "less" if delta_pp < 0 else ("more" if delta_pp > 0 else "the same as")
    st.markdown(
        f"That's **{abs(delta_pp):.0f} percentage points {direction_word} "
        f"than a 50/50 baseline**.  Confidence in this call: {tier_badge_md(tier)}.",
        unsafe_allow_html=True,
    )
    if shock:
        old_hedge = shock.baseline_summary.avg_hedge_now
        st.info(
            f"Before the scenario: **{old_hedge*100:.0f}%** buy now.  "
            f"After: **{hedge*100:.0f}%** ({(hedge-old_hedge)*100:+.0f} pp).  "
            f"The system shifted because the scenario implies "
            f"{'higher' if hedge > old_hedge else 'lower'} expected prices."
        )
    st.markdown("</div>", unsafe_allow_html=True)

with c_right:
    st.markdown("<div class='hedge-card'>", unsafe_allow_html=True)
    st.markdown("### Today's price")
    st.markdown(
        f"<div class='eur-big'>€{bundle.spot:.2f}<span class='muted'>/MWh</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='muted'>{bundle.display_name}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        f"<div class='muted'>Forecast horizon: 6 months "
        f"({pd.Timestamp(bundle.bands[0].date).strftime('%b %Y')} – "
        f"{pd.Timestamp(bundle.bands[-1].date).strftime('%b %Y')})</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ============================== EUR ROLLUP ==================================

st.markdown("### What this means in EUR")
roll = eur_rollup(bundle, monthly_mwh, months=3)
st.caption(
    f"Over the next quarter you'll buy "
    f"**{roll['volume_mwh']:,.0f} MWh** "
    f"(3 months × {monthly_mwh:,.0f} MWh/month)."
)

c1, c2, c3 = st.columns(3)
c1.metric(
    "Our recommendation — likely cost",
    fmt_eur(roll["rule_ev"]),
    help="Expected total cost over the next quarter under our hedge ratio. "
         "Computed as h × today's price + (1-h) × forecast median, summed."
)
c2.metric(
    "Naive 50/50 — likely cost",
    fmt_eur(roll["baseline_ev"]),
    delta=fmt_eur(-roll["savings_vs_base"]),
    delta_color="inverse",
    help="What you'd pay if you ignored the model and hedged exactly half. "
         "Negative delta means our rule is cheaper."
)
c3.metric(
    "Bad case (10% chance)",
    fmt_eur(roll["rule_p90"]),
    delta=fmt_eur(roll["rule_p90"] - roll["rule_ev"]),
    delta_color="inverse",
    help="90th-percentile total — what you pay if prices land at q90. "
         "Delta is the risk premium vs the likely case."
)

if roll["savings_vs_base"] != 0:
    sign = "+" if roll["savings_vs_base"] > 0 else ""
    st.success(
        f"**Vs naive 50/50:** {sign}{fmt_eur(roll['savings_vs_base'])} "
        f"saved in expectation across this quarter.  "
        f"**Vs locking it all in now:** "
        f"{'+' if roll['savings_vs_now']>0 else ''}{fmt_eur(roll['savings_vs_now'])} "
        f"saved in expectation, with **{fmt_eur(abs(roll['risk_vs_now']))} extra "
        f"worst-case spend**."
    )

st.divider()

# ============================== WHY ========================================

st.markdown("### Why this recommendation?")

w1, w2 = st.columns(2)
with w1:
    avg_drift = sum(r.drift_pct for r in active_rows) / max(1, len(active_rows))
    drift_word = "fall" if avg_drift < -0.02 else ("rise" if avg_drift > 0.02 else "stay flat")
    st.markdown("#### 📊 Where prices are headed")
    st.markdown(
        f"Over the next 6 months the model expects prices to **{drift_word}** "
        f"by **{abs(avg_drift)*100:.0f}% on average** versus today's €{bundle.spot:.2f}."
    )
    st.markdown(
        f"That's the central reason we're "
        f"{'below' if hedge < baseline else 'above' if hedge > baseline else 'at'} "
        f"the 50/50 baseline — waiting "
        f"{'saves' if avg_drift < 0 else 'costs'} money in expectation."
    )

with w2:
    st.markdown("#### 📡 Which signals matter most")
    kept = [c for c in bundle.curated if c.decision != curation.DROPPED][:3]
    if kept:
        for c in kept:
            st.markdown(f"- {driver_phrase(c.driver.direction_overall, c.driver.name)}")
    n_dropped = sum(1 for c in bundle.curated if c.decision == curation.DROPPED)
    st.markdown(
        f"<div class='muted'>(We dropped {n_dropped} weak signals that scored "
        f"high by coincidence — e.g. country-level population data unrelated "
        f"to gas prices.)</div>",
        unsafe_allow_html=True,
    )

st.markdown("#### ⚠ Trust caveat — how good has the model been historically?")
if bundle.trust is not None:
    t = bundle.trust
    if t.mape is not None:
        mape_str = (
            f"On recent forecasts the model was off by about "
            f"**{t.mape*100:.0f}% on average** "
            f"({'reasonable' if t.mape < 0.20 else 'high'} for monthly prices)."
        )
    else:
        mape_str = "Track record unavailable."
    mase_str = ""
    if t.mase is not None and t.mase > 10:
        mase_str = (
            f"  Another error metric (MASE = {t.mase:.0f}) looks anomalously bad — "
            f"the model is much worse than a simple seasonal baseline on that scale. "
        )
    st.markdown(
        f"{mape_str}{mase_str}"
        f"Because of this, we widen the **'too uncertain'** zone — "
        f"{confidence_phrase(t.trust)}, so we treat the recommendation as "
        f"**directional**, not a precise commitment."
    )
else:
    st.markdown("No historical accuracy data on file.")

st.divider()

# ============================== FORECAST CHART ==============================

st.markdown("### Forecast — next 6 months")
shocked_bands = None
if shock is not None:
    shocked_bands = adaptive.shift_bands(
        bundle.bands, shock.delta_pressure, sensitivity=shock.scenario.sensitivity,
    )
st.pyplot(forecast_chart(bundle, shocked_bands, show_advanced), use_container_width=True)
st.caption(
    "**What you're seeing:** The line in the middle is the most-likely price "
    "each month. The shaded area is the range where prices land roughly 80% "
    "of the time. The dotted line is today's price for comparison."
    + (
        "  After the scenario, the red bands sit above the blue ones — the "
        "shock pushes prices higher."
        if shock is not None and shock.applied_shift_pct > 0
        else "  After the scenario, the red bands sit below the blue ones — the "
             "shock pushes prices lower." if shock is not None else ""
      )
)

st.divider()

# ============================== MONTH-BY-MONTH ==============================

st.markdown("### Month-by-month detail")
df = per_month_friendly_df(bundle, shock)
st.dataframe(df, use_container_width=True, hide_index=True)
st.caption(
    "Confidence dots: more = tighter forecast band = stronger conviction. "
    "When the band is too wide we mark the month **Too uncertain** and lean "
    "back toward the 50/50 baseline."
)

st.divider()

# ============================== NARRATIVE ===================================

st.markdown("### Executive memo")

if gen_narration:
    use_llm = ("Featherless" in narrate_mode)
    with st.spinner("Generating…"):
        b = bundle if shock is None else narrator.DecisionBundle(
            **{**bundle.__dict__, "adaptive": shock}
        )
        st.session_state["narration"] = narrator.narrate(b, use_llm=use_llm)

narration: narrator.Narration | None = st.session_state.get("narration")
if narration is None:
    st.caption("No memo yet — press *Generate memo* in the sidebar.")
else:
    src = "LLM (Featherless)" if narration.used_llm else "Template"
    st.caption(
        f"Generated by: **{src}** &nbsp;•&nbsp; "
        f"This memo only uses the numbers from this page. It never proposes "
        f"a different recommendation than the engine."
    )
    with st.expander("📄 Board memo", expanded=True):
        st.write(narration.memo)
    with st.expander("🛑 Devil's-advocate: what could be wrong?", expanded=False):
        st.write(narration.counter_case)
    if show_advanced:
        with st.expander("🔬 Raw facts fed to the LLM", expanded=False):
            st.code(narration.facts, language="text")

st.divider()

# ============================== ADVANCED ====================================

if show_advanced:
    st.markdown("## Advanced details")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("#### Top kept drivers (after curation)")
        kept_rows = []
        for c in [x for x in bundle.curated if x.decision != curation.DROPPED][:12]:
            kept_rows.append({
                "name": c.driver.name,
                "imp (adj)": f"{c.adjusted_importance_overall:.1f}",
                "amp": (f"x{c.amplification_factor:.2f}"
                        if c.amplified else "—"),
                "matched": ", ".join(c.matched_tokens) if c.matched_tokens else "—",
                "direction": ("↑" if c.driver.direction_overall > 0.01
                              else ("↓" if c.driver.direction_overall < -0.01 else "→")),
                "corr": (f"{c.driver.correlation:+.2f}"
                         if c.driver.correlation is not None else "—"),
                "tag": c.decision,
            })
        st.dataframe(pd.DataFrame(kept_rows), use_container_width=True, hide_index=True)
        n_kept = sum(1 for c in bundle.curated if c.decision == curation.KEPT)
        n_dem = sum(1 for c in bundle.curated if c.decision == curation.DEMOTED)
        n_drp = sum(1 for c in bundle.curated if c.decision == curation.DROPPED)
        n_amp = sum(1 for c in bundle.curated if c.amplified and c.decision != curation.DROPPED)
        st.caption(
            f"Curation: **{n_kept}** kept, **{n_dem}** demoted, "
            f"**{n_drp}** dropped, **{n_amp}** amplified by keyword match. "
            f"`amp` shows the multiplier, `matched` shows which token(s) fired the boost."
        )

    with a2:
        st.markdown("#### Top dropped (the noise we cut)")
        dropped = sorted(
            [c for c in bundle.curated if c.decision == curation.DROPPED],
            key=lambda c: c.driver.importance_overall, reverse=True,
        )[:8]
        st.dataframe(
            pd.DataFrame([
                {"raw imp": f"{c.driver.importance_overall:.1f}",
                 "category": c.driver.category,
                 "name": c.driver.name}
                for c in dropped
            ]),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "These scored high purely by trend coincidence — "
            "population time-series of unrelated countries, etc."
        )

    st.markdown("#### Backtest grounding (raw)")
    if bundle.trust is not None:
        t = bundle.trust
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("MAPE", f"{t.mape*100:.1f}%" if t.mape is not None else "—",
                    help="Mean absolute percentage error on rolling backtest.")
        bc2.metric("MASE", f"{t.mase:.2f}" if t.mase is not None else "—",
                    help="Mean absolute scaled error vs seasonal-naive baseline. "
                         "1 = equal to naive; >1 = worse than naive.")
        bc3.metric("Trust factor", f"{t.trust:.2f}",
                    help="Composite score, 1.0 = full trust, 0 = ignore model.")
        bc4.metric("Threshold shift", f"{t.threshold_shift:.3f}",
                    help="How much we shrink the confident zones when trust is low.")
        st.caption(
            f"Effective ACT band-width = {t.effective_act_max:.3f} "
            f"(default {decision.ACT_MAX_WIDTH}), RECOMMEND = "
            f"{t.effective_recommend_max:.3f} (default "
            f"{decision.RECOMMEND_MAX_WIDTH})."
        )
        if t.notes:
            for n in t.notes:
                st.caption(f"• {n}")

    st.markdown("#### Raw decision rows")
    raw_rows = []
    for r in active_rows:
        raw_rows.append({
            "date": r.date,
            "q10": r.q10, "q50": r.q50, "q90": r.q90,
            "band_width": r.band_width,
            "drift": r.drift_pct, "upside_tail": r.upside_tail_pct,
            "hedge_ratio": r.hedge_ratio, "tier": r.tier,
        })
    st.dataframe(pd.DataFrame(raw_rows), use_container_width=True, hide_index=True)

    if shock is not None:
        st.markdown("#### Live alerts surfaced under the scenario")
        if shock.alerts_surfaced:
            st.dataframe(
                pd.DataFrame([
                    {"pct_change": f"{float(a.get('pct_change', 0)):+.2f}%",
                     "name": a.get("name", "?"),
                     "news_items": len(a.get("news") or [])}
                    for a in shock.alerts_surfaced[:10]
                ]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No live alerts surfaced (simulated mode or empty result).")
        st.caption(
            f"Baseline pressure: {shock.baseline_pressure.pressure:+.3f}  •  "
            f"Shocked: {shock.shocked_pressure.pressure:+.3f}  •  "
            f"Delta: {shock.delta_pressure:+.3f}  •  "
            f"Band shift: {shock.applied_shift_pct*100:+.2f}%"
        )

# Footer
st.divider()
st.caption(
    f"Cache: `{Path(cache_dir).name[:24]}…`  •  "
    f"Forecasts pre-cached, live path uses only sync `/alerts` and `/drivers`.  •  "
    f"Built for the Zero One Hack."
)
