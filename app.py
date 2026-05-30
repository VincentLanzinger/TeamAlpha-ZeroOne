"""Hedge Decision Cockpit — Streamlit demo.

Run with:
    streamlit run app.py

Architecture
------------
- Forecasts are pre-cached in cache/<hash>/. The cockpit reads them on load.
- /alerts is sync + billable. The 'Trigger shock' button is the ONLY live
  call from the UI; we wrap it in st.spinner.
- /drivers and the LLM (Featherless) are optional toggles. Both are sync.
- The recommendation, curated drivers, cost-of-waiting, and backtest grounding
  are all computed locally from the cached artifacts — they re-render in
  milliseconds on every interaction.

Layout
------
+----------------------------------------------------------------+
|  HEADER: title, spot, cache, shock controls                    |
+---------------------------+------------------------------------+
|  LEFT (wide)              |  RIGHT (narrow)                    |
|  - Recommendation card    |  - Curated driver watchlist        |
|  - Forecast band chart    |  - Curation summary (kept/dropped) |
|  - Per-month hedge table  |  - Backtest grounding              |
|  - Cost-of-waiting table  |                                    |
|  - LLM memo + counter     |                                    |
+---------------------------+------------------------------------+
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import adaptive, config, curation, data, decision, economics, narrator
from src.signals import HorizonBand, parse_drivers, parse_forecast_bands
from src.sybilion_client import has_token

REPO = Path(__file__).resolve().parent
TIER_COLOR = {"ACT": "#1f9d55", "RECOMMEND": "#d69e2e", "ABSTAIN": "#c53030"}


# -- caching -----------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _latest_cache_dir() -> str:
    cands = [
        d for d in (REPO / "cache").iterdir()
        if d.is_dir() and (d / "forecast.json").exists()
    ]
    if not cands:
        raise FileNotFoundError("No cached forecast under cache/. Run scripts/hour_one_gate.py first.")
    return str(max(cands, key=lambda d: d.stat().st_mtime))


@st.cache_data(show_spinner=False)
def _load_bundle(cache_dir: str) -> narrator.DecisionBundle:
    return narrator.build_bundle_from_cache(Path(cache_dir))


@st.cache_data(show_spinner=False)
def _load_cached_drivers(cache_dir: str):
    return parse_drivers(json.loads(
        (Path(cache_dir) / "external_signals.json").read_text(encoding="utf-8")
    ))


# -- rendering helpers -------------------------------------------------------

def _tier_badge(tier: str) -> str:
    color = TIER_COLOR.get(tier, "#666")
    return (
        f"<span style='background:{color}; color:white; "
        f"padding:2px 8px; border-radius:4px; font-weight:600;'>{tier}</span>"
    )


def _band_chart(bundle: narrator.DecisionBundle,
                shocked_bands: list[HorizonBand] | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    dates = [b.date for b in bundle.bands]
    q10 = [b.q10 for b in bundle.bands]
    q50 = [b.q50 for b in bundle.bands]
    q90 = [b.q90 for b in bundle.bands]
    ax.fill_between(dates, q10, q90, alpha=0.20, color="#5078b5", label="q10–q90 (baseline)")
    ax.plot(dates, q50, color="#274171", lw=2, label="median (baseline)")
    if shocked_bands is not None:
        sq10 = [b.q10 for b in shocked_bands]
        sq50 = [b.q50 for b in shocked_bands]
        sq90 = [b.q90 for b in shocked_bands]
        ax.fill_between(dates, sq10, sq90, alpha=0.20, color="#c0392b", label="q10–q90 (shocked)")
        ax.plot(dates, sq50, color="#7a1f15", lw=2, linestyle="--", label="median (shocked)")
    ax.axhline(bundle.spot, color="black", lw=1, linestyle=":", label=f"spot {bundle.spot:.2f}")
    ax.set_ylabel(f"price ({bundle.unit})")
    ax.set_title(f"{bundle.display_name} — forecast bands")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _per_month_df(
    bundle: narrator.DecisionBundle,
    shock: adaptive.AdaptiveResult | None,
) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(bundle.hedge_rows):
        base = {
            "month": r.date,
            "band %": f"{r.band_width*100:.1f}",
            "drift %": f"{r.drift_pct*100:+.1f}",
            "tail %": f"{r.upside_tail_pct*100:.1f}",
            "hedge %": f"{r.hedge_ratio*100:.1f}",
            "tier": r.tier,
        }
        if shock is not None and i < len(shock.shocked_rows):
            sr = shock.shocked_rows[i]
            base.update({
                "new drift %": f"{sr.drift_pct*100:+.1f}",
                "new hedge %": f"{sr.hedge_ratio*100:.1f}",
                "delta pp": f"{(sr.hedge_ratio - r.hedge_ratio)*100:+.1f}",
                "new tier": sr.tier,
            })
        rows.append(base)
    return pd.DataFrame(rows)


def _curated_df(curated: list[curation.CurationDecision], n: int) -> pd.DataFrame:
    kept = [c for c in curated if c.decision != curation.DROPPED][:n]
    rows = []
    for i, c in enumerate(kept, start=1):
        corr = (
            f"{c.driver.correlation:+.2f}" if c.driver.correlation is not None else "—"
        )
        rows.append({
            "#": i,
            "imp": f"{c.adjusted_importance_overall:.1f}",
            "dir": "+" if c.driver.direction_overall > 0.01 else
                   ("-" if c.driver.direction_overall < -0.01 else "0"),
            "corr": corr,
            "category": c.driver.category,
            "name": c.driver.name,
            "tag": c.decision,
        })
    return pd.DataFrame(rows)


def _cost_df(bundle: narrator.DecisionBundle) -> pd.DataFrame:
    s = bundle.cost_summary
    return pd.DataFrame([
        {"strategy": "rule",
         "EV cost (EUR/MWh)": f"{s.get('rule_cost_ev', 0):.2f}",
         "p90 cost (EUR/MWh)": f"{s.get('rule_cost_p90', 0):.2f}",
         "E[regret] (EUR/MWh)": f"{s.get('rule_expected_regret', 0):.2f}"},
        {"strategy": "naive 50%",
         "EV cost (EUR/MWh)": f"{s.get('baseline_cost_ev', 0):.2f}",
         "p90 cost (EUR/MWh)": f"{s.get('baseline_cost_p90', 0):.2f}",
         "E[regret] (EUR/MWh)": f"{s.get('baseline_expected_regret', 0):.2f}"},
        {"strategy": "all now (100%)",
         "EV cost (EUR/MWh)": f"{s.get('all_now_cost', 0):.2f}",
         "p90 cost (EUR/MWh)": f"{s.get('all_now_cost', 0):.2f}",
         "E[regret] (EUR/MWh)": "—"},
        {"strategy": "all wait (0%)",
         "EV cost (EUR/MWh)": f"{s.get('all_wait_cost_ev', 0):.2f}",
         "p90 cost (EUR/MWh)": f"{s.get('all_wait_cost_p90', 0):.2f}",
         "E[regret] (EUR/MWh)": "—"},
    ])


# -- app body ---------------------------------------------------------------

st.set_page_config(
    page_title="Hedge Decision Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar — controls
st.sidebar.title("Controls")
st.sidebar.caption(f"Ticker: **{config.TICKER}** ({config.active_ticker().display_name})")
st.sidebar.caption(f"Sybilion token: {'present' if has_token() else 'MISSING'}")
st.sidebar.caption(f"Featherless: {'present' if narrator.has_llm() else 'no key (template fallback)'}")

st.sidebar.divider()
st.sidebar.subheader("Adaptive shock")
scenario_name = st.sidebar.selectbox("Scenario", list(adaptive.SCENARIOS), index=0)
shock_mode = st.sidebar.radio(
    "Mode",
    ["Live /alerts", "Simulated pressure"],
    index=0,
    help="Live mode calls /alerts (sync, billable). Simulated bypasses the API for "
         "deterministic stage runs.",
)
simulated_pressure = None
if shock_mode == "Simulated pressure":
    simulated_pressure = st.sidebar.slider(
        "Pressure delta", min_value=-0.30, max_value=+0.30, value=0.20, step=0.05,
        help="Positive = bullish shock (hedge MORE). Negative = bearish (hedge LESS).",
    )
sensitivity = st.sidebar.slider(
    "Band-shift sensitivity", 0.10, 1.50, 0.70, 0.10,
    help="Multiplier on delta_pressure when shifting bands.",
)
trigger = st.sidebar.button("⚡ Trigger shock", type="primary", use_container_width=True)
clear = st.sidebar.button("Reset (clear shock)", use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("LLM narration")
narrate_mode = st.sidebar.radio(
    "Narrator", ["Template (deterministic)", "Featherless LLM"], index=0
)
gen_narration = st.sidebar.button("📝 Generate memo + counter-case",
                                    use_container_width=True)

# Main — header
try:
    cache_dir = _latest_cache_dir()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

bundle = _load_bundle(cache_dir)
cached_drivers = _load_cached_drivers(cache_dir)

st.title("Hedge Decision Agent")
st.markdown(
    f"**{bundle.display_name}** &nbsp;•&nbsp; "
    f"spot **{bundle.spot:.2f} {bundle.unit}** &nbsp;•&nbsp; "
    f"cache `{Path(cache_dir).name[:12]}…` &nbsp;•&nbsp; "
    f"{len(bundle.bands)} horizon months, {len(bundle.bands[0].quantiles)} quantile levels",
    unsafe_allow_html=True,
)

# Shock state
if clear:
    st.session_state.pop("shock", None)
if trigger:
    spec = adaptive.SCENARIOS[scenario_name]
    spec = adaptive.ShockScenario(**{**spec.__dict__, "sensitivity": sensitivity})
    if shock_mode == "Live /alerts" and not has_token():
        st.error("Live mode requires SYBILION_API_TOKEN in .env. Switching to simulated.")
    else:
        with st.spinner(
            f"Running {scenario_name} shock "
            f"({'live /alerts' if simulated_pressure is None else 'simulated'})…"
        ):
            try:
                result = adaptive.run_shock(
                    spec,
                    bands=bundle.bands,
                    spot=bundle.spot,
                    cached_drivers=cached_drivers,
                    simulated_pressure=simulated_pressure,
                )
                st.session_state["shock"] = result
                st.session_state.pop("narration", None)  # invalidate stale memo
            except Exception as e:
                st.error(f"Shock failed: {type(e).__name__}: {e}")

shock: adaptive.AdaptiveResult | None = st.session_state.get("shock")

# --- main grid ---
left, right = st.columns([2, 1])

# === LEFT ===
with left:
    st.subheader("Recommendation (next quarter)")
    s = bundle.summary
    if shock is None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Buy now",
                  f"{s.avg_hedge_now*100:.1f}%",
                  delta=f"{(s.avg_hedge_now - s.avg_baseline)*100:+.1f} pp vs naive")
        c2.metric("Wait", f"{(1 - s.avg_hedge_now)*100:.1f}%")
        c3.markdown(f"**Window tier**<br>{_tier_badge(s.weakest_tier)}",
                    unsafe_allow_html=True)
        c4.metric("Trust factor", f"{bundle.trust.trust:.2f}" if bundle.trust else "—")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        new = shock.shocked_summary.avg_hedge_now
        old = shock.baseline_summary.avg_hedge_now
        c1.metric("Buy now",
                  f"{new*100:.1f}%",
                  delta=f"{(new - old)*100:+.1f} pp vs baseline")
        c2.metric("Was", f"{old*100:.1f}%")
        c3.metric("Δ pressure", f"{shock.delta_pressure:+.3f}")
        c4.metric("Band shift", f"{shock.applied_shift_pct*100:+.2f}%")
        c5.markdown(
            f"**Tier**<br>{_tier_badge(shock.baseline_summary.weakest_tier)} → "
            f"{_tier_badge(shock.shocked_summary.weakest_tier)}",
            unsafe_allow_html=True,
        )
        st.info(f"**Scenario:** {shock.scenario.name} — {shock.scenario.description}")

    # Forecast chart
    shocked_bands = None
    if shock is not None:
        shocked_bands = adaptive.shift_bands(
            bundle.bands, shock.delta_pressure, sensitivity=shock.scenario.sensitivity,
        )
    st.pyplot(_band_chart(bundle, shocked_bands), use_container_width=True)

    # Per-month table
    st.subheader("Per-month hedge ratios")
    df = _per_month_df(bundle, shock)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Cost-of-waiting
    st.subheader("Cost of waiting (per MWh, quarter avg)")
    st.dataframe(_cost_df(bundle), use_container_width=True, hide_index=True)
    sav = bundle.cost_summary.get("rule_savings_vs_baseline_ev", 0)
    risk = bundle.cost_summary.get("rule_extra_risk_vs_all_now", 0)
    st.caption(
        f"Rule saves **{sav:+.3f} EUR/MWh** vs naive in EV; "
        f"bears **{risk:+.2f} EUR/MWh** extra worst-case spend vs full-hedge."
    )

    # LLM section
    st.subheader("Board memo + devil's-advocate counter-case")
    if gen_narration:
        use_llm = (narrate_mode == "Featherless LLM")
        with st.spinner(f"Generating narration ({'Featherless' if use_llm else 'template'})…"):
            # Inject shock into bundle if present.
            b = bundle
            if shock is not None:
                b = narrator.DecisionBundle(**{**bundle.__dict__, "adaptive": shock})
            st.session_state["narration"] = narrator.narrate(b, use_llm=use_llm)
    narration: narrator.Narration | None = st.session_state.get("narration")
    if narration is None:
        st.caption("No narration yet — press ‘Generate’ in the sidebar.")
    else:
        st.markdown(f"**Source:** `{narration.model}` &nbsp; (LLM used: "
                     f"{'yes' if narration.used_llm else 'no'})")
        with st.expander("Memo", expanded=True):
            st.write(narration.memo)
        with st.expander("Counter-case", expanded=False):
            st.write(narration.counter_case)
        with st.expander("Facts fed to the LLM (verbatim)", expanded=False):
            st.code(narration.facts, language="text")
        if narration.notes:
            st.caption(" • ".join(narration.notes))

# === RIGHT ===
with right:
    st.subheader("Curated drivers")
    st.caption(
        f"Whitelist + scope curation • "
        f"{sum(1 for c in bundle.curated if c.decision == curation.KEPT)} kept, "
        f"{sum(1 for c in bundle.curated if c.decision == curation.DEMOTED)} demoted, "
        f"{sum(1 for c in bundle.curated if c.decision == curation.DROPPED)} dropped"
    )
    st.dataframe(_curated_df(bundle.curated, 12), use_container_width=True, hide_index=True)

    if shock is not None and shock.alerts_surfaced:
        st.subheader("Live alerts surfaced")
        alert_rows = []
        for a in shock.alerts_surfaced[:8]:
            alert_rows.append({
                "pct_change": f"{float(a.get('pct_change', 0)):+.2f}%",
                "name": a.get("name", "?"),
                "news": len(a.get("news") or []),
            })
        st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, hide_index=True)

    st.subheader("Backtest grounding")
    t = bundle.trust
    if t is None:
        st.write("(no backtest)")
    else:
        c1, c2 = st.columns(2)
        c1.metric("MAPE", f"{t.mape*100:.1f}%" if t.mape is not None else "—")
        c2.metric("MASE", f"{t.mase:.2f}" if t.mase is not None else "—")
        c1.metric("Trust factor", f"{t.trust:.2f}")
        c2.metric("Shift", f"{t.threshold_shift:.3f}")
        st.caption(
            f"Effective ACT_MAX width **{t.effective_act_max:.3f}** "
            f"(was {decision.ACT_MAX_WIDTH}), RECOMMEND_MAX **{t.effective_recommend_max:.3f}** "
            f"(was {decision.RECOMMEND_MAX_WIDTH}). Threshold shrinkage widens the abstain zone "
            f"when backtest is poor."
        )
        if t.notes:
            for n in t.notes:
                st.caption(f"• {n}")

    st.subheader("Top dropped (the noise we cut)")
    dropped = [c for c in bundle.curated if c.decision == curation.DROPPED]
    dropped_top = sorted(dropped, key=lambda c: c.driver.importance_overall, reverse=True)[:5]
    st.dataframe(
        pd.DataFrame([
            {"raw imp": f"{c.driver.importance_overall:.1f}",
             "category": c.driver.category,
             "name": c.driver.name}
            for c in dropped_top
        ]),
        use_container_width=True, hide_index=True,
    )

# Footer
st.divider()
st.caption(
    f"Cache: {Path(cache_dir).name[:24]}…  •  "
    f"Phases 0–8 wired  •  "
    f"Forecasts pre-cached; live path uses only /alerts and /drivers (sync)."
)
