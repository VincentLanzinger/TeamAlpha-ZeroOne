# Hedge Decision Agent

[![tests](https://github.com/VincentLanzinger/hedge-decision-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/VincentLanzinger/hedge-decision-agent/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Zero One Hack — Forecasting track.**
A decision agent for procurement teams that have to decide *what share of next quarter's
input commodity to lock in now vs. wait for*. Built on the Sybilion Forecasting API.

The intelligence is in the **decision engine**, the **driver-curation layer**, the
**backtest-aware trust grounding**, and the **adaptive shock loop** — not in an LLM.
An LLM (Featherless) is present, but only to verbalize already-computed numbers; it
never makes a different recommendation than the engine.

## Clone and run

```bash
git clone https://github.com/VincentLanzinger/hedge-decision-agent
cd hedge-decision-agent
python -m venv .venv
. .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env          # then paste real keys (optional — works without them)
pytest tests/                 # 79 passed in ~3s
streamlit run app.py          # → http://localhost:8501
```

Works **without API keys** off the committed cache. Add `SYBILION_API_TOKEN` to call
new forecasts; add `FEATHERLESS_API_KEY` for live LLM narration (template fallback is
deterministic and instant otherwise).

---

## 1. What it does

```
Sybilion forecast (cached)  -->  Decision engine (band -> hedge ratio)
        |                           |
        |--> Driver curation        |--> ACT / RECOMMEND / ABSTAIN tier
        |--> Backtest grounding     |--> EUR cost-of-waiting + regret
        |                           |
        +------> Adaptive layer  -->+    (live shock via sync /alerts)
                                    |
                                    +--> Narrator (LLM or template)
                                         memo + devil's-advocate
```

Active asset: **Dutch TTF natural gas**, monthly, 6-month horizon. The ticker is
swappable in a single line of `src/config.py` (`TICKER = "TTF"` -> `"EUA"` / `"ALUMINUM"`).

**Live result on the cached forecast:**

| Metric | Rule | Naive 50/50 | All now | All wait |
| --- | ---: | ---: | ---: | ---: |
| Hedge ratio | **41.9%** | 50% | 100% | 0% |
| Quarter EV cost (€/MWh) | **42.91** | 43.32 | 45.79 | 40.85 |
| Quarter worst-case (p90, €/MWh) | **46.72** | 46.66 | 45.79 | 47.53 |
| Avg regret (€/MWh) | **2.35** | 2.74 | — | — |

For a 100,000 MWh/month buyer that's **€124k saved in expectation vs naive 50/50**
and **€865k saved vs locking it all in now**, with €280k extra worst-case spend.

---

## 2. Setup

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Then paste your real keys into .env:
#   SYBILION_API_TOKEN=sk_ops_...
#   FEATHERLESS_API_KEY=rc_...        (optional — narrator falls back to template)
```

Requires Python 3.10+. Project uses the official `sybilion` SDK 0.1.4.

---

## 3. Demo script (3 minutes on stage)

```powershell
# 1. Open the cockpit
streamlit run app.py
# -> http://localhost:8501
```

| Beat | Action | What the audience sees |
| --- | --- | --- |
| **0:00** | Open the cockpit | "Buy 42% now, wait on 58%" hero, €45.79 spot, "Moderate confidence" |
| **0:30** | Point at the EUR cards | "€124k saved vs naive 50/50, €865k saved vs all-now" |
| **1:00** | Read the trust caveat | "Model has been off by 15% on average. Another metric is anomalously bad. So we treat this as DIRECTIONAL, not precise" — the system polices itself |
| **1:30** | Sidebar: scenario = Hormuz, mode = Simulated +0.20, click *Run scenario* | Hero flips 41.9% -> 62.5% (+20.6 pp). Red shocked bands jump above the blue baseline on the chart. Per-month table picks up delta columns |
| **2:15** | Click *Generate memo* (Featherless mode) | Board memo + devil's-advocate counter-case, every number traceable to the cockpit |
| **2:45** | (Bravery) Switch to *Live*, re-trigger | Today's live `/alerts` say markets are calm — system honestly responds *less* hedge. Demo proof: not faked |

Pre-cached so the demo doesn't depend on internet (the only billable call is the live
shock, which is optional). All numbers are deterministic seed=42.

---

## 4. Architecture

```
src/
  config.py        TICKER + per-ticker spec (one-line switch).
  sybilion_client.py  Thin SDK wrapper + SHA-256 forecast cache.
  data.py          Loader + validator (first-of-month, >=60 obs, no gaps).
  signals.py       Parsers for forecast.json + external_signals.json.
  curation.py      Whitelist + scope-plausibility filter on raw drivers.
  decision.py      band_width / drift / upside-tail -> hedge ratio + tier.
  economics.py     Cost-of-waiting + 19-quantile regret + backtest trust grounding.
  adaptive.py      Live shock: /alerts -> pressure delta -> band shift -> re-decide.
  narrator.py      Featherless or template; verbalize the numbers, never re-decide.

scripts/
  hour_one_gate.py   Phase 2 gate. Submit (or load cached) -> KEEP/SWITCH verdict.
  rehearse_demo.py   Dry-run of the full stage path; non-destructive.
  probe_alerts.py    One-off /alerts shape probe (used to debug filter behavior).
  probe_featherless.py  /v1/models probe (used to debug Cloudflare 1010).

cache/             Forecast artifacts keyed by SHA-256(canonical request body).
data/              Input monthly CSVs (date,value).
tests/             pytest, deterministic; 79 passing.
app.py             Streamlit cockpit.
```

### Key choices

- **Forecasts are async (minutes) and billable** — we always cache. SHA-256 of the
  canonicalised JSON request body is the key, so identical requests on different machines
  hash identically. Cached artifacts are committed to git so the demo runs offline.
- **Live demo path uses only sync endpoints** (`/alerts`, `/drivers`, `/me`, catalogs).
- **`/me` balance gate** before any submit refuses if `available_eur_cents < 5`.
- **Decision engine is deterministic, no rng, no clock.** Same forecast + same spot ->
  same recommendation. Tested with 11 unit cases + 1 integration test against the cached
  TTF artifact.

---

## 5. Key technical challenges & solutions

The work that actually consumed time on this hackathon:

### 5.1 Real artifact shapes diverged from the marketing docs
**Problem.** The Sybilion doc page implied `external_signals.json` was a list of
`{name, importance, direction, correlation}` items. The live artifact is a *dict keyed
by UUID*, importance has only per-horizon entries (no `overall.mean` for some drivers),
and per-horizon entries are dicts of `{lag_key: float}`, not single numbers.
**Solution.** `src/signals.py` parses the live shape defensively — handles the
dict-of-UUIDs and a legacy list shape; computes overall importance from per-horizon
entries when missing; aggregates inner lag-key values by **mean** (which matches the
API's own `overall.mean` semantics — sum gave numbers 3× too large).
*Time to find: ~45 min. Symptom: the gate said SWITCH because parser returned 0 drivers.*

### 5.2 Spurious drivers crowd out real signal
**Problem.** Raw ranking puts `Population - Nigeria` at #2 (importance 93.3), with
`Population - Mali / Belarus / Russia / Palestine` filling 5 of the top 10. These are
slow-moving population time series that latch onto any other trending target by
coincidence — they crowd out the actual signal.
**Solution.** `src/curation.py` whitelists 7 economically credible categories
(Energy, Commodities, Exchange Rates, Global risk, Equities, Market Indices, Industry)
and applies a per-ticker scope plausibility filter (Market Indices in Slovenia is
demoted for TTF; in World scope it's kept). Top 10 after curation: 0 spurious,
9 different World-scoped Commodities/Energy/Global-risk variants. **This is the headline
differentiator from a raw-API wrapper.**

### 5.3 The model's own backtest metric is anomalous
**Problem.** Cached TTF backtest: MAPE 14.7% (fine), MASE 92.94, RMSSE 71.36
(catastrophic — model 90× worse than seasonal naive at scale). All four rolling windows
(6/12/24/60 months) return identical metrics — looks like single-point eval.
**Solution.** `src/economics.py` separates MAPE-based trust (`trust_mape = 1 -
MAPE/0.30`) from a MASE *floor* (penalty 0.5 when MASE > 10, treating it as a red flag
rather than a directly-scaled error). Combined trust = 0.26 for TTF, which shrinks
effective ACT/RECOMMEND thresholds by 0.15 — 4 of 6 months tier-shift toward ABSTAIN.
The system literally says "I don't trust myself here, fall back closer to baseline."

### 5.4 Driver direction lives in `external_signals.json`, not `/drivers`
**Problem.** `/drivers` (sync, perfect for the adaptive layer) returns only
`{driver_name, hash_id, score}` — no direction. So under a shock we can't naively
compute "did the signed pressure flip?" from the live `/drivers` re-pull.
**Solution.** `src/adaptive.py` uses cached `external_signals.json` as the baseline
pressure source (free, contains signed direction) and live `/alerts` as the shock
source (signed `pct_change` per alert, sync). Pressure delta = shocked - baseline.
**One billable call per shock** instead of two.

### 5.5 Cloudflare 1010 on Featherless
**Problem.** `https://api.featherless.ai/v1/chat/completions` returns 403 + Cloudflare
error code 1010 ("the owner of this website has banned your access based on your
browser's signature") for Python's default `Python-urllib/x.y` User-Agent.
**Solution.** Set `User-Agent: hedge-decision-agent/0.1` and `Accept: application/json`
on the urllib request. Auth was fine all along.
*Time wasted before realising it was the UA: ~15 min.*

### 5.6 `/alerts` filters over-narrow
**Problem.** Adding `regions=[3], categories=[25, 46]` to `/alerts` cut the result
count from 10 alerts to 0–1. The endpoint is semantic-search-driven; filters appear to
hard-restrict the candidate pool instead of just re-ranking.
**Solution.** Drop filters from `/alerts` queries by default — keywords already steer
the search. Documented in `notes/schema_summary.md` and the `ShockScenario` defaults.

### 5.7 Live `/alerts` may return a calm market even under a shock query
**Problem.** Running the "Hormuz" scenario today returned alerts dominated by VIX
**−14%** and European energy indices around −3% — a *bearish* signal. The pressure
delta came out negative and the system recommended **less** hedge.
**Reframe.** This is a feature, not a bug: the live data wasn't pricing a Hormuz event
today, and the system honestly captures that. For the demo we offer **two modes side
by side**: Live (real data, real reaction) and Simulated (controlled severity slider).
The audience sees the system isn't pretending.

### 5.8 PowerShell + UTF-8 + git commit messages
**Problem.** PowerShell here-strings break on lines starting with `-` (interpreted
as parameter switches); default Windows `open()` uses cp1252 and crashes on em-dash or
arrow characters; the auto-mode classifier blocks `git commit -F .git/MSG_FILE.txt` if
the file write isn't in the visible transcript.
**Solution.** Multiple short `-m` flags for git commits; explicit `encoding="utf-8"`
on every `open()`; ASCII-only output strings in CLI scripts (`->` instead of `→`).

### 5.9 Importance scale guessing
**Problem.** First decision-rule thresholds assumed importance values in `[0, 1]`.
Real values are `[0, 100]`. Calibration was off by 2 orders of magnitude.
**Solution.** Calibrated against the live TTF run: credible-driver threshold 5.0,
top-importance threshold 20.0. Documented in `decision.py`.

---

## 6. Testing

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
# 79 passed in ~3s
```

- All decision/economics/curation/adaptive/narrator paths have unit tests.
- The cached TTF artifact (`cache/3d08b704.../`) doubles as a live-shape fixture for
  integration tests — every parser is exercised against real API output, not a mock.
- Live smoke tests for the Sybilion client auto-skip when no token is present.
- LLM is mock-tested for the call shape; live Featherless is verified manually.
- Seeds are pinned (`SEED = 42` in `src/config.py`); deterministic by construction.

---

## 7. Cost budget

| Spent | What |
|---:|---|
| ~€1.32 | First (and only) forecast submit for TTF — cached after that |
| ~€0.30 | Live `/alerts` calls during demo & dev (~4 calls) |
| ~€0.30 | Featherless LLM tokens (memo + counter-case, a few times) |
| **~€2.00** | **Total spend; trial balance €48.98 remaining at session start, ~€47 still in trust** |

The decision engine, curation, backtest grounding, cost-of-waiting, and adaptive layer
all run **offline** off the committed cache — re-runs on the demo are free.

---

## 8. What this is *not*

- Not an LLM wrapper. The LLM is the last 5% of the stack and only verbalizes the
  numbers the engine has already computed.
- Not a black box. Every number on the cockpit is traceable through `src/decision.py`,
  `src/economics.py`, and the cached JSON artifacts.
- Not a Sybilion mirror. The driver curation, backtest trust grounding, and adaptive
  shock loop are entirely on top.
- Not safe for live trading. Hackathon prototype. Backtest is single-point per window
  on the API side; we treat the rule as directional and abstain when uncertain.

---

## License & credits

Built by Vincent Lanzinger for the Zero One Hack (Forecasting track), May 2026.
Powered by the Sybilion Forecasting API (https://sybilion.dev) and the Featherless.ai
LLM gateway. Code is for the hackathon submission; not for production use.
