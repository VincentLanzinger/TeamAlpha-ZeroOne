# Hedge Decision Agent — Zero One Hack (Forecasting track)

A decision agent on top of the Sybilion Forecasting API. NOT an LLM wrapper — the
intelligence lives in the decision logic (forecast band -> hedge ratio), the backtest
of that rule, and the adaptive layer that re-prices the decision when shocks land.

## Domain

Procurement / input-cost hedging for an EU industrial emitter.
Default metric: **EU carbon allowance price (EUA)**, monthly.
Decision: *"What % of next quarter's allowances do we forward-buy now vs. wait?"*

The ticker is swappable in **one line** in `src/config.py` (`TICKER = "EUA"`).
Fallbacks: `"TTF"` (natural gas) or `"ALUMINUM"`. The pipeline (data shape, decision
rule, reasoning, demo) is identical across tickers.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env       # then paste your real key into .env
```

## Structure

```
hedge-decision-agent/
  src/
    config.py            # TICKER + per-ticker metadata (one-line switch)
    sybilion_client.py   # Phase 1 — thin wrapper over the official SDK
    decision.py          # Phase 4 — band -> hedge ratio
    reasoning.py         # Phase 5 — structured decision surface
    adaptive.py          # Phase 7 — shock -> re-decide (sync endpoints only)
  tests/                 # pytest, seeds fixed
  data/                  # eua_monthly.csv etc. (>=60 monthly pts at horizon 6)
  cache/                 # cached forecast artifacts, keyed by request-body hash
  notes/
    schema_summary.md    # what the Sybilion API actually returns
  app.py                 # Phase 8 — Streamlit demo
```

## Demo path (Phase 8)

Forecasts are async (minutes). The live demo uses **pre-cached forecast artifacts**
in `cache/` plus **live sync calls** to `/drivers` and `/alerts` for the adaptive
layer. The cache is keyed by a hash of the forecast request body — identical
requests never re-bill.
