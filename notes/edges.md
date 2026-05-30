# Driver Curation — the "edge"

This is the differentiator: the agent doesn't just consume Sybilion's raw driver
ranking — it filters and re-weights it through three layers of economic reasoning
that nothing in the underlying API performs.

Implemented in [`src/curation.py`](../src/curation.py).
Tested in [`tests/test_curation.py`](../tests/test_curation.py).

## 1. Purpose

Separate **credible signals** from **spurious proxies**. The Sybilion API
publishes 70+ driver candidates per forecast — most are honest, some latch onto
the target by trend coincidence (e.g. monotone population time series). Without
curation those crowd out the real signal.

## 2. Three-stage curation pipeline

Each driver passes through three filters; the final score = raw × scope_factor × amp_factor.

### Stage 1 — Category whitelist (binary keep/drop)

`DEFAULT_WHITELIST` keeps only categories with a defensible mechanism for
driving European gas:

| Whitelisted | Why |
|---|---|
| Energy | Direct: gas, power, neighbour markets |
| Commodities | Brent / coal / LNG / carbon — all gas-price coupled |
| Exchange Rates | EUR weakness raises EUR-denominated gas cost |
| Global risk | Geopolitical premium (Hormuz, Ukraine, Middle East) |
| Equities | Risk-on/off proxy via the cycle |
| Market Indices | Macro state signals |
| Industry | Demand-side proxies |

Everything else (Population, Bonds, Labour, Public finance, Earnings, Justice,
Health, Education, …) is **dropped**. The decision is per-category, by
*economic mechanism*, not by name.

### Stage 2 — Scope plausibility (×0.30 demote)

A whitelisted driver can still have an implausible *country scope*. Slovenian
market indices probably don't drive Dutch TTF; Belgian energy probably does.

`PLAUSIBLE_SCOPES_TTF` lists the country/region tags that are economically
linked to European gas: World, EU, Germany, Netherlands, Belgium, France,
Italy, Spain, Norway, UK, Denmark, Austria, Poland — plus the major
LNG/gas-exporting nations (US, Russia, Qatar, Australia, China).

In-whitelist + implausible scope → multiplier **0.30** (kept but down-weighted).
In-whitelist + plausible scope → multiplier **1.00**.

Per-ticker tables exist for TTF and Aluminum; new tickers add their own.

### Stage 3 — Dynamic keyword amplification (×1.0 – ×1.5 boost)

This is what turns "blunt whitelist" into "graduated conviction." Driver names
are scanned for high-value tokens; the strongest match's factor multiplies the
score. The table (`AMPLIFY_TOKENS_TTF`):

| Tier | Tokens | Factor |
|---|---|---|
| 1. Direct gas-market signals | brent, lng, carbon, natural gas | **×1.45 – 1.50** |
| 2. Risk / volatility | vix, volatility, geopolitical, global risk | ×1.35 – 1.40 |
| 3. Macro fundamentals | exchange rate, eur/usd, usd/eur, pmi, inflation, heating, storage | ×1.20 – 1.30 |
| 4. Broad-category | energy, commodities | ×1.10 – 1.15 |

Multiple matches → strongest factor wins (we never compound, to avoid weak
tokens stacking into dominance).

`amplify_table=None` disables Stage 3 for A/B comparisons.

## 3. Live evidence on the TTF run

Raw API ranking (top 10):

```
#1  Energy - Belgium                  98.02
#2  Population - Nigeria              93.28   ← spurious
#3  Global risk - United Kingdom      90.35
#4  Commodities - World               86.76
#5  Population - Mali                 85.93   ← spurious
#6  Commodities - World               84.16
#7  Population - Belarus              80.68   ← spurious
#8  Commodities - World               74.02
#9  Population - Belarus              71.66   ← spurious
#10 Population - Russian Federation   69.79   ← spurious
```

Curated ranking after all three stages (top 10):

```
#1  Global risk - United Kingdom    121.97   amp x1.35  ← promoted from raw #3
#2  Energy - Belgium                112.72   amp x1.15  ← raw #1
#3  Commodities - World              95.43   amp x1.10
#4  Commodities - World              92.58   amp x1.10
#5  Commodities - World              81.42   amp x1.10
#6  Commodities - World              63.81   amp x1.10
#7  Exchange Rates - World           60.89   amp x1.30  ← strongest tier-3 promotion
#8  Commodities - World              60.42   amp x1.10
#9  Global risk - World              50.62   amp x1.35
#10 Commodities - World              49.36   amp x1.10
```

**Effect:**
- **0% spurious** drivers in the top 10 (was 50%)
- **Global risk - UK jumped from #3 to #1** (×1.35 amplification re-ranked it
  above the raw leader)
- 71 raw drivers → 34 kept, 3 demoted, 34 dropped; **15 of the kept got an
  amplification boost** because their names matched a tier token

## 4. Why this is non-trivial

- **Not just keep/drop**: a binary whitelist would treat Energy-Belgium and
  Commodities-World identically. Amplification graduates that — direct
  gas-market tokens (Brent, LNG, Carbon, VIX) get a stronger voice than
  generic-category drivers.
- **Not an LLM black box**: the table is 18 explicit tokens with documented
  factors. A reviewer can change the table without retraining anything.
- **Composes with the rest of the engine**: amplified importance flows into
  the per-horizon decision rule and the cost-of-waiting math. A shock that
  brings VIX up amplifies (×1.40) the same VIX entries it surfaces in
  `/alerts`, so the rule reacts proportionally.
- **A/B-able**: the `amplify_table=None` flag turns Stage 3 off, so a judge
  can see exactly how much it moved the recommendation.

## 5. Where to look in the code

| Concern | File |
|---|---|
| Category whitelist + scope sets | `src/curation.py` (`DEFAULT_WHITELIST`, `PLAUSIBLE_SCOPES_TTF`) |
| Keyword amplification table | `src/curation.py` (`AMPLIFY_TOKENS_TTF`) |
| Amplification helper | `src/curation.py` (`amplification_factor`) |
| Apply curation | `src/curation.py` (`curate`) |
| Side-by-side raw vs curated CLI | `python -m src.curation` |
| Tests | `tests/test_curation.py` (16 tests: whitelist, demotion, amplification, compounding, A/B, real-artifact integration) |
| Cockpit display | `app.py` — *Show advanced details* → *Top kept drivers* table has `amp` and `matched` columns |

## 6. Limitations honestly stated

- The amplification table is **hand-curated** for TTF. A second ticker needs
  its own table — `AMPLIFY_TOKENS_TTF` doesn't auto-generalise to aluminum.
- **No machine-learned weights**: tokens are hand-picked from domain
  knowledge. That's a feature for explainability, but it means we can't
  claim "the model learned which drivers matter" — we *encoded* it.
- The factor ceiling is **×1.50**. A driver can be at most ~50% boosted above
  its whitelist-adjusted score. This keeps any single token from dominating.
- Tokens are **substring matches**, not regex. "Brent" matches "Brent crude"
  AND "Cornelius Brent Holdings" — false positives are possible in principle
  but haven't shown up in the live data.
