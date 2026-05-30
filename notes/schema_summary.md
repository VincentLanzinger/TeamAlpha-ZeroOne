# Sybilion API — schema notes (Phase 0)

Base URL: `https://api.sybilion.dev`
Path prefix: `/api/v1`
Auth header: `Authorization: Bearer $SYBILION_API_TOKEN` (key starts `sk_ops_…`).

**Use the official SDK** (`pip install sybilion`, currently 0.1.4):

```python
from sybilion import Client
client = Client(token=os.environ["SYBILION_API_TOKEN"])
job = client.submit_forecast(body)             # async submit
client.wait_forecast(job.job_id, poll_s=5, timeout_s=600)
art = client.get_forecast_artifact(job.job_id, "forecast.json")
drivers  = client.get_drivers(body_dri)        # sync
alerts   = client.get_alerts(body_alr)         # sync
me       = client.me()                          # sync — check available_eur_cents BEFORE submitting
```
SDK runtime deps: urllib3 ≥2, python-dateutil, pydantic ≥2, typing-extensions. Py ≥3.10.

## Endpoint map

| Method | Path | Sync? | Used by |
|---|---|---|---|
| GET | `/api/v1/me` | sync | check_account |
| GET | `/api/v1/usage` | sync | check_account, billing audit |
| GET | `/api/v1/jobs` | sync | (optional) job history |
| POST | `/api/v1/forecasts` | **async (202)** | submit_forecast |
| GET | `/api/v1/forecasts/{id}` | sync poll | poll_status |
| GET | `/api/v1/forecasts/{id}/artifacts/{name}` | sync | get_artifacts |
| POST | `/api/v1/drivers` | **sync** | get_drivers |
| POST | `/api/v1/alerts` | **sync** | get_alerts |
| GET | `/api/v1/regions` | sync | list_regions |
| GET | `/api/v1/categories` | sync | list_categories |

Sync vs async matters for the demo: live adaptation in Phase 7 / Phase 8 goes
through `/drivers` + `/alerts` only.

## POST /api/v1/forecasts — request

```yaml
required: [pipeline_version, frequency, recency_factor, timeseries_metadata, timeseries]
pipeline_version: "v1"
frequency: "monthly"
soft_horizon: 1..12        # at least one of soft/hard — we use 6 ("next 2 quarters")
hard_horizon: 1..12
recency_factor: 0..1
backtest: bool             # we always pass true to get backtest_metrics + trajectories
strictly_positive: bool    # true for prices
timeseries: { "YYYY-MM-DD": number, ... }
  # Dates MUST be first-of-month at monthly frequency.
  # Latest point must be within the last 12 months.
  # Min obs: 40 (h ≤ 3) / 60 (h ≤ 6) / 120 (h ≤ 12). We target ≥60.
timeseries_metadata:
  title: string, 20..511 chars  (required)
  description?: string, <=2048
  keywords?: string[<=20], each 1..255
filters?:
  regions?: int[]    # ids from /regions
  categories?: int[] # ids from /categories
  limit?: 0..1000
```

Response 202: `{ job_id (uuid), poll_url, run_id, workflow }`

## GET /api/v1/forecasts/{id} — response

```yaml
job_id, status: [queued|running|completed|failed|canceled], settled: bool,
pipeline_error: object|null,
artifacts: [{name, href, content_type, size}]
```

Poll until `settled == true` and `status == "completed"`.

## Artifacts (GET …/artifacts/{name})

- `forecast.json` — point + quantile bands per horizon month
  ```json
  { "forecast_series": {
      "YYYY-MM-DD": { "forecast": 78.4,
                      "quantile_forecast": { "0.1": 68.2, "0.5": 78.4, "0.9": 89.1 } } } }
  ```
- `external_signals.json` — ranked drivers used by the model (per-forecast attribution).
  Per-driver fields: `importance`, `direction` (up/down), `correlation`. THIS is where
  per-horizon driver direction lives; `/drivers` itself only returns score.
- `backtest_metrics.json` — MAPE / RMSE over 6m/12m/24m/60m windows (only if `backtest: true`)
- `backtest_trajectories.json` — per-fold actual vs forecast for last 12 months
- `input.json` — processed input timeseries as the pipeline saw it

## POST /api/v1/drivers — sync

Request:
```yaml
required: [version, recency_factor, timeseries_metadata]
version: "v1"
recency_factor: 0..1
timeseries_metadata: TimeseriesMetadata
timeseries?: { "YYYY-MM-DD": number, ... }
filters?: Filters
```
Response: `{ status, message, data: { drivers: [DriverItemV1, ...] } }`
DriverItemV1: `{ driver_name, hash_id, score }` — `score` = relevance (higher = more relevant).

**Note:** driver `direction` (up/down) is **NOT** in `/drivers`. It surfaces in
`external_signals.json` on a forecast, and in alerts via `pct_change` sign.

## POST /api/v1/alerts — sync

Request:
```yaml
required: [metadata, context_enriched]
metadata: TimeseriesMetadata        # field name is `metadata`, NOT `timeseries_metadata`
context_enriched: bool
date_from?: "YYYY-MM-DD"
date_to?: "YYYY-MM-DD"
filters?: Filters
```
Response: `{ alerts: [AlertItemV1, ...] }`
AlertItemV1: `{ name, pct_change, trending, news: [NewsItemV1, ...] }`
NewsItemV1: `{ title, description, url, source_name, category, published_at, trending }`

## GET /api/v1/me — response

```yaml
user_id (uuid), balance_eur_cents, available_eur_cents, api_usage_tier,
lifetime_paid_cents, payment_count, has_ever_paid, euro_tranches,
auto_recharge, signup_trial: {granted_at, expires_at, initial_eur_cents, remaining_eur_cents} | null
```

## Minimum observations by horizon

| Horizon (months) | Min obs |
|---|---|
| 1–3 | 40 |
| 4–6 | 60 |
| 7–12 | 120 |

Our `SOFT_HORIZON = 6` → **need ≥60 monthly observations**. Covers two quarters of
forward decisioning, which is what the hedge ratio sweep is built around.

## Regions & categories (`GET /api/v1/regions`, `GET /api/v1/categories`)

Read-only catalogs of integer ids accepted by the `filters` object. **Not billed.**
Returned in full (no pagination), sorted by id ascending. Each item has
`{id, name, ...}`; regions also include hierarchy metadata (parent, path,
coordinates); categories include optional classification codes.

We'll resolve the EU/Europe id once in Phase 3 to scope EUA driver queries.

## Cost & cache discipline

- Billing fires on **2xx** with a pre-charge hold on submit. Quota is limited.
- Before any submit: call `client.me()` and gate on `available_eur_cents`.
- **Every** forecast request goes through a disk cache in `cache/` keyed by a stable
  hash (SHA-256) of the canonicalised JSON body. Identical requests never re-bill.
- Catalog calls (`/regions`, `/categories`) are free; alerts/drivers are sync but
  metered, so cache them when content allows.

## Gotchas to remember

1. `/alerts` uses `metadata`, all others use `timeseries_metadata`. Easy to typo.
2. `title` has a **20-char minimum**. Short titles get rejected.
3. `/drivers` returns no direction — only `driver_name`, `hash_id`, `score`.
   Per-horizon direction lives in forecast `external_signals.json`; pct_change sign
   in alerts is another direction proxy.
4. Forecast submit returns **202**, not 200. Don't treat 202 as failure.
5. `strictly_positive: true` matters for prices (prevents negative quantile tails).
6. Timeseries dates must be **first-of-month** strings at monthly frequency, and the
   **latest point must be within the last 12 months** or submit will reject.
