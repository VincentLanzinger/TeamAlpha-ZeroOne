"""Project-wide config.

The whole pipeline is parameterised by a single ticker string. Flip TICKER below
to switch the system end-to-end (data file, forecast metadata, drivers query,
alert query, demo labels). Decision rule, reasoning, and adaptive layer stay
identical.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# === ONE-LINE TICKER SWITCH ===
TICKER: str = "TTF"
# ==============================

SEED: int = 42

# Sybilion request defaults
PIPELINE_VERSION: str = "v1"
FREQUENCY: str = "monthly"
SOFT_HORIZON: int = 6       # 4–6 mo horizon → requires >=60 monthly observations
RECENCY_FACTOR: float = 0.5
BACKTEST: bool = True
STRICTLY_POSITIVE: bool = True   # prices are positive

# API token envvar names, in priority order. Canonical is SYBILION_API_TOKEN
# (matches the official docs); SYBILION_API_KEY is accepted as a fallback.
API_KEY_ENV_NAMES: tuple[str, ...] = ("SYBILION_API_TOKEN", "SYBILION_API_KEY")
BASE_URL: str = "https://api.sybilion.dev"

# Minimum observation count required by the API at the chosen horizon.
# 1–3 mo: 40, 4–6 mo: 60, 7–12 mo: 120.
MIN_OBS_BY_HORIZON: dict[int, int] = {3: 40, 6: 60, 12: 120}


def min_observations() -> int:
    """Required min observation count for the active SOFT_HORIZON."""
    if SOFT_HORIZON <= 3:
        return 40
    if SOFT_HORIZON <= 6:
        return 60
    return 120

# Polling
POLL_INTERVAL_S: float = 5.0
POLL_TIMEOUT_S: float = 600.0    # 10 min cap on a single forecast job
POLL_BACKOFF_MAX_S: float = 30.0


@dataclass(frozen=True)
class TickerSpec:
    """Everything the Sybilion API needs to recognise this series."""
    symbol: str
    csv_path: str
    metadata_title: str
    metadata_description: str
    keywords: tuple[str, ...] = field(default_factory=tuple)
    unit: str = ""
    display_name: str = ""
    # Per-ticker overrides for the forecast request.
    recency_factor_override: float | None = None
    forecast_regions: tuple[int, ...] = ()
    forecast_categories: tuple[int, ...] = ()


# Registry — add more tickers here; flipping TICKER above switches the whole pipeline.
# Titles are >=20 chars (API requires that); descriptions are short and concrete.
TICKER_REGISTRY: dict[str, TickerSpec] = {
    "EUA": TickerSpec(
        symbol="EUA",
        csv_path="data/eua_monthly.csv",
        metadata_title="EU ETS carbon allowance monthly settlement price (EUA)",
        metadata_description=(
            "Monthly EUR-per-tonne settlement price for EU Emissions Trading System "
            "carbon allowances (EUA). Driver of compliance cost for EU industrial "
            "emitters; used here as the input for a procurement hedging decision."
        ),
        keywords=(
            "EU ETS", "carbon allowance", "EUA", "carbon price",
            "emissions trading", "compliance cost", "industrial emitter",
        ),
        unit="EUR / tCO2",
        display_name="EU Carbon Allowance (EUA)",
    ),
    "TTF": TickerSpec(
        symbol="TTF",
        csv_path="data/ttf_monthly.csv",
        metadata_title="Dutch TTF natural gas monthly front-month settlement price",
        metadata_description=(
            "Monthly EUR-per-MWh front-month settlement for Dutch TTF natural gas. "
            "Primary European gas benchmark and a key cost input for power and "
            "industrial users; here used as a hedging decision target. Drivers "
            "include LNG flows and inventory, gas storage levels, Brent oil, "
            "European electricity prices, and heating demand."
        ),
        keywords=(
            "TTF", "LNG", "gas storage", "Brent", "electricity",
            "heating demand", "European energy",
        ),
        unit="EUR / MWh",
        display_name="Dutch TTF Natural Gas",
        # Hour-one gate uses recency_factor 0.7 (TTF moves fast; recent regime matters).
        recency_factor_override=0.7,
        # Filter scope chosen for the gate: regions=[3] (Europe), categories=[25 Energy, 46 Commodities].
        forecast_regions=(3,),
        forecast_categories=(25, 46),
    ),
    "ALUMINUM": TickerSpec(
        symbol="ALUMINUM",
        csv_path="data/aluminum_monthly.csv",
        metadata_title="LME aluminum monthly cash settlement price benchmark",
        metadata_description=(
            "Monthly USD-per-tonne LME aluminum cash settlement. Benchmark for "
            "industrial aluminum procurement contracts; here used as a hedging "
            "decision target for an industrial buyer."
        ),
        keywords=(
            "aluminum", "LME", "metals", "industrial metal", "commodity",
            "procurement", "input cost",
        ),
        unit="USD / tonne",
        display_name="LME Aluminum",
    ),
}


def active_ticker() -> TickerSpec:
    """Return the currently selected ticker spec."""
    if TICKER not in TICKER_REGISTRY:
        raise KeyError(
            f"TICKER={TICKER!r} not in TICKER_REGISTRY (have: "
            f"{sorted(TICKER_REGISTRY)}). Add a TickerSpec or fix config.py."
        )
    return TICKER_REGISTRY[TICKER]
