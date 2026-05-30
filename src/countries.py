"""Country-specific gas-market specs.

The underlying price series stays Dutch TTF (the European gas benchmark — every
EU procurement team aligns to it). What changes per country:

1. **Region filter** sent to Sybilion's `/forecasts` and `/drivers` — different
   region surfaces different drivers (e.g. Germany surfaces Russian/Norwegian
   supply signals, Spain surfaces Algerian pipeline / LNG terminal signals).

2. **Amplification table** — each country's gas economy has its own
   high-conviction tokens. Germany cares more about heating + industrial PMI;
   Spain cares more about LNG terminals + Algerian supply; UK cares more about
   the North Sea + storage. The weighting reflects that.

3. **Forecast metadata** (title / description / keywords) — Sybilion's NLP
   models read these to steer driver selection.

The structure is meant to be auditable: a procurement analyst can read
`amplify_tokens` for their country and challenge the weights. Tokens are
hand-encoded, not learned.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CountrySpec:
    code: str                          # "DE", "FR", ...
    name: str                          # "Germany"
    flag: str                          # "🇩🇪"
    region_id: int                     # Sybilion region id (from /regions)
    # Plain-English line shown in the country picker.
    one_liner: str
    # Longer rationale — surfaced once the country is selected.
    rationale: str
    # Per-country keyword amplification (replaces AMPLIFY_TOKENS_TTF).
    amplify_tokens: dict[str, float] = field(default_factory=dict)
    # Per-country forecast metadata. The price *data* is always TTF.
    forecast_title: str = ""
    forecast_description: str = ""
    forecast_keywords: tuple[str, ...] = field(default_factory=tuple)
    # Marker for the one country whose body matches the committed cache.
    cached_in_repo: bool = False


# -- common defaults reused across countries ---------------------------------

_DEFAULTS = {
    "brent":         1.40,
    "carbon":        1.40,
    "vix":           1.35,
    "volatility":    1.35,
    "geopolitical":  1.35,
    "global risk":   1.35,
    "exchange rate": 1.25,
    "eur/usd":       1.25,
    "usd/eur":       1.25,
    "pmi":           1.20,
    "inflation":     1.15,
    "natural gas":   1.45,
    "energy":        1.15,
    "commodities":   1.10,
}


def _merge(*tables: dict[str, float]) -> dict[str, float]:
    """Combine tables; later table wins on conflicting tokens."""
    out: dict[str, float] = {}
    for t in tables:
        out.update(t)
    return out


# ===========================================================================
# Country registry
# ===========================================================================

COUNTRIES: dict[str, CountrySpec] = {

    # The Europe-wide spec is special — its body matches the committed cache
    # so picking it gives an instant offline demo with no live spend.
    "EU": CountrySpec(
        code="EU", name="Europe (region-wide)", flag="🇪🇺", region_id=3,
        one_liner="Demo default — uses the pre-cached forecast (no spend)",
        rationale=(
            "Europe-wide perspective with no single-country lean. This is the "
            "forecast that's committed to the repo cache — picking it loads "
            "instantly without billing the API."
        ),
        amplify_tokens=_merge(_DEFAULTS, {
            "lng":           1.50,
            "storage":       1.30,
            "heating":       1.25,
        }),
        # Body matches the committed cache 3d08b704... exactly.
        forecast_title="Dutch TTF natural gas monthly front-month settlement price",
        forecast_description=(
            "Monthly EUR-per-MWh front-month settlement for Dutch TTF natural gas. "
            "Primary European gas benchmark and a key cost input for power and "
            "industrial users; here used as a hedging decision target. Drivers "
            "include LNG flows and inventory, gas storage levels, Brent oil, "
            "European electricity prices, and heating demand."
        ),
        forecast_keywords=(
            "TTF", "LNG", "gas storage", "Brent", "electricity",
            "heating demand", "European energy",
        ),
        cached_in_repo=True,
    ),

    "DE": CountrySpec(
        code="DE", name="Germany", flag="🇩🇪", region_id=1083,
        one_liner="Industrial demand · heating · post-Russian LNG transition",
        rationale=(
            "Largest EU gas consumer. Heavy industrial gas use (chemicals, "
            "metals); large residential heating share; recent pivot from "
            "Russian pipeline gas to LNG terminals (Wilhelmshaven, Brunsbüttel) "
            "and Norwegian supply via Nord Stream replacement routes. "
            "Sensitive to PMI, Norwegian flows, and storage fill levels."
        ),
        amplify_tokens=_merge(_DEFAULTS, {
            "heating":       1.50,   # biggest residential gas market in EU
            "industrial":    1.40,
            "manufacturing": 1.35,
            "pmi":           1.35,
            "storage":       1.45,   # storage politics is a national topic
            "lng":           1.50,
            "norwegian":     1.35,
            "russian":       1.30,
            "chemical":      1.25,
        }),
        forecast_title="Dutch TTF gas price benchmark for German industrial procurement",
        forecast_description=(
            "Monthly Dutch TTF front-month settlement (EUR/MWh), used as the "
            "reference price for a German industrial procurement team. Drivers "
            "of interest: German PMI, residential heating demand, LNG inflows "
            "via Wilhelmshaven and Brunsbüttel, Norwegian pipeline supply, "
            "storage fill levels, and the post-Russian-gas transition."
        ),
        forecast_keywords=(
            "TTF", "Germany", "heating", "industrial", "LNG", "Norwegian gas",
            "gas storage", "PMI", "Wilhelmshaven", "chemical industry",
        ),
    ),

    "FR": CountrySpec(
        code="FR", name="France", flag="🇫🇷", region_id=1076,
        one_liner="Nuclear-heavy · LNG via Dunkerque · electricity-gas spread",
        rationale=(
            "Heavily nuclear-powered, so gas demand is more peaking and "
            "heating-driven than industrial. LNG arrives via Dunkerque, "
            "Fos-sur-Mer, Montoir. Sensitive to French nuclear availability — "
            "outages spike gas-fired generation demand."
        ),
        amplify_tokens=_merge(_DEFAULTS, {
            "nuclear":       1.45,
            "electricity":   1.40,
            "heating":       1.35,
            "lng":           1.45,
            "power":         1.30,
        }),
        forecast_title="Dutch TTF gas price benchmark for French industrial procurement",
        forecast_description=(
            "Monthly Dutch TTF front-month settlement (EUR/MWh) for a "
            "France-based procurement context. Key drivers: French nuclear "
            "availability (drives peaking gas demand), electricity-gas spread, "
            "LNG flows via Dunkerque / Fos-sur-Mer, heating demand."
        ),
        forecast_keywords=(
            "TTF", "France", "nuclear", "electricity", "LNG", "Dunkerque",
            "heating", "European energy",
        ),
    ),

    "IT": CountrySpec(
        code="IT", name="Italy", flag="🇮🇹", region_id=1110,
        one_liner="Mediterranean supply · Algerian pipeline · industrial demand",
        rationale=(
            "Major gas consumer, historically reliant on Russian pipeline gas. "
            "Now pivoting to Algerian (TransMed), Libyan (GreenStream), and "
            "LNG. Sensitive to Mediterranean security and Italian industrial "
            "PMI."
        ),
        amplify_tokens=_merge(_DEFAULTS, {
            "lng":           1.45,
            "algeria":       1.40,
            "mediterranean": 1.35,
            "industrial":    1.30,
            "heating":       1.25,
        }),
        forecast_title="Dutch TTF gas price benchmark for Italian industrial procurement",
        forecast_description=(
            "Monthly Dutch TTF front-month settlement (EUR/MWh) for "
            "Italy-based procurement. Drivers: Algerian / Libyan / LNG supply, "
            "Mediterranean security, Italian PMI, heating demand."
        ),
        forecast_keywords=(
            "TTF", "Italy", "Algeria", "LNG", "Mediterranean", "TransMed",
            "industrial", "European energy",
        ),
    ),

    "ES": CountrySpec(
        code="ES", name="Spain", flag="🇪🇸", region_id=1209,
        one_liner="LNG terminals · Algerian pipeline · renewables-balancing",
        rationale=(
            "Most LNG regasification capacity in Europe (Barcelona, Sagunto, "
            "Bilbao, Huelva). Algerian gas via Medgaz / Maghreb-Europe. "
            "Increasing renewables share — gas is a balancing fuel. "
            "Sensitive to global LNG prices and North African politics."
        ),
        amplify_tokens=_merge(_DEFAULTS, {
            "lng":           1.50,
            "algeria":       1.45,
            "renewable":     1.30,
            "electricity":   1.30,
            "maghreb":       1.40,
        }),
        forecast_title="Dutch TTF gas price benchmark for Spanish industrial procurement",
        forecast_description=(
            "Monthly Dutch TTF front-month settlement (EUR/MWh) for "
            "Spain-based procurement. Key drivers: global LNG prices "
            "(Spain is Europe's LNG gateway), Algerian Medgaz / Maghreb-Europe "
            "pipeline flows, electricity balancing demand."
        ),
        forecast_keywords=(
            "TTF", "Spain", "LNG", "Algeria", "Maghreb", "Medgaz",
            "Barcelona", "regasification", "renewables",
        ),
    ),

    "NL": CountrySpec(
        code="NL", name="Netherlands", flag="🇳🇱", region_id=1156,
        one_liner="TTF hub · Groningen phase-out · LNG gateway",
        rationale=(
            "Home of the TTF benchmark itself. Domestic Groningen field "
            "phased out by 2024 — now a net importer via LNG (Gate terminal "
            "Rotterdam) and North Sea pipelines. Trading hub role amplifies "
            "any signal."
        ),
        amplify_tokens=_merge(_DEFAULTS, {
            "lng":           1.50,
            "storage":       1.40,
            "north sea":     1.35,
            "rotterdam":     1.30,
        }),
        forecast_title="Dutch TTF gas price for Netherlands-based procurement",
        forecast_description=(
            "Monthly Dutch TTF front-month (EUR/MWh) for a Netherlands-based "
            "procurement context. Drivers: Gate LNG terminal flows, North Sea "
            "supply, storage levels, the role of Rotterdam as European gas "
            "transit hub."
        ),
        forecast_keywords=(
            "TTF", "Netherlands", "LNG", "Gate terminal", "Rotterdam",
            "North Sea", "gas storage", "Groningen",
        ),
    ),

    "BE": CountrySpec(
        code="BE", name="Belgium", flag="🇧🇪", region_id=1021,
        one_liner="Zeebrugge LNG · transit hub · industrial intensity",
        rationale=(
            "Major LNG terminal at Zeebrugge — handles re-export to UK and "
            "transit to Germany. Heavy chemical and steel industry. ZTP hub "
            "alongside TTF."
        ),
        amplify_tokens=_merge(_DEFAULTS, {
            "lng":           1.50,
            "zeebrugge":     1.45,
            "industrial":    1.35,
            "chemical":      1.30,
            "storage":       1.30,
        }),
        forecast_title="Dutch TTF gas price benchmark for Belgian industrial procurement",
        forecast_description=(
            "Monthly Dutch TTF front-month (EUR/MWh) for a Belgium-based "
            "procurement context. Key drivers: Zeebrugge LNG flows and re-"
            "export, chemical industry demand, ZTP hub dynamics."
        ),
        forecast_keywords=(
            "TTF", "Belgium", "Zeebrugge", "LNG", "ZTP", "chemical", "industrial",
        ),
    ),

    "GB": CountrySpec(
        code="GB", name="United Kingdom", flag="🇬🇧", region_id=1234,
        one_liner="NBP hub · Norwegian + LNG · North Sea decline",
        rationale=(
            "Post-Brexit NBP price benchmark, but still tightly coupled to "
            "TTF via the Interconnector. North Sea domestic production is in "
            "decline; reliance on Norwegian pipeline and LNG (Isle of Grain, "
            "South Hook, Dragon)."
        ),
        amplify_tokens=_merge(_DEFAULTS, {
            "lng":           1.45,
            "norwegian":     1.45,
            "north sea":     1.40,
            "storage":       1.35,
            "heating":       1.35,
            "nbp":           1.30,
        }),
        forecast_title="Dutch TTF gas price benchmark for UK industrial procurement",
        forecast_description=(
            "Monthly Dutch TTF front-month (EUR/MWh) used as the European "
            "reference for a UK-based procurement context (NBP tracks TTF "
            "closely via the Interconnector). Drivers: Norwegian pipeline "
            "flows, LNG via Isle of Grain / South Hook / Dragon, North Sea "
            "production decline, storage levels."
        ),
        forecast_keywords=(
            "TTF", "NBP", "United Kingdom", "Norwegian gas", "LNG",
            "North Sea", "Isle of Grain", "storage", "Interconnector",
        ),
    ),

    "PL": CountrySpec(
        code="PL", name="Poland", flag="🇵🇱", region_id=1177,
        one_liner="Baltic Pipe · Świnoujście LNG · post-Russian transition",
        rationale=(
            "Aggressive diversification away from Russian gas: Baltic Pipe "
            "(Norwegian gas via Denmark) commissioned 2022, Świnoujście LNG "
            "expansion. Heavy industrial base; cold winters drive heating "
            "demand."
        ),
        amplify_tokens=_merge(_DEFAULTS, {
            "lng":           1.45,
            "norwegian":     1.45,
            "baltic":        1.40,
            "heating":       1.40,
            "industrial":    1.30,
            "russian":       1.25,
        }),
        forecast_title="Dutch TTF gas price benchmark for Polish industrial procurement",
        forecast_description=(
            "Monthly Dutch TTF front-month (EUR/MWh) for a Poland-based "
            "procurement context. Drivers: Baltic Pipe Norwegian flows, "
            "Świnoujście LNG capacity, heating demand, post-Russian-gas "
            "transition."
        ),
        forecast_keywords=(
            "TTF", "Poland", "Baltic Pipe", "Norwegian gas", "LNG",
            "Świnoujście", "heating", "industrial",
        ),
    ),
}


def get_country(code: str) -> CountrySpec:
    if code not in COUNTRIES:
        raise KeyError(
            f"Country code {code!r} not in registry. "
            f"Available: {sorted(COUNTRIES)}"
        )
    return COUNTRIES[code]


def cached_country_codes() -> list[str]:
    """Codes whose forecast body matches a committed cache (no live spend)."""
    return [c for c, spec in COUNTRIES.items() if spec.cached_in_repo]
