"""Parsers for Sybilion forecast artifacts.

Lifted out of scripts/hour_one_gate.py so that curation, decision, reasoning,
and the Streamlit cockpit can all share the same parsed structures.

Real artifact shapes (observed against a live TTF forecast):

  forecast.json
    { version, data: {
        forecast_horizon: int,
        forecast_start: "YYYY-MM-DD",
        forecast_series: {
            "YYYY-MM-DD": {
                forecast: float,
                quantile_forecast: {"0.05": .., "0.1": .., ..., "0.95": ..}
            }, ...
        }
    }}

  external_signals.json
    { version,
      data: {
        "<uuid>": {
            driver_name: "Category - Scope",
            importance: {
                horizon_1: {<lag_key str>: float, ...},
                ...
                horizon_6: {<lag_key str>: float, ...},
                # optional `overall: {mean, min, max}` injected by API
            },
            direction: {
                horizon_0?: {<lag_key str>: float, ...},
                horizon_1..6?: {<lag_key str>: float, ...},
                overall?: {mean, min, max}
            },
            pearson_correlation: {
                lag_3..6: float,
                overall: {mean, min, max}
            }
        },
        ...
      }
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

HORIZON_KEYS: tuple[str, ...] = (
    "horizon_1", "horizon_2", "horizon_3", "horizon_4", "horizon_5", "horizon_6",
)
HORIZONS: tuple[int, ...] = tuple(range(1, 7))


# ---------- forecast.json ----------------------------------------------------

@dataclass(frozen=True)
class HorizonBand:
    date: str
    q10: float
    q50: float
    q90: float
    point: float

    @property
    def width(self) -> float:
        return self.q90 - self.q10

    @property
    def width_pct(self) -> float:
        return (self.q90 - self.q10) / self.q50 if self.q50 else float("nan")


def parse_forecast_bands(forecast_json: dict[str, Any]) -> list[HorizonBand]:
    series = (
        forecast_json.get("data", {}).get("forecast_series")
        or forecast_json.get("forecast_series")
        or {}
    )
    rows: list[HorizonBand] = []
    for date, row in sorted(series.items()):
        q = row.get("quantile_forecast", {})
        point = row.get("forecast")
        rows.append(
            HorizonBand(
                date=date,
                q10=float(q.get("0.1") or q.get("0.10") or float("nan")),
                q50=float(q.get("0.5") or q.get("0.50") or float("nan")),
                q90=float(q.get("0.9") or q.get("0.90") or float("nan")),
                point=float(point) if point is not None else float("nan"),
            )
        )
    return rows


# ---------- external_signals.json -------------------------------------------

@dataclass(frozen=True)
class Driver:
    """A parsed driver from external_signals.json.

    The API publishes `importance.overall.mean` (= mean over lag-key values
    *within* each horizon, then identical across horizons in practice). We
    prefer that value when present and fall back to computing it from the
    per-horizon entries — same aggregation rule (mean over inner lag keys).
    """
    uuid: str
    name: str               # full "Category - Scope"
    category: str           # before the " - "
    scope: str              # after the " - " (e.g. "World", "Belgium")
    importance_overall: float = 0.0
    direction_overall: float = 0.0
    importance_by_h: dict[int, float] = field(default_factory=dict)
    direction_by_h: dict[int, float] = field(default_factory=dict)
    correlation: float | None = None  # pearson_correlation.overall.mean

    def direction_sign(self, threshold: float = 0.01) -> str:
        v = self.direction_overall
        return "+" if v > threshold else ("-" if v < -threshold else "0")


def _agg_inner(node: Any) -> float:
    """A per-horizon node is `{<lag_key str>: float, ...}`. The API's
    `importance.overall.mean` semantics is the mean over those lag values,
    so we match that.
    """
    if isinstance(node, dict):
        try:
            vals = [float(v) for v in node.values()]
        except (TypeError, ValueError):
            return 0.0
        return sum(vals) / len(vals) if vals else 0.0
    try:
        return float(node) if node is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _per_horizon(node: Any) -> dict[int, float]:
    if not isinstance(node, dict):
        return {}
    out: dict[int, float] = {}
    for h in HORIZONS:
        v = node.get(f"horizon_{h}")
        if v is not None:
            out[h] = _agg_inner(v)
    return out


def _overall_mean(node: Any) -> float | None:
    """Read node.overall.mean if present (the API publishes this)."""
    if isinstance(node, dict):
        ov = node.get("overall")
        if isinstance(ov, dict) and "mean" in ov:
            try:
                return float(ov["mean"])
            except (TypeError, ValueError):
                return None
    return None


def _split_name(name: str) -> tuple[str, str]:
    if " - " in name:
        cat, scope = name.split(" - ", 1)
        return cat.strip(), scope.strip()
    return name.strip(), ""


def _correlation(node: dict[str, Any]) -> float | None:
    corr = node.get("pearson_correlation") or node.get("correlation")
    if isinstance(corr, dict):
        ov = corr.get("overall")
        if isinstance(ov, dict) and "mean" in ov:
            return float(ov["mean"])
        if "mean" in corr:
            return float(corr["mean"])
        return None
    try:
        return float(corr) if corr is not None else None
    except (TypeError, ValueError):
        return None


def parse_drivers(external_signals: dict[str, Any]) -> list[Driver]:
    """Return drivers ranked by importance_overall (desc).

    Handles the live shape (`data` is a dict keyed by UUID) and a legacy
    flat list shape if encountered.
    """
    raw = external_signals.get("data") or external_signals.get("drivers") or []
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = [(str(d.get("hash_id", "")), d) for d in raw]
    else:
        items = []
    drivers: list[Driver] = []
    for uid, d in items:
        if not isinstance(d, dict):
            continue
        name = str(d.get("driver_name") or d.get("name") or d.get("hash_id") or "?")
        cat, scope = _split_name(name)
        imp_h = _per_horizon(d.get("importance"))
        dir_h = _per_horizon(d.get("direction"))
        # Prefer the API-published overall.mean; else fall back to the mean
        # over per-horizon (lag-mean) values we just computed.
        imp_overall = _overall_mean(d.get("importance"))
        if imp_overall is None:
            imp_overall = sum(imp_h.values()) / len(imp_h) if imp_h else 0.0
        dir_overall = _overall_mean(d.get("direction"))
        if dir_overall is None:
            dir_overall = sum(dir_h.values()) / len(dir_h) if dir_h else 0.0
        drivers.append(
            Driver(
                uuid=str(uid),
                name=name,
                category=cat,
                scope=scope,
                importance_overall=imp_overall,
                direction_overall=dir_overall,
                importance_by_h=imp_h,
                direction_by_h=dir_h,
                correlation=_correlation(d),
            )
        )
    drivers.sort(key=lambda x: x.importance_overall, reverse=True)
    return drivers


def rank_by_horizon(drivers: Iterable[Driver], h: int) -> list[Driver]:
    """Return drivers sorted by their importance at horizon h (desc)."""
    return sorted(drivers, key=lambda d: d.importance_by_h.get(h, 0.0), reverse=True)
