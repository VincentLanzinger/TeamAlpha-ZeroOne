"""Thin wrapper over the official `sybilion` SDK with SHA-256 disk caching.

Design notes
------------
- `check_account()` is the FREE balance gate (`/me` + `/usage`). Call before any
  forecast submit — submit_and_wait_forecast() does this automatically.
- `submit_and_wait_forecast()` always goes through `cache/<sha256>/`. The cache
  key is SHA-256 of the canonicalised JSON body (sorted keys, no whitespace) so
  identical requests on different machines hash identically.
- Live demo path: forecasts come from cache; only `/drivers`, `/alerts`,
  `/regions`, `/categories`, `/me` are called live (they're sync).
- `get_drivers()` wraps the SDK call whose return annotation is incorrectly
  declared as `None`; the SDK actually returns a pydantic model.
- `/alerts` body uses `metadata` (NOT `timeseries_metadata`) — the SDK handles
  this internally; we expose a keyword-only function that mirrors the SDK.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from sybilion.client import (
    Client,
    Filters,
    ForecastRequestV1,
    RecommendRequestV1,
    TimeseriesMetadata,
)

from src import config

load_dotenv()  # populate os.environ from .env if present

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "cache"
ARTIFACT_NAMES: tuple[str, ...] = (
    "forecast.json",
    "external_signals.json",
    "backtest_metrics.json",
    "backtest_trajectories.json",
    "input.json",
)


# -- env / client ------------------------------------------------------------

def _resolve_token() -> str:
    for name in config.API_KEY_ENV_NAMES:
        v = os.environ.get(name)
        if v:
            return v.strip()
    raise RuntimeError(
        "No Sybilion token in env. Set one of "
        f"{config.API_KEY_ENV_NAMES} in .env or your shell."
    )


def has_token() -> bool:
    return any(os.environ.get(n) for n in config.API_KEY_ENV_NAMES)


def make_client(token: str | None = None) -> Client:
    return Client(token=token or _resolve_token())


# -- cache -------------------------------------------------------------------

def canonical_body_json(body: dict[str, Any]) -> str:
    """Stable serialisation safe for hashing across processes/machines."""
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def cache_key(body: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical request body."""
    return hashlib.sha256(canonical_body_json(body).encode("utf-8")).hexdigest()


def cache_dir_for(body: dict[str, Any]) -> Path:
    return CACHE_DIR / cache_key(body)


def _load_cached(cdir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"_cache_hit": True, "_cache_dir": str(cdir)}
    for name in ARTIFACT_NAMES:
        p = cdir / name
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    body_p = cdir / "_request_body.json"
    if body_p.exists():
        out["_request_body"] = json.loads(body_p.read_text(encoding="utf-8"))
    return out


def _save_to_cache(
    cdir: Path,
    body: dict[str, Any],
    artifacts: dict[str, bytes],
) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "_request_body.json").write_text(
        json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    for name, raw in artifacts.items():
        try:
            payload = json.loads(raw.decode("utf-8"))
            (cdir / name).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            (cdir / name).write_bytes(bytes(raw))


# -- typed body builders -----------------------------------------------------

def _make_metadata(
    title: str, description: str | None, keywords: Iterable[str]
) -> TimeseriesMetadata:
    kws = list(keywords) if keywords else None
    return TimeseriesMetadata(title=title, description=description, keywords=kws)


def _maybe_filters(
    regions: list[int] | None,
    categories: list[int] | None,
    limit: int | None,
) -> Filters | None:
    if not regions and not categories and limit is None:
        return None
    return Filters(regions=regions, categories=categories, limit=limit)


def build_forecast_body(
    timeseries: dict[str, float],
    *,
    title: str,
    description: str | None,
    keywords: Iterable[str],
    soft_horizon: int = config.SOFT_HORIZON,
    recency_factor: float = config.RECENCY_FACTOR,
    backtest: bool = config.BACKTEST,
    strictly_positive: bool = config.STRICTLY_POSITIVE,
    regions: list[int] | None = None,
    categories: list[int] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build a canonical JSON-serialisable forecast request body.

    Validates via the SDK's pydantic models, then dumps to a dict so callers can
    hash / cache / log it before submission.
    """
    req = ForecastRequestV1(
        pipeline_version=config.PIPELINE_VERSION,
        frequency=config.FREQUENCY,
        soft_horizon=soft_horizon,
        recency_factor=recency_factor,
        backtest=backtest,
        strictly_positive=strictly_positive,
        timeseries=timeseries,
        timeseries_metadata=_make_metadata(title, description, keywords),
        filters=_maybe_filters(regions, categories, limit),
    )
    return req.to_dict()


def build_drivers_body(
    *,
    title: str,
    description: str | None,
    keywords: Iterable[str],
    timeseries: dict[str, float] | None = None,
    recency_factor: float = config.RECENCY_FACTOR,
    regions: list[int] | None = None,
    categories: list[int] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    req = RecommendRequestV1(
        version=config.PIPELINE_VERSION,
        recency_factor=recency_factor,
        timeseries_metadata=_make_metadata(title, description, keywords),
        timeseries=timeseries,
        filters=_maybe_filters(regions, categories, limit),
    )
    return req.to_dict()


# -- public API --------------------------------------------------------------

@dataclass(frozen=True)
class AccountInfo:
    user_id: str
    balance_eur_cents: int
    available_eur_cents: int
    api_usage_tier: int
    recent_usage_events: int
    recent_spend_eur_cents: int

    @property
    def available_eur(self) -> float:
        return self.available_eur_cents / 100

    @property
    def balance_eur(self) -> float:
        return self.balance_eur_cents / 100

    def summary(self) -> str:
        return (
            f"user_id              = {self.user_id}\n"
            f"api_usage_tier       = {self.api_usage_tier}\n"
            f"balance              = {self.balance_eur:>8.2f} EUR "
            f"({self.balance_eur_cents} cents)\n"
            f"available (spendable)= {self.available_eur:>8.2f} EUR "
            f"({self.available_eur_cents} cents)\n"
            f"recent usage         = {self.recent_usage_events} events, "
            f"{self.recent_spend_eur_cents/100:.2f} EUR billed"
        )


def check_account(token: str | None = None) -> AccountInfo:
    """Hit `/me` and `/usage`. Free, sync, no spend."""
    c = make_client(token)
    me = c.me()
    usage = c.get_usage(limit=20)
    events = list(getattr(usage, "usage_events", []) or [])
    return AccountInfo(
        user_id=str(me.user_id),
        balance_eur_cents=int(me.balance_eur_cents),
        available_eur_cents=int(me.available_eur_cents),
        api_usage_tier=int(me.api_usage_tier),
        recent_usage_events=len(events),
        recent_spend_eur_cents=sum(int(e.eur_cents_charged) for e in events),
    )


def submit_and_wait_forecast(
    body: dict[str, Any],
    *,
    token: str | None = None,
    poll_s: float = config.POLL_INTERVAL_S,
    timeout_s: float = config.POLL_TIMEOUT_S,
    skip_cache: bool = False,
    min_required_eur_cents: int = 5,
) -> dict[str, Any]:
    """Submit (or load from cache), poll until completed, download all artifacts.

    Returns a dict keyed by artifact name (parsed JSON) plus meta keys:
      `_cache_hit`, `_cache_dir`, `_cache_key`, `_job_id` (on cache miss),
      `_request_body`.
    """
    key = cache_key(body)
    cdir = CACHE_DIR / key
    if not skip_cache and cdir.exists() and (cdir / "forecast.json").exists():
        out = _load_cached(cdir)
        out["_cache_key"] = key
        return out

    c = make_client(token)
    me = c.me()
    if int(me.available_eur_cents) < min_required_eur_cents:
        raise RuntimeError(
            f"Balance too low to submit: available_eur_cents={me.available_eur_cents} "
            f"< {min_required_eur_cents}. Top up before proceeding."
        )

    req = ForecastRequestV1(**body)
    submitted = c.submit_forecast(req)
    job_id = submitted.job_id
    c.wait_forecast(job_id, poll_s=poll_s, timeout_s=timeout_s)

    artifacts: dict[str, bytes] = {}
    for name in ARTIFACT_NAMES:
        try:
            raw = c.get_forecast_artifact(job_id, name)
            artifacts[name] = bytes(raw)
        except Exception:  # backtest_* absent if backtest=False; ignore
            continue
    _save_to_cache(cdir, body, artifacts)
    out = _load_cached(cdir)
    out["_cache_key"] = key
    out["_cache_hit"] = False
    out["_job_id"] = str(job_id)
    return out


def get_drivers(body: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
    """POST /api/v1/drivers (SYNC). Returns the parsed model as a dict."""
    c = make_client(token)
    req = RecommendRequestV1(**body)
    raw = c.get_drivers(req)
    if raw is None:
        return {}
    if hasattr(raw, "to_dict"):
        return raw.to_dict()
    return dict(raw)


def get_alerts(
    *,
    title: str,
    description: str | None,
    keywords: Iterable[str],
    context_enriched: bool = True,
    regions: list[int] | None = None,
    categories: list[int] | None = None,
    limit: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """POST /api/v1/alerts (SYNC). Body field is `metadata`, NOT `timeseries_metadata`."""
    c = make_client(token)
    meta = _make_metadata(title, description, keywords)
    filt = _maybe_filters(regions, categories, limit)
    return c.get_alerts(
        metadata=meta,
        context_enriched=context_enriched,
        filters=filt,
        date_from=date_from,
        date_to=date_to,
    )


def list_regions(*, token: str | None = None) -> list[dict[str, Any]]:
    """GET /api/v1/regions — free, sync."""
    c = make_client(token)
    out = c.list_regions()
    items = getattr(out, "items", None) or []
    return [it.to_dict() if hasattr(it, "to_dict") else dict(it) for it in items]


def list_categories(*, token: str | None = None) -> list[dict[str, Any]]:
    """GET /api/v1/categories — free, sync."""
    c = make_client(token)
    out = c.list_categories()
    items = getattr(out, "items", None) or []
    return [it.to_dict() if hasattr(it, "to_dict") else dict(it) for it in items]


# -- CLI ---------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m src.sybilion_client",
        description="hedge-decision-agent — Sybilion client smoke utility",
    )
    p.add_argument(
        "cmd",
        nargs="?",
        default="check",
        choices=("check", "regions", "categories"),
        help="check_account (default) | list regions | list categories",
    )
    args = p.parse_args(argv)
    if args.cmd == "check":
        info = check_account()
        print(info.summary())
    elif args.cmd == "regions":
        for r in list_regions():
            print(f"{r.get('id'):>5}  {r.get('name')}")
    elif args.cmd == "categories":
        for r in list_categories():
            print(f"{r.get('id'):>5}  {r.get('name')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
