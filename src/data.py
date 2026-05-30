"""Monthly timeseries loader + validator.

Sybilion's monthly forecast requires:
  - YYYY-MM-DD keys, all first-of-month
  - >=60 observations at soft_horizon=6
  - no gaps in the monthly grid
  - latest point within the last 12 months (else submit rejects)

`load_series` returns a pandas DataFrame with columns `date` (datetime64[ns],
first-of-month) and `value` (float). `to_api_payload` converts that to the dict
the API actually wants.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import config


@dataclass(frozen=True)
class SeriesStats:
    start: pd.Timestamp
    end: pd.Timestamp
    n: int
    min: float
    max: float
    mean: float
    months_since_latest: int

    def summary(self, unit: str = "") -> str:
        return (
            f"n             = {self.n}\n"
            f"start         = {self.start.date()}\n"
            f"end           = {self.end.date()}\n"
            f"latest age    = {self.months_since_latest} month(s) ago\n"
            f"min / mean / max = {self.min:.2f} / {self.mean:.2f} / {self.max:.2f} {unit}".rstrip()
        )


class SeriesValidationError(ValueError):
    """Raised when an input series fails an API precondition."""


def load_series(path: str | Path) -> pd.DataFrame:
    """Load a date,value CSV. Coerces date to first-of-month timestamp."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"CSV not found at {p.resolve()}. "
            "Place a 2-column 'date,value' monthly file there."
        )
    df = pd.read_csv(p)
    expected = {"date", "value"}
    if set(df.columns) != expected:
        raise SeriesValidationError(
            f"CSV columns {list(df.columns)} != expected {sorted(expected)}"
        )
    df = df.dropna(subset=["date", "value"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()]
        raise SeriesValidationError(f"Unparseable date(s) in CSV: {bad.head()}")
    # Snap to first-of-month so a raw 2024-01-15 becomes 2024-01-01.
    df["date"] = df["date"].values.astype("datetime64[M]")
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df["value"].isna().any():
        raise SeriesValidationError("Non-numeric value(s) in CSV.")
    df = df.sort_values("date").reset_index(drop=True)
    # Collapse possible duplicate months (keep last observation per month).
    df = df.groupby("date", as_index=False)["value"].last()
    return df


def validate_series(
    df: pd.DataFrame,
    *,
    min_obs: int = config.min_observations(),
    max_months_stale: int = 12,
    reference_today: pd.Timestamp | None = None,
) -> None:
    """Enforce Sybilion's preconditions. Raises SeriesValidationError on failure."""
    if len(df) < min_obs:
        raise SeriesValidationError(
            f"Series has {len(df)} obs, need >= {min_obs} for "
            f"soft_horizon={config.SOFT_HORIZON}."
        )
    # No gaps in the monthly grid.
    expected = pd.date_range(start=df["date"].iloc[0], end=df["date"].iloc[-1], freq="MS")
    actual = pd.DatetimeIndex(df["date"])
    missing = expected.difference(actual)
    if len(missing) > 0:
        sample = [str(m.date()) for m in missing[:5]]
        raise SeriesValidationError(
            f"Series has {len(missing)} gap month(s): first few = {sample}"
        )
    # Every date must be first-of-month.
    bad = df[df["date"].dt.day != 1]
    if len(bad) > 0:
        raise SeriesValidationError(
            f"{len(bad)} date(s) not first-of-month. First: {bad['date'].iloc[0].date()}"
        )
    # Latest point must be within the last `max_months_stale` months.
    today = reference_today or pd.Timestamp.utcnow().tz_localize(None).normalize()
    months_old = months_between(df["date"].iloc[-1], today)
    if months_old > max_months_stale:
        raise SeriesValidationError(
            f"Latest point is {months_old} months old (>{max_months_stale}). "
            "API will reject — refresh the series."
        )


def months_between(earlier: pd.Timestamp, later: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def stats(df: pd.DataFrame, reference_today: pd.Timestamp | None = None) -> SeriesStats:
    today = reference_today or pd.Timestamp.utcnow().tz_localize(None).normalize()
    return SeriesStats(
        start=df["date"].iloc[0],
        end=df["date"].iloc[-1],
        n=len(df),
        min=float(df["value"].min()),
        max=float(df["value"].max()),
        mean=float(df["value"].mean()),
        months_since_latest=months_between(df["date"].iloc[-1], today),
    )


def to_api_payload(df: pd.DataFrame) -> dict[str, float]:
    """Convert the validated DataFrame to the {YYYY-MM-DD: value} dict the API expects."""
    return {
        ts.strftime("%Y-%m-%d"): float(v)
        for ts, v in zip(df["date"], df["value"], strict=True)
    }


def current_spot(df: pd.DataFrame) -> float:
    """Most recent observed value — used as 'spot' in the decision rule."""
    return float(df["value"].iloc[-1])
