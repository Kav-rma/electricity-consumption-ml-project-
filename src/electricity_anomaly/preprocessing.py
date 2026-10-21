"""Cleaning, validation and aggregation of half-hourly meter readings.

Design decisions (see docs/methodology.md for the full rationale):

* Readings are cleaned chunk by chunk (``clean_raw_chunk``) so the real file
  can be streamed. Every exclusion is counted in a ``Counter`` that ends up in
  ``preprocessing_stats.json``; nothing is dropped silently.
* Missing, non-numeric, negative and physically implausible readings become
  missing slots. They are never filled in. A valid zero reading is kept.
* Behavioural features use *local* clock time. Source timestamps in UTC are
  converted to Europe/London, so the spring-forward day has 46 half-hours and
  the fall-back day has 50 (the repeated hour is averaged into its slot).
* The canonical processed table is a *slot matrix*: one row per
  (customer, local date), 48 columns of kWh, NaN where nothing was recorded.
  Daily totals are "24-hour equivalent": mean of observed slots x 48, and only
  for days with at least ``min_day_coverage`` of slots observed.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SLOTS_PER_DAY = 48
SLOT_COLUMNS = list(range(SLOTS_PER_DAY))


@dataclass
class ProcessedData:
    """Output of preprocessing: slot matrix, daily table, coverage and stats."""

    slot_matrix: pd.DataFrame        # index (customer_id, date), columns 0..47
    daily: pd.DataFrame              # index (customer_id, date)
    coverage: pd.DataFrame           # index customer_id
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def eligible_customers(self) -> list[str]:
        return self.coverage.index[self.coverage["eligible"]].tolist()


# ---------------------------------------------------------------------------
# Row-level cleaning
# ---------------------------------------------------------------------------
def clean_raw_chunk(raw: pd.DataFrame, colmap: dict[str, str], cfg: dict[str, Any]) -> tuple[pd.DataFrame, Counter]:
    """Validate and type one chunk of raw readings.

    Args:
        raw: Raw chunk; columns may be strings (as read from CSV) or typed.
        colmap: Mapping of canonical names (customer_id, timestamp, consumption)
            to the column names in ``raw``.
        cfg: Full configuration.

    Returns:
        A frame with columns ``customer_id`` (str), ``timestamp`` (source time,
        naive), ``local_time`` (local wall clock, naive) and ``kwh`` (float32),
        plus a ``Counter`` describing every row that was excluded.
    """
    schema = cfg["schema"]
    max_kwh = float(cfg["preprocessing"]["max_kwh_per_interval"])
    stats: Counter = Counter()
    stats["rows_read"] += len(raw)

    cid = raw[colmap["customer_id"]].astype("string").str.strip()
    missing_id = cid.isna() | (cid == "")

    ts = _parse_timestamps(raw[colmap["timestamp"]])
    bad_ts = ts.isna()

    values, missing_token, non_numeric = _parse_consumption(raw[colmap["consumption"]], schema["missing_value_tokens"])
    negative = values < 0
    implausible = values > max_kwh

    stats["rows_missing_customer_id"] += int(missing_id.sum())
    stats["rows_malformed_timestamp"] += int((bad_ts & ~missing_id).sum())
    usable = ~missing_id & ~bad_ts
    stats["rows_missing_reading"] += int((missing_token & usable).sum())
    stats["rows_non_numeric_reading"] += int((non_numeric & usable).sum())
    stats["rows_negative_reading"] += int((negative & usable).sum())
    stats["rows_implausible_reading"] += int((implausible & usable).sum())
    stats["zero_readings_kept"] += int(((values == 0) & usable).sum())

    keep = usable & values.notna() & ~negative & ~implausible
    out = pd.DataFrame({
        "customer_id": cid[keep].astype(str).to_numpy(),
        "timestamp": ts[keep].to_numpy(),
        "kwh": values[keep].astype("float32").to_numpy(),
    })

    irregular = out["timestamp"].dt.floor("30min") != out["timestamp"]
    stats["rows_irregular_timestamp"] += int(irregular.sum())
    out = out.loc[~irregular].reset_index(drop=True)
    out["local_time"] = to_local_wall_clock(out["timestamp"], schema["source_timezone"], schema["local_timezone"])
    stats["rows_valid"] += len(out)
    return out, stats


def _parse_timestamps(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        ts = series
    else:
        ts = pd.to_datetime(series.astype("string").str.strip(), errors="coerce", format="ISO8601")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts


def _parse_consumption(series: pd.Series, tokens: list[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (values, missing_token_mask, non_numeric_mask)."""
    if pd.api.types.is_numeric_dtype(series):
        values = series.astype("float64")
        missing = values.isna()
        return values, missing, pd.Series(False, index=series.index)
    text = series.astype("string").str.strip()
    missing = text.isna() | text.isin(tokens)
    values = pd.to_numeric(text.where(~missing), errors="coerce").astype("float64")
    non_numeric = values.isna() & ~missing
    return values, missing, non_numeric


def to_local_wall_clock(ts: pd.Series, source_tz: str, local_tz: str) -> pd.Series:
    """Convert naive source timestamps to naive local wall-clock time.

    If the source is already local wall-clock time it is returned unchanged.
    """
    if source_tz == local_tz:
        return ts
    return ts.dt.tz_localize(source_tz).dt.tz_convert(local_tz).dt.tz_localize(None)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    """Remove exact duplicates and resolve conflicting duplicates.

    Duplicates are defined on the *source* timestamp, so the repeated local
    hour on the autumn DST day is not mistaken for a duplicate when the source
    is UTC. Conflicting readings for the same (customer, timestamp) are
    replaced by their mean and counted.
    """
    stats: Counter = Counter()
    exact = df.duplicated(["customer_id", "timestamp", "kwh"])
    stats["exact_duplicate_rows_removed"] = int(exact.sum())
    df = df.loc[~exact]
    conflict = df.duplicated(["customer_id", "timestamp"], keep=False)
    stats["conflicting_duplicate_rows"] = int(conflict.sum())
    if conflict.any():
        resolved = (df.loc[conflict]
                    .groupby(["customer_id", "timestamp"], as_index=False, observed=True)
                    .agg(kwh=("kwh", "mean"), local_time=("local_time", "first")))
        stats["conflicting_duplicate_keys_resolved_by_mean"] = len(resolved)
        df = pd.concat([df.loc[~conflict], resolved], ignore_index=True)
    return df.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def build_slot_matrix(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, Counter]:
    """Pivot clean readings into a (customer, local date) x 48 slot matrix.

    Every customer gets a row for every calendar date in [start, end]; dates
    without readings are all-NaN rows so gaps stay visible.
    """
    stats: Counter = Counter()
    local = df["local_time"]
    in_range = (local >= start) & (local < end + pd.Timedelta(days=1))
    stats["rows_outside_study_period"] = int((~in_range).sum())
    df = df.loc[in_range]
    date = df["local_time"].dt.normalize()
    slot = (df["local_time"].dt.hour * 2 + df["local_time"].dt.minute // 30).astype("int16")
    keys = pd.DataFrame({"customer_id": df["customer_id"].to_numpy(), "date": date.to_numpy(),
                         "slot": slot.to_numpy(), "kwh": df["kwh"].to_numpy()})
    grouped = keys.groupby(["customer_id", "date", "slot"], sort=True)["kwh"]
    sizes = grouped.size()
    stats["local_slots_with_repeated_readings"] = int((sizes > 1).sum())
    matrix = grouped.mean().unstack("slot")
    matrix = matrix.reindex(columns=SLOT_COLUMNS)

    customers = sorted(keys["customer_id"].unique())
    dates = pd.date_range(start, end, freq="D")
    full_index = pd.MultiIndex.from_product([customers, dates], names=["customer_id", "date"])
    matrix = matrix.reindex(full_index).astype("float32")
    matrix.columns = SLOT_COLUMNS
    stats["customer_days_in_matrix"] = len(matrix)
    return matrix, stats


def daily_from_matrix(matrix: pd.DataFrame, min_day_coverage: float) -> pd.DataFrame:
    """Daily coverage and 24-hour-equivalent consumption for each customer-day."""
    values = matrix.to_numpy(dtype="float64")
    n_slots = np.isfinite(values).sum(axis=1)
    coverage = n_slots / SLOTS_PER_DAY
    valid = coverage >= min_day_coverage
    with np.errstate(invalid="ignore"):
        mean_slot = np.where(n_slots > 0, np.nansum(values, axis=1) / np.maximum(n_slots, 1), np.nan)
    daily_kwh = np.where(valid, mean_slot * SLOTS_PER_DAY, np.nan)
    dates = matrix.index.get_level_values("date")
    return pd.DataFrame({
        "n_slots": n_slots.astype("int16"),
        "coverage": coverage,
        "valid": valid,
        "daily_kwh": daily_kwh,
        "dayofweek": dates.dayofweek,
        "is_weekend": dates.dayofweek >= 5,
    }, index=matrix.index)


def _longest_run(flags: np.ndarray) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def coverage_summary(daily: pd.DataFrame, windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
                     min_window_coverage: float) -> pd.DataFrame:
    """Per-customer coverage for each window and overall eligibility.

    A customer is eligible only if every window has at least
    ``min_window_coverage`` valid days, so both approaches are evaluated on
    exactly the same customers.
    """
    dates = daily.index.get_level_values("date")
    rows = {}
    for name, (start, end) in windows.items():
        in_window = (dates >= start) & (dates <= end)
        rows[f"coverage_{name}"] = daily.loc[in_window, "valid"].groupby(level="customer_id").mean()
    cov = pd.DataFrame(rows)
    grouped = daily["valid"].groupby(level="customer_id")
    cov["valid_days"] = grouped.sum()
    cov["total_days"] = grouped.size()
    cov["longest_missing_run_days"] = grouped.agg(lambda s: _longest_run(~s.to_numpy()))
    valid_dates = daily.index[daily["valid"].to_numpy()].to_frame(index=False)
    span = valid_dates.groupby("customer_id")["date"].agg(["min", "max"])
    cov["first_valid_date"] = span["min"]
    cov["last_valid_date"] = span["max"]
    window_cols = [c for c in cov.columns if c.startswith("coverage_")]
    cov[window_cols] = cov[window_cols].fillna(0.0)
    cov["eligible"] = (cov[window_cols] >= min_window_coverage).all(axis=1)
    return cov


def preprocess(clean_long: pd.DataFrame, cfg: dict[str, Any], windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
               upstream_stats: Counter | None = None) -> ProcessedData:
    """Run deduplication, slot aggregation, daily aggregation and coverage checks.

    Args:
        clean_long: Concatenated output of ``clean_raw_chunk``.
        cfg: Full configuration.
        windows: Chronological windows from ``config.get_windows``.
        upstream_stats: Counts from loading / chunk cleaning, merged into the
            returned statistics.
    """
    stats: Counter = Counter(upstream_stats or {})
    stats["customers_after_cleaning"] = int(clean_long["customer_id"].nunique())
    deduped, dup_stats = deduplicate(clean_long)
    stats.update(dup_stats)

    start = min(w[0] for w in windows.values())
    end = max(w[1] for w in windows.values())
    matrix, mat_stats = build_slot_matrix(deduped, start, end)
    stats.update(mat_stats)

    pre = cfg["preprocessing"]
    daily = daily_from_matrix(matrix, pre["min_day_coverage"])
    coverage = coverage_summary(daily, windows, pre["min_window_day_coverage"])

    stats["customer_days_valid"] = int(daily["valid"].sum())
    stats["customer_days_partial_excluded"] = int(((daily["n_slots"] > 0) & ~daily["valid"]).sum())
    stats["customer_days_no_data"] = int((daily["n_slots"] == 0).sum())
    stats["missing_slots_in_matrix"] = int(matrix.isna().to_numpy().sum())
    stats["customers_in_study_period"] = int(len(coverage))
    stats["customers_eligible"] = int(coverage["eligible"].sum())
    stats["customers_ineligible_low_coverage"] = int((~coverage["eligible"]).sum())
    logger.info("Preprocessing: %d customers in study period, %d eligible", len(coverage), coverage["eligible"].sum())
    return ProcessedData(slot_matrix=matrix, daily=daily, coverage=coverage,
                         stats={k: int(v) if isinstance(v, (int, np.integer)) else v for k, v in stats.items()})


def restrict_customers(data: ProcessedData, customers: list[str]) -> ProcessedData:
    """Return a copy of ``data`` restricted to ``customers``."""
    keep = sorted(customers)
    matrix = data.slot_matrix.loc[data.slot_matrix.index.get_level_values("customer_id").isin(keep)]
    daily = data.daily.loc[matrix.index]
    stats = dict(data.stats)
    stats["customers_retained_after_sampling"] = len(keep)
    return ProcessedData(slot_matrix=matrix, daily=daily, coverage=data.coverage.loc[keep], stats=stats)


def window_slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Rows of a (customer_id, date)-indexed frame within [start, end]."""
    dates = frame.index.get_level_values("date")
    return frame.loc[(dates >= start) & (dates <= end)]
