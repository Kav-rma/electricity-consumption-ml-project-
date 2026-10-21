"""Synthetic half-hourly household consumption with realistic data defects.

The generator is used by tests, the dashboard demo and the controlled
experiments. It produces readings in the same raw schema as the London file
(``LCLid, stdorToU, DateTime, KWH/hh (per half hour)``) with UTC timestamps,
so the identical ingestion and cleaning code path is exercised.

Behaviour model (per customer):
    slot kWh = daily level x seasonal factor x weekday/weekend factor
               x shared weather shock x AR(1) day noise x load shape(slot)
               x multiplicative reading noise
Households belong to one of five archetypes with different daily load
shapes. A minority have an away period (a legitimate low-consumption spell),
so the clean data already contains benign irregularities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import SLOTS_PER_DAY

logger = logging.getLogger(__name__)

RAW_COLUMNS = ["LCLid", "stdorToU", "DateTime", "KWH/hh (per half hour) "]

# (probability, level multiplier, seasonal amplitude, bumps as (hour, width_h, weight))
ARCHETYPES: dict[str, dict[str, Any]] = {
    "commuter": dict(p=0.35, level=1.0, season=0.20,
                     weekday=[(7.5, 1.0, 1.0), (19.5, 2.0, 2.2)], weekend=[(10.0, 2.0, 1.2), (13.5, 2.5, 1.0), (19.0, 2.0, 2.0)]),
    "home_daytime": dict(p=0.25, level=1.15, season=0.20,
                         weekday=[(9.0, 2.0, 1.0), (13.0, 2.5, 1.3), (19.0, 2.0, 1.8)], weekend=[(10.0, 2.0, 1.2), (13.5, 2.5, 1.2), (19.0, 2.0, 1.8)]),
    "early_retired": dict(p=0.20, level=0.85, season=0.25,
                          weekday=[(7.0, 1.5, 1.2), (12.0, 2.0, 1.2), (17.5, 2.0, 1.6)], weekend=[(7.5, 1.5, 1.2), (12.5, 2.0, 1.3), (17.5, 2.0, 1.6)]),
    "night_storage": dict(p=0.08, level=1.6, season=0.45,
                          weekday=[(3.0, 2.2, 3.0), (8.0, 1.0, 0.6), (19.0, 2.0, 1.2)], weekend=[(3.0, 2.2, 3.0), (11.0, 2.0, 0.8), (19.0, 2.0, 1.2)]),
    "low_user": dict(p=0.12, level=0.45, season=0.10,
                     weekday=[(8.0, 1.0, 0.6), (20.0, 2.0, 1.0)], weekend=[(11.0, 2.0, 0.8), (20.0, 2.0, 1.0)]),
}


def _shape(bumps: list[tuple[float, float, float]], baseload: float) -> np.ndarray:
    """48-slot normalised load shape from circular Gaussian bumps plus baseload."""
    hours = (np.arange(SLOTS_PER_DAY) + 0.5) / 2.0
    shape = np.full(SLOTS_PER_DAY, baseload)
    for centre, width, weight in bumps:
        d = np.minimum(np.abs(hours - centre), 24 - np.abs(hours - centre))
        shape += weight * np.exp(-0.5 * (d / width) ** 2)
    return shape / shape.sum()


def _ar1(rng: np.random.Generator, n: int, rho: float, sigma: float) -> np.ndarray:
    eps = rng.normal(0.0, sigma * np.sqrt(1 - rho**2), n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + eps[t]
    return out


def generate_local_matrix(n_customers: int, start: str, end: str, seed: int,
                          holiday_probability: float = 0.3) -> tuple[np.ndarray, pd.DatetimeIndex, pd.DataFrame]:
    """Clean behavioural slot matrix in local time.

    Returns:
        ``(values, dates, customers)`` where ``values`` has shape
        (n_customers, n_days, 48) in kWh per half hour and ``customers``
        describes each household's archetype and parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)
    names = list(ARCHETYPES)
    probs = np.array([ARCHETYPES[a]["p"] for a in names])
    archetype = rng.choice(names, size=n_customers, p=probs / probs.sum())

    doy = dates.dayofyear.to_numpy()
    winter = np.cos(2 * np.pi * (doy - 15) / 365.25)          # +1 mid-January, -1 mid-July
    weather = _ar1(rng, n_days, rho=0.8, sigma=0.06)           # shared by all households
    weekend = np.asarray(dates.dayofweek >= 5)

    values = np.empty((n_customers, n_days, SLOTS_PER_DAY))
    rows = []
    for c in range(n_customers):
        spec = ARCHETYPES[archetype[c]]
        level = spec["level"] * rng.lognormal(np.log(9.0), 0.35)      # kWh/day
        season_amp = spec["season"] * rng.uniform(0.6, 1.4)
        weekend_factor = rng.normal(1.08, 0.06)
        noise_sd = rng.uniform(0.08, 0.22)
        baseload = rng.uniform(0.15, 0.45)
        jitter = rng.normal(0, 0.4, size=8)
        wd_bumps = [(h + jitter[i], w, wt * rng.uniform(0.8, 1.2)) for i, (h, w, wt) in enumerate(spec["weekday"])]
        we_bumps = [(h + jitter[4 + i % 4], w, wt * rng.uniform(0.8, 1.2)) for i, (h, w, wt) in enumerate(spec["weekend"])]
        shape_wd = _shape(wd_bumps, baseload)
        shape_we = _shape(we_bumps, baseload)

        daily = (level * (1 + season_amp * winter) * np.where(weekend, weekend_factor, 1.0)
                 * np.exp(weather * rng.uniform(0.5, 1.5)) * np.exp(_ar1(rng, n_days, 0.3, noise_sd)))
        flat = np.full(SLOTS_PER_DAY, 1 / SLOTS_PER_DAY)
        shapes = np.where(weekend[:, None], shape_we[None, :], shape_wd[None, :])
        holiday = None
        if rng.random() < holiday_probability:
            h_start = int(rng.integers(0, n_days - 15))
            h_len = int(rng.integers(7, 15))
            daily[h_start:h_start + h_len] *= rng.uniform(0.2, 0.4)
            shapes[h_start:h_start + h_len] = flat
            holiday = (str(dates[h_start].date()), h_len)
        values[c] = daily[:, None] * shapes
        rows.append(dict(customer_id=f"SYN{c:04d}", archetype=archetype[c], mean_level_kwh_day=level,
                         seasonal_amplitude=season_amp, away_period=holiday))
    return values, dates, pd.DataFrame(rows).set_index("customer_id")


def matrix_to_utc_readings(values: np.ndarray, dates: pd.DatetimeIndex, customer_ids: list[str],
                           local_tz: str, seed: int, reading_noise_shape: float = 6.0) -> pd.DataFrame:
    """Sample half-hourly UTC readings from a local-time slot matrix.

    Each UTC half-hour is mapped to its local (date, slot). On the autumn DST
    day two UTC readings map to the same local slot (a 25-hour day); on the
    spring day two local slots receive no reading (a 23-hour day), exactly as
    a UTC-stamped meter would record.
    """
    rng = np.random.default_rng(seed + 1)
    start_utc = pd.Timestamp(dates[0]).tz_localize(local_tz).tz_convert("UTC")
    end_utc = (pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)).tz_localize(local_tz).tz_convert("UTC")
    utc = pd.date_range(start_utc, end_utc, freq="30min", inclusive="left")
    local = utc.tz_convert(local_tz).tz_localize(None)
    day_idx = ((local.normalize() - dates[0]) // pd.Timedelta(days=1)).to_numpy()
    slot_idx = (local.hour * 2 + local.minute // 30).to_numpy()

    n_c, n_t = len(customer_ids), len(utc)
    kwh = values[:, day_idx, slot_idx]
    kwh = kwh * rng.gamma(reading_noise_shape, 1 / reading_noise_shape, size=(n_c, n_t))
    return pd.DataFrame({
        RAW_COLUMNS[0]: np.repeat(np.asarray(customer_ids, dtype=object), n_t),
        RAW_COLUMNS[1]: "Std",
        RAW_COLUMNS[2]: np.tile(utc.tz_localize(None).to_numpy(), n_c),
        RAW_COLUMNS[3]: np.round(kwh.ravel(), 3),
    })


def add_data_defects(df: pd.DataFrame, cfg_syn: dict[str, Any], seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Inject missing data, outages, late starts, duplicates and invalid values.

    Returns the defective frame and the number of each defect introduced,
    which tests use to check that preprocessing detects them.
    """
    rng = np.random.default_rng(seed + 2)
    df = df.copy()
    kcol, idcol, tcol = RAW_COLUMNS[3], RAW_COLUMNS[0], RAW_COLUMNS[2]
    n = len(df)
    log: dict[str, int] = {}

    missing = rng.random(n) < cfg_syn["missing_slot_rate"]
    df.loc[missing, kcol] = np.nan
    log["missing_slots"] = int(missing.sum())

    customers = df[idcol].unique()
    ts = df[tcol]
    t_min, t_max = ts.min(), ts.max()
    n_outage = int(round(cfg_syn["outage_customer_fraction"] * len(customers)))
    outage_mask = np.zeros(n, dtype=bool)
    for cid in rng.choice(customers, size=n_outage, replace=False):
        length = pd.Timedelta(days=int(rng.integers(2, 21)))
        begin = t_min + (t_max - t_min - length) * rng.random()
        outage_mask |= ((df[idcol] == cid) & (ts >= begin) & (ts < begin + length)).to_numpy()
    log["outage_customers"] = n_outage

    n_late = int(round(cfg_syn["late_start_fraction"] * len(customers)))
    late_customers = rng.choice(customers, size=n_late, replace=False)
    for cid in late_customers:
        first = t_min + pd.Timedelta(days=int(rng.integers(120, 270)))
        outage_mask |= ((df[idcol] == cid) & (ts < first)).to_numpy()
    log["late_start_customers"] = n_late
    df = df.loc[~outage_mask].reset_index(drop=True)
    log["rows_removed_by_outages_and_late_starts"] = int(outage_mask.sum())

    n = len(df)
    neg = rng.random(n) < cfg_syn["negative_value_rate"]
    df.loc[neg & df[kcol].notna(), kcol] = -df.loc[neg & df[kcol].notna(), kcol] - 0.01
    log["negative_values"] = int((neg & df[kcol].notna()).sum())
    big = (rng.random(n) < cfg_syn["implausible_value_rate"]) & ~neg
    df.loc[big, kcol] = 50.0
    log["implausible_values"] = int(big.sum())

    dup_idx = rng.choice(n, size=int(cfg_syn["duplicate_rate"] * n), replace=False)
    conflict_idx = rng.choice(np.setdiff1d(np.arange(n), dup_idx), size=int(cfg_syn["conflicting_duplicate_rate"] * n), replace=False)
    conflicts = df.iloc[conflict_idx].copy()
    conflicts[kcol] = conflicts[kcol] + 0.123
    df = pd.concat([df, df.iloc[dup_idx], conflicts], ignore_index=True)
    log["exact_duplicates"] = len(dup_idx)
    log["conflicting_duplicates"] = len(conflict_idx)
    log["late_start_customer_ids"] = sorted(map(str, late_customers))  # type: ignore[assignment]
    return df, log


def generate_synthetic_readings(cfg: dict[str, Any], seed: int, n_customers: int | None = None,
                                start: str | None = None, end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate raw-schema synthetic readings with defects, in memory.

    Returns:
        ``(raw_readings, customer_metadata, defect_log)``. The metadata is for
        analysis only and is never passed to a model.
    """
    syn = cfg["synthetic"]
    n_customers = n_customers or syn["n_customers"]
    start = start or syn["start_date"]
    end = end or syn["end_date"]
    values, dates, meta = generate_local_matrix(n_customers, start, end, seed, syn["holiday_probability"])
    raw = matrix_to_utc_readings(values, dates, meta.index.tolist(), cfg["schema"]["local_timezone"], seed)
    raw, defects = add_data_defects(raw, syn, seed)
    return raw, meta, defects


def write_synthetic_csv(cfg: dict[str, Any], out_dir: Path, seed: int) -> Path:
    """Write the synthetic dataset as CSV (``Null`` for missing, like the source).

    A few malformed timestamp rows are appended to the file to exercise the
    parser.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, meta, defects = generate_synthetic_readings(cfg, seed)
    path = out_dir / cfg["synthetic"]["csv_name"]
    logger.info("Writing %s synthetic readings to %s", f"{len(raw):,}", path)
    raw.to_csv(path, index=False, na_rep="Null", date_format="%Y-%m-%d %H:%M:%S.0000000")
    n_bad = int(cfg["synthetic"]["malformed_timestamp_rows"])
    with open(path, "a", encoding="utf-8", newline="") as fh:
        for i in range(n_bad):
            fh.write(f"SYN{i:04d},Std,not-a-timestamp,0.250\n")
    defects["malformed_timestamps"] = n_bad
    meta.to_csv(out_dir / "synthetic_customer_metadata.csv")
    pd.Series(defects, dtype=object).to_json(out_dir / "synthetic_defect_log.json", indent=2)
    return path
