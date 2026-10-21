"""Customer-level behavioural features and observation-level (daily) features.

Every feature is defined in docs/methodology.md (units, formula, meaning).
Customer features are computed for one *window* and, for the change group,
compared with the customer's preceding *history* window. Nothing from after
the window end is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import SLOT_COLUMNS, SLOTS_PER_DAY, daily_from_matrix, window_slice

LEVEL_FEATURES = ["mean_daily_kwh", "median_daily_kwh", "std_daily_kwh", "iqr_daily_kwh",
                  "mad_daily_kwh", "p95_daily_kwh", "max_daily_kwh"]
VARIABILITY_FEATURES = ["cv_daily", "mean_abs_change_rel", "high_day_freq", "low_day_freq",
                        "profile_stability"]
TEMPORAL_FEATURES = ["night_share", "morning_share", "afternoon_share", "evening_share",
                     "peak_to_average", "baseload_ratio", "weekend_weekday_ratio", "dow_cv",
                     "peak_hour_sin", "peak_hour_cos"]
CHANGE_FEATURES = ["level_change_log", "peak_change_log", "variability_change", "night_share_change",
                   "weekend_ratio_change_log", "profile_shape_distance", "trend_slope_rel",
                   "half_split_change_log"]

FEATURE_GROUPS = {
    "level": LEVEL_FEATURES,
    "variability": VARIABILITY_FEATURES,
    "temporal": TEMPORAL_FEATURES,
    "change": CHANGE_FEATURES,
}
ALL_FEATURES = LEVEL_FEATURES + VARIABILITY_FEATURES + TEMPORAL_FEATURES + CHANGE_FEATURES
# Features describing *typical* behaviour, used to define peer groups.
PROFILE_FEATURES = LEVEL_FEATURES + VARIABILITY_FEATURES + TEMPORAL_FEATURES
# Strictly positive, right-skewed features modelled on a log scale.
LOG_FEATURES = LEVEL_FEATURES + ["peak_to_average", "weekend_weekday_ratio"]

OBS_RELATIVE_FEATURES = ["z_log_kwh", "z_night_share", "z_log_peak_ratio", "z_day_change"]
OBS_GLOBAL_FEATURES = ["log_kwh", "night_share_day", "log_peak_ratio", "day_change", "dow_sin", "dow_cos"]
OBS_CALENDAR = ["is_weekend"]


def safe_divide(num, den, fallback: float):
    """Element-wise ``num / den`` returning ``fallback`` where ``den`` is ~0 or non-finite."""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    bad = ~np.isfinite(out) | (np.abs(den) < 1e-12)
    return np.where(bad, fallback, out)


def log_ratio(a, b, eps: float):
    """``log((a + eps) / (b + eps))``; zero when both are zero."""
    return np.log((np.asarray(a, dtype="float64") + eps) / (np.asarray(b, dtype="float64") + eps))


def _period_slots(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    return {name: np.arange(2 * lo, 2 * hi) for name, (lo, hi) in cfg["features"]["periods"].items()}


# ---------------------------------------------------------------------------
# Customer-level features
# ---------------------------------------------------------------------------
def _window_basics(matrix: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Daily table (valid days only), per-customer mean load profile, and per-day profiles."""
    daily = daily_from_matrix(matrix, cfg["preprocessing"]["min_day_coverage"])
    valid = daily["valid"].to_numpy()
    days = daily.loc[valid, ["daily_kwh", "is_weekend", "dayofweek"]].copy()
    slots = matrix.loc[valid]
    profile = slots.groupby(level="customer_id").mean()          # mean kWh per local slot
    return days, profile, slots


def _basic_stats(days: pd.DataFrame, profile: pd.DataFrame, slots: pd.DataFrame,
                 cfg: dict[str, Any]) -> pd.DataFrame:
    """Level, variability and temporal features for one window."""
    fcfg = cfg["features"]
    g = days.groupby(level="customer_id")["daily_kwh"]
    out = pd.DataFrame(index=profile.index)
    out["n_valid_days"] = g.size()
    out["mean_daily_kwh"] = g.mean()
    out["median_daily_kwh"] = g.median()
    out["std_daily_kwh"] = g.std(ddof=1).fillna(0.0)
    q75, q25 = g.quantile(0.75), g.quantile(0.25)
    out["iqr_daily_kwh"] = q75 - q25
    absdev = (days["daily_kwh"] - g.transform("median")).abs()
    out["mad_daily_kwh"] = absdev.groupby(level="customer_id").median()
    out["p95_daily_kwh"] = g.quantile(0.95)
    out["max_daily_kwh"] = g.max()

    out["cv_daily"] = safe_divide(out["std_daily_kwh"], out["mean_daily_kwh"], 0.0)
    # Day-to-day change only between consecutive calendar days that are both valid.
    d = days.reset_index()
    d["prev_kwh"] = d.groupby("customer_id")["daily_kwh"].shift(1)
    d["gap"] = d.groupby("customer_id")["date"].diff().dt.days
    consecutive = d["gap"] == 1
    abs_change = (d["daily_kwh"] - d["prev_kwh"]).abs().where(consecutive)
    mean_abs_change = abs_change.groupby(d["customer_id"]).mean().reindex(out.index)
    out["mean_abs_change_rel"] = safe_divide(mean_abs_change.fillna(0.0), out["mean_daily_kwh"], 0.0)
    med = g.transform("median")
    out["high_day_freq"] = (days["daily_kwh"] > fcfg["high_day_factor"] * med).groupby(level="customer_id").mean()
    out["low_day_freq"] = (days["daily_kwh"] < fcfg["low_day_factor"] * med).groupby(level="customer_id").mean()
    out["profile_stability"] = _profile_stability(slots, profile)

    prof = profile[SLOT_COLUMNS].to_numpy(dtype="float64")
    total = prof.sum(axis=1)
    for name, idx in _period_slots(cfg).items():
        out[f"{name}_share"] = safe_divide(prof[:, idx].sum(axis=1), total, len(idx) / SLOTS_PER_DAY)
    mean_slot = prof.mean(axis=1)
    out["peak_to_average"] = safe_divide(prof.max(axis=1), mean_slot, 1.0)
    out["baseload_ratio"] = safe_divide(prof.min(axis=1), mean_slot, 1.0)
    weekend_mean = days.loc[days["is_weekend"], "daily_kwh"].groupby(level="customer_id").mean().reindex(out.index)
    weekday_mean = days.loc[~days["is_weekend"], "daily_kwh"].groupby(level="customer_id").mean().reindex(out.index)
    out["weekend_weekday_ratio"] = np.clip(safe_divide(weekend_mean, weekday_mean, 1.0), 0.0, 10.0)
    dow_means = days.groupby([days.index.get_level_values("customer_id"), "dayofweek"])["daily_kwh"].mean().unstack()
    dow_means = dow_means.reindex(out.index)
    out["dow_cv"] = safe_divide(dow_means.std(axis=1, ddof=0), dow_means.mean(axis=1), 0.0)
    peak_hour = (np.argmax(prof, axis=1) + 0.5) / 2.0
    out["peak_hour_sin"] = np.sin(2 * np.pi * peak_hour / 24)
    out["peak_hour_cos"] = np.cos(2 * np.pi * peak_hour / 24)
    out["_mean_slot"] = mean_slot
    return out


def _profile_stability(slots: pd.DataFrame, profile: pd.DataFrame) -> pd.Series:
    """Mean Pearson correlation between each day's 48-slot profile and the window mean profile.

    Missing slots in a day are filled from the mean profile for this
    calculation only. Days with a constant profile have no defined correlation
    and are skipped; a customer with no defined correlation gets 1.0.
    """
    cust = slots.index.get_level_values("customer_id")
    day = slots.to_numpy(dtype="float64")
    ref = profile.reindex(cust).to_numpy(dtype="float64")
    day = np.where(np.isnan(day), ref, day)
    dc = day - day.mean(axis=1, keepdims=True)
    rc = ref - ref.mean(axis=1, keepdims=True)
    den = np.sqrt((dc**2).sum(axis=1) * (rc**2).sum(axis=1))
    corr = np.where(den > 1e-12, (dc * rc).sum(axis=1) / np.where(den > 1e-12, den, 1.0), np.nan)
    return pd.Series(corr, index=cust).groupby(level=0).mean().reindex(profile.index).fillna(1.0)


def _trend_and_split(days: pd.DataFrame, start: pd.Timestamp, eps: float, mean_kwh: pd.Series) -> pd.DataFrame:
    """Within-window linear trend (relative, per 30 days) and second-half vs first-half change."""
    d = days.reset_index()
    t = (d["date"] - start).dt.days.astype("float64")
    d["t"] = t
    d["ty"] = t * d["daily_kwh"]
    d["tt"] = t * t
    g = d.groupby("customer_id")
    n = g.size()
    st, sy, sty, stt = g["t"].sum(), g["daily_kwh"].sum(), g["ty"].sum(), g["tt"].sum()
    var_t = stt - st**2 / n
    slope = safe_divide(sty - st * sy / n, var_t, 0.0)
    out = pd.DataFrame(index=n.index)
    out["trend_slope_rel"] = safe_divide(slope * 30.0, mean_kwh.reindex(n.index), 0.0)
    mid = d["t"].max() / 2 if len(d) else 0
    first = d.loc[d["t"] <= mid].groupby("customer_id")["daily_kwh"].mean().reindex(n.index)
    second = d.loc[d["t"] > mid].groupby("customer_id")["daily_kwh"].mean().reindex(n.index)
    out["half_split_change_log"] = np.nan_to_num(log_ratio(second, first, eps), nan=0.0)
    return out


def customer_features(slot_matrix: pd.DataFrame, window: tuple[pd.Timestamp, pd.Timestamp],
                      history: tuple[pd.Timestamp, pd.Timestamp], cfg: dict[str, Any],
                      customers: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Behavioural feature vector per customer for ``window`` relative to ``history``.

    Args:
        slot_matrix: (customer_id, date) x 48 matrix.
        window: Inclusive (start, end) of the observation window.
        history: Inclusive (start, end) of the preceding reference window.
        cfg: Configuration.
        customers: Restrict to these customers (e.g. the eligible set).

    Returns:
        ``(features, profiles)``: features indexed by customer_id with
        ``ALL_FEATURES`` plus ``n_valid_days``; profiles holds the window's mean
        48-slot load profile for visualisation. Customers without valid days in
        either window are omitted.
    """
    if history[1] >= window[0]:
        raise ValueError("History window must end before the observation window starts (no look-ahead).")
    if customers is not None:
        slot_matrix = slot_matrix.loc[slot_matrix.index.get_level_values("customer_id").isin(customers)]
    eps = cfg["features"]["log_epsilon"]

    w_days, w_prof, w_slots = _window_basics(window_slice(slot_matrix, *window), cfg)
    h_days, h_prof, h_slots = _window_basics(window_slice(slot_matrix, *history), cfg)
    cur = _basic_stats(w_days, w_prof, w_slots, cfg)
    hist = _basic_stats(h_days, h_prof, h_slots, cfg)
    common = cur.index.intersection(hist.index)
    cur, hist = cur.loc[common], hist.loc[common]

    feats = cur[["n_valid_days"] + LEVEL_FEATURES + VARIABILITY_FEATURES + TEMPORAL_FEATURES].copy()
    feats["level_change_log"] = log_ratio(cur["mean_daily_kwh"], hist["mean_daily_kwh"], eps)
    feats["peak_change_log"] = log_ratio(cur["p95_daily_kwh"], hist["p95_daily_kwh"], eps)
    feats["variability_change"] = cur["cv_daily"] - hist["cv_daily"]
    feats["night_share_change"] = cur["night_share"] - hist["night_share"]
    feats["weekend_ratio_change_log"] = np.log(cur["weekend_weekday_ratio"].clip(lower=1e-3)
                                               / hist["weekend_weekday_ratio"].clip(lower=1e-3))
    wp = w_prof.loc[common, SLOT_COLUMNS].to_numpy(dtype="float64")
    hp = h_prof.loc[common, SLOT_COLUMNS].to_numpy(dtype="float64")
    wc, hc = wp - wp.mean(1, keepdims=True), hp - hp.mean(1, keepdims=True)
    den = np.sqrt((wc**2).sum(1) * (hc**2).sum(1))
    corr = safe_divide((wc * hc).sum(1), den, 1.0)
    feats["profile_shape_distance"] = 1.0 - corr
    ts = _trend_and_split(w_days.loc[w_days.index.get_level_values("customer_id").isin(common)],
                          window[0], eps, cur["mean_daily_kwh"])
    feats = feats.join(ts)
    feats[["trend_slope_rel", "half_split_change_log"]] = feats[["trend_slope_rel", "half_split_change_log"]].fillna(0.0)
    feats.index.name = "customer_id"
    if not np.isfinite(feats[ALL_FEATURES].to_numpy()).all():
        bad = feats[ALL_FEATURES].columns[~np.isfinite(feats[ALL_FEATURES].to_numpy()).all(axis=0)].tolist()
        raise FloatingPointError(f"Non-finite customer features produced: {bad}")
    return feats, w_prof.loc[common, SLOT_COLUMNS]


# ---------------------------------------------------------------------------
# Observation-level (daily) features
# ---------------------------------------------------------------------------
def _day_table(matrix: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Per-day raw features for valid days."""
    eps = cfg["features"]["log_epsilon"]
    daily = daily_from_matrix(matrix, cfg["preprocessing"]["min_day_coverage"])
    values = matrix.to_numpy(dtype="float64")
    n_obs = np.isfinite(values).sum(axis=1)
    mean_slot = safe_divide(np.nansum(values, axis=1), n_obs, 0.0)
    night = _period_slots(cfg)["night"]
    night_mean = safe_divide(np.nansum(values[:, night], axis=1), np.isfinite(values[:, night]).sum(axis=1), 0.0)
    day = pd.DataFrame(index=matrix.index)
    day["valid"] = daily["valid"].to_numpy()
    day["daily_kwh"] = daily["daily_kwh"].to_numpy()
    day["is_weekend"] = daily["is_weekend"].to_numpy().astype(float)
    dow = daily["dayofweek"].to_numpy()
    day["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    day["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    day["log_kwh"] = np.log(day["daily_kwh"] + eps)
    day["night_share_day"] = safe_divide(night_mean * len(night), mean_slot * SLOTS_PER_DAY, len(night) / SLOTS_PER_DAY)
    with np.errstate(all="ignore"):
        day["log_peak_ratio"] = np.log(safe_divide(np.nanmax(np.where(np.isfinite(values), values, -np.inf), axis=1),
                                                   mean_slot, 1.0).clip(min=1.0))
    prev = day.groupby(level="customer_id")["log_kwh"].shift(1)
    day["day_change"] = (day["log_kwh"] - prev).fillna(0.0)
    return day.loc[day["valid"]].drop(columns="valid")


def observation_features(slot_matrix: pd.DataFrame, window: tuple[pd.Timestamp, pd.Timestamp],
                         history: tuple[pd.Timestamp, pd.Timestamp], cfg: dict[str, Any],
                         customers: list[str] | None = None) -> pd.DataFrame:
    """Daily observation features for ``window``.

    Returns both representations:

    * global: raw daily values plus day-of-week encoding;
    * relative: robust z-scores of each daily value against the same
      customer's median and MAD in ``history`` for the same day type
      (weekday/weekend). Each day is still scored independently; history only
      supplies the scale.

    Customer ID is never a feature.
    """
    if history[1] >= window[0]:
        raise ValueError("History window must end before the observation window starts (no look-ahead).")
    if customers is not None:
        slot_matrix = slot_matrix.loc[slot_matrix.index.get_level_values("customer_id").isin(customers)]
    # Include the day before the window so the first day's day-to-day change is defined.
    start = window[0] - pd.Timedelta(days=1)
    cur = _day_table(window_slice(slot_matrix, start, window[1]), cfg)
    cur = cur.loc[cur.index.get_level_values("date") >= window[0]]
    hist = _day_table(window_slice(slot_matrix, *history), cfg)

    floors = {"log_kwh": 0.05, "night_share_day": 0.01, "log_peak_ratio": 0.05, "day_change": 0.05}
    names = {"log_kwh": "z_log_kwh", "night_share_day": "z_night_share",
             "log_peak_ratio": "z_log_peak_ratio", "day_change": "z_day_change"}
    hkey = [hist.index.get_level_values("customer_id"), hist["is_weekend"]]
    ckey = pd.MultiIndex.from_arrays([cur.index.get_level_values("customer_id"), cur["is_weekend"]])
    for col, floor in floors.items():
        med = hist.groupby(hkey)[col].median()
        mad = (hist[col] - hist.groupby(hkey)[col].transform("median")).abs().groupby(hkey).median()
        scale = np.maximum(1.4826 * mad, floor)
        m = med.reindex(ckey).to_numpy()
        s = scale.reindex(ckey).to_numpy()
        cur[names[col]] = safe_divide(cur[col].to_numpy() - m, s, 0.0)
    # Customers without history for a day type cannot be normalised; their z is 0 (neutral).
    cur[list(names.values())] = cur[list(names.values())].fillna(0.0)
    return cur
