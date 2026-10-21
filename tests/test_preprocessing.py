"""Cleaning, duplicates, invalid values, DST handling, aggregation and coverage."""

import numpy as np
import pandas as pd
import pytest

from electricity_anomaly.config import get_windows, with_overrides
from electricity_anomaly.data_loader import detect_columns
from electricity_anomaly.preprocessing import (build_slot_matrix, clean_raw_chunk, coverage_summary,
                                               daily_from_matrix, deduplicate, preprocess)

from conftest import raw_frame


def _clean(rows, cfg):
    raw = raw_frame(rows)
    return clean_raw_chunk(raw, detect_columns(list(raw.columns), cfg["schema"]), cfg)


def test_missing_negative_implausible_and_zero(small_cfg):
    rows = [
        ["A", "Std", "2013-01-07 00:00:00", "0.5"],
        ["A", "Std", "2013-01-07 00:30:00", "Null"],
        ["A", "Std", "2013-01-07 01:00:00", "-0.2"],
        ["A", "Std", "2013-01-07 01:30:00", "99"],
        ["A", "Std", "2013-01-07 02:00:00", "0"],
        ["A", "Std", "2013-01-07 02:30:00", "abc"],
    ]
    df, stats = _clean(rows, small_cfg)
    assert stats["rows_missing_reading"] == 1
    assert stats["rows_negative_reading"] == 1
    assert stats["rows_implausible_reading"] == 1
    assert stats["rows_non_numeric_reading"] == 1
    assert stats["zero_readings_kept"] == 1
    assert sorted(df["kwh"].tolist()) == [0.0, 0.5]      # zero kept, invalid ones dropped


def test_irregular_and_malformed_timestamps(small_cfg):
    rows = [["A", "Std", "2013-01-07 00:13:00", "0.5"], ["A", "Std", "2013-13-45 00:00:00", "0.5"],
            ["A", "Std", "2013-01-07 00:30:00", "0.5"]]
    df, stats = _clean(rows, small_cfg)
    assert stats["rows_irregular_timestamp"] == 1
    assert stats["rows_malformed_timestamp"] == 1
    assert len(df) == 1


def test_duplicates_exact_and_conflicting(small_cfg):
    rows = [["A", "Std", "2013-01-07 00:00:00", "0.5"], ["A", "Std", "2013-01-07 00:00:00", "0.5"],
            ["A", "Std", "2013-01-07 00:30:00", "0.2"], ["A", "Std", "2013-01-07 00:30:00", "0.4"]]
    df, _ = _clean(rows, small_cfg)
    out, stats = deduplicate(df)
    assert stats["exact_duplicate_rows_removed"] == 1
    assert stats["conflicting_duplicate_rows"] == 2
    assert len(out) == 2
    assert out.sort_values("timestamp")["kwh"].iloc[1] == pytest.approx(0.3)


def _full_day_utc(local_date: str, tz="Europe/London"):
    start = pd.Timestamp(local_date).tz_localize(tz).tz_convert("UTC")
    end = (pd.Timestamp(local_date) + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC")
    return pd.date_range(start, end, freq="30min", inclusive="left").tz_localize(None)


def test_dst_spring_forward_day_has_46_local_slots(small_cfg):
    ts = _full_day_utc("2013-03-31")
    assert len(ts) == 46                                  # a 23-hour local day
    rows = [["A", "Std", str(t), "0.5"] for t in ts]
    df, _ = _clean(rows, small_cfg)
    mat, stats = build_slot_matrix(df, pd.Timestamp("2013-03-31"), pd.Timestamp("2013-03-31"))
    row = mat.iloc[0]
    assert row[[2, 3]].isna().all() and row.notna().sum() == 46   # 01:00-02:00 does not exist locally
    daily = daily_from_matrix(mat, 0.9)
    assert daily["valid"].iloc[0]
    assert daily["daily_kwh"].iloc[0] == pytest.approx(0.5 * 48)  # 24-hour equivalent


def test_dst_fall_back_day_repeated_hour_is_not_a_duplicate(small_cfg):
    ts = _full_day_utc("2013-10-27")
    assert len(ts) == 50                                  # a 25-hour local day
    rows = [["A", "Std", str(t), str(0.2 if i < 10 else 0.4)] for i, t in enumerate(ts)]
    df, _ = _clean(rows, small_cfg)
    df, dup = deduplicate(df)
    assert dup["exact_duplicate_rows_removed"] == 0 and dup["conflicting_duplicate_rows"] == 0
    mat, stats = build_slot_matrix(df, pd.Timestamp("2013-10-27"), pd.Timestamp("2013-10-27"))
    assert stats["local_slots_with_repeated_readings"] == 2
    assert mat.iloc[0].notna().all()


def test_local_source_timezone_keeps_wall_clock(small_cfg):
    cfg = with_overrides(small_cfg, {"schema.source_timezone": "Europe/London"})
    df, _ = _clean([["A", "Std", "2013-07-01 08:00:00", "0.5"]], cfg)
    assert df["local_time"].iloc[0] == pd.Timestamp("2013-07-01 08:00:00")
    df2, _ = _clean([["A", "Std", "2013-07-01 08:00:00", "0.5"]], small_cfg)   # UTC source -> BST
    assert df2["local_time"].iloc[0] == pd.Timestamp("2013-07-01 09:00:00")


def test_missing_days_stay_missing_and_coverage_eligibility(small_cfg):
    windows = get_windows(small_cfg)
    days = pd.date_range("2013-01-07", "2013-04-28", freq="D")
    rows = []
    for cid, keep in (("FULL", days), ("PARTIAL", days[days >= "2013-03-20"])):
        for d in keep:
            for t in pd.date_range(d, periods=48, freq="30min"):
                rows.append([cid, "Std", str(t), "0.1"])
    cfg = with_overrides(small_cfg, {"schema.source_timezone": "Europe/London"})
    df, stats = _clean(rows, cfg)
    data = preprocess(df, cfg, windows, stats)
    assert data.coverage.loc["FULL", "eligible"]
    assert not data.coverage.loc["PARTIAL", "eligible"]
    gap = data.daily.loc[("PARTIAL", pd.Timestamp("2013-02-01"))]
    assert gap["n_slots"] == 0 and np.isnan(gap["daily_kwh"])          # not fabricated
    assert data.stats["customers_eligible"] == 1


def test_partial_day_below_coverage_is_invalid():
    mat = pd.DataFrame([np.r_[np.full(40, 0.1), np.full(8, np.nan)]], columns=range(48),
                       index=pd.MultiIndex.from_tuples([("A", pd.Timestamp("2013-01-07"))], names=["customer_id", "date"]))
    d = daily_from_matrix(mat, 0.9)
    assert not d["valid"].iloc[0] and np.isnan(d["daily_kwh"].iloc[0])
    d2 = daily_from_matrix(mat, 0.8)
    assert d2["daily_kwh"].iloc[0] == pytest.approx(0.1 * 48)


def test_coverage_summary_longest_gap():
    idx = pd.MultiIndex.from_product([["A"], pd.date_range("2013-01-01", periods=10)], names=["customer_id", "date"])
    daily = pd.DataFrame({"valid": [True, False, False, False, True, True, False, True, True, True]}, index=idx)
    w = {"w": (pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-10"))}
    cov = coverage_summary(daily, w, 0.7)
    assert cov.loc["A", "longest_missing_run_days"] == 3
    assert cov.loc["A", "coverage_w"] == pytest.approx(0.6) and not cov.loc["A", "eligible"]
