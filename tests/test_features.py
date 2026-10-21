"""Feature calculations, denominator handling, leakage and scaling."""

import numpy as np
import pandas as pd
import pytest

from electricity_anomaly.feature_engineering import (ALL_FEATURES, OBS_GLOBAL_FEATURES, OBS_RELATIVE_FEATURES,
                                                     customer_features, log_ratio, observation_features, safe_divide)
from electricity_anomaly.scoring import RobustFeatureScaler

from conftest import make_matrix

HIST = (pd.Timestamp("2013-01-07"), pd.Timestamp("2013-02-03"))
WIN = (pd.Timestamp("2013-02-04"), pd.Timestamp("2013-03-03"))


def _weekday_weekend(n_days, weekday, weekend, start="2013-01-07"):
    dates = pd.date_range(start, periods=n_days)
    return np.where(dates.dayofweek >= 5, weekend, weekday).astype(float)


def test_known_feature_values(small_cfg):
    shape = np.ones(48)
    shape[:12] = 3.0                                   # night slots carry 3x the load
    mat = make_matrix({"A": np.r_[np.full(28, 10.0), _weekday_weekend(28, 10.0, 20.0, "2013-02-04")]}, shape=shape)
    feats, prof = customer_features(mat, WIN, HIST, small_cfg)
    f = feats.loc["A"]
    assert f["weekend_weekday_ratio"] == pytest.approx(2.0, rel=1e-4)
    assert f["night_share"] == pytest.approx(36 / 72, rel=1e-4)
    assert f["peak_to_average"] == pytest.approx(3 / 1.5, rel=1e-4)
    assert f["n_valid_days"] == 28
    expected_level = np.log((f["mean_daily_kwh"] + 0.1) / (10 + 0.1))
    assert f["level_change_log"] == pytest.approx(expected_level, rel=1e-6)
    assert f["profile_shape_distance"] == pytest.approx(0.0, abs=1e-6)
    assert prof.shape == (1, 48)


def test_constant_and_zero_histories_are_finite(small_cfg):
    mat = make_matrix({"CONST": np.full(56, 8.0), "ZERO": np.zeros(56)})
    feats, _ = customer_features(mat, WIN, HIST, small_cfg)
    assert np.isfinite(feats[ALL_FEATURES].to_numpy()).all()
    c, z = feats.loc["CONST"], feats.loc["ZERO"]
    assert c["cv_daily"] == 0 and c["std_daily_kwh"] == 0 and c["mad_daily_kwh"] == 0
    assert c["profile_stability"] == 1.0                     # constant profile: correlation undefined -> 1
    assert c["night_share"] == pytest.approx(12 / 48)
    assert z["peak_to_average"] == 1.0 and z["night_share"] == pytest.approx(12 / 48)
    assert z["level_change_log"] == 0.0 and z["trend_slope_rel"] == 0.0


def test_trend_and_half_split_detect_drift(small_cfg):
    ramp = np.r_[np.full(28, 10.0), np.linspace(10, 20, 28)]
    feats, _ = customer_features(make_matrix({"A": ramp}), WIN, HIST, small_cfg)
    assert feats.loc["A", "trend_slope_rel"] > 0.2
    assert feats.loc["A", "half_split_change_log"] > 0.2


def test_history_must_precede_window(small_cfg):
    mat = make_matrix({"A": np.full(56, 8.0)})
    with pytest.raises(ValueError, match="look-ahead"):
        customer_features(mat, HIST, WIN, small_cfg)
    with pytest.raises(ValueError, match="look-ahead"):
        observation_features(mat, HIST, WIN, small_cfg)


def test_future_readings_do_not_change_features(small_cfg):
    rng = np.random.default_rng(0)
    base = rng.uniform(5, 15, 84)
    future = base.copy()
    future[56:] *= 10                                      # change only data after the window end
    f1, _ = customer_features(make_matrix({"A": base}), WIN, HIST, small_cfg)
    f2, _ = customer_features(make_matrix({"A": future}), WIN, HIST, small_cfg)
    pd.testing.assert_frame_equal(f1, f2)
    o1 = observation_features(make_matrix({"A": base}), WIN, HIST, small_cfg)
    o2 = observation_features(make_matrix({"A": future}), WIN, HIST, small_cfg)
    pd.testing.assert_frame_equal(o1, o2)


def test_observation_features_relative_to_own_history(small_cfg):
    daily = np.full(56, 10.0)
    daily[40] = 40.0                                       # one spike day in the window
    obs = observation_features(make_matrix({"A": daily}), WIN, HIST, small_cfg)
    assert set(OBS_RELATIVE_FEATURES + OBS_GLOBAL_FEATURES) <= set(obs.columns)
    assert "customer_id" not in obs.columns                # the ID is only an index, never a feature
    spike = obs.loc[("A", pd.Timestamp("2013-01-07") + pd.Timedelta(days=40))]
    others = obs.drop(index=("A", pd.Timestamp("2013-01-07") + pd.Timedelta(days=40)))
    assert spike["z_log_kwh"] > 10 and others["z_log_kwh"].abs().max() < 1e-6
    assert np.isfinite(obs[OBS_RELATIVE_FEATURES].to_numpy()).all()


def test_missing_days_reduce_valid_day_count(small_cfg):
    mat = make_matrix({"A": np.full(56, 8.0)})
    mat.loc[("A", pd.Timestamp("2013-02-10"))] = np.nan
    feats, _ = customer_features(mat, WIN, HIST, small_cfg)
    assert feats.loc["A", "n_valid_days"] == 27


def test_safe_divide_and_log_ratio():
    out = safe_divide([1.0, 1.0, 0.0], [2.0, 0.0, 0.0], fallback=-1.0)
    assert out.tolist() == [0.5, -1.0, -1.0]
    assert log_ratio(0.0, 0.0, 0.1) == 0.0


def test_robust_scaler_finite_and_alignment():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.lognormal(2, 0.5, 200), "b": rng.normal(0, 1, 200), "c": np.ones(200)})
    sc = RobustFeatureScaler(["a", "b", "c"], log_features=["a"], align=True).fit(train)
    z = sc.transform(train)
    assert np.isfinite(z.to_numpy()).all()
    assert z.median().abs().max() < 1e-9                   # centred on the training median
    assert (z["c"] == 0).all()                             # zero IQR handled
    shifted = train.assign(b=train["b"] + 5)               # population-wide shift
    assert sc.transform(shifted)["b"].median() == pytest.approx(0, abs=1e-9)
    assert sc.transform(shifted, align=False)["b"].median() > 2
    with_nan = train.copy()
    with_nan.iloc[0, 1] = np.nan
    assert np.isfinite(sc.transform(with_nan).to_numpy()).all()
