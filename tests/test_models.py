"""Model training, scoring direction, determinism, aggregation and leakage-free thresholds."""

import numpy as np
import pandas as pd
import pytest

from electricity_anomaly import experiments
from electricity_anomaly.config import get_windows
from electricity_anomaly.customer_models import CustomerIsolationForest, top_contributions
from electricity_anomaly.feature_engineering import ALL_FEATURES, CHANGE_FEATURES, PROFILE_FEATURES
from electricity_anomaly.observation_models import (ObservationIsolationForest, aggregate_to_customers,
                                                    customer_score_from_observations)
from electricity_anomaly.peer_grouping import PeerGroupModel, select_k
from electricity_anomaly.pipeline import B1, fit_suite, prepare_synthetic, score_window
from electricity_anomaly.scoring import empirical_cdf, f1_optimal_threshold, hybrid_score, quantile_threshold


def _features(n=120, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(0, 1, size=(n, len(ALL_FEATURES))), columns=ALL_FEATURES,
                      index=[f"C{i:03d}" for i in range(n)])
    for f in ["mean_daily_kwh", "median_daily_kwh", "std_daily_kwh", "iqr_daily_kwh", "mad_daily_kwh",
              "p95_daily_kwh", "max_daily_kwh", "peak_to_average", "weekend_weekday_ratio"]:
        df[f] = rng.lognormal(1, 0.3, n)
    return df


def test_customer_iforest_score_direction_and_determinism():
    """Higher score = more anomalous; same seed = same scores.

    Uses the change-feature subset: with all 30 features a deviation in a few
    of them is diluted, because Isolation Forest picks split features at
    random and cannot score values beyond the training range any higher than
    the most extreme training value (documented in docs/limitations.md).
    """
    train = _features()
    test = _features(seed=1)
    for f in ("level_change_log", "peak_change_log", "half_split_change_log", "trend_slope_rel"):
        test.loc["C000", f] = 10.0
    m1 = CustomerIsolationForest(features=CHANGE_FEATURES, n_estimators=100, random_state=3).fit(train)
    m2 = CustomerIsolationForest(features=CHANGE_FEATURES, n_estimators=100, random_state=3).fit(train)
    s = m1.score(test)
    assert s.idxmax() == "C000"
    pd.testing.assert_series_equal(s, m2.score(test))
    contrib = m1.feature_contributions(test.loc[["C000"]])
    changed = {"level_change_log", "peak_change_log", "half_split_change_log", "trend_slope_rel"}
    assert contrib.loc["C000"].idxmax() in changed
    assert contrib.loc["C000"].idxmax() in top_contributions(contrib, test.loc[["C000"]]).loc["C000"]
    full = CustomerIsolationForest(n_estimators=100, random_state=3).fit(train)
    assert np.isfinite(full.score(test)).all()


def test_peer_model_scores_outlier_highest_and_selects_k():
    rng = np.random.default_rng(0)
    train = _features(150)
    train.loc[train.index[:75], "night_share"] += 6       # two clear behavioural groups
    model = PeerGroupModel(k_min=2, k_max=4, n_init=3, stability_seeds=2, random_state=0).fit(train)
    assert model.k_ in (2, 3, 4) and len(model.k_selection_) == 3
    test = train.copy()
    test.loc["C010", "variability_change"] = 25.0
    clusters = model.assign(train[PROFILE_FEATURES])
    scores = model.score(test, clusters)
    assert scores.idxmax() == "C010"
    assert "variability_change" in model.principal_deviations(test.loc[["C010"]], clusters).loc["C010"]
    with pytest.raises(ValueError):
        model.deviations(test, clusters.iloc[:5])
    assert set(select_k(rng.normal(size=(40, 3)), 2, 3, 2, [0, 1])["k"]) == {2, 3}


def _obs(n_customers=30, n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([[f"C{i}" for i in range(n_customers)],
                                      pd.date_range("2013-02-04", periods=n_days)], names=["customer_id", "date"])
    df = pd.DataFrame(rng.normal(0, 1, size=(len(idx), 4)), index=idx,
                      columns=["z_log_kwh", "z_night_share", "z_log_peak_ratio", "z_day_change"])
    df["is_weekend"] = (idx.get_level_values("date").dayofweek >= 5).astype(float)
    return df


def test_observation_iforest_spike_scores_high_and_threshold():
    train, test = _obs(), _obs(seed=1)
    spike = ("C3", pd.Timestamp("2013-02-10"))
    # A spike day is high relative to its own history and relative to the previous day.
    test.loc[spike, ["z_log_kwh", "z_day_change"]] = 30.0
    model = ObservationIsolationForest(n_estimators=100, contamination=0.02, random_state=0).fit(train)
    assert np.mean(model.train_scores_ > model.threshold_) == pytest.approx(0.02, abs=0.01)
    s = model.score(test)
    # Flagged and among the top 1% of days. (Isolation Forest cannot rank a value far beyond the
    # training range above the most extreme training points, so it need not be the single top day.)
    assert s.loc[spike] > model.threshold_
    assert (s > s.loc[spike]).mean() < 0.01


def test_aggregation_frequency_persistence_severity():
    idx = pd.MultiIndex.from_product([["A", "B"], pd.date_range("2013-01-01", periods=6)], names=["customer_id", "date"])
    scores = pd.Series([0.9, 0.9, 0.9, 0.1, 0.9, 0.1,   0.1, 0.1, 0.95, 0.1, 0.1, 0.1], index=idx)
    out = aggregate_to_customers(scores, threshold=0.5)
    assert out.loc["A", "observation_anomaly_count"] == 4 and out.loc["B", "observation_anomaly_count"] == 1
    assert out.loc["A", "observation_anomaly_rate"] == pytest.approx(4 / 6)
    assert out.loc["A", "obs_longest_flag_run"] == 3 and out.loc["B", "obs_longest_flag_run"] == 1
    assert out.loc["A", "obs_severity"] == pytest.approx(0.4)
    rate = customer_score_from_observations(out, "flag_rate")
    mx = customer_score_from_observations(out, "max")
    assert rate["A"] > rate["B"]                          # persistence beats a single day
    assert mx["B"] > mx["A"]                              # max-day favours the single spike
    assert customer_score_from_observations(out, "top3_mean")["A"] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        customer_score_from_observations(out, "median")


def test_thresholds_and_calibration():
    ref = np.arange(100.0)
    assert quantile_threshold(ref, 0.1) == pytest.approx(89.1)
    assert empirical_cdf([-1, 49.5, 1000], ref).tolist() == [0.0, 0.5, 1.0]
    labels = np.r_[np.zeros(90, bool), np.ones(10, bool)]
    assert f1_optimal_threshold(ref, labels) == pytest.approx(90.0)
    with pytest.raises(ValueError):
        f1_optimal_threshold(ref, np.zeros(100, bool))
    h = hybrid_score(pd.Series([1.0, 50.0]), pd.Series([99.0, 0.0]), ref, ref, 0.5)
    assert h.tolist() == pytest.approx([0.5 * 0.02 + 0.5 * 1.0, 0.5 * 0.51 + 0.5 * 0.01])


def test_fit_suite_deterministic_and_consistent_direction(small_cfg):
    data = prepare_synthetic(small_cfg, seed=0)
    w = get_windows(small_cfg)
    s1 = fit_suite(data.slot_matrix, small_cfg, data.eligible_customers, seed=5, with_ablations=True)
    s2 = fit_suite(data.slot_matrix, small_cfg, data.eligible_customers, seed=5, with_ablations=True)
    v1 = score_window(s1, data.slot_matrix, w["validation"], w["train"], s1.train_features)
    v2 = score_window(s2, data.slot_matrix, w["validation"], w["train"], s2.train_features)
    pd.testing.assert_frame_equal(v1.customer_scores, v2.customer_scores)
    assert np.isfinite(v1.customer_scores.to_numpy()).all()
    # Quadrupling one customer's validation window must raise the peer-deviation score and the
    # change-feature Isolation Forest score. (Full-feature B2 can fall when a formerly atypical
    # customer's level becomes typical - see docs/limitations.md - so it is not asserted here.)
    cid = v1.customer_scores.index[0]
    mat = data.slot_matrix.copy()
    dates = mat.index.get_level_values("date")
    sel = (mat.index.get_level_values("customer_id") == cid) & (dates >= w["validation"][0]) & (dates <= w["validation"][1])
    mat.loc[sel] = mat.loc[sel] * 4
    v3 = score_window(s1, mat, w["validation"], w["train"], s1.train_features)
    for m in (B1, "B2_change_only"):
        assert v3.customer_scores.loc[cid, m] > v1.customer_scores.loc[cid, m]
        assert v3.customer_scores[m].rank(ascending=False)[cid] <= 0.25 * len(v3.customer_scores)


def test_thresholds_use_development_labels_only(small_cfg, monkeypatch):
    """Thresholds are chosen after the validation-window injection and before any test-window injection."""
    w = get_windows(small_cfg)
    log = []
    real_inject, real_f1 = experiments.inject_anomalies, experiments.f1_optimal_threshold

    def spy_inject(matrix, customers, window, *a, **k):
        log.append(("inject", "validation" if window == w["validation"] else "test"))
        return real_inject(matrix, customers, window, *a, **k)

    def spy_f1(scores, labels):
        log.append(("threshold", None))
        return real_f1(scores, labels)

    monkeypatch.setattr(experiments, "inject_anomalies", spy_inject)
    monkeypatch.setattr(experiments, "f1_optimal_threshold", spy_f1)
    experiments.run_seed(small_cfg, "synthetic", 0, quick=True)
    first_test = next(i for i, e in enumerate(log) if e == ("inject", "test"))
    thresholds = [i for i, e in enumerate(log) if e[0] == "threshold"]
    dev = log.index(("inject", "validation"))
    assert thresholds and all(dev < i < first_test for i in thresholds)
    assert log.count(("inject", "validation")) == 1
