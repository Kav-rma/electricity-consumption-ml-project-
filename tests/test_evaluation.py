"""Metric calculations, statistics, and controlled anomaly injection labels."""

import numpy as np
import pandas as pd
import pytest

from electricity_anomaly.anomaly_injection import (ALL_SCENARIOS, CONTROL, PERSISTENT, TRANSIENT,
                                                   inject_anomalies)
from electricity_anomaly.evaluation import (binary_metrics, holm_adjust, mean_ci, paired_comparison,
                                            precision_recall_at_k, ranking_metrics, score_stability)

from conftest import make_matrix

HIST = (pd.Timestamp("2013-01-07"), pd.Timestamp("2013-02-03"))
WIN = (pd.Timestamp("2013-02-04"), pd.Timestamp("2013-03-03"))


# --------------------------------------------------------------------------- metrics
def test_binary_metrics_by_hand():
    m = binary_metrics([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert binary_metrics([0, 0], [0, 0])["f1"] == 0.0     # undefined -> 0, no crash


def test_precision_recall_at_k_and_ranking_metrics():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    perfect = np.arange(10, 0, -1).astype(float)
    assert precision_recall_at_k(y, perfect, 2) == (1.0, 1.0)
    assert precision_recall_at_k(y, perfect, 4) == (0.5, 1.0)
    r = ranking_metrics(y, perfect, [0.1, 0.2])
    assert r["average_precision"] == 1.0 and r["roc_auc"] == 1.0 and r["r_precision"] == 1.0
    assert r["precision_at_10pct"] == 1.0 and r["recall_at_10pct"] == 0.5 and r["recall_at_20pct"] == 1.0
    worst = ranking_metrics(y, -perfect, [0.1])
    assert worst["roc_auc"] == 0.0 and worst["precision_at_10pct"] == 0.0
    assert np.isnan(ranking_metrics(np.zeros(5, bool), np.arange(5.0), [0.2])["average_precision"])


def test_accuracy_would_mislead():
    """At 10% prevalence flagging nobody has 90% accuracy but zero recall - why accuracy is not used."""
    y = np.r_[np.ones(10, bool), np.zeros(90, bool)]
    m = binary_metrics(y, np.zeros(100, bool))
    assert m["recall"] == 0 and (100 - m["fn"]) / 100 == 0.9


def test_mean_ci_and_paired_comparison():
    s = mean_ci([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["ci_low"] < 2.0 < s["ci_high"] and s["n_runs"] == 3
    same = paired_comparison([0.5] * 6, [0.5] * 6)
    assert same["p_value"] == 1.0 and same["ties"] == 6
    better = paired_comparison(np.linspace(0.6, 0.9, 10), np.linspace(0.1, 0.4, 10), n_boot=200)
    assert better["mean_diff"] == pytest.approx(0.5) and better["wins_a"] == 10 and better["p_value"] < 0.01
    assert better["ci_low"] == pytest.approx(0.5) and better["ci_high"] == pytest.approx(0.5)


def test_holm_adjust():
    adj = holm_adjust([0.01, 0.04, 0.03, np.nan])
    assert adj[0] == pytest.approx(0.03) and adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[3])


def test_score_stability():
    a = pd.Series(np.arange(20.0), index=[f"c{i}" for i in range(20)])
    assert score_stability(a, a, 0.1) == {"spearman": 1.0, "top_jaccard": 1.0, "n_customers": 20}
    assert score_stability(a, -a, 0.1)["top_jaccard"] == 0.0


# --------------------------------------------------------------------------- injection
def _matrix(n=20):
    rng = np.random.default_rng(0)
    return make_matrix({f"C{i:02d}": rng.uniform(5, 15, 56) for i in range(n)})


def test_injection_copies_and_labels(small_cfg):
    mat = _matrix()
    original = mat.copy()
    inj = inject_anomalies(mat, mat.index.unique("customer_id").tolist(), WIN, HIST, ["persistent_increase"],
                           0.2, "medium", small_cfg, seed=1, onset="start")
    pd.testing.assert_frame_equal(mat, original)           # reference data untouched
    lab = inj.customer_labels
    assert len(lab) == 4 and lab["is_anomalous"].all() and (lab["magnitude"] == 0.5).all()
    assert set(lab.columns) >= {"customer_id", "scenario", "magnitude", "onset", "end", "n_affected_days"}
    cid = lab["customer_id"].iloc[0]
    ratio = (inj.matrix.loc[cid] / original.loc[cid]).to_numpy()
    dates = original.loc[cid].index
    assert np.allclose(ratio[dates >= WIN[0]], 1.5) and np.allclose(ratio[dates < WIN[0]], 1.0)
    assert len(inj.day_labels) == 4 * 28
    untouched = [c for c in original.index.unique("customer_id") if c not in set(lab["customer_id"])]
    pd.testing.assert_frame_equal(inj.matrix.loc[untouched], original.loc[untouched])
    truth = inj.customer_truth(original.index.unique("customer_id"))
    assert truth.sum() == 4


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_every_scenario_changes_data_and_records_truth(small_cfg, scenario):
    mat = _matrix()
    inj = inject_anomalies(mat, mat.index.unique("customer_id").tolist(), WIN, HIST, [scenario], 0.1, "high",
                           small_cfg, seed=3)
    lab = inj.customer_labels.iloc[0]
    assert lab["scenario"] == scenario
    cid = lab["customer_id"]
    assert not np.allclose(inj.matrix.loc[cid].to_numpy(), mat.loc[cid].to_numpy())
    days = inj.day_labels.loc[inj.day_labels["customer_id"] == cid]
    if scenario in TRANSIENT:
        assert lab["kind"] == "transient" and 0 < len(days) < 28
    elif scenario in PERSISTENT:
        assert lab["kind"] == "persistent" and len(days) >= 28 * 2 // 3
    else:
        assert scenario == CONTROL and not lab["is_anomalous"] and lab["is_control"] and len(days) == 0
        assert pd.Timestamp(lab["onset"]) == HIST[0]       # control is high in history too


def test_injection_seed_reproducible_and_controls(small_cfg):
    mat = _matrix(40)
    cust = mat.index.unique("customer_id").tolist()
    a = inject_anomalies(mat, cust, WIN, HIST, PERSISTENT + TRANSIENT, 0.2, "low", small_cfg, seed=7, n_controls=2)
    b = inject_anomalies(mat, cust, WIN, HIST, PERSISTENT + TRANSIENT, 0.2, "low", small_cfg, seed=7, n_controls=2)
    pd.testing.assert_frame_equal(a.customer_labels, b.customer_labels)
    assert a.customer_labels["is_anomalous"].sum() == 8 and a.customer_labels["is_control"].sum() == 2
    assert a.customer_labels["customer_id"].is_unique
