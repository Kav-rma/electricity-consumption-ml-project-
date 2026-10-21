"""End-to-end runs on small synthetic data, output schema, and dashboard start-up."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from electricity_anomaly.config import resolve_path
from electricity_anomaly.data_loader import DataUnavailableError
from electricity_anomaly.experiments import run_experiments
from electricity_anomaly.pipeline import run_pipeline
from electricity_anomaly.synthetic import write_synthetic_csv

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ["observation_count", "mean_daily_kwh", "behavioural_cluster", "observation_anomaly_count",
                    "observation_anomaly_rate", "customer_anomaly_score", "customer_anomaly_flag",
                    "peer_deviation_score", "principal_behavioural_deviations", "model_version", "evaluation_period"]


@pytest.fixture()
def pipeline_outputs(small_cfg):
    write_synthetic_csv(small_cfg, resolve_path(small_cfg, "synthetic_dir"), seed=1)
    meta = run_pipeline(small_cfg, "synthetic", inject_demo=True, make_figures=True)
    return small_cfg, meta


def test_synthetic_pipeline_end_to_end(pipeline_outputs):
    cfg, meta = pipeline_outputs
    out = resolve_path(cfg, "outputs_dir")
    pipe = out / "pipeline" / "synthetic"
    res = pd.read_csv(pipe / "customer_results.csv", index_col=0)
    assert res.index.name == "customer_id" and res.index.is_unique
    assert set(REQUIRED_COLUMNS) <= set(res.columns)
    assert len(res) == meta["n_scored_customers"] >= 30
    for col in ("customer_anomaly_score", "observation_level_score", "hybrid_score"):
        assert res[col].between(0, 1).all()
    assert np.isfinite(res["peer_deviation_score"]).all()
    assert (res["observation_anomaly_count"] <= res["observation_count"]).all()
    assert res["evaluation_period"].iloc[0] == "2013-04-01/2013-04-28"
    assert (res["data_source"] == "synthetic").all()
    stats = json.loads((pipe / "preprocessing_stats.json").read_text())
    assert stats["rows_malformed_timestamp"] == cfg["synthetic"]["malformed_timestamp_rows"]
    assert stats["exact_duplicate_rows_removed"] > 0 and stats["rows_negative_reading"] > 0
    truth = pd.read_csv(pipe / "demo_ground_truth_customers.csv")
    assert truth["is_anomalous"].sum() >= 1
    assert len(list((out / "figures" / "pipeline_synthetic").glob("*.png"))) >= 6
    assert not (out / "pipeline" / "real").exists()         # synthetic never written as real


def test_real_pipeline_without_data_fails_clearly(small_cfg):
    with pytest.raises(DataUnavailableError, match="Real dataset not found"):
        run_pipeline(small_cfg, "real")


def test_experiments_end_to_end_quick(small_cfg):
    out = run_experiments(small_cfg, "synthetic", seeds=[0], quick=True, make_figures=True)
    runs = pd.read_csv(out / "runs_customer.csv")
    assert {"mixed", "single_scenario"} <= set(runs["experiment"])
    ap = runs["average_precision"].dropna()
    assert len(ap) and ap.between(0, 1).all()
    hyp = pd.read_csv(out / "hypotheses.csv")
    assert hyp["hypothesis"].str.startswith(("H1", "H2", "H3")).all()
    assert (out / "summary_customer.csv").exists() and (out / "paired_comparisons.csv").exists()
    delay = pd.read_csv(out / "runs_delay.csv")
    assert delay["detection_rate_at_k"].between(0, 1).all()
    assert len(list((resolve_path(small_cfg, "outputs_dir") / "figures" / "experiments_synthetic").glob("*.png"))) >= 6


def _app_test(outputs_dir: Path, features_dir: Path):
    from streamlit.testing.v1 import AppTest

    os.environ["ELECTRICITY_OUTPUTS_DIR"] = str(outputs_dir)
    os.environ["ELECTRICITY_FEATURES_DIR"] = str(features_dir)
    try:
        return AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    finally:
        os.environ.pop("ELECTRICITY_OUTPUTS_DIR")
        os.environ.pop("ELECTRICITY_FEATURES_DIR")


def test_dashboard_starts_without_outputs(tmp_path):
    at = _app_test(tmp_path / "empty_outputs", tmp_path / "empty_features")
    assert not at.exception
    assert any("No pipeline outputs found" in i.value for i in at.info)


def test_dashboard_starts_with_outputs(pipeline_outputs):
    cfg, _ = pipeline_outputs
    at = _app_test(resolve_path(cfg, "outputs_dir"), resolve_path(cfg, "features_dir"))
    assert not at.exception
    assert any(m.label == "Eligible customers scored" for m in at.metric)
