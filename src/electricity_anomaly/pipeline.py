"""End-to-end pipeline: data preparation, model fitting, window scoring, outputs.

The same ``fit_suite`` / ``score_window`` functions are used by the pipeline
and by the experiment runner, so both approaches always see identical
customers, windows and training data.

Window roles (config ``windows``):
    fit on ``train`` (history ``train_history``)
    calibrate on clean ``validation`` (history ``train``)
    evaluate on ``test`` (history ``validation``)
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .anomaly_injection import ANOMALY_SCENARIOS, inject_anomalies
from .config import config_hash, get_windows, resolve_path
from .customer_models import CustomerIsolationForest, build_customer_iforest, top_contributions
from .data_loader import detect_columns, load_clean_readings, load_real_readings
from .evaluation import score_stability
from .feature_engineering import (CHANGE_FEATURES, LEVEL_FEATURES, PROFILE_FEATURES, TEMPORAL_FEATURES,
                                  VARIABILITY_FEATURES, customer_features, observation_features)
from .observation_models import (ObservationIsolationForest, aggregate_to_customers, build_observation_model,
                                 customer_score_from_observations)
from .peer_grouping import PeerGroupModel, build_peer_model, cluster_profiles
from .preprocessing import ProcessedData, clean_raw_chunk, preprocess, restrict_customers
from .scoring import empirical_cdf, hybrid_score, quantile_threshold
from .synthetic import generate_synthetic_readings

logger = logging.getLogger(__name__)

A_REL, A_GLOB, B1, B2, HYB = "A_obs_relative", "A_obs_global", "B1_peer_kmeans", "B2_customer_iforest", "H_hybrid"
PRIMARY_MODELS = [A_REL, A_GLOB, B1, B2, HYB]
OBS_AGG_VARIANTS = ["max", "mean", "top3_mean"]
ABLATION_SETS = {
    "B2_level_only": LEVEL_FEATURES,
    "B2_level_variability": LEVEL_FEATURES + VARIABILITY_FEATURES,
    "B2_temporal_only": TEMPORAL_FEATURES,
    "B2_profile_no_change": PROFILE_FEATURES,
    "B2_change_only": CHANGE_FEATURES,
}
MODEL_DESCRIPTIONS = {
    A_REL: "Approach A: Isolation Forest on daily observations, features relative to the customer's own history; "
           "customer score = fraction of days flagged",
    A_GLOB: "Approach A: Isolation Forest on raw daily observations (no customer context); "
            "customer score = fraction of days flagged",
    B1: "Approach B1: K-Means peer groups; score = mean of 3 largest robust z-deviations from peer group",
    B2: "Approach B2: Isolation Forest on customer behavioural feature vectors",
    HYB: "Optional extension: mean of calibrated B2 and A_obs_relative percentiles",
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_synthetic(cfg: dict[str, Any], seed: int, n_customers: int | None = None,
                      csv_path: Path | None = None) -> ProcessedData:
    """Synthetic data, either read from the generated CSV or generated in memory."""
    windows = get_windows(cfg)
    if csv_path is not None:
        clean, stats = load_clean_readings(csv_path, cfg, sample_fraction=1.0, chunksize=1_000_000)
    else:
        raw, _, _ = generate_synthetic_readings(cfg, seed, n_customers=n_customers)
        colmap = detect_columns(list(raw.columns), cfg["schema"])
        clean, stats = clean_raw_chunk(raw, colmap, cfg)
        stats["raw_rows_scanned"] = len(raw)
    stats["data_source"] = "synthetic"
    return preprocess(clean, cfg, windows, stats)


def prepare_real(cfg: dict[str, Any], use_cache: bool = True) -> ProcessedData:
    """Real London data: stream, sample households, preprocess, cap the customer count."""
    windows = get_windows(cfg)
    clean, stats = load_real_readings(cfg, use_cache=use_cache)
    data = preprocess(clean, cfg, windows, stats)
    eligible = data.eligible_customers
    cap = int(cfg["real"]["max_households"])
    if len(eligible) > cap:
        rng = np.random.default_rng(cfg["random_seed"])
        eligible = sorted(rng.choice(eligible, size=cap, replace=False).tolist())
        logger.info("Randomly retained %d of %d eligible households (seed %d)", cap,
                    len(data.eligible_customers), cfg["random_seed"])
    data = restrict_customers(data, eligible)
    data.stats["data_source"] = "real"
    return data


# ---------------------------------------------------------------------------
# Fitting and scoring
# ---------------------------------------------------------------------------
@dataclass
class ModelSuite:
    """All detectors fitted on the same training window and customers."""

    cfg: dict[str, Any]
    seed: int
    customers: list[str]
    obs_models: dict[str, ObservationIsolationForest]
    peer: PeerGroupModel
    iforest: CustomerIsolationForest
    ablations: dict[str, CustomerIsolationForest]
    train_features: pd.DataFrame
    train_profiles: pd.DataFrame
    resources: list[dict[str, Any]] = field(default_factory=list)


def _measure(fn, *args, track_memory: bool = False, **kwargs):
    """Run ``fn`` and return (result, seconds, peak traced memory in MB or NaN).

    ``tracemalloc`` slows allocation-heavy code considerably, so peak memory is
    only traced when ``track_memory`` is set, and timings from such calls
    include that overhead.
    """
    if track_memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        peak = tracemalloc.get_traced_memory()[1] / 2**20 if track_memory else float("nan")
        if track_memory:
            tracemalloc.stop()
    return result, time.perf_counter() - t0, peak


def fit_suite(matrix: pd.DataFrame, cfg: dict[str, Any], customers: list[str], seed: int,
              with_ablations: bool = True, track_memory: bool = False) -> ModelSuite:
    """Fit Approach A (two representations), B1, B2 and B2 ablations on the training window.

    With ``track_memory`` the peak Python-traced memory of each step is
    recorded (slower); otherwise only wall-clock time is recorded.
    """
    w = get_windows(cfg)
    tm = track_memory
    (F_train, prof_train), t_cf, m_cf = _measure(customer_features, matrix, w["train"], w["train_history"], cfg, customers, track_memory=tm)
    obs_train, t_of, m_of = _measure(observation_features, matrix, w["train"], w["train_history"], cfg, customers, track_memory=tm)
    resources = [dict(stage="features", model="customer_features", seconds=t_cf, peak_mb=m_cf, rows=len(F_train)),
                 dict(stage="features", model="observation_features", seconds=t_of, peak_mb=m_of, rows=len(obs_train))]
    obs_models = {}
    for rep in ("relative", "global"):
        model, t, mem = _measure(build_observation_model(cfg, rep, seed).fit, obs_train, track_memory=tm)
        obs_models[rep] = model
        resources.append(dict(stage="fit", model=A_REL if rep == "relative" else A_GLOB, seconds=t, peak_mb=mem, rows=len(obs_train)))
    peer, t, mem = _measure(build_peer_model(cfg, seed).fit, F_train, track_memory=tm)
    resources.append(dict(stage="fit", model=B1, seconds=t, peak_mb=mem, rows=len(F_train)))
    iforest, t, mem = _measure(build_customer_iforest(cfg, seed).fit, F_train, track_memory=tm)
    resources.append(dict(stage="fit", model=B2, seconds=t, peak_mb=mem, rows=len(F_train)))
    ablations = {}
    if with_ablations:
        for name, feats in ABLATION_SETS.items():
            ablations[name] = build_customer_iforest(cfg, seed, features=feats).fit(F_train)
    return ModelSuite(cfg=cfg, seed=seed, customers=list(F_train.index), obs_models=obs_models, peer=peer,
                      iforest=iforest, ablations=ablations, train_features=F_train, train_profiles=prof_train,
                      resources=resources)


@dataclass
class WindowScores:
    """Scores of every detector for one evaluation window."""

    customer_scores: pd.DataFrame            # index customer_id, one column per model
    obs_day_scores: dict[str, pd.Series]     # representation -> day scores
    obs_summary: dict[str, pd.DataFrame]     # representation -> customer aggregation
    features: pd.DataFrame
    profiles: pd.DataFrame
    clusters: pd.Series
    window: tuple[pd.Timestamp, pd.Timestamp]


def score_window(suite: ModelSuite, matrix: pd.DataFrame, window: tuple[pd.Timestamp, pd.Timestamp],
                 history: tuple[pd.Timestamp, pd.Timestamp], peer_reference_features: pd.DataFrame,
                 calibration: pd.DataFrame | None = None, timings: list | None = None) -> WindowScores:
    """Score all detectors on one window.

    Args:
        peer_reference_features: Customer features from the window *before*
            ``window`` (clean); their profile part assigns peer groups.
        calibration: Clean validation-window scores used to calibrate the
            hybrid; when ``None`` the hybrid is omitted.
        timings: Optional list receiving per-model scoring times.
    """
    cfg = suite.cfg
    agg = cfg["observation_model"]["aggregation"]
    t0 = time.perf_counter()
    F, prof = customer_features(matrix, window, history, cfg, suite.customers)
    t_feat = time.perf_counter() - t0
    t0 = time.perf_counter()
    obs = observation_features(matrix, window, history, cfg, suite.customers)
    t_obs_feat = time.perf_counter() - t0
    common = F.index.intersection(obs.index.get_level_values("customer_id").unique()).intersection(peer_reference_features.index)
    F, prof = F.loc[common], prof.loc[common]
    obs = obs.loc[obs.index.get_level_values("customer_id").isin(common)]

    scores: dict[str, pd.Series] = {}
    day_scores, summaries = {}, {}
    timing = {}
    for rep, model in suite.obs_models.items():
        t0 = time.perf_counter()
        s = model.score(obs)
        summ = aggregate_to_customers(s, model.threshold_)
        name = A_REL if rep == "relative" else A_GLOB
        scores[name] = customer_score_from_observations(summ, agg)
        timing[name] = time.perf_counter() - t0 + t_obs_feat
        day_scores[rep], summaries[rep] = s, summ
        if rep == "relative":
            for variant in OBS_AGG_VARIANTS:
                scores[f"{A_REL}_{variant}"] = customer_score_from_observations(summ, variant)
    t0 = time.perf_counter()
    clusters = suite.peer.assign(peer_reference_features.loc[common])
    scores[B1] = suite.peer.score(F, clusters)
    timing[B1] = time.perf_counter() - t0 + t_feat
    t0 = time.perf_counter()
    scores[B2] = suite.iforest.score(F)
    timing[B2] = time.perf_counter() - t0 + t_feat
    for name, model in suite.ablations.items():
        scores[name] = model.score(F)
    table = pd.DataFrame(scores).loc[common]
    if calibration is not None:
        table[HYB] = hybrid_score(table[B2], table[A_REL], calibration[B2], calibration[A_REL],
                                  cfg["customer_models"]["hybrid_weight"])
    if timings is not None:
        n_days = len(obs)
        timings.extend(dict(stage="score", model=m, seconds=t, rows=len(common), observation_rows=n_days)
                       for m, t in timing.items())
    return WindowScores(customer_scores=table, obs_day_scores=day_scores, obs_summary=summaries,
                        features=F, profiles=prof, clusters=clusters, window=window)


def calibrate_on_validation(suite: ModelSuite, matrix: pd.DataFrame) -> WindowScores:
    """Score the clean validation window; its scores calibrate thresholds and the hybrid."""
    w = get_windows(suite.cfg)
    val = score_window(suite, matrix, w["validation"], w["train"], suite.train_features)
    # The hybrid of the calibration window is calibrated against itself.
    val.customer_scores[HYB] = hybrid_score(val.customer_scores[B2], val.customer_scores[A_REL],
                                            val.customer_scores[B2], val.customer_scores[A_REL],
                                            suite.cfg["customer_models"]["hybrid_weight"])
    return val


# ---------------------------------------------------------------------------
# Pipeline run (one dataset, one configuration)
# ---------------------------------------------------------------------------
def _k_sensitivity(suite: ModelSuite, val: WindowScores, cfg: dict[str, Any]) -> pd.DataFrame:
    """Rank agreement of B1 scores across cluster counts and random initialisations."""
    from scipy.stats import spearmanr

    km = cfg["customer_models"]["kmeans"]
    base = val.customer_scores[B1]
    rows = []
    for k in range(km["k_min"], km["k_max"] + 1):
        for s in range(3):
            model = build_peer_model(cfg, suite.seed + 100 + s, k=k)
            model.stability_seeds = 1
            model.k_min = model.k_max = k
            model.fit(suite.train_features)
            clusters = model.assign(suite.train_features.loc[val.features.index])
            sc = model.score(val.features, clusters)
            rows.append(dict(k=k, init_seed=suite.seed + 100 + s,
                             spearman_vs_selected=float(spearmanr(sc, base.loc[sc.index]).statistic),
                             top10_overlap=len(set(sc.nlargest(max(1, len(sc) // 10)).index)
                                               & set(base.nlargest(max(1, len(base) // 10)).index)) / max(1, len(sc) // 10)))
    return pd.DataFrame(rows)


def build_results_table(suite: ModelSuite, test: WindowScores, val: WindowScores, thresholds: dict[str, float],
                        cfg: dict[str, Any], data_kind: str) -> pd.DataFrame:
    """Customer-level output table in the schema required by the project brief."""
    w = test.window
    cs = test.customer_scores
    summ = test.obs_summary["relative"]
    contrib = suite.iforest.feature_contributions(test.features)
    out = pd.DataFrame(index=cs.index)
    out.index.name = "customer_id"
    out["observation_count"] = summ["observation_count"]
    out["mean_daily_kwh"] = test.features["mean_daily_kwh"].round(3)
    out["behavioural_cluster"] = test.clusters
    out["observation_anomaly_count"] = summ["observation_anomaly_count"]
    out["observation_anomaly_rate"] = summ["observation_anomaly_rate"].round(4)
    out["observation_longest_flag_run_days"] = summ["obs_longest_flag_run"]
    out["observation_severity"] = summ["obs_severity"].round(4)
    out["observation_level_score"] = empirical_cdf(cs[A_REL], val.customer_scores[A_REL]).round(4)
    out["observation_level_flag"] = cs[A_REL] > thresholds[A_REL]
    out["customer_anomaly_score"] = empirical_cdf(cs[B2], val.customer_scores[B2]).round(4)
    out["customer_anomaly_raw_score"] = cs[B2].round(5)
    out["customer_anomaly_flag"] = cs[B2] > thresholds[B2]
    out["peer_deviation_score"] = cs[B1].round(3)
    out["peer_deviation_flag"] = cs[B1] > thresholds[B1]
    out["hybrid_score"] = cs[HYB].round(4)
    out["hybrid_flag"] = cs[HYB] > thresholds[HYB]
    out["principal_behavioural_deviations"] = suite.peer.principal_deviations(test.features, test.clusters)
    out["iforest_top_features"] = top_contributions(contrib, test.features)
    out["model_version"] = f"{cfg['project']['model_version']}+{config_hash(cfg)}"
    out["evaluation_period"] = f"{w[0].date()}/{w[1].date()}"
    out["data_source"] = data_kind
    return out.sort_values("customer_anomaly_score", ascending=False)


def output_dirs(cfg: dict[str, Any], data_kind: str) -> dict[str, Path]:
    """Output folders for a dataset kind; synthetic and real never share a folder."""
    root = resolve_path(cfg, "outputs_dir")
    dirs = {"pipeline": root / "pipeline" / data_kind, "figures": root / "figures" / f"pipeline_{data_kind}",
            "features": resolve_path(cfg, "features_dir") / data_kind}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def run_pipeline(cfg: dict[str, Any], data_kind: str, inject_demo: bool = True, use_cache: bool = True,
                 make_figures: bool = True) -> dict[str, Any]:
    """Run preprocessing, fitting, calibration and test-window scoring; save outputs.

    For synthetic data a documented mix of anomalies is injected into the test
    window (``inject_demo``) so the dashboard has known cases to show; the
    ground truth is saved separately and never used by the models. Real data
    is never injected here.
    """
    if data_kind not in ("synthetic", "real"):
        raise ValueError("data_kind must be 'synthetic' or 'real'")
    seed = int(cfg["random_seed"])
    dirs = output_dirs(cfg, data_kind)
    w = get_windows(cfg)
    t_start = time.time()

    if data_kind == "synthetic":
        csv = resolve_path(cfg, "synthetic_dir") / cfg["synthetic"]["csv_name"]
        if not csv.exists():
            raise FileNotFoundError(f"{csv} not found. Run: python scripts/generate_synthetic_data.py")
        data = prepare_synthetic(cfg, seed, csv_path=csv)
    else:
        data = prepare_real(cfg, use_cache=use_cache)
    customers = data.eligible_customers
    if len(customers) < 30:
        raise RuntimeError(f"Only {len(customers)} eligible customers; at least 30 are needed. "
                           "Increase the sample or relax preprocessing.min_window_day_coverage.")
    logger.info("Fitting models on %d eligible customers", len(customers))
    suite = fit_suite(data.slot_matrix, cfg, customers, seed, track_memory=True)
    val = calibrate_on_validation(suite, data.slot_matrix)
    alert_rate = cfg["customer_models"]["alert_rate"]
    thresholds = {m: quantile_threshold(val.customer_scores[m], alert_rate) for m in PRIMARY_MODELS}

    test_matrix = data.slot_matrix
    truth = None
    if data_kind == "synthetic" and inject_demo:
        n_controls = max(2, int(round(0.02 * len(customers))))
        inj = inject_anomalies(data.slot_matrix, customers, w["test"], w["validation"], ANOMALY_SCENARIOS,
                               cfg["experiments"]["scenario_prevalence"], "medium", cfg, seed, n_controls=n_controls)
        test_matrix = inj.matrix
        truth = inj
        inj.customer_labels.to_csv(dirs["pipeline"] / "demo_ground_truth_customers.csv", index=False)
        inj.day_labels.to_csv(dirs["pipeline"] / "demo_ground_truth_days.csv", index=False)
    timings: list = []
    test = score_window(suite, test_matrix, w["test"], w["validation"], val.features,
                        calibration=val.customer_scores, timings=timings)
    clean_test = test if truth is None else score_window(suite, data.slot_matrix, w["test"], w["validation"],
                                                         val.features, calibration=val.customer_scores)

    results = build_results_table(suite, test, val, thresholds, cfg, data_kind)
    if truth is not None:
        lab = truth.customer_labels.set_index("customer_id")
        results["injected_scenario"] = lab["scenario"].reindex(results.index).fillna("none")
    results.to_csv(dirs["pipeline"] / "customer_results.csv")

    rel_model = suite.obs_models["relative"]
    day = test.obs_day_scores["relative"].to_frame("observation_score")
    day["observation_flag"] = day["observation_score"] > rel_model.threshold_
    from .preprocessing import daily_from_matrix
    test_daily = daily_from_matrix(test_matrix, cfg["preprocessing"]["min_day_coverage"])
    test_daily = test_daily.loc[test_daily.index.get_level_values("customer_id").isin(results.index), ["daily_kwh", "coverage", "valid"]]
    test_daily.join(day).reset_index().to_parquet(dirs["pipeline"] / "daily_with_scores.parquet", index=False)
    test.obs_day_scores["global"].to_frame("observation_score_global").reset_index().to_parquet(
        dirs["pipeline"] / "observation_scores_global.parquet", index=False)

    suite.train_features.to_parquet(dirs["features"] / "customer_features_train.parquet")
    val.features.to_parquet(dirs["features"] / "customer_features_validation.parquet")
    test.features.to_parquet(dirs["features"] / "customer_features_test.parquet")
    profiles = pd.concat({"train": suite.train_profiles, "validation": val.profiles, "test": test.profiles},
                         names=["window"])
    profiles.columns = [str(c) for c in profiles.columns]
    profiles.to_parquet(dirs["pipeline"] / "load_profiles.parquet")
    summary, cluster_prof = cluster_profiles(suite.train_features, suite.peer.train_clusters_, suite.train_profiles)
    summary.to_csv(dirs["pipeline"] / "cluster_summary.csv")
    cluster_prof.columns = [str(c) for c in cluster_prof.columns]
    cluster_prof.to_csv(dirs["pipeline"] / "cluster_load_profiles.csv")
    suite.peer.k_selection_.to_csv(dirs["pipeline"] / "k_selection.csv", index=False)
    suite.peer.train_clusters_.to_csv(dirs["pipeline"] / "train_clusters.csv")
    _k_sensitivity(suite, val, cfg).to_csv(dirs["pipeline"] / "k_sensitivity.csv", index=False)
    test.customer_scores.to_csv(dirs["pipeline"] / "all_model_scores_test.csv")
    val.customer_scores.to_csv(dirs["pipeline"] / "all_model_scores_validation.csv")
    data.coverage.to_csv(dirs["pipeline"] / "coverage.csv")

    stability = pd.DataFrame([{**score_stability(val.customer_scores[m], clean_test.customer_scores[m], alert_rate), "model": m}
                              for m in PRIMARY_MODELS])
    stability.to_csv(dirs["pipeline"] / "score_stability_validation_vs_test.csv", index=False)
    agreement = _model_agreement(test.customer_scores, alert_rate)
    agreement.to_csv(dirs["pipeline"] / "model_agreement.csv", index=False)
    pd.DataFrame(suite.resources + timings).to_csv(dirs["pipeline"] / "resources.csv", index=False)

    with open(dirs["pipeline"] / "preprocessing_stats.json", "w") as fh:
        json.dump(data.stats, fh, indent=2, default=str)
    meta = {
        "data_source": data_kind,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "model_version": cfg["project"]["model_version"],
        "config_hash": config_hash(cfg),
        "random_seed": seed,
        "windows": {k: [str(a.date()), str(b.date())] for k, (a, b) in w.items()},
        "n_eligible_customers": len(customers),
        "n_scored_customers": int(len(results)),
        "selected_k": suite.peer.k_,
        "thresholds_from_clean_validation": thresholds,
        "alert_rate": alert_rate,
        "observation_day_threshold": rel_model.threshold_,
        "demo_injection": bool(truth is not None),
        "model_descriptions": MODEL_DESCRIPTIONS,
        "n_flagged": {m: int((test.customer_scores[m] > thresholds[m]).sum()) for m in PRIMARY_MODELS},
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open(dirs["pipeline"] / "run_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    if make_figures:
        from .visualization import pipeline_figures
        pipeline_figures(dirs["pipeline"], dirs["figures"], data_kind)
    logger.info("Pipeline (%s) finished in %.0fs; outputs in %s", data_kind, time.time() - t_start, dirs["pipeline"])
    return meta


def _model_agreement(scores: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Pairwise Spearman correlation and top-K overlap between detectors."""
    rows = []
    for i, a in enumerate(PRIMARY_MODELS):
        for b in PRIMARY_MODELS[i + 1:]:
            s = score_stability(scores[a], scores[b], budget)
            rows.append(dict(model_a=a, model_b=b, spearman=s["spearman"], top_jaccard=s["top_jaccard"]))
    return pd.DataFrame(rows)
