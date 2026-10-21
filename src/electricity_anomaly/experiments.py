"""Controlled experiments (Experiment 1 and 2) and their statistical summary.

For each seed:

1. Background data: a fresh synthetic population (synthetic) or the fixed
   real sample (real background, i.e. semi-synthetic).
2. Fit every detector on the clean training window.
3. Score the clean validation window -> calibration distribution.
4. Development set: mixed anomalies injected into the *validation* window
   (independent random draw). F1-optimal thresholds are chosen here.
5. Test sets: anomalies injected into the *test* window with new random
   draws: mixed scenarios at each prevalence x magnitude, and each scenario
   alone at each magnitude. Ranking metrics plus metrics at the development
   thresholds are recorded. Final test labels never influence any choice.
6. Detection delay: persistent anomalies starting on the first test day,
   scored on growing partial windows.
7. Clean-data score stability between validation and test windows, and
   computational cost.

All per-run rows are saved; summaries (mean, SD, 95% CI across seeds) and
paired Wilcoxon tests across seeds are computed from them.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .anomaly_injection import ALL_SCENARIOS, ANOMALY_SCENARIOS, CONTROL, PERSISTENT, TRANSIENT, inject_anomalies
from .config import config_hash, get_windows, resolve_path
from .evaluation import binary_metrics, holm_adjust, paired_comparison, ranking_metrics, score_stability, summarise_runs
from .pipeline import (A_GLOB, A_REL, ABLATION_SETS, B1, B2, HYB, OBS_AGG_VARIANTS, PRIMARY_MODELS,
                       calibrate_on_validation, fit_suite, prepare_real, prepare_synthetic, score_window)
from .preprocessing import ProcessedData
from .scoring import f1_optimal_threshold

logger = logging.getLogger(__name__)

ALL_MODELS = PRIMARY_MODELS + [f"{A_REL}_{v}" for v in OBS_AGG_VARIANTS] + list(ABLATION_SETS)
CUSTOMER_METRICS = ["average_precision", "roc_auc", "r_precision", "precision_at_5pct", "recall_at_5pct",
                    "precision_at_10pct", "recall_at_10pct", "precision", "recall", "f1"]


def _evaluate_customers(scores: pd.DataFrame, labels: pd.DataFrame, thresholds: dict[str, float],
                        budgets: list[float], models: list[str]) -> list[dict[str, Any]]:
    """Customer-level metrics for every model on one injected test set.

    Benign controls count as negatives; their false-alarm rate is reported
    separately in ``control_flag_rate`` / ``control_in_top10pct``.
    """
    positives = set(labels.loc[labels["is_anomalous"], "customer_id"])
    controls = set(labels.loc[labels["is_control"] & ~labels["is_anomalous"], "customer_id"])
    y = scores.index.isin(list(positives))
    is_ctrl = scores.index.isin(list(controls))
    rows = []
    for m in models:
        s = scores[m].to_numpy()
        row: dict[str, Any] = {"model": m}
        if y.any():
            row.update(ranking_metrics(y, s, budgets))
            row.update(binary_metrics(y, s > thresholds[m]))
        if is_ctrl.any():
            k10 = max(1, int(round(0.10 * len(s))))
            top = np.zeros(len(s), dtype=bool)
            top[np.argsort(-s, kind="mergesort")[:k10]] = True
            row["control_flag_rate"] = float((s[is_ctrl] > thresholds[m]).mean())
            row["control_in_top10pct"] = float(top[is_ctrl].mean())
            normal = ~y & ~is_ctrl
            row["normal_flag_rate"] = float((s[normal] > thresholds[m]).mean())
            from sklearn.metrics import roc_auc_score
            row["control_vs_normal_auc"] = float(roc_auc_score(is_ctrl[is_ctrl | normal], s[is_ctrl | normal]))
        rows.append(row)
    return rows


def _evaluate_days(ws, day_labels: pd.DataFrame, suite) -> list[dict[str, Any]]:
    """Observation-level (day) metrics for Approach A on transient scenarios."""
    from sklearn.metrics import average_precision_score

    rows = []
    affected = pd.MultiIndex.from_frame(day_labels[["customer_id", "date"]]) if len(day_labels) else pd.MultiIndex.from_tuples([], names=["customer_id", "date"])
    for rep, name in (("relative", A_REL), ("global", A_GLOB)):
        s = ws.obs_day_scores[rep]
        y = s.index.isin(affected)
        if not y.any():
            continue
        flags = s.to_numpy() > suite.obs_models[rep].threshold_
        rows.append({"model": name, "day_average_precision": float(average_precision_score(y, s.to_numpy())),
                     **{f"day_{k}": v for k, v in binary_metrics(y, flags).items()},
                     "n_days": int(len(y)), "n_anomalous_days": int(y.sum())})
    return rows


def run_seed(cfg: dict[str, Any], data_kind: str, seed: int, base: ProcessedData | None = None,
             quick: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Run every experiment for one seed and return result rows by table name."""
    exp = cfg["experiments"]
    w = get_windows(cfg)
    budgets = exp["alert_budgets"]
    t_seed = time.time()
    if data_kind == "synthetic":
        data = prepare_synthetic(cfg, seed, n_customers=exp["synthetic_customers"])
    else:
        if base is None:
            raise ValueError("Real-background experiments need the prepared real data")
        data = base
    matrix = data.slot_matrix
    customers = data.eligible_customers
    suite = fit_suite(matrix, cfg, customers, seed, track_memory=True)
    customers = suite.customers
    n = len(customers)
    n_controls = max(2, int(round(0.02 * n)))
    out: dict[str, list] = {k: [] for k in ("customer", "day", "delay", "stability", "resources", "thresholds", "k_selection")}

    val = calibrate_on_validation(suite, matrix)
    calib = val.customer_scores
    clean_test = score_window(suite, matrix, w["test"], w["validation"], val.features, calibration=calib)
    for m in ALL_MODELS:
        out["stability"].append({"seed": seed, "model": m,
                                 **score_stability(calib[m], clean_test.customer_scores[m], 0.10)})
    ks = suite.peer.k_selection_.copy()
    ks["seed"] = seed
    ks["selected"] = ks["k"] == suite.peer.k_
    out["k_selection"].extend(ks.to_dict("records"))

    # Development set (validation window) -> thresholds.
    dev = inject_anomalies(matrix, customers, w["validation"], w["train"], ANOMALY_SCENARIOS,
                           exp["scenario_prevalence"], exp["dev_magnitude"], cfg, seed=seed * 1000 + 1,
                           n_controls=n_controls)
    dev_scores = score_window(suite, dev.matrix, w["validation"], w["train"], suite.train_features, calibration=calib)
    y_dev = dev.customer_truth(dev_scores.customer_scores.index)
    thresholds = {m: f1_optimal_threshold(dev_scores.customer_scores[m], y_dev) for m in ALL_MODELS}
    out["thresholds"].extend({"seed": seed, "model": m, "threshold": t} for m, t in thresholds.items())

    def record(experiment: str, scenario: str, magnitude: str, prevalence: float, inj, run_seed: int) -> None:
        timings: list = []
        ws = score_window(suite, inj.matrix, w["test"], w["validation"], val.features, calibration=calib, timings=timings)
        meta = dict(seed=seed, experiment=experiment, scenario=scenario, magnitude=magnitude, prevalence=prevalence,
                    n_customers=len(ws.customer_scores))
        for row in _evaluate_customers(ws.customer_scores, inj.customer_labels, thresholds, budgets, ALL_MODELS):
            out["customer"].append({**meta, **row})
        if scenario in TRANSIENT or experiment == "mixed":
            for row in _evaluate_days(ws, inj.day_labels.loc[inj.day_labels["scenario"].isin(TRANSIENT)]
                                      if len(inj.day_labels) else inj.day_labels, suite):
                out["day"].append({**meta, **row})
        if experiment == "mixed" and prevalence == exp["scenario_prevalence"] and magnitude == exp["dev_magnitude"]:
            out["resources"].extend({**t, "seed": seed} for t in timings)

    magnitudes = list(exp["magnitudes"])
    if quick:
        magnitudes = [exp["dev_magnitude"]]
    run_id = 0
    for prevalence in exp["mixed_prevalences"]:
        for mag in magnitudes:
            run_id += 1
            inj = inject_anomalies(matrix, customers, w["test"], w["validation"], ANOMALY_SCENARIOS, prevalence, mag,
                                   cfg, seed=seed * 1000 + 100 + run_id, n_controls=n_controls)
            record("mixed", "mixed", mag, prevalence, inj, run_id)
    for scenario in ALL_SCENARIOS:
        for mag in magnitudes:
            run_id += 1
            inj = inject_anomalies(matrix, customers, w["test"], w["validation"], [scenario],
                                   exp["scenario_prevalence"] if scenario != CONTROL else 0.0, mag, cfg,
                                   seed=seed * 1000 + 100 + run_id,
                                   n_controls=int(round(exp["scenario_prevalence"] * n)) if scenario == CONTROL else 0)
            record("single_scenario", scenario, mag, exp["scenario_prevalence"], inj, run_id)

    # Detection delay for persistent scenarios (onset on the first test day).
    checkpoints = exp["delay_checkpoints_days"] if not quick else exp["delay_checkpoints_days"][::2]
    k = max(1, int(round(exp["scenario_prevalence"] * n)))
    for scenario in PERSISTENT:
        run_id += 1
        inj = inject_anomalies(matrix, customers, w["test"], w["validation"], [scenario], exp["scenario_prevalence"],
                               exp["dev_magnitude"], cfg, seed=seed * 1000 + 100 + run_id, onset="start")
        positives = set(inj.customer_labels.loc[inj.customer_labels["is_anomalous"], "customer_id"])
        for days in checkpoints:
            end = min(w["test"][0] + pd.Timedelta(days=days - 1), w["test"][1])
            ws = score_window(suite, inj.matrix, (w["test"][0], end), w["validation"], val.features, calibration=calib)
            for m in PRIMARY_MODELS:
                s = ws.customer_scores[m]
                top = set(s.nlargest(k).index)
                detected = len(top & positives) / max(1, len(positives))
                out["delay"].append(dict(seed=seed, scenario=scenario, days_since_onset=days, model=m,
                                         detection_rate_at_k=detected, k=k))
    out["resources"].extend({**r, "seed": seed} for r in suite.resources)
    logger.info("Seed %d (%s) done in %.0fs", seed, data_kind, time.time() - t_seed)
    return out


# ---------------------------------------------------------------------------
# Summaries and hypothesis tests
# ---------------------------------------------------------------------------
def summarise(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Aggregate per-seed rows into means/CIs and paired comparisons."""
    exp = cfg["experiments"]
    cust = tables["customer"]
    out: dict[str, pd.DataFrame] = {}
    metric_cols = [c for c in CUSTOMER_METRICS + ["control_flag_rate", "control_in_top10pct", "control_vs_normal_auc",
                                                  "normal_flag_rate"] if c in cust]
    out["summary_customer"] = summarise_runs(cust, ["experiment", "scenario", "magnitude", "prevalence", "model"], metric_cols)
    if len(tables["day"]):
        day_cols = [c for c in tables["day"].columns if c.startswith("day_") and c not in ("day_tp", "day_fp", "day_fn")]
        out["summary_day"] = summarise_runs(tables["day"], ["experiment", "scenario", "magnitude", "model"], day_cols)
    out["summary_delay"] = summarise_runs(tables["delay"], ["scenario", "days_since_onset", "model"], ["detection_rate_at_k"])
    out["summary_stability"] = summarise_runs(tables["stability"], ["model"], ["spearman", "top_jaccard"])
    res = tables["resources"]
    out["summary_resources"] = res.groupby(["stage", "model"])[["seconds"] + (["peak_mb"] if "peak_mb" in res else [])].agg(["mean", "std"]).reset_index()
    out["summary_resources"].columns = ["_".join(c).strip("_") for c in out["summary_resources"].columns]

    comps = []
    pairs = [(B2, A_REL), (B1, A_REL), (B2, A_GLOB), (B2, B1), (HYB, B2), (HYB, A_REL), (f"{A_REL}_max", B2)]
    single = cust.loc[cust["experiment"] == "single_scenario"]
    for (scenario, mag), grp in single.groupby(["scenario", "magnitude"]):
        if scenario == CONTROL:
            continue
        for a, b in pairs:
            for metric in ("average_precision", "precision_at_10pct"):
                ga = grp.loc[grp["model"] == a].set_index("seed")[metric]
                gb = grp.loc[grp["model"] == b].set_index("seed")[metric]
                idx = ga.index.intersection(gb.index)
                comps.append(dict(experiment="single_scenario", scenario=scenario, magnitude=mag, model_a=a, model_b=b,
                                  metric=metric, **paired_comparison(ga.loc[idx], gb.loc[idx],
                                                                     exp["bootstrap_resamples"])))
    mixed = cust.loc[cust["experiment"] == "mixed"]
    for (prev, mag), grp in mixed.groupby(["prevalence", "magnitude"]):
        for a, b in pairs:
            for metric in ("average_precision", "precision_at_10pct", "f1"):
                ga = grp.loc[grp["model"] == a].set_index("seed")[metric]
                gb = grp.loc[grp["model"] == b].set_index("seed")[metric]
                idx = ga.index.intersection(gb.index)
                comps.append(dict(experiment="mixed", scenario=f"mixed_p{prev}", magnitude=mag, model_a=a, model_b=b,
                                  metric=metric, **paired_comparison(ga.loc[idx], gb.loc[idx], exp["bootstrap_resamples"])))
    comp = pd.DataFrame(comps)
    if len(comp):
        comp["p_holm"] = np.nan
        for key, grp in comp.groupby(["experiment", "model_a", "model_b", "metric"]):
            comp.loc[grp.index, "p_holm"] = holm_adjust(grp["p_value"].to_numpy())
    out["paired_comparisons"] = comp
    out["hypotheses"] = evaluate_hypotheses(cust, exp)
    return out


def _per_seed_mean(cust: pd.DataFrame, scenarios: list[str], magnitude: str, model: str, metric: str) -> pd.Series:
    sel = cust.loc[(cust["experiment"] == "single_scenario") & cust["scenario"].isin(scenarios)
                   & (cust["magnitude"] == magnitude) & (cust["model"] == model)]
    return sel.groupby("seed")[metric].mean()


def evaluate_hypotheses(cust: pd.DataFrame, exp: dict[str, Any]) -> pd.DataFrame:
    """Apply the pre-registered decision rules (docs/experiment_design.md).

    H1: on persistent scenarios, B2 vs A_obs_relative customer-level AP,
        averaged over persistent scenarios per seed (medium magnitude), paired
        Wilcoxon across seeds; also B1 vs A_obs_relative.
    H2: on transient scenarios, A_obs_relative_max vs B2 customer-level AP
        (same procedure); A_obs_relative (flag-rate) reported as well.
    H3: on the mixed set (prevalence = scenario_prevalence, medium magnitude),
        H_hybrid vs each component.
    Supported = mean difference > 0, Wilcoxon p < 0.05, and the direction holds
    for every magnitude level. Not supported otherwise; "reversed" if the
    opposite difference is significant.
    """
    mag = exp["dev_magnitude"]
    rows = []

    def decide(name, a, b, scenarios, mixed=False):
        per_mag = {}
        for m in exp["magnitudes"]:
            if mixed:
                sel = cust.loc[(cust["experiment"] == "mixed") & (cust["prevalence"] == exp["scenario_prevalence"])
                               & (cust["magnitude"] == m)]
                sa = sel.loc[sel["model"] == a].set_index("seed")["average_precision"]
                sb = sel.loc[sel["model"] == b].set_index("seed")["average_precision"]
            else:
                sa = _per_seed_mean(cust, scenarios, m, a, "average_precision")
                sb = _per_seed_mean(cust, scenarios, m, b, "average_precision")
            if len(sa) and len(sb):
                idx = sa.index.intersection(sb.index)
                per_mag[m] = paired_comparison(sa.loc[idx], sb.loc[idx], exp["bootstrap_resamples"])
        if mag not in per_mag:
            return
        main = per_mag[mag]
        consistent = all(r["mean_diff"] > 0 for r in per_mag.values())
        if main["mean_diff"] > 0 and main["p_value"] < 0.05 and consistent:
            verdict = "supported"
        elif main["mean_diff"] < 0 and main["p_value"] < 0.05:
            verdict = "not supported (opposite difference significant)"
        elif main["mean_diff"] > 0 and main["p_value"] < 0.05:
            verdict = "partially supported (significant at medium magnitude, not at every magnitude)"
        else:
            verdict = "not supported (no significant difference)"
        rows.append(dict(hypothesis=name, model_a=a, model_b=b, scenarios=",".join(scenarios) if not mixed else "mixed",
                         magnitude=mag, mean_ap_diff=main["mean_diff"], ci_low=main["ci_low"], ci_high=main["ci_high"],
                         wins_a=main["wins_a"], wins_b=main["wins_b"], p_value=main["p_value"],
                         direction_by_magnitude="; ".join(f"{m}: {r['mean_diff']:+.3f}" for m, r in per_mag.items()),
                         verdict=verdict))

    decide("H1 (B2 > A on persistent)", B2, A_REL, PERSISTENT)
    decide("H1 (B1 > A on persistent)", B1, A_REL, PERSISTENT)
    decide("H2 (A_max > B2 on transient)", f"{A_REL}_max", B2, TRANSIENT)
    decide("H2 (A flag-rate > B2 on transient)", A_REL, B2, TRANSIENT)
    decide("H3 (hybrid > B2, mixed)", HYB, B2, [], mixed=True)
    decide("H3 (hybrid > A, mixed)", HYB, A_REL, [], mixed=True)
    return pd.DataFrame(rows)


def run_experiments(cfg: dict[str, Any], data_kind: str, seeds: list[int] | None = None, quick: bool = False,
                    make_figures: bool = True) -> Path:
    """Run all seeds, save raw rows, summaries, hypotheses and figures."""
    exp = cfg["experiments"]
    if seeds is None:
        seeds = exp["seeds"] if data_kind == "synthetic" else exp["real_seeds"]
    out_dir = resolve_path(cfg, "outputs_dir") / "experiments" / (data_kind if data_kind == "synthetic" else "real_background")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = prepare_real(cfg) if data_kind == "real" else None
    t0 = time.time()
    tables: dict[str, list] = {}
    for seed in seeds:
        logger.info("=== %s experiments, seed %d (%d/%d) ===", data_kind, seed, seeds.index(seed) + 1, len(seeds))
        res = run_seed(cfg, data_kind, seed, base=base, quick=quick)
        for k, rows in res.items():
            tables.setdefault(k, []).extend(rows)
    frames = {k: pd.DataFrame(v) for k, v in tables.items()}
    for k, df in frames.items():
        df.to_csv(out_dir / f"runs_{k}.csv", index=False)
    summaries = summarise(frames, cfg)
    for k, df in summaries.items():
        df.to_csv(out_dir / f"{k}.csv", index=False)
    meta = {"data_background": data_kind, "seeds": seeds, "quick": quick, "config_hash": config_hash(cfg),
            "runtime_seconds": round(time.time() - t0, 1), "n_customers_per_seed":
                frames["customer"].groupby("seed")["n_customers"].first().to_dict(),
            "label_high_regular_as_anomaly": exp["label_high_regular_as_anomaly"]}
    (out_dir / "experiment_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    if make_figures:
        from .visualization import experiment_figures
        experiment_figures(out_dir, resolve_path(cfg, "outputs_dir") / "figures" / f"experiments_{out_dir.name}")
    logger.info("Experiments finished in %.0fs; results in %s", time.time() - t0, out_dir)
    return out_dir
