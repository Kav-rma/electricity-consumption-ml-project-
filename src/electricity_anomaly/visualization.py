"""Static figures generated directly from saved pipeline and experiment outputs.

Every figure reads CSV/Parquet files written by the pipeline or the
experiment runner; no number is typed in by hand. Colours follow the model
(entity), never its rank, using a fixed colour-blind-checked categorical order.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_COLORS = {
    "A_obs_relative": "#2a78d6",
    "A_obs_global": "#eb6834",
    "B1_peer_kmeans": "#1baf7a",
    "B2_customer_iforest": "#eda100",
    "H_hybrid": "#e87ba4",
    "A_obs_relative_max": "#008300",
    "A_obs_relative_mean": "#6250d6",
    "A_obs_relative_top3_mean": "#e34948",
}
MODEL_LABELS = {
    "A_obs_relative": "A: obs IF (relative)",
    "A_obs_global": "A: obs IF (global)",
    "B1_peer_kmeans": "B1: K-Means peers",
    "B2_customer_iforest": "B2: profile IF",
    "H_hybrid": "Hybrid (B2+A)",
    "A_obs_relative_max": "A: obs IF, max-day",
}
PRIMARY = ["A_obs_relative", "A_obs_global", "B1_peer_kmeans", "B2_customer_iforest", "H_hybrid"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _label(m: str) -> str:
    return MODEL_LABELS.get(m, m)


def grouped_bars(ax, table: pd.DataFrame, err: pd.DataFrame | None, models: list[str]) -> None:
    """Grouped bars: rows of ``table`` are groups, columns are models."""
    n = len(models)
    width = 0.8 / n
    x = np.arange(len(table))
    for i, m in enumerate(models):
        if m not in table:
            continue
        yerr = None
        if err is not None and m in err:
            yerr = np.vstack([table[m] - err[m]["low"], err[m]["high"] - table[m]]).clip(min=0)
        ax.bar(x + (i - (n - 1) / 2) * width, table[m], width * 0.92, color=MODEL_COLORS.get(m, MUTED),
               label=_label(m), yerr=yerr, error_kw=dict(ecolor=MUTED, elinewidth=0.8, capsize=0))
    ax.set_xticks(x)
    ax.set_xticklabels([str(i).replace("_", "\n", 1) for i in table.index], fontsize=8)


# ---------------------------------------------------------------------------
# Pipeline figures
# ---------------------------------------------------------------------------
def pipeline_figures(pipeline_dir: Path, fig_dir: Path, data_kind: str) -> list[Path]:
    """Figures for one pipeline run (synthetic or real)."""
    _style()
    out = []
    tag = "synthetic data" if data_kind == "synthetic" else "real London data (unlabelled)"
    res = pd.read_csv(pipeline_dir / "customer_results.csv", index_col=0)
    scores = pd.read_csv(pipeline_dir / "all_model_scores_test.csv", index_col=0)
    meta = json.loads((pipeline_dir / "run_metadata.json").read_text())

    # 1. Preprocessing exclusions
    stats = json.loads((pipeline_dir / "preprocessing_stats.json").read_text())
    keys = [k for k in ("rows_missing_reading", "rows_non_numeric_reading", "rows_negative_reading",
                        "rows_implausible_reading", "rows_malformed_timestamp", "rows_irregular_timestamp",
                        "exact_duplicate_rows_removed", "conflicting_duplicate_rows") if stats.get(k)]
    if keys:
        fig, ax = plt.subplots(figsize=(6.5, 3))
        vals = [stats[k] for k in keys]
        ax.barh([k.replace("_", " ") for k in keys], vals, color="#2a78d6", height=0.6)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:,}", va="center", fontsize=8, color=TEXT)
        ax.set_xscale("log")
        ax.set_xlabel("rows (log scale)")
        ax.set_title(f"Readings excluded or resolved during cleaning - {tag}")
        out.append(_save(fig, fig_dir / "preprocessing_exclusions.png"))

    # 2. Score distributions (calibrated percentiles vs clean validation)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, (col, m, title) in zip(axes, [("observation_level_score", "A_obs_relative", "A: observation-level"),
                                          ("customer_anomaly_score", "B2_customer_iforest", "B2: customer profile"),
                                          ("hybrid_score", "H_hybrid", "Hybrid")]):
        ax.hist(res[col], bins=20, color=MODEL_COLORS[m], edgecolor="white", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("score percentile vs clean validation window")
    axes[0].set_ylabel("customers")
    fig.suptitle(f"Test-window anomaly score distributions - {tag}", fontsize=10)
    out.append(_save(fig, fig_dir / "score_distributions.png"))

    # 3. K selection
    ks = pd.read_csv(pipeline_dir / "k_selection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    axes[0].plot(ks["k"], ks["silhouette"], marker="o", color="#2a78d6", markersize=6)
    axes[0].axvline(meta["selected_k"], color=MUTED, linestyle=":", linewidth=1)
    axes[0].set(xlabel="number of clusters k", ylabel="mean silhouette", title="Silhouette (training profiles)")
    axes[1].plot(ks["k"], ks["seed_stability_ari"], marker="o", color="#eb6834", markersize=6)
    axes[1].set(xlabel="number of clusters k", ylabel="mean ARI between seeds", title="Partition stability", ylim=(0, 1.05))
    out.append(_save(fig, fig_dir / "k_selection.png"))

    # 4. Cluster load profiles
    cp = pd.read_csv(pipeline_dir / "cluster_load_profiles.csv", index_col=0)
    summ = pd.read_csv(pipeline_dir / "cluster_summary.csv", index_col=0)
    palette = list(MODEL_COLORS.values())
    fig, ax = plt.subplots(figsize=(7, 3.2))
    hours = (np.arange(48) + 0.5) / 2
    for i, (c, row) in enumerate(cp.iterrows()):
        ax.plot(hours, row.to_numpy() * 2, color=palette[i % len(palette)],
                label=f"cluster {c} (n={int(summ.loc[c, 'n_customers'])})")
    ax.set(xlabel="local hour of day", ylabel="mean demand (kW)", title=f"Peer-group load profiles (training window) - {tag}",
           xticks=range(0, 25, 3))
    ax.legend(fontsize=8, ncol=2)
    out.append(_save(fig, fig_dir / "cluster_load_profiles.png"))

    # 5. Observation-level vs customer-level score
    fig, ax = plt.subplots(figsize=(5, 4))
    if "injected_scenario" in res:
        normal = res["injected_scenario"] == "none"
        ax.scatter(res.loc[normal, "observation_level_score"], res.loc[normal, "customer_anomaly_score"], s=14,
                   color="#b5b4ad", label="not injected")
        ax.scatter(res.loc[~normal, "observation_level_score"], res.loc[~normal, "customer_anomaly_score"], s=22,
                   color="#e34948", label="injected (demo labels)", edgecolor="white", linewidth=0.8)
        ax.legend(fontsize=8)
    else:
        ax.scatter(res["observation_level_score"], res["customer_anomaly_score"], s=14, color="#2a78d6")
    ax.set(xlabel="A: observation-level score (percentile)", ylabel="B2: customer-level score (percentile)",
           title="Do the two approaches agree?")
    out.append(_save(fig, fig_dir / "observation_vs_customer_scores.png"))

    # 6. Agreement heatmap
    corr = scores[[m for m in PRIMARY if m in scores]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(corr.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(corr)), [_label(m) for m in corr.columns], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), [_label(m) for m in corr.index], fontsize=8)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else TEXT)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Spearman rank correlation")
    ax.set_title("Rank agreement between detectors (test window)")
    out.append(_save(fig, fig_dir / "model_agreement.png"))

    # 7. Top customers' daily consumption with flagged days
    daily = pd.read_parquet(pipeline_dir / "daily_with_scores.parquet")
    top = res.sort_values("customer_anomaly_score", ascending=False).index[:6]
    fig, axes = plt.subplots(3, 2, figsize=(10, 6.5), sharex=True)
    for ax, cid in zip(axes.ravel(), top):
        d = daily.loc[daily["customer_id"] == cid].sort_values("date")
        ax.plot(d["date"], d["daily_kwh"], color="#2a78d6", linewidth=1.5)
        f = d.loc[d["observation_flag"].eq(True)]
        ax.scatter(f["date"], f["daily_kwh"], color="#e34948", s=16, zorder=3, label="day flagged by A")
        extra = f" | injected: {res.loc[cid, 'injected_scenario']}" if "injected_scenario" in res else ""
        ax.set_title(f"{cid}  B2 pct={res.loc[cid, 'customer_anomaly_score']:.2f}{extra}", fontsize=8)
        ax.set_ylabel("kWh/day")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Highest-ranked customers by B2 (test window) - {tag}", fontsize=10)
    out.append(_save(fig, fig_dir / "top_customers_daily.png"))
    logger.info("Saved %d pipeline figures to %s", len(out), fig_dir)
    return out


# ---------------------------------------------------------------------------
# Experiment figures
# ---------------------------------------------------------------------------
def _ci_table(summary: pd.DataFrame, metric: str, index_col: str, models: list[str]) -> tuple[pd.DataFrame, dict]:
    s = summary.loc[(summary["metric"] == metric) & summary["model"].isin(models)]
    mean = s.pivot_table(index=index_col, columns="model", values="mean")
    err = {m: {"low": s.loc[s["model"] == m].set_index(index_col)["ci_low"].reindex(mean.index),
               "high": s.loc[s["model"] == m].set_index(index_col)["ci_high"].reindex(mean.index)} for m in mean.columns}
    return mean, err


def experiment_figures(exp_dir: Path, fig_dir: Path) -> list[Path]:
    """Figures for one experiment background (synthetic or real-background)."""
    _style()
    out = []
    bg = "synthetic background" if exp_dir.name == "synthetic" else "real-data background (semi-synthetic)"
    summ = pd.read_csv(exp_dir / "summary_customer.csv")
    meta = json.loads((exp_dir / "experiment_metadata.json").read_text())
    n_seeds = len(meta["seeds"])
    mags = ["low", "medium", "high"]
    medium = "medium" if "medium" in set(summ["magnitude"]) else summ["magnitude"].iloc[0]
    models = PRIMARY + ["A_obs_relative_max"]

    single = summ.loc[(summ["experiment"] == "single_scenario") & (summ["magnitude"] == medium)
                      & (summ["scenario"] != "high_regular_control")]
    order = ["persistent_increase", "persistent_decrease", "gradual_drift", "night_increase", "weekday_weekend_shift",
             "variability_increase", "isolated_spikes", "intermittent_days"]
    for metric, fname, ylabel in [("average_precision", "ap_by_scenario.png", "average precision"),
                                  ("precision_at_10pct", "precision_at_10pct_by_scenario.png", "precision@10% of customers")]:
        mean, err = _ci_table(single, metric, "scenario", models)
        mean = mean.reindex([o for o in order if o in mean.index])
        fig, ax = plt.subplots(figsize=(11, 3.6))
        grouped_bars(ax, mean, {m: {k: v.reindex(mean.index) for k, v in e.items()} for m, e in err.items()},
                     [m for m in models if m in mean])
        prev = single["prevalence"].iloc[0] if len(single) else np.nan
        ax.axhline(prev, color=MUTED, linestyle=":", linewidth=1)
        ax.text(-0.45, prev, "random ranking", va="bottom", ha="left", fontsize=7, color=MUTED)
        ax.set(ylabel=ylabel, ylim=(0, 1.05),
               title=f"Customer-level {ylabel} by scenario, {medium} magnitude ({bg}; mean and 95% CI over {n_seeds} seeds)")
        ax.legend(fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        out.append(_save(fig, fig_dir / fname))

    # AP versus magnitude, persistent vs transient
    sc = summ.loc[(summ["experiment"] == "single_scenario") & (summ["metric"] == "average_precision")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (fam, scen) in zip(axes, [("persistent", order[:6]), ("transient", order[6:])]):
        sub = sc.loc[sc["scenario"].isin(scen)].groupby(["magnitude", "model"])["mean"].mean().unstack()
        sub = sub.reindex([m for m in mags if m in sub.index])
        for m in ["A_obs_relative", "A_obs_relative_max", "B1_peer_kmeans", "B2_customer_iforest", "H_hybrid"]:
            if m in sub:
                ax.plot(sub.index, sub[m], marker="o", markersize=6, color=MODEL_COLORS[m], label=_label(m))
        ax.set(title=f"{fam} scenarios (mean over scenarios)", xlabel="injection magnitude", ylim=(0, 1.02))
    axes[0].set_ylabel("average precision")
    axes[1].legend(fontsize=7)
    fig.suptitle(f"Detection quality vs anomaly magnitude - {bg}", fontsize=10)
    out.append(_save(fig, fig_dir / "ap_vs_magnitude.png"))

    # Mixed scenario: threshold metrics at dev-selected thresholds
    mixed = summ.loc[(summ["experiment"] == "mixed") & (summ["magnitude"] == medium)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        mean, err = _ci_table(mixed, metric, "prevalence", PRIMARY)
        grouped_bars(ax, mean, err, [m for m in PRIMARY if m in mean])
        ax.set_xticklabels([f"prevalence {p:.0%}" for p in mean.index])
        ax.set(title=f"{metric} at dev-selected threshold", ylim=(0, 1.05))
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Mixed anomaly test sets, {medium} magnitude - {bg}", fontsize=10)
    out.append(_save(fig, fig_dir / "mixed_threshold_metrics.png"))

    # Detection delay
    delay = pd.read_csv(exp_dir / "summary_delay.csv")
    d = delay.groupby(["days_since_onset", "model"])["mean"].mean().unstack()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for m in PRIMARY:
        if m in d:
            ax.plot(d.index, d[m], marker="o", markersize=6, color=MODEL_COLORS[m], label=_label(m))
    ax.set(xlabel="days since anomaly onset", ylabel="share of injected customers in top-K",
           title=f"Detection of persistent changes over time ({bg})", ylim=(0, 1.02), xticks=list(d.index))
    ax.legend(fontsize=7)
    out.append(_save(fig, fig_dir / "detection_delay.png"))

    # Ablation
    abl = summ.loc[(summ["experiment"] == "single_scenario") & (summ["metric"] == "average_precision")
                   & (summ["magnitude"] == medium) & (summ["scenario"] != "high_regular_control")].copy()
    abl["family"] = np.where(abl["scenario"].isin(order[:6]), "persistent", "transient")
    abl_models = ["B2_level_only", "B2_level_variability", "B2_temporal_only", "B2_profile_no_change", "B2_change_only",
                  "B2_customer_iforest", "B1_peer_kmeans"]
    t = abl.loc[abl["model"].isin(abl_models)].groupby(["model", "family"])["mean"].mean().unstack().reindex(abl_models)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t["persistent"], 0.38, color="#2a78d6", label="persistent scenarios")
    ax.barh(y + 0.2, t["transient"], 0.38, color="#eb6834", label="transient scenarios")
    ax.set_yticks(y, [m.replace("B2_", "IF: ").replace("_", " ") for m in t.index], fontsize=8)
    ax.set(xlabel="mean average precision", title=f"Feature-set ablation, {medium} magnitude ({bg})", xlim=(0, 1))
    ax.legend(fontsize=8)
    out.append(_save(fig, fig_dir / "ablation.png"))

    # Benign high-but-regular controls
    ctrl = summ.loc[(summ["scenario"] == "high_regular_control") & (summ["metric"] == "control_in_top10pct")
                    & summ["model"].isin(PRIMARY)]
    if len(ctrl):
        mean = ctrl.pivot_table(index="magnitude", columns="model", values="mean").reindex(mags)
        fig, ax = plt.subplots(figsize=(7, 3))
        grouped_bars(ax, mean, None, [m for m in PRIMARY if m in mean])
        ax.axhline(0.10, color=MUTED, linestyle=":", linewidth=1)
        ax.set(ylabel="share of controls in top 10%", ylim=(0, 1.05),
               title=f"False alarms on benign high-but-regular customers ({bg}; lower is better)")
        ax.set_xticklabels([f"{m} (x{1 + 2 * {'low': .2, 'medium': .5, 'high': 1.}[m]:.1f} level)" for m in mean.index])
        ax.legend(fontsize=7, ncol=3)
        out.append(_save(fig, fig_dir / "control_false_alarms.png"))

    # Day-level metrics for transient scenarios
    if (exp_dir / "summary_day.csv").exists():
        sd = pd.read_csv(exp_dir / "summary_day.csv")
        sd = sd.loc[sd["scenario"].isin(["isolated_spikes", "intermittent_days"]) & (sd["metric"] == "day_average_precision")]
        if len(sd):
            mean = sd.pivot_table(index=["scenario", "magnitude"], columns="model", values="mean")
            mean.index = [f"{a}\n{b}" for a, b in mean.index]
            fig, ax = plt.subplots(figsize=(8, 3))
            grouped_bars(ax, mean, None, [m for m in ("A_obs_relative", "A_obs_global") if m in mean])
            ax.set(ylabel="day-level average precision", ylim=(0, 1.05),
                   title=f"Observation-level detection of anomalous days ({bg})")
            ax.legend(fontsize=8)
            out.append(_save(fig, fig_dir / "day_level_ap.png"))

    # Stability and runtime
    stab = pd.read_csv(exp_dir / "summary_stability.csv")
    stab = stab.loc[stab["model"].isin(PRIMARY) & (stab["metric"] == "spearman")].set_index("model").reindex(PRIMARY)
    res = pd.read_csv(exp_dir / "summary_resources.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].bar(range(len(stab)), stab["mean"], color=[MODEL_COLORS[m] for m in stab.index],
                yerr=np.vstack([stab["mean"] - stab["ci_low"], stab["ci_high"] - stab["mean"]]).clip(min=0),
                error_kw=dict(ecolor=MUTED, elinewidth=0.8))
    axes[0].set_xticks(range(len(stab)), [_label(m) for m in stab.index], rotation=25, ha="right", fontsize=8)
    axes[0].set(ylabel="Spearman rho", title="Score stability: clean validation vs test window")
    fit = res.loc[res["stage"] == "fit"].set_index("model")["seconds_mean"]
    score = res.loc[res["stage"] == "score"].set_index("model")["seconds_mean"]
    ms = [m for m in PRIMARY if m in fit.index or m in score.index]
    x = np.arange(len(ms))
    axes[1].bar(x - 0.2, fit.reindex(ms).fillna(0), 0.38, color="#2a78d6", label="fit (train window)")
    axes[1].bar(x + 0.2, score.reindex(ms).fillna(0), 0.38, color="#eb6834", label="score one window (incl. features)")
    axes[1].set_xticks(x, [_label(m) for m in ms], rotation=25, ha="right", fontsize=8)
    axes[1].set(ylabel="seconds (mean over seeds)", title="Computation time (fit includes memory-tracing overhead)")
    axes[1].legend(fontsize=7)
    out.append(_save(fig, fig_dir / "stability_and_runtime.png"))
    logger.info("Saved %d experiment figures to %s", len(out), fig_dir)
    return out
