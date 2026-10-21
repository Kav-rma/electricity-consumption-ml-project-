"""Build the Markdown project report from saved results.

``docs/report_template.md`` holds the prose; ``{{NAME}}`` placeholders are
replaced with tables generated from files in ``outputs/``. If a result file is
missing the placeholder becomes an explicit "not yet run" note, so the report
never contains invented numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, resolve_path

PRIMARY = ["A_obs_relative", "A_obs_relative_max", "A_obs_global", "B1_peer_kmeans", "B2_customer_iforest", "H_hybrid"]
SHORT = {"A_obs_relative": "A (flag-rate)", "A_obs_relative_max": "A (max-day)", "A_obs_global": "A-global",
         "B1_peer_kmeans": "B1 peers", "B2_customer_iforest": "B2 profile-IF", "H_hybrid": "Hybrid",
         "B2_level_only": "IF: level", "B2_level_variability": "IF: level+var", "B2_temporal_only": "IF: temporal",
         "B2_profile_no_change": "IF: no change", "B2_change_only": "IF: change only",
         "A_obs_relative_mean": "A (mean)", "A_obs_relative_top3_mean": "A (top-3)"}
SCEN_ORDER = ["persistent_increase", "persistent_decrease", "gradual_drift", "night_increase", "weekday_weekend_shift",
              "variability_increase", "isolated_spikes", "intermittent_days"]


class Missing(Exception):
    """A results file needed for a placeholder does not exist."""


def _md(df: pd.DataFrame, index: bool = True) -> str:
    """Render a DataFrame as a GitHub Markdown table without extra dependencies."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row.to_numpy()) + " |")
    return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "–"
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


def _read(path: Path, **kw) -> pd.DataFrame:
    if not path.exists():
        raise Missing(str(path.relative_to(PROJECT_ROOT)))
    return pd.read_csv(path, **kw)


def _mean_ci(summary: pd.DataFrame, metric: str, rows: str, models: list[str]) -> pd.DataFrame:
    s = summary.loc[(summary["metric"] == metric) & summary["model"].isin(models)].copy()
    s["cell"] = s.apply(lambda r: f"{r['mean']:.3f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]", axis=1)
    t = s.pivot_table(index=rows, columns="model", values="cell", aggfunc="first")
    return t[[m for m in models if m in t.columns]].rename(columns=SHORT)


class ReportBuilder:
    """Generates every placeholder table from the outputs folder."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out = resolve_path(cfg, "outputs_dir")

    def exp(self, kind: str) -> Path:
        return self.out / "experiments" / kind

    # ----------------------------------------------------------------- data
    def preprocessing(self, kind: str) -> str:
        p = self.out / "pipeline" / kind / "preprocessing_stats.json"
        if not p.exists():
            raise Missing(str(p.relative_to(PROJECT_ROOT)))
        stats = json.loads(p.read_text())
        keys = ["raw_rows_scanned", "rows_other_tariff", "rows_not_in_household_sample", "rows_read",
                "rows_missing_reading", "rows_non_numeric_reading", "rows_negative_reading", "rows_implausible_reading",
                "rows_malformed_timestamp", "rows_irregular_timestamp", "zero_readings_kept", "rows_valid",
                "exact_duplicate_rows_removed", "conflicting_duplicate_rows", "local_slots_with_repeated_readings",
                "households_in_sample", "customers_in_study_period", "customer_days_valid",
                "customer_days_partial_excluded", "customer_days_no_data", "customers_eligible",
                "customers_ineligible_low_coverage", "customers_retained_after_sampling"]
        rows = [(k.replace("_", " "), stats[k]) for k in keys if k in stats]
        return _md(pd.DataFrame(rows, columns=["stage / count", "value"]), index=False)

    # ----------------------------------------------------------------- experiments
    def scenario_table(self, kind: str, metric: str = "average_precision") -> str:
        s = _read(self.exp(kind) / "summary_customer.csv")
        s = s.loc[(s["experiment"] == "single_scenario") & (s["magnitude"] == "medium")]
        t = _mean_ci(s, metric, "scenario", PRIMARY).reindex(SCEN_ORDER)
        return _md(t)

    def magnitude_table(self, kind: str) -> str:
        s = _read(self.exp(kind) / "summary_customer.csv")
        s = s.loc[(s["experiment"] == "single_scenario") & (s["metric"] == "average_precision")
                  & s["scenario"].isin(SCEN_ORDER) & s["model"].isin(PRIMARY)].copy()
        s["family"] = np.where(s["scenario"].isin(SCEN_ORDER[:6]), "persistent", "transient")
        t = s.groupby(["family", "magnitude", "model"])["mean"].mean().unstack()
        t = t.reindex(pd.MultiIndex.from_product([["persistent", "transient"], ["low", "medium", "high"]],
                                                 names=["family", "magnitude"]))
        return _md(t[[m for m in PRIMARY if m in t]].rename(columns=SHORT))

    def mixed_table(self, kind: str) -> str:
        s = _read(self.exp(kind) / "summary_customer.csv")
        s = s.loc[(s["experiment"] == "mixed") & s["model"].isin(PRIMARY)
                  & s["metric"].isin(["average_precision", "precision_at_10pct", "precision", "recall", "f1"])]
        t = s.pivot_table(index=["prevalence", "magnitude", "model"], columns="metric", values="mean")
        t = t[["average_precision", "precision_at_10pct", "precision", "recall", "f1"]]
        t = t.reset_index()
        t["model"] = t["model"].map(SHORT)
        t["magnitude"] = pd.Categorical(t["magnitude"], ["low", "medium", "high"])
        return _md(t.sort_values(["prevalence", "magnitude", "model"]), index=False)

    def hypotheses(self, kind: str) -> str:
        h = _read(self.exp(kind) / "hypotheses.csv")
        h = h[["hypothesis", "mean_ap_diff", "ci_low", "ci_high", "wins_a", "wins_b", "p_value",
               "direction_by_magnitude", "verdict"]]
        return _md(h, index=False)

    def paired(self, kind: str) -> str:
        c = _read(self.exp(kind) / "paired_comparisons.csv")
        c = c.loc[(c["experiment"] == "single_scenario") & (c["magnitude"] == "medium") & (c["metric"] == "average_precision")
                  & c["model_a"].isin(["B2_customer_iforest", "B1_peer_kmeans", "A_obs_relative_max"])
                  & (c["model_b"].isin(["A_obs_relative", "B2_customer_iforest"]))]
        c = c.loc[~((c["model_a"] == "B2_customer_iforest") & (c["model_b"] == "B2_customer_iforest"))]
        c = c[["scenario", "model_a", "model_b", "mean_diff", "ci_low", "ci_high", "wins_a", "wins_b", "p_value", "p_holm"]]
        c["model_a"] = c["model_a"].map(SHORT)
        c["model_b"] = c["model_b"].map(SHORT)
        c["scenario"] = pd.Categorical(c["scenario"], SCEN_ORDER)
        return _md(c.sort_values(["model_a", "model_b", "scenario"]), index=False)

    def ablation(self, kind: str) -> str:
        s = _read(self.exp(kind) / "summary_customer.csv")
        models = ["B2_level_only", "B2_level_variability", "B2_temporal_only", "B2_profile_no_change",
                  "B2_change_only", "B2_customer_iforest", "B1_peer_kmeans", "A_obs_relative", "A_obs_relative_max",
                  "A_obs_relative_mean", "A_obs_relative_top3_mean"]
        s = s.loc[(s["experiment"] == "single_scenario") & (s["magnitude"] == "medium") & (s["metric"] == "average_precision")
                  & s["model"].isin(models) & s["scenario"].isin(SCEN_ORDER)]
        t = s.pivot_table(index="model", columns="scenario", values="mean")[SCEN_ORDER]
        t["mean persistent"] = t[SCEN_ORDER[:6]].mean(axis=1)
        t["mean transient"] = t[SCEN_ORDER[6:]].mean(axis=1)
        t = t.reindex(models)
        t.index = [SHORT.get(m, m) for m in t.index]
        t.columns = [c.replace("persistent_", "p_").replace("_shift", "").replace("variability_", "var_")
                     .replace("intermittent_", "interm_") for c in t.columns]
        return _md(t)

    def day_table(self, kind: str) -> str:
        d = _read(self.exp(kind) / "summary_day.csv")
        d = d.loc[d["scenario"].isin(["isolated_spikes", "intermittent_days"])
                  & d["metric"].isin(["day_average_precision", "day_precision", "day_recall", "day_f1"])]
        t = d.pivot_table(index=["scenario", "magnitude", "model"], columns="metric", values="mean")
        t = t.reset_index()
        t["magnitude"] = pd.Categorical(t["magnitude"], ["low", "medium", "high"])
        t["model"] = t["model"].map(SHORT)
        return _md(t.sort_values(["scenario", "magnitude", "model"]), index=False)

    def delay(self, kind: str) -> str:
        d = _read(self.exp(kind) / "summary_delay.csv")
        t = d.groupby(["days_since_onset", "model"])["mean"].mean().unstack()
        return _md(t[[m for m in PRIMARY if m in t]].rename(columns=SHORT))

    def stability(self, kind: str) -> str:
        s = _read(self.exp(kind) / "summary_stability.csv")
        s = s.loc[s["model"].isin(PRIMARY)]
        s["cell"] = s.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
        t = s.pivot_table(index="model", columns="metric", values="cell", aggfunc="first").reindex(PRIMARY)
        t.index = [SHORT[m] for m in t.index]
        return _md(t)

    def resources(self, kind: str) -> str:
        r = _read(self.exp(kind) / "summary_resources.csv")
        r = r.loc[r["model"].isin(PRIMARY + ["customer_features", "observation_features"])].copy()
        keep = [c for c in ["stage", "model", "seconds_mean", "seconds_std", "peak_mb_mean"] if c in r]
        r = r[keep]
        r["model"] = r["model"].map(lambda m: SHORT.get(m, m))
        return _md(r, index=False)

    def controls(self, kind: str) -> str:
        s = _read(self.exp(kind) / "summary_customer.csv")
        s = s.loc[(s["scenario"] == "high_regular_control") & s["model"].isin(PRIMARY)
                  & s["metric"].isin(["control_in_top10pct", "control_flag_rate", "control_vs_normal_auc", "normal_flag_rate"])]
        t = s.pivot_table(index=["magnitude", "model"], columns="metric", values="mean").reset_index()
        t["magnitude"] = pd.Categorical(t["magnitude"], ["low", "medium", "high"])
        t["model"] = t["model"].map(SHORT)
        return _md(t.sort_values(["magnitude", "model"]), index=False)

    def exp_meta(self, kind: str) -> str:
        p = self.exp(kind) / "experiment_metadata.json"
        if not p.exists():
            raise Missing(str(p.relative_to(PROJECT_ROOT)))
        m = json.loads(p.read_text())
        n = sorted(set(m["n_customers_per_seed"].values()))
        return (f"{len(m['seeds'])} seeds ({', '.join(map(str, m['seeds']))}); eligible customers per seed: "
                f"{', '.join(map(str, n))}; run time {m['runtime_seconds'] / 60:.0f} min; config hash `{m['config_hash']}`.")

    def references(self) -> str:
        text = (PROJECT_ROOT / "docs" / "references.md").read_text(encoding="utf-8")
        body = text.split("\n", 1)[1].strip()           # drop the document's own H1 title
        return body.replace("## ", "### ")

    # ----------------------------------------------------------------- real pipeline
    def pipeline_summary(self, kind: str) -> str:
        p = self.out / "pipeline" / kind
        meta_p = p / "run_metadata.json"
        if not meta_p.exists():
            raise Missing(str(meta_p.relative_to(PROJECT_ROOT)))
        meta = json.loads(meta_p.read_text())
        rows = [("customers scored", meta["n_scored_customers"]), ("selected k (silhouette)", meta["selected_k"]),
                ("alert rate for flags", meta["alert_rate"]),
                ("observation-level day threshold (score)", round(meta["observation_day_threshold"], 4))]
        rows += [(f"customers flagged: {SHORT.get(m, m)}", v) for m, v in meta["n_flagged"].items()]
        rows += [("evaluation window", "/".join(meta["windows"]["test"])), ("config hash", meta["config_hash"]),
                 ("run time (s)", meta["runtime_seconds"])]
        return _md(pd.DataFrame(rows, columns=["item", "value"]), index=False)

    def agreement(self, kind: str) -> str:
        a = _read(self.out / "pipeline" / kind / "model_agreement.csv")
        a["model_a"] = a["model_a"].map(lambda m: SHORT.get(m, m))
        a["model_b"] = a["model_b"].map(lambda m: SHORT.get(m, m))
        return _md(a, index=False)

    def pipeline_stability(self, kind: str) -> str:
        s = _read(self.out / "pipeline" / kind / "score_stability_validation_vs_test.csv")
        s["model"] = s["model"].map(lambda m: SHORT.get(m, m))
        return _md(s[["model", "spearman", "top_jaccard", "n_customers"]], index=False)

    def clusters(self, kind: str) -> str:
        c = _read(self.out / "pipeline" / kind / "cluster_summary.csv", index_col=0)
        return _md(c)

    def k_selection(self, kind: str) -> str:
        return _md(_read(self.out / "pipeline" / kind / "k_selection.csv"), index=False)

    def top_customers(self, kind: str, n: int = 10) -> str:
        r = _read(self.out / "pipeline" / kind / "customer_results.csv", index_col=0)
        r = r.sort_values("customer_anomaly_score", ascending=False).head(n)
        r["pattern"] = np.where(r["observation_longest_flag_run_days"] >= 7, "persistent (>= 7-day flagged run)",
                                np.where(r["observation_anomaly_count"] <= 2, "isolated / none at day level",
                                         "intermittent"))
        cols = ["customer_anomaly_score", "peer_deviation_score", "observation_level_score", "mean_daily_kwh",
                "behavioural_cluster", "observation_anomaly_count", "observation_longest_flag_run_days", "pattern",
                "principal_behavioural_deviations"]
        return _md(r[cols])

    def pattern_counts(self, kind: str) -> str:
        r = _read(self.out / "pipeline" / kind / "customer_results.csv", index_col=0)
        rows = []
        for name, flag in (("B2 flagged", "customer_anomaly_flag"), ("B1 flagged", "peer_deviation_flag"),
                           ("A flagged", "observation_level_flag"), ("not flagged by any", None)):
            sel = r.loc[r[flag]] if flag else r.loc[~(r["customer_anomaly_flag"] | r["peer_deviation_flag"]
                                                       | r["observation_level_flag"])]
            rows.append(dict(group=name, customers=len(sel),
                             median_flagged_days=sel["observation_anomaly_count"].median(),
                             median_longest_run=sel["observation_longest_flag_run_days"].median(),
                             share_with_7day_run=(sel["observation_longest_flag_run_days"] >= 7).mean() if len(sel) else np.nan,
                             share_with_0_to_2_flagged_days=(sel["observation_anomaly_count"] <= 2).mean() if len(sel) else np.nan))
        both = r["customer_anomaly_flag"] & r["observation_level_flag"]
        rows.append(dict(group="flagged by both B2 and A", customers=int(both.sum())))
        return _md(pd.DataFrame(rows), index=False)

    def deviation_counts(self, kind: str) -> str:
        r = _read(self.out / "pipeline" / kind / "customer_results.csv", index_col=0)
        flagged = r.loc[r["customer_anomaly_flag"] | r["peer_deviation_flag"], "principal_behavioural_deviations"]
        feats = flagged.str.split("; ").explode().str.extract(r"^(\w+)")[0]
        return _md(feats.value_counts().head(10).rename_axis("feature").to_frame("times among top-3 deviations of flagged customers"))


def build_report(cfg: dict, template: Path | None = None, out_path: Path | None = None) -> Path:
    """Fill ``docs/report_template.md`` and write ``outputs/reports/project_report.md``."""
    template = template or PROJECT_ROOT / "docs" / "report_template.md"
    out_path = out_path or resolve_path(cfg, "outputs_dir") / "reports" / "project_report.md"
    rb = ReportBuilder(cfg)
    text = template.read_text(encoding="utf-8")

    def replace(match: re.Match) -> str:
        name, *args = match.group(1).split(":")
        fn: Callable[..., str] | None = getattr(rb, name, None)
        if fn is None:
            raise KeyError(f"Unknown report placeholder {match.group(0)}")
        try:
            return fn(*args)
        except Missing as exc:
            return f"> *Result not available: `{exc}` is missing. Run the corresponding command in docs/reproducibility.md.*"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(re.sub(r"\{\{([\w:]+)\}\}", replace, text), encoding="utf-8")
    return out_path
