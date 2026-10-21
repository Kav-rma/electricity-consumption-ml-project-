"""Interactive dashboard for the electricity anomaly-detection project.

Run from the project root:
    streamlit run app/streamlit_app.py

The app only *reads* saved outputs; it never retrains models when a filter
changes. The "Settings & runs" tab can start the pipeline as a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from electricity_anomaly.config import DEFAULT_CONFIG_PATH, ConfigError, load_config, validate_config  # noqa: E402
from electricity_anomaly.feature_engineering import ALL_FEATURES, FEATURE_GROUPS  # noqa: E402

# ELECTRICITY_OUTPUTS_DIR lets tests point the app at an empty or temporary folder.
OUT = Path(os.environ.get("ELECTRICITY_OUTPUTS_DIR", ROOT / "outputs"))
FEATURES = Path(os.environ.get("ELECTRICITY_FEATURES_DIR", ROOT / "data" / "features"))
DASHBOARD_CONFIG = ROOT / "config" / "dashboard_config.yaml"
MODEL_COLORS = {"A_obs_relative": "#2a78d6", "A_obs_global": "#eb6834", "B1_peer_kmeans": "#1baf7a",
                "B2_customer_iforest": "#eda100", "H_hybrid": "#e87ba4", "A_obs_relative_max": "#008300"}
MODEL_LABELS = {"A_obs_relative": "A: observation IF (relative)", "A_obs_global": "A: observation IF (global)",
                "B1_peer_kmeans": "B1: K-Means peer deviation", "B2_customer_iforest": "B2: customer-profile IF",
                "H_hybrid": "Hybrid (B2 + A)", "A_obs_relative_max": "A: observation IF, max-day"}
PRIMARY = list(MODEL_LABELS)[:5]
SCORE_DEFINITIONS = {
    "customer_anomaly_score": "B2 percentile: share of clean validation-window customers with a lower Isolation "
                              "Forest score (0-1, higher = more unusual profile).",
    "observation_level_score": "A percentile: share of validation customers with a lower flagged-day rate "
                               "(days whose observation score exceeds the training 98th percentile).",
    "peer_deviation_score": "B1: mean of the 3 largest robust z-scores (in peer-group MADs) between the customer's "
                            "features and their peer group's training median.",
    "hybrid_score": "Mean of the B2 and A percentiles.",
}

st.set_page_config(page_title="Electricity anomaly detection", page_icon="⚡", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders (keyed on file modification time so new runs are picked up)
# ---------------------------------------------------------------------------
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def read_csv(path: str, mtime: float, **kw) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_parquet(path: str, mtime: float) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_json(path: str, mtime: float) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def csv(path: Path, **kw):
    return read_csv(str(path), _mtime(path), **kw)


def parquet(path: Path):
    return read_parquet(str(path), _mtime(path))


def jsonf(path: Path):
    return read_json(str(path), _mtime(path))


def available_pipelines() -> list[str]:
    return [k for k in ("synthetic", "real") if (OUT / "pipeline" / k / "customer_results.csv").exists()]


def missing_message(what: str, command: str) -> None:
    st.info(f"No {what} found yet. Run:\n\n```powershell\n{command}\n```")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ Anomaly detection")
pipes = available_pipelines()
data_kind = st.sidebar.radio("Dataset", pipes, format_func=lambda k: "Synthetic (demo labels)" if k == "synthetic"
                             else "Real London households (unlabelled)") if pipes else None
st.sidebar.caption("Synthetic and real results are stored and shown separately. Real-data rankings are "
                   "signals for investigation, not evidence of theft or wrongdoing.")
PIPE = OUT / "pipeline" / (data_kind or "none")
meta = jsonf(PIPE / "run_metadata.json") if data_kind else None
if meta:
    st.sidebar.markdown(f"**Run:** {meta['created_utc']}  \n**Config hash:** `{meta['config_hash']}`  \n"
                        f"**Model version:** {meta['model_version']}  \n**Seed:** {meta['random_seed']}")

tabs = st.tabs(["Overview", "Customer explorer", "Anomaly leaderboard", "Model comparison",
                "Behavioural profiles", "Settings & runs"])


def need_pipeline() -> bool:
    if not data_kind:
        missing_message("pipeline outputs", "python scripts/generate_synthetic_data.py\n"
                                            "python scripts/run_pipeline.py --data synthetic")
        return False
    return True


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Overview")
    if need_pipeline():
        res = csv(PIPE / "customer_results.csv", index_col=0)
        stats = jsonf(PIPE / "preprocessing_stats.json") or {}
        c = st.columns(5)
        c[0].metric("Eligible customers scored", f"{len(res):,}")
        c[1].metric("Valid readings processed", f"{stats.get('rows_valid', 0):,}")
        c[2].metric("Valid customer-days", f"{stats.get('customer_days_valid', 0):,}")
        c[3].metric("Customers in study period", f"{stats.get('customers_in_study_period', 0):,}")
        c[4].metric("Selected peer groups (k)", meta.get("selected_k", "–"))
        st.caption(f"Evaluation period: {res['evaluation_period'].iloc[0]} · windows: "
                   + ", ".join(f"{k} {v[0]}→{v[1]}" for k, v in meta["windows"].items()))

        left, right = st.columns(2)
        with left:
            st.subheader("Customers flagged by each model")
            flagged = pd.DataFrame({"model": list(meta["n_flagged"]), "flagged": list(meta["n_flagged"].values())})
            fig = px.bar(flagged, x="model", y="flagged", color="model", color_discrete_map=MODEL_COLORS,
                         labels={"flagged": "customers flagged"})
            fig.update_layout(showlegend=False, height=320, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Threshold per model: the top {meta['alert_rate']:.0%} of clean validation-window scores "
                       "(unsupervised calibration, no labels).")
        with right:
            st.subheader("Distribution of customer anomaly scores")
            col = st.selectbox("Score", list(SCORE_DEFINITIONS), key="ov_score")
            fig = px.histogram(res, x=col, nbins=30, color_discrete_sequence=["#2a78d6"])
            fig.update_layout(height=320, margin=dict(t=10), bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(SCORE_DEFINITIONS[col])

        st.subheader("Preprocessing statistics")
        cov = csv(PIPE / "coverage.csv", index_col=0)
        a, b = st.columns([1, 1])
        a.dataframe(pd.Series(stats, name="value").astype(str), use_container_width=True, height=360)
        if cov is not None:
            cov_cols = [c for c in cov.columns if c.startswith("coverage_")]
            long = cov[cov_cols].melt(var_name="window", value_name="share of valid days")
            fig = px.box(long, x="window", y="share of valid days", points=False)
            fig.update_layout(height=360, margin=dict(t=10))
            b.plotly_chart(fig, use_container_width=True)
            b.caption("Per-customer share of valid days in each window (customers retained after sampling).")


# ---------------------------------------------------------------------------
# Customer explorer
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Customer explorer")
    if need_pipeline():
        res = csv(PIPE / "customer_results.csv", index_col=0)
        daily = parquet(PIPE / "daily_with_scores.parquet")
        feats_test = parquet(FEATURES / data_kind / "customer_features_test.parquet")
        feats_train = parquet(FEATURES / data_kind / "customer_features_train.parquet")
        clusters_train = csv(PIPE / "train_clusters.csv", index_col=0)
        profiles = parquet(PIPE / "load_profiles.parquet")
        cluster_prof = csv(PIPE / "cluster_load_profiles.csv", index_col=0)

        options = res.index.tolist()
        query = st.text_input("Search customer ID", "")
        if query:
            options = [o for o in options if query.lower() in o.lower()] or options
        cid = st.selectbox("Customer (sorted by B2 score)", options)
        row = res.loc[cid]
        c = st.columns(5)
        c[0].metric("B2 profile score", f"{row['customer_anomaly_score']:.2f}", "flagged" if row["customer_anomaly_flag"] else None)
        c[1].metric("A observation score", f"{row['observation_level_score']:.2f}", "flagged" if row["observation_level_flag"] else None)
        c[2].metric("B1 peer deviation", f"{row['peer_deviation_score']:.1f}", "flagged" if row["peer_deviation_flag"] else None)
        c[3].metric("Flagged days (A)", f"{int(row['observation_anomaly_count'])} / {int(row['observation_count'])}")
        c[4].metric("Peer group", int(row["behavioural_cluster"]))
        if "injected_scenario" in res:
            st.caption(f"Synthetic demo ground truth: **{row['injected_scenario']}** (never shown to the models).")

        d = daily.loc[daily["customer_id"] == cid].sort_values("date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["date"], y=d["daily_kwh"], mode="lines", name="daily kWh",
                                 line=dict(color="#2a78d6", width=1.6)))
        f = d.loc[d["observation_flag"].eq(True)]
        fig.add_trace(go.Scatter(x=f["date"], y=f["daily_kwh"], mode="markers", name="day flagged by A",
                                 marker=dict(color="#e34948", size=8, line=dict(color="white", width=1))))
        start, end = row["evaluation_period"].split("/")
        fig.add_vrect(x0=start, x1=end, fillcolor="#86b6ef", opacity=0.12, line_width=0,
                      annotation_text="evaluation window", annotation_position="top left")
        fig.update_layout(height=340, margin=dict(t=30), yaxis_title="kWh per day (24-h equivalent)",
                          hovermode="x unified", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Why was this customer ranked here?")
        a, b = st.columns(2)
        a.markdown("**Largest deviations from peer group (B1)**")
        for item in str(row["principal_behavioural_deviations"]).split("; "):
            a.markdown(f"- `{item}`")
        b.markdown("**Features that raised the B2 Isolation Forest score** (occlusion attribution)")
        for item in str(row["iforest_top_features"]).split("; "):
            b.markdown(f"- `{item}`")
        st.caption(f"A: longest run of consecutive flagged days = {int(row['observation_longest_flag_run_days'])}; "
                   f"mean exceedance on flagged days = {row['observation_severity']:.3f}. "
                   "A long run suggests a persistent change; one or two isolated days suggest a transient event.")

        a, b = st.columns(2)
        with a:
            st.markdown("**Load profile vs peer-group reference**")
            fig = go.Figure()
            hours = (np.arange(48) + 0.5) / 2
            if profiles is not None and ("test", cid) in profiles.index:
                fig.add_trace(go.Scatter(x=hours, y=profiles.loc[("test", cid)].to_numpy() * 2, name="customer (evaluation window)",
                                         line=dict(color="#2a78d6")))
            if profiles is not None and ("validation", cid) in profiles.index:
                fig.add_trace(go.Scatter(x=hours, y=profiles.loc[("validation", cid)].to_numpy() * 2, name="customer (previous window)",
                                         line=dict(color="#2a78d6", dash="dot")))
            cl = int(row["behavioural_cluster"])
            if cluster_prof is not None and cl in cluster_prof.index:
                fig.add_trace(go.Scatter(x=hours, y=cluster_prof.loc[cl].to_numpy() * 2, name=f"peer group {cl} mean (training)",
                                         line=dict(color="#eb6834")))
            fig.update_layout(height=320, margin=dict(t=10), xaxis_title="local hour", yaxis_title="mean kW",
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)
        with b:
            st.markdown("**Behavioural features vs peer-group median (training window)**")
            if feats_test is not None and cid in feats_test.index:
                group = st.selectbox("Feature group", list(FEATURE_GROUPS), key="fg")
                cols = FEATURE_GROUPS[group]
                members = clusters_train.index[clusters_train["behavioural_cluster"] == cl]
                peer_med = feats_train.loc[feats_train.index.intersection(members), cols].median()
                st.dataframe(pd.DataFrame({"customer": feats_test.loc[cid, cols], "peer median": peer_med}).round(3),
                             use_container_width=True, height=290)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Anomaly leaderboard")
    if need_pipeline():
        res = csv(PIPE / "customer_results.csv", index_col=0)
        thresholds = meta["thresholds_from_clean_validation"]
        a, b, c = st.columns(3)
        ranking = a.radio("Ranking", ["Customer-level (B2)", "Peer deviation (B1)", "Observation-level (A)", "Hybrid"])
        sort_col = {"Customer-level (B2)": "customer_anomaly_score", "Peer deviation (B1)": "peer_deviation_score",
                    "Observation-level (A)": "observation_level_score", "Hybrid": "hybrid_score"}[ranking]
        clusters = sorted(res["behavioural_cluster"].unique())
        sel_clusters = b.multiselect("Peer group", clusters, default=clusters)
        only_flagged = b.checkbox("Only customers flagged by this ranking")
        scen = None
        if "injected_scenario" in res:
            scen = c.multiselect("Injected scenario (synthetic demo)", sorted(res["injected_scenario"].unique()))
        c.caption(f"Evaluation period: {res['evaluation_period'].iloc[0]}")
        view = res.loc[res["behavioural_cluster"].isin(sel_clusters)]
        flag_col = {"customer_anomaly_score": "customer_anomaly_flag", "peer_deviation_score": "peer_deviation_flag",
                    "observation_level_score": "observation_level_flag", "hybrid_score": "hybrid_flag"}[sort_col]
        if only_flagged:
            view = view.loc[view[flag_col]]
        if scen:
            view = view.loc[view["injected_scenario"].isin(scen)]
        view = view.sort_values(sort_col, ascending=False)
        show = ["customer_anomaly_score", "observation_level_score", "peer_deviation_score", "hybrid_score",
                flag_col, "behavioural_cluster", "mean_daily_kwh", "observation_anomaly_count",
                "observation_longest_flag_run_days", "principal_behavioural_deviations"]
        if "injected_scenario" in res:
            show.append("injected_scenario")
        st.dataframe(view[list(dict.fromkeys(show))], use_container_width=True, height=480)
        st.markdown(f"**Score definition:** {SCORE_DEFINITIONS[sort_col]}")
        st.markdown("**Thresholds** (raw model scores at the top-"
                    f"{meta['alert_rate']:.0%} of clean validation scores): "
                    + ", ".join(f"`{m}` > {v:.4g}" for m, v in thresholds.items()))
        st.download_button("Download table (CSV)", view.to_csv().encode(), file_name=f"leaderboard_{data_kind}.csv")


# ---------------------------------------------------------------------------
# Model comparison (experiments)
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Model comparison")
    exp_kinds = [k for k in ("synthetic", "real_background") if (OUT / "experiments" / k / "summary_customer.csv").exists()]
    if not exp_kinds:
        missing_message("experiment results", "python scripts/run_experiments.py --data synthetic")
    else:
        kind = st.radio("Experiment background", exp_kinds, horizontal=True,
                        format_func=lambda k: "Synthetic households" if k == "synthetic"
                        else "Real households + injected anomalies (semi-synthetic)")
        E = OUT / "experiments" / kind
        emeta = jsonf(E / "experiment_metadata.json")
        st.caption(f"Labelled controlled experiment · {len(emeta['seeds'])} seeds · config `{emeta['config_hash']}`. "
                   "Metrics are computed against injected ground truth. Unlabelled real-data rankings are in the "
                   "other tabs and have no accuracy figures.")
        summ = csv(E / "summary_customer.csv")
        hyp = csv(E / "hypotheses.csv")
        if hyp is not None:
            st.subheader("Pre-registered hypotheses")
            st.dataframe(hyp[["hypothesis", "mean_ap_diff", "ci_low", "ci_high", "wins_a", "wins_b", "p_value",
                              "direction_by_magnitude", "verdict"]].round(4), use_container_width=True)
        a, b, c = st.columns(3)
        metric = a.selectbox("Metric", ["average_precision", "precision_at_10pct", "recall_at_10pct", "precision_at_5pct",
                                        "r_precision", "roc_auc", "precision", "recall", "f1"])
        mags = [m for m in ("low", "medium", "high") if m in set(summ["magnitude"])]
        mag = b.selectbox("Magnitude", mags, index=mags.index("medium") if "medium" in mags else 0)
        models = c.multiselect("Models", sorted(summ["model"].unique()), default=[m for m in PRIMARY + ["A_obs_relative_max"]
                                                                                  if m in set(summ["model"])])
        sel = summ.loc[(summ["experiment"] == "single_scenario") & (summ["magnitude"] == mag) & (summ["metric"] == metric)
                       & summ["model"].isin(models) & (summ["scenario"] != "high_regular_control")]
        fig = px.bar(sel, x="scenario", y="mean", color="model", barmode="group", color_discrete_map=MODEL_COLORS,
                     error_y=sel["ci_high"] - sel["mean"], error_y_minus=sel["mean"] - sel["ci_low"],
                     labels={"mean": metric})
        fig.update_layout(height=420, legend=dict(orientation="h", y=-0.35), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bars: mean over seeds; whiskers: 95% t-interval. Precision/recall/F1 use thresholds selected on a "
                   "separate development injection in the validation window.")

        st.subheader("Mixed-scenario test sets")
        mixed = summ.loc[(summ["experiment"] == "mixed") & (summ["metric"].isin(["average_precision", "precision_at_10pct",
                                                                                   "precision", "recall", "f1"]))
                         & summ["model"].isin(models)]
        st.dataframe(mixed.pivot_table(index=["prevalence", "magnitude", "model"], columns="metric", values="mean").round(3),
                     use_container_width=True)

        a, b = st.columns(2)
        with a:
            st.subheader("Detection delay (persistent)")
            delay = csv(E / "summary_delay.csv")
            dd = delay.groupby(["days_since_onset", "model"])["mean"].mean().reset_index()
            fig = px.line(dd, x="days_since_onset", y="mean", color="model", markers=True, color_discrete_map=MODEL_COLORS,
                          labels={"mean": "share of injected customers in top 10%"})
            fig.update_layout(height=340, margin=dict(t=10), legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig, use_container_width=True)
        with b:
            st.subheader("Score stability and run time")
            stab = csv(E / "summary_stability.csv")
            res_t = csv(E / "summary_resources.csv")
            st.dataframe(stab.loc[stab["model"].isin(PRIMARY)].pivot_table(index="model", columns="metric", values="mean").round(3),
                         use_container_width=True)
            st.dataframe(res_t.round(3), use_container_width=True, height=220)

        st.subheader("Benign high-but-regular controls (false alarms; lower is better)")
        ctrl = summ.loc[(summ["scenario"] == "high_regular_control") & summ["metric"].isin(["control_in_top10pct", "control_flag_rate",
                                                                                           "control_vs_normal_auc"])]
        st.dataframe(ctrl.pivot_table(index=["magnitude", "model"], columns="metric", values="mean").round(3), use_container_width=True)
        day = csv(E / "summary_day.csv")
        if day is not None:
            st.subheader("Observation-level (day) metrics for Approach A")
            st.dataframe(day.loc[day["scenario"].isin(["isolated_spikes", "intermittent_days"])]
                         .pivot_table(index=["scenario", "magnitude", "model"], columns="metric", values="mean").round(3),
                         use_container_width=True)
        figs = OUT / "figures" / f"experiments_{kind}"
        if figs.exists():
            with st.expander("Static report figures"):
                for p in sorted(figs.glob("*.png")):
                    st.image(str(p), caption=p.name)


# ---------------------------------------------------------------------------
# Behavioural profiles
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Behavioural profiles")
    if need_pipeline():
        cp = csv(PIPE / "cluster_load_profiles.csv", index_col=0)
        summ = csv(PIPE / "cluster_summary.csv", index_col=0)
        ks = csv(PIPE / "k_selection.csv")
        kss = csv(PIPE / "k_sensitivity.csv")
        res = csv(PIPE / "customer_results.csv", index_col=0)
        feats = parquet(FEATURES / data_kind / "customer_features_test.parquet")
        a, b = st.columns([3, 2])
        with a:
            st.subheader("Representative load profile of each peer group")
            long = cp.T
            long.index = (np.arange(48) + 0.5) / 2
            long = (long * 2).reset_index().melt(id_vars="index", var_name="cluster", value_name="kW")
            fig = px.line(long, x="index", y="kW", color="cluster", labels={"index": "local hour"},
                          color_discrete_sequence=list(MODEL_COLORS.values()))
            fig.update_layout(height=360, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        with b:
            st.subheader("Cluster summaries (training window)")
            st.dataframe(summ.round(3), use_container_width=True)
        st.subheader("Customers in feature space")
        c1, c2 = st.columns(2)
        fx = c1.selectbox("x feature", ALL_FEATURES, index=ALL_FEATURES.index("mean_daily_kwh"))
        fy = c2.selectbox("y feature", ALL_FEATURES, index=ALL_FEATURES.index("night_share"))
        plot = feats[[fx, fy]].join(res[["behavioural_cluster", "peer_deviation_score", "customer_anomaly_score"]], how="inner")
        plot["behavioural_cluster"] = plot["behavioural_cluster"].astype(str)
        fig = px.scatter(plot, x=fx, y=fy, color="behavioural_cluster", size=plot["peer_deviation_score"].clip(upper=15) + 1,
                         hover_name=plot.index, hover_data=["peer_deviation_score", "customer_anomaly_score"],
                         color_discrete_sequence=list(MODEL_COLORS.values()))
        fig.update_layout(height=420, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Point size = peer deviation score (B1). Peer groups use consumption behaviour only; no demographic "
                   "data is available, so 'peers' means customers who consumed similarly in the previous window.")
        a, b = st.columns(2)
        with a:
            st.subheader("Choosing k (training data only)")
            st.dataframe(ks.round(3), use_container_width=True)
        with b:
            st.subheader("Sensitivity of B1 rankings to k and initialisation")
            if kss is not None:
                st.dataframe(kss.groupby("k")[["spearman_vs_selected", "top10_overlap"]].agg(["mean", "min"]).round(3),
                             use_container_width=True)


# ---------------------------------------------------------------------------
# Settings & runs
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("Data and experiment settings")
    try:
        base = load_config(DASHBOARD_CONFIG if DASHBOARD_CONFIG.exists() else DEFAULT_CONFIG_PATH)
    except ConfigError as exc:
        st.error(f"{exc}. Delete config/dashboard_config.yaml to return to the defaults.")
        st.stop()
    if meta:
        st.markdown("**Configuration that produced the current pipeline results**")
        st.json({k: meta[k] for k in ("data_source", "created_utc", "config_hash", "random_seed", "windows",
                                      "n_eligible_customers", "selected_k", "alert_rate", "demo_injection",
                                      "model_descriptions")}, expanded=False)
    st.markdown("Edit supported parameters below. They are saved to `config/dashboard_config.yaml` "
                "(the main `config.yaml` is never modified).")
    with st.form("settings"):
        a, b, c = st.columns(3)
        n_cust = a.number_input("Synthetic customers", 30, 2000, int(base["synthetic"]["n_customers"]), step=10)
        seed = a.number_input("Random seed", 0, 10_000, int(base["random_seed"]))
        alert = b.slider("Alert rate (share of customers flagged)", 0.01, 0.30, float(base["customer_models"]["alert_rate"]))
        contam = b.slider("Observation day contamination", 0.005, 0.10, float(base["observation_model"]["observation_contamination"]))
        rep = c.selectbox("Observation representation", ["relative", "global"],
                          index=["relative", "global"].index(base["observation_model"]["representation"]))
        k_max = c.number_input("K-Means max k", 2, 15, int(base["customer_models"]["kmeans"]["k_max"]))
        max_hh = a.number_input("Real data: max households", 30, 6000, int(base["real"]["max_households"]), step=50)
        frac = b.slider("Real data: household sample fraction", 0.01, 1.0, float(base["real"]["household_sample_fraction"]))
        if st.form_submit_button("Save settings"):
            new = dict(base)
            new["synthetic"] = {**base["synthetic"], "n_customers": int(n_cust)}
            new["random_seed"] = int(seed)
            new["customer_models"] = {**base["customer_models"], "alert_rate": float(alert),
                                      "kmeans": {**base["customer_models"]["kmeans"], "k_max": int(k_max)}}
            new["observation_model"] = {**base["observation_model"], "observation_contamination": float(contam),
                                        "representation": rep}
            new["real"] = {**base["real"], "max_households": int(max_hh), "household_sample_fraction": float(frac)}
            try:
                validate_config(new)
                DASHBOARD_CONFIG.write_text(yaml.safe_dump(new, sort_keys=False))
                st.success(f"Saved {DASHBOARD_CONFIG.relative_to(ROOT)}")
            except ConfigError as exc:
                st.error(f"Invalid settings: {exc}")

    cfg_arg = ["--config", str(DASHBOARD_CONFIG)] if DASHBOARD_CONFIG.exists() else []
    st.subheader("Run")
    st.markdown("Commands (PowerShell, project root):")
    st.code("python scripts/generate_synthetic_data.py " + " ".join(cfg_arg) + "\n"
            "python scripts/run_pipeline.py --data synthetic " + " ".join(cfg_arg) + "\n"
            "python scripts/run_pipeline.py --data real " + " ".join(cfg_arg) + "\n"
            "python scripts/run_experiments.py --data synthetic " + " ".join(cfg_arg), language="powershell")
    a, b = st.columns(2)
    which = a.selectbox("Pipeline to run now", ["synthetic", "real"])
    if b.button("Run pipeline (takes 1-5 minutes)"):
        with st.spinner(f"Running the {which} pipeline..."):
            proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--data", which, *cfg_arg],
                                  cwd=ROOT, capture_output=True, text=True)
        if proc.returncode == 0:
            st.success("Pipeline finished. Reload the page to see the new results.")
            st.cache_data.clear()
        else:
            st.error("Pipeline failed")
        st.code((proc.stdout + proc.stderr)[-4000:])
    st.caption("Full experiments take tens of minutes; run them from a terminal with the command above.")
