"""Approach A: observation-level anomaly detection with Isolation Forest.

Each customer-day is an independent observation. A day is flagged when its
score exceeds the (1 - contamination) quantile of *training* day scores.
Day results are then aggregated to the customer level with a documented rule
so that both approaches can be compared on the same customers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .feature_engineering import OBS_CALENDAR, OBS_GLOBAL_FEATURES, OBS_RELATIVE_FEATURES
from .scoring import RobustFeatureScaler

AGGREGATIONS = ("flag_rate", "max", "mean", "top3_mean")


class ObservationIsolationForest:
    """Isolation Forest over daily observations.

    Args:
        representation: ``"relative"`` (z-scores against the customer's own
            history) or ``"global"`` (raw daily values).
        contamination: Fraction of training days treated as anomalous when
            setting the day-level threshold.
        align: Remove per-date population shifts (weather, season) by
            subtracting the cross-sectional median of each feature on each date.
    """

    def __init__(self, representation: str = "relative", n_estimators: int = 200, max_samples: int | str = 256,
                 contamination: float = 0.02, align: bool = True, random_state: int = 0):
        if representation not in ("relative", "global"):
            raise ValueError("representation must be 'relative' or 'global'")
        self.representation = representation
        self.features = (OBS_RELATIVE_FEATURES + OBS_CALENDAR) if representation == "relative" else OBS_GLOBAL_FEATURES
        self.continuous = [f for f in self.features if f not in OBS_CALENDAR + ["dow_sin", "dow_cos"]]
        self.contamination = contamination
        self.align = align
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, random_state=random_state, n_jobs=1)

    def _matrix(self, obs: pd.DataFrame) -> pd.DataFrame:
        X = obs[self.features].astype("float64").copy()
        if self.align:
            dates = obs.index.get_level_values("date")
            counts = X.groupby(dates)[self.continuous[0]].transform("size")
            shift = X[self.continuous].groupby(dates).transform("median") - self.train_median_
            X.loc[counts.to_numpy() >= 20, self.continuous] -= shift.loc[counts.to_numpy() >= 20]
        return self.scaler_.transform(X, align=False)

    def fit(self, obs: pd.DataFrame) -> "ObservationIsolationForest":
        """Fit on training-window daily observations and set the day threshold."""
        self.train_median_ = obs[self.continuous].median()
        self.scaler_ = RobustFeatureScaler(self.features, align=False).fit(obs)
        X = self._matrix(obs)
        self.model.fit(X.to_numpy())
        self.train_scores_ = -self.model.score_samples(X.to_numpy())
        self.threshold_ = float(np.quantile(self.train_scores_, 1 - self.contamination))
        return self

    def score(self, obs: pd.DataFrame) -> pd.Series:
        """Anomaly score per day (higher = more anomalous)."""
        X = self._matrix(obs)
        return pd.Series(-self.model.score_samples(X.to_numpy()), index=obs.index, name="observation_score")


def _longest_flag_run(frame: pd.DataFrame) -> pd.Series:
    """Longest run of consecutive calendar days flagged, per customer."""
    f = frame.loc[frame["flag"]].reset_index()[["customer_id", "date"]]
    if f.empty:
        return pd.Series(dtype="float64")
    gap = f.groupby("customer_id")["date"].diff().dt.days.ne(1)
    run_id = gap.cumsum()
    return f.groupby(["customer_id", run_id]).size().groupby(level=0).max()


def aggregate_to_customers(scores: pd.Series, threshold: float) -> pd.DataFrame:
    """Summarise day-level scores and flags for each customer.

    Columns: observation_count, observation_anomaly_count,
    observation_anomaly_rate (frequency), obs_longest_flag_run (persistence),
    obs_severity (mean exceedance over the threshold on flagged days),
    obs_max_score, obs_mean_score, obs_top3_mean_score.
    """
    frame = pd.DataFrame({"score": scores, "flag": scores > threshold})
    g = frame.groupby(level="customer_id")
    out = pd.DataFrame({
        "observation_count": g.size(),
        "observation_anomaly_count": g["flag"].sum().astype(int),
    })
    out["observation_anomaly_rate"] = out["observation_anomaly_count"] / out["observation_count"]
    out["obs_longest_flag_run"] = _longest_flag_run(frame).reindex(out.index).fillna(0).astype(int)
    exceed = (frame["score"] - threshold).where(frame["flag"])
    out["obs_severity"] = exceed.groupby(level="customer_id").mean().reindex(out.index).fillna(0.0)
    out["obs_max_score"] = g["score"].max()
    out["obs_mean_score"] = g["score"].mean()
    top3 = frame["score"].sort_values(ascending=False).groupby(level="customer_id").head(3)
    out["obs_top3_mean_score"] = top3.groupby(level="customer_id").mean()
    return out


def customer_score_from_observations(summary: pd.DataFrame, method: str) -> pd.Series:
    """Customer-level score from aggregated day results.

    * ``flag_rate`` (default): fraction of days flagged. Ties are broken by the
      mean day score (added at 1e-3 scale, smaller than one day's change in the
      rate for windows under 1000 days), so the ranking is lexicographic.
      A single flagged day therefore counts for little in a 90-day window.
    * ``max``: the single most anomalous day (sensitive to isolated spikes).
    * ``mean`` / ``top3_mean``: average over all days / the three worst days.
    """
    if method == "flag_rate":
        return summary["observation_anomaly_rate"] + 1e-3 * summary["obs_mean_score"]
    if method == "max":
        return summary["obs_max_score"].copy()
    if method == "mean":
        return summary["obs_mean_score"].copy()
    if method == "top3_mean":
        return summary["obs_top3_mean_score"].copy()
    raise ValueError(f"Unknown aggregation {method!r}; choose from {AGGREGATIONS}")


def build_observation_model(cfg: dict[str, Any], representation: str, seed: int) -> ObservationIsolationForest:
    """Construct an observation model from configuration."""
    oc = cfg["observation_model"]
    return ObservationIsolationForest(representation=representation, n_estimators=oc["n_estimators"],
                                      max_samples=oc["max_samples"], contamination=oc["observation_contamination"],
                                      align=cfg["features"]["align_population"], random_state=seed)
