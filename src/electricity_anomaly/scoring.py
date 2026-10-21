"""Score normalisation, threshold selection and score fusion.

Convention used everywhere: a higher score means a greater anomaly risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


class RobustFeatureScaler:
    """Log-transform, population-align, impute and robust-scale a feature table.

    * ``log_features`` are transformed with ``log1p`` (non-negative inputs).
    * ``align``: each scored batch is shifted so that its per-feature median
      equals the training median. This removes shifts that affect the whole
      population in the same window (season, weather) but keeps differences
      between customers. It needs a reasonably large batch, so it is skipped
      for batches smaller than ``min_align_rows``.
    * Missing values are imputed with the training median.
    * Scaling uses the training median and IQR (IQR of 0 falls back to 1).
    """

    def __init__(self, features: list[str], log_features: list[str] | None = None, align: bool = True,
                 clip: float | None = None, min_align_rows: int = 20):
        self.features = list(features)
        self.log_features = [f for f in (log_features or []) if f in self.features]
        self.align = align
        self.clip = clip
        self.min_align_rows = min_align_rows

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise KeyError(f"Missing features: {missing}")
        X = df[self.features].astype("float64").copy()
        for f in self.log_features:
            X[f] = np.log1p(X[f].clip(lower=0))
        return X

    def fit(self, df: pd.DataFrame) -> "RobustFeatureScaler":
        X = self._prepare(df)
        self.median_ = X.median()
        iqr = X.quantile(0.75) - X.quantile(0.25)
        self.scale_ = iqr.where(iqr > 1e-9, 1.0)
        return self

    def unscaled(self, df: pd.DataFrame, align: bool | None = None) -> pd.DataFrame:
        """Log-transformed, aligned and imputed values (before scaling)."""
        X = self._prepare(df)
        do_align = self.align if align is None else align
        if do_align and len(X) >= self.min_align_rows:
            X = X - X.median() + self.median_
        return X.fillna(self.median_)

    def transform(self, df: pd.DataFrame, align: bool | None = None) -> pd.DataFrame:
        Z = (self.unscaled(df, align) - self.median_) / self.scale_
        if self.clip is not None:
            Z = Z.clip(-self.clip, self.clip)
        return Z


def empirical_cdf(scores, reference) -> np.ndarray:
    """Fraction of ``reference`` scores less than or equal to each score (0-1)."""
    ref = np.sort(np.asarray(reference, dtype="float64"))
    if ref.size == 0:
        raise ValueError("Reference distribution is empty")
    return np.searchsorted(ref, np.asarray(scores, dtype="float64"), side="right") / ref.size


def quantile_threshold(reference_scores, alert_rate: float) -> float:
    """Threshold that flags ``alert_rate`` of the reference (unlabelled) scores."""
    if not 0 < alert_rate < 1:
        raise ValueError("alert_rate must be in (0, 1)")
    return float(np.quantile(np.asarray(reference_scores, dtype="float64"), 1 - alert_rate))


def f1_optimal_threshold(scores, labels) -> float:
    """Threshold maximising F1 on a *development* set with known labels.

    Must never be called with final test labels; the experiment runner only
    passes development-set (validation window) injections.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype="float64")
    if labels.sum() == 0:
        raise ValueError("Development set has no positive labels")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best]) if len(thresholds) else float(scores.max())


def top_k_flags(scores: pd.Series, k: int) -> pd.Series:
    """Boolean flags for the ``k`` highest scores (ties broken by order)."""
    k = int(max(0, min(k, len(scores))))
    flags = pd.Series(False, index=scores.index)
    if k:
        flags.loc[scores.sort_values(ascending=False, kind="mergesort").index[:k]] = True
    return flags


def hybrid_score(profile_scores: pd.Series, observation_scores: pd.Series,
                 profile_reference, observation_reference, weight: float) -> pd.Series:
    """Weighted mean of calibrated (empirical-CDF) scores of two detectors.

    Each component is first mapped to its percentile within its own clean
    validation-window score distribution, so the two scales are comparable.
    """
    idx = profile_scores.index.intersection(observation_scores.index)
    p = empirical_cdf(profile_scores.loc[idx], profile_reference)
    o = empirical_cdf(observation_scores.loc[idx], observation_reference)
    return pd.Series(weight * p + (1 - weight) * o, index=idx)
