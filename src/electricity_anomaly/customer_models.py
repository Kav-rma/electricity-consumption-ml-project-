"""Approach B2: Isolation Forest on customer behavioural profiles.

The model is trained on training-window customer profiles only and then
scores profiles from later windows. The continuous score
(``-score_samples``, higher = more anomalous) is used for ranking; the
model's built-in binary prediction is not used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .feature_engineering import ALL_FEATURES, LOG_FEATURES
from .scoring import RobustFeatureScaler


class CustomerIsolationForest:
    """Isolation Forest over customer feature vectors."""

    def __init__(self, features: list[str] | None = None, n_estimators: int = 300, max_samples: int | str = "auto",
                 align: bool = True, random_state: int = 0):
        self.features = list(features or ALL_FEATURES)
        self.scaler = RobustFeatureScaler(self.features, LOG_FEATURES, align=align)
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                     random_state=random_state, n_jobs=1)

    def fit(self, train_features: pd.DataFrame) -> "CustomerIsolationForest":
        X = self.scaler.fit(train_features).transform(train_features)
        self.model.fit(X.to_numpy())
        self.train_scores_ = pd.Series(-self.model.score_samples(X.to_numpy()), index=train_features.index)
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Customer anomaly score (higher = more anomalous)."""
        X = self.scaler.transform(features)
        return pd.Series(-self.model.score_samples(X.to_numpy()), index=features.index, name="customer_iforest_score")

    def feature_contributions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Occlusion attribution: score drop when one feature is reset to the training median.

        For customer i and feature j, contribution = score(x_i) - score(x_i with
        feature j set to its training median). Positive values mean the
        feature's value made the customer look more anomalous. This is a simple,
        model-agnostic explanation; it ignores feature interactions.
        """
        X = self.scaler.transform(features).to_numpy()
        base = -self.model.score_samples(X)
        contrib = np.empty_like(X)
        for j in range(X.shape[1]):
            Xj = X.copy()
            Xj[:, j] = 0.0                     # scaled training median
            contrib[:, j] = base - (-self.model.score_samples(Xj))
        return pd.DataFrame(contrib, index=features.index, columns=self.features)


def top_contributions(contrib: pd.DataFrame, features: pd.DataFrame, n: int = 3) -> pd.Series:
    """Text summary of the ``n`` features that increased each customer's score most."""
    out = {}
    for cid, row in contrib.iterrows():
        top = row.nlargest(n)
        top = top[top > 0]
        out[cid] = "; ".join(f"{f}={features.at[cid, f]:.3g} (+{v:.3f})" for f, v in top.items()) or "no single dominant feature"
    return pd.Series(out, name="iforest_top_features")


def build_customer_iforest(cfg: dict[str, Any], seed: int, features: list[str] | None = None) -> CustomerIsolationForest:
    """Construct the customer-profile Isolation Forest from configuration."""
    ic = cfg["customer_models"]["isolation_forest"]
    return CustomerIsolationForest(features=features, n_estimators=ic["n_estimators"], max_samples=ic["max_samples"],
                                   align=cfg["features"]["align_population"], random_state=seed)
