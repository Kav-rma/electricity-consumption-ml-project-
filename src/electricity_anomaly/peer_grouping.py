"""Peer groups from consumption behaviour (K-Means) and within-group deviation.

Peer groups are defined from *profile* features only (level, variability,
temporal shape) because they describe what is typical for a customer; change
features are excluded from clustering. When scoring a window, a customer is
assigned to a peer group using their profile in the *preceding* window, so a
customer whose behaviour changes is compared against the peers they used to
resemble rather than being silently re-assigned to a group that fits the new
behaviour.

No demographic metadata is used: peers are "customers who consumed
similarly", which is a limitation discussed in docs/limitations.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from .feature_engineering import ALL_FEATURES, LOG_FEATURES, PROFILE_FEATURES
from .scoring import RobustFeatureScaler


def select_k(X: np.ndarray, k_min: int, k_max: int, n_init: int, seeds: list[int]) -> pd.DataFrame:
    """Compare candidate cluster counts on *training* profiles.

    For each k: mean silhouette over seeds, inertia, and stability measured as
    the mean adjusted Rand index between partitions from different seeds.
    Silhouette favours compact, well-separated spherical clusters and can
    prefer small k; it is a heuristic, not proof of a "true" k.
    """
    rows = []
    n = len(X)
    # Multi-threaded K-Means is ~10x slower than one thread on a few hundred rows.
    with threadpool_limits(1):
        for k in range(k_min, min(k_max, n - 1) + 1):
            labels, sils, inertias = [], [], []
            for s in seeds:
                km = KMeans(n_clusters=k, n_init=n_init, random_state=s).fit(X)
                labels.append(km.labels_)
                inertias.append(km.inertia_)
                sils.append(silhouette_score(X, km.labels_) if len(set(km.labels_)) > 1 else np.nan)
            aris = [adjusted_rand_score(labels[i], labels[j]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
            rows.append(dict(k=k, silhouette=float(np.nanmean(sils)), silhouette_std=float(np.nanstd(sils)),
                             inertia=float(np.mean(inertias)), seed_stability_ari=float(np.mean(aris)) if aris else 1.0,
                             min_cluster_size=int(min(np.bincount(labels[0])))))
    return pd.DataFrame(rows)


class PeerGroupModel:
    """B1: K-Means peer groups with robust within-group deviation scoring.

    Score for a customer in cluster c:
        z_j = (x_j - median_c(x_j)) / max(1.4826 * MAD_c(x_j), 0.25 * IQR_j / 1.349)
        score = mean of the ``top_k`` largest |z_j| (each |z_j| capped at 20)
    where x are log-transformed, population-aligned features, medians and MADs
    come from training customers in cluster c, and the floor (a quarter of the
    population robust standard deviation) prevents a tiny within-cluster
    spread from inflating z-scores. Clusters with fewer than
    ``min_cluster_size`` training members use population statistics.
    """

    def __init__(self, k: int | None = None, k_min: int = 2, k_max: int = 8, n_init: int = 10, clip: float = 5.0,
                 top_k: int = 3, align: bool = True, random_state: int = 0, stability_seeds: int = 5,
                 deviation_features: list[str] | None = None, min_cluster_size: int = 5):
        self.k = k
        self.k_min, self.k_max, self.n_init = k_min, k_max, n_init
        self.clip, self.top_k, self.align = clip, top_k, align
        self.random_state = random_state
        self.stability_seeds = stability_seeds
        self.deviation_features = deviation_features or ALL_FEATURES
        self.min_cluster_size = min_cluster_size

    def fit(self, train_features: pd.DataFrame) -> "PeerGroupModel":
        self.cluster_scaler_ = RobustFeatureScaler(PROFILE_FEATURES, LOG_FEATURES, align=self.align, clip=self.clip).fit(train_features)
        Xc = self.cluster_scaler_.transform(train_features).to_numpy()
        seeds = [self.random_state + i for i in range(self.stability_seeds)]
        self.k_selection_ = select_k(Xc, self.k_min, self.k_max, self.n_init, seeds)
        if self.k is None:
            self.k_ = int(self.k_selection_.loc[self.k_selection_["silhouette"].idxmax(), "k"])
        else:
            self.k_ = int(self.k)
        with threadpool_limits(1):
            self.kmeans_ = KMeans(n_clusters=self.k_, n_init=self.n_init, random_state=self.random_state).fit(Xc)
        self.train_clusters_ = pd.Series(self.kmeans_.labels_, index=train_features.index, name="behavioural_cluster")

        self.dev_scaler_ = RobustFeatureScaler(self.deviation_features, LOG_FEATURES, align=self.align).fit(train_features)
        U = self.dev_scaler_.unscaled(train_features)
        pop_floor = 0.25 * self.dev_scaler_.scale_ / 1.349
        self.floor_ = pop_floor.where(pop_floor > 1e-6, 1e-3)
        pop_med = U.median()
        pop_mad = 1.4826 * (U - pop_med).abs().median()
        self.cluster_median_, self.cluster_scale_ = {}, {}
        for c in range(self.k_):
            members = U.loc[self.train_clusters_ == c]
            if len(members) >= self.min_cluster_size:
                med = members.median()
                mad = 1.4826 * (members - med).abs().median()
            else:
                med, mad = pop_med, pop_mad
            self.cluster_median_[c] = med
            self.cluster_scale_[c] = np.maximum(mad, self.floor_)
        self.train_scores_ = self.score(train_features, self.train_clusters_)
        return self

    def assign(self, profile_features: pd.DataFrame) -> pd.Series:
        """Nearest-centroid peer group for each customer from profile features."""
        Xc = self.cluster_scaler_.transform(profile_features).to_numpy()
        return pd.Series(self.kmeans_.predict(Xc), index=profile_features.index, name="behavioural_cluster")

    def deviations(self, features: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
        """Robust z-score of every deviation feature relative to the peer group."""
        U = self.dev_scaler_.unscaled(features)
        clusters = clusters.reindex(U.index)
        if clusters.isna().any():
            raise ValueError("Some customers have no peer-group assignment")
        med = pd.DataFrame([self.cluster_median_[int(c)] for c in clusters], index=U.index)
        scale = pd.DataFrame([self.cluster_scale_[int(c)] for c in clusters], index=U.index)
        return ((U - med) / scale).clip(-20, 20)

    def score(self, features: pd.DataFrame, clusters: pd.Series) -> pd.Series:
        """Peer deviation score (higher = more unusual relative to peers)."""
        Z = self.deviations(features, clusters).abs().to_numpy()
        k = min(self.top_k, Z.shape[1])
        top = -np.sort(-Z, axis=1)[:, :k]
        return pd.Series(top.mean(axis=1), index=features.index, name="peer_deviation_score")

    def principal_deviations(self, features: pd.DataFrame, clusters: pd.Series, n: int = 3) -> pd.Series:
        """Text summary of the ``n`` features deviating most from the peer group."""
        Z = self.deviations(features, clusters)
        out = {}
        for cid, row in Z.iterrows():
            top = row.abs().nlargest(n).index
            out[cid] = "; ".join(f"{f} ({row[f]:+.1f} peer-MAD)" for f in top)
        return pd.Series(out, name="principal_behavioural_deviations")

    def centroid_distance(self, profile_features: pd.DataFrame, clusters: pd.Series) -> pd.Series:
        """Euclidean distance to the assigned centroid in clustering space (diagnostic)."""
        Xc = self.cluster_scaler_.transform(profile_features).to_numpy()
        cent = self.kmeans_.cluster_centers_[clusters.reindex(profile_features.index).to_numpy().astype(int)]
        return pd.Series(np.linalg.norm(Xc - cent, axis=1), index=profile_features.index, name="centroid_distance")


def cluster_profiles(features: pd.DataFrame, clusters: pd.Series, load_profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median feature values and mean 48-slot load profile for each cluster."""
    summary_cols = ["mean_daily_kwh", "cv_daily", "night_share", "morning_share", "afternoon_share",
                    "evening_share", "peak_to_average", "baseload_ratio", "weekend_weekday_ratio"]
    f = features.join(clusters)
    summary = f.groupby("behavioural_cluster")[summary_cols].median()
    summary.insert(0, "n_customers", f.groupby("behavioural_cluster").size())
    prof = load_profiles.join(clusters).groupby("behavioural_cluster").mean()
    return summary, prof


def build_peer_model(cfg: dict[str, Any], seed: int, deviation_features: list[str] | None = None,
                     k: int | None = None) -> PeerGroupModel:
    """Construct the peer-group model from configuration."""
    km = cfg["customer_models"]["kmeans"]
    return PeerGroupModel(k=k, k_min=km["k_min"], k_max=km["k_max"], n_init=km["n_init"], clip=km["clip_value"],
                          top_k=km["top_k_deviations"], align=cfg["features"]["align_population"],
                          random_state=seed, stability_seeds=km["stability_seeds"],
                          deviation_features=deviation_features)
