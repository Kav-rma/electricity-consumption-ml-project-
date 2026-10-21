"""Evaluation metrics and statistical comparison utilities.

Accuracy is deliberately not used as a headline metric: with 5-10% anomalous
customers a detector that flags nobody is 90-95% "accurate". Precision,
recall, F1, Precision@K and average precision focus on the rare class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(y_true, y_pred) -> dict[str, float]:
    """Precision, recall and F1 for boolean predictions (0 when undefined)."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "n_flagged": int(y_pred.sum())}


def precision_recall_at_k(y_true, scores, k: int) -> tuple[float, float]:
    """Precision@K and Recall@K for the K highest scores (stable tie order)."""
    y_true = np.asarray(y_true, dtype=bool)
    k = int(min(max(k, 1), len(y_true)))
    order = np.argsort(-np.asarray(scores, dtype="float64"), kind="mergesort")[:k]
    hits = int(y_true[order].sum())
    n_pos = int(y_true.sum())
    return hits / k, (hits / n_pos if n_pos else 0.0)


def ranking_metrics(y_true, scores, budgets: list[float]) -> dict[str, float]:
    """Threshold-free metrics for one detector.

    Returns average precision, ROC-AUC, R-precision (Precision@K with K equal
    to the number of positives) and Precision/Recall@K for each alert budget
    (fraction of customers investigated).
    """
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype="float64")
    out: dict[str, float] = {"n": int(len(y_true)), "n_pos": int(y_true.sum())}
    if 0 < y_true.sum() < len(y_true):
        out["average_precision"] = float(average_precision_score(y_true, scores))
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["r_precision"] = precision_recall_at_k(y_true, scores, int(y_true.sum()))[0]
    else:
        out["average_precision"] = out["roc_auc"] = out["r_precision"] = np.nan
    for b in budgets:
        k = max(1, int(round(b * len(y_true))))
        p, r = precision_recall_at_k(y_true, scores, k)
        out[f"precision_at_{int(round(b * 100))}pct"] = p
        out[f"recall_at_{int(round(b * 100))}pct"] = r
    out["prevalence"] = float(y_true.mean()) if len(y_true) else np.nan
    return out


def mean_ci(values, confidence: float = 0.95) -> dict[str, float]:
    """Mean, standard deviation and t-based confidence interval of repeated runs."""
    v = np.asarray(pd.Series(values).dropna(), dtype="float64")
    n = len(v)
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_runs": 0}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "std": sd, "ci_low": mean - half, "ci_high": mean + half, "n_runs": n}


def summarise_runs(runs: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    """Mean / std / 95% CI of each metric across seeds for each group."""
    rows = []
    for key, grp in runs.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for metric in metric_cols:
            if metric not in grp:
                continue
            rows.append({**dict(zip(group_cols, key)), "metric": metric, **mean_ci(grp[metric])})
    return pd.DataFrame(rows)


def paired_comparison(a, b, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Paired comparison of two detectors across matched runs (seeds).

    The unit of analysis is the experimental run, not individual readings or
    customers, so observations are independent replications. Reports the mean
    difference (a - b), a percentile bootstrap 95% CI, the number of runs
    where a > b, and a two-sided Wilcoxon signed-rank test (assumes paired,
    independent runs and a difference distribution symmetric about its
    median; no normality assumption). With all differences zero, p = 1.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = len(d)
    out = {"n_pairs": n, "mean_diff": float(d.mean()) if n else np.nan,
           "wins_a": int((d > 0).sum()), "wins_b": int((d < 0).sum()), "ties": int((d == 0).sum())}
    if n == 0:
        out.update(ci_low=np.nan, ci_high=np.nan, wilcoxon_stat=np.nan, p_value=np.nan)
        return out
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    out["ci_low"], out["ci_high"] = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    if np.all(d == 0) or n < 2:
        out["wilcoxon_stat"], out["p_value"] = np.nan, 1.0
    else:
        res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        out["wilcoxon_stat"], out["p_value"] = float(res.statistic), float(res.pvalue)
    return out


def holm_adjust(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(p_values, dtype="float64")
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out


def score_stability(scores_a: pd.Series, scores_b: pd.Series, budget: float) -> dict[str, float]:
    """Agreement of one detector's scores across two windows.

    Spearman rank correlation over common customers and Jaccard overlap of the
    top ``budget`` fraction in each window.
    """
    idx = scores_a.index.intersection(scores_b.index)
    a, b = scores_a.loc[idx], scores_b.loc[idx]
    rho = float(stats.spearmanr(a, b).statistic) if len(idx) > 2 else np.nan
    k = max(1, int(round(budget * len(idx))))
    top_a = set(a.nlargest(k).index)
    top_b = set(b.nlargest(k).index)
    return {"spearman": rho, "top_jaccard": len(top_a & top_b) / len(top_a | top_b), "n_customers": len(idx)}
