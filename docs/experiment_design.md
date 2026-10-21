# Experiment design and pre-registration

**Research question.** Is customer-level behavioural modelling more effective than observation-level anomaly detection for identifying abnormal electricity usage?

**Registration note.** The hypotheses, metrics and decision rules below were written on 2026-10-21, before the full multi-seed experiments were run. One single-seed, reduced-size smoke test (100 customers, medium magnitude only) was run first to debug the code. Its numbers were not used to choose models, features, thresholds or decision rules. Model hyperparameters are the configuration defaults in `config/config.yaml`. None were tuned on test labels.

---

## 1. Hypotheses

| ID | Hypothesis | Primary comparison |
|---|---|---|
| **H1** | Customer-level modelling identifies persistent deviations more effectively than observation-level detection, when performance is evaluated at the customer level. | B2 (profile Isolation Forest) vs A (observation IF, relative representation, flag-rate aggregation), customer-level average precision (AP) on the six persistent scenarios. Secondary: B1 vs A. |
| **H2** | Observation-level detection identifies isolated spikes more effectively than a model based only on aggregated customer profiles. | A with max-day aggregation vs B2, customer-level AP on the two transient scenarios. Secondary: A with flag-rate aggregation vs B2. Descriptive: day-level AP / P / R / F1 of A. |
| **H3** | Combining behavioural profiles with observation-level evidence may improve prioritisation compared with either approach alone. | Hybrid (mean of calibrated B2 and A percentiles) vs B2 and vs A, customer-level AP on the mixed test set (prevalence 10%). |

**Why the aggregation differs between H1 and H2.** It was fixed in advance from the design rationale, not from results. Flag-rate measures *persistence*, so it is the natural customer-level summary for H1. The single most anomalous day (max) is the natural summary for isolated spikes. Both aggregations are reported for every scenario, so a reader can check the alternative.

## 2. Decision rules

For each hypothesis, the per-seed metric is averaged over the relevant scenarios. The two methods are then compared across seeds with a two-sided Wilcoxon signed-rank test, using paired seeds. The unit of analysis is the seed (an independent replication), never a reading or a customer-day.

* **Supported:** at medium magnitude the mean difference is > 0 and p < 0.05, *and* the difference is positive at all three magnitudes.
* **Partially supported:** significant at medium magnitude but not positive at every magnitude.
* **Not supported (no significant difference):** p ≥ 0.05.
* **Not supported (opposite difference significant):** the mean difference is < 0 and p < 0.05.

Per-scenario paired comparisons are reported with Holm-adjusted p-values within each family (`paired_comparisons.csv`). With 10 seeds the smallest attainable two-sided Wilcoxon p-value is about 0.002, so the test has limited power. Effect sizes (mean difference with a bootstrap 95% CI) are therefore reported alongside every p-value.

## 3. Data partitions (chronological, never random)

| Role | Window (default) | Used for |
|---|---|---|
| `train_history` | 2013-01-01 – 2013-03-31 | Reference history for the training window |
| `train` | 2013-04-01 – 2013-06-30 | Fitting every model (features vs `train_history`) |
| `validation` | 2013-07-01 – 2013-09-30 | Clean: calibration distributions and pipeline thresholds. Development injections: F1-optimal thresholds |
| `test` | 2013-10-01 – 2013-12-31 | Final evaluation injections only |

* Every window's features use only that window and the immediately preceding window.
* Models are fitted once per seed on clean training data. Injections never touch the training window.
* Both approaches use the **same eligible customers** (≥ 70% valid days in every window), the **same windows**, and the **same alert budgets**.

## 4. Controlled anomaly injection (Experiment 1)

Anomalies are injected into a copy of the half-hourly slot matrix, inside the evaluation window. The original matrix is kept unchanged. Magnitude *m* ∈ {low 0.2, medium 0.5, high 1.0}.

| # | Scenario | Kind | Transformation (from onset to window end unless noted) |
|---|---|---|---|
| 1 | persistent_increase | persistent | all slots × (1 + m) |
| 2 | persistent_decrease | persistent | all slots ÷ (1 + m) |
| 3 | gradual_drift | persistent | linear multiplier 1 → 1 + 2m |
| 4 | night_increase | persistent | 00:00–06:00 slots × (1 + 3m) |
| 5 | weekday_weekend_shift | persistent | weekdays × (1 + m), weekends ÷ (1 + m) |
| 6 | variability_increase | persistent | each day × exp(N(−m²/2, m)), which preserves the mean |
| 7 | isolated_spikes | transient | 2 random days × (1 + 4m) |
| 8 | intermittent_days | transient | 15% of days × (1 + 2m) |
| 9 | high_regular_control | **benign control** | history *and* window × (1 + 2m) |

* **Onset.** Persistent anomalies start at a random day within the first third of the window. Detection-delay runs start on day 1.
* **Ground truth.** Recorded for every injection: affected customer, affected days, scenario, magnitude, onset date, end date.
* **Scenario 9.** This customer is consistently high in both the history and the evaluation window, with a regular pattern. Following the brief ("high overall but otherwise regular"), it is treated as **not anomalous** by default (`label_high_regular_as_anomaly: false`). Its false-alarm rate is reported separately: the share of controls flagged, and the share of controls in the top 10%. A good detector should *not* flag it.
* **Prevalence.** Single-scenario runs use 10% of eligible customers. Mixed runs use 5% and 10%, with scenarios 1–8 assigned round-robin plus 2% benign controls.
* **Development vs test.** The development set (validation window) and the test sets (test window) use independent random draws. Thresholds come from the development set only.
* **Seeds.** 10 seeds on the synthetic background, each with a newly generated 200-household population. 5 seeds on the real-data background, with the same households but new injections and model random states.

## 5. Methods compared

| Name | Approach | Unit scored | Customer score |
|---|---|---|---|
| A_obs_relative | Observation-level IF; day features as robust z-scores vs the customer's own history (same day type) | customer-day | fraction of days above the training 98th-percentile day score (tie-break: mean day score) |
| A_obs_relative_max / _mean / _top3_mean | same detector, different aggregation | customer-day | max / mean / mean of the 3 highest day scores |
| A_obs_global | Observation-level IF on raw daily values + day-of-week | customer-day | fraction of flagged days |
| B1_peer_kmeans | K-Means peer groups (k by silhouette on training data); robust z-deviation from the peer group | customer | mean of the 3 largest abs(z) |
| B2_customer_iforest | IF on the 30-feature customer vector | customer | −score_samples |
| B2 ablations | IF on level / level+variability / temporal / profile-without-change / change-only features | customer | −score_samples |
| H_hybrid (optional) | 0.5·percentile(B2) + 0.5·percentile(A_obs_relative), with percentiles taken against the clean validation window | customer | weighted percentile |

## 6. Metrics (Experiment 2)

* **Customer level, threshold-free:** average precision (primary), ROC-AUC, R-precision, Precision@K and Recall@K at K = 5% and 10% of customers.
* **Customer level, thresholded:** precision, recall and F1 at the development-set F1-optimal threshold.
* **Observation level (transient scenarios, model A):** day-level AP, plus precision / recall / F1 at the training-quantile day threshold.
* **Detection delay:** the share of injected customers in the top-K (K = 10%) after 7, 14, 28, 56 and 91 days, for persistent scenarios (medium magnitude).
* **Score stability:** Spearman ρ and top-10% Jaccard overlap between clean validation-window and clean test-window scores.
* **Cost:** fit and score wall-clock time per model, and peak Python-traced memory.
* **Reporting:** mean, SD and t-based 95% CI across seeds for every metric.

**Why not accuracy?** At 10% prevalence, flagging nobody is 90% accurate. Accuracy is therefore not reported as a headline metric.

## 7. Experiment 3: real data (unlabelled)

* Both approaches run on the sampled London households: train on Q2 2013 and score Q4 2013.
* No ground truth exists, so **no precision, recall or detection rate is reported for real data**.
* Reported instead:
  * score distributions;
  * rank agreement between methods;
  * cluster profiles;
  * the highest-ranked customers' daily histories;
  * whether their flagged behaviour is persistent (long flagged runs, change features) or isolated (one or two flagged days).
* The semi-synthetic experiment (injections into real households) shows how the methods behave on realistic background variation. Its results are labelled separately from the purely synthetic ones.

## 8. Deviation from the pre-registration (recorded 2026-10-21)

**Number of real-background seeds: 5 → 10.**

* **The flaw.** With 5 paired runs, the exact two-sided Wilcoxon signed-rank test has a minimum attainable p-value of 2/2⁵ = 0.0625. The pre-registered rule (p < 0.05) could therefore never be met on the real background, even if all five seeds agreed. This was a design error in the registration, found when the 5-seed run finished.
* **What was preserved.** The 5-seed results are kept unchanged in `outputs/experiments/real_background_5seed_preregistered/`.
* **What was changed.** The real-background experiment was extended to seeds 0–9, matching the synthetic experiment. Seeds 0–4 are deterministic and reproduce the original runs exactly.
* **How it is reported.** Both versions appear in the report.
* **What did not change.** No model, feature, threshold rule, metric or decision rule was changed.
* **Caveat.** Extending a sample after seeing interim results is a form of optional stopping. The reader should weigh the 10-seed p-values with this in mind. Effect sizes and their direction across seeds are the more informative evidence.

## 9. Threats to validity (summary; see limitations.md)

* Injected anomalies are stylised. Detectors may do well on them and still fail on real abnormal use.
* The synthetic generator was designed by the same author as the features, which risks circularity. The real-background experiment mitigates this partly.
* Peer groups use consumption data only. No demographic or dwelling data is available.
* Default hyperparameters were not tuned. A different configuration could change the ranking of methods.
