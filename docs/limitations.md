# Limitations and ethical considerations

## 1. What an "anomaly" means here

* **A flag is not an accusation.** Every score in this project measures how *unusual* a customer's consumption is relative to their own history, their peers or the population. It is a signal for investigation, not evidence of electricity theft, fraud or any wrongdoing.
* **Legitimate causes.** Many reasons for unusual consumption are legitimate, for example:
  * a new baby, illness or a home-bound occupant;
  * a new electric vehicle, heat pump or storage heater;
  * holidays or a second home;
  * a change of occupants;
  * a faulty appliance;
  * metering faults.
* **No real fraud figures.** The real London data has **no verified labels**. No precision, recall, accuracy or "fraud-detection rate" is reported for it. Real-data results are descriptive: rankings, agreement between methods, and whether flagged behaviour is persistent or isolated.

## 2. Synthetic anomaly injection

* **Stylised anomalies.** The injected anomalies are clean multiplicative transformations (see `experiment_design.md`). Real abnormal behaviour can be subtler: partial meter bypass during certain hours, tampering that mimics a neighbour's profile, or slow changes over years. Good performance on injected anomalies is necessary but not sufficient evidence of usefulness.
* **Possible circularity.** The same author designed the features, the synthetic generator and the anomaly types. Features may match the injected changes better than they would match real ones. The *real-background* experiment injects anomalies into real households to reduce this risk, but the anomalies themselves are still stylised.
* **Deviation is not undesirability.** The experiments measure detection of *deviations from a customer's normal behaviour*. A customer who has always consumed abnormally (for example, a long-standing bypass) has no change to detect. The project treats a consistently high but regular customer as **benign** by design, so this case is deliberately not detected.
* **Magnitude and prevalence choices.** Magnitudes (×1.2 to ×3 depending on scenario) and prevalence (5–10%) are assumptions. Real prevalence of abnormal use is unknown and probably lower, which would reduce precision for every method.

## 3. Data limitations

* **Coverage.**
  * Only 2013 is analysed, so all four windows fall within one year.
  * Only households with ≥ 70% valid days in every quarter are included. Customers with patchy data, who may be the most interesting, are excluded.
  * The real sample is 600 of the 1,012 eligible flat-tariff households in a 25% hash sample. Results may differ on the full population.
* **ToU households excluded.** Their 2013 behaviour was shaped by the pricing trial and would be confused with anomalies. Including them is a natural extension.
* **No weather or metadata.** Seasonal effects are removed only by cross-sectional median alignment (§4 of methodology). Household-specific weather sensitivity (for example, electric heating) is not modelled, which produces benign false positives in winter.
* **Zero readings.** These are kept as valid. Long runs of zeros in the real data may be metering problems rather than vacancy, and they are not separately flagged.
* **Timestamp assumption.** Treating the timestamps as UTC is justified empirically (48 readings on DST days) but not documented by the publisher.

## 4. Method limitations

* **Isolation Forest saturation and dilution** (found while writing unit tests):
  * A value far beyond the training range scores no higher than the most extreme training value.
  * With 30 features, of which 7 are correlated level features, a deviation confined to a few change features is diluted, because the forest picks split features at random.
  * In one test case, quadrupling a *low-consuming* customer's usage *lowered* their B2 score: their level features became typical while their change features saturated. The peer-deviation model (B1) and the change-feature ablation ranked the same customer highly.
  * Feature design and selection matter as much as the choice of algorithm.
* **Zero-inflated features.**
  * Features such as `low_day_freq` are zero for most customers, so their peer-group robust spread is tiny. A few legitimately low days (for example Christmas absences in Q4) then give a saturated z-score of 20.
  * On the real data, `low_day_freq` appeared in the top-3 deviations of 71 flagged customers.
  * A larger floor, or a count-based model for such features, would reduce these benign alerts.
* **Calibration drift between quarters.**
  * Pipeline thresholds were set to flag 10% of customers in Q3. In Q4 they flagged about 15–18%, because residual seasonal effects remain after median alignment: heterogeneous heating, holidays.
  * Fixed-budget ranking (Precision@K) is more robust to this than fixed thresholds.
* **Statistical power.** As pre-registered, five real-background seeds made p < 0.05 unattainable for the exact Wilcoxon test. This is corrected by extending to ten seeds (see `experiment_design.md` §8).
* **Post-hoc observations.** These are not hypothesis tests. For example, the *mean*-day aggregation of Approach A performed best on persistent synthetic anomalies, but it was not a pre-registered primary method. It is reported as exploratory, and would need confirmation on fresh data.
* **K-Means assumptions.**
  * K-Means assumes roughly spherical, similar-sized clusters and is sensitive to outliers.
  * Clipping the scaled features to ±5 reduces outlier influence, but does not make the method robust.
  * Silhouette often prefers k = 2. On the synthetic data it separated night-storage heating households from everyone else, which is a coarse peer definition.
  * Density-based (DBSCAN) or robust alternatives (k-medoids, Gaussian mixtures with a noise component) may suit heavy-tailed real data better. They were not evaluated.
* **Peer groups from consumption alone.**
  * Peers are "customers who consumed similarly in the previous window", not households of similar size, dwelling or demographics.
  * This choice determines who is flagged. For example, a single household with electric heating is unusual among gas-heated peers, but normal within an electric-heating peer group.
  * Peer assignment from the *previous* window means a customer who changed behaviour gradually across windows may already have moved to new peers.
* **Aggregation rules.** The choice of rule for turning day scores into a customer score (flag rate vs max-day) changes which customers rank highest. Both were pre-specified, and all variants are reported.
* **Thresholds.**
  * Development-set thresholds come from injected anomalies with the same scenario mix as the test set. This is optimistic compared with a real deployment, where the anomaly mix is unknown.
  * Unsupervised pipeline thresholds (top 10% of validation scores) fix the alert volume rather than any error rate.
* **Statistical power.**
  * Ten seeds give a minimum two-sided Wilcoxon p-value of about 0.002.
  * Each seed is one replication. Customers within a seed are not independent units, and the analysis does not treat them as such.
  * Small differences may be real but undetectable, so effect sizes with confidence intervals are reported throughout.
* **Hyperparameters.** No hyperparameter tuning was done, to avoid optimistic bias and complexity. Other settings (contamination, number of trees, k, top-k deviations) could change the ranking of methods.

## 5. Ethical considerations

* **Human review.** Scores should support, not replace, human review. Any operational use would need a human-in-the-loop process, an appeal route, and checks that flags are not concentrated on particular groups (for example, low-income, elderly or medically dependent households, whose usage may be atypical for legitimate reasons).
* **Privacy.** Half-hourly smart-meter data can reveal occupancy, routines and lifestyle. This project:
  * uses only the published, pseudonymised dataset under its CC BY 4.0 licence;
  * attempts no re-identification;
  * keeps raw and processed data out of version control.
* **Proportionality.** Low prevalence means that even a precise model produces many false alarms in absolute terms. Contacting customers on the basis of a flag alone would be inappropriate.
