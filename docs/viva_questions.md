# Likely viva questions and answers

Answers refer to the code and the saved results. Numbers quoted from the experiments are in `outputs/reports/project_report.md`. Re-check them if the experiments are re-run.

## A. Problem and design

**1. What exactly is the difference between observation-level and customer-level detection here?**

* **Observation level (Approach A).** Each customer-day is a separate data point, and Isolation Forest scores it. Customer scores come *afterwards*, by aggregating day flags; the default is the fraction of flagged days.
* **Customer level (B1, B2).** Each customer is one data point described by a 30-feature behavioural profile (level, variability, timing, and change against their own history). The model scores the profile directly.

**2. Why not just use supervised learning?**

* There are no verified anomaly or theft labels for the London data.
* In practice, inspection labels are scarce and biased towards customers who were already suspected (Glauner et al., 2017).
* Unsupervised scores need no labels. Controlled injection then gives a labelled *test*, without training on labels.

**3. Is a high score evidence of theft?**

No. A high score means the consumption is unusual relative to the customer's history or peers. Legitimate causes include vacancy, a new electric vehicle or heater, illness, a new baby, a meter fault, or seasonal electric heating. On the real data the top customers included:

* a meter reporting near-zero all year;
* a household whose usage stepped up in winter, consistent with electric heating.

**4. Why Isolation Forest?**

* It needs no distance matrix, so it is linear in data size and has no O(n²) pairwise cost, which suits a 16 GB laptop.
* It handles mixed-scale features after scaling.
* It gives a continuous score.
* It is standard and explainable: anomalies are isolated in fewer random splits.

**5. Why K-Means for peers, and what are its weaknesses?**

* It is the standard load-profiling baseline. It is fast and gives explicit centroids that can be described as "typical profiles".
* Weaknesses:
  * it assumes spherical clusters of similar size;
  * it is sensitive to outliers (mitigated by clipping the scaled features to ±5);
  * k must be chosen.
* Silhouette picked k = 2 on both datasets. The k-sensitivity table shows how much B1 rankings change for k = 3–8.
* DBSCAN or k-medoids would be the next alternatives to try.

## B. Data and preprocessing

**6. How did you handle the 10 GB file on a 16 GB laptop?**

* The CSV is streamed from the zip in 2M-row chunks, with a custom Deflate64 decompressor, because Python's `zipfile` cannot read method 9.
* Only rows from a deterministic 25% household hash sample are kept, flat tariff, 2013 only.
* Customer IDs are stored as categoricals.
* The result is cached as Parquet. Peak memory stays well under 4 GB.

**7. How do you know the timestamps are UTC?**

On the spring DST day (2013-03-31), every household has 48 readings per calendar date. A local-time record would have only 46.

**8. What about the 49 readings on 2013-10-27?**

These are exact duplicates of the 00:00 reading. They recur for almost every household about every 31 days, so this is a systematic artefact of the published file, not a DST effect. Exact deduplication removes them and counts them.

**9. How are missing values different from zeros?**

* `Null`, non-numeric, negative and >12 kWh readings become *missing slots* (NaN), and each type is counted.
* A recorded 0 is kept as valid consumption.
* Daily totals need ≥ 90% of slots; otherwise the day is missing.
* Gaps are never filled.

**10. Why not remove outliers?**

Extreme but physically plausible readings may be exactly the anomalies under study. Only physically impossible values (> 24 kW sustained) are treated as invalid.

**11. How is DST handled?**

* UTC timestamps are converted to Europe/London, and each reading is placed in a local clock slot.
* The spring day has 46 slots.
* On the autumn day, the repeated hour's two readings are averaged into one slot.
* Daily totals are 24-hour equivalents, so DST days are comparable with ordinary days.

## C. Features and leakage

**12. Give an example feature and its formula.**

`level_change_log = ln((mean daily kWh in window + 0.1) / (mean daily kWh in previous window + 0.1))`.

* 0 means no change.
* +0.69 means doubled.
* −0.69 means halved.

**13. How do you prevent data leakage?**

* **Chronological windows.** Train is Q2 (history Q1); validation is Q3; test is Q4.
* **Features look back only.** A window's features use only that window and the window before it. The code raises an error otherwise, and a test checks that changing future data leaves the features unchanged.
* **Models.** Every model is fitted once on clean training data.
* **Thresholds.** These come from a separate development injection in Q3. A test checks they are computed before any Q4 injection.
* **Labels.** Injected labels are never used as features.

**14. What is "population alignment" and why is it needed?**

* London consumption is much higher in Q4 than in Q2. Without alignment, every customer would look anomalous in winter.
* For each window (customer features) or each date (day features), the cross-sectional median of every feature is shifted to the training median.
* This removes shifts common to everyone but keeps differences between customers.
* Limitation: it would also hide an anomaly affecting most customers at once.

## D. Experiments and statistics

**15. Why inject anomalies? Isn't that circular?**

* Without labels, injection is the only way to measure detection performance with known ground truth.
* It is partly circular, because the same person designed the features and the anomalies. This is mitigated by:
  * injecting into *real* households (the semi-synthetic background);
  * including transient anomalies and benign controls;
  * pre-registering the hypotheses.
* The limitation is stated explicitly.

**16. Why is the benign control (scenario 9) not labelled as an anomaly?**

* That customer is *consistently* high, in both history and evaluation window, with a regular pattern. That is not a change in behaviour.
* The test checks whether a detector merely flags big consumers.
* A detector that flags such controls would generate false alarms on large but legitimate households.

**17. Why average precision rather than accuracy?**

* Prevalence is 5–10%, so accuracy rewards flagging no one.
* AP summarises the precision-recall curve and focuses on the rare class.
* Precision@K matches an operational budget of "inspect the top K customers".

**18. How did you choose thresholds fairly?**

* For every model, the F1-optimal threshold is chosen on a *development* set: a separate injection into the validation quarter.
* That threshold is then applied unchanged to the test quarter.
* Ranking metrics (AP, P@K) need no threshold at all.

**19. Why the Wilcoxon test, and what is the unit of analysis?**

* Each seed is an independent replication: a new population or injection, and new model seeds. Readings and customers within a seed are *not* independent, so they are not used as samples.
* The paired Wilcoxon signed-rank test compares two methods across seeds. It assumes no normality, only paired, independent replications with a roughly symmetric difference distribution (Demšar, 2006).
* With 10 seeds, the minimum p-value is about 0.002, so effect sizes with bootstrap CIs are also reported.

**20. What is Holm correction and why use it?**

* Several scenarios are tested, which inflates the false-positive risk.
* Holm–Bonferroni orders the p-values and multiplies the i-th smallest by (m − i + 1), keeping the running maximum. This controls the family-wise error rate with more power than plain Bonferroni.

## E. Results and interpretation

**21. So — is customer-level modelling better?**

Partly, and the answer depends on which customer-level method and which anomaly type.

* **Supported (H1).** Peer-deviation modelling (B1) beat observation-level detection (A) on persistent anomalies, on every seed:
  * synthetic: +0.11 average precision;
  * real background: +0.03.

  The gain came almost entirely from **gradual drift** and **changed weekday/weekend routine**. These change the shape or trend of behaviour without producing extreme days.
* **Not better.**
  * For night-time increases and variability increases, A, using customer-relative day features, was just as good.
  * For plain ±50% level shifts, every method was weak, because natural Q3→Q4 changes are of a similar size.
  * The full-feature profile Isolation Forest (B2) was *worse* than A on synthetic data (−0.07 AP), and only marginally better on the real background (+0.01).
* **Not supported (H2).** For isolated spikes, neither approach was better at the customer level.
* **One-sentence answer:** customer-level modelling is more effective for persistent *pattern* changes when it compares customers with relevant references, but it is not uniformly superior.

**22. Why can full-feature Isolation Forest (B2) miss a large level change?**

Two properties, both found while writing unit tests:

1. **Saturation.** IF can only split within the training range, so a value 10× beyond the range scores the same as one just beyond it.
2. **Dilution.** With 30 features, of which 7 are correlated level features, a change in a few features is diluted, because splits pick features at random.

Example: quadrupling a *low* user's consumption made their level features look typical while their change features saturated, and B2 slightly *lowered* the score. The peer-deviation model (B1), which takes the largest z-deviations, and the change-only IF ablation ranked that customer highly.

**23. What do the ablations tell you?**

* An Isolation Forest on the 8 **change features alone** beat the full 30-feature B2 on persistent anomalies:
  * synthetic: 0.66 vs 0.49;
  * real background: 0.22 vs 0.19.
* Temporal-shape features mainly helped with the weekday/weekend shift.
* Level-only profiles were weakest, because they cannot see within-customer change.
* Among A's aggregations, the *mean* day score did best on persistent anomalies. This was an exploratory, not pre-registered, finding.
* Overall, *what* a customer is compared against (their own history, their peers) matters more than the algorithm.

**24. How stable are the rankings over time?**

* Profile-based scores (B2, B1) are much more stable between quarters than the flag-rate of Approach A.
* For normal customers, A's flag rate is near zero and mostly noise.
* This matters operationally: a stable ranking produces consistent investigation lists.

**25. What would you do next?**

* Add weather and tariff data to model heating-driven seasonality per customer.
* Include time-of-use households, with tariff-aware features.
* Try robust or density-based peers (k-medoids, DBSCAN) and feature selection to reduce IF dilution.
* Use rolling windows for earlier detection.
* Evaluate against real inspection outcomes if they become available, with fairness checks across household types.
