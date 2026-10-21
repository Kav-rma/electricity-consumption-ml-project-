# Methodology

This document defines every processing step, feature and model in this project, so that the results can be reproduced and explained. Code references are given as `module.function`.

---

## 1. Data

### 1.1 Real dataset

**Source.** *SmartMeter Energy Consumption Data in London Households*, published by UK Power Networks on the London Datastore (Low Carbon London project), licensed CC BY 4.0. It covers 5,567 households from November 2011 to February 2014, with readings of kWh per half hour.

**File used.** `LCL-FullData.zip` (764.5 MB) from the official download link (see `config.yaml → real.download_url`). It contains one CSV, `CC_LCL-FullData.csv` (8.54 GB, 167,932,474 rows).

**Schema (verified by inspecting the file).**

| Column | Content |
|---|---|
| `LCLid` | Household identifier |
| `stdorToU` | Tariff: `Std` (flat) or `ToU` (dynamic time-of-use in 2013) |
| `DateTime` | Timestamp, e.g. `2012-10-12 00:30:00.0000000` |
| `KWH/hh (per half hour) ` | Consumption. The header has a trailing space, and values have trailing spaces |

**Missing values** are encoded as the text `Null`.

**Compression.** The zip uses **Deflate64** (method 9), which Python's `zipfile` cannot read. `data_loader._Deflate64Stream` decompresses it as a stream using `inflate64`, so the 8.5 GB CSV is never extracted or held in memory. A manually extracted CSV, or the 168-file partitioned release, also works (`real.local_path` can be a zip, a CSV or a folder).

**Timestamps are UTC/GMT.** This was established empirically. On the spring DST day (2013-03-31), households have **48** readings per calendar date. A local-time record would have only 46 (`docs/reproducibility.md` shows the check). `schema.source_timezone: UTC`.

**Duplicate midnight readings.** The real file contains a *systematic* exact duplicate of the 00:00 reading for almost every household, roughly every 31 days (2013-01-21, 02-21, 03-24, 04-24, …, 10-27). It shows up as 49 readings on those calendar dates. Exact deduplication removes these rows (12,643 in the sample) and counts them in `exact_duplicate_rows_removed`.

**Sampling.** Households are sampled deterministically while streaming:

* A household is kept if `crc32("seed:LCLid") / 2³² < 0.25`. The sample is therefore reproducible and independent of file order.
* Only `Std` tariff households are kept. In 2013 the ToU households received price signals designed to change their behaviour, which would confound "abnormal" with "responding to the trial".
* Rows from 2013 only are kept (plus one-day margins for time-zone conversion).
* After the coverage filter (§1.4), at most `real.max_households = 600` eligible households are retained by seeded random choice.
* Counts at every stage are written to `outputs/pipeline/real/preprocessing_stats.json`.

### 1.2 Synthetic dataset (`synthetic.py`)

The generator writes the **same raw schema**, so ingestion and cleaning are exercised identically.

**Behaviour model.** Each household belongs to one of five archetypes:

| Archetype | Share |
|---|---|
| commuter | 35% |
| home-daytime | 25% |
| early-retired | 20% |
| night-storage heating | 8% |
| low user | 12% |

Each archetype has its own 48-slot weekday and weekend load shape, built from Gaussian bumps plus baseload. Daily kWh is:

```
level × (1 + seasonal_amplitude · cos(2π(doy − 15)/365))
      × weekend factor
      × shared weather shock (AR(1))
      × household AR(1) noise
```

Each reading also gets multiplicative Gamma noise (shape 6). 30% of households have a 7–14 day away period at 20–40% of normal consumption. This is a *benign* irregularity.

**UTC sampling.** Readings are generated on the local clock and written in UTC, so DST days have 46 and 50 readings, exactly as a UTC-stamped meter would record them.

**Data defects, with known counts** (`synthetic_defect_log.json`):

* isolated missing readings (`Null`);
* multi-day outages (10% of households);
* late-starting households (5%, which become ineligible);
* exact and conflicting duplicates;
* negative and physically implausible values;
* malformed timestamps.

### 1.3 Cleaning (`preprocessing.clean_raw_chunk`, `deduplicate`)

| Check | Handling | Recorded as |
|---|---|---|
| Missing customer ID | row dropped | `rows_missing_customer_id` |
| Unparseable timestamp | row dropped | `rows_malformed_timestamp` |
| Timestamp not on the 30-min grid | row dropped | `rows_irregular_timestamp` |
| `Null` / empty reading | becomes a missing slot | `rows_missing_reading` |
| Non-numeric reading | missing slot | `rows_non_numeric_reading` |
| Negative reading | missing slot (meter error; never set to zero) | `rows_negative_reading` |
| Reading > 12 kWh per half hour (24 kW sustained, above a ~23 kW 100 A single-phase supply) | missing slot | `rows_implausible_reading` |
| Zero reading | **kept as valid** | `zero_readings_kept` |
| Exact duplicate (customer, timestamp, value) | one copy kept | `exact_duplicate_rows_removed` |
| Conflicting duplicate (same customer and timestamp, different values) | replaced by their mean | `conflicting_duplicate_rows` |

**Extreme values.** Extreme but *plausible* readings are **not removed**, because they may be the anomalies under study.

**Duplicate key.** Duplicates are defined on the *source* (UTC) timestamp. The repeated local hour on the autumn DST day is therefore not mistaken for a duplicate.

### 1.4 Aggregation and coverage (`build_slot_matrix`, `daily_from_matrix`, `coverage_summary`)

* **Local time.** Timestamps are converted to Europe/London local time. Each reading is assigned to a (local date, slot 0–47) pair.
* **Slot matrix.** The canonical processed table is the *slot matrix*: one row per (customer, date) for **every calendar date** in the study period, with 48 kWh columns. NaN means *not measured*; nothing is imputed.
* **DST days.**
  * On the spring day, slots 2–3 (01:00–02:00) do not exist, so the day has 46 observed slots.
  * On the autumn day, the two readings for 01:00–02:00 are averaged into their slots. This is counted in `local_slots_with_repeated_readings`.
* **Valid day.** At least 90% of the 48 slots are observed (`min_day_coverage`).
* **Daily consumption.** Defined as the 24-hour-equivalent value, `mean(observed slots) × 48`. This rescales DST days by 24/23 and 24/25, and partially observed valid days by their coverage. Days below 90% coverage are missing. Long gaps are never filled.
* **Eligible customer.** At least 70% valid days in *each* of the four windows (`min_window_day_coverage`). Both approaches use exactly this customer set.

### 1.5 Chronological windows (`config.yaml → windows`)

| Window | Dates | Role |
|---|---|---|
| train_history | 2013-01-01 – 03-31 | History for the training window |
| train | 2013-04-01 – 06-30 | Model fitting |
| validation | 2013-07-01 – 09-30 | Calibration and threshold selection |
| test | 2013-10-01 – 12-31 | Evaluation |

* The "history" of a window is the window immediately before it.
* No feature of a window uses data after the window's end. This is enforced in code: `customer_features` raises an error if the history window does not precede the observation window, and a unit test checks that changing future readings leaves the features unchanged.

---

## 2. Customer-level features (`feature_engineering.customer_features`)

### 2.1 Notation

| Symbol | Meaning |
|---|---|
| *W* | the observation window |
| *H* | the history window |
| D | the set of valid days in *W* |
| *d_t* | daily kWh on day *t* |
| *p_s* | mean kWh in local slot *s* over D (the mean load profile) |
| med, IQR, MAD | median, inter-quartile range, median absolute deviation |
| ε | 0.1 kWh/day, added inside logs of daily totals |

### 2.2 Safe-division rule

Every ratio uses `safe_divide`. A zero or non-finite denominator returns a stated neutral fallback, given in brackets in the tables below.

### 2.3 Consumption level (kWh/day)

| Feature | Formula | Interpretation |
|---|---|---|
| mean_daily_kwh | mean(*d_t*) | typical level |
| median_daily_kwh | med(*d_t*) | robust level |
| std_daily_kwh | SD(*d_t*), ddof = 1 (0 if one day) | absolute spread |
| iqr_daily_kwh | Q75 − Q25 | robust spread |
| mad_daily_kwh | med(abs(*d_t* − med *d*)) | robust spread |
| p95_daily_kwh | 95th percentile of *d_t* | high-end consumption |
| max_daily_kwh | max(*d_t*) over valid days (≥ 90% coverage only) | peak day |

### 2.4 Variability and regularity (dimensionless)

| Feature | Formula | Interpretation |
|---|---|---|
| cv_daily | std / mean [0] | relative variability |
| mean_abs_change_rel | mean(abs(*d_t* − *d_{t−1}*)) over consecutive valid calendar days / mean [0] | day-to-day volatility |
| high_day_freq | share of days with *d_t* > 1.5 × med(*d*) | frequency of unusually high days |
| low_day_freq | share of days with *d_t* < 0.5 × med(*d*) | frequency of unusually low days (e.g. absence) |
| profile_stability | mean Pearson correlation between each day's 48-slot profile and the window mean profile. Missing slots in a day are filled from the mean profile for this calculation only. A constant day is undefined and skipped [1.0 if none defined] | regularity of the daily routine |

### 2.5 Temporal behaviour (local time)

| Feature | Formula | Interpretation |
|---|---|---|
| night_share | Σ*p_s* over 00:00–06:00 / Σ*p_s* [12/48] | night-time consumption ratio |
| morning_share | Σ*p_s* over 06:00–12:00 / Σ*p_s* [12/48] | morning share |
| afternoon_share | Σ*p_s* over 12:00–18:00 / Σ*p_s* [12/48] | afternoon share |
| evening_share | Σ*p_s* over 18:00–24:00 / Σ*p_s* [12/48] | evening share |
| peak_to_average | max *p_s* / mean *p_s* [1] | peakiness of the daily shape |
| baseload_ratio | min *p_s* / mean *p_s* [1] | always-on load relative to average |
| weekend_weekday_ratio | mean weekend *d_t* / mean weekday *d_t* [1], clipped to [0, 10] | weekly routine |
| dow_cv | SD / mean of the seven day-of-week means [0] | day-of-week pattern strength |
| peak_hour_sin, peak_hour_cos | sin, cos(2π h*/24), where h* is the hour of the max *p_s* | time of the daily peak (circular encoding) |

The full 48-slot mean profile is stored separately (`load_profiles.parquet`) for visualisation and the peer reference. It is not used directly as model input, because 48 correlated columns would dominate the distance computations.

### 2.6 Long-term change (relative to the customer's own history *H*)

| Feature | Formula | Interpretation |
|---|---|---|
| level_change_log | ln((mean_W + ε)/(mean_H + ε)) | persistent level change |
| peak_change_log | ln((p95_W + ε)/(p95_H + ε)) | change in high-end use |
| variability_change | cv_W − cv_H | change in variability |
| night_share_change | night_share_W − night_share_H | change in night-time use |
| weekend_ratio_change_log | ln(ratio_W / ratio_H), ratios clipped ≥ 0.001 | changed weekly routine |
| profile_shape_distance | 1 − corr(*p*^W, *p*^H) [0] | change in daily shape |
| trend_slope_rel | OLS slope of *d_t* on day index × 30 / mean_W [0] | within-window drift, as a fraction of mean per 30 days |
| half_split_change_log | ln((mean of 2nd half + ε)/(mean of 1st half + ε)) | change within the window |

**Not computed: demographic features.** These are not available (the ACORN group exists only in other releases of the data), and none are fabricated. Customer IDs are never features.

**Validation.** If any feature is non-finite after the safe-division rules, the code raises `FloatingPointError` instead of continuing silently.

---

## 3. Observation-level (daily) features (`observation_features`)

### 3.1 Raw ("global") day features

| Feature | Definition |
|---|---|
| log_kwh | ln(*d_t* + ε) |
| night_share_day | the day's 00:00–06:00 share |
| log_peak_ratio | ln(max slot / mean slot) |
| day_change | log_kwh_t − log_kwh_{t−1} (0 if the previous day is missing) |
| dow_sin, dow_cos | day-of-week encoding |

### 3.2 Relative day features (default representation)

The first four features are expressed as robust z-scores against the same customer's history, for the same day type (weekday or weekend):

```
z = (x − med_H) / max(1.4826 · MAD_H, floor)
```

The floors are 0.05, 0.01, 0.05 and 0.05 respectively.

* `is_weekend` is added as a feature.
* Each day is still scored **independently**. The history only provides a scale, so this remains an observation-level method.

---

## 4. Population alignment (both approaches)

Seasonal and weather effects shift every customer together. For example, Q4 consumption is higher than Q2. A model trained in Q2 would otherwise see every Q4 customer as anomalous.

| Approach | Alignment |
|---|---|
| **Customer features** | After the log transform, each scored batch is shifted so its per-feature median equals the training median (`RobustFeatureScaler`, `align=True`) |
| **Day features** | The cross-sectional median of each continuous feature on each date is replaced by the training median (≥ 20 customers per date required) |

**Limitation.** This removes population-wide changes, which is the intent, but it would also hide an anomaly affecting most customers at once. A per-feature median is robust as long as fewer than half of the customers are anomalous.

---

## 5. Models

### 5.1 Approach A: observation-level Isolation Forest (`observation_models.py`)

**Training.**

* `IsolationForest(n_estimators=200, max_samples=256)` is fitted on training-window days (relative or global representation).
* Features are robust-scaled using the training median and IQR.
* The customer ID is not a feature.

**Day score and threshold.**

* Day score = −`score_samples` (higher = more anomalous).
* A day is flagged when its score exceeds the 98th percentile of training-day scores (`observation_contamination = 0.02`).

**Aggregation to customers** (`aggregate_to_customers`):

| Output | Meaning |
|---|---|
| observation_count | number of valid days |
| observation_anomaly_count | number of flagged days (frequency) |
| observation_anomaly_rate | flagged days / valid days |
| obs_longest_flag_run | longest run of consecutive flagged calendar days (persistence) |
| obs_severity | mean exceedance over the threshold on flagged days |
| obs_max_score, obs_mean_score, obs_top3_mean_score | max, mean and top-3 mean of day scores |

**Customer score.**

* **Default: flag rate.** Ties are broken by the mean day score, added at a 10⁻³ scale, which is below one day's increment of the rate. A single flagged day in a 92-day window therefore gives a rate of only 0.011. One reading alone cannot make a customer rank highly unless no other customer has flagged days.
* **Ablation variants:** max-day, mean and top-3 mean.

### 5.2 Approach B1: K-Means peer groups (`peer_grouping.PeerGroupModel`)

**Clustering features.** Only the 22 profile features (level, variability and temporal groups) are used: log-transformed, aligned, robust-scaled and clipped to ±5 so outliers do not drag centroids.

**Choosing k.**

* k ∈ {2,…,8} is evaluated on **training** profiles only. For each k, five K-Means runs (`n_init=10`, k-means++) are made with different seeds.
* The criteria are mean silhouette, inertia, and seed stability (mean pairwise adjusted Rand index).
* The k with the highest silhouette is chosen.
* Silhouette is a heuristic that favours compact, spherical and often few clusters. Sensitivity to k and to initialisation is reported in `k_sensitivity.csv`: Spearman correlation of the B1 scores, and top-10% overlap, against the selected model.

**Peer assignment.** When a window is scored, each customer is assigned to the nearest centroid using their profile from the **preceding window**. A customer whose behaviour changes is therefore compared with their former peers, rather than silently moved into a group that fits the new behaviour.

**Deviation score.** For each of the 30 features *j*:

```
z_j = (x_j − med_c(x_j)) / max(1.4826 · MAD_c(x_j), 0.25 · IQR_j/1.349)
```

* x is the log-transformed, aligned value.
* med_c and MAD_c come from training customers in cluster *c*.
* The floor is a quarter of the population robust SD. It stops tiny within-cluster spreads from inflating z.
* Clusters with fewer than 5 training members use population statistics.

`peer_deviation_score` is the mean of the 3 largest abs(z_j), each capped at 20. `principal_behavioural_deviations` lists those three features with their signed z.

### 5.3 Approach B2: Isolation Forest on customer profiles (`customer_models.py`)

**Model.** `IsolationForest(n_estimators=200, max_samples="auto")` is fitted on the 30-feature training-window vectors: log-transformed, aligned, median-imputed and robust-scaled.

**Score.** The score is −`score_samples`. The built-in binary `predict` is not used.

**Explanation (occlusion attribution).** For each feature, the score drop is measured when that feature is reset to its training median. The top 3 positive contributions are reported (`iforest_top_features`). This ignores interactions.

**Known property, found in the unit tests.** The IF score of a value beyond the training range is no higher than that of the most extreme training value. In addition, a deviation confined to a few of 30 features is diluted, because split features are drawn at random. The feature-group ablations (B2_level_only, …, B2_change_only) quantify how much this matters.

### 5.4 Optional hybrid (H3)

```
hybrid = 0.5 · F_B2(score_B2) + 0.5 · F_A(score_A)
```

F is the empirical CDF of the model's scores on the clean validation window. Both primary approaches remain independently evaluable.

---

## 6. Score direction, normalisation and thresholds

**Direction.** For every model, a higher raw score means greater anomaly risk. The tests assert this for B1 and the change-feature IF on a customer whose consumption is multiplied by four.

**Output-table scores.** `customer_anomaly_score` and `observation_level_score` are percentiles against the clean validation-window scores (0–1). This makes them comparable across models.

**Limitations of the percentiles:**

* They depend on the validation population.
* They are capped at 1.0, so many top customers tie.
* They are not probabilities of wrongdoing.

**Pipeline flags (unlabelled).** The threshold is the raw score at the top `alert_rate = 10%` of the clean validation window's scores.

**Experiment thresholds.** The threshold is the F1-optimal threshold on a *development* injection in the validation window. A unit test checks that thresholds are computed before any test-window injection.

---

## 7. Output schema (`customer_results.csv`)

| Column | Meaning |
|---|---|
| customer_id | household identifier (index) |
| observation_count | valid days in the evaluation window |
| mean_daily_kwh | mean kWh/day in the evaluation window |
| behavioural_cluster | peer group, assigned from the previous window |
| observation_anomaly_count / rate | flagged days and their share (Approach A, relative) |
| observation_longest_flag_run_days, observation_severity | persistence and severity of A's flags |
| observation_level_score / flag | A percentile and flag |
| customer_anomaly_score / flag | B2 percentile and flag |
| customer_anomaly_raw_score | raw B2 score |
| peer_deviation_score / flag | B1 score and flag |
| hybrid_score / flag | hybrid percentile and flag |
| principal_behavioural_deviations | top-3 peer z-deviations |
| iforest_top_features | top-3 occlusion contributions |
| model_version | package version + config hash |
| evaluation_period | start/end of the evaluation window |
| data_source | `synthetic` or `real` |
| injected_scenario | synthetic demo only |
