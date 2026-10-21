# Final presentation outline (15–20 minutes + live demo)

1. **Title and question** (1 min)
   * Project title, and the research question stated verbatim.
   * The one-line answer: *it depends on the anomaly type*. Defer the details to the results slides.

2. **Motivation** (1.5 min)
   * Smart meters give 17,520 readings per household per year.
   * Abnormal use (faults, vacancy, non-technical losses) matters, but labels are scarce.
   * Two ways to look for it: unusual *readings* or unusual *customers*.

3. **Dataset** (1.5 min)
   * London Datastore, Low Carbon London: 5,567 households, 168M rows, CC BY 4.0.
   * What inspection revealed:
     * Deflate64 compression;
     * UTC timestamps (shown by the DST-day check);
     * systematic duplicate midnight readings;
     * `Null` tokens.
   * Sample: 600 flat-tariff households in 2013. Synthetic generator for controlled tests.
   * *Figure:* `outputs/figures/pipeline_real/preprocessing_exclusions.png`

4. **Pipeline** (2 min)
   * Streaming → cleaning (counted, never silent) → local-time slot matrix → daily totals → chronological windows Q1–Q4.
   * Leakage controls: features look back only; thresholds come from a development set.

5. **Two approaches** (2.5 min)
   * A: Isolation Forest on days, then flag rate per customer.
   * B1: K-Means peer groups with robust z-deviation. B2: Isolation Forest on 30-feature profiles.
   * Feature groups: level, variability, temporal, change.
   * *Figure:* cluster load profiles.

6. **Experimental design** (2 min)
   * 8 scenarios + benign control, 3 magnitudes, 10 + 5 seeds.
   * Pre-registered H1–H3 and decision rules.
   * Metrics: AP and Precision@K, plus F1 at development thresholds; Wilcoxon test across seeds.

7. **Results: by scenario** (3 min)
   * *Figure:* `outputs/figures/experiments_synthetic/ap_by_scenario.png`, and its real-background version.
   * Hypothesis table, with verdicts read directly from `hypotheses.csv`.

8. **Results: why** (2 min)
   * Ablations: which feature groups matter.
   * Isolation Forest saturation and dilution (the unit-test finding).
   * Detection delay and score stability.

9. **Real data, unlabelled** (2 min)
   * Top-ranked households: a near-zero meter, step changes, winter heating.
   * Persistent vs isolated flags; agreement between methods.
   * *Figure:* `outputs/figures/pipeline_real/top_customers_daily.png`.
   * Emphasise: signals for investigation, not accusations.

10. **Live demo** (3 min): `streamlit run app/streamlit_app.py`
    * Overview → leaderboard (switch rankings) → customer explorer for a top customer ("why flagged") → model comparison tab.

11. **Limitations and ethics** (1 min)
    * Stylised injections; no labels; peers defined by consumption only; seasonal heating; privacy; human review.

12. **Conclusion and future work** (1 min)
    * Answer the RQ only to the extent the results support.
    * Next steps: weather data, ToU households, robust peers, rolling detection.

**Backup slides:**

* k-selection and sensitivity;
* threshold selection procedure;
* feature formulas;
* runtime and memory;
* test-suite coverage.
