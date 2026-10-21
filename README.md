# Electricity Consumer Behaviour Anomaly Detection

An unsupervised machine-learning framework that investigates one research question:

> **Is customer-level behavioural modelling more effective than observation-level anomaly detection for identifying abnormal electricity usage?**

The project compares two families of detectors on half-hourly smart-meter data:

* **Approach A – observation level.** An Isolation Forest scores each customer-day. Day flags are then aggregated per customer: frequency, persistence and severity.
* **Approach B – customer level.** Each customer gets a 30-feature behavioural profile covering level, variability, daily/weekly shape, and change against their own history. Profiles are scored by:
  * **B1** – K-Means peer groups with robust within-group deviation;
  * **B2** – an Isolation Forest on the profiles.

  An optional **hybrid** combines B2 and A.

The comparison is made through **controlled anomaly injection**:

* 8 anomaly scenarios plus a benign control;
* 3 magnitudes;
* 10 seeds on synthetic households and 10 seeds on real London households (5 pre-registered, extended to 10; see `docs/experiment_design.md` §8);
* chronological train/validation/test windows;
* thresholds chosen on a separate development set;
* paired Wilcoxon tests across seeds.

The real London data (no labels) is then analysed qualitatively.

> **An anomaly score is a signal for investigation, not evidence of theft, fraud or wrongdoing.** No accuracy or fraud-detection figures are reported for the real data, because it has no verified labels.

**Results** are in [`outputs/reports/project_report.md`](outputs/reports/project_report.md), which is generated from saved outputs. Each run also writes its own summary tables (see *Outputs* below).

### Key findings (runs of 2026-10-21; numbers from the generated report)

* **H1, persistent changes.** Customer-level **peer-deviation (B1) beat observation-level A** on every seed:
  * synthetic: +0.11 average precision (AP);
  * real-household background: +0.03 AP.

  The gain came mainly from gradual drift and changed weekday/weekend routines. The **profile Isolation Forest (B2) was worse than A on synthetic data** (−0.07 AP) and only marginally better on real data (+0.01). Its weakness traces to Isolation Forest saturation and to feature dilution.
* **H2, isolated spikes.** **Not supported.** At the customer level, neither approach was better. Only A localises anomalous *days*, and it does so with high recall but low precision.
* **H3, hybrid.** **Partially supported.** The hybrid beat the weaker component on each background, never both.
* **Trade-offs.** A detects persistent changes earliest. Customer-level scores are far more stable between quarters, but flag more legitimately high consumers.
* **Real data, unlabelled.** The top-ranked households show persistent shifts (near-zero meters, step changes, winter heating), not isolated spikes. These are signals for review only.
* **Overall.** Customer-level modelling helps for persistent *pattern* changes when it compares customers with relevant references. It is not uniformly superior.

---

## Repository structure

```
├── README.md                 this file
├── requirements.txt          pinned dependencies (Python 3.11)
├── pyproject.toml            package metadata (pip install -e . optional)
├── .env.example              optional environment variables (none required)
├── config/config.yaml        every path, window, feature setting, model parameter and seed
├── data/                     raw / processed / features / synthetic (git-ignored contents)
├── notebooks/exploratory_analysis.ipynb
├── src/electricity_anomaly/
│   ├── config.py             load + validate config
│   ├── data_loader.py        schema detection, streaming CSV/zip (Deflate64), hash sampling, download, cache
│   ├── preprocessing.py      cleaning, duplicates, DST, slot matrix, daily totals, coverage
│   ├── synthetic.py          synthetic households with realistic defects
│   ├── feature_engineering.py  30 customer features + daily observation features
│   ├── observation_models.py   Approach A + customer aggregation
│   ├── peer_grouping.py      Approach B1 (K-Means peers, k selection, deviations)
│   ├── customer_models.py    Approach B2 (profile Isolation Forest + occlusion attribution)
│   ├── scoring.py            scaling/alignment, calibration, thresholds, hybrid
│   ├── anomaly_injection.py  controlled anomalies with ground truth
│   ├── evaluation.py         metrics, CIs, paired tests, Holm, stability
│   ├── experiments.py        Experiments 1-2 runner + hypothesis decisions
│   ├── pipeline.py           shared fit/score functions + end-to-end pipeline
│   ├── visualization.py      figures from saved outputs
│   └── reporting.py          fills the report template from saved outputs
├── app/streamlit_app.py      dashboard
├── scripts/                  download_data, generate_synthetic_data, run_pipeline, run_experiments, build_report
├── tests/                    pytest suite (no real data needed)
├── outputs/                  pipeline/, experiments/, figures/, reports/
└── docs/                     methodology, experiment_design (pre-registration), limitations,
                              reproducibility, references, presentation_outline, viva_questions, report_template
```

The project root is this folder. The brief's suggested `electricity-anomaly-detection/` layout is used inside it.

---

## Quick start (Windows PowerShell)

```powershell
# 1. Create and activate a virtual environment (Python 3.11)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 2. Synthetic data -> pipeline -> controlled experiments
python scripts/generate_synthetic_data.py
python scripts/run_pipeline.py --data synthetic
python scripts/run_experiments.py --data synthetic          # ~15-25 min; add --quick --seeds 0 1 for a fast check

# 3. Real London data (764.5 MB download; one-off ~12 min stream to a Parquet cache)
python scripts/download_data.py
python scripts/run_pipeline.py --data real
python scripts/run_experiments.py --data real               # ~30 min; anomalies injected into real households

# 4. Report and dashboard
python scripts/build_report.py
streamlit run app/streamlit_app.py

# 5. Tests
python -m pytest -q
```

**If activation is blocked:** run `Set-ExecutionPolicy -Scope Process RemoteSigned`, or call `.\.venv\Scripts\python.exe` directly.

### Script options

| Script | Options |
|---|---|
| `generate_synthetic_data.py` | `--customers N`, `--seed S`, `--config PATH` |
| `run_pipeline.py` | `--data {synthetic,real}` (required), `--no-demo-injection`, `--no-cache`, `--no-figures`, `--config PATH` |
| `run_experiments.py` | `--data {synthetic,real}` (required), `--seeds 0 1 ...`, `--quick`, `--no-figures`, `--config PATH` |
| `download_data.py` | `--skip-download`, `--no-prepare`, `--rebuild`, `--config PATH` |
| `build_report.py` | `--config PATH` |

---

## Data

### Real data

*SmartMeter Energy Consumption Data in London Households* (UK Power Networks / Low Carbon London, London Datastore, CC BY 4.0): 5,567 households, Nov 2011 – Feb 2014, kWh per half hour.

* **Source:** the official zip URL is in `config.yaml → real.download_url`.
* **Manual alternative:** download from the landing page and set `real.local_path` to the zip, an extracted CSV, or a folder of CSVs (for example the 168-file release).
* **Streaming:** the file is read in 2M-row chunks without extraction.
* **Sampling:**
  * a deterministic 25% household hash sample;
  * flat-tariff (`Std`) households only;
  * calendar year 2013;
  * households need ≥ 70% valid days in every quarter;
  * a seeded cap of 600 households.
* **Counts:** every stage is recorded in `outputs/pipeline/real/preprocessing_stats.json`.

### Synthetic data

* 200 households from five behavioural archetypes, with seasonality, weather and away periods.
* Data defects with known counts: missing values, outages, late starts, duplicates, negative or implausible values, malformed timestamps.
* Written in the same raw schema as the London file.
* Synthetic results and real results are always stored in separate folders.

---

## Outputs

| Path | Content |
|---|---|
| `outputs/pipeline/<synthetic or real>/customer_results.csv` | one row per customer: `customer_id`, `observation_count`, `mean_daily_kwh`, `behavioural_cluster`, `observation_anomaly_count`, `observation_anomaly_rate`, `customer_anomaly_score`, `customer_anomaly_flag`, `peer_deviation_score`, `principal_behavioural_deviations`, `model_version`, `evaluation_period`, … |
| `outputs/pipeline/<…>/` | preprocessing stats, coverage, cluster summaries / load profiles, k selection and sensitivity, all model scores, stability, model agreement, resources, run metadata |
| `outputs/experiments/<synthetic or real_background>/` | `runs_*.csv` (raw per-seed rows), `summary_*.csv` (mean, SD, 95% CI), `paired_comparisons.csv`, `hypotheses.csv` |
| `outputs/figures/` | PNG figures generated from the files above |
| `outputs/reports/project_report.md` | full academic report |

## Dashboard

Run `streamlit run app/streamlit_app.py`. The dashboard has six tabs:

| Tab | Content |
|---|---|
| Overview | dataset and preprocessing statistics; flagged counts; score distributions |
| Customer explorer | daily history with flagged days; peer reference profile; features vs peer median; "why flagged" |
| Anomaly leaderboard | separate A / B1 / B2 / hybrid rankings with filters, score definitions and thresholds |
| Model comparison | labelled experiment metrics, hypotheses, delay, stability, run time |
| Behavioural profiles | clusters and k selection |
| Settings & runs | editable parameters (saved to `config/dashboard_config.yaml`) and a run button |

The dashboard only reads saved files. It never retrains models when a filter changes, and it starts without errors when outputs are missing.

## Tests

`python -m pytest -q` runs 55 tests in about 1–3 minutes. No download is needed. The suite covers:

* schema detection, CSV/zip/Deflate64/folder reading and hash sampling;
* missing, negative, implausible and zero readings;
* exact and conflicting duplicates;
* irregular and malformed timestamps;
* DST spring and autumn days;
* coverage eligibility;
* known feature values, constant and zero histories, look-ahead prevention, scaling and alignment;
* model determinism and score direction;
* day-to-customer aggregation;
* leakage-free threshold ordering;
* every injection scenario and its labels;
* metric calculations, paired tests and Holm correction;
* end-to-end pipeline and experiments;
* dashboard start-up with and without outputs.

## Documentation

| Document | Content |
|---|---|
| [docs/methodology.md](docs/methodology.md) | every feature (formula, units, meaning), preprocessing rule and model |
| [docs/experiment_design.md](docs/experiment_design.md) | pre-registered hypotheses, decision rules, scenarios, metrics |
| [docs/limitations.md](docs/limitations.md) | limitations and ethics |
| [docs/reproducibility.md](docs/reproducibility.md) | environment, seeds, run times, where files go |
| [docs/references.md](docs/references.md) | verified references |
| [docs/presentation_outline.md](docs/presentation_outline.md), [docs/viva_questions.md](docs/viva_questions.md) | demonstration and viva preparation |
