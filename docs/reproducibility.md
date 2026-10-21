# Reproducibility guide

## Environment

| Item | Setting |
|---|---|
| Tested on | Windows 11, Python **3.11.2**, CPU only, 16 GB RAM |
| Dependencies | Pinned in `requirements.txt` (numpy 2.4.2, pandas 2.3.3, scikit-learn 1.9.0, streamlit 1.54.0, inflate64 1.0.3, …) |
| Random seeds | `config.yaml → random_seed` (pipeline, default 42) and `experiments.seeds` (0–9 synthetic) / `experiments.real_seeds` (0–4) |
| Configuration hash | Every output records `config_hash` (SHA-256 of the full configuration), so a result can be matched to the settings that produced it |

**What the seeds control:**

* synthetic population generation;
* anomaly injection;
* model `random_state`;
* the real-household sample (hash seed);
* the final 600-household cap.

## Step-by-step (PowerShell, from the project root)

```powershell
# 1. Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 2. Tests (55 tests, no real data needed; about 1-3 minutes)
python -m pytest -q

# 3. Synthetic data, pipeline, experiments
python scripts/generate_synthetic_data.py
python scripts/run_pipeline.py --data synthetic
python scripts/run_experiments.py --data synthetic

# 4. Real data (764.5 MB download; streaming the full file takes about 12 minutes once)
python scripts/download_data.py
python scripts/run_pipeline.py --data real
python scripts/run_experiments.py --data real

# 5. Report and dashboard
python scripts/build_report.py
streamlit run app/streamlit_app.py
```

**If PowerShell blocks `Activate.ps1`.** Run `Set-ExecutionPolicy -Scope Process RemoteSigned` in that window, or call `.\.venv\Scripts\python.exe` directly.

## Approximate run times (this laptop)

| Step | Time |
|---|---|
| Generate synthetic CSV (200 households, 3.4M rows) | ~45 s |
| Synthetic pipeline | ~0.5–1.5 min |
| Synthetic experiments, 10 seeds × (6 mixed + 27 single-scenario + 6 delay×5 checkpoints) test sets | ~15–25 min |
| Stream the real zip and build the Parquet cache (one-off) | ~12 min |
| Real pipeline (600 households) | ~1–3 min |
| Real-background experiments, 10 seeds (600 households) | ~30 min |
| `pytest` (55 tests) | ~1–3 min |

**Faster runs.** `--quick` (medium magnitude only) and `--seeds 0 1` give a quick check.

## Where results are written

| Path | Content |
|---|---|
| `data/processed/real_clean_halfhourly.parquet` | Cleaned sampled real readings (+ `.meta.json` with counts and the cache key) |
| `data/features/<data>/customer_features_*.parquet` | Customer feature matrices per window |
| `outputs/pipeline/<synthetic or real>/` | `customer_results.csv`, `preprocessing_stats.json`, `run_metadata.json`, cluster tables, all model scores, stability, agreement, resources |
| `outputs/experiments/<synthetic or real_background>/` | `runs_*.csv` (one row per seed × test set × model), `summary_*.csv` (mean, SD, 95% CI), `paired_comparisons.csv`, `hypotheses.csv` |
| `outputs/figures/` | All PNG figures, regenerated from the files above |
| `outputs/reports/project_report.md` | Report with result tables filled from the files above |

## Verifying the UTC timestamp assumption

```powershell
python -c "import sys; sys.path.insert(0,'src'); import pandas as pd; from pathlib import Path; from electricity_anomaly.data_loader import iter_csv_chunks; c=next(iter_csv_chunks(Path('data/raw/LCL-FullData.zip'),3000000)); t=pd.to_datetime(c['DateTime'],format='ISO8601'); print(c[t.dt.normalize()=='2013-03-31'].groupby('LCLid').size().value_counts())"
```

**Expected result.** All households show 48 readings. A local-time file would show 46.

## Determinism

**What is fixed by seed.** Re-running with the same configuration gives identical scores for:

* the synthetic generator;
* injection;
* model fitting;
* sampling.

`tests/test_models.py::test_fit_suite_deterministic_and_consistent_direction` checks this.

**What varies.** Timing and memory figures change between runs and machines.
