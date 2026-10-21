"""Shared fixtures: a small, fast configuration and tiny hand-built datasets."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from electricity_anomaly.config import load_config, with_overrides  # noqa: E402
from electricity_anomaly.preprocessing import SLOT_COLUMNS  # noqa: E402

SMALL_WINDOWS = {
    "windows.train_history": ["2013-01-07", "2013-02-03"],
    "windows.train": ["2013-02-04", "2013-03-03"],
    "windows.validation": ["2013-03-04", "2013-03-31"],
    "windows.test": ["2013-04-01", "2013-04-28"],
}


@pytest.fixture(scope="session")
def base_cfg():
    return load_config()


@pytest.fixture()
def small_cfg(base_cfg, tmp_path):
    """Four 28-day windows, 40 customers, small forests; outputs in a temp dir."""
    return with_overrides(base_cfg, {
        **SMALL_WINDOWS,
        "synthetic.n_customers": 40,
        "synthetic.start_date": "2013-01-01",
        "synthetic.end_date": "2013-04-30",
        "experiments.synthetic_customers": 40,
        "experiments.seeds": [0],
        "experiments.delay_checkpoints_days": [7, 14, 28],
        "experiments.bootstrap_resamples": 200,
        "observation_model.n_estimators": 50,
        "customer_models.isolation_forest.n_estimators": 50,
        "customer_models.kmeans.k_max": 4,
        "customer_models.kmeans.stability_seeds": 2,
        "customer_models.kmeans.n_init": 3,
        "paths.outputs_dir": str(tmp_path / "outputs"),
        "paths.synthetic_dir": str(tmp_path / "synthetic"),
        "paths.features_dir": str(tmp_path / "features"),
        "paths.processed_dir": str(tmp_path / "processed"),
        "real.local_path": str(tmp_path / "raw" / "missing.zip"),
    })


def make_matrix(daily_by_customer: dict[str, np.ndarray], start: str = "2013-01-07",
                shape: np.ndarray | None = None) -> pd.DataFrame:
    """Slot matrix from per-customer daily totals and a 48-slot shape (default flat)."""
    shape = np.full(48, 1 / 48) if shape is None else shape / shape.sum()
    frames = []
    for cid, daily in daily_by_customer.items():
        dates = pd.date_range(start, periods=len(daily), freq="D")
        idx = pd.MultiIndex.from_product([[cid], dates], names=["customer_id", "date"])
        frames.append(pd.DataFrame(np.outer(daily, shape), index=idx, columns=SLOT_COLUMNS))
    return pd.concat(frames).astype("float32")


def raw_frame(rows):
    """Raw string-typed frame in the London schema."""
    return pd.DataFrame(rows, columns=["LCLid", "stdorToU", "DateTime", "KWH/hh (per half hour) "])
