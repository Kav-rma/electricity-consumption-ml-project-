"""Configuration loading and validation.

The configuration is a plain nested dictionary read from ``config/config.yaml``.
``load_config`` validates the values the rest of the code relies on so that a
typo fails early with a clear message instead of deep inside a model.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

WINDOW_ORDER = ("train_history", "train", "validation", "test")


class ConfigError(ValueError):
    """Raised when the configuration file contains invalid values."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the YAML configuration.

    Args:
        path: Path to a YAML file. Defaults to ``config/config.yaml``.

    Returns:
        The validated configuration dictionary.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Configuration file {path} is empty or not a mapping")
    validate_config(cfg)
    return cfg


def _require(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"Missing configuration key: {dotted}")
        node = node[key]
    return node


def _check_fraction(cfg: dict, dotted: str, allow_zero: bool = True) -> None:
    value = _require(cfg, dotted)
    low_ok = value >= 0 if allow_zero else value > 0
    if not isinstance(value, (int, float)) or not low_ok or value > 1:
        raise ConfigError(f"{dotted} must be a fraction in [0, 1], got {value!r}")


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate configuration values; raises ``ConfigError`` on problems."""
    for key in ("paths", "schema", "synthetic", "real", "preprocessing", "windows",
                "features", "observation_model", "customer_models", "experiments"):
        _require(cfg, key)

    if _require(cfg, "preprocessing.interval_minutes") != 30:
        raise ConfigError("Only 30-minute intervals are supported (48 slots per day)")
    if _require(cfg, "preprocessing.max_kwh_per_interval") <= 0:
        raise ConfigError("preprocessing.max_kwh_per_interval must be positive")
    for frac in ("preprocessing.min_day_coverage", "preprocessing.min_window_day_coverage",
                 "real.household_sample_fraction", "customer_models.alert_rate",
                 "observation_model.observation_contamination",
                 "experiments.scenario_prevalence", "customer_models.hybrid_weight"):
        _check_fraction(cfg, frac)

    windows = get_windows(cfg)
    previous_end = None
    for name in WINDOW_ORDER:
        start, end = windows[name]
        if end < start:
            raise ConfigError(f"Window {name} ends before it starts")
        if previous_end is not None and start <= previous_end:
            raise ConfigError(f"Window {name} overlaps the previous window; windows must be chronological")
        previous_end = end

    rep = _require(cfg, "observation_model.representation")
    if rep not in ("relative", "global"):
        raise ConfigError("observation_model.representation must be 'relative' or 'global'")
    agg = _require(cfg, "observation_model.aggregation")
    if agg not in ("flag_rate", "max", "mean", "top3_mean"):
        raise ConfigError("observation_model.aggregation must be flag_rate, max, mean or top3_mean")

    km = _require(cfg, "customer_models.kmeans")
    if not 2 <= km["k_min"] <= km["k_max"]:
        raise ConfigError("customer_models.kmeans requires 2 <= k_min <= k_max")

    for period, bounds in _require(cfg, "features.periods").items():
        if len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] <= 24:
            raise ConfigError(f"features.periods.{period} must be [start_hour, end_hour] within 0-24")

    if _require(cfg, "synthetic.n_customers") < 10:
        raise ConfigError("synthetic.n_customers must be at least 10")


def get_windows(cfg: dict[str, Any]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the chronological windows as (start, end) timestamps (inclusive)."""
    raw = _require(cfg, "windows")
    out = {}
    for name in WINDOW_ORDER:
        if name not in raw:
            raise ConfigError(f"windows.{name} is missing")
        start, end = raw[name]
        out[name] = (pd.Timestamp(start), pd.Timestamp(end))
    return out


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a ``paths.<key>`` entry against the project root."""
    path = Path(_require(cfg, f"paths.{key}"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_path(relative: str | Path) -> Path:
    """Resolve an arbitrary configured path against the project root."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


def config_hash(cfg: dict[str, Any]) -> str:
    """Short, stable hash of the configuration, stored with every output."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def with_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a validated deep copy of ``cfg`` with dotted-key overrides applied."""
    new = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = new
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    validate_config(new)
    return new
