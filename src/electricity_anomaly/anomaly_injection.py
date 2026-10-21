"""Controlled anomaly injection with ground-truth labels (Experiment 1).

Anomalies are injected into a *copy* of the slot matrix inside an evaluation
window; the original matrix is never modified. Every injection records the
affected customer, scenario, magnitude, onset/end dates and the affected days.
Labels are only used for evaluation and (on a separate development window)
threshold selection; they are never model inputs.

Magnitude ``m`` (0.2 / 0.5 / 1.0 by default) maps to each scenario as follows:

=========================  ==================================================
persistent_increase         all slots x (1 + m) from onset to window end
persistent_decrease         all slots / (1 + m) from onset to window end
gradual_drift               multiplier rises linearly from 1 to (1 + 2m)
night_increase              night slots (00:00-06:00) x (1 + 3m)
weekday_weekend_shift       weekdays x (1 + m), weekends / (1 + m)
variability_increase        each day x exp(N(-s^2/2, s)), s = m (mean-preserving)
isolated_spikes             ``spike_days`` single days x (1 + 4m)
intermittent_days           ``intermittent_day_fraction`` of days x (1 + 2m)
high_regular_control        whole history + window x (1 + 2m); benign control
=========================  ==================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

PERSISTENT = ["persistent_increase", "persistent_decrease", "gradual_drift", "night_increase",
              "weekday_weekend_shift", "variability_increase"]
TRANSIENT = ["isolated_spikes", "intermittent_days"]
CONTROL = "high_regular_control"
ANOMALY_SCENARIOS = PERSISTENT + TRANSIENT
ALL_SCENARIOS = ANOMALY_SCENARIOS + [CONTROL]


@dataclass
class InjectionResult:
    """Injected matrix plus customer-level and day-level ground truth."""

    matrix: pd.DataFrame
    customer_labels: pd.DataFrame     # one row per injected customer
    day_labels: pd.DataFrame          # (customer_id, date) of affected days

    def customer_truth(self, customers: list[str] | pd.Index) -> pd.Series:
        """Boolean anomaly label for every customer in ``customers``."""
        pos = self.customer_labels.loc[self.customer_labels["is_anomalous"], "customer_id"]
        return pd.Series(pd.Index(customers).isin(pos), index=pd.Index(customers, name="customer_id"))


def _apply(block: np.ndarray, dates: pd.DatetimeIndex, scenario: str, m: float, rng: np.random.Generator,
           cfg_exp: dict[str, Any], night_slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply one scenario to a (days x 48) block; return block and affected-day mask."""
    n = len(block)
    affected = np.ones(n, dtype=bool)
    if scenario == "persistent_increase":
        block = block * (1 + m)
    elif scenario == "persistent_decrease":
        block = block / (1 + m)
    elif scenario == "gradual_drift":
        ramp = 1 + 2 * m * (np.arange(1, n + 1) / n)
        block = block * ramp[:, None]
    elif scenario == "night_increase":
        block = block.copy()
        block[:, night_slots] *= (1 + 3 * m)
    elif scenario == "weekday_weekend_shift":
        weekend = np.asarray(dates.dayofweek >= 5)
        block = block * np.where(weekend, 1 / (1 + m), 1 + m)[:, None]
    elif scenario == "variability_increase":
        factor = np.exp(rng.normal(-m**2 / 2, m, size=n))
        block = block * factor[:, None]
    elif scenario == "isolated_spikes":
        candidates = np.flatnonzero(np.isfinite(block).any(axis=1))   # days with data
        affected = np.zeros(n, dtype=bool)
        affected[rng.choice(candidates, size=min(int(cfg_exp["spike_days"]), len(candidates)), replace=False)] = True
        block = block * np.where(affected, 1 + 4 * m, 1.0)[:, None]
    elif scenario == "intermittent_days":
        candidates = np.flatnonzero(np.isfinite(block).any(axis=1))
        k = max(2, int(round(cfg_exp["intermittent_day_fraction"] * len(candidates))))
        affected = np.zeros(n, dtype=bool)
        affected[rng.choice(candidates, size=min(k, len(candidates)), replace=False)] = True
        block = block * np.where(affected, 1 + 2 * m, 1.0)[:, None]
    elif scenario == CONTROL:
        block = block * (1 + 2 * m)
    else:
        raise ValueError(f"Unknown scenario {scenario!r}")
    return block, affected


def inject_anomalies(slot_matrix: pd.DataFrame, customers: list[str], window: tuple[pd.Timestamp, pd.Timestamp],
                     history: tuple[pd.Timestamp, pd.Timestamp], scenarios: list[str], prevalence: float,
                     magnitude_label: str, cfg: dict[str, Any], seed: int, onset: str = "random",
                     n_controls: int | None = None) -> InjectionResult:
    """Inject anomalies into a copy of ``slot_matrix``.

    Args:
        customers: Eligible customers to sample from.
        window: Evaluation window receiving the anomalies.
        history: The window's reference history (only the benign control is
            scaled there, so the customer is "consistently high").
        scenarios: Anomaly scenarios assigned round-robin to the selected
            customers. Pass ``[CONTROL]`` to inject only benign controls.
        prevalence: Fraction of ``customers`` receiving an anomaly.
        magnitude_label: Key of ``experiments.magnitudes``.
        onset: ``"random"`` (within the first ``onset_max_fraction`` of the
            window) or ``"start"`` (window start; used for detection delay).
        n_controls: Additional benign controls (mixed experiments). Defaults
            to none.
    """
    exp = cfg["experiments"]
    m = float(exp["magnitudes"][magnitude_label])
    rng = np.random.default_rng(seed)
    night_lo, night_hi = cfg["features"]["periods"]["night"]
    night_slots = np.arange(2 * night_lo, 2 * night_hi)

    pool = np.array(sorted(customers))
    n_target = max(1, int(round(prevalence * len(pool))))
    n_controls = int(n_controls or 0)
    chosen = rng.permutation(pool)[: n_target + n_controls]
    order = list(rng.permutation(scenarios))
    assignment = [(c, order[i % len(order)]) for i, c in enumerate(chosen[:n_target])]
    assignment += [(c, CONTROL) for c in chosen[n_target:]]

    matrix = slot_matrix.copy()
    values = matrix.to_numpy(copy=True)
    cust_idx = matrix.index.get_level_values("customer_id")
    dates = matrix.index.get_level_values("date")
    label_controls_positive = bool(exp["label_high_regular_as_anomaly"])
    n_window = (window[1] - window[0]).days + 1

    rows, day_rows = [], []
    for cid, scenario in assignment:
        if scenario in PERSISTENT and onset == "random":
            start = window[0] + pd.Timedelta(days=int(rng.integers(0, max(1, int(exp["onset_max_fraction"] * n_window)))))
        else:
            start = window[0]
        if scenario == CONTROL:
            start = history[0]
        mask = (cust_idx == cid) & (dates >= start) & (dates <= window[1])
        pos = np.flatnonzero(mask)
        block, affected = _apply(values[pos], dates[pos], scenario, m, rng, exp, night_slots)
        values[pos] = block
        is_anom = scenario != CONTROL or label_controls_positive
        if scenario != CONTROL:
            day_rows.append(pd.DataFrame({"customer_id": cid, "date": dates[pos][affected], "scenario": scenario}))
        rows.append(dict(customer_id=cid, scenario=scenario, magnitude_label=magnitude_label, magnitude=m,
                         onset=start.date(), end=window[1].date(), n_affected_days=int(affected.sum()),
                         is_anomalous=is_anom, is_control=scenario == CONTROL,
                         kind="transient" if scenario in TRANSIENT else ("control" if scenario == CONTROL else "persistent")))
    matrix.iloc[:, :] = values
    day_labels = (pd.concat(day_rows, ignore_index=True) if day_rows
                  else pd.DataFrame(columns=["customer_id", "date", "scenario"]))
    return InjectionResult(matrix=matrix, customer_labels=pd.DataFrame(rows), day_labels=day_labels)
