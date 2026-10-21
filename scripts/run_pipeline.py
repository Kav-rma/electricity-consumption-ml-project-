"""Run preprocessing, feature engineering, model fitting and scoring.

Examples:
    python scripts/run_pipeline.py --data synthetic
    python scripts/run_pipeline.py --data real
    python scripts/run_pipeline.py --data synthetic --no-demo-injection

Outputs go to outputs/pipeline/<data>/ and outputs/figures/pipeline_<data>/.
Synthetic and real outputs are always kept in separate folders.
"""

import argparse
import json

from _bootstrap import setup_logging

from electricity_anomaly.config import load_config
from electricity_anomaly.data_loader import DataUnavailableError
from electricity_anomaly.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=["synthetic", "real"], required=True)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--no-demo-injection", action="store_true",
                        help="Synthetic only: do not inject labelled demo anomalies into the test window")
    parser.add_argument("--no-cache", action="store_true", help="Real only: re-read the raw file")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    try:
        meta = run_pipeline(cfg, args.data, inject_demo=not args.no_demo_injection, use_cache=not args.no_cache,
                            make_figures=not args.no_figures)
    except (DataUnavailableError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(json.dumps({k: meta[k] for k in ("data_source", "n_scored_customers", "selected_k", "n_flagged",
                                           "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
