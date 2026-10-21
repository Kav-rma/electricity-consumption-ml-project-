"""Generate the synthetic half-hourly dataset (CSV in the London file's schema).

Example:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --customers 300 --seed 7
"""

import argparse
import logging

from _bootstrap import setup_logging

from electricity_anomaly.config import load_config, resolve_path, with_overrides
from electricity_anomaly.synthetic import write_synthetic_csv

logger = logging.getLogger("generate_synthetic_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--customers", type=int, default=None, help="Number of households (default: config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: config random_seed)")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.customers:
        cfg = with_overrides(cfg, {"synthetic.n_customers": args.customers})
    seed = cfg["random_seed"] if args.seed is None else args.seed
    path = write_synthetic_csv(cfg, resolve_path(cfg, "synthetic_dir"), seed)
    logger.info("Synthetic dataset written to %s", path)


if __name__ == "__main__":
    main()
