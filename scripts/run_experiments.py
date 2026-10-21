"""Run the controlled anomaly-injection experiments (Experiments 1 and 2).

Examples:
    python scripts/run_experiments.py --data synthetic            # 10 seeds (config)
    python scripts/run_experiments.py --data synthetic --quick --seeds 0 1
    python scripts/run_experiments.py --data real                 # injections into real data

--data real injects synthetic anomalies into the real London households
(semi-synthetic); real data itself has no anomaly labels. Results go to
outputs/experiments/<synthetic|real_background>/.
"""

import argparse

from _bootstrap import setup_logging

from electricity_anomaly.config import load_config
from electricity_anomaly.data_loader import DataUnavailableError
from electricity_anomaly.experiments import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=["synthetic", "real"], required=True)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override the configured seeds")
    parser.add_argument("--quick", action="store_true", help="Medium magnitude only and fewer delay checkpoints")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    try:
        out = run_experiments(cfg, args.data, seeds=args.seeds, quick=args.quick, make_figures=not args.no_figures)
    except DataUnavailableError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
