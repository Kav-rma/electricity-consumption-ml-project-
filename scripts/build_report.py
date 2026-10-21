"""Fill docs/report_template.md with tables from saved results.

Example:
    python scripts/build_report.py
Writes outputs/reports/project_report.md. Missing results become explicit
"not available" notes rather than numbers.
"""

import argparse
import logging

from _bootstrap import setup_logging

from electricity_anomaly.config import load_config
from electricity_anomaly.reporting import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    setup_logging()
    path = build_report(load_config(args.config))
    logging.getLogger("build_report").info("Report written to %s", path)


if __name__ == "__main__":
    main()
