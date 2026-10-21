"""Download the London smart-meter dataset and build the sampled Parquet cache.

Examples (PowerShell, from the project root):
    python scripts/download_data.py              # download (if missing) + build cache
    python scripts/download_data.py --no-prepare # download only
    python scripts/download_data.py --skip-download --rebuild

If the download fails, get LCL-FullData.zip manually from the London Datastore
page listed in config.yaml (real.landing_page), put it in data/raw/, or point
real.local_path at a zip, a CSV, or a folder of CSVs.
"""

import argparse
import json
import logging

from _bootstrap import setup_logging

from electricity_anomaly.config import load_config
from electricity_anomaly.data_loader import download_file, list_csv_sources, load_real_readings, real_data_path

logger = logging.getLogger("download_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--skip-download", action="store_true", help="Use the existing local file")
    parser.add_argument("--no-prepare", action="store_true", help="Do not build the sampled Parquet cache")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the cache even if it is up to date")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    path = real_data_path(cfg)

    if not args.skip_download:
        if path.suffix.lower() != ".zip":
            raise SystemExit(f"real.local_path ({path}) is not a .zip; download manually or use --skip-download")
        logger.info("Source: %s", cfg["real"]["download_url"])
        download_file(cfg["real"]["download_url"], path)
    if not path.exists():
        raise SystemExit(f"{path} does not exist. See {cfg['real']['landing_page']}")
    logger.info("Dataset files: %s", list_csv_sources(path))

    if not args.no_prepare:
        df, stats = load_real_readings(cfg, use_cache=not args.rebuild)
        logger.info("Cached %s clean readings for %d households", f"{len(df):,}", df["customer_id"].nunique())
        print(json.dumps(dict(stats), indent=2, default=str))


if __name__ == "__main__":
    main()
