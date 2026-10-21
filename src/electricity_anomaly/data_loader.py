"""Dataset access: downloading, streaming CSV chunks, sampling and caching.

The London file is ~8.5 GB of CSV inside a Deflate64-compressed zip. It is
read as a stream in chunks; only rows for a deterministic sample of
households are kept, so memory use stays proportional to the sample.
"""

from __future__ import annotations

import io
import json
import logging
import struct
import time
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from pandas.api.types import union_categoricals

from .config import config_hash, get_windows, project_path, resolve_path
from .preprocessing import clean_raw_chunk

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = ("customer_id", "timestamp", "consumption")
DEFLATE64 = 9


class SchemaError(ValueError):
    """Raised when required columns cannot be identified."""


class DataUnavailableError(FileNotFoundError):
    """Raised when a dataset file is missing, with instructions to obtain it."""


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------
def _normalise(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def detect_columns(columns: list[str], schema: dict[str, Any]) -> dict[str, str]:
    """Map canonical column names to the actual names in a file.

    Matching ignores case and surrounding whitespace. Raises ``SchemaError``
    listing the columns found if a required column is absent.
    """
    lookup = {_normalise(c): c for c in columns}
    candidates = {
        "customer_id": schema["customer_id_candidates"],
        "timestamp": schema["timestamp_candidates"],
        "consumption": schema["consumption_candidates"],
        "tariff": schema.get("tariff_candidates", []),
    }
    mapping: dict[str, str] = {}
    for canonical, names in candidates.items():
        for name in names:
            if _normalise(name) in lookup:
                mapping[canonical] = lookup[_normalise(name)]
                break
    missing = [c for c in CANONICAL_COLUMNS if c not in mapping]
    if missing:
        raise SchemaError(f"Could not identify column(s) {missing}; file has columns {list(columns)}. "
                          "Add the names to schema.*_candidates in config.yaml.")
    return mapping


# ---------------------------------------------------------------------------
# Streaming readers
# ---------------------------------------------------------------------------
class _Deflate64Stream(io.RawIOBase):
    """Read-only stream that inflates a Deflate64 zip member on the fly."""

    def __init__(self, path: Path, info: zipfile.ZipInfo, block_size: int = 4 << 20):
        try:
            import inflate64
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("This zip uses Deflate64 compression; install it with "
                              "'python -m pip install inflate64' or extract the CSV manually "
                              "and point real.local_path at it.") from exc
        self._fh = open(path, "rb")
        self._fh.seek(info.header_offset)
        header = self._fh.read(30)
        signature, *_, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
        if signature != 0x04034B50:
            raise zipfile.BadZipFile(f"Bad local header for {info.filename}")
        self._fh.seek(info.header_offset + 30 + name_len + extra_len)
        self._remaining = info.compress_size
        self._inflater = inflate64.Inflater()
        self._block = block_size
        self._buffer = b""

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self._buffer and self._remaining > 0:
            raw = self._fh.read(min(self._block, self._remaining))
            if not raw:
                break
            self._remaining -= len(raw)
            self._buffer = self._inflater.inflate(raw)
        n = min(len(b), len(self._buffer))
        b[:n] = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return n

    def close(self) -> None:
        self._fh.close()
        super().close()


def _open_zip_member(path: Path, info: zipfile.ZipInfo, archive: zipfile.ZipFile) -> io.BufferedIOBase:
    if info.compress_type == DEFLATE64:
        return io.BufferedReader(_Deflate64Stream(path, info), buffer_size=1 << 20)
    return archive.open(info)


def list_csv_sources(path: str | Path) -> list[str]:
    """Describe the CSV files a path resolves to (zip members or files)."""
    path = Path(path)
    if path.is_dir():
        return [str(p) for p in sorted(path.glob("*.csv"))]
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            return [i.filename for i in archive.infolist() if i.filename.lower().endswith(".csv")]
    return [str(path)]


def iter_csv_chunks(path: str | Path, chunksize: int) -> Iterator[pd.DataFrame]:
    """Yield string-typed chunks from a CSV, a zip of CSVs, or a directory of CSVs.

    Values are read as strings without automatic NA conversion so that the
    cleaning step can count missing-value tokens such as ``Null`` explicitly.
    """
    path = Path(path)
    if not path.exists():
        raise DataUnavailableError(f"Dataset path does not exist: {path}")
    read_kw = dict(chunksize=chunksize, dtype=str, keep_default_na=False, na_filter=False)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise DataUnavailableError(f"No CSV files in directory {path}")
        for file in files:
            yield from pd.read_csv(file, **read_kw)
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [i for i in archive.infolist() if i.filename.lower().endswith(".csv")]
            if not members:
                raise DataUnavailableError(f"No CSV inside {path}")
            for info in members:
                with _open_zip_member(path, info, archive) as fh:
                    yield from pd.read_csv(fh, **read_kw)
    else:
        yield from pd.read_csv(path, **read_kw)


# ---------------------------------------------------------------------------
# Sampling and loading
# ---------------------------------------------------------------------------
def hash_sample_mask(ids: pd.Series, fraction: float, seed: int) -> pd.Series:
    """Deterministic household sample independent of file order.

    A household is kept when ``crc32("seed:id") / 2**32 < fraction``. The
    same household is therefore always in or out for a given seed, which
    makes a single streaming pass reproducible.
    """
    if fraction >= 1:
        return pd.Series(True, index=ids.index)
    uniques = pd.unique(ids)
    keep = {u: (zlib.crc32(f"{seed}:{u}".encode()) / 2**32) < fraction for u in uniques}
    return ids.map(keep).astype(bool)


def load_clean_readings(path: str | Path, cfg: dict[str, Any], sample_fraction: float = 1.0,
                        tariff: str | None = None, chunksize: int | None = None,
                        seed: int | None = None) -> tuple[pd.DataFrame, Counter]:
    """Stream a raw file, sample households, clean readings and filter dates.

    Returns:
        Clean long-format readings and a ``Counter`` of exclusions.
    """
    windows = get_windows(cfg)
    start = min(w[0] for w in windows.values()) - pd.Timedelta(days=1)
    end = max(w[1] for w in windows.values()) + pd.Timedelta(days=2)
    seed = cfg["random_seed"] if seed is None else seed
    chunksize = chunksize or cfg["real"]["chunksize"]

    stats: Counter = Counter()
    kept: list[pd.DataFrame] = []
    colmap = None
    t0 = time.time()
    for i, chunk in enumerate(iter_csv_chunks(path, chunksize)):
        if colmap is None:
            colmap = detect_columns(list(chunk.columns), cfg["schema"])
            logger.info("Detected columns: %s", colmap)
        stats["raw_rows_scanned"] += len(chunk)
        if tariff and "tariff" in colmap:
            tariff_mask = chunk[colmap["tariff"]].str.strip() == tariff
            stats["rows_other_tariff"] += int((~tariff_mask).sum())
            chunk = chunk.loc[tariff_mask]
        sample = hash_sample_mask(chunk[colmap["customer_id"]].str.strip(), sample_fraction, seed)
        stats["rows_not_in_household_sample"] += int((~sample).sum())
        chunk = chunk.loc[sample]
        if chunk.empty:
            continue
        clean, chunk_stats = clean_raw_chunk(chunk, colmap, cfg)
        stats.update(chunk_stats)
        in_period = (clean["timestamp"] >= start) & (clean["timestamp"] < end)
        stats["rows_outside_load_period"] += int((~in_period).sum())
        clean = clean.loc[in_period]
        clean["customer_id"] = clean["customer_id"].astype("category")   # small memory footprint
        kept.append(clean)
        if i % 10 == 0:
            logger.info("  chunk %d: %s rows scanned, %s kept (%.0fs)", i, f"{stats['raw_rows_scanned']:,}",
                        f"{sum(len(k) for k in kept):,}", time.time() - t0)
    if colmap is None:
        raise DataUnavailableError(f"No rows could be read from {path}")
    if not kept:
        raise DataUnavailableError(f"No readings in {path} matched the sample and study period")
    ids = union_categoricals([k["customer_id"] for k in kept])
    df = pd.concat([k.drop(columns="customer_id") for k in kept], ignore_index=True)
    df.insert(0, "customer_id", ids)
    stats["households_in_sample"] = int(df["customer_id"].nunique())
    logger.info("Loaded %s clean readings for %d households in %.0fs", f"{len(df):,}",
                stats["households_in_sample"], time.time() - t0)
    return df, stats


def real_data_path(cfg: dict[str, Any]) -> Path:
    """Configured location of the real dataset (zip, CSV or directory)."""
    return project_path(cfg["real"]["local_path"])


def load_real_readings(cfg: dict[str, Any], use_cache: bool = True) -> tuple[pd.DataFrame, Counter]:
    """Load the sampled real dataset, using a Parquet cache when available."""
    real = cfg["real"]
    cache_dir = resolve_path(cfg, "processed_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_cfg = {k: real[k] for k in ("household_sample_fraction", "tariff_filter", "start_date", "end_date")}
    key_cfg.update(windows=cfg["windows"], seed=cfg["random_seed"], schema=cfg["schema"],
                   max_kwh=cfg["preprocessing"]["max_kwh_per_interval"])
    key = config_hash(key_cfg)
    cache = cache_dir / real["cache_name"]
    meta_path = cache.with_suffix(".meta.json")
    if use_cache and cache.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("key") == key:
            logger.info("Using cached real readings: %s", cache)
            return pd.read_parquet(cache), Counter(meta["stats"])
        logger.info("Cache %s was built with different settings; rebuilding", cache)

    path = real_data_path(cfg)
    if not path.exists():
        raise DataUnavailableError(
            f"Real dataset not found at {path}.\n"
            f"  Download it with:  python scripts/download_data.py\n"
            f"  or manually from {real['landing_page']}\n"
            f"  and set real.local_path in config/config.yaml (zip, CSV, or folder of CSVs).")
    logger.info("Streaming real dataset from %s (sample fraction %.2f, tariff %s)",
                path, real["household_sample_fraction"], real["tariff_filter"])
    df, stats = load_clean_readings(path, cfg, sample_fraction=real["household_sample_fraction"],
                                    tariff=real["tariff_filter"])
    df.to_parquet(cache, index=False)
    meta_path.write_text(json.dumps({"key": key, "source": str(path), "stats": dict(stats)}, indent=2))
    return df, stats


def download_file(url: str, dest: str | Path, overwrite: bool = False, timeout: int = 60) -> Path:
    """Stream a file from ``url`` to ``dest`` with progress messages."""
    import requests

    dest = Path(dest)
    if dest.exists() and not overwrite:
        logger.info("%s already exists; skipping download", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        next_report = 0.0
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(chunk_size=1 << 20):
                fh.write(block)
                done += len(block)
                if total and done / total >= next_report:
                    logger.info("  downloaded %.0f%% (%d MB)", 100 * done / total, done >> 20)
                    next_report += 0.1
    tmp.replace(dest)
    return dest
