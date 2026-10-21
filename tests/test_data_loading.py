"""Dataset loading, schema detection, streaming readers and sampling."""

import struct
import zipfile

import pandas as pd
import pytest

from electricity_anomaly.data_loader import (DataUnavailableError, SchemaError, detect_columns, hash_sample_mask,
                                             iter_csv_chunks, load_clean_readings, load_real_readings)

CSV_TEXT = (
    "LCLid,stdorToU,DateTime,KWH/hh (per half hour) \n"
    "MAC1,Std,2013-01-07 00:00:00.0000000,0.25 \n"
    "MAC1,Std,2013-01-07 00:30:00.0000000,Null\n"
    "MAC1,Std,not-a-date,0.3\n"
    "MAC2,ToU,2013-01-07 00:00:00.0000000,0.4\n"
    ",Std,2013-01-07 00:00:00.0000000,0.4\n"
)


def test_detect_columns_handles_trailing_space_and_case(base_cfg):
    cols = ["lclid", "stdorToU", "DateTime", "KWH/hh (per half hour) "]
    mapping = detect_columns(cols, base_cfg["schema"])
    assert mapping == {"customer_id": "lclid", "timestamp": "DateTime",
                       "consumption": "KWH/hh (per half hour) ", "tariff": "stdorToU"}


def test_detect_columns_missing_raises(base_cfg):
    with pytest.raises(SchemaError, match="consumption"):
        detect_columns(["LCLid", "DateTime", "something"], base_cfg["schema"])


def test_iter_csv_chunks_csv_zip_and_directory(tmp_path):
    csv = tmp_path / "a.csv"
    csv.write_text(CSV_TEXT)
    assert sum(len(c) for c in iter_csv_chunks(csv, 2)) == 5
    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, "inner.csv")
    chunks = list(iter_csv_chunks(zpath, 10))
    assert len(chunks[0]) == 5 and chunks[0].iloc[1, 3] == "Null"   # tokens kept as text
    folder = tmp_path / "parts"
    folder.mkdir()
    (folder / "p1.csv").write_text(CSV_TEXT)
    (folder / "p2.csv").write_text(CSV_TEXT)
    assert sum(len(c) for c in iter_csv_chunks(folder, 100)) == 10


def _deflate64_zip(path, name: str, payload: bytes) -> None:
    """Write a zip whose single member uses Deflate64 (method 9), like the London file."""
    import inflate64

    deflater = inflate64.Deflater()
    comp = deflater.deflate(payload) + deflater.flush()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr(name, comp)
    raw = bytearray(path.read_bytes())
    # Patch the compression method in the local header (offset 8) and central directory (offset 10).
    struct.pack_into("<H", raw, 8, 9)
    cd = raw.rfind(b"PK\x01\x02")
    struct.pack_into("<H", raw, cd + 10, 9)
    path.write_bytes(bytes(raw))


def test_deflate64_zip_streaming(tmp_path):
    zpath = tmp_path / "d64.zip"
    _deflate64_zip(zpath, "data.csv", CSV_TEXT.encode())
    with zipfile.ZipFile(zpath) as z:
        assert z.infolist()[0].compress_type == 9
    chunk = next(iter_csv_chunks(zpath, 100))
    assert len(chunk) == 5 and list(chunk.columns)[0] == "LCLid"


def test_missing_path_raises(tmp_path):
    with pytest.raises(DataUnavailableError):
        next(iter_csv_chunks(tmp_path / "nope.csv", 10))


def test_hash_sample_is_deterministic_and_order_independent():
    ids = pd.Series([f"MAC{i:06d}" for i in range(4000)])
    m1 = hash_sample_mask(ids, 0.25, seed=1)
    m2 = hash_sample_mask(ids.iloc[::-1], 0.25, seed=1).iloc[::-1]
    assert (m1.to_numpy() == m2.to_numpy()).all()
    assert 0.2 < m1.mean() < 0.3
    assert (hash_sample_mask(ids, 0.25, seed=2) != m1).any()
    assert hash_sample_mask(ids, 1.0, seed=1).all()


def test_load_clean_readings_counts_exclusions(tmp_path, small_cfg):
    csv = tmp_path / "a.csv"
    csv.write_text(CSV_TEXT)
    df, stats = load_clean_readings(csv, small_cfg, sample_fraction=1.0, tariff="Std", chunksize=2)
    assert stats["raw_rows_scanned"] == 5
    assert stats["rows_other_tariff"] == 1
    assert stats["rows_missing_reading"] == 1
    assert stats["rows_malformed_timestamp"] == 1
    assert stats["rows_missing_customer_id"] == 1
    assert len(df) == 1 and df["kwh"].iloc[0] == pytest.approx(0.25)
    assert set(df.columns) == {"customer_id", "timestamp", "kwh", "local_time"}


def test_real_data_missing_gives_instructions(small_cfg):
    with pytest.raises(DataUnavailableError, match="download_data.py"):
        load_real_readings(small_cfg, use_cache=False)
