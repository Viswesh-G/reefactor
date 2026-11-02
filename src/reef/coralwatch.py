"""Stream the public chart survey export and retain only anonymized GBR rows."""
import csv
import hashlib
import io
import json
import tempfile
import time
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit

import polars as pl
import pyarrow.parquet as pq
import requests

from .observability import exclusive_run
from .pipeline import atomic_replace, file_hash, write_json_atomic
from .region import contains_points, geometry

API = "https://coralwatchdata.org/api/download"
SOURCE = "https://coralwatch.org/new-coralwatch-interactive-data-map/"
COLUMNS = {"Activity ID": "survey_id", "Latitude": "lat", "Longitude": "lon",
           "Observation date": "observation_timestamp", "Light condition": "light_condition",
           "Depth (metres)": "depth_m", "Water temperature (deg. C)": "water_temperature_c",
           "Coral No.": "coral_number", "Colour Code Lightest": "lightest",
           "Colour Code Darkest": "darkest", "Average": "chart_mean", "Coral Type": "coral_type"}


class BoundedReader(io.RawIOBase):
    """Give the standard CSV decoder a byte-limited, hashed stream."""
    def __init__(self, stream, maximum, deadline):
        self.stream, self.maximum, self.deadline = stream, maximum, deadline
        self.count, self.digest = 0, hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.stream.read(len(buffer))
        self.count += len(data)
        if self.count > self.maximum or time.monotonic() > self.deadline:
            raise ValueError("Survey export exceeded configured time or byte budget")
        self.digest.update(data)
        buffer[:len(data)] = data
        return len(data)


def subset_surveys(stream, output):
    reader = csv.DictReader(stream)
    if not set(COLUMNS).issubset(reader.fieldnames or []):
        raise ValueError("CoralWatch export schema changed")
    west, south, east, north = geometry().bounds
    writer = None
    total = 0
    try:
        while batch := list(islice(reader, 4096)):
            total += len(batch)
            # Select permitted columns before constructing or writing a table.
            frame = pl.DataFrame({name: [row[col] for row in batch] for col, name in COLUMNS.items()},
                                 schema={name: pl.String for name in COLUMNS.values()})
            frame = frame.with_columns(pl.col("lat", "lon", "depth_m", "water_temperature_c", "chart_mean")
                                       .cast(pl.Float64, strict=False))
            frame = frame.filter(pl.col("lat").is_between(south, north) & pl.col("lon").is_between(west, east))
            frame = frame.filter(pl.Series(contains_points(frame["lat"].to_numpy(), frame["lon"].to_numpy())))
            if frame.is_empty():
                continue
            # Preserve original timestamp. A date-only survey is often encoded at
            # local midnight; the original UTC day can then be one day earlier.
            frame = frame.with_columns(
                pl.col("observation_timestamp").str.to_datetime("%+", time_zone="UTC", strict=False)
                .dt.convert_time_zone("Australia/Brisbane").dt.date().alias("local_survey_date"),
                pl.col("chart_mean").is_between(1, 6).fill_null(False).alias("valid_chart_range"))
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("No survey coordinates inside the official GBR boundary")
    return total


def fetch_coralwatch(root, execution, expected_sha=None):
    root = Path(root)
    target = root / "coralwatch"
    with exclusive_run(root):
        if target.exists():
            meta = json.loads((target / "provenance.json").read_text())
            if meta["parquet_sha256"] != file_hash(target / "surveys.parquet") or (
                    expected_sha and meta["raw_sha256"] != expected_sha):
                raise ValueError("CoralWatch snapshot checksum mismatch")
            return meta
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            stage = Path(tmp) / "ready"
            stage.mkdir()
            output = stage / "surveys.parquet"
            timeout = (10, execution["timeout_seconds"])
            # This is the research download offered by the public map. Its signed
            # asset URL expires and is never included in logs or provenance.
            with requests.post(API, json={"downloadMode": "all"}, timeout=timeout) as lookup:
                lookup.raise_for_status()
                payload = lookup.json()
                if payload.get("success") is not True or urlsplit(payload.get("url", "")).scheme != "https":
                    raise ValueError("Publisher did not return an HTTPS research export")
            try:
                asset = requests.get(payload["url"], stream=True, timeout=timeout)
            except requests.RequestException:
                raise RuntimeError("Could not connect to the publisher's survey export") from None
            with asset as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Survey export returned HTTP {response.status_code}")
                response.raw.decode_content = True
                reader = BoundedReader(response.raw, execution["max_download_mb"] * 1024**2,
                                       time.monotonic() + execution["source_deadline_seconds"])
                with io.TextIOWrapper(io.BufferedReader(reader), encoding="utf-8-sig", newline="") as stream:
                    total = subset_surveys(stream, output)
            digest = reader.digest.hexdigest()
            if expected_sha and digest != expected_sha:
                raise ValueError("Publisher export differs from requested snapshot")
            frame = pl.scan_parquet(output)
            counts = frame.select(pl.len().alias("gbr_rows"), pl.col("survey_id").n_unique().alias("survey_ids"),
                pl.col("local_survey_date").min().cast(pl.String).alias("first"),
                pl.col("local_survey_date").max().cast(pl.String).alias("last"),
                pl.col("local_survey_date").null_count().alias("undated_rows"),
                pl.col("valid_chart_range").sum().alias("valid_chart_rows")).collect().to_dicts()[0]
            meta = {**counts, "source": SOURCE, "endpoint": API, "global_rows_streamed": total,
                    "raw_sha256": digest, "decoded_bytes": reader.count, "parquet_sha256": file_hash(output),
                    "stored_bytes": output.stat().st_size, "endpoint_type": "citizen_science_colour_chart"}
            write_json_atomic(stage / "provenance.json", meta)
            atomic_replace(stage, target)
    return meta
