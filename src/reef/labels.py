"""Versioned field observations, kept sparse instead of fabricating a global label raster."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from .pipeline import file_hash, write_json_atomic
from .sources import download
from .catalog import LABEL_SOURCE

LABEL_URL, LABEL_DOI = LABEL_SOURCE["url"], LABEL_SOURCE["doi"]
CONTEXT = list(LABEL_SOURCE["context"])
IDENTIFIERS = ["Site_ID", "Sample_ID", "Data_Source", "Reef_ID", "Bleaching_Level"]


def convert_labels(csv_path, output):
    """Drop duplicated climate summaries (some use full-series/future statistics)."""
    frame = pl.scan_csv(csv_path, null_values=["nd", "NA", ""], infer_schema_length=0)
    required = set(IDENTIFIERS + CONTEXT + ["Date", "Latitude_Degrees", "Longitude_Degrees", "Percent_Bleaching"])
    missing = required-set(frame.collect_schema().names())
    if missing:
        raise ValueError(f"Label schema changed: {sorted(missing)}")
    selected = frame.with_row_index("source_row").select(
        "source_row", *IDENTIFIERS, pl.col("Date").str.to_date("%Y-%m-%d", strict=True).alias("survey_date"),
        pl.col("Latitude_Degrees").cast(pl.Float64, strict=False).alias("lat"),
        pl.col("Longitude_Degrees").cast(pl.Float64, strict=False).alias("lon"),
        pl.col("Percent_Bleaching").cast(pl.Float32, strict=False).alias("bleaching_percent"),
        *[pl.col(c) if c == "Exposure" else pl.col(c).cast(pl.Float32, strict=False) for c in CONTEXT],
    ).with_columns((pl.col("bleaching_percent").is_between(0, 100) &
                    pl.col("lat").is_between(-90, 90) & pl.col("lon").is_between(-180, 180) &
                    pl.col("survey_date").is_not_null()).fill_null(False).alias("target_valid"))
    # No removal of repeated surveys or aggregation across incompatible protocols.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    selected.sink_parquet(temporary)
    os.replace(temporary, output)
    stored = pl.scan_parquet(output)
    summary = stored.select(pl.len().alias("rows"), pl.col("Site_ID").n_unique().alias("sites"),
        pl.col("survey_date").min().cast(pl.String).alias("first"),
        pl.col("survey_date").max().cast(pl.String).alias("last"),
        pl.col("target_valid").sum().alias("valid_targets")).collect().to_dicts()[0]
    return summary


def fetch_labels(root, execution):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "bleaching_v2.parquet"
    if output.exists():
        raise FileExistsError("Label output exists; use its provenance or choose a new directory")
    with tempfile.TemporaryDirectory(prefix="labels_", dir=root) as scratch:
        raw = Path(scratch)/"bleaching.csv"
        provenance = download(LABEL_URL, raw, execution)
        summary = convert_labels(raw, output)
    metadata = {"doi": LABEL_DOI, "version": 2, "url": LABEL_URL, "license": "CC-BY-4.0",
                "provenance": provenance, "summary": summary, "parquet_sha256": file_hash(output),
                "parquet_bytes": output.stat().st_size,
                "excluded_inputs": "all precomputed thermal/climate summaries; protocol fields are not predictors"}
    write_json_atomic(root / "bleaching_v2.json", metadata)
    return metadata


def sample_history(cube, *, lat, lon, survey_date, features, lookback=84, lead_days=7):
    """Sample only a site's containing cell and history ending BEFORE target day.

    Returns an xarray object, still lazy. Missing cells remain missing. The label
    timestamp can be approximate in survey records; retain original survey metadata.
    """
    if lookback < 1 or lead_days < 1:
        raise ValueError("lookback and lead_days must be positive")
    if len(cube.lat) < 2 or len(cube.lon) < 2:
        raise ValueError("Need at least two coordinates to identify cell edges")
    dy, dx = float(cube.lat[1]-cube.lat[0]), float(cube.lon[1]-cube.lon[0])
    row = int(np.floor((lat-(float(cube.lat[0])-dy/2))/dy))
    col = int(np.floor((lon-(float(cube.lon[0])-dx/2))/dx))
    if not (0 <= row < len(cube.lat) and 0 <= col < len(cube.lon)):
        raise ValueError("Site is outside the cube; refusing nearest-edge extrapolation")
    end = np.datetime64(survey_date, "D")-np.timedelta64(lead_days, "D")
    start = end-np.timedelta64(lookback-1, "D")
    sample = cube[features].isel(lat=row, lon=col).sel(time=slice(start, end))
    if sample.sizes["time"] != lookback:
        raise ValueError("Incomplete pre-issuance history")
    if "available_not_before" in sample.coords and bool((sample.available_not_before > end).any()):
        raise ValueError("History contains context unavailable at issuance")
    return sample
