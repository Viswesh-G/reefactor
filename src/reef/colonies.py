"""AIMS 2022 field observations and their pre-survey environmental histories."""
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from .labels import gbr_history_end, sample_history
from .observability import exclusive_run
from .pipeline import atomic_replace, file_hash, open_cube, write_json_atomic
from .region import contains_points
from .sources import download


RELEASE = "https://doi.org/10.5281/zenodo.14752226"
URL = ("https://raw.githubusercontent.com/open-AIMS/2022-Bleaching-central-GBR/"
       "1d6bb8b3e25fc9b0a9622a1e1cb5342a2e0c6176/Data/bleaching_data.csv")
RAW_SHA = "9cabe865fe6720c2551c10f4c15690ede6aa34543ae4941577657ef1378193d7"


def prepare_colonies(raw):
    frame = pl.read_csv(raw, null_values="NA").with_row_index("source_row").rename(
        {"long": "lon", "Date_bleaching": "survey_date", "Date_2022": "initial_date"})
    frame = frame.with_columns(pl.col("survey_date", "initial_date").str.to_date("%d/%m/%Y"))
    if frame.filter(~pl.col("Bleaching").is_between(1, 6) | pl.col("Bleaching").is_null()).height:
        raise ValueError("Unexpected colony score in AIMS release")
    # Keep the original score. Categories 5 and 6 record mortality, a competing outcome.
    frame = frame.with_columns(pl.col("Bleaching").is_between(1, 4).alias("severity_eligible"))
    return frame.filter(pl.Series(contains_points(frame["lat"].to_numpy(), frame["lon"].to_numpy())))


def fetch_colonies(root, execution):
    root = Path(root)
    target = root / "aims_colonies_2022"
    with exclusive_run(root):
        if target.exists():
            meta = json.loads((target / "provenance.json").read_text())
            if meta["raw_sha256"] != RAW_SHA or file_hash(target / "observations.parquet") != meta["parquet_sha256"]:
                raise ValueError("Stored colony observations differ from provenance")
            return meta
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            raw = Path(temporary) / "observations.csv"
            transfer = download(URL, raw, execution)
            if transfer["raw_sha256"] != RAW_SHA:
                raise ValueError("Pinned AIMS input checksum mismatch")
            frame = prepare_colonies(raw)
            stage = Path(temporary) / "ready"
            stage.mkdir()
            parquet = stage / "observations.parquet"
            frame.write_parquet(parquet)
            meta = {"citation": "Álvarez-Noriega et al. (2025)", "archive": RELEASE,
                    "url": URL, "license": "CC-BY-4.0", **transfer,
                    "parquet_sha256": file_hash(parquet), "stored_bytes": parquet.stat().st_size,
                    "rows": frame.height, "ordinal_rows": frame["severity_eligible"].sum(),
                    "sites": frame["Site"].n_unique(), "taxa": frame["Taxa"].n_unique(),
                    "first": str(frame["survey_date"].min()), "last": str(frame["survey_date"].max())}
            write_json_atomic(stage / "provenance.json", meta)
            atomic_replace(stage, target)
    return meta


def audit_colony_overlap(root, parquet, lookback=14, lead=7):
    """Count exact support at observed sites. Compute each shared cell-history once."""
    root = Path(root)
    parquet = Path(parquet)
    manifest = json.loads((root / "manifest.json").read_text())
    plan = manifest["plan"]
    if plan["cadence"] != "daily" or min(lookback, lead) < 1:
        raise ValueError("A daily cube and positive history and lead are required")
    frame = pl.scan_parquet(parquet).group_by("lat", "lon", "survey_date").agg(
        pl.len().alias("rows"), pl.col("severity_eligible").sum().alias("eligible")).collect()
    fields = plan["feature_names"]
    grid = plan["grid"]
    counts = {"outside_grid": 0, "incomplete_history": 0, "outside_domain": 0, "matched": 0}
    histories = {}
    with open_cube(root) as cube:
        mask = cube.model_domain.values
        for lat, lon, day, rows, eligible in frame.iter_rows():
            if not (grid["lat_min"] <= lat < grid["lat_max"] and grid["lon_min"] <= lon < grid["lon_max"]):
                counts["outside_grid"] += rows
                continue
            row = int(np.floor((lat-grid["lat_min"])/grid["resolution"]))
            col = int(np.floor((lon-grid["lon_min"])/grid["resolution"]))
            if not mask[row, col]:
                counts["outside_domain"] += rows
                continue
            end = gbr_history_end(day, lead)
            start = end - timedelta(days=lookback-1)
            if start < date.fromisoformat(plan["requested"]["start"]) or end > date.fromisoformat(plan["requested"]["end"]):
                counts["incomplete_history"] += rows
                continue
            key = (row, col, day)
            if key not in histories:
                window = sample_history(cube, lat=lat, lon=lon, survey_date=day, features=fields,
                                        lookback=lookback, lead_days=(day-end).days, resolution=grid["resolution"])
                if not bool(np.isfinite(window.to_array()).all().compute(scheduler="synchronous")):
                    raise ValueError("Nonfinite value on the asserted common domain")
                histories[key] = {"row": row, "col": col, "survey_date": str(day),
                                  "history_start": str(start), "history_end": str(end),
                                  "rows": 0, "ordinal_rows": 0}
            histories[key]["rows"] += rows
            histories[key]["ordinal_rows"] += eligible
            counts["matched"] += rows
    return {"fingerprint": manifest["fingerprint"], "label_sha256": file_hash(parquet),
            "fields_per_history": len(fields), "lookback_days": lookback, "lead_days": lead,
            "row_counts": counts, "distinct_cells": len({(r,c) for r,c,_ in histories}),
            "distinct_histories": len(histories), "histories": [histories[k] for k in sorted(histories)]}
