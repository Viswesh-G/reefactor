"""Design audit for the observational endpoint. No outcomes used to allocate folds."""
import hashlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from .pipeline import file_hash
from .region import contains_points


def chart_surveys(path):
    """One candidate endpoint per survey, with explicit denominator and QC."""
    hard = ["Branching corals", "Boulder corals", "Plate corals"]
    frame = pl.scan_parquet(path).filter(pl.col("coral_type").is_in(hard))
    scores = [pl.col(c).str.extract(r"^[BCDE]([1-6])$", 1).cast(pl.Float64) for c in ("lightest", "darkest")]
    chart_ok = (pl.col("valid_chart_range") & (scores[0] <= scores[1]) &
                ((pl.col("chart_mean") - (scores[0]+scores[1])/2).abs() < 1e-6)).fill_null(False)
    return frame.group_by("survey_id").agg(
        pl.first("lat", "lon", "local_survey_date"), pl.len().alias("hard_corals"),
        (pl.col("chart_mean") < 2.1).sum().alias("pale_corals"),
        (chart_ok.all() & pl.col("coral_number").is_not_null().all() &
         (pl.col("coral_number").n_unique() == pl.len()) &
         (pl.struct("lat", "lon", "local_survey_date").n_unique() == 1)).alias("qc")
    ).with_columns((pl.col("pale_corals")*10 >= pl.col("hard_corals")*3).alias("chart_flag")).collect()


def spatial_group(lat, lon):
    block = f"{int(np.floor(lat))},{int(np.floor(lon))}"
    bucket = int.from_bytes(hashlib.sha256(("reefactor-v1/"+block).encode()).digest()[:4], "big") % 5
    return bucket, block


def away_from(group, others, buffer_km):
    """Apply a physical-distance buffer between distinct coarse exposure cells."""
    if not len(group) or not len(others):
        return group

    def xyz(frame):
        lat, lon = np.deg2rad(frame.select("cell_lat", "cell_lon").to_numpy()).T
        return np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))

    reference = cKDTree(xyz(others.select("cell_lat", "cell_lon").unique()))
    chord, _ = reference.query(xyz(group))
    min_chord = 2*np.sin(buffer_km/(2*6371.0))
    return group.filter(pl.Series(chord >= min_chord))


def blocked_folds(frame, rotation, gap, buffer_km):
    """Test each later-event block once over five prespecified rotations."""
    day = pl.col("local_survey_date")
    bucket = pl.col("bucket")
    val_bucket = (rotation+1) % 5
    test = frame.filter((bucket == rotation) & (day >= date(2025,1,1)+gap))
    validation = frame.filter((bucket == val_bucket) & day.is_between(date(2022,1,1)+gap, date(2024,12,31)))
    validation = away_from(validation, test, buffer_km)
    train = frame.filter(~bucket.is_in([rotation, val_bucket]) & (day < date(2022,1,1)))
    train = away_from(train, pl.concat([validation, test]), buffer_km)
    return {"train": train, "validation": validation, "test": test}


def design_chart_study(path, cfg, lookback=14, lead=7, buffer_km=50):
    """Return design counts before any full-cube support or skill claim."""
    if min(lookback, lead) < 1 or buffer_km <= 0:
        raise ValueError("Require positive history, lead and buffer")
    path = Path(path)
    surveys = chart_surveys(path)
    usable = surveys.filter(pl.col("qc") & (pl.col("hard_corals") >= 15) &
                            pl.col("local_survey_date").is_not_null() & (pl.col("survey_id") != ""))
    start, end = (date.fromisoformat(cfg["time"][k]) for k in ("start", "end"))
    # Local-day labels. Use whole UTC feature days and reserve a full lead before
    # the earliest possible survey (00:00 AEST = previous UTC date at 14:00).
    earliest = start + timedelta(days=lookback+lead+1)
    grid = cfg["grid"]
    r = grid["resolution"]
    within = usable.filter(pl.col("local_survey_date").is_between(earliest, end) &
        (pl.col("lat") >= grid["lat_min"]) & (pl.col("lat") < grid["lat_max"]) &
        (pl.col("lon") >= grid["lon_min"]) & (pl.col("lon") < grid["lon_max"]))
    within = within.with_columns(*[
        (grid[f"{axis}_min"] + (((pl.col(axis)-grid[f"{axis}_min"])/r).floor()+0.5)*r).alias("cell_"+axis)
        for axis in ("lat", "lon")])
    within = within.filter(pl.Series(contains_points(within["cell_lat"].to_numpy(), within["cell_lon"].to_numpy())))
    groups = [spatial_group(lat, lon) for lat, lon in within.select("cell_lat", "cell_lon").iter_rows()]
    within = within.with_columns(pl.Series("bucket", [g for g,b in groups], dtype=pl.Int8),
                                 pl.Series("block", [b for g,b in groups], dtype=pl.String))
    # Purge input histories crossing either temporal split, including the UTC/local margin.
    gap = timedelta(days=lookback+lead+1)
    counts = []
    for rotation in range(5):
        folds = blocked_folds(within, rotation, gap, buffer_km)
        counts.append({name: {"surveys": f.height, "cells": f.select("cell_lat","cell_lon").n_unique(),
                              "blocks": f["block"].n_unique(), "first": str(f["local_survey_date"].min()) if len(f) else None,
                              "last": str(f["local_survey_date"].max()) if len(f) else None}
                       for name, f in folds.items()})
    return {"chart_sha256": file_hash(path), "profile": cfg["profile"], "requested": cfg["time"],
            "eligible_dated_surveys": usable.height, "calendar_geometry_eligible": within.height,
            "first_eligible_in_window": str(within["local_survey_date"].min()) if len(within) else None,
            "last_eligible_in_window": str(within["local_survey_date"].max()) if len(within) else None,
            "lookback_days": lookback, "lead_days": lead, "spatial_buffer_km": buffer_km, "folds": counts,
            "status": "design_only_pending_full_cube_domain_and_out_of_sample_evaluation"}
