"""Validate a request before I/O and yield bounded, reproducible partition jobs."""
import calendar
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import yaml

from .catalog import SOURCES, serializable_catalog
from .region import boundary_identity, region_mask

GBR_BOUNDS = {"lat_min": -24.0, "lat_max": -10.0, "lon_min": 142.0, "lon_max": 154.0}


@dataclass(frozen=True)
class Job:
    key: str
    start: str
    end: str
    lat_edges: tuple[float, float]
    lon_edges: tuple[float, float]
    resolution: float

    def centers(self):
        def axis(edges):
            count = round((edges[1] - edges[0]) / self.resolution)
            return edges[0] + (np.arange(count) + 0.5) * self.resolution
        return axis(self.lat_edges), axis(self.lon_edges)


def load_config(path, profile=None, overrides=None):
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if profile:
        cfg["profile"] = profile
    for section, values in (overrides or {}).items():
        cfg[section].update(values)
    validate_config(cfg)
    return cfg


def source_names(cfg):
    return cfg["profiles"][cfg["profile"]]


def validate_config(cfg):
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported schema_version")
    names = source_names(cfg)
    if not names or len(set(names)) != len(names) or set(names) - SOURCES.keys():
        raise ValueError("Profile must contain unique registered sources")
    cadences = {SOURCES[n]["cadence"] for n in names}
    if ("monthly" in cadences or "static" in cadences) and len(cadences) > 1:
        raise ValueError("Monthly/static context must use separate stores")
    start, end = (date.fromisoformat(cfg["time"][k]) for k in ("start", "end"))
    if start > end:
        raise ValueError("start must be <= end")
    if "monthly" in cadences and (start.day != 1 or end.day != calendar.monthrange(end.year, end.month)[1]):
        raise ValueError("Monthly context requires whole calendar months")
    grid = cfg["grid"]
    if cfg.get("study", {}).get("region", "GBR") != "GBR":
        raise ValueError("This research path is scoped to GBR only")
    if any(grid[k] < GBR_BOUNDS[k] for k in ("lat_min", "lon_min")) or any(
            grid[k] > GBR_BOUNDS[k] for k in ("lat_max", "lon_max")):
        raise ValueError("Requested bbox lies outside the configured GBR study envelope")
    r = float(grid["resolution"])
    if not math.isfinite(r) or r <= 0 or r > 5:
        raise ValueError("resolution must be positive and <= 5 degrees")
    if not (-90 <= grid["lat_min"] < grid["lat_max"] <= 90 and
            -180 <= grid["lon_min"] < grid["lon_max"] <= 180):
        raise ValueError("Invalid edges; split antimeridian-crossing requests")
    for axis in ("lat", "lon"):
        cells = (grid[f"{axis}_max"] - grid[f"{axis}_min"]) / r
        if not np.isclose(cells, round(cells)):
            raise ValueError("Grid edges must span whole cells")
    if not isinstance(grid["tile_cells"], int) or grid["tile_cells"] < 1:
        raise ValueError("tile_cells must be a positive integer")
    for name in names:
        source = SOURCES[name]
        if r + 1e-9 < source["resolution"]:
            raise ValueError(f"{name}: refusing artificial spatial upsampling")
        if start.isoformat() < source["start"] or end.isoformat() > source["catalog_end"]:
            raise ValueError(f"{name}: request outside reviewed catalog window; review catalog first")
        if grid["lat_min"] < source["lat_bounds"][0] or grid["lat_max"] > source["lat_bounds"][1]:
            raise ValueError(f"{name}: request exceeds latitude support")
        if cfg["execution"].get("require_pinned_versions") and not (
                cfg.get("versions", {}).get(name) or source["version"]):
            raise ValueError(f"Pin a live dataset version for {name}")
    exe = cfg["execution"]
    for key in ("workers", "max_attempts", "max_jobs"):
        if not isinstance(exe[key], int) or exe[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("timeout_seconds", "max_download_mb", "max_tile_mb", "max_scratch_mb", "max_output_mb"):
        if not math.isfinite(exe[key]) or exe[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not math.isfinite(exe.get("source_deadline_seconds", 180)) or exe.get("source_deadline_seconds", 180) <= 0:
        raise ValueError("source_deadline_seconds must be positive and finite")
    if not 0 <= exe["max_missing_fraction"] < 1 or exe["retry_base_seconds"] < 0:
        raise ValueError("Invalid missing fraction or retry delay")
    if not math.isfinite(exe.get("remote_cache_mb", 0)) or exe.get("remote_cache_mb", 0) < 0:
        raise ValueError("remote_cache_mb must be nonnegative and finite")
    if not Path(cfg["storage"]["root"]).name:
        raise ValueError("Storage root must name a directory")


def effective_tile_cells(cfg):
    """Budget the selected native arrays, not only their smaller output grid."""
    tile, r = cfg["grid"]["tile_cells"], cfg["grid"]["resolution"]
    start, end = (date.fromisoformat(cfg["time"][key]) for key in ("start", "end"))
    days = min(31, (end-start).days+1)
    maximum = cfg["execution"]["max_tile_mb"]*1024**2
    while True:
        largest = 0
        for name in source_names(cfg):
            source = SOURCES[name]
            steps = 1 if source["cadence"] in ("monthly", "static") else days
            if source["cadence"] == "hourly":
                steps = (days+1)*24
            native_side = math.ceil(tile*r/source["resolution"])+4  # interpolation/area halo
            largest = max(largest, native_side**2*steps*len(source["features"])*8)
        if largest <= maximum:
            return tile
        if tile == 1:
            raise ValueError("Even one target cell exceeds the native-array memory budget")
        tile = max(1, tile//2)


def iter_jobs(cfg):
    grid = cfg["grid"]
    start, end = (date.fromisoformat(cfg["time"][k]) for k in ("start", "end"))
    nlat = round((grid["lat_max"] - grid["lat_min"]) / grid["resolution"])
    nlon = round((grid["lon_max"] - grid["lon_min"]) / grid["resolution"])
    tile, r = effective_tile_cells(cfg), grid["resolution"]
    geographic_support = region_mask(grid["lat_min"]+(np.arange(nlat)+0.5)*r,
                                     grid["lon_min"]+(np.arange(nlon)+0.5)*r)
    static = all(SOURCES[n]["cadence"] == "static" for n in source_names(cfg))
    current = start
    count = 0
    while current <= end:
        last = min(end, date(current.year, current.month, calendar.monthrange(current.year, current.month)[1]))
        if static:
            last = end
        for row in range(0, nlat, tile):
            for col in range(0, nlon, tile):
                if not geographic_support[row:row+tile, col:col+tile].any():
                    continue  # Geometry-only exclusion, identical for every feature.
                count += 1
                if count > cfg["execution"]["max_jobs"]:
                    raise ValueError("Request exceeds max_jobs; narrow region or raise the explicit budget")
                yield Job(f"{current.isoformat()}_{row:05d}_{col:05d}", current.isoformat(), last.isoformat(),
                    (grid["lat_min"] + row*r, grid["lat_min"] + min(row+tile, nlat)*r),
                    (grid["lon_min"] + col*r, grid["lon_min"] + min(col+tile, nlon)*r), r)
        current = last + timedelta(days=1)


def fingerprint(cfg):
    catalog = serializable_catalog()
    # Universal newline reading keeps semantic code identity stable on Windows/Linux.
    code = {path.name: hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))}
    payload = {"implementation_contract": 2, "code_sha256": code, "config": cfg, "boundary": boundary_identity(),
               "sources": {n: catalog[n] for n in source_names(cfg)}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def make_plan(cfg):
    names = source_names(cfg)
    count, largest, cells_days = 0, 0, 0
    nvars = sum(len(SOURCES[n]["features"]) for n in names)
    for job in iter_jobs(cfg):
        lat, lon = job.centers()
        days = (date.fromisoformat(job.end) - date.fromisoformat(job.start)).days + 1
        if SOURCES[names[0]]["cadence"] in ("monthly", "static"):
            days = 1
        elements = days * len(lat) * len(lon)
        largest = max(largest, elements * nvars * 4)
        cells_days += elements
        count += 1
    estimated = cells_days * (nvars * 4 + nvars + 1)
    if not count:
        raise ValueError("No requested cell centers are inside the official GBR boundary")
    if largest > cfg["execution"]["max_tile_mb"] * 1024**2:
        raise ValueError("Tile exceeds memory budget; lower tile_cells")
    if estimated > cfg["execution"]["max_output_mb"] * 1024**2:
        raise ValueError("Uncompressed output estimate exceeds disk budget; narrow region/time")
    if cfg["execution"]["workers"] * cfg["execution"]["max_download_mb"] > cfg["execution"]["max_scratch_mb"]:
        raise ValueError("Concurrent maximum downloads exceed scratch budget")
    return {
        "fingerprint": fingerprint(cfg), "profile": cfg["profile"], "sources": names,
        "jobs": count, "fields": nvars, "largest_float32_tile_bytes": largest,
        "feature_names": [f.name for n in names for f in SOURCES[n]["features"]],
        "cadence": ("daily" if SOURCES[names[0]]["cadence"] == "hourly" else SOURCES[names[0]]["cadence"]),
        "study": {"region": "GBR", "envelope": GBR_BOUNDS, "boundary": boundary_identity()},
        "effective_tile_cells": effective_tile_cells(cfg),
        "unity": "all_selected_features_at_every_retained_coordinate_for_the_entire_request",
        "estimated_uncompressed_output_bytes": estimated,
        "reviewed_intersection": {"start": max(SOURCES[n]["start"] for n in names),
                                  "end": min(SOURCES[n]["catalog_end"] for n in names)},
        "coverage_status": "catalog_only_until_partitions_pass_runtime_validation",
        "requested": cfg["time"], "grid": cfg["grid"],
        "source_versions": {n: cfg.get("versions", {}).get(n) or SOURCES[n]["version"] for n in names},
        "unresolved_versions": [n for n in names if not (cfg.get("versions", {}).get(n) or SOURCES[n]["version"])],
    }
