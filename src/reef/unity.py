"""One fixed geographic support and one feature schema for every model sample."""
import json
from pathlib import Path

import numpy as np
import xarray as xr


def required_fields(cfg):
    from .catalog import SOURCES
    from .planning import source_names
    return [f.name for name in source_names(cfg) for f in SOURCES[name]["features"]]


def check_tile_axes(ds, fields, job, cadence):
    missing = set(fields)-set(ds.data_vars)
    if missing:
        raise ValueError(f"Tile {job.key} is missing required features: {sorted(missing)}")
    lat, lon = job.centers()
    expected = {"lat": lat, "lon": lon}
    dims = ("lat", "lon") if cadence == "static" else ("time", "lat", "lon")
    if cadence != "static":
        unit = "M" if cadence == "monthly" else "D"
        expected["time"] = np.arange(np.datetime64(job.start, unit), np.datetime64(job.end, unit)+1,
                                     dtype=f"datetime64[{unit}]").astype("datetime64[ns]")
    for name in fields:
        if ds[name].dims != dims:
            raise ValueError(f"{job.key}/{name}: dimensions differ from the common contract")
        for axis, values in expected.items():
            actual = ds[name][axis].values
            if axis == "time":
                actual = actual.astype("datetime64[ns]")
            if not np.array_equal(actual, values):
                raise ValueError(f"{job.key}/{name}: {axis} differs from the requested common coordinates")


def finalize_domain(root, manifest):
    """Intersect ALL feature/time support across months; never create feature regions.

    Only one spatial tile and a small 2-D GBR mask are resident. Raw source masks
    remain available for audit, while model reads apply this same mask to all fields.
    """
    from .planning import Job
    from .region import region_mask
    grid, fields = manifest["plan"]["grid"], manifest["plan"]["feature_names"]
    r = grid["resolution"]
    lat = grid["lat_min"]+(np.arange(round((grid["lat_max"]-grid["lat_min"])/r))+0.5)*r
    lon = grid["lon_min"]+(np.arange(round((grid["lon_max"]-grid["lon_min"])/r))+0.5)*r
    geographic = region_mask(lat, lon)
    common = geographic.copy()
    visits = np.zeros(common.shape, dtype=np.uint32)
    for relative in manifest["partitions"]:
        path = Path(root)/relative
        metadata = json.loads((path/"_SUCCESS.json").read_text(encoding="utf-8"))
        job = Job(**metadata["job"])
        with xr.open_zarr(path, consolidated=True) as tile:
            check_tile_axes(tile, fields, job, manifest["plan"]["cadence"])
            # Recompute from actual arrays, not a possibly stale per-source mask.
            support = None
            for name in fields:
                valid = np.isfinite(tile[name])
                if "time" in valid.dims:
                    valid = valid.all("time")
                support = valid if support is None else support & valid
            support = support.compute(scheduler="synchronous").values
        row = round((job.lat_edges[0]-grid["lat_min"])/r)
        col = round((job.lon_edges[0]-grid["lon_min"])/r)
        selection = np.s_[row:row+support.shape[0], col:col+support.shape[1]]
        common[selection] &= support
        visits[selection] += 1
    if not visits[geographic].all() or not np.all(visits[geographic] == visits[geographic].flat[0]):
        raise ValueError("Spatial tiles do not cover the same complete temporal request")
    count, total = int(common.sum()), int(common.size)
    if not count:
        raise ValueError("No coordinate has ALL required features for ALL requested times")
    domain = xr.Dataset({"model_domain": (("lat", "lon"), common)}, coords={"lat": lat, "lon": lon},
        attrs={"fingerprint": manifest["fingerprint"], "feature_names": json.dumps(fields),
               "rule": "all_features_finite_for_all_requested_times", "complete_cells": count,
               "excluded_cells": total-count})
    path = Path(root)/"model_domain.zarr"
    if path.exists():
        with xr.open_zarr(path, consolidated=True) as old:
            if old.attrs.get("fingerprint") != manifest["fingerprint"] or not np.array_equal(old.model_domain.values, common):
                raise ValueError("Existing model domain differs from the recomputed contract")
    else:
        from .pipeline import atomic_replace
        import tempfile
        with tempfile.TemporaryDirectory(prefix="domain_", suffix=".partial", dir=root) as temporary:
            stage = Path(temporary)/"model_domain.zarr"
            domain.to_zarr(stage, mode="w", consolidated=True, zarr_format=2)
            atomic_replace(stage, path)
    return {"path": path.name, "complete_cells": count, "excluded_cells": total-count,
            "feature_names": fields, "rule": domain.attrs["rule"]}


def apply_domain(ds, root, manifest):
    if "model_domain" not in manifest:
        raise ValueError("Run has no fixed model domain; rerun ingestion to finalize its feature contract")
    with xr.open_zarr(Path(root)/manifest["model_domain"]["path"], consolidated=True) as saved:
        domain = saved.model_domain.load()
    if not np.isin(ds.lat.values, domain.lat.values).all() or not np.isin(ds.lon.values, domain.lon.values).all():
        raise ValueError("Stored coordinates differ from the official common domain grid")
    # Geometrically skipped rows/columns are restored only as uniformly masked
    # cells. No source data is extrapolated into those locations.
    ds = ds.reindex(lat=domain.lat, lon=domain.lon)
    for name in manifest["plan"]["feature_names"]:
        ds[name] = ds[name].where(domain)
        ds[name+"__valid"] = domain.broadcast_like(ds[name])
    ds["common_valid"] = domain.broadcast_like(ds[manifest["plan"]["feature_names"][0]])
    ds["model_domain"] = domain
    ds.attrs["feature_contract"] = json.dumps(manifest["model_domain"])
    return ds
