"""Provider adapters. Credentials come from environment variables, never manifests."""
import hashlib
import os
import random
import tempfile
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import requests
import xarray as xr

from .catalog import SOURCES
from .harmonize import harmonize

NETCDF_LOCK = threading.Lock()  # netCDF C libraries are not generally thread safe.


class PermanentSourceError(RuntimeError):
    pass


def download(url, path, execution):
    """Bounded streamed HTTP download, finite retries and per-request timeouts."""
    maximum = int(execution["max_download_mb"] * 1024**2)
    for attempt in range(1, execution["max_attempts"] + 1):
        try:
            started = time.monotonic()
            with requests.get(url, stream=True, timeout=(10, execution["timeout_seconds"])) as response:
                if response.status_code in (400, 401, 403, 404):
                    raise PermanentSourceError(f"HTTP {response.status_code}; review endpoint/credentials/request")
                response.raise_for_status()
                if int(response.headers.get("Content-Length", 0)) > maximum:
                    raise PermanentSourceError("Server response exceeds max_download_mb")
                size, digest = 0, hashlib.sha256()
                with open(path, "wb") as handle:
                    for block in response.iter_content(1024 * 256):
                        size += len(block)
                        if size > maximum or time.monotonic()-started > execution["timeout_seconds"]:
                            raise PermanentSourceError("Download exceeded byte/time budget; reduce partition")
                        digest.update(block)
                        handle.write(block)
                return {"attempts": attempt, "download_bytes": size, "raw_sha256": digest.hexdigest(),
                        "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}
        except (requests.RequestException, OSError):
            if attempt == execution["max_attempts"]:
                raise
            time.sleep(min(30, execution["retry_base_seconds"] * 2**(attempt-1)) + random.uniform(0, 0.2))


def preflight(cfg, names):
    for name in names:
        backend = SOURCES[name]["backend"]
        if backend == "copernicus" and not SOURCES[name].get("zarr_urls") and not (os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") and
                                             os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")):
            raise PermanentSourceError("Copernicus requires free account credentials in COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD")
        if backend == "cds" and not os.environ.get("CDSAPI_KEY"):
            raise PermanentSourceError("ERA5 CDS requires a free account, accepted licence, and CDSAPI_KEY")


def bounds(source, job):
    halo = source["resolution"] * 1.1
    return (max(source["lat_bounds"][0], job.lat_edges[0] - halo),
            min(source["lat_bounds"][1], job.lat_edges[1] + halo),
            max(-180, job.lon_edges[0] - halo), min(179.99999, job.lon_edges[1] + halo))


def erddap_url(source, job):
    south, north, west, east = bounds(source, job)
    dimensions = (f"[({job.start}T00:00:00Z):({job.end}T23:59:59Z)]"
                  f"[({south:.6f}):({north:.6f})][({west:.6f}):({east:.6f})]")
    return source["url"] + ".nc?" + ",".join(f.native + dimensions for f in source["features"])


def _read_netcdf(path, max_bytes):
    # Only a bounded request is loaded. The whole archive is never materialized.
    with NETCDF_LOCK, xr.open_dataset(path) as remote:
        if remote.nbytes > max_bytes:
            raise PermanentSourceError("Decoded NetCDF exceeds memory budget; reduce partition")
        return remote.load()


def fetch_source(name, job, cfg, scratch, cache=None):
    source, exe = SOURCES[name], cfg["execution"]
    started = time.monotonic()
    version = cfg.get("versions", {}).get(name) or source["version"]
    provenance = {"source": name, "product": source["product"], "dataset_id": source["dataset_id"],
                  "requested_version": version, "version_status": "pinned" if version else "unresolved",
                  "download_bytes": 0, "attempts": 1}
    south, north, west, east = bounds(source, job)
    if source["backend"] == "erddap":
        path = Path(scratch) / f"{name}.nc"
        provenance.update(download(erddap_url(source, job), path, exe))
        try:
            native = _read_netcdf(path, exe["max_tile_mb"]*1024**2)
        finally:
            path.unlink(missing_ok=True)
        output = harmonize(native, name, job)
    elif source["backend"] == "copernicus":
        if source.get("zarr_urls"):
            from .remote_zarr import fetch_public_zarr
            return fetch_public_zarr(source, name, job, cfg, cache=cache)
        import copernicusmarine
        arguments = dict(dataset_id=source["dataset_id"], dataset_version=version,
            variables=[f.native for f in source["features"]], minimum_latitude=south,
            maximum_latitude=north, minimum_longitude=west, maximum_longitude=east,
            chunk_size_limit=16, coordinates_selection_method="outside")
        if source["cadence"] != "static":
            arguments.update(start_datetime=job.start, end_datetime=job.end + "T23:59:59")
            arguments.update(minimum_depth=0, maximum_depth=1)
        with copernicusmarine.open_dataset(**arguments) as native:
            # Work remains lazy through subsetting and regridding; one tile is computed.
            output = harmonize(native, name, job).compute(scheduler="synchronous")
            provenance["remote_attrs"] = {k: str(v) for k, v in native.attrs.items()}
            provenance["download_bytes"] = None  # SDK throughput is not measured here.
    elif source["backend"] == "cds":
        import cdsapi
        # Include next-day 00 UTC: hourly means describe the preceding hour.
        first, last = date.fromisoformat(job.start), date.fromisoformat(job.end) + timedelta(days=1)
        groups = {}
        current = first
        while current <= last:
            groups.setdefault((current.year, current.month), []).append(current.day)
            current += timedelta(days=1)
        pieces = []
        client = cdsapi.Client(url=os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api"),
                               key=os.environ["CDSAPI_KEY"], timeout=exe["timeout_seconds"],
                               retry_max=exe["max_attempts"], quiet=True)
        for (year, month), days in groups.items():
            request = {"product_type": ["reanalysis"], "variable": [f.native for f in source["features"]],
                "year": [str(year)], "month": [f"{month:02}"], "day": [f"{d:02}" for d in days],
                "time": [f"{h:02}:00" for h in range(24)], "area": [north, west, south, east],
                "data_format": "netcdf", "download_format": "unarchived"}
            path = Path(scratch) / f"era5_{year}_{month}.nc"
            # The CDS SDK queues jobs synchronously; its service wait can exceed HTTP timeout.
            result = client.retrieve(source["dataset_id"], request)
            provenance["attempts"] = None  # Provider SDK manages these retries.
            provenance.setdefault("downloads", []).append(download(result.location, path, exe))
            try:
                pieces.append(_read_netcdf(path, exe["max_tile_mb"]*1024**2))
            finally:
                path.unlink(missing_ok=True)
        native = xr.concat(pieces, dim="valid_time" if "valid_time" in pieces[0].dims else "time")
        output = harmonize(native, name, job)
        provenance["download_bytes"] = sum(p["download_bytes"] for p in provenance["downloads"])
    elif source["backend"] == "opendap":
        from .opendap import fetch_opendap
        return fetch_opendap(source, name, job, cfg)
    else:
        raise ValueError(f"Unsupported backend {source['backend']}")
    output = output.compute(scheduler="synchronous")
    provenance["seconds"] = time.monotonic() - started
    return output, provenance


def fetch_job(job, cfg, names, scratch_root, cache=None):
    parts, provenance = [], []
    with tempfile.TemporaryDirectory(prefix=f"{job.key}_", dir=scratch_root) as scratch:
        for name in names:
            part, details = fetch_source(name, job, cfg, scratch, cache=cache)
            parts.append(part)
            provenance.append(details)
        return xr.merge(parts, join="exact", compat="no_conflicts"), provenance
