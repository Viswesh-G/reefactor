"""Anonymous read-only HTTPS Zarr access with explicit transfer/decode budgets.

No filesystem cache and no bucket listings. Consolidated metadata supplies keys.
Each worker owns its session, and Dask executes its tile graph synchronously.
"""
import hashlib
import json
import math
import random
import time
from collections.abc import MutableMapping
from urllib.parse import quote

import numpy as np
import requests
import xarray as xr

from .harmonize import harmonize, normalize_coordinates
from .sources import PermanentSourceError


class HTTPZarrStore(MutableMapping):
    def __init__(self, url, execution, cache=None):
        self.url, self.execution = url.rstrip("/"), execution
        self.session, self.bytes, self.requests, self.attempts = requests.Session(), 0, 0, 0
        self.started = time.monotonic()
        self.digests = {}
        self.cache, self.cache_hits, self.cache_bytes = cache, 0, 0
        self.metadata_bytes = self._get(".zmetadata")
        self.metadata = json.loads(self.metadata_bytes)["metadata"]

    def _get(self, key):
        if self.cache is not None:
            data, hit = self.cache.get((self.url, key), lambda: self._request(key))
            if hit:
                self.cache_hits += 1
                self.cache_bytes += len(data)
                self.digests[key] = hashlib.sha256(data).hexdigest()
            return data
        return self._request(key)

    def _request(self, key):
        exe = self.execution
        limit = int(exe["max_download_mb"]*1024**2)
        for attempt in range(exe["max_attempts"]):
            self.attempts += 1
            started = time.monotonic()
            try:
                with self.session.get(self.url+"/"+quote(key, safe="/."), stream=True,
                                      timeout=(10, exe["timeout_seconds"])) as response:
                    self.requests += 1
                    if response.status_code == 404:
                        # Zarr legitimately asks for absent optional metadata/chunks.
                        raise KeyError(key)
                    if response.status_code in (400, 401, 403):
                        raise PermanentSourceError(f"Zarr HTTP {response.status_code} for {self.url}/{key}")
                    response.raise_for_status()
                    blocks = []
                    for block in response.iter_content(256*1024):
                        self.bytes += len(block)
                        if self.bytes > limit or time.monotonic()-started > exe["timeout_seconds"]:
                            raise PermanentSourceError("Remote Zarr exceeded source transfer/time budget")
                        blocks.append(block)
                    data = b"".join(blocks)
                self.digests[key] = hashlib.sha256(data).hexdigest()
                return data
            except requests.RequestException:
                if attempt == exe["max_attempts"]-1:
                    raise
                time.sleep(min(30, exe["retry_base_seconds"]*2**attempt)+random.uniform(0, 0.2))

    def __getitem__(self, key):
        if key == ".zmetadata":
            return self.metadata_bytes
        if key in self.metadata:
            return json.dumps(self.metadata[key]).encode()
        variable = key.split("/")[0]
        zarray = self.metadata.get(variable+"/.zarray")
        if zarray:
            decoded_bytes = math.prod(zarray["chunks"])*np.dtype(zarray["dtype"]).itemsize
            if decoded_bytes > self.execution["max_tile_mb"]*1024**2:
                raise PermanentSourceError("Native Zarr chunk exceeds decode budget; choose another layout")
        return self._get(key)

    def __iter__(self):
        return iter([".zmetadata", *self.metadata])

    def __len__(self):
        return len(self.metadata)+1

    def __setitem__(self, key, value):
        raise TypeError("Remote source is read-only")

    def __delitem__(self, key):
        raise TypeError("Remote source is read-only")

    def close(self):
        self.session.close()


def fetch_public_zarr(source, name, job, cfg, cache=None):
    from .sources import bounds
    version = cfg.get("versions", {}).get(name) or source["zarr_version"]
    if version != source["zarr_version"]:
        raise ValueError("Public Zarr URL version differs from requested version; refresh catalog")
    # Small reef tiles favor time-series chunks. The alternate spatial layout is
    # explicitly available for later cost-based selection, rather than silently used.
    store = HTTPZarrStore(source["zarr_urls"][0], cfg["execution"], cache=cache)
    try:
        # Backend-lazy indexing first: creating a Dask graph for the global 4-D
        # ocean can itself consume gigabytes, even if we later select one tile.
        with xr.open_zarr(store, consolidated=True, chunks=None) as remote:
            native = normalize_coordinates(remote[[f.native for f in source["features"]]])
            south, north, west, east = bounds(source, job)
            native = native.sel(lat=slice(south, north), lon=slice(west, east))
            if "time" in native.dims:
                native = native.sel(time=slice(job.start, job.end+"T23:59:59"))
            output = harmonize(native, name, job).compute(scheduler="synchronous")
        return output, {"source": name, "dataset_id": source["dataset_id"], "product": source["product"],
            "requested_version": version, "version_status": "pinned", "url": store.url,
            "download_bytes": store.bytes, "requests": store.requests, "attempts": store.attempts,
            "ram_cache_hits": store.cache_hits, "ram_cache_bytes_served": store.cache_bytes,
            "remote_chunk_sha256": store.digests, "seconds": time.monotonic()-store.started}
    finally:
        store.close()
