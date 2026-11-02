"""Bounded, timeout-controlled OPeNDAP reads for public rectilinear GBR products."""
import hashlib
import time
from urllib.parse import urlsplit
from http import HTTPStatus

import requests
import xarray as xr

from .harmonize import harmonize, normalize_coordinates
from .sources import PermanentSourceError, bounds


class BoundedSession(requests.Session):
    def __init__(self, execution):
        super().__init__()
        self.execution, self.bytes, self.requests, self.attempts = execution, 0, 0, 0
        self.hashes = {}
        self.started = time.monotonic()

    def send(self, request, **kwargs):
        kwargs["stream"] = True
        maximum = self.execution["max_download_mb"]*1024**2
        for attempt in range(min(2, self.execution["max_attempts"])):
            remaining = self.execution.get("source_deadline_seconds", 180)-(time.monotonic()-self.started)
            if remaining <= 0:
                raise PermanentSourceError("OPeNDAP exceeded the overall source deadline")
            kwargs["timeout"] = min(float(self.execution["timeout_seconds"]), 60, remaining)
            self.attempts += 1
            started = time.monotonic()
            try:
                response = super().send(request, **kwargs)
                self.requests += 1
                if self.requests % 20 == 0:
                    print(f"  eReefs: {self.requests} responses, {self.bytes/1024**2:.2f} MiB", flush=True)
                if response.status_code in (400, 401, 403, 404):
                    response.close()
                    raise PermanentSourceError(f"OPeNDAP HTTP {response.status_code}")
                response.raise_for_status()
                content = []
                for block in response.iter_content(256*1024):
                    self.bytes += len(block)
                    if (self.bytes > maximum or time.monotonic()-started > self.execution["timeout_seconds"] or
                            time.monotonic()-self.started > self.execution.get("source_deadline_seconds", 180)):
                        response.close()
                        raise PermanentSourceError("OPeNDAP source exceeded transfer/time budget")
                    content.append(block)
                response._content = b"".join(content)
                response._content_consumed = True
                self.hashes[request.url] = hashlib.sha256(response.content).hexdigest()
                return response
            except requests.RequestException:
                if attempt+1 == min(2, self.execution["max_attempts"]):
                    raise
                time.sleep(1)


def fetch_opendap(source, name, job, cfg):
    requested = cfg.get("versions", {}).get(name) or source["version"]
    if requested != source["version"]:
        raise PermanentSourceError("Requested eReefs version differs from the verified endpoint")
    started = time.monotonic()
    with BoundedSession(cfg["execution"]) as session:
        from pydap.client import open_url
        origin = urlsplit(source["url"])
        def transport(environ, start_response):
            # PyDAP 3.5 clones HTTP sessions, dropping custom byte/deadline guards.
            # Its supported WSGI transport routes EVERY request through our owned
            # bounded session without global monkeypatches or a local web server.
            url = f"{origin.scheme}://{origin.netloc}"+environ["PATH_INFO"]
            if environ.get("QUERY_STRING"):
                url += "?"+environ["QUERY_STRING"]
            response = session.get(url)
            start_response(f"{response.status_code} {HTTPStatus(response.status_code).phrase}",
                [("Content-Type", response.headers.get("Content-Type", "application/octet-stream")),
                 ("Content-Length", str(len(response.content)))])
            return [response.content]
        dataset = open_url(source["url"], application=transport, output_grid=False, protocol="dap2", use_cache=False)
        with xr.open_dataset(xr.backends.PydapDataStore(dataset), chunks=None) as remote:
            native = normalize_coordinates(remote[[f.native for f in source["features"]]])
            south, north, west, east = bounds(source, job)
            native = native.sel(lat=slice(south, north), lon=slice(west, east),
                                time=slice(job.start, job.end+"T23:59:59"))
            if native.nbytes > cfg["execution"]["max_tile_mb"]*1024**2:
                raise PermanentSourceError("Decoded OPeNDAP subset exceeds memory budget")
            # Read one bounded subset before conservative area operations. No
            # full-domain Dask graph and no NetCDF files or sqlite request cache.
            native.load()
            output = harmonize(native, name, job)
        provenance = {"source": name, "url": source["url"], "product": source["product"],
            "requested_version": source["version"], "version_status": "pinned_model_run",
            "temporal_statistic": source["temporal_statistic"], "download_bytes": session.bytes,
            "requests": session.requests, "attempts": session.attempts,
            "response_sha256": session.hashes, "seconds": time.monotonic()-started}
    return output, provenance
