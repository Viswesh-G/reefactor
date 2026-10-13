"""Strict real-data ingestion FSM — no synthetic fallbacks.

States: INIT -> CHECK_COVERAGE -> INGESTING -> RESOLVING -> VALIDATING -> STORING -> READY (ERROR on failure).

Principles:
  - REAL DATA ONLY: any source that cannot be fetched after all retries -> RuntimeError.
    No random/synthetic generation anywhere (publishability requirement).
  - STORAGE-MINIMAL: server-side subsetting (ERDDAP .nc query = only GBR bbox bytes
    transferred), yearly time-chunking (no 7-yr single download), temp files deleted
    immediately, no raw archive kept, single Zarr write with float32 + compression.
  - SAME CANVAS: every variable lands on identical (time, lat, lon):
    daily-max resample -> bilinear interp to 0.05 deg GBR canvas -> common time reindex.
"""

import gc
import os
import shutil
import time as _time

import numpy as np
import xarray as xr

from src.config_validator import ConfigValidator
from src.grid import build_axis


class PipelineState:
    INIT = "INIT"
    CHECK_COVERAGE = "CHECK_COVERAGE"
    INGESTING = "INGESTING"
    RESOLVING = "RESOLVING"
    VALIDATING = "VALIDATING"
    STORING = "STORING"
    READY = "READY"
    ERROR = "ERROR"


LAG_STRATEGY_MAP = {"max": "max", "min": "min", "avg": "mean"}
LAG_AGGREGATION_MAP = {
    "max": "max", "min": "min", "avg": "mean", "mean": "mean",
    "last": "last", "first": "first",
}
LAG_CYCLE_MAP = {"daily": "1D", "weekly": "7D", "monthly": "MS", "annually": "YS"}

# Defaults if pipeline block missing (legacy). Canonical values live in
# config.yaml >> pipeline.ingestion / pipeline.validation.
_DEFAULTS = {"chunk_years": 1, "max_retries": 4, "retry_backoff_s": 5.0,
             "fetch_timeout_s": 600, "max_nan_fraction": 0.05,
             "zarr_chunk_time": 30, "pacing_s": 0.0}


class DataIngestionFSM:
    """Strict FSM: real data only, one shared spatio-temporal canvas."""

    def __init__(self, config):
        self.state = PipelineState.INIT
        self.config = config
        self.data_store = {}   # name -> xr.Dataset (native grid, chunked fetch)
        self.provenance = {}   # name -> dict(source_url, time_range, bytes, attempts)
        pipe = config.get("pipeline", {})
        ing, val = pipe.get("ingestion", {}), pipe.get("validation", {})
        # All tunables from config — no hardcoded pipeline params.
        self.max_retries = int(ing.get("max_retries", _DEFAULTS["max_retries"]))
        self.retry_backoff_s = float(ing.get("retry_backoff_s",
                                             _DEFAULTS["retry_backoff_s"]))
        self.fetch_timeout_s = float(ing.get("fetch_timeout_s",
                                             _DEFAULTS["fetch_timeout_s"]))
        self.fetch_order = ing.get("fetch_order", ["erddap_nc", "pydap"])
        # Gentle pause between chunk requests (be kind to flaky public mirrors).
        self.pacing_s = float(ing.get("pacing_s", _DEFAULTS["pacing_s"]))
        self.chunk_years = int(ing.get("chunk_years", 1))
        # Monthly chunks (~30MB/req) — yearly requests time out on ERDDAP.
        if "chunk_months" in ing:
            self.chunk_months = max(1, int(ing["chunk_months"]))
        elif "chunk_years" in ing:
            self.chunk_months = max(1, int(ing["chunk_years"]) * 12)
        else:
            self.chunk_months = 1
        self.zarr_chunk_time = int(ing.get("zarr_chunk_time",
                                           _DEFAULTS["zarr_chunk_time"]))
        self.overwrite_zarr = bool(ing.get("overwrite_zarr", True))
        self.aggregation = dict(
            ing.get("aggregation", {
                "sea_surface_temperature": "max",
                "degree_heating_weeks": "last",
                "water_u": "mean",
                "water_v": "mean",
            })
        )
        # Resumable ingestion: cache each fetched monthly chunk to disk so a
        # re-run after a flaky-mirror failure skips chunks already fetched.
        self.use_chunk_cache = bool(ing.get("use_chunk_cache", True))
        self.raw_cache_dir = str(
            self.config.get("storage", {}).get("raw_cache_dir", "data/raw_cache"))
        self.max_nan_fraction = float(val.get("max_nan_fraction",
                                              _DEFAULTS["max_nan_fraction"]))
        self.sst_min = float(val.get("sst_min", -2.0))
        self.sst_max = float(val.get("sst_max", 40.0))
        self.current_max = float(val.get("current_max", 3.0))

    # ── driver ──────────────────────────────────────────────
    def run(self):
        print("=== Starting Data Ingestion Pipeline (strict, real-only) ===")
        try:
            self.transition_to(PipelineState.CHECK_COVERAGE)
            self._check_coverage()
            self.transition_to(PipelineState.INGESTING)
            self._ingest_data()
            self.transition_to(PipelineState.RESOLVING)
            self._resolve_data()
            self.transition_to(PipelineState.VALIDATING)
            self._validate_data()
            self.transition_to(PipelineState.STORING)
            self._store_data()
            self.transition_to(PipelineState.READY)
            print("=== Pipeline Finished Successfully! ===")
        except Exception as e:
            self.transition_to(PipelineState.ERROR)
            print(f"PIPELINE FAILED (no synthetic fallback by design): {e}")
            raise

    def transition_to(self, new_state):
        print(f"  [{self.state}] ---> [{new_state}]")
        self.state = new_state

    # ── CHECK_COVERAGE ──────────────────────────────────────
    def _check_coverage(self):
        cfg = self.config
        if "common_coverage" not in cfg:
            print("  -> No coverage blocks in config; skipping coverage gate.")
            return
        print("  -> Coverage gate: requested canvas must be inside maximal, "
              "ideally inside common (all-vars-real). See ConfigValidator.")
        # Validator already enforces; here we just log the decision.
        print(f"     requested: {cfg['temporal_canvas']['start_date']}.."
              f"{cfg['temporal_canvas']['end_date']}")
        print(f"     common (minimal): {cfg['common_coverage']['start_date']}.."
              f"{cfg['common_coverage']['end_date']}")
        print(f"     maximal (envelope): {cfg['maximal_coverage']['start_date']}.."
              f"{cfg['maximal_coverage']['end_date']}")

    # ── INGESTING ───────────────────────────────────────────
    def fetch_variables(self, opendap_url, variables, t_start, t_end):
        """Public helper: fetch one bbox/time window for given variables.

        Reuses the same ERDDAP .nc query + retry logic as the main ingestion.
        Handy for small sidecar pulls (e.g. deriving the CRW MMM climatology).
        """
        canvas = self.config["spatial_canvas"]
        nc_url = self._build_erddap_nc_url(
            opendap_url, variables, t_start, t_end,
            canvas["lat_min"], canvas["lat_max"], canvas["lon_min"], canvas["lon_max"])
        ds, _att = self._fetch_nc_chunk(nc_url)
        if "latitude" in ds.dims:
            ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        return ds

    def _build_erddap_nc_url(self, opendap_url, variables,
                              t_start, t_end, lat_min, lat_max, lon_min, lon_max):
        host = opendap_url.split("/erddap/")[0]
        dataset_id = opendap_url.rstrip("/").split("/")[-1]
        var_parts = ",".join(
            f"{v}[({t_start}):({t_end})]"
            f"[({lat_min:.4f}):({lat_max:.4f})]"
            f"[({lon_min:.4f}):({lon_max:.4f})]"
            for v in variables
        )
        return f"{host}/erddap/griddap/{dataset_id}.nc?{var_parts}"

    def _chunk_generator(self, start, end):
        import datetime as _dt
        import calendar
        s_date = _dt.datetime.strptime(start, "%Y-%m-%d").date()
        e_date = _dt.datetime.strptime(end, "%Y-%m-%d").date()
        cur = s_date
        step_m = getattr(self, "chunk_months", 1)
        while cur <= e_date:
            m = cur.month + step_m - 1
            y = cur.year + (m - 1) // 12
            m = (m - 1) % 12 + 1
            _, last_day = calendar.monthrange(y, m)
            chunk_end = _dt.date(y, m, last_day)
            if chunk_end > e_date:
                chunk_end = e_date
            yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
            cur = chunk_end + _dt.timedelta(days=1)

    def _year_chunks(self, start, end):
        return self._chunk_generator(start, end)

    def _chunk_cache_path(self, name, cs, ce):
        """Disk path for a cached raw chunk (one file per source x month window)."""
        d = os.path.join(self.raw_cache_dir, name)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{cs}__{ce}.nc")

    def _fetch_nc_chunk(self, nc_url, timeout=None):
        import os as _os
        import tempfile
        import urllib.request
        import socket
        timeout = timeout or self.fetch_timeout_s
        socket.setdefaulttimeout(timeout)
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    urllib.request.urlretrieve(nc_url, tmp_path)
                    # open + load then release file handle immediately (storage-minimal)
                    with xr.open_dataset(tmp_path) as ds:
                        out = ds.load()
                finally:
                    if _os.path.exists(tmp_path):
                        _os.unlink(tmp_path)
                return out, attempt
            except Exception as e:
                last_err = e
                print(f"       attempt {attempt}/{self.max_retries} failed "
                      f"({type(e).__name__}: {str(e)[:100]})")
                if attempt < self.max_retries:
                    _time.sleep(self.retry_backoff_s * attempt)
        raise RuntimeError(f"ERDDAP .nc fetch failed after {self.max_retries}: {last_err}")

    def _fetch_pydap(self, opendap_url, variables, drop_dims,
                     t_start, t_end, lat_min, lat_max, lon_min, lon_max):
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                pydap_url = opendap_url.replace("https://", "dap4://", 1).replace(
                    "http://", "dap4://", 1)
                try:
                    ds_remote = xr.open_dataset(pydap_url, engine="pydap", chunks={})
                except Exception:
                    ds_remote = xr.open_dataset(opendap_url, engine="pydap", chunks={})
                if "latitude" in ds_remote.dims:
                    ds_remote = ds_remote.rename({"latitude": "lat", "longitude": "lon"})
                avail = [v for v in variables if v in ds_remote.data_vars]
                if not avail:
                    raise ValueError(f"None of {variables} in {list(ds_remote.data_vars)}")
                ds_remote = ds_remote[avail]
                for dim in drop_dims:
                    if dim in ds_remote.dims:
                        ds_remote = ds_remote.isel({dim: 0}, drop=True)
                lat_slice = (slice(lat_max, lat_min) if ds_remote.lat.values[0] >
                             ds_remote.lat.values[-1] else slice(lat_min, lat_max))
                sub = ds_remote.sel(time=slice(t_start, t_end),
                                    lat=lat_slice, lon=slice(lon_min, lon_max))
                if len(sub.time) == 0:
                    raise ValueError("Time slice returned 0 records.")
                return sub.load(), attempt
            except Exception as e:
                last_err = e
                print(f"       pydap attempt {attempt}/{self.max_retries} failed "
                      f"({type(e).__name__}: {str(e)[:100]})")
                if attempt < self.max_retries:
                    _time.sleep(self.retry_backoff_s * attempt)
        raise RuntimeError(f"pydap fetch failed after {self.max_retries}: {last_err}")

    def _ingest_one_source(self, name, info, canvas):
        opendap_url = info.get("opendap_url")
        if not opendap_url:
            raise RuntimeError(f"No opendap_url for '{name}'. Real-data-only: refusing synthetic.")
        if self.config.get("pipeline", {}).get("allow_synthetic", False) is not False:
            raise RuntimeError("pipeline.allow_synthetic must be false (strict real-only).")
        # Candidate mirrors: primary first, then fallbacks (order preserved, dups dropped).
        mirrors = list(dict.fromkeys(m for m in (info.get("opendap_mirrors") or [opendap_url]) if m))
        variables = info.get("variables", [])
        rename_map = info.get("rename_vars", {})
        drop_dims = info.get("drop_dims", [])
        lat_min, lat_max = canvas["lat_min"], canvas["lat_max"]
        lon_min, lon_max = canvas["lon_min"], canvas["lon_max"]
        t_canvas = self.config["temporal_canvas"]

        parts, total_attempts, bytes_est = [], 0, 0
        mirrors_used = set()
        for cs, ce in self._year_chunks(t_canvas["start_date"], t_canvas["end_date"]):
            ds_c, att, used_mirror = None, 0, None
            last_err = None
            cache_path = self._chunk_cache_path(name, cs, ce) if self.use_chunk_cache else None
            # 1) Serve from disk cache if present (resume after a prior failure).
            if cache_path and os.path.exists(cache_path):
                try:
                    with xr.open_dataset(cache_path) as cds:
                        ds_c = cds.load()
                    used_mirror = "cache"
                    print(f"     [cache] {name} {cs}..{ce}")
                except Exception as e:
                    print(f"     [cache] {name} {cs}..{ce} unreadable "
                          f"({str(e)[:60]}) — refetching")
                    ds_c = None
            # 2) Otherwise fetch over the network: try each method, each mirror.
            if ds_c is None:
                for method in self.fetch_order:
                    for mirror in mirrors:
                        try:
                            if method == "erddap_nc":
                                nc_url = self._build_erddap_nc_url(
                                    mirror, variables, cs, ce, lat_min, lat_max,
                                    lon_min, lon_max)
                                print(f"     [nc] {name} {cs}..{ce} <- {mirror.split('/')[2]}")
                                ds_c, att = self._fetch_nc_chunk(nc_url)
                            elif method == "pydap":
                                print(f"     [pydap] {name} {cs}..{ce} <- {mirror.split('/')[2]}")
                                ds_c, att = self._fetch_pydap(
                                    mirror, variables, drop_dims, cs, ce,
                                    lat_min, lat_max, lon_min, lon_max)
                            else:
                                raise ValueError(f"Unknown fetch method: {method}")
                            used_mirror = mirror
                            break  # success
                        except Exception as e:
                            last_err = e
                            print(f"     [{method}@{mirror.split('/')[2]}] chunk failed "
                                  f"({str(e)[:70]})")
                    if ds_c is not None:
                        break
                # Persist the freshly fetched chunk so future runs can resume.
                if ds_c is not None and cache_path:
                    try:
                        ds_c.to_netcdf(cache_path)
                    except Exception as e:
                        print(f"     [cache] could not write {cs}..{ce} ({str(e)[:60]})")
            if ds_c is None:
                raise RuntimeError(f"[{name}] chunk {cs}..{ce} failed via "
                                   f"{self.fetch_order} on {mirrors}: {last_err}. "
                                   f"No synthetic fallback.")
            total_attempts += att
            if used_mirror and used_mirror != "cache":
                mirrors_used.add(used_mirror)
            if "latitude" in ds_c.dims:
                ds_c = ds_c.rename({"latitude": "lat", "longitude": "lon"})
            ren = {k: v for k, v in rename_map.items() if k in ds_c.data_vars}
            if ren:
                ds_c = ds_c.rename(ren)
            bytes_est += int(ds_c.nbytes)
            parts.append(ds_c)
            gc.collect()  # keep peak RAM flat across years
            if self.pacing_s > 0:
                _time.sleep(self.pacing_s)  # pacing between chunks

        ds_all = xr.concat(sorted(parts, key=lambda d: d.time.values[0]),
                           dim="time").sortby("time")
        for p in parts:
            p.close()
        del parts
        gc.collect()
        self.provenance[name] = {"url": opendap_url, "mirrors": sorted(mirrors_used),
                                 "variables": variables,
                                 "attempts": total_attempts, "bytes": bytes_est,
                                 "time_first": str(ds_all.time.values[0]),
                                 "time_last": str(ds_all.time.values[-1])}
        print(f"     [{name}] OK: {len(ds_all.time)} steps, ~{bytes_est/1e6:.1f} MB, "
              f"{total_attempts} attempts total.")
        return ds_all

    def _ingest_data(self):
        canvas = self.config["spatial_canvas"]
        for name, info in self.config["data_sources"].items():
            print(f"  -> Fetching '{name}': {info.get('opendap_url')}")
            # Raises on failure — NO synthetic fallback by design.
            self.data_store[name] = self._ingest_one_source(name, info, canvas)
            print(f"     Variables: {list(self.data_store[name].data_vars)}")

    # ── RESOLVING (single shared canvas) ────────────────────
    def _resolve_data(self):
        canvas = self.config["spatial_canvas"]
        t_canvas = self.config["temporal_canvas"]
        target_lat = build_axis(canvas["lat_min"], canvas["lat_max"],
                                canvas["target_resolution"])
        target_lon = build_axis(canvas["lon_min"], canvas["lon_max"],
                                canvas["target_resolution"])
        freq = LAG_CYCLE_MAP[t_canvas["time_lag_cycle"]]
        method = LAG_STRATEGY_MAP[t_canvas["lag_strategy"]]

        # Common daily time index: every source reindexed here -> identical time axis.
        t0 = np.datetime64(t_canvas["start_date"])
        t1 = np.datetime64(t_canvas["end_date"])
        common_time = np.arange(t0, t1 + np.timedelta64(1, "D"),
                                np.timedelta64(1, "D"))
        print(f"  -> Common canvas: {len(common_time)} days, "
              f"{len(target_lat)}x{len(target_lon)} @ {canvas['target_resolution']} deg")

        for name, ds in self.data_store.items():
            print(f"  -> [{name}] daily-{method} resample")
            resampled = {}
            for var in ds.data_vars:
                aggregation = self.aggregation.get(var, method)
                if aggregation not in LAG_AGGREGATION_MAP:
                    raise ValueError(
                        f"Unsupported aggregation '{aggregation}' for variable '{var}'"
                    )
                reducer = LAG_AGGREGATION_MAP[aggregation]
                resampled[var] = getattr(ds[var].resample(time=freq), reducer)()
            ds = xr.Dataset(resampled, attrs=ds.attrs)
            print(f"  -> [{name}] bilinear interp to unified 0.05 deg canvas")
            ds = ds.interp(lat=target_lat, lon=target_lon, method="linear",
                           kwargs={"fill_value": np.nan})
            # Identical time axis for all vars (NaN where a source truly missing).
            ds = ds.reindex(time=common_time)
            # Storage-minimal dtype: float32 is plenty for SST/currents.
            for v in ds.data_vars:
                if ds[v].dtype == np.float64:
                    ds[v] = ds[v].astype(np.float32)
            self.data_store[name] = ds
        gc.collect()

    # ── VALIDATING (fail-fast, no silent fill of real gaps) ─
    def _validate_data(self):
        for name, ds in self.data_store.items():
            # Time continuity: no missing days on the common axis.
            n_expected = len(ds.time)
            if n_expected != len(np.unique(ds.time.values)):
                raise RuntimeError(f"[{name}] duplicate timestamps after resolving.")
            if int(ds.time.isnull().sum().values):
                raise RuntimeError(f"[{name}] null timestamps on common axis.")
            print(f"  -> [{name}] time steps: {n_expected} (continuous)")
            for var in ds.data_vars:
                arr = ds[var]
                # Distinguish PERMANENT NaN (land pixels, NaN on every timestep)
                # from TRANSIENT NaN (real data gaps). The GBR bbox legitimately
                # contains ~35% land; counting land as "missing data" is wrong.
                static_nan = arr.isnull().all("time")          # (lat, lon) land mask
                ever_nan   = arr.isnull().any("time")
                total_frac = float(arr.isnull().mean().values)
                land_frac  = float(static_nan.mean().values)
                gap_frac   = float((ever_nan & ~static_nan).mean().values)
                print(f"  -> [{name}/{var}] NaN: {total_frac*100:.2f}% "
                      f"(land {land_frac*100:.2f}%, transient gaps {gap_frac*100:.2f}%)")
                if gap_frac > self.max_nan_fraction:
                    raise RuntimeError(
                        f"[{name}/{var}] transient gap fraction {gap_frac:.3f} > "
                        f"{self.max_nan_fraction}. Real-data-only: refusing to "
                        f"interpolate large gaps. Check ERDDAP window.")
                if gap_frac > 0:
                    # Fill only the small transient gaps, then re-assert the land
                    # mask so we never invent ocean values over land pixels.
                    filled = (arr.interpolate_na(dim="lon", method="nearest")
                                 .interpolate_na(dim="lat", method="nearest"))
                    ds[var] = filled.where(~static_nan)
            if "sea_surface_temperature" in ds.data_vars:
                smin = float(ds["sea_surface_temperature"].min().values)
                smax = float(ds["sea_surface_temperature"].max().values)
                print(f"  -> SST range: {smin:.2f}..{smax:.2f} C")
                if smin < self.sst_min or smax > self.sst_max:
                    raise RuntimeError(
                        f"SST outside physical bounds [{self.sst_min},{self.sst_max}].")
            if "water_u" in ds.data_vars:
                umax = max(float(abs(ds["water_u"]).max().values),
                           float(abs(ds["water_v"]).max().values))
                print(f"  -> |current| max: {umax:.2f} m/s")
                if umax > self.current_max:
                    raise RuntimeError(
                        f"Geostrophic current >{self.current_max} m/s implausible.")
            self.data_store[name] = ds

    # ── STORING (one Zarr, compressed, provenance attrs) ────
    def _store_data(self):
        out_path = self.config["storage"]["processed_zarr_path"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        combined = xr.merge(list(self.data_store.values()))
        # Time-major chunks match training reads (14+7d windows scan time).
        combined = combined.chunk({"time": self.zarr_chunk_time, "lat": -1, "lon": -1})
        combined.attrs["ingestion"] = "strict real-only, no synthetic"
        combined.attrs["common_coverage"] = str(self.config.get("common_coverage", ""))
        combined.attrs["provenance"] = str(self.provenance)
        print(f"  -> Writing Zarr: {out_path} vars={list(combined.data_vars)} "
              f"shape={dict(combined.sizes)}")
        if os.path.exists(out_path):
            if not self.overwrite_zarr:
                raise RuntimeError(f"Zarr exists at {out_path} and overwrite_zarr=false.")
            shutil.rmtree(out_path)
        # zarr v3 (this env) uses default zstd compressors; do NOT pass numcodecs
        # Blosc here (v2 codec -> TypeError on v3). dtype-only encoding is portable.
        enc = {v: {"dtype": "float32"} for v in combined.data_vars}
        combined.to_zarr(out_path, mode="w", consolidated=True, encoding=enc)
        print("  -> Zarr written. Temp .nc files were deleted inline; no raw archive kept.")
        # Storage-minimal: drop the resumable chunk cache once the Zarr is safely
        # written. (If the store had failed, the cache would have been kept.)
        if self.use_chunk_cache and self.config.get("storage", {}).get(
                "delete_raw_after_processing", True):
            cache_root = os.path.join(self.raw_cache_dir)
            if os.path.isdir(cache_root):
                shutil.rmtree(cache_root, ignore_errors=True)
                print("  -> Removed raw chunk cache (storage-minimal).")
        # Free ingested data immediately (storage-minimal peak RAM).
        self.data_store.clear()
        gc.collect()


if __name__ == "__main__":
    validator = ConfigValidator("../config.yaml")
    valid_config = validator.validate()
    pipeline = DataIngestionFSM(valid_config)
    pipeline.run()
