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
LAG_CYCLE_MAP = {"daily": "1D", "weekly": "7D", "monthly": "MS", "annually": "YS"}

# Yearly chunks keep each ERDDAP request small (timeouts were the #1 failure).
CHUNK_YEARS = 1
MAX_RETRIES = 4
RETRY_BACKOFF_S = 5.0
# Fail fast instead of silently training on garbage.
MAX_NAN_FRACTION = 0.05


class DataIngestionFSM:
    """Strict FSM: real data only, one shared spatio-temporal canvas."""

    def __init__(self, config):
        self.state = PipelineState.INIT
        self.config = config
        self.data_store = {}   # name -> xr.Dataset (native grid, chunked fetch)
        self.provenance = {}   # name -> dict(source_url, time_range, bytes, attempts)

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

    def _year_chunks(self, start, end):
        sy, ey = int(start[:4]), int(end[:4])
        for y in range(sy, ey + 1):
            yield (f"{y}-01-01" if y > sy else start,
                   f"{y}-12-31" if y < ey else end)

    def _fetch_nc_chunk(self, nc_url, timeout=600):
        import os as _os
        import tempfile
        import urllib.request
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
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
                print(f"       attempt {attempt}/{MAX_RETRIES} failed "
                      f"({type(e).__name__}: {str(e)[:100]})")
                if attempt < MAX_RETRIES:
                    _time.sleep(RETRY_BACKOFF_S * attempt)
        raise RuntimeError(f"ERDDAP .nc fetch failed after {MAX_RETRIES}: {last_err}")

    def _fetch_pydap(self, opendap_url, variables, drop_dims,
                     t_start, t_end, lat_min, lat_max, lon_min, lon_max):
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
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
                print(f"       pydap attempt {attempt}/{MAX_RETRIES} failed "
                      f"({type(e).__name__}: {str(e)[:100]})")
                if attempt < MAX_RETRIES:
                    _time.sleep(RETRY_BACKOFF_S * attempt)
        raise RuntimeError(f"pydap fetch failed after {MAX_RETRIES}: {last_err}")

    def _ingest_one_source(self, name, info, canvas):
        opendap_url = info.get("opendap_url")
        if not opendap_url:
            raise RuntimeError(f"No opendap_url for '{name}'. Real-data-only: refusing synthetic.")
        variables = info.get("variables", [])
        rename_map = info.get("rename_vars", {})
        drop_dims = info.get("drop_dims", [])
        lat_min, lat_max = canvas["lat_min"], canvas["lat_max"]
        lon_min, lon_max = canvas["lon_min"], canvas["lon_max"]
        t_canvas = self.config["temporal_canvas"]

        parts, total_attempts, bytes_est = [], 0, 0
        for cs, ce in self._year_chunks(t_canvas["start_date"], t_canvas["end_date"]):
            nc_url = self._build_erddap_nc_url(
                opendap_url, variables, cs, ce, lat_min, lat_max, lon_min, lon_max)
            try:
                print(f"     [nc] {name} {cs}..{ce}")
                ds_c, att = self._fetch_nc_chunk(nc_url)
            except Exception as e1:
                print(f"     [nc] failed ({str(e1)[:80]}), trying pydap...")
                ds_c, att = self._fetch_pydap(
                    opendap_url, variables, drop_dims, cs, ce,
                    lat_min, lat_max, lon_min, lon_max)
            total_attempts += att
            if "latitude" in ds_c.dims:
                ds_c = ds_c.rename({"latitude": "lat", "longitude": "lon"})
            ren = {k: v for k, v in rename_map.items() if k in ds_c.data_vars}
            if ren:
                ds_c = ds_c.rename(ren)
            bytes_est += int(ds_c.nbytes)
            parts.append(ds_c)
            gc.collect()  # keep peak RAM flat across years

        ds_all = xr.concat(sorted(parts, key=lambda d: d.time.values[0]),
                           dim="time").sortby("time")
        for p in parts:
            p.close()
        del parts
        gc.collect()
        self.provenance[name] = {"url": opendap_url, "variables": variables,
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
        target_lat = np.arange(canvas["lat_min"], canvas["lat_max"],
                               canvas["target_resolution"])
        target_lon = np.arange(canvas["lon_min"], canvas["lon_max"],
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
            ds = getattr(ds.resample(time=freq), method)()
            print(f"  -> [{name}] bilinear interp to unified 0.05 deg canvas")
            ds = ds.interp(lat=target_lat, lon=target_lon, method="linear",
                           kwargs={"fill_value": "extrapolate"})
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
            if int(ds.time.isnull().sum().values):
                raise RuntimeError(f"[{name}] null timestamps on common axis.")
            print(f"  -> [{name}] time steps: {n_expected} (continuous)")
            for var in ds.data_vars:
                arr = ds[var]
                nan_frac = float(arr.isnull().mean().values)
                print(f"  -> [{name}/{var}] NaN: {nan_frac*100:.2f}%")
                if nan_frac > MAX_NAN_FRACTION:
                    raise RuntimeError(
                        f"[{name}/{var}] NaN fraction {nan_frac:.3f} > "
                        f"{MAX_NAN_FRACTION}. Real-data-only: refusing to "
                        f"interpolate large gaps. Check ERDDAP window.")
                if 0 < nan_frac <= MAX_NAN_FRACTION:
                    ds[var] = (arr.interpolate_na(dim="lon", method="nearest")
                                  .interpolate_na(dim="lat", method="nearest"))
            if "sea_surface_temperature" in ds.data_vars:
                smin = float(ds["sea_surface_temperature"].min().values)
                smax = float(ds["sea_surface_temperature"].max().values)
                print(f"  -> SST range: {smin:.2f}..{smax:.2f} C")
                if smin < -2.0 or smax > 40.0:
                    raise RuntimeError("SST outside physical bounds [-2,40].")
            if "water_u" in ds.data_vars:
                umax = max(float(abs(ds["water_u"]).max().values),
                           float(abs(ds["water_v"]).max().values))
                print(f"  -> |current| max: {umax:.2f} m/s")
                if umax > 3.0:
                    raise RuntimeError("Geostrophic current >3 m/s implausible.")
            self.data_store[name] = ds

    # ── STORING (one Zarr, compressed, provenance attrs) ────
    def _store_data(self):
        out_path = self.config["storage"]["processed_zarr_path"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        combined = xr.merge(list(self.data_store.values()))
        # Time-major chunks match training reads (14+7d windows scan time).
        combined = combined.chunk({"time": 30, "lat": -1, "lon": -1})
        combined.attrs["ingestion"] = "strict real-only, no synthetic"
        combined.attrs["common_coverage"] = str(self.config.get("common_coverage", ""))
        combined.attrs["provenance"] = str(self.provenance)
        print(f"  -> Writing Zarr: {out_path} vars={list(combined.data_vars)} "
              f"shape={dict(combined.sizes)}")
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        comp = {"compressor": None}  # keep default; set Blosc/zstd if available
        try:
            import numcodecs as _nc  # noqa
            comp = {"compressor": _nc.Blosc(cname="zstd", clevel=3)}
        except Exception:
            pass
        enc = {v: {"dtype": "float32", **comp} for v in combined.data_vars}
        combined.to_zarr(out_path, mode="w", consolidated=True, encoding=enc)
        print("  -> Zarr written. Temp .nc files were deleted inline; no raw archive kept.")
        # Free ingested data immediately (storage-minimal peak RAM).
        self.data_store.clear()
        gc.collect()


if __name__ == "__main__":
    validator = ConfigValidator("../config.yaml")
    valid_config = validator.validate()
    pipeline = DataIngestionFSM(valid_config)
    pipeline.run()
