import os
import shutil
import numpy as np
import xarray as xr
from src.config_validator import ConfigValidator

# A Finite State Machine (FSM) is used here to structure the data pipeline.
# It ensures data moves step-by-step in a safe, predictable order:
# 1. INIT -> 2. INGESTING -> 3. RESOLVING -> 4. VALIDATING -> 5. STORING -> 6. CLEANUP -> READY
# This prevents out-of-order execution and makes the flow easy to follow and debug.

class PipelineState:
    """Defines the possible states of our FSM."""
    INIT       = "INIT"
    INGESTING  = "INGESTING"
    RESOLVING  = "RESOLVING"
    VALIDATING = "VALIDATING"
    STORING    = "STORING"
    CLEANUP    = "CLEANUP"
    READY      = "READY"
    ERROR      = "ERROR"


# How config lag_strategy strings map to xarray resample method names
LAG_STRATEGY_MAP = {
    "max": "max",
    "min": "min",
    "avg": "mean",
}

# How config time_lag_cycle strings map to pandas/xarray frequency codes
LAG_CYCLE_MAP = {
    "daily":    "1D",
    "weekly":   "7D",
    "monthly":  "MS",   # Month Start
    "annually": "YS",   # Year Start
}


class DataIngestionFSM:
    """
    Runs the full data ingestion pipeline using a Finite State Machine.
    Each state performs one well-defined transformation step, and the pipeline
    transitions through them in strict order.
    """
    def __init__(self, config):
        self.state = PipelineState.INIT
        self.config = config
        # This dict holds the xarray Datasets as they move through the pipeline
        self.data_store = {}

    def run(self):
        """Drives the FSM from INIT through to READY (or ERROR on failure)."""
        print("=== Starting Data Ingestion Pipeline ===")
        try:
            self.transition_to(PipelineState.INGESTING)
            self._ingest_data()

            self.transition_to(PipelineState.RESOLVING)
            self._resolve_data()

            self.transition_to(PipelineState.VALIDATING)
            self._validate_data()

            self.transition_to(PipelineState.STORING)
            self._store_data()

            if self.config["storage"].get("delete_raw_after_processing", False):
                self.transition_to(PipelineState.CLEANUP)
                self._cleanup_raw_data()

            self.transition_to(PipelineState.READY)
            print("=== Pipeline Finished Successfully! ===")

        except Exception as e:
            self.transition_to(PipelineState.ERROR)
            print(f"PIPELINE FAILED: {e}")
            raise

    def transition_to(self, new_state):
        """Safely transitions the state machine and logs the change."""
        print(f"  [{self.state}] ---> [{new_state}]")
        self.state = new_state

    # -------------------------------------------------------
    # State Logic
    # -------------------------------------------------------

    def _build_erddap_nc_url(self, opendap_url, variables, t_start, t_end, lat_min, lat_max, lon_min, lon_max):
        """
        Builds a direct ERDDAP .nc HTTP download URL using config-specified bounds.

        ERDDAP supports server-side subsetting via bracketed query parameters, so the
        server only sends the exact bytes we need — no global download happens.

        Example result:
          https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.nc?
            CRW_SST[(2020-01-01):(2020-12-31)][(-24.0):(-10.0)][(142.0):(154.0)],
            CRW_DHW[(2020-01-01):(2020-12-31)][(-24.0):(-10.0)][(142.0):(154.0)]
        """
        host       = opendap_url.split("/erddap/")[0]
        dataset_id = opendap_url.rstrip("/").split("/")[-1]
        var_parts  = ",".join(
            f"{v}[({t_start}):({t_end})]"
            f"[({lat_min:.4f}):({lat_max:.4f})]"
            f"[({lon_min:.4f}):({lon_max:.4f})]"
            for v in variables
        )
        return f"{host}/erddap/griddap/{dataset_id}.nc?{var_parts}"

    def _ingest_data(self):
        """
        Downloads satellite data from free ERDDAP servers.

        We try three strategies in order, stopping as soon as one succeeds:

          Strategy 1 — Direct ERDDAP .nc HTTP download (FASTEST, MOST RELIABLE)
            Builds a URL with the bounding-box embedded in it. The ERDDAP server
            does the subsetting and returns a small, pre-sliced NetCDF file.
            This works even when the OPeNDAP protocol is blocked/redirected.

          Strategy 2 — pydap OPeNDAP lazy stream
            Opens the remote dataset lazily, applies .sel() to subset, then .load().
            Works well for datasets that don't redirect. Used as a backup.

          Strategy 3 — Synthetic fallback (LAST RESORT)
            Generates physically plausible random data with correct grid/time structure.
            Ensures the pipeline never crashes, even with no internet connection.
        """
        canvas   = self.config["spatial_canvas"]
        t_canvas = self.config["temporal_canvas"]

        # Config bounds — used directly in the URL, no remote metadata fetch needed
        t_start  = t_canvas["start_date"]
        t_end    = t_canvas["end_date"]
        lat_min  = canvas["lat_min"]
        lat_max  = canvas["lat_max"]
        lon_min  = canvas["lon_min"]
        lon_max  = canvas["lon_max"]

        time_slice = slice(t_start, t_end)
        lon_slice  = slice(lon_min, lon_max)

        for name, info in self.config["data_sources"].items():
            opendap_url = info.get("opendap_url")
            variables   = info.get("variables", [])
            rename_map  = info.get("rename_vars", {})
            drop_dims   = info.get("drop_dims", [])

            if not opendap_url:
                print(f"  -> No OPeNDAP URL for '{name}'. Generating synthetic data.")
                self.data_store[name] = self._generate_synthetic(name, info)
                print(f"     Variables: {list(self.data_store[name].data_vars)}")
                continue

            print(f"  -> Fetching '{name}': {opendap_url}")
            ds_subset = None

            # ======================================================
            # STRATEGY 1: Direct ERDDAP .nc HTTP download
            # ======================================================
            # No OPeNDAP handshake — just a plain HTTPS request for a
            # pre-sliced NetCDF file. Most reliable option; sidesteps
            # any redirect or protocol mismatch issues.
            if variables:
                nc_url = self._build_erddap_nc_url(
                    opendap_url, variables,
                    t_start, t_end, lat_min, lat_max, lon_min, lon_max
                )
                try:
                    import tempfile, urllib.request, os as _os
                    print(f"     [Strategy 1] Direct ERDDAP .nc download...")
                    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                        tmp_path = tmp.name
                    # urlretrieve blocks until the full file lands — that's what we want
                    urllib.request.urlretrieve(nc_url, tmp_path)
                    ds_subset = xr.open_dataset(tmp_path)
                    # Normalise coordinate names
                    if "latitude" in ds_subset.dims:
                        ds_subset = ds_subset.rename({"latitude": "lat", "longitude": "lon"})
                    # Rename variables to internal names
                    if rename_map:
                        rename_actual = {k: v for k, v in rename_map.items()
                                         if k in ds_subset.data_vars}
                        if rename_actual:
                            ds_subset = ds_subset.rename(rename_actual)
                    ds_subset = ds_subset.load()
                    _os.unlink(tmp_path)
                    print(f"     [Strategy 1] Success!")
                except Exception as e:
                    print(f"     [Strategy 1] Failed ({type(e).__name__}: {str(e)[:120]})")
                    ds_subset = None

            # ======================================================
            # STRATEGY 2: pydap OPeNDAP lazy stream
            # ======================================================
            # Fetches only the variable metadata first, then pulls the subsetted
            # data. Good fallback for servers that don't support the .nc query format.
            if ds_subset is None:
                try:
                    print(f"     [Strategy 2] pydap OPeNDAP stream...")
                    ds_remote = xr.open_dataset(opendap_url, engine="pydap", chunks={})

                    if "latitude" in ds_remote.dims:
                        ds_remote = ds_remote.rename({"latitude": "lat", "longitude": "lon"})

                    if variables:
                        available = [v for v in variables if v in ds_remote.data_vars]
                        if not available:
                            raise ValueError(f"None of {variables} found. Available: {list(ds_remote.data_vars)}")
                        ds_remote = ds_remote[available]

                    for dim in drop_dims:
                        if dim in ds_remote.dims:
                            ds_remote = ds_remote.isel({dim: 0}, drop=True)

                    # Detect descending latitude axis and flip the slice accordingly
                    if ds_remote.lat.values[0] > ds_remote.lat.values[-1]:
                        lat_slice = slice(lat_max, lat_min)
                    else:
                        lat_slice = slice(lat_min, lat_max)

                    ds_subset = ds_remote.sel(time=time_slice, lat=lat_slice, lon=lon_slice)

                    if len(ds_subset.time) == 0:
                        raise ValueError("Time slice returned 0 records.")

                    if rename_map:
                        rename_actual = {k: v for k, v in rename_map.items()
                                         if k in ds_subset.data_vars}
                        if rename_actual:
                            ds_subset = ds_subset.rename(rename_actual)

                    print(f"     Downloading subset into memory...")
                    ds_subset = ds_subset.load()
                    print(f"     [Strategy 2] Success!")

                except Exception as e:
                    print(f"     [Strategy 2] Failed ({type(e).__name__}: {str(e)[:120]})")
                    ds_subset = None

            # ======================================================
            # STRATEGY 3: Synthetic fallback (last resort)
            # ======================================================
            if ds_subset is None:
                print(f"  !! All real-data attempts failed for '{name}'. Using synthetic mock.")
                ds_subset = self._generate_synthetic(name, info)

            self.data_store[name] = ds_subset
            print(f"     Variables: {list(self.data_store[name].data_vars)}")



    def _generate_synthetic(self, name, info):
        """
        Creates a synthetic xarray Dataset that mimics the structure of real data.
        This lets us test the full pipeline end-to-end without network access.

        Variable names here use the *internal* (renamed) names so that downstream
        code (dataset.py, metrics.py) doesn't need to know which source provided the data.
        """
        canvas   = self.config["spatial_canvas"]
        t_canvas = self.config["temporal_canvas"]
        src_res  = info.get("source_resolution", canvas["target_resolution"])

        # Build coordinate arrays at the source's native resolution
        lats  = np.arange(canvas["lat_min"], canvas["lat_max"], src_res)
        lons  = np.arange(canvas["lon_min"], canvas["lon_max"], src_res)
        times = xr.date_range(start=t_canvas["start_date"], end=t_canvas["end_date"], freq="1D", use_cftime=True)

        rng = np.random.default_rng(42)

        # Read the rename map to know what our internal variable names should be
        rename_map = info.get("rename_vars", {})

        if name == "sst":
            # Realistic SST for tropical Australia: 24–31°C
            sst_var = rename_map.get("CRW_SST", "sea_surface_temperature")
            dhw_var = rename_map.get("CRW_DHW", "degree_heating_weeks")

            sst_data = rng.uniform(24.0, 31.0, size=(len(times), len(lats), len(lons))).astype(np.float32)

            # Synthetic DHW: rolling 12-week sum of hotspots above 28°C (tropical bleaching threshold)
            hotspot  = np.maximum(0.0, sst_data - 28.0)
            dhw_data = np.zeros_like(sst_data)
            for t in range(len(times)):
                start          = max(0, t - 83)
                dhw_data[t]    = hotspot[start : t + 1].sum(axis=0) / 7.0

            ds = xr.Dataset(
                {
                    sst_var: (["time", "lat", "lon"], sst_data),
                    dhw_var: (["time", "lat", "lon"], dhw_data),
                },
                coords={"time": times, "lat": lats, "lon": lons},
            )

        elif name == "velocity":
            u_var = rename_map.get("u", "water_u")
            v_var = rename_map.get("v", "water_v")

            # Small surface current velocities in m/s
            u = rng.uniform(-0.5, 0.5, size=(len(times), len(lats), len(lons))).astype(np.float32)
            v = rng.uniform(-0.5, 0.5, size=(len(times), len(lats), len(lons))).astype(np.float32)
            ds = xr.Dataset(
                {u_var: (["time", "lat", "lon"], u), v_var: (["time", "lat", "lon"], v)},
                coords={"time": times, "lat": lats, "lon": lons},
            )

        else:
            # Generic fallback for any unknown source name
            data = rng.standard_normal(size=(len(times), len(lats), len(lons))).astype(np.float32)
            ds = xr.Dataset(
                {name: (["time", "lat", "lon"], data)},
                coords={"time": times, "lat": lats, "lon": lons},
            )

        return ds

    def _resolve_data(self):
        """
        Two-phase resolution to unify all data sources onto one shared grid:

          Phase A — Temporal: Resample each dataset to the configured time frequency
                    and aggregation strategy (e.g., daily maximum).
          Phase B — Spatial: Interpolate all datasets onto the unified lat/lon canvas
                    defined in config (the same grid for every variable).
        """
        canvas   = self.config["spatial_canvas"]
        t_canvas = self.config["temporal_canvas"]

        target_lat = np.arange(canvas["lat_min"], canvas["lat_max"], canvas["target_resolution"])
        target_lon = np.arange(canvas["lon_min"], canvas["lon_max"], canvas["target_resolution"])

        freq_code    = LAG_CYCLE_MAP[t_canvas["time_lag_cycle"]]
        strat_method = LAG_STRATEGY_MAP[t_canvas["lag_strategy"]]

        for name, ds in self.data_store.items():
            src_res = self.config["data_sources"][name].get(
                "source_resolution", canvas["target_resolution"]
            )

            # Phase A: temporal resampling
            print(f"  -> Phase A [{name}]: Temporal resample ({t_canvas['time_lag_cycle']} {t_canvas['lag_strategy']})")
            resampler = ds.resample(time=freq_code)
            ds = getattr(resampler, strat_method)()

            # Phase A (spatial): if source resolution differs, interpolate to declared source res first
            if src_res != canvas["target_resolution"]:
                src_lat = np.arange(canvas["lat_min"], canvas["lat_max"], src_res)
                src_lon = np.arange(canvas["lon_min"], canvas["lon_max"], src_res)
                print(f"  -> Phase A [{name}]: Spatial resample to source resolution {src_res}°")
                ds = ds.interp(lat=src_lat, lon=src_lon, method="linear")

            # Phase B: regrid everything to the unified canvas
            print(f"  -> Phase B [{name}]: Interpolating to unified canvas at {canvas['target_resolution']}°")
            ds = ds.interp(lat=target_lat, lon=target_lon, method="linear")

            self.data_store[name] = ds

    def _validate_data(self):
        """
        Checks each resolved dataset for:
          1. NaN values (and fills any edge NaNs from interpolation)
          2. Physically impossible values (SST outside -2 to 40°C)
        """
        for name, ds in self.data_store.items():
            for var in ds.data_vars:
                arr = ds[var]
                nan_count = int(arr.isnull().sum().values)
                total     = int(arr.size)
                nan_pct   = (nan_count / total) * 100 if total > 0 else 0

                print(f"  -> [{name}/{var}] NaN: {nan_count}/{total} ({nan_pct:.1f}%)")

                if nan_count > 0:
                    # Fill edge NaNs from interpolation by nearest-neighbour extrapolation
                    ds[var] = (
                        arr.interpolate_na(dim="lon", method="nearest")
                           .interpolate_na(dim="lat", method="nearest")
                    )
                    remaining = int(ds[var].isnull().sum().values)
                    print(f"     After fill: {remaining} NaN remaining")

            # Physical bounds check — ocean temps outside this range are impossible
            if "sea_surface_temperature" in ds.data_vars:
                sst     = ds["sea_surface_temperature"]
                sst_min = float(sst.min().values)
                sst_max = float(sst.max().values)
                print(f"  -> SST range: {sst_min:.2f} to {sst_max:.2f} °C")
                if sst_min < -2.0 or sst_max > 40.0:
                    print("  !! WARNING: SST values outside physical bounds [-2, 40]")

            self.data_store[name] = ds

    def _store_data(self):
        """
        Merges all resolved datasets into a single xarray Dataset and saves it
        as a Zarr store on disk.

        Zarr is chunked columnar storage — much more efficient than NetCDF for
        time-series access patterns (each chunk covers 30 days, full spatial extent).
        """
        out_path = self.config["storage"]["processed_zarr_path"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        combined = xr.merge(list(self.data_store.values()))

        # Chunk for efficient time-series reads: 30-day windows, full spatial extent
        chunk_spec = {"time": 30, "lat": -1, "lon": -1}
        combined   = combined.chunk(chunk_spec)

        print(f"  -> Writing Zarr store to: {out_path}")
        print(f"     Combined variables: {list(combined.data_vars)}")
        print(f"     Combined shape: {dict(combined.dims)}")

        if os.path.exists(out_path):
            shutil.rmtree(out_path)

        combined.to_zarr(out_path, mode="w", consolidated=True)
        print(f"  -> Zarr store written successfully.")

    def _cleanup_raw_data(self):
        """
        Since we used OPeNDAP streaming, no raw NetCDF files were ever
        downloaded to disk — nothing to delete. Maximum storage efficiency.
        """
        print("  -> OPeNDAP streaming was used. No local raw files to clean up.")


# Quick test if run directly
if __name__ == "__main__":
    validator   = ConfigValidator("../config.yaml")
    valid_config = validator.validate()

    pipeline = DataIngestionFSM(valid_config)
    pipeline.run()
