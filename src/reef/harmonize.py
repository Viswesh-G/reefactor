"""Conservative rectilinear area means, explicit UTC days and missing-data masks."""
import numpy as np
import xarray as xr

from .catalog import ERA5_SHORT, SOURCES


def normalize_coordinates(ds):
    names = {old: new for old, new in [("latitude", "lat"), ("longitude", "lon"),
             ("valid_time", "time")] if old in ds.dims and new not in ds.dims}
    ds = ds.rename(names)
    if "zc" in ds and ds.zc.dims == ("k",):
        ds = ds.swap_dims({"k": "zc"}).rename(zc="depth")
        ds = ds.assign_coords(depth=-ds.depth)
    if "elevation" in ds.dims:
        ds = ds.rename(elevation="depth")
        ds = ds.assign_coords(depth=-ds.depth)
    if "depth" in ds.dims:
        # Surface model level only; never silently average the whole water column.
        if float(ds.depth.min()) > 2:
            raise ValueError("Requested surface field has no near-surface depth level")
        index = int(ds.depth.argmin(dim="depth"))
        selected_depth = float(ds.depth[index])
        for name in ds.data_vars:
            if "depth" in ds[name].dims:
                ds[name].attrs = {**ds[name].attrs, "selected_depth_m": selected_depth,
                                  "vertical_selection": "native_near_surface_model_layer"}
        ds = ds.isel(depth=index, drop=True)
    for dim in ("number", "expver"):
        if dim in ds.dims:
            if ds.sizes[dim] != 1:
                raise ValueError(f"Resolve {dim} explicitly before harmonization")
            ds = ds.isel({dim: 0}, drop=True)
    ds = ds.assign_coords(lon=(ds.lon + 180) % 360 - 180).sortby("lat").sortby("lon")
    for axis in ("lat", "lon"):
        if ds.sizes[axis] < 2 or not np.isfinite(ds[axis].values).all() or np.any(np.diff(ds[axis].values) <= 0):
            raise ValueError(f"{axis} must contain unique increasing coordinates with a halo")
    if "time" in ds.dims:
        ds = ds.sortby("time")
        if not np.issubdtype(ds.time.dtype, np.datetime64) or bool(ds.time.isnull().any()) or np.any(np.diff(ds.time.values) <= np.timedelta64(0, "ns")):
            raise ValueError("Time must use unique increasing Gregorian UTC timestamps")
    return ds


def _edges(centers):
    middle = (centers[1:] + centers[:-1]) / 2
    return np.r_[centers[0] - (centers[1]-centers[0])/2, middle,
                 centers[-1] + (centers[-1]-centers[-2])/2]


def overlap_weights(native, target, resolution, latitude=False):
    native_edges = _edges(native)
    target_edges = np.r_[target - resolution/2, target[-1] + resolution/2]
    if latitude:
        native_edges = np.sin(np.deg2rad(np.clip(native_edges, -90, 90)))
        target_edges = np.sin(np.deg2rad(np.clip(target_edges, -90, 90)))
    low = np.maximum(target_edges[:-1, None], native_edges[None, :-1])
    high = np.minimum(target_edges[1:, None], native_edges[None, 1:])
    weights = np.maximum(high-low, 0)
    expected = np.diff(target_edges)
    if not np.allclose(weights.sum(axis=1), expected, rtol=1e-5, atol=1e-10):
        raise ValueError("Native subset does not cover target cell edges; refusing extrapolation")
    return weights


def area_mean(array, lat, lon, resolution):
    wy = xr.DataArray(overlap_weights(array.lat.values, lat, resolution, True),
                      dims=("target_lat", "lat"), coords={"target_lat": lat, "lat": array.lat})
    wx = xr.DataArray(overlap_weights(array.lon.values, lon, resolution),
                      dims=("target_lon", "lon"), coords={"target_lon": lon, "lon": array.lon})
    valid = array.notnull()
    numerator = xr.dot(xr.dot(array.fillna(0), wy, dim="lat"), wx, dim="lon")
    denominator = xr.dot(xr.dot(valid.astype("float64"), wy, dim="lat"), wx, dim="lon")
    full_area = wy.sum("lat") * wx.sum("lon")
    # Partial coast cells retain a water-only average only with >=50% support.
    result = (numerator / denominator.where(denominator > 0)).where(denominator >= 0.5 * full_area)
    return result.rename(target_lat="lat", target_lon="lon")


UNIT_ALIASES = {
    "degree_Celsius": {"degree_celsius", "degrees_celsius", "degree_c", "degrees_c", "degc", "celsius"},
    "degree_Celsius_week": {"degree_celsius_week", "degree_celsius_weeks", "degree_celsiusweeks", "degc-weeks", "celsiusweeks"},
    "m s-1": {"ms-1", "m/s", "ms**-1", "ms^-1"},
    "W m-2": {"wm-2", "w/m^2", "wm**-2", "wm^-2"},
    "kg m-2 s-1": {"kgm-2s-1", "kgm**-2s**-1", "kgm^-2s^-1"},
    "mmol m-3": {"mmolm-3", "mmol/m^3", "mmolm**-3", "mmolm^-3"},
    "mg m-3": {"mgm-3", "mg/m^3", "mgm**-3", "mgm^-3"},
    "mg m-3 day-1": {"mgm-3day-1", "mgm-3d-1", "mgm**-3day**-1"},
    "1e-3": {"1e-3", "psu", "0.001", "1e-03", "10^-3"},
    "1": {"1", "dimensionless", "(0-1)", "0-1"},
}


def check_units(actual, expected):
    compact = str(actual).lower().replace(" ", "").replace("_per_", "/")
    acceptable = UNIT_ALIASES.get(expected, {expected.lower().replace(" ", "")})
    if compact not in acceptable:
        raise ValueError(f"Unit mismatch: received {actual!r}; expected {expected!r}")


def harmonize(ds, source_name, job):
    source = SOURCES[source_name]
    ds = normalize_coordinates(ds)
    lat, lon = job.centers()
    output = {}
    for feature in source["features"]:
        candidates = [feature.native, ERA5_SHORT.get(feature.native, ""), feature.name]
        native = next((n for n in candidates if n in ds.data_vars), None)
        if native is None:
            raise ValueError(f"{source_name}: missing required field {feature.native}")
        field = ds[native]
        if source["cadence"] == "static" and native != "mask" and "mask" in ds:
            field = field.where(ds["mask"] > 0)
        check_units(field.attrs.get("units"), feature.units)
        expected_dims = {"lat", "lon"} if source["cadence"] == "static" else {"time", "lat", "lon"}
        if set(field.dims) != expected_dims:
            raise ValueError(f"{native}: unexpected dimensions {field.dims}")
        if source["cadence"] == "3hourly":
            # Instantaneous values on the exact eight UTC slots. Missing native
            # values invalidate that cell-day before spatial averaging.
            slots = np.arange(np.datetime64(job.start, "h"), np.datetime64(job.end, "h") + 24,
                              np.timedelta64(3, "h"))
            if not np.array_equal(field.time.values, slots):
                raise ValueError(f"{native}: incomplete or shifted three-hourly UTC slots")
            field = field.resample(time="1D").mean(skipna=False, keep_attrs=True)
        elif source["cadence"] == "hourly":
            # ECMWF mean fluxes end at validity time. Shift only those fields.
            if feature.native.startswith("mean_") and feature.name != "mean_sea_level_pressure":
                field = field.assign_coords(time=field.time - np.timedelta64(1, "h"))
            field = field.sel(time=slice(job.start, job.end))
            counts = field.resample(time="1D").count()
            field = field.resample(time="1D").mean(skipna=False).where(counts == 24)
        elif source["cadence"] == "daily":
            days = field.time.values.astype("datetime64[D]")
            if len(np.unique(days)) != len(days):
                raise ValueError("Daily source contains duplicate calendar days")
            field = field.assign_coords(time=days).sel(time=slice(job.start, job.end))
        elif source["cadence"] == "monthly":
            months = field.time.values.astype("datetime64[M]").astype("datetime64[ns]")
            if len(np.unique(months)) != len(months):
                raise ValueError("Monthly source contains duplicate months")
            field = field.assign_coords(time=months)
        if source["cadence"] != "static":
            unit = "M" if source["cadence"] == "monthly" else "D"
            expected = np.arange(np.datetime64(job.start, unit), np.datetime64(job.end, unit)+1,
                                 dtype=f"datetime64[{unit}]").astype("datetime64[ns]")
            if not np.array_equal(field.time.values.astype("datetime64[ns]"), expected):
                raise ValueError(f"{native}: incomplete time support")
            # A verified UTC day may arrive at second or nanosecond precision.
            # Canonicalize AFTER comparison so exact joins cannot hide a shift.
            field = field.assign_coords(time=expected)
        result = area_mean(field, lat, lon, job.resolution)
        dims = ("lat", "lon") if source["cadence"] == "static" else ("time", "lat", "lon")
        result = result.transpose(*dims).astype("float32")
        result.attrs = {"units": feature.units, "source": source_name, "native_variable": native,
                        "native_resolution_degrees": source["resolution"], "native_cadence": source["cadence"],
                        "source_kind": source["kind"], "role": feature.role,
                        "source_long_name": field.attrs.get("long_name", ""),
                        "vertical_selection": field.attrs.get("vertical_selection", "source_2D_field"),
                        "spatial_method": "spherical_cell_overlap_water_area_mean_min_support_0.5"}
        if "selected_depth_m" in field.attrs:
            result.attrs["selected_depth_m"] = field.attrs["selected_depth_m"]
        output[feature.name] = result
    result = xr.Dataset(output)
    if source["cadence"] == "monthly":
        result = result.assign_coords(available_not_before=("time",
            (result.time.values.astype("datetime64[M]")+1).astype("datetime64[ns]")))
    return result


def validate_partition(ds, names, max_missing):
    """No temporal interpolation. Unknown permanent support stays explicitly masked."""
    metrics = {}
    common = None
    for name in names:
        for feature in SOURCES[name]["features"]:
            array = ds[feature.name]
            finite = np.isfinite(array.values)
            if not finite.any():
                raise ValueError(f"{feature.name}: no finite support; use an ocean/reef region")
            if np.isinf(array.values).any():
                raise ValueError(f"{feature.name}: infinite values")
            values = array.values[finite]
            if feature.minimum is not None and values.min() < feature.minimum - 1e-6:
                raise ValueError(f"{feature.name}: below physical sanity bound")
            if feature.maximum is not None and values.max() > feature.maximum + 1e-6:
                raise ValueError(f"{feature.name}: above physical sanity bound")
            supported = finite.any(axis=0) if "time" in array.dims else finite
            fraction = 1 - float(finite[:, supported].mean()) if "time" in array.dims else 0.0
            if fraction > max_missing:
                raise ValueError(f"{feature.name}: intermittent missing fraction {fraction:.3f}")
            metrics[feature.name] = {"min": float(values.min()), "max": float(values.max()),
                "missing_fraction_on_ever_valid_support": fraction, "supported_cell_fraction": float(supported.mean())}
            mask = xr.DataArray(finite, coords=array.coords, dims=array.dims)
            ds[f"{feature.name}__valid"] = mask
            common = mask if common is None else common & mask
    ds["common_valid"] = common
    if not bool(common.any()):
        raise ValueError("No common finite support between selected fields")
    metrics["common_valid_fraction"] = float(common.mean())
    return ds, metrics
