"""Algebraic features computed lazily from independent source fields."""
import numpy as np
import xarray as xr


def derive_features(ds, official_mmm=None):
    result = ds.copy(deep=False)
    for name, u, v in [("wind_speed", "wind_u", "wind_v"), ("current_speed", "water_u", "water_v")]:
        if u in ds and v in ds:
            result[name] = np.hypot(ds[u], ds[v])
            result[name].attrs["units"] = "m s-1"
    if {"air_temperature", "dewpoint"} <= set(ds):
        t, d = ds.air_temperature - 273.15, ds.dewpoint - 273.15
        result["relative_humidity"] = (100*np.exp(17.625*d/(243.04+d) - 17.625*t/(243.04+t))).clip(0, 100)
    fluxes = ["shortwave_net", "longwave_net", "latent_heat_flux", "sensible_heat_flux"]
    if set(fluxes) <= set(ds):
        result["net_heat_flux"] = sum(ds[f] for f in fluxes)
        result.net_heat_flux.attrs = {"units": "W m-2", "positive": "downward"}
    if {"precipitation", "evaporation"} <= set(ds):
        result["freshwater_flux"] = ds.precipitation + ds.evaporation
        result.freshwater_flux.attrs = {"units": "kg m-2 s-1", "positive": "into_ocean"}
    if "net_heat_flux" in result and "mixed_layer_depth" in ds:
        result["heat_tendency"] = result.net_heat_flux/(1025*3990*ds.mixed_layer_depth.where(ds.mixed_layer_depth > 0))
    if {"sea_surface_temperature", "ocean_temperature"} <= set(ds):
        result["surface_model_temperature_difference"] = ds.sea_surface_temperature - ds.ocean_temperature
    if "time" in ds.dims:
        phase = 2*np.pi*(ds.time.dt.dayofyear-1)/365.2425
        result["season_sin"], result["season_cos"] = np.sin(phase), np.cos(phase)
    if official_mmm is not None:
        sst, mmm = xr.align(ds.sea_surface_temperature, official_mmm, join="exact")
        result["hotspot"] = (sst-mmm).clip(min=0)
        # 84 complete daily records needed, including history before the forecast window.
        if np.any(np.diff(ds.time.values) != np.timedelta64(1, "D")):
            raise ValueError("DHW requires contiguous UTC daily records")
        dose = xr.where(result.hotspot >= 1, result.hotspot, 0).where(result.hotspot.notnull())
        result["derived_dhw"] = dose.rolling(time=84, min_periods=84).sum()/7
    return result
