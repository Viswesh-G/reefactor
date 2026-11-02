"""One reviewed feature registry; source fields and derived features stay distinct.

Catalog dates are documentation snapshots, not evidence of fetched coverage.
Remote variable names and units are checked again on every fetched partition.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Feature:
    name: str
    native: str
    units: str
    role: str = "input"
    reducer: str = "mean"
    minimum: float | None = None
    maximum: float | None = None


SOURCES = {
    "noaa": {
        "provider": "NOAA CRW / PacIOOS mirror", "product": "CoralTemp v3.1",
        "backend": "erddap", "dataset_id": "dhw_5km", "version": "v3.1",
        "url": "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km",
        "documentation": "https://coralreefwatch.noaa.gov/product/5km/index_5km_sst.php",
        # Actual mirror axis (checked 2026-11-02); product itself starts January.
        "start": "1985-04-01", "catalog_end": "2026-10-29", "resolution": 0.05,
        "cadence": "daily", "kind": "satellite_analysis", "lat_bounds": [-89.975, 89.975],
        "features": [Feature("sea_surface_temperature", "CRW_SST", "degree_Celsius", minimum=-2, maximum=40),
                     Feature("degree_heating_weeks", "CRW_DHW", "degree_Celsius_week", "reference", "last", 0, 100)],
    },
    "physics": {
        "provider": "Copernicus Marine", "product": "GLOBAL_MULTIYEAR_PHY_001_030",
        "backend": "copernicus", "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "version": None, "start": "1993-01-01", "catalog_end": "2026-09-22",
        "documentation": "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description",
        "resolution": 1 / 12, "cadence": "daily", "kind": "assimilative_reanalysis",
        "lat_bounds": [-80, 90],
        "features": [Feature("ocean_temperature", "thetao", "degree_Celsius", minimum=-3, maximum=45),
                     Feature("salinity", "so", "1e-3", minimum=0, maximum=50),
                     Feature("water_u", "uo", "m s-1", minimum=-5, maximum=5),
                     Feature("water_v", "vo", "m s-1", minimum=-5, maximum=5),
                     Feature("sea_level", "zos", "m", minimum=-10, maximum=10),
                     Feature("mixed_layer_depth", "mlotst", "m", minimum=0, maximum=6000)],
    },
    "era5": {
        "provider": "ECMWF / Copernicus CDS", "product": "reanalysis-era5-single-levels",
        "backend": "cds", "dataset_id": "reanalysis-era5-single-levels", "version": "ERA5",
        "documentation": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview",
        "start": "1940-01-01", "catalog_end": "2026-10-26", "resolution": 0.25,
        "cadence": "hourly", "kind": "atmospheric_reanalysis", "lat_bounds": [-90, 90],
        # Use hourly MEAN fluxes: no ambiguous accumulation/reset conversion.
        "features": [Feature("wind_u", "10m_u_component_of_wind", "m s-1"),
                     Feature("wind_v", "10m_v_component_of_wind", "m s-1"),
                     Feature("air_temperature", "2m_temperature", "K", minimum=180, maximum=340),
                     Feature("dewpoint", "2m_dewpoint_temperature", "K", minimum=150, maximum=340),
                     Feature("surface_pressure", "surface_pressure", "Pa", minimum=50000, maximum=110000),
                     Feature("mean_sea_level_pressure", "mean_sea_level_pressure", "Pa"),
                     Feature("cloud_fraction", "total_cloud_cover", "1", minimum=0, maximum=1),
                     Feature("shortwave_down", "mean_surface_downward_short_wave_radiation_flux", "W m-2"),
                     Feature("longwave_down", "mean_surface_downward_long_wave_radiation_flux", "W m-2"),
                     Feature("shortwave_net", "mean_surface_net_short_wave_radiation_flux", "W m-2"),
                     Feature("longwave_net", "mean_surface_net_long_wave_radiation_flux", "W m-2"),
                     Feature("latent_heat_flux", "mean_surface_latent_heat_flux", "W m-2"),
                     Feature("sensible_heat_flux", "mean_surface_sensible_heat_flux", "W m-2"),
                     Feature("precipitation", "mean_total_precipitation_rate", "kg m-2 s-1", minimum=0),
                     Feature("evaporation", "mean_evaporation_rate", "kg m-2 s-1"),
                     Feature("uv_down", "mean_surface_downward_uv_radiation_flux", "W m-2")],
    },
    "bgc": {
        "provider": "Copernicus Marine", "product": "GLOBAL_MULTIYEAR_BGC_001_029",
        "backend": "copernicus", "dataset_id": "cmems_mod_glo_bgc_my_0.25deg_P1D-m",
        "version": None, "start": "1993-01-01", "catalog_end": "2026-08-31",
        "documentation": "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_BGC_001_029/description",
        "resolution": 0.25, "cadence": "daily", "kind": "unassimilated_hindcast",
        "lat_bounds": [-80, 90],
        "features": [Feature("chlorophyll", "chl", "mg m-3", minimum=0),
                     Feature("nitrate", "no3", "mmol m-3", minimum=0),
                     Feature("phosphate", "po4", "mmol m-3", minimum=0),
                     Feature("silicate", "si", "mmol m-3", minimum=0),
                     Feature("oxygen", "o2", "mmol m-3", minimum=0),
                     Feature("primary_production", "nppv", "mg m-3 day-1", minimum=0)],
    },
}
SOURCES["bgc_monthly"] = {
    **SOURCES["bgc"], "dataset_id": "cmems_mod_glo_bgc_my_0.25deg_P1M-m", "cadence": "monthly",
    "features": [Feature("iron", "fe", "mmol m-3", minimum=0),
                 Feature("phytoplankton_carbon", "phyc", "mmol m-3", minimum=0),
                 Feature("ph", "ph", "1", minimum=6, maximum=10),
                 Feature("surface_pco2", "spco2", "Pa", minimum=0)],
}
SOURCES["static"] = {
    **SOURCES["bgc"], "dataset_id": "cmems_mod_glo_bgc_my_0.25deg_static", "cadence": "static",
    "features": [Feature("bathymetry", "deptho", "m", "context", minimum=0),
                 Feature("ocean_fraction", "mask", "1", "mask", minimum=0, maximum=1)],
}

# Anonymous HTTPS Zarr URLs published by the Copernicus describe API, checked
# 2026-11-02. These are mirrors of the SAME product, not additional data sources.
for _name, _number, _version in [("physics", "025", "202311"), ("bgc", "018", "202406"),
                                ("bgc_monthly", "018", "202406")]:
    _spec = SOURCES[_name]
    _product, _dataset = _spec["product"], _spec["dataset_id"]
    _spec["zarr_version"] = _version
    _spec["zarr_urls"] = [
        f"https://s3.waw3-1.cloudferro.com/mdl-arco-geo-{_number}/arco/{_product}/{_dataset}_{_version}/geoChunked.zarr",
        f"https://s3.waw3-1.cloudferro.com/mdl-arco-time-{_number}/arco/{_product}/{_dataset}_{_version}/timeChunked.zarr",
    ]
SOURCES["static"]["zarr_version"] = "202406"
SOURCES["static"]["zarr_urls"] = [
    "https://s3.waw3-1.cloudferro.com/mdl-arco-time-018/arco/GLOBAL_MULTIYEAR_BGC_001_029/"
    "cmems_mod_glo_bgc_my_0.25deg_static_202406--ext--mask/static.zarr"]

# Full published rectilinear baseline inventory, captured from the live DAS.
# These 106 columns include diagnostics and model biology, not 106 proven causes.
EREEFS_FIELDS = json.loads((Path(__file__).parent/"data"/"ereefs_bgc_fields.json").read_text(encoding="utf-8"))
SOURCES["ereefs_bgc"] = {
    "provider": "AIMS / CSIRO eReefs", "product": "GBR4_H2p0_B3p1_Cq3b_Dhnd",
    "backend": "opendap", "dataset_id": "GBR4_H2p0_B3p1_Cq3b_Dhnd/daily.nc",
    "version": "B3p1_Cq3b_Dhnd_AIMS_2021", "start": "2010-12-01", "catalog_end": "2019-04-30",
    "url": "https://thredds.ereefs.aims.gov.au/thredds/dodsC/GBR4_H2p0_B3p1_Cq3b_Dhnd/daily.nc",
    "documentation": "https://thredds.ereefs.aims.gov.au/thredds/catalog/ereefs/GBR4_H2p0_B3p1_Cq3b_Dhnd.html",
    "resolution": 0.03, "cadence": "daily", "kind": "regional_biogeochemical_model_baseline",
    "temporal_statistic": "daily_snapshot_02_UTC", "lat_bounds": [-28.696022, -7.036022],
    "features": [Feature("ereefs_"+item["native"].lower(), item["native"], item["units"],
                  "simulated_response" if item["native"].startswith(("CS_", "CH_", "SG", "MA_")) else
                  "diagnostic" if item["native"].startswith(("R_", "month_")) else "input")
                 for item in EREEFS_FIELDS],
}

# CDS NetCDF output short names, verified against the ECMWF parameter database.
ERA5_SHORT = dict(zip([f.native for f in SOURCES["era5"]["features"]],
    ["u10", "v10", "t2m", "d2m", "sp", "msl", "tcc", "msdwswrf", "msdwlwrf",
     "msnswrf", "msnlwrf", "mslhf", "msshf", "mtpr", "mer", "msdwuvrf"]))

DERIVED = {
    "wind_speed": ["wind_u", "wind_v"], "current_speed": ["water_u", "water_v"],
    "relative_humidity": ["air_temperature", "dewpoint"],
    "net_heat_flux": ["shortwave_net", "longwave_net", "latent_heat_flux", "sensible_heat_flux"],
    "freshwater_flux": ["precipitation", "evaporation"],
    "hotspot": ["sea_surface_temperature", "official_mmm"],
    "derived_dhw": ["hotspot"], "heat_tendency": ["net_heat_flux", "mixed_layer_depth"],
    "season_sin": ["time"], "season_cos": ["time"],
    "surface_model_temperature_difference": ["sea_surface_temperature", "ocean_temperature"],
}

LABEL_SOURCE = {
    "provider": "BCO-DMO", "dataset_id": "773466", "version": 2,
    "doi": "10.26008/1912/bco-dmo.773466.2", "license": "CC-BY-4.0",
    "documentation": "https://www.bco-dmo.org/dataset/773466/description",
    "url": "https://datadocs.bco-dmo.org/dataset/773466/file/B11vA82u7y2Owp/global_bleaching_environmental.csv",
    "start": "1980-06-15", "end": "2020-08-15", "support": "sparse_surveys",
    "target": {"native": "Percent_Bleaching", "units": "percent", "role": "biological_target"},
    "context": {"Depth_m": "m", "Distance_to_Shore": "m", "Exposure": "categorical",
                "Turbidity": "m-1 (Kd490 proxy, 100 km buffer)",
                "Cyclone_Frequency": "events in 1964-2014; causal availability audit required"},
}


def serializable_catalog():
    return {key: {**value, "features": [asdict(f) for f in value["features"]]}
            for key, value in SOURCES.items()}
