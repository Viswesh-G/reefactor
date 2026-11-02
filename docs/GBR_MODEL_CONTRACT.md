# GBR model contract and driver expansion — 2 November 2026

## Scope and coordinate guarantee

The active study is **GBR only**. Global products supply GBR subsets. Requests
outside the study envelope are rejected. The [official Reef Authority boundary](https://www.gbrmpa.gov.au/about-us/spatial-data-information-services)
is stored under CC BY 4.0, in EPSG:4326, with its checksum. A retained grid-cell
center must intersect that polygon. Source halos support averaging; they do not
become model targets. A coarse grid cell is not a claim of colony-scale coverage.

Every tile must contain the **same required feature list**, exact requested
coordinates, dimensions and dates. A missing field or shifted coordinate fails
before commit. Tiles are storage partitions, never temperature-only versus
nutrient-only geographic feature groups.

After ingestion the FSM enters **FINALIZING**, computing:

`domain(lat,lon) = inside_GBR AND all(isfinite(field(time,lat,lon)) for ALL required fields and ALL requested times)`

This one fixed mask applies to **every field at every time**. An oxygen gap in
February excludes that location from all fields in January and February. An
empty intersection fails the run. Exclusions and the feature contract are
recorded in the manifest and `model_domain.zarr`. Geometry-only tile pruning
is identical for every feature. Empty rectangle rows/columns are restored as
uniformly masked cells, never extrapolated values.

`open_cube(root)` serves the model contract. Raw variable-specific availability
is accessible explicitly with `model_ready=False` for audits. Old stores without
a fixed domain are refused as model data. One spatial tile and a small 2-D mask
are resident during finalization; no full-history array is loaded.
Tile width is reduced automatically when the selected native source arrays would
exceed the memory budget; budgeting only the smaller output grid would be unsafe.

This is retrospective availability-based domain selection. Changing the time
window may change its geography. Report that selection bias; a future operational
deployment should freeze a training-era domain and fail on newly missing inputs.
Training statistics still use training dates only.

## Coverage contracts

There are now **142 registered gridded source fields**, including 106 newly
integrated eReefs fields. Counts include diagnostics, derived products and
simulated biological states; they are not counts of independent bleaching causes.
Across verification runs 126 fields have been fetched; 16 ERA5 fields still
await live CDS verification with credentials/licence acceptance.

| Whole-study configuration | Fields everywhere on its retained domain | Catalog-supported period |
|---|---:|---|
| `gbr_long` | 30: NOAA + GLORYS + ERA5 + global BGC | 1993-01-01–2026-08-31 |
| `gbr_rich` | 130: NOAA + GLORYS + ERA5 + eReefs baseline | 2010-12-01–2019-04-30 |
| `gbr_rich_public` | 114: NOAA + GLORYS + eReefs baseline | 2010-12-01–2019-04-30 |

These are separate experiment contracts, **not different feature sets in different
GBR boxes of one model**. Never merge their tiles or change a checkpoint's feature
order by location. Temporal catalog limits do not prove complete fetched coverage.

The pilot is 2018-01-01–03, edges 18–17°S and 146–147°E, daily 0.25°. The full-GBR
planning configuration is `configs/gbr_study.yaml`, envelope 24–10°S, 142–154°E,
with official-polygon clipping. It does not automatically download decades of data.

```sh
python run_research.py plan --config configs/gbr_study.yaml
python run_research.py plan --config configs/gbr_study.yaml --profile gbr_rich --start 2010-12-01 --end 2019-04-30
python run_research.py ingest --profile gbr_rich_public
```

The coarsest chosen source limits shared resolution. A 0.25° cell is about 28 km
meridionally. Regridding does not make ERA5 or GLORYS into independent 4 km reef
measurements, nor resolve tide pools, colony shading or nocturnal hypoxia.

## Non-thermal fields actually integrated

The [AIMS/CSIRO eReefs rectilinear baseline](https://thredds.ereefs.aims.gov.au/thredds/catalog/ereefs/GBR4_H2p0_B3p1_Cq3b_Dhnd.html)
exposes **106 data variables** on the live endpoint. All were enumerated from
DDS/DAS metadata and fetched through a bounded OPeNDAP adapter. The full native
names, units, descriptions and dimension coordinates are retained in
`src/reef/data/ereefs_bgc_fields.json` and the exported evidence catalog.

They include underwater PAR, PAR at depth, Kd490, Secchi depth, fluorescence,
suspended inorganic matter, fine sediment, TSS, dust, ammonium, DIN/DIP,
nitrogen/phosphorus pools, DIC, alkalinity, carbonate/bicarbonate, aragonite
saturation, pH, pCO2, oxygen saturation, oxygen demand, plankton, detritus and
production. Spectral reflectance diagnostics and coral/symbiont, macroalgal and
seagrass model states are also preserved, with explicit roles.
The baseline uses specified catchment-condition assumptions, including a 2019
condition applied to earlier periods. It is a retrospective modeled scenario,
not an archive of the exact information available to a forecast issued in 2010.
Operational claims require production vintages and an explicit as-of audit.

Simulated biological responses are labeled `simulated_response` and refused as
training predictors. In particular, `CS_bleach` is a modeled rate, **not observed
bleaching truth**. Default training uses every admitted input-role field; explicit
`--features` defines a whole-study ablation. More columns from a coupled simulator
are not more independent observations. Diagnostics are not default input fields.

The verified axis is **2010-12-01 02:00 UTC–2019-04-30 02:00 UTC**. UTC+10 is
decoded before calendar-day assignment. These are daily snapshots, not means or
maxima. Native units remain in separate `ereefs_*` fields: eReefs pCO2 ppmv and
PH metadata `log(mM)` are not silently equated to Copernicus Pa/total-scale pH.
Resolve these semantics before imposing carbonate chemistry equations.
Three-dimensional variables select the native near-surface model layer, with
its actual selected depth recorded. Two-dimensional benthic/model-state fields
retain their different meaning; sharing a map cell does not make them measurements
at the same physical depth.

Additional sources inspected but not silently admitted:

| Source | Actual axis / issue | Required work |
|---|---|---|
| AIMS hydro daily | 2010-08-31 14:00 UTC–2024-01-16 14:00 UTC; local-day averages | Native hourly aggregation for exact UTC forcing |
| [Native eReefs BGC v4.2](https://thredds.nci.org.au/thredds/catalog/catalogs/fx3/catalog.html?dataset=GBR4_H4p0_ABARRAr2_OBRAN2020_FG2Gv3_B4p2_Cq5b_Dhnd) | 2011-01-01–2022-10-30; curvilinear grid and different model run | Validated remapping and overlap/version-bias assessment; no unreviewed stitching |
| [GBR1 diuron scenario](https://thredds.nci.org.au/thredds/catalog/catalogs/fx3/catalog.html?dataset=GBR1_H2p0_Cq3pe1_Dhnd) | 2016-01-01–2018-06-23; scenario-specific catchment assumptions | Validate concentrations, model geometry and meaning; not measured all-GBR pesticide exposure |

## Driver inventory beyond heat

`python run_research.py drivers` lists **174 inventory entries in 16 families**,
with sources and acquisition/admission status. Entries include grouped quantities,
derived indices, susceptibility modifiers, confounders and screening hypotheses.
They are not 174 additional independent gridded measurements.

| Family | Examples and source route |
|---|---|
| Thermal history | Cold shocks, variability, warming rate, acclimatization: NOAA/GLORYS/AIMS |
| Light/optics | PAR, UV, attenuation, cloud/shading: eReefs/ERA5/ocean colour |
| Mixing/circulation | Currents, vertical exchange, stratification, residence time: GLORYS/eReefs |
| Waves/tides | Waves, Stokes drift, emersion, cyclones: Copernicus/ERA5/BoM/native hydro |
| Freshwater | Salinity, rainfall, discharge, plumes: eReefs/ERA5/BoM/Queensland gauges |
| Oxygen/redox | Oxygen, saturation, hypoxia, BOD, sulfide: models and site monitoring |
| Nutrients | Nitrogen/phosphorus species and imbalance, iron: eReefs/Copernicus/Queensland/AIMS |
| Carbonate chemistry | pH, DIC, alkalinity, carbon species and saturation: eReefs/observations |
| Sediments | TSS, fines, dust, deposition, abrasion: eReefs/Queensland/AIMS |
| Food/energy | Plankton, detritus, heterotrophy, production: eReefs/physiological observations |
| Pesticides | Diuron, atrazine and other compounds: Queensland monitoring/native scenario |
| Other pollutants | Metals, hydrocarbons, sewage, sunscreen ingredients, plastics: sparse/experimental data |
| Host biology | Taxon, genotype, age, size, reserves: Coral Traits/AIMS/specimen data |
| Symbiosis/microbes | Symbiont identity, photophysiology, disease: AIMS/specimen-linked studies |
| Habitat | Depth, reef zone, substrate, orientation, cover: Reef Authority GIS/AIMS/bathymetry |
| Context | Predation, fishing, dredging, protection, ENSO/IOD: AIMS/Reef Authority/BoM |

Light/nutrients are recognized stressors ([NOAA](https://oceanservice.noaa.gov/facts/coral_bleach.html));
nutrient imbalance has experimental support ([Wiedenmann et al.](https://www.nature.com/articles/nclimate1661));
oxygen responses vary by species ([deoxygenation study](https://www.nature.com/articles/s41598-021-01078-9)).
This does not establish every candidate pollutant or contextual variable as a
direct independent bleaching cause.

[Queensland monitoring](https://www.reefplan.qld.gov.au/tracking-progress/paddock-to-reef/modelling-and-monitoring)
provides sediment, nutrient and pesticide sampling, but observations are site/event
specific. River concentrations cannot be copied to offshore cells. The
[Coral Trait Database](https://www.coraltraits.org/procedures) records individual/species
information, not a complete historical map of colony genotype and symbionts.

**All possible causes, at every coordinate, since the 1990s is not an available
dataset or a closed scientific list.** Unmeasured host biology, microbial states,
toxins and newly discovered mechanisms cannot honestly be filled into it. The
inventory records these gaps. Sparse factors need site-level analyses or validated
mapping models before admission to this all-coordinate contract.

## Biological target and model size

The model now supports a shared multivariate encoder, SST decoder and an
**observed bleaching-percent head**. Joint training uses survey targets only at
their actual cell and forecast-end date. Unsurveyed cells are not negatives;
incompatible observations are not aggregated into invented targets. Validation
reports observed-percent MAE/RMSE and counts separately from SST RMSE. Training
fails if no real bleaching targets reach its windows.

Official-boundary filtering of BCO-DMO v2 yields **2,081 GBR rows, 985 Site_ID
values and 1,865 valid targets**, dated **1992-04-15–2017-11-04**. The Parquet file
is 50,652 bytes. The global title through 2020 does not establish GBR coverage
through 2020. The current 2018 pilot cannot demonstrate biological training skill
with this file.

AIMS reports [surveyed events through 2025](https://www.aims.gov.au/research-topics/environmental-issues/coral-bleaching/coral-bleaching-events).
Some [EcoRRAP 2022/2024 data](https://researchdata.edu.au/ecorrap-coral-demographic-data/2832666)
require contact for the full dataset. Reports/maps are not substituted for compatible
dated raw labels; no email or data request was sent. Recent raw-label acquisition
remains necessary. Environmental history and labeled history are different limits.

Capacity tests with **130 feature values plus validity channels** produce:

| Requested weight budget | Actual active trainable weights |
|---:|---:|
| 1,000 | 1,000 |
| 10,000 | 10,000 |
| 100,000 | 100,000 |
| 1,000,000 | 999,998 |

Projection, recurrent width and nonlinear readout width scale without dummy
parameters. Every input enters the same projection everywhere. The tiny model
has a severe bottleneck; capacity is not predictive skill. Trainable weights are
different from the number of environmental input variables.

```sh
python run_research.py model --profile gbr_rich --parameters 1000000
python train_research.py --root data/research/<run> --output data/experiments/<new-name> --parameters 1000000 --labels data/labels_gbr/bleaching_v2_gbr.parquet
```

Checkpoints preserve architecture, feature order, normalization and the data
fingerprint. Physics-enabled training still needs actual heat forcing, currents
and mixed-layer depth; absent ERA5 fluxes are never replaced by zero.

## Verification and remaining work

The real 114-field pilot retained **14 GBR centers × 3 days × all 114 fields**.
Two bounding-rectangle centers were outside the official boundary and uniformly
excluded. It took about **114 seconds**, peaked at **234 MiB RSS**, transferred
about **17 MB** including native chunks, and stored a **453 KB** partition.
eReefs itself used 114 responses and about 1.7 MB. These are pilot measurements,
not throughput promises for full-GBR decades.

PyDAP's HTTP-session cloning initially bypassed custom byte guards. The supported
per-reader WSGI transport now routes every request through the bounded session,
without global monkeypatches or a local server. A protocol-level fixture verifies
byte counts and transfer-limit failure. The source has request timeouts, a whole-source
deadline, transfer/decode limits and no disk HTTP cache. Tests also cover cross-month
domain intersection, missing fields, shifted coordinates and all-channel gradients.

Next: acquire recent compatible GBR labels; validate newer curvilinear remapping;
resolve chemistry scales; audit wave/tide, pesticide and host-factor coverage; finish
ERA5 retrieval; run GBR-wide support audits and blocked site/time experiments.
Docker configuration parses, but its engine is unavailable, so the build remains
unverified. No trained biological predictive skill is claimed.

Branch: `codex/global-reef-data-fsm`. Merge destination: `main` only, after the
requested signal.
