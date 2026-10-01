# REEFS_ML — Physics-Informed Neural Network for Coral Bleaching Prediction

A research-grade machine learning pipeline that predicts Sea Surface Temperature (SST) and Degree Heating Weeks (DHW) across the Great Barrier Reef, using a **Physics-Informed Neural Network (PINN)** — a ConvLSTM model whose training loss includes a term enforcing the Advection-Diffusion equation of ocean heat transport.

The project is designed for a comparative research paper benchmarking PINN predictions against the standard NOAA DHW threshold metric.

---

## Why a PINN?

NOAA's operational coral bleaching alert system uses a simple accumulator: if SST exceeds the local climatological maximum by 1°C, it counts as a "hot day," and 12 weeks of hot days constitute a bleaching alert. This works but ignores the physical movement of heat — ocean currents can advect thermal stress into or away from a reef faster than a static threshold detects.

A PINN adds a physics penalty to the loss function:

```
Total Loss = MSE(predictions, observed_SST)
           + λ · Residual(Advection-Diffusion equation)
```

The Advection-Diffusion residual checks that the rate of temperature change in each predicted grid cell is consistent with the local ocean current velocity field. If a prediction is physically impossible (e.g., heat accumulates where currents should be dispersing it), the model is penalised.

This makes the model extrapolate better to extreme events (marine heatwaves) it has never seen, because it cannot cheat its way to low training loss with non-physical shortcuts.

---

## Project Structure

```
REEFS_ML/
├── config.yaml           ← Single control panel: region, time range, data sources, hyperparams
├── run_ingestion.py      ← Entry point: validate config → stream satellite data → write Zarr
├── run_pinns.py          ← Entry point: load Zarr → build model → train → save checkpoints
├── run_eval.py           ← Entry point: load checkpoint → compute metrics → save plot
├── Makefile              ← Shortcuts: make ingest | train | eval | install | clean
├── requirements.txt      ← Python dependencies
│
├── docs/
│   ├── README.md                              ← This file
│   ├── PINN_Coral_Bleaching_Research_History.md
│   └── PINN_Coral_Bleaching_UNABRIDGED_History.md
│
└── src/
    ├── config_validator.py   ← Validates config.yaml before any heavy operations run
    ├── data_pipeline.py      ← FSM: OPeNDAP stream → resample → validate → save Zarr
    ├── dataset.py            ← PyTorch Dataset: sliding-window samples from the Zarr store
    ├── models.py             ← 2-layer ConvLSTM encoder-decoder architecture
    ├── physics.py            ← Custom PhyLoss: Advection-Diffusion residual via fixed conv kernels
    ├── train.py              ← Training loop with dynamic gradient-annealing + checkpointing
    └── metrics.py            ← RMSE, ACC, NSE, and Degree Heating Weeks computation
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
make install
# or: pip install -r requirements.txt
```

---

## Running the Pipeline

The full pipeline runs in three sequential steps:

### Step 1 — Ingest Data

```bash
make ingest
```

Streams satellite data directly from free public ERDDAP servers over OPeNDAP — **no files are downloaded to disk**. The resulting unified dataset is written to `data/processed/unified_gbr_canvas.zarr`.

**Data sources (all free, no login required):**

| Variable | Source | ERDDAP Dataset |
|---|---|---|
| Sea Surface Temperature (daily, 5km) | NOAA Coral Reef Watch | `NOAA_DHW_5km` |
| Degree Heating Weeks (pre-computed) | NOAA Coral Reef Watch | `NOAA_DHW_5km` |
| Ocean Surface Currents (5-day, 0.33°) | OSCAR / JPL | `jplOscar` |

If either server is unreachable, the pipeline automatically falls back to physically realistic synthetic data so you can continue developing and testing.

### Step 2 — Train

```bash
make train
```

Trains the ConvLSTM-PINN for the number of epochs set in `config.yaml`. A checkpoint is saved after every epoch. The model with the lowest validation loss is saved separately as `data/checkpoints/best_model.pt`.

Training logs print: Data Loss | Physics Loss | λ_phy (dynamic weight) | Validation Loss.

### Step 3 — Evaluate

```bash
make eval
```

Loads `best_model.pt`, runs inference on the held-out validation split, and prints:

| Metric | What it measures |
|---|---|
| RMSE (°C) | Average SST prediction error |
| Anomaly Correlation (ACC) | How well spatial/temporal patterns are captured |
| Nash-Sutcliffe Efficiency | Skill vs. always predicting the mean (1.0 = perfect, 0.0 = no skill) |
| DHW RMSE (°C-weeks) | Error in derived Degree Heating Weeks vs. NOAA ground truth |
| DHW Correlation | How well bleaching stress patterns are predicted |

Also saves `data/checkpoints/eval_comparison.png` — a 3-panel map showing True SST, Predicted SST, and their difference for the first validation sample.

---

## Configuration

All parameters live in `config.yaml`. Key settings:

```yaml
spatial_canvas:
  lat_min: -24.0   # Great Barrier Reef bounding box
  lat_max: -10.0
  lon_min: 142.0
  lon_max: 154.0
  target_resolution: 0.05   # Degrees (~5km)

temporal_canvas:
  start_date: "2020-01-01"
  end_date:   "2020-12-31"

dataset:
  lookback: 14        # Days of past SST fed to encoder
  horizon: 7          # Days ahead to predict
  val_fraction: 0.2   # Last 20% of time for validation
  patch_size: null    # null = full GBR; set to 64 for CPU testing

training:
  epochs: 50
  batch_size: 4
  physics_K: 0.05     # Thermal diffusivity in the Advection-Diffusion equation
```

---

## Architecture

```
Past 14 days of SST
   │
   ▼
┌──────────────────────────────────────────┐
│  SST_ConvLSTM  (Encoder → Decoder)       │
│                                          │
│  Encoder: ConvLSTMCell × 2 layers        │
│    → reads 14 past SST maps, builds      │
│      spatiotemporal memory state         │
│                                          │
│  Decoder: ConvLSTMCell × 2 layers        │
│    → predicts 7 future SST maps          │
│      autoregressively                    │
│                                          │
│  final_conv: 64ch → 1ch per pixel        │
└──────────────────────────────────────────┘
   │
   ▼
Predicted SST for next 7 days

Loss = MSE(pred, true_SST)
     + λ · PhyLoss(pred, u_current, v_current)

PhyLoss checks: dT/dt + u·∂T/∂x + v·∂T/∂y = K·∇²T
(Advection-Diffusion equation)
λ is dynamically adjusted each batch via gradient annealing.
```

---

## Key Design Choices

**OPeNDAP streaming, no bulk downloads.**
All satellite data is fetched lazily over the network. Only the exact spatial bounding box and time range requested are transferred. The entire 1-year GBR SST dataset compresses to roughly 200–400 MB in Zarr format (chunked, float32).

**Zarr over NetCDF.**
Zarr is chunk-optimised for time-series access: loading a 14-day sliding window reads exactly one chunk, not the whole file.

**Chronological train/val split.**
For time series, shuffling across the split boundary leaks future information into training. The last 20% of the time axis is always held out.

**Normalization stats from training split only.**
Computing mean/std on the full dataset (including validation) would be data leakage. Stats are computed from the first 80% only and applied consistently.

**Dynamic gradient annealing (Wang et al., 2021).**
The physics weight λ is adjusted each batch by comparing gradient magnitudes, preventing either loss from dominating and ensuring the physics constraint is always actively shaping the model.

---

## Research Paper Benchmarks

For the paper, the PINN should be compared against:

1. **NOAA DHW operational algorithm** — the current operational standard (no ML)
2. **Vanilla ConvLSTM** — same architecture, MSE loss only, no physics penalty
3. **Ablation: PINN with λ = 0** — confirm physics loss is what drives improvement
4. **Extreme heatwave generalisation** — test on 2016 or 2022 bleaching events (out-of-distribution)

---

## References

- Wang, S., Teng, Y., Perdikaris, P. (2021). *Understanding and mitigating gradient pathologies in physics-informed neural networks.* SIAM Journal on Scientific Computing.
- Shi, X. et al. (2015). *Convolutional LSTM network: A machine learning approach for precipitation nowcasting.* NeurIPS.
- NOAA Coral Reef Watch: https://coralreefwatch.noaa.gov
- NOAA CoastWatch ERDDAP: https://coastwatch.pfeg.noaa.gov/erddap
