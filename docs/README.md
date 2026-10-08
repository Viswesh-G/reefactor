# REEFS_ML — Physics-Informed ConvLSTM for Coral Bleaching Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CPU-Compatible](https://img.shields.io/badge/GPU-Not%20Required-green.svg)]()
[![Open Data](https://img.shields.io/badge/data-NOAA%20Open%20Access-blue.svg)](https://coralreefwatch.noaa.gov)

A research-grade, fully reproducible ML pipeline that predicts Sea Surface Temperature (SST) and Degree Heating Weeks (DHW) across the Great Barrier Reef using a **Physics-Informed ConvLSTM (PhyConvLSTM)**. The model's training loss includes a term enforcing the ocean Advection-Diffusion equation, using real geostrophic surface currents as the advection forcing.

> **Novel contribution:** The first GBR-scale PINN to link ocean heat transport physics directly to satellite-derived biological coral stress metrics (DHW). See [ARCHITECTURE.md](ARCHITECTURE.md) for design details.

---

## Quick Start

```bash
git clone https://github.com/[repo]/REEFS_ML
cd REEFS_ML
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS
make install
make all                       # Full pipeline: ingest → train → eval → plot
```

> No GPU required. No proprietary data. No API keys.

---

## The Research Problem

NOAA's operational coral bleaching alert system — Degree Heating Weeks (DHW) — accumulates thermal stress using a static threshold applied to raw satellite SST. It ignores the physical movement of heat by ocean currents. A marine heatwave can be *advected* into a reef faster than the static accumulator detects, and a bleaching event can be missed.

**PhyConvLSTM-DHW** adds an advection-diffusion physics residual to the ConvLSTM loss:

```
Total Loss = MSE(ŷ, y_SST)  +  λ_phy · ||∂T/∂t + u·∂T/∂x + v·∂T/∂y − K·∇²T||²
```

where `(u, v)` are the real NESDIS geostrophic surface current fields. The physics weight `λ_phy` is dynamically adjusted each batch via gradient-magnitude annealing (Wang et al., 2021) to prevent gradient pathology.

---

## Pipeline Steps

### `make install` — Install dependencies
```bash
make install
```

### `make ingest` — Stream satellite data → Zarr
```bash
make ingest
```
Streams lazily from free public ERDDAP servers over OPeNDAP. Only the GBR bounding box is transferred. Falls back to synthetic data if servers are unreachable.

| Variable | Source | ERDDAP ID |
|---|---|---|
| SST + DHW (0.05°, daily) | NOAA CoralTemp | `NOAA_DHW` |
| Currents u, v (0.25°, daily) | NESDIS Altimetry | `nesdisSSH1day` |

### `make train` — Train PhyConvLSTM
```bash
make train
# Resume after interruption:
venv\Scripts\python.exe run_pinns.py --resume
```
Saves `best_model.pt` and `latest_checkpoint.pt` every epoch. Writes `training_log.csv` with epoch, data_loss, phy_loss, λ_phy, val_loss.

### `make baseline` — Vanilla ConvLSTM (no physics)
```bash
make baseline
```
Same architecture, MSE-only loss. Saves to `data/checkpoints/baseline/`.

### `make ablation` — Physics ablation (K=0, advection only)
```bash
make ablation
```
Same PINN loop with diffusivity K=0 — isolates whether the diffusion term specifically drives performance improvement.

### `make noaa_baseline` — NOAA CRW operational baseline
```bash
make noaa_baseline
```
Applies the NOAA DHW algorithm directly to raw Zarr SST. Establishes the operational performance ceiling without ML.

### `make eval` — Evaluate and compute metrics
```bash
make eval
```
Evaluates `best_model.pt` on the held-out 2023 validation year. Outputs:
- `eval_metrics.csv` — RMSE, ACC, NSE, DHW RMSE, DHW ACC, Bleaching Alert F1/IoU
- `eval_lead_rmse.csv` — per-day RMSE for Figure 2
- `eval_comparison.png` — True vs. Predicted vs. Difference maps

### `make plot` — Generate all paper figures
```bash
make plot
```
Generates four publication-ready figures in `data/checkpoints/figures/`:

| Figure | Content |
|---|---|
| `figure1_dhw_spatial_comparison.png` | NOAA vs. PhyConvLSTM vs. Vanilla ConvLSTM DHW maps |
| `figure2_lead_time_rmse.png` | RMSE vs. forecast lead day (1–7) |
| `figure3_lambda_convergence.png` | λ_phy convergence + validation loss curves |
| `figure4_bleaching_alert_overlay.png` | 2023 GBR bleaching event alert overlay |

---

## Configuration

All parameters live in `config.yaml`. Nothing is hardcoded.

```yaml
spatial_canvas:
  lat_min: -24.0         # GBR bounding box
  lat_max: -10.0
  lon_min: 142.0
  lon_max: 154.0
  target_resolution: 0.05  # ~5km, matches NOAA CoralTemp native grid

temporal_canvas:
  start_date: "2017-01-01"
  end_date:   "2023-12-31"

dataset:
  lookback: 14       # Days of past SST → encoder input
  horizon: 7         # Days ahead to predict
  val_fraction: 0.2  # Last 20% of time axis → validation (2023 = record bleaching year)
  patch_size: 64     # Spatial crop for CPU training; set to null for full GBR (GPU)

training:
  epochs: 50
  batch_size: 4
  learning_rate: 0.001
  physics_K: 0.05    # Thermal diffusivity K in dT/dt + u·∂T/∂x + v·∂T/∂y = K·∇²T
  ema_alpha: 0.9     # EMA smoothing for dynamic λ_phy balancing
```

---

## Experimental Conditions

| Condition | Script | physics_K | λ_phy balancing |
|---|---|---|---|
| **PhyConvLSTM (ours)** | `run_pinns.py` | 0.05 | ✅ Wang et al. 2021 |
| Ablation (advection only) | `run_ablation.py` | 0.00 | ✅ Wang et al. 2021 |
| Vanilla ConvLSTM | `run_baseline.py` | N/A | ❌ no physics |
| NOAA CRW operational | `run_noaa_dhw_baseline.py` | N/A | N/A |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| RMSE (°C) | Root mean square SST prediction error |
| ACC | Anomaly Correlation Coefficient — spatial/temporal pattern fidelity |
| NSE | Nash-Sutcliffe Efficiency — skill vs. always-predict-mean baseline |
| DHW RMSE (°C-weeks) | Error in accumulated thermal stress |
| DHW ACC | Spatial correlation of bleaching hotspot maps |
| DHW>4 F1 / IoU | Alert Level 1 classification agreement |
| DHW>8 F1 / IoU | Alert Level 2 classification agreement |

---

## Repository Structure

```
REEFS_ML/
├── config.yaml                  ← SINGLE control panel
├── run_all.py                   ← Full pipeline orchestrator (--from phase)
├── run_ingestion.py             ← Data ingest entry point
├── run_pinns.py                 ← PINN training (--resume supported)
├── run_baseline.py              ← Vanilla ConvLSTM baseline
├── run_ablation.py              ← Physics ablation (K=0)
├── run_noaa_dhw_baseline.py     ← NOAA CRW operational baseline
├── run_eval.py                  ← Evaluation + metric tables
├── run_plot_figures.py          ← Paper figure generation
├── Makefile                     ← make all/train/eval/...
├── requirements.txt
├── src/
│   ├── config_validator.py
│   ├── data_pipeline.py         ← DataIngestionFSM
│   ├── dataset.py               ← ReefDataset (random patch support)
│   ├── models.py                ← SST_ConvLSTM
│   ├── physics.py               ← PhyLoss (advection-diffusion)
│   ├── train.py                 ← train_pinn, resume_from_checkpoint, count_parameters
│   ├── metrics.py               ← RMSE, ACC, NSE, DHW, bleaching_alert_agreement
│   └── ablation.py              ← train_ablation (configurable K)
└── docs/
    ├── README.md                ← This file
    └── ARCHITECTURE.md          ← Deep-dive: data flow, design decisions, extension guide
```

---

## References

- Wang, S., Teng, Y., Perdikaris, P. (2021). *Understanding and mitigating gradient pathologies in physics-informed neural networks.* SIAM Journal on Scientific Computing.
- Shi, X. et al. (2015). *Convolutional LSTM network: A machine learning approach for precipitation nowcasting.* NeurIPS.
- Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). *Physics-informed neural networks.* Journal of Computational Physics.
- Hughes, T.P. et al. (2018). *Global warming transforms coral reef assemblages.* Nature, 556, 492–496.
- NOAA Coral Reef Watch. https://coralreefwatch.noaa.gov
- NOAA CoastWatch ERDDAP. https://coastwatch.pfeg.noaa.gov/erddap
