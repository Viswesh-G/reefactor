# REEFS_ML — Physics-Informed ConvLSTM for Coral Bleaching Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CPU-Compatible](https://img.shields.io/badge/GPU-Not%20Required-green.svg)]()
[![Open Data](https://img.shields.io/badge/data-NOAA%20Open%20Access-blue.svg)](https://coralreefwatch.noaa.gov)

A research-grade, fully reproducible ML pipeline that predicts Sea Surface Temperature (SST) and Degree Heating Weeks (DHW) across the Great Barrier Reef using a **Physics-Informed ConvLSTM (PhyConvLSTM)**. The model's training loss includes a term enforcing the ocean Advection-Diffusion equation, using real geostrophic surface currents as the advection forcing.

> **Novel contribution (under review):** linking ocean heat-transport physics to
> satellite-derived coral stress (DHW), plus three differentiators — a *learned*
> eddy diffusivity K, a climatology-matched DHW baseline, and a physics-residual
> early-warning signal. See [ARCHITECTURE.md](ARCHITECTURE.md) for design details and
> [ANALYSIS_AND_NOVELTY_REPORT.md](ANALYSIS_AND_NOVELTY_REPORT.md) for an honest
> novelty/publishability assessment against the 2024–2026 literature.

---

## Quick Start

```bash
git clone https://github.com/[repo]/REEFS_ML
cd REEFS_ML
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS
make install
make all                       # Full pipeline: ingest → MMM → train → baselines → eval → warning → plot
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
Streams lazily from free public ERDDAP servers over OPeNDAP. Only the GBR bounding box is transferred. Runs in **strict real-only mode**: if a source is unreachable it raises rather than generating synthetic data. A list of mirror URLs is tried per chunk, with gentle pacing between requests.

| Variable | Source | ERDDAP ID |
|---|---|---|
| SST + DHW (0.05°, daily) | NOAA CoralTemp | `dhw_5km` (PacIOOS) |
| Currents u, v (0.25°, daily) | NESDIS Altimetry | `nesdisSSH1day` |

### `make derive_mmm` — Recover the true CRW MMM climatology
```bash
make derive_mmm
```
Fetches a short `CRW_SST`/`CRW_HOTSPOT` snapshot and recovers the exact per-pixel
NOAA **Maximum Monthly Mean** via `MMM = CRW_SST − CRW_HOTSPOT` (the HotSpot anomaly
is raw and time-invariant, so this is exact). Saves a ~270 KB sidecar
`data/processed/crw_mmm.npz` so our predicted DHW is computed against the *same*
climatology NOAA used — an apples-to-apples comparison without storing full HotSpot
time series (storage-minimal). All DHW comparisons automatically prefer this sidecar
and fall back to a train-split climatology if absent.

### `make train` — Train PhyConvLSTM
```bash
make train
# Resume after interruption:
venv\Scripts\python.exe run_pinns.py --resume
```
Saves `best_model.pt` and `latest_checkpoint.pt` every epoch. Writes `training_log.csv` with epoch, data_loss, phy_loss, λ_phy, **K_eff**, val_loss. With `physics_learn_K: true` the diffusivity K is learned in log-space (clamped to `[physics_K_min, physics_K_max]`) — the converged `K_eff` is a physical result we can compare against literature.

### `make baseline` — Vanilla ConvLSTM (no physics)
```bash
make baseline
```
Same architecture, MSE-only loss. Saves to `data/checkpoints/baseline/`.

### `make ablation` — Physics ablation (K=0, advection only)
```bash
make ablation
```
Same PINN loop with diffusivity `K=0` (advection only) — isolates whether diffusion specifically drives performance improvement.

### `make noaa_baseline` — NOAA CRW operational baseline
```bash
make noaa_baseline
```
Applies the NOAA DHW algorithm directly to raw Zarr SST. Establishes the operational performance ceiling without ML.

### `make eval` — Evaluate and compute metrics
```bash
make eval
```
Evaluates `best_model.pt` on the held-out validation period. Outputs:
- `eval_metrics.csv` — RMSE, ACC, NSE, DHW RMSE, DHW ACC, Bleaching Alert F1/IoU
- `eval_lead_rmse.csv` — per-day RMSE for Figure 2
- `eval_comparison.png` — True vs. Predicted vs. Difference maps

### `make early_warning` — Physics-residual precursor analysis
```bash
make early_warning
```
Computes the spatial-mean advection-diffusion residual over the continuous forecast
timeline and cross-correlates it against the DHW accumulation rate across lead times.
If the residual rises before heat stress accumulates, it is an interpretable,
physics-grounded early-warning signal that a black-box model cannot provide. Outputs
`early_warning_series.csv`, `early_warning_leadlag.csv`, and `early_warning.png`.

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
| `figure4_bleaching_alert_overlay.png` | GBR bleaching-alert overlay for the validation period |

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
  start_date: "2017-02-13"   # NESDIS altimetry currents begin here
  end_date:   "2018-12-31"

dataset:
  lookback: 14       # Days of past SST → encoder input
  horizon: 7         # Days ahead to predict
  val_fraction: 0.2  # Last 20% of time axis → held-out validation period
  patch_size: 64     # Spatial crop for CPU training; set to null for full GBR (GPU)

training:
  epochs: 50
  batch_size: 4
  learning_rate: 0.001
  # SI units: K in m^2/s, dx/dy in m, dt in s (K~500 typical at 5km grids)
  physics_K: 500.0
  physics_learn_K: true    # learn K in log-space, clamped to [K_min, K_max]
  physics_K_min: 10.0
  physics_K_max: 5000.0
  ema_alpha: 0.9           # EMA smoothing for dynamic λ_phy balancing
```

---

## Experimental Conditions

| Condition | Script | physics_K | λ_phy balancing |
|---|---|---|---|
| **PhyConvLSTM (ours)** | `run_pinns.py` | learned (init 500.0) | ✅ Wang et al. 2021 |
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
├── run_derive_mmm.py            ← Recover true CRW MMM climatology sidecar
├── run_pinns.py                 ← PINN training (--resume supported)
├── run_baseline.py              ← Vanilla ConvLSTM baseline
├── run_ablation.py              ← Physics ablation (K=0)
├── run_noaa_dhw_baseline.py     ← NOAA CRW operational baseline
├── run_eval.py                  ← Evaluation + metric tables
├── run_early_warning.py         ← Physics-residual early-warning analysis
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
