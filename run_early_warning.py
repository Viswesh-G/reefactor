"""
run_early_warning.py — Physics-residual early-warning differentiator (PIVOT).

IDEA
----
A well-trained PINN should only violate the heat-transport physics when the ocean
is doing something the model cannot explain — e.g. the onset of a marine heatwave
before it shows up as accumulated heat stress. If the spatial-mean |advection-
diffusion residual| rises BEFORE the DHW accumulation rate peaks, the residual is
an interpretable, physics-grounded *early-warning precursor* — something a pure
black-box LSTM cannot provide.

WHAT THIS DOES
--------------
  1. Loads the trained PINN checkpoint.
  2. Runs the FULL-grid validation split, denormalizes predictions to degC.
  3. Reconstructs the continuous daily-mean forecast timeline (preds, u, v).
  4. Computes: predicted DHW, its daily rate, and the physics residual series.
  5. Cross-correlates residual(t) against DHW-rate(t + lead) to find the lead time
     at which the residual best predicts accelerating heat stress.
  6. Saves a CSV + a 2-panel figure (time series; correlation vs lead).

Usage:
    python run_early_warning.py
"""
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config_validator import ConfigValidator
from src.models import SST_ConvLSTM
from src.dataset import ReefDataset
from src.reproducibility import set_seed, get_seed, get_device
from src.metrics import (
    load_crw_mmm,
    compute_mmm_climatology,
    compute_dhw_from_sst,
    reconstruct_continuous,
    physics_residual_series,
)


def _spatial_mean_series(field_tHW):
    """(T, H, W) -> (T,) spatial mean (NaN-safe)."""
    return np.nanmean(np.nanmean(field_tHW, axis=-1), axis=-1)


def _cross_corr_lead(residual, dhw_rate, max_lead=21):
    """Correlation of residual(t) with dhw_rate(t+lead) for leads 0..max_lead."""
    L = min(len(residual), len(dhw_rate))
    r = (residual[:L] - residual[:L].mean()) / (residual[:L].std() + 1e-12)
    d = (dhw_rate[:L] - dhw_rate[:L].mean()) / (dhw_rate[:L].std() + 1e-12)
    leads, corrs = [], []
    for lead in range(0, max_lead + 1):
        a = r[:L - lead]
        b = d[lead:L]
        if len(a) < 5:
            break
        leads.append(lead)
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(leads), np.array(corrs)


def main():
    validator = ConfigValidator("config.yaml")
    config    = validator.validate()
    set_seed(get_seed(config))
    device    = get_device(config)

    zarr_path       = config["storage"]["processed_zarr_path"]
    checkpoint_dir  = config["storage"]["checkpoint_dir"]
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.exists(best_model_path):
        print(f"\nNo trained model found at '{best_model_path}'. Run `make train` first.\n")
        return

    ds_cfg       = config.get("dataset", {})
    lookback     = ds_cfg.get("lookback", 14)
    horizon      = ds_cfg.get("horizon", 7)
    val_fraction = ds_cfg.get("val_fraction", 0.2)

    tr = config.get("training", {})
    dx = float(tr.get("physics_dx", 5500.0))
    dy = float(tr.get("physics_dy", 5500.0))
    dt = float(tr.get("physics_dt", 86400.0))
    K  = float(tr.get("physics_K", 500.0))

    print(f"\nLoading validation split from: {zarr_path}")
    try:
        val_dataset = ReefDataset(zarr_path, lookback=lookback, horizon=horizon,
                                  split="val", val_fraction=val_fraction, patch_size=None)
    except Exception as e:
        print(f"Could not load Zarr data: {e}\nRun `make ingest` first.\n")
        return

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=horizon)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Collect full-grid windows (preds, u, v) ---
    preds_list, u_list, v_list = [], [], []
    with torch.no_grad():
        for x_sst, _y, u_vel, v_vel in val_loader:
            preds_list.append(model(x_sst.to(device)).cpu())
            u_list.append(u_vel)
            v_list.append(v_vel)

    preds = torch.cat(preds_list, dim=0)                 # (N, H, 1, H, W)
    u_all = torch.cat(u_list, dim=0)                     # (N, H, 1, H, W)
    v_all = torch.cat(v_list, dim=0)

    # Denormalize predictions to degC
    preds_c = preds * val_dataset.sst_std + val_dataset.sst_mean

    # Continuous daily-mean timeline: (T, 1, H, W) -> (T, H, W)
    preds_t = reconstruct_continuous(preds_c).squeeze(1).numpy()
    u_t     = reconstruct_continuous(u_all).squeeze(1).numpy()
    v_t     = reconstruct_continuous(v_all).squeeze(1).numpy()

    # --- Predicted DHW series (climatology-matched) ---
    import xarray as xr
    ds = xr.open_zarr(zarr_path, consolidated=True)
    n_times   = ds.sizes["time"]
    val_start = int(n_times * (1.0 - val_fraction))
    sst_train_da = ds["sea_surface_temperature"].isel(time=slice(0, val_start))
    mmm = load_crw_mmm(config)
    if mmm is None:
        mmm = compute_mmm_climatology(sst_train_da)

    dhw_pred = compute_dhw_from_sst(preds_t, climatology_mean=mmm)   # (T, H, W)
    dhw_series = _spatial_mean_series(dhw_pred)
    dhw_rate   = np.diff(dhw_series)

    # --- Physics residual series ---
    residual = physics_residual_series(preds_t, u_t, v_t, dx=dx, dy=dy, dt=dt, K=K)

    # --- Lead-lag analysis ---
    leads, corrs = _cross_corr_lead(residual, dhw_rate)
    best = int(np.argmax(corrs)) if len(corrs) else 0
    print(f"\nResidual vs DHW-rate: best lead = {leads[best]} day(s), "
          f"r = {corrs[best]:.3f} (lead 0 r = {corrs[0]:.3f})")

    # --- Save CSV ---
    L = min(len(dhw_series) - 1, len(residual))
    series_csv = os.path.join(checkpoint_dir, "early_warning_series.csv")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(series_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "pred_dhw_mean", "dhw_rate", "physics_residual"])
        for t in range(L):
            w.writerow([t, f"{dhw_series[t]:.6f}", f"{dhw_rate[t]:.6f}", f"{residual[t]:.6f}"])
    print(f"  Series saved: {series_csv}")

    lead_csv = os.path.join(checkpoint_dir, "early_warning_leadlag.csv")
    with open(lead_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lead_days", "correlation"])
        for ld, c in zip(leads, corrs):
            w.writerow([int(ld), f"{c:.6f}"])
    print(f"  Lead-lag saved: {lead_csv}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        ax = axes[0]
        ax.plot(dhw_series, color="#c0392b", label="Predicted DHW (spatial mean)")
        ax.set_ylabel("DHW (°C-weeks)", color="#c0392b")
        ax.tick_params(axis="y", labelcolor="#c0392b")
        ax2 = ax.twinx()
        ax2.plot(residual, color="#2471a3", alpha=0.8, label="Physics residual |r|")
        ax2.set_ylabel("|advection-diffusion residual|", color="#2471a3")
        ax2.tick_params(axis="y", labelcolor="#2471a3")
        ax.set_xlabel("Forecast day")
        ax.set_title("DHW accumulation vs physics residual")

        ax = axes[1]
        if len(leads):
            ax.bar(leads, corrs, color="#7d3c98")
            ax.axvline(leads[best], color="k", ls="--", lw=1,
                       label=f"best lead = {leads[best]}d (r={corrs[best]:.2f})")
            ax.legend()
        ax.set_xlabel("Lead (days): residual(t) vs DHW-rate(t+lead)")
        ax.set_ylabel("Correlation")
        ax.set_title("Lead-lag cross-correlation")

        plt.tight_layout()
        plot_path = os.path.join(checkpoint_dir, "early_warning.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {plot_path}")
    except ImportError:
        print("(matplotlib not found — no plot saved.)")


if __name__ == "__main__":
    main()
