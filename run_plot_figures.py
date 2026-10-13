"""
run_plot_figures.py — Generate all paper figures from saved checkpoints

Produces the four key figures described in the-idea.md:

    Figure 1: Spatial DHW map comparison
              (NOAA ground truth vs. PhyConvLSTM vs. Vanilla ConvLSTM)

    Figure 2: Lead-time RMSE decay curves
              (all baselines vs. PhyConvLSTM, error vs. forecast day 1–7)

    Figure 3: λ_phy convergence plot over training epochs
              (physics weight trajectory, loaded from training_log.csv)

    Figure 4: Bleaching alert overlay for the peak stress period
              (DHW > 4 and DHW > 8 alert contours over GBR map)

Usage:
    python run_plot_figures.py
  or
    make plot

All figures are saved to data/checkpoints/figures/
"""
import os
import csv
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from src.config_validator import ConfigValidator
from src.models import SST_ConvLSTM
from src.dataset import ReefDataset
from src.reproducibility import set_seed, get_seed
from src.metrics import (
    rmse,
    compute_dhw_from_sst,
    compute_mmm_climatology,
    load_crw_mmm,
    reconstruct_continuous_forecast,
)


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def load_training_log(csv_path: str) -> dict:
    """Read a training_log.csv and return columns as lists of floats."""
    data = {"epoch": [], "data_loss": [], "phy_loss": [], "lambda_phy": [], "val_loss": []}
    if not os.path.exists(csv_path):
        return data
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                try:
                    data[key].append(float(row[key]))
                except (ValueError, KeyError):
                    data[key].append(float("nan"))
    return data


def collect_preds(model, val_loader, val_dataset):
    """Run inference on the full validation set; return denormalized tensors."""
    all_preds   = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for x_sst, y_sst, _, _ in val_loader:
            preds = model(x_sst)
            all_preds.append(preds)
            all_targets.append(y_sst)
    all_preds   = torch.cat(all_preds,   dim=0)   # (N, horizon, 1, H, W)
    all_targets = torch.cat(all_targets, dim=0)
    # Denormalize
    preds_c   = all_preds   * val_dataset.sst_std + val_dataset.sst_mean
    targets_c = all_targets * val_dataset.sst_std + val_dataset.sst_mean
    return preds_c, targets_c


# ────────────────────────────────────────────────────────────────
# Figure 1: Spatial DHW map comparison
# ────────────────────────────────────────────────────────────────

def plot_figure1(ds, pinn_preds_np, baseline_preds_np, clim_per_pixel,
                 val_start, lookback, out_dir):
    """
    Three-panel DHW map comparison at the last validation time step:
        Left:   NOAA Ground-Truth DHW
        Middle: PhyConvLSTM predicted DHW
        Right:  Vanilla ConvLSTM predicted DHW
    """
    print("  → Figure 1: Spatial DHW map comparison")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dhw_true_full = ds["degree_heating_weeks"].isel(time=slice(val_start + lookback, None)).values

    dhw_pinn     = compute_dhw_from_sst(pinn_preds_np,     clim_per_pixel)
    dhw_baseline = compute_dhw_from_sst(baseline_preds_np, clim_per_pixel)

    min_len = min(dhw_true_full.shape[0], dhw_pinn.shape[0], dhw_baseline.shape[0])
    true_map = dhw_true_full[min_len - 1]
    pinn_map = dhw_pinn[min_len - 1]
    base_map = dhw_baseline[min_len - 1]

    lats = ds.lat.values
    lons = ds.lon.values
    vmax = max(true_map.max(), pinn_map.max(), base_map.max(), 4.0)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Figure 1 — Degree Heating Weeks: Spatial Comparison (Validation Period End)",
                 fontsize=13, fontweight="bold")

    titles = [
        "NOAA Ground-Truth DHW\n(Satellite Observation)",
        "PhyConvLSTM Predicted DHW\n(This Work — Physics Constrained)",
        "Vanilla ConvLSTM Predicted DHW\n(Baseline — No Physics)",
    ]
    maps = [true_map, pinn_map, base_map]

    for ax, title, data in zip(axes, titles, maps):
        im = ax.pcolormesh(lons, lats, data, vmin=0, vmax=vmax, cmap="YlOrRd")
        # Bleaching alert contours (DHW > 4 and DHW > 8)
        ax.contour(lons, lats, data, levels=[4.0], colors="red",   linewidths=1.0, linestyles="--")
        ax.contour(lons, lats, data, levels=[8.0], colors="maroon",linewidths=1.5, linestyles="-")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°S)")
        plt.colorbar(im, ax=ax, fraction=0.046, label="DHW (°C-weeks)")

    plt.tight_layout()
    path = os.path.join(out_dir, "figure1_dhw_spatial_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {path}")


# ────────────────────────────────────────────────────────────────
# Figure 2: Lead-time RMSE decay curves
# ────────────────────────────────────────────────────────────────

def plot_figure2(pinn_preds, pinn_targets, baseline_preds, baseline_targets,
                 horizon, out_dir):
    """
    RMSE vs. forecast lead day (1 to horizon) for both models.
    Shows how error accumulates as the prediction horizon grows.
    """
    print("  → Figure 2: Lead-time RMSE decay curves")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    def lead_rmse(preds, targets, h):
        return [
            rmse(preds[:, d, :, :, :], targets[:, d, :, :, :])
            for d in range(h)
        ]

    days          = list(range(1, horizon + 1))
    pinn_rmse     = lead_rmse(pinn_preds,     pinn_targets,     horizon)
    baseline_rmse = lead_rmse(baseline_preds, baseline_targets, horizon)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(days, pinn_rmse,     "o-",  color="#E94F37", lw=2, ms=7, label="PhyConvLSTM (ours)")
    ax.plot(days, baseline_rmse, "s--", color="#393E41", lw=2, ms=7, label="Vanilla ConvLSTM")
    ax.set_xlabel("Forecast Lead Day", fontsize=12)
    ax.set_ylabel("RMSE (°C)", fontsize=12)
    ax.set_title("Figure 2 — SST Forecast RMSE vs. Lead Day", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "figure2_lead_time_rmse.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {path}")


# ────────────────────────────────────────────────────────────────
# Figure 3: λ_phy convergence over training
# ────────────────────────────────────────────────────────────────

def plot_figure3(pinn_log_path, baseline_log_path, out_dir):
    """
    Dual-panel plot:
        Top:    λ_phy trajectory (physics weight) over epochs (PhyConvLSTM only)
        Bottom: Validation loss comparison (PhyConvLSTM vs. Vanilla ConvLSTM)
    """
    print("  → Figure 3: λ_phy convergence and validation loss")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pinn_log = load_training_log(pinn_log_path)
    base_log = load_training_log(baseline_log_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=False)
    fig.suptitle("Figure 3 — Training Dynamics", fontsize=13, fontweight="bold")

    # Top panel: λ_phy
    if pinn_log["epoch"]:
        ax1.plot(pinn_log["epoch"], pinn_log["lambda_phy"],
                 color="#E94F37", lw=2, label="λ_phy (PhyConvLSTM)")
        ax1.axhline(1.0, color="gray", lw=1, ls="--", label="Initial λ = 1.0")
    ax1.set_ylabel("Physics Weight λ_phy", fontsize=11)
    ax1.set_title("Dynamic Physics Weight Convergence (Wang et al. 2021 EMA)", fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Epoch")

    # Bottom panel: val loss comparison
    if pinn_log["epoch"]:
        ax2.plot(pinn_log["epoch"], pinn_log["val_loss"],
                 "o-", color="#E94F37", lw=2, ms=4, label="PhyConvLSTM Val Loss")
    if base_log["epoch"]:
        ax2.plot(base_log["epoch"], base_log["val_loss"],
                 "s--", color="#393E41", lw=2, ms=4, label="Vanilla ConvLSTM Val Loss")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation MSE Loss", fontsize=11)
    ax2.set_title("Validation Loss Convergence", fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "figure3_lambda_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {path}")


# ────────────────────────────────────────────────────────────────
# Figure 4: Bleaching alert overlay for the validation period peak stress
# ────────────────────────────────────────────────────────────────

def plot_figure4(ds, pinn_preds_np, clim_per_pixel, val_start, lookback, out_dir):
    """
    Bleaching alert level overlay for the peak stress period in the validation year.
    Shows Alert Level 1 (DHW > 4) and Alert Level 2 (DHW > 8) regions,
    matching the NOAA Bleaching Alert Area product convention.
    """
    print("  → Figure 4: Validation-period bleaching alert overlay")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dhw_pred = compute_dhw_from_sst(pinn_preds_np, clim_per_pixel)
    dhw_true = ds["degree_heating_weeks"].isel(time=slice(val_start + lookback, None)).values

    # Find the time step with the highest spatial mean DHW (peak stress)
    peak_idx  = int(np.nanmean(dhw_pred, axis=(1, 2)).argmax())
    pred_peak = dhw_pred[peak_idx]
    true_peak = dhw_true[min(peak_idx, dhw_true.shape[0] - 1)]

    lats = ds.lat.values
    lons = ds.lon.values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Figure 4 — GBR Bleaching Event: Alert Level Overlay\n"
        "(Red = Alert Level 1, DHW > 4 °C-wks; Dark Red = Alert Level 2, DHW > 8 °C-wks)",
        fontsize=11, fontweight="bold"
    )

    vmax = max(pred_peak.max(), true_peak.max(), 8.0)

    for ax, data, title in zip(axes,
                                [true_peak, pred_peak],
                                ["NOAA Ground-Truth DHW", "PhyConvLSTM Predicted DHW"]):
        im = ax.pcolormesh(lons, lats, data, vmin=0, vmax=vmax, cmap="YlOrRd", alpha=0.85)
        ax.contourf(lons, lats, data, levels=[4.0, 8.0], colors=["#FF4444"],  alpha=0.35)
        ax.contourf(lons, lats, data, levels=[8.0, vmax+1], colors=["#8B0000"], alpha=0.5)
        ax.contour(lons, lats, data,  levels=[4.0], colors="red",   lw=1.5, linestyles="--")
        ax.contour(lons, lats, data,  levels=[8.0], colors="maroon",lw=2.0, linestyles="-")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°S)")
        plt.colorbar(im, ax=ax, fraction=0.046, label="DHW (°C-weeks)")

    plt.tight_layout()
    path = os.path.join(out_dir, "figure4_bleaching_alert_overlay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {path}")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    print("=== Generating Paper Figures ===\n")

    validator = ConfigValidator("config.yaml")
    config    = validator.validate()
    set_seed(get_seed(config))

    zarr_path      = config["storage"]["processed_zarr_path"]
    checkpoint_dir = config["storage"]["checkpoint_dir"]
    val_fraction   = config.get("dataset", {}).get("val_fraction", 0.2)
    ds_cfg         = config.get("dataset", {})
    lookback       = ds_cfg.get("lookback", 14)
    horizon        = ds_cfg.get("horizon", 7)
    patch_size     = ds_cfg.get("patch_size", None)

    out_dir = os.path.join(checkpoint_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Figure output directory: {out_dir}\n")

    # --- Load Zarr ---
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except Exception as e:
        print(f"Could not open Zarr store ({e}). Run `make ingest` first.")
        return

    n_times   = ds.dims["time"]
    val_start = int(n_times * (1.0 - val_fraction))

    # --- Per-pixel climatology (prefer TRUE CRW MMM; else training split) ---
    sst_train_da   = ds["sea_surface_temperature"].isel(time=slice(0, val_start))
    clim_per_pixel = load_crw_mmm(config)
    if clim_per_pixel is None:
        clim_per_pixel = compute_mmm_climatology(sst_train_da)  # (H, W)

    # --- Load validation dataset ---
    try:
        val_dataset = ReefDataset(
            zarr_path, lookback=lookback, horizon=horizon,
            split="val", val_fraction=val_fraction, patch_size=None,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Could not load ReefDataset ({e}). Run `make ingest` first.")
        return

    # --- Load PhyConvLSTM model ---
    pinn_ckpt     = os.path.join(checkpoint_dir, "best_model.pt")
    baseline_ckpt = os.path.join(checkpoint_dir, "baseline", "baseline_best_model.pt")

    pinn_available     = os.path.exists(pinn_ckpt)
    baseline_available = os.path.exists(baseline_ckpt)

    pinn_preds = pinn_targets = base_preds = base_targets = None

    if pinn_available:
        print(f"Loading PhyConvLSTM checkpoint: {pinn_ckpt}")
        pinn_model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=horizon)
        pinn_model.load_state_dict(torch.load(pinn_ckpt, map_location="cpu"))
        pinn_preds, pinn_targets = collect_preds(pinn_model, val_loader, val_dataset)
        # Continuous daily forecast timeline (average overlapping 7-day windows).
        pinn_preds_np = reconstruct_continuous_forecast(pinn_preds).squeeze(1).numpy()
    else:
        print(f"⚠ PhyConvLSTM checkpoint not found at {pinn_ckpt}. Run `make train` first.")

    if baseline_available:
        print(f"Loading Vanilla ConvLSTM checkpoint: {baseline_ckpt}")
        base_model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=horizon)
        base_model.load_state_dict(torch.load(baseline_ckpt, map_location="cpu"))
        base_preds, base_targets = collect_preds(base_model, val_loader, val_dataset)
        base_preds_np = reconstruct_continuous_forecast(base_preds).squeeze(1).numpy()
    else:
        print(f"⚠ Baseline checkpoint not found at {baseline_ckpt}. Run `make baseline` first.")

    # --- Figures ---
    print("\nGenerating figures...")

    if pinn_available and baseline_available and "degree_heating_weeks" in ds.data_vars:
        plot_figure1(ds, pinn_preds_np, base_preds_np, clim_per_pixel, val_start, lookback, out_dir)
    else:
        print("  ⚠ Figure 1 skipped (need both models trained + DHW in Zarr)")

    if pinn_available and baseline_available:
        plot_figure2(pinn_preds, pinn_targets, base_preds, base_targets, horizon, out_dir)
    else:
        print("  ⚠ Figure 2 skipped (need both models trained)")

    pinn_log     = os.path.join(checkpoint_dir, "training_log.csv")
    baseline_log = os.path.join(checkpoint_dir, "baseline", "baseline_training_log.csv")
    if os.path.exists(pinn_log) or os.path.exists(baseline_log):
        plot_figure3(pinn_log, baseline_log, out_dir)
    else:
        print("  ⚠ Figure 3 skipped (no training logs found — run make train / make baseline)")

    if pinn_available and "degree_heating_weeks" in ds.data_vars:
        plot_figure4(ds, pinn_preds_np, clim_per_pixel, val_start, lookback, out_dir)
    else:
        print("  ⚠ Figure 4 skipped (need PhyConvLSTM trained + DHW in Zarr)")

    print(f"\nAll available figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
