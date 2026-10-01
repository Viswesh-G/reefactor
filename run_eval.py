"""
Evaluation script for the PINN coral bleaching model.

Run this after training has completed:
    python run_eval.py
  or
    make eval

What this does:
  1. Loads the best saved checkpoint from data/checkpoints/best_model.pt
  2. Runs the model on the held-out validation split of the Zarr dataset
  3. Denormalizes predictions back to real degrees Celsius
  4. Computes three oceanographic skill metrics:
       - RMSE  (Root Mean Square Error, in °C)
       - ACC   (Anomaly Correlation Coefficient)
       - NSE   (Nash-Sutcliffe Efficiency)
  5. Computes Degree Heating Weeks (DHW) from predicted SST
  6. Compares predicted DHW to the NOAA ground-truth DHW (if available in Zarr)
  7. Prints a metrics summary table
  8. Saves a side-by-side comparison plot to data/checkpoints/eval_comparison.png
"""
import os
import torch
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader

from src.config_validator import ConfigValidator
from src.models import SST_ConvLSTM
from src.dataset import ReefDataset
from src.metrics import (
    rmse,
    anomaly_correlation,
    nash_sutcliffe,
    compute_dhw_from_sst,
    print_metrics_table,
)


def main():
    # --- Config ---
    validator = ConfigValidator("config.yaml")
    config    = validator.validate()

    zarr_path      = config["storage"]["processed_zarr_path"]
    checkpoint_dir = config["storage"]["checkpoint_dir"]
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.exists(best_model_path):
        print(f"\nNo trained model found at '{best_model_path}'.")
        print("Run `make train` first to produce a trained model.\n")
        return

    ds_cfg       = config.get("dataset", {})
    lookback     = ds_cfg.get("lookback", 14)
    horizon      = ds_cfg.get("horizon", 7)
    val_fraction = ds_cfg.get("val_fraction", 0.2)
    patch_size   = ds_cfg.get("patch_size", None)

    # --- Load Validation Split ---
    print(f"\nLoading validation split from: {zarr_path}")
    try:
        val_dataset = ReefDataset(
            zarr_path, lookback=lookback, horizon=horizon,
            split="val", val_fraction=val_fraction, patch_size=patch_size,
        )
    except Exception as e:
        print(f"Could not load Zarr data: {e}")
        print("Run `make ingest` first.\n")
        return

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Validation samples: {len(val_dataset)}")

    # --- Load Model ---
    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=horizon)
    model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    model.eval()
    print(f"  Loaded model: {best_model_path}")

    # --- Collect Predictions ---
    all_preds   = []
    all_targets = []

    print("\nRunning inference on validation set...")
    with torch.no_grad():
        for x_sst, y_sst, _, _ in val_loader:
            preds = model(x_sst)
            all_preds.append(preds)
            all_targets.append(y_sst)

    all_preds   = torch.cat(all_preds,   dim=0)   # (N, horizon, 1, H, W)
    all_targets = torch.cat(all_targets, dim=0)   # (N, horizon, 1, H, W)

    # Denormalize from z-score back to real °C using the training split's stats
    preds_celsius   = all_preds   * val_dataset.sst_std + val_dataset.sst_mean
    targets_celsius = all_targets * val_dataset.sst_std + val_dataset.sst_mean

    # --- Standard Metrics ---
    metrics = {
        "RMSE (°C)":                  rmse(preds_celsius, targets_celsius),
        "Anomaly Correlation (ACC)":   anomaly_correlation(preds_celsius, targets_celsius),
        "Nash-Sutcliffe Efficiency":   nash_sutcliffe(preds_celsius, targets_celsius),
    }

    # --- DHW Comparison (if ground-truth DHW was ingested) ---
    ds = xr.open_zarr(zarr_path, consolidated=True)
    if "degree_heating_weeks" in ds.data_vars:
        print("  Computing Degree Heating Weeks from predictions...")

        # Use the mean SST of the full dataset as the climatology reference
        clim_mean = float(ds["sea_surface_temperature"].mean().values)

        # Flatten predictions from (N, horizon, 1, H, W) to (T, H, W)
        n, h_out, _, h, w = all_preds.shape
        preds_np = preds_celsius.squeeze(2).numpy().reshape(n * h_out, h, w)

        dhw_pred = compute_dhw_from_sst(preds_np, climatology_mean=clim_mean)

        # Load the matching slice of true NOAA DHW from the Zarr store
        n_times   = ds.dims["time"]
        val_start = int(n_times * (1.0 - val_fraction))
        true_dhw  = ds["degree_heating_weeks"].isel(time=slice(val_start, None)).values

        # Align lengths before comparing (preds may be slightly shorter due to window edge)
        min_len   = min(dhw_pred.shape[0], true_dhw.shape[0])
        dhw_pred_t = torch.from_numpy(dhw_pred[:min_len]).float()
        dhw_true_t = torch.from_numpy(true_dhw[:min_len]).float()

        metrics["DHW RMSE (°C-weeks)"]   = rmse(dhw_pred_t, dhw_true_t)
        metrics["DHW Correlation (ACC)"]  = anomaly_correlation(dhw_pred_t, dhw_true_t)
    else:
        print("  Note: 'degree_heating_weeks' not in Zarr — DHW metrics skipped.")

    # --- Print Results ---
    print_metrics_table(metrics)

    # --- Comparison Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")   # Non-interactive backend: saves to file without a display
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle("PINN Validation — Day 1 Forecast (First Sample)", fontsize=13)

        # Show the first sample, first predicted day
        pred_map = preds_celsius[0, 0, 0].numpy()
        true_map = targets_celsius[0, 0, 0].numpy()
        diff_map = pred_map - true_map

        vmin = min(pred_map.min(), true_map.min())
        vmax = max(pred_map.max(), true_map.max())
        v_diff = max(abs(diff_map.min()), abs(diff_map.max()))

        im0 = axes[0].imshow(true_map,  origin="lower", vmin=vmin, vmax=vmax, cmap="RdYlBu_r")
        axes[0].set_title("True SST (°C)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(pred_map,  origin="lower", vmin=vmin, vmax=vmax, cmap="RdYlBu_r")
        axes[1].set_title("Predicted SST (°C)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(diff_map,  origin="lower", vmin=-v_diff, vmax=v_diff, cmap="coolwarm")
        axes[2].set_title("Difference (Pred − True, °C)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        plot_path = os.path.join(checkpoint_dir, "eval_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to: {plot_path}")

    except ImportError:
        print("(matplotlib not found — no plot saved. Run: pip install matplotlib)")


if __name__ == "__main__":
    main()
