"""
run_noaa_dhw_baseline.py — NOAA CRW Operational DHW Baseline

This is Baseline #1 in the paper: the current operational standard that the
PINN is benchmarked against. It does NOT use any machine learning.

The NOAA Coral Reef Watch (CRW) DHW algorithm:
    1. For each pixel, MMM = warmest of the 12 monthly-mean SSTs; threshold = MMM + 1°C
    2. Coral Bleaching HotSpot = SST − MMM (negative allowed)
    3. Accumulate the FULL HotSpot only on days it reaches the threshold (HotSpot >= 1),
       over the trailing 12 weeks (84 days), then divide by 7 → DHW in °C-weeks
    (e.g. HotSpots 1.0, 2.0, 0.8, 1.2 twice-weekly → (1.0+2.0+1.2)*0.5 = 2.1 °C-weeks)

This script applies that algorithm directly to the raw satellite SST already
stored in the Zarr store, then computes the same metrics (RMSE, ACC) against the
NOAA ground-truth DHW product that was also ingested. This gives us the performance
ceiling of the operational algorithm on the same held-out validation period.

Usage:
    python run_noaa_dhw_baseline.py
  or
    make noaa_baseline

Output:
    • Metrics table printed to stdout
    • data/checkpoints/noaa_dhw_baseline_comparison.png — spatial DHW comparison map
    • data/checkpoints/noaa_dhw_baseline_metrics.csv   — metrics in CSV form
"""
import os
import csv
import numpy as np
import torch
import xarray as xr

from src.config_validator import ConfigValidator
from src.metrics import (
    rmse,
    anomaly_correlation,
    nash_sutcliffe,
    compute_dhw_from_sst,
    compute_mmm_climatology,
    load_crw_mmm,
    print_metrics_table,
)


def main():
    print("=== NOAA CRW Operational DHW Baseline ===\n")

    validator = ConfigValidator("config.yaml")
    config    = validator.validate()

    zarr_path      = config["storage"]["processed_zarr_path"]
    checkpoint_dir = config["storage"]["checkpoint_dir"]
    val_fraction   = config.get("dataset", {}).get("val_fraction", 0.2)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Load the full Zarr store ---
    print(f"Loading Zarr store: {zarr_path}")
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except Exception as e:
        print(f"Could not open Zarr store: {e}")
        print("Run `make ingest` first.\n")
        return

    if "sea_surface_temperature" not in ds.data_vars:
        print("ERROR: 'sea_surface_temperature' not found in Zarr store.")
        return

    n_times   = ds.dims["time"]
    val_start = int(n_times * (1.0 - val_fraction))
    print(f"  Total time steps: {n_times}")
    print(f"  Val split starts at index: {val_start} ({val_fraction*100:.0f}% held out)\n")

    # --- Build per-pixel monthly climatology (prefer TRUE CRW MMM sidecar) ---
    # Using the same MMM NOAA used makes the DHW comparison apples-to-apples.
    sst_train_da = ds["sea_surface_temperature"].isel(time=slice(0, val_start))
    clim_per_pixel = load_crw_mmm(config)
    if clim_per_pixel is not None:
        print("  Using TRUE CRW MMM sidecar (climatology-matched).")
    else:
        print("Building per-pixel MMM climatology (max of monthly means) from train split...")
        clim_per_pixel = compute_mmm_climatology(sst_train_da)  # (H, W) NOAA MMM
    print(f"  Climatology shape: {clim_per_pixel.shape}")
    print(f"  Mean threshold across grid: {clim_per_pixel.mean():.2f} °C (+1°C bleach offset)\n")

    # --- Apply NOAA DHW algorithm to the raw SST in the validation period ---
    # We load the full SST array (train + val) so the 84-day rolling window at the
    # start of the val period has proper warm-up from the preceding training days.
    print("Loading full SST time series for DHW computation (with 84-day warm-up)...")
    sst_full = ds["sea_surface_temperature"].values   # (T, H, W)

    print("Computing NOAA DHW algorithm on full time series (this may take a minute)...")
    dhw_computed = compute_dhw_from_sst(sst_full, climatology_mean=clim_per_pixel)
    # Restrict to the validation period for metrics
    dhw_pred_val = dhw_computed[val_start:]           # (val_T, H, W)

    # --- Load ground-truth NOAA DHW from Zarr ---
    if "degree_heating_weeks" not in ds.data_vars:
        print("WARNING: 'degree_heating_weeks' not found in Zarr store.")
        print("         DHW comparison metrics cannot be computed.")
        print("         Ensure the SST ERDDAP source includes CRW_DHW in config.yaml.\n")
        return

    print("Loading ground-truth NOAA DHW from Zarr...")
    dhw_true_val = ds["degree_heating_weeks"].isel(time=slice(val_start, None)).values

    # --- Align lengths ---
    min_len    = min(dhw_pred_val.shape[0], dhw_true_val.shape[0])
    dhw_pred_t = torch.from_numpy(dhw_pred_val[:min_len]).float()
    dhw_true_t = torch.from_numpy(dhw_true_val[:min_len]).float()

    # --- Metrics ---
    metrics = {
        "DHW RMSE (°C-weeks)":   rmse(dhw_pred_t, dhw_true_t),
        "DHW Correlation (ACC)": anomaly_correlation(dhw_pred_t, dhw_true_t),
        "DHW NSE":               nash_sutcliffe(dhw_pred_t, dhw_true_t),
    }
    print("\nNOAA CRW Operational DHW Baseline — Validation Metrics:")
    print_metrics_table(metrics)

    # --- Save metrics to CSV ---
    metrics_csv = os.path.join(checkpoint_dir, "noaa_dhw_baseline_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for name, val in metrics.items():
            writer.writerow({"metric": name, "value": f"{val:.6f}"})
    print(f"Metrics saved to: {metrics_csv}")

    # --- Comparison Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            "NOAA CRW DHW Baseline — Validation Period (Last Time Step)",
            fontsize=13,
        )

        # Last time step of the validation period for a snapshot comparison
        pred_map = dhw_pred_val[-1]
        true_map = dhw_true_val[-1]
        diff_map = pred_map - true_map

        vmax = max(pred_map.max(), true_map.max(), 1.0)
        v_diff = max(abs(diff_map.min()), abs(diff_map.max()), 0.5)

        lats = ds.lat.values
        lons = ds.lon.values

        im0 = axes[0].pcolormesh(lons, lats, true_map,  vmin=0, vmax=vmax, cmap="YlOrRd")
        axes[0].set_title("NOAA Ground-Truth DHW (°C-weeks)")
        axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].pcolormesh(lons, lats, pred_map,  vmin=0, vmax=vmax, cmap="YlOrRd")
        axes[1].set_title("NOAA Algorithm DHW (°C-weeks)\n[Applied to raw Zarr SST]")
        axes[1].set_xlabel("Longitude")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].pcolormesh(lons, lats, diff_map, vmin=-v_diff, vmax=v_diff, cmap="coolwarm")
        axes[2].set_title("Difference (Computed − Ground Truth, °C-weeks)")
        axes[2].set_xlabel("Longitude")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        plot_path = os.path.join(checkpoint_dir, "noaa_dhw_baseline_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to: {plot_path}\n")

    except ImportError:
        print("(matplotlib not found — no plot saved. Run: pip install matplotlib)")


if __name__ == "__main__":
    main()
