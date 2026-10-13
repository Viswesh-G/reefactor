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
from src.reproducibility import set_seed, get_seed, get_device
from src.metrics import (
    rmse,
    anomaly_correlation,
    nash_sutcliffe,
    compute_dhw_from_sst,
    compute_monthly_climatology,
    compute_mmm_climatology,
    load_crw_mmm,
    reconstruct_continuous_forecast,
    persistence_baseline,
    climatology_baseline,
    bleaching_alert_agreement,
    compute_lead_rmse,
    print_metrics_table,
)


def main():
    # --- Config ---
    validator = ConfigValidator("config.yaml")
    config    = validator.validate()
    set_seed(get_seed(config))
    device    = get_device(config)

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
    # IMPORTANT: evaluate on the FULL spatial field (patch_size=None), never on
    # random crops. Training uses 64x64 patches for CPU memory, but each validation
    # day must cover the whole GBR grid so the forecast timeline is spatially
    # consistent (required for a valid 84-day DHW accumulation).
    print(f"\nLoading validation split from: {zarr_path}")
    try:
        val_dataset = ReefDataset(
            zarr_path, lookback=lookback, horizon=horizon,
            split="val", val_fraction=val_fraction, patch_size=None,
        )
    except Exception as e:
        print(f"Could not load Zarr data: {e}")
        print("Run `make ingest` first.\n")
        return

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Validation samples: {len(val_dataset)}")

    # --- Load Model ---
    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=horizon)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Loaded model: {best_model_path} (device={device})")

    # --- Collect Predictions (+ inputs for persistence baseline) ---
    all_preds   = []
    all_targets = []
    all_inputs  = []

    print("\nRunning inference on validation set...")
    with torch.no_grad():
        for x_sst, y_sst, _, _ in val_loader:
            x_sst = x_sst.to(device)
            preds = model(x_sst)
            all_preds.append(preds.cpu())
            all_targets.append(y_sst)
            all_inputs.append(x_sst.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)   # (N, horizon, 1, H, W)
    all_targets = torch.cat(all_targets, dim=0)   # (N, horizon, 1, H, W)
    all_inputs  = torch.cat(all_inputs,  dim=0)   # (N, lookback, 1, H, W)

    # Denormalize from z-score back to real °C using the training split's stats
    preds_celsius   = all_preds   * val_dataset.sst_std + val_dataset.sst_mean
    targets_celsius = all_targets * val_dataset.sst_std + val_dataset.sst_mean

    # --- Standard Metrics + persistence & climatology baselines (reviewers demand) ---
    persist_norm = persistence_baseline(all_inputs, horizon)
    persist_c = persist_norm * val_dataset.sst_std + val_dataset.sst_mean
    metrics = {
        "RMSE (°C)":                  rmse(preds_celsius, targets_celsius),
        "Anomaly Correlation (ACC)":   anomaly_correlation(preds_celsius, targets_celsius),
        "Nash-Sutcliffe Efficiency":   nash_sutcliffe(preds_celsius, targets_celsius),
        "Persistence RMSE (°C)":       rmse(persist_c, targets_celsius),
        "Persistence ACC":              anomaly_correlation(persist_c, targets_celsius),
    }

    # --- DHW Comparison (if ground-truth DHW was ingested) ---
    # True NOAA: MMM = per-pixel MAX of monthly means (training split only),
    # threshold MMM+1, 84d window. Forecast timeline is the continuous daily-mean
    # reconstruction (never stitched overlapping windows).
    ds = xr.open_zarr(zarr_path, consolidated=True)
    if "degree_heating_weeks" in ds.data_vars:
        print("  Computing Degree Heating Weeks from predictions...")

        n_times   = ds.sizes["time"]
        val_start = int(n_times * (1.0 - val_fraction))
        sst_train_da = ds["sea_surface_temperature"].isel(time=slice(0, val_start))
        # Prefer the TRUE CRW MMM sidecar (climatology-matched to NOAA DHW).
        mmm = load_crw_mmm(config)
        if mmm is not None:
            print("  Using TRUE CRW MMM sidecar (climatology-matched to NOAA DHW).")
        else:
            print("  MMM sidecar not found — falling back to train-split MMM "
                  "(run run_derive_mmm.py for a matched comparison).")
            mmm = compute_mmm_climatology(sst_train_da)  # (H, W)

        # Skill of the honest "no-skill" reference: predict the training-split
        # climatology of the *target calendar month* for every validation day.
        clim12 = compute_monthly_climatology(sst_train_da)  # (12, H, W) in °C
        times  = ds.time.values
        months = np.empty((len(val_dataset), horizon), dtype=int)
        for i in range(len(val_dataset)):
            t_split = val_start + i + lookback
            for d in range(horizon):
                months[i, d] = int(times[t_split + d].astype("datetime64[M]").astype(int) % 12)
        clim_fc = climatology_baseline(clim12.astype(np.float32), months)  # °C
        metrics["Climatology RMSE (°C)"] = rmse(clim_fc, targets_celsius)

        # Continuous forecast timeline for valid 84d DHW accumulation
        cont_pred = reconstruct_continuous_forecast(preds_celsius)  # (T,1,H,W)
        preds_np = cont_pred.squeeze(1).numpy()  # (T, H, W)

        dhw_pred = compute_dhw_from_sst(preds_np, climatology_mean=mmm)

        # Load the matching slice of true NOAA DHW from the Zarr store.
        # Cont timeline starts at val_start+lookback (first forecast day), not val_start.
        dhw_start = val_start + lookback
        true_dhw  = ds["degree_heating_weeks"].isel(time=slice(dhw_start, None)).values

        # Align lengths before comparing (tail window edge)
        min_len   = min(dhw_pred.shape[0], true_dhw.shape[0])
        dhw_pred = dhw_pred[:min_len]
        # Warm-up guard: first 83 DHW values lack full 84d history — compare from day 83
        # only when timeline is long enough; else compare all (short 2017-2018 test).
        w = 83 if min_len > 100 else 0
        dhw_pred_t = torch.from_numpy(dhw_pred[w:]).float()
        dhw_true_t = torch.from_numpy(true_dhw[w:min_len]).float()

        metrics["DHW RMSE (°C-weeks)"]   = rmse(dhw_pred_t, dhw_true_t)
        metrics["DHW Correlation (ACC)"]  = anomaly_correlation(dhw_pred_t, dhw_true_t)

        # --- Bleaching Alert Level Agreement (DHW classification metrics) ---
        dhw_pred_np = dhw_pred[w:]
        dhw_true_np = true_dhw[w:min_len]

        for thr in [4.0, 8.0]:
            alert = bleaching_alert_agreement(dhw_pred_np, dhw_true_np, threshold=thr)
            label = f"DHW>{int(thr)}"
            metrics[f"{label} Precision"] = alert["precision"]
            metrics[f"{label} Recall"]    = alert["recall"]
            metrics[f"{label} F1"]        = alert["f1"]
            metrics[f"{label} IoU"]       = alert["iou"]
    else:
        print("  Note: 'degree_heating_weeks' not in Zarr — DHW metrics skipped.")

    # --- Lead-Time RMSE (per forecast day) — needed for Figure 2 ---
    lead_rmses = compute_lead_rmse(preds_celsius, targets_celsius, horizon)
    lead_csv_path = os.path.join(checkpoint_dir, "eval_lead_rmse.csv")
    import csv as _csv
    with open(lead_csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=["lead_day", "rmse_celsius"])
        writer.writeheader()
        for day, r in enumerate(lead_rmses, start=1):
            writer.writerow({"lead_day": day, "rmse_celsius": f"{r:.6f}"})
    print(f"  Lead-time RMSE saved to: {lead_csv_path}")
    for day, r in enumerate(lead_rmses, start=1):
        metrics[f"RMSE Day {day} (°C)"] = r

    # --- Print Results ---
    print_metrics_table(metrics)

    # --- Save all metrics to CSV ---
    metrics_csv = os.path.join(checkpoint_dir, "eval_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for name, val in metrics.items():
            writer.writerow({"metric": name, "value": f"{val:.6f}" if isinstance(val, float) else str(val)})
    print(f"  All metrics saved to: {metrics_csv}")

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
