"""
Evaluation metrics for the PINN coral bleaching model.

Three standard oceanographic skill metrics:
  - RMSE  : Root Mean Square Error — raw prediction error in °C
  - ACC   : Anomaly Correlation Coefficient — how well the spatial/temporal pattern is captured (0 to 1)
  - NSE   : Nash-Sutcliffe Efficiency — skill vs. a naive "always predict the mean" baseline (-∞ to 1)

Plus the key biological stress metric used in the paper:
  - DHW   : Degree Heating Weeks — accumulated thermal stress computed from predicted SST,
            following the NOAA CRW definition (12-week rolling sum of daily hotspots).
"""

import numpy as np
import torch
import xarray as xr


def compute_monthly_climatology(sst_da: xr.DataArray) -> np.ndarray:
    """
    Compute per-pixel monthly mean SST climatology from an xarray DataArray.

    This follows the NOAA CRW definition for the bleaching threshold baseline:
      - For each calendar month (Jan–Dec), compute the mean SST at every grid point
      - Return a (12, H, W) array where index 0 = January, 11 = December

    Unlike a single global mean, this captures the fact that some reef sites are
    chronically warmer or cooler than others, so each pixel gets its own threshold.
    This is essential for computing DHW correctly relative to local baselines.

    Args:
        sst_da: xarray DataArray with dims (time, lat, lon) and a cftime/datetime time axis.

    Returns:
        clim: np.ndarray of shape (12, H, W) — monthly mean SST per pixel, in °C.
    """
    # Group by calendar month and compute the mean across all years for each pixel
    monthly = sst_da.groupby("time.month").mean(dim="time")
    # Ensure months are ordered 1–12 and return as numpy array
    clim = monthly.sortby("month").values   # shape: (12, H, W)
    return clim


def lookup_climatology(clim12: np.ndarray, month_index: int) -> np.ndarray:
    """
    Retrieve the climatology map for a given calendar month (0-indexed: Jan=0).

    Args:
        clim12:      Output of compute_monthly_climatology(), shape (12, H, W).
        month_index: 0-based month index (0=January, 11=December).

    Returns:
        clim_map: shape (H, W) — per-pixel climatological SST for that month.
    """
    return clim12[month_index % 12]



def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Root Mean Square Error between prediction and ground truth (in the unit of the data)."""
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def anomaly_correlation(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Anomaly Correlation Coefficient (ACC).
    Removes the mean from both arrays, then measures how well the remaining
    spatial/temporal patterns line up.
      - 1.0 → perfect pattern agreement
      - 0.0 → no correlation
      - negative → anti-correlated (worse than random)
    """
    pred_anom = preds - preds.mean()
    true_anom = targets - targets.mean()
    numerator = (pred_anom * true_anom).sum()
    denominator = torch.sqrt((pred_anom ** 2).sum() * (true_anom ** 2).sum()) + 1e-8
    return (numerator / denominator).item()


def nash_sutcliffe(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Nash-Sutcliffe Efficiency (NSE).
    Compares model error against the simplest possible baseline: always predicting the mean.
      - 1.0 → perfect model
      - 0.0 → model is exactly as good as a constant mean prediction
      - negative → model is actually worse than just using the mean
    This is a standard metric in hydrology and oceanography.
    """
    mean_target = targets.mean()
    ss_res = torch.sum((preds - targets) ** 2)        # our model's squared error
    ss_tot = torch.sum((targets - mean_target) ** 2)  # naive baseline's squared error
    return (1.0 - ss_res / (ss_tot + 1e-8)).item()


def compute_dhw_from_sst(
    sst: np.ndarray,
    climatology_mean,
    bleach_offset: float = 1.0,
    window_days: int = 84,
) -> np.ndarray:
    """
    Compute Degree Heating Weeks (DHW) from a time series of daily SST maps.

    DHW is the primary biological stress metric NOAA uses to predict coral bleaching:
      - DHW > 4 °C-weeks  → bleaching likely to begin
      - DHW > 8 °C-weeks  → severe bleaching and coral mortality likely

    NOAA's definition (which we follow here):
      1. Bleaching threshold = long-term monthly mean SST climatology + 1°C
      2. Daily hotspot       = max(0,  SST − bleaching_threshold)
      3. DHW                 = sum of daily hotspots over the last 84 days (12 weeks) ÷ 7

    Args:
        sst:              Shape (Time, H, W) — daily SST maps in °C.
        climatology_mean: Either:
                            • A scalar  — applied uniformly to every pixel (fast, approximate).
                            • A (H, W) np.ndarray — per-pixel threshold matching NOAA CRW.
                          Use compute_monthly_climatology() to derive the per-pixel version.
        bleach_offset:    Degrees above climatology that defines the stress onset (default 1°C).
        window_days:      Rolling window in days (default 84 = 12 weeks).

    Returns:
        dhw: Shape (Time, H, W) — accumulated thermal stress in °C-weeks.
    """
    # Support both scalar and (H, W) spatial climatology maps.
    # np.atleast_1d / broadcasting handles both cases cleanly.
    bleaching_threshold = np.asarray(climatology_mean) + bleach_offset

    # Hotspot: how many degrees each pixel exceeds the bleaching threshold on each day
    # When bleaching_threshold is (H, W), broadcasting handles the (T, H, W) sst array.
    hotspot = np.maximum(0.0, sst - bleaching_threshold)

    # Rolling sum of the past 12 weeks of hotspots, converted to weeks
    dhw = np.zeros_like(sst)
    for t in range(sst.shape[0]):
        start = max(0, t - window_days + 1)
        dhw[t] = hotspot[start : t + 1].sum(axis=0) / 7.0

    return dhw


def bleaching_alert_agreement(
    dhw_pred: np.ndarray,
    dhw_true: np.ndarray,
    threshold: float = 4.0,
) -> dict:
    """
    Compute pixel-level classification metrics for bleaching alert agreement.

    Treats DHW prediction as a binary classifier:
        Positive class = DHW exceeds `threshold` (bleaching alert active)
        Negative class = DHW below `threshold` (no alert)

    NOAA uses two operational thresholds:
        DHW > 4  °C-weeks → Bleaching Alert Level 1 (bleaching likely)
        DHW > 8  °C-weeks → Bleaching Alert Level 2 (mortality likely)

    Args:
        dhw_pred:  Predicted DHW array,  shape (T, H, W) or (H, W).
        dhw_true:  Ground-truth DHW array, same shape.
        threshold: Alert threshold in °C-weeks (4.0 or 8.0).

    Returns:
        dict with keys: precision, recall, f1, iou, threshold
    """
    pred_alert = (dhw_pred >= threshold).astype(bool)
    true_alert = (dhw_true >= threshold).astype(bool)

    tp = int((pred_alert &  true_alert).sum())
    fp = int((pred_alert & ~true_alert).sum())
    fn = int((~pred_alert & true_alert).sum())
    tn = int((~pred_alert & ~true_alert).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)   # Intersection over Union

    return {
        "threshold":  threshold,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "iou":        iou,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_lead_rmse(
    preds: torch.Tensor,
    targets: torch.Tensor,
    horizon: int,
) -> list:
    """
    Compute RMSE at each forecast lead day independently.

    Args:
        preds:   Shape (N, horizon, 1, H, W) — denormalized predictions.
        targets: Shape (N, horizon, 1, H, W) — denormalized ground truth.
        horizon: Number of forecast days.

    Returns:
        List of floats, length=horizon. Index 0 = Day 1 RMSE, etc.
    """
    return [
        rmse(preds[:, d, :, :, :], targets[:, d, :, :, :])
        for d in range(horizon)
    ]


def print_metrics_table(metrics_dict: dict):
    """Prints a clean, human-readable table of evaluation results."""
    print("\n" + "=" * 52)
    print(f"  {'Metric':<32} {'Value':>10}")
    print("=" * 52)
    for name, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"  {name:<32} {value:>10.4f}")
        else:
            print(f"  {name:<32} {str(value):>10}")
    print("=" * 52 + "\n")

