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
    climatology_mean: float,
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
        climatology_mean: Scalar — long-term monthly mean SST for the region (°C).
        bleach_offset:    Degrees above climatology that defines the stress onset (default 1°C).
        window_days:      Rolling window in days (default 84 = 12 weeks).

    Returns:
        dhw: Shape (Time, H, W) — accumulated thermal stress in °C-weeks.
    """
    bleaching_threshold = climatology_mean + bleach_offset

    # Hotspot: how many degrees each pixel exceeds the bleaching threshold on each day
    hotspot = np.maximum(0.0, sst - bleaching_threshold)

    # Rolling sum of the past 12 weeks of hotspots, converted to weeks
    dhw = np.zeros_like(sst)
    for t in range(sst.shape[0]):
        start = max(0, t - window_days + 1)
        dhw[t] = hotspot[start : t + 1].sum(axis=0) / 7.0

    return dhw


def print_metrics_table(metrics_dict: dict):
    """Prints a clean, human-readable table of evaluation results."""
    print("\n" + "=" * 52)
    print(f"  {'Metric':<32} {'Value':>10}")
    print("=" * 52)
    for name, value in metrics_dict.items():
        print(f"  {name:<32} {value:>10.4f}")
    print("=" * 52 + "\n")
