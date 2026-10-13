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
import os


def load_crw_mmm(config: dict):
    """Load the true CRW MMM sidecar if present (shape (H, W) on the canvas grid).

    Produced by `run_derive_mmm.py` from MMM = CRW_SST - CRW_HOTSPOT. Using it
    makes our predicted DHW climatology-matched to NOAA's ground-truth DHW.
    Returns None if the sidecar does not exist (caller falls back to a
    train-split climatology).
    """
    path = config.get("storage", {}).get("crw_mmm_path")
    if path and os.path.exists(path):
        with np.load(path) as d:
            return d["mmm"].astype(np.float64)
    return None


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


def compute_mmm_climatology(sst_da: xr.DataArray) -> np.ndarray:
    """NOAA CRW MMM: per-pixel MAXIMUM of 12 monthly means. Shape (H, W).

    Bleaching threshold = MMM + 1C. HotSpot = SST - MMM; DHW accumulates
    HotSpot/7 only when HotSpot >= 1 (i.e. SST >= MMM+1).
    """
    clim12 = compute_monthly_climatology(sst_da)  # (12, H, W)
    return clim12.max(axis=0)  # (H, W)


def reconstruct_continuous(all_windows: torch.Tensor) -> torch.Tensor:
    """Average overlapping forecast windows into a continuous (T, C, H, W) series.

    Input : (N, H_out, C, H, W) — N sliding windows, each predicting H_out days.
    Output: (T, C, H, W) with T = N + H_out - 1. Each absolute day is the mean of
    all windows that cover it (a daily-mean forecast timeline). This is the only
    valid input for an 84-day DHW accumulation or a residual time series.
    """
    n, h_out = all_windows.shape[0], all_windows.shape[1]
    t_total = n + h_out - 1
    acc = torch.zeros(t_total, *all_windows.shape[2:])   # (T, C, H, W)
    cnt = torch.zeros(t_total, 1, 1, 1)
    for i in range(n):
        acc[i:i + h_out] += all_windows[i]
        cnt[i:i + h_out] += 1
    return acc / cnt.clamp_min(1)


def reconstruct_continuous_forecast(all_preds: torch.Tensor) -> torch.Tensor:
    """Collapse overlapping (N, H, 1, H, W) windows to a continuous (T, 1, H, W).

    Val windows use stride 1, so each absolute day has up to H predictions.
    We average all predictions covering each day (daily-mean forecast timeline).
    """
    return reconstruct_continuous(all_preds)


def physics_residual_series(
    preds: np.ndarray,
    u_vel: np.ndarray,
    v_vel: np.ndarray,
    dx: float = 5500.0,
    dy: float = 5500.0,
    dt: float = 86400.0,
    K: float = 500.0,
) -> np.ndarray:
    """Daily spatial-mean |advection-diffusion residual| over a forecast timeline.

    PIVOT differentiator: the residual measures how much the network's own forecast
    violates the heat-transport physics. If it rises before accumulated heat stress
    (DHW) peaks, it is an interpretable early-warning precursor.

    Args:
        preds: (T, H, W) continuous SST (°C).
        u_vel, v_vel: (T, H, W) velocity (m/s) on the SAME timeline.
        dx, dy: grid spacing (m). dt: time step (s). K: diffusivity (m^2/s).

    Returns:
        (T-1,) array; index t = mean |residual| between forecast days t and t+1.
    """
    res = np.empty(preds.shape[0] - 1, dtype=np.float64)
    for t in range(preds.shape[0] - 1):
        Tp = preds[t]
        dTdt = (preds[t + 1] - Tp) / dt
        dTdx = np.gradient(Tp, dx, axis=1)               # along W (longitude)
        dTdy = np.gradient(Tp, dy, axis=0)               # along H (latitude)
        d2dx2 = np.gradient(dTdx, dx, axis=1)
        d2dy2 = np.gradient(dTdy, dy, axis=0)
        residual = dTdt + u_vel[t] * dTdx + v_vel[t] * dTdy - K * (d2dx2 + d2dy2)
        res[t] = np.mean(np.abs(residual))
    return res


def persistence_baseline(all_inputs: torch.Tensor, horizon: int) -> torch.Tensor:
    """Repeat last observed frame over horizon: (N, H, 1, H, W)."""
    last = all_inputs[:, -1:, :, :, :]  # (N, 1, 1, H, W)
    return last.repeat(1, horizon, 1, 1, 1)


def climatology_baseline(clim12: np.ndarray, target_months: np.ndarray) -> torch.Tensor:
    """Monthly-climatology baseline forecast.

    The honest "no-skill" reference for SST forecasting is the long-term
    monthly mean for the *calendar month being predicted* (not a single annual
    value). We build it from the training-split climatology only.

    Args:
        clim12:        (12, H, W) monthly-mean SST per pixel (Jan=index 0).
        target_months: (N, horizon) int array, 0-based calendar month per target day.

    Returns:
        (N, horizon, 1, H, W) tensor of climatological SST in °C.
    """
    n, horizon = target_months.shape
    h, w = clim12.shape[-2], clim12.shape[-1]
    out = np.empty((n, horizon, 1, h, w), dtype=np.float32)
    for i in range(n):
        for d in range(horizon):
            out[i, d, 0] = clim12[target_months[i, d] % 12]
    return torch.from_numpy(out)


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

    True NOAA CRW definition (use compute_mmm_climatology for climatology_mean):
      1. MMM = per-pixel MAXIMUM of 12 monthly-mean SST (warmest-month baseline)
      2. Coral Bleaching HotSpot = SST - MMM  (can be negative; range 0..+5 shown)
      3. Bleaching threshold = MMM + bleach_offset (1°C). Each day whose HotSpot
         reaches/exceeds the threshold contributes its FULL HotSpot value; days
         below the threshold contribute nothing. DHW = sum(HotSpot)/7 over 84 days.
         e.g. HotSpots 1.0, 2.0, 0.8, 1.2 (2x/week) -> (1.0+2.0+1.2)*0.5 = 2.1.

    Callers: pass MMM (H, W) from TRAINING split only. Do NOT pass annual means.
    Input sst must be a CONTINUOUS daily timeline (see reconstruct_continuous_forecast),
    never stitched overlapping windows.

    Args:
        sst:              Shape (Time, H, W) — CONTINUOUS daily SST maps in °C.
        climatology_mean: MMM (H, W) per-pixel, or scalar fallback.
        bleach_offset:    HotSpot gate defining stress onset (default 1°C NOAA).
        window_days:      Rolling window in days (default 84 = 12 weeks).

    Returns:
        dhw: Shape (Time, H, W) — accumulated thermal stress in °C-weeks.
    """
    # Support both scalar and (H, W) spatial climatology maps.
    # Broadcasting handles both cases cleanly.
    mmm = np.asarray(climatology_mean, dtype=np.float64)

    # Coral Bleaching HotSpot = SST - MMM. Accumulate the FULL HotSpot value, but
    # only on days where it reaches the bleaching threshold (>= bleach_offset).
    hotspot = sst - mmm
    hotspot = np.where(hotspot >= bleach_offset, hotspot, 0.0)

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

