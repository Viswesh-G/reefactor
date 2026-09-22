import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

# This Dataset reads processed data from a Zarr store and serves
# sliding-window samples to the PyTorch DataLoader.
# Each sample contains:
#   - x_sst:    past SST maps          (lookback, 1, H, W)
#   - y_sst:    future SST maps        (horizon,  1, H, W)
#   - u_vel:    future east-west current  (horizon,  1, H, W)
#   - v_vel:    future north-south current (horizon, 1, H, W)

class ReefDataset(Dataset):
    """
    Lazily loads data from a Zarr store produced by the ingestion pipeline.
    Uses a sliding window to create (input, target) pairs for the ConvLSTM.
    """
    def __init__(self, zarr_path, lookback=14, horizon=7):
        """
        Args:
            zarr_path: Path to the Zarr directory (e.g., data/processed/unified_gbr_canvas.zarr)
            lookback:  Number of past time steps fed to the encoder (default 14 days)
            horizon:   Number of future time steps to predict (default 7 days)
        """
        self.lookback = lookback
        self.horizon = horizon

        # Open the Zarr store lazily (data stays on disk until accessed)
        self.ds = xr.open_zarr(zarr_path, consolidated=True)

        # Total number of time steps available
        self.n_times = self.ds.dims["time"]

        # We need at least (lookback + horizon) contiguous days to form one sample
        self.n_samples = self.n_times - lookback - horizon + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough time steps ({self.n_times}) for lookback={lookback} + horizon={horizon}."
            )

        # Precompute normalization stats from the SST variable
        # (computed once on the full dataset, stored as scalars)
        sst = self.ds["sea_surface_temperature"]
        self.sst_mean = float(sst.mean().values)
        self.sst_std = float(sst.std().values)
        if self.sst_std == 0:
            self.sst_std = 1.0  # avoid division by zero on constant data

        # Check which velocity variables are available
        self.has_velocity = ("water_u" in self.ds.data_vars and "water_v" in self.ds.data_vars)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns one training sample as four tensors.
        The SST values are z-score normalized; velocities are left in m/s.
        """
        t_start = idx
        t_split = t_start + self.lookback
        t_end = t_split + self.horizon

        # Slice the SST for input (past) and target (future)
        sst_input = self.ds["sea_surface_temperature"].isel(time=slice(t_start, t_split)).values
        sst_target = self.ds["sea_surface_temperature"].isel(time=slice(t_split, t_end)).values

        # Normalize SST
        sst_input = (sst_input - self.sst_mean) / self.sst_std
        sst_target = (sst_target - self.sst_mean) / self.sst_std

        # Reshape from (T, H, W) to (T, 1, H, W) to add the channel dimension
        x_sst = torch.from_numpy(sst_input).unsqueeze(1).float()
        y_sst = torch.from_numpy(sst_target).unsqueeze(1).float()

        # Crop to 16x16 to prevent Out-Of-Memory errors during local CPU testing
        x_sst = x_sst[:, :, :16, :16]
        y_sst = y_sst[:, :, :16, :16]

        # Velocities for the forecast window (needed by PhyLoss)
        if self.has_velocity:
            u = self.ds["water_u"].isel(time=slice(t_split, t_end)).values
            v = self.ds["water_v"].isel(time=slice(t_split, t_end)).values
            u_vel = torch.from_numpy(u).unsqueeze(1).float()
            v_vel = torch.from_numpy(v).unsqueeze(1).float()
            u_vel = u_vel[:, :, :16, :16]
            v_vel = v_vel[:, :, :16, :16]
        else:
            # If velocity data is missing, use zeros (diffusion-only mode)
            u_vel = torch.zeros(self.horizon, 1, 16, 16)
            v_vel = torch.zeros(self.horizon, 1, 16, 16)

        return x_sst, y_sst, u_vel, v_vel
