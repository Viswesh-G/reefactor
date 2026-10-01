import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

# This Dataset reads from the Zarr store produced by the ingestion pipeline
# and serves sliding-window (input, target) pairs to the PyTorch DataLoader.
#
# Each sample contains four tensors:
#   x_sst  — past SST maps fed into the encoder      (lookback, 1, H, W)
#   y_sst  — future SST maps used as training targets (horizon,  1, H, W)
#   u_vel  — east-west ocean current during forecast  (horizon,  1, H, W)
#   v_vel  — north-south ocean current during forecast (horizon, 1, H, W)
#
# The u/v velocities are NOT predicted — they are used by the physics loss
# (PhyLoss) to verify that predicted SST changes are physically consistent
# with how the ocean currents would actually move heat.


class ReefDataset(Dataset):
    """
    Lazily reads from a Zarr store and produces sliding-window samples
    for the ConvLSTM encoder-decoder.

    Supports a chronological train/val split so that:
      - Training always uses the earlier portion of the time series
      - Validation uses the later, held-out portion
    This is correct for time series — never shuffle across the split boundary.
    """
    def __init__(
        self,
        zarr_path: str,
        lookback: int = 14,
        horizon: int = 7,
        split: str = "train",       # "train" or "val"
        val_fraction: float = 0.2,  # Fraction of time held out for validation
        patch_size: int = None,     # None = full spatial extent; int = crop to NxN patch
    ):
        """
        Args:
            zarr_path:    Path to the Zarr directory from the ingestion pipeline.
            lookback:     How many past days to feed into the encoder.
            horizon:      How many future days to predict.
            split:        "train" uses the first (1 - val_fraction) of time steps;
                          "val" uses the last val_fraction.
            val_fraction: Fraction of total time reserved for validation (default 0.2 = 20%).
            patch_size:   If set, each sample is cropped to a (patch_size × patch_size) region.
                          Useful for training on CPU where the full GBR grid is too large.
                          Set to None (default) to use the full spatial extent.
        """
        self.lookback   = lookback
        self.horizon    = horizon
        self.patch_size = patch_size

        # Open the Zarr store lazily — data stays on disk until we slice it in __getitem__
        self.ds = xr.open_zarr(zarr_path, consolidated=True)
        total_times = self.ds.dims["time"]

        # Chronological split: training on the first portion, validation on the last
        split_idx = int(total_times * (1.0 - val_fraction))
        if split == "train":
            self.time_offset = 0
            self.n_times = split_idx
        else:
            self.time_offset = split_idx
            self.n_times = total_times - split_idx

        # We need at least (lookback + horizon) contiguous days for one sample
        self.n_samples = self.n_times - lookback - horizon + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough time steps ({self.n_times}) in the '{split}' split for "
                f"lookback={lookback} + horizon={horizon}. "
                f"Try a shorter lookback/horizon or reduce val_fraction."
            )

        # Compute normalization stats using the training portion ONLY.
        # Using val/test data to compute stats would be data leakage.
        sst_train     = self.ds["sea_surface_temperature"].isel(time=slice(0, split_idx))
        self.sst_mean = float(sst_train.mean().values)
        self.sst_std  = float(sst_train.std().values)
        if self.sst_std == 0:
            self.sst_std = 1.0  # Safety guard against constant synthetic data

        # Check which velocity variables are available (they may not be if ingestion failed)
        self.has_velocity = (
            "water_u" in self.ds.data_vars and "water_v" in self.ds.data_vars
        )

        # Store spatial dimensions for use when building zero-velocity fallback tensors
        self.n_lat = self.ds.dims["lat"]
        self.n_lon = self.ds.dims["lon"]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns one (input, target, u_velocity, v_velocity) sample.
        SST is z-score normalized using training stats; velocities are left in m/s.
        """
        # Map the sample index to absolute positions in the full time axis
        t_start = self.time_offset + idx
        t_split = t_start + self.lookback
        t_end   = t_split + self.horizon

        # Load the SST slices from the Zarr store (triggers disk read)
        sst_input  = self.ds["sea_surface_temperature"].isel(time=slice(t_start, t_split)).values
        sst_target = self.ds["sea_surface_temperature"].isel(time=slice(t_split, t_end)).values

        # Z-score normalization: subtract mean, divide by std
        # This keeps training numerically stable regardless of absolute SST values
        sst_input  = (sst_input  - self.sst_mean) / self.sst_std
        sst_target = (sst_target - self.sst_mean) / self.sst_std

        # Reshape from (Time, H, W) to (Time, 1, H, W) — add a "channel" dimension
        # PyTorch convolutions expect (Batch, Channels, H, W) per time step
        x_sst = torch.from_numpy(sst_input).unsqueeze(1).float()
        y_sst = torch.from_numpy(sst_target).unsqueeze(1).float()

        if self.has_velocity:
            u = self.ds["water_u"].isel(time=slice(t_split, t_end)).values
            v = self.ds["water_v"].isel(time=slice(t_split, t_end)).values
            u_vel = torch.from_numpy(u).unsqueeze(1).float()
            v_vel = torch.from_numpy(v).unsqueeze(1).float()
        else:
            # If velocity data is missing, use zero tensors.
            # The physics loss will then only apply diffusion (no advection),
            # which is a physically valid degraded mode.
            H = x_sst.shape[-2]
            W = x_sst.shape[-1]
            u_vel = torch.zeros(self.horizon, 1, H, W)
            v_vel = torch.zeros(self.horizon, 1, H, W)

        # --- Optional spatial patch crop ---
        # For memory-constrained environments (e.g., CPU training on a laptop),
        # set patch_size in config.yaml to crop to a manageable spatial region.
        # Uses a deterministic but varied offset per sample for good spatial coverage.
        if self.patch_size is not None:
            h_max = max(0, self.n_lat - self.patch_size)
            w_max = max(0, self.n_lon - self.patch_size)
            # Simple but deterministic: different offset for each sample index
            h0 = (idx * 7)  % (h_max + 1) if h_max > 0 else 0
            w0 = (idx * 11) % (w_max + 1) if w_max > 0 else 0
            p  = self.patch_size
            x_sst = x_sst[:, :, h0:h0+p, w0:w0+p]
            y_sst = y_sst[:, :, h0:h0+p, w0:w0+p]
            u_vel = u_vel[:, :, h0:h0+p, w0:w0+p]
            v_vel = v_vel[:, :, h0:h0+p, w0:w0+p]

        return x_sst, y_sst, u_vel, v_vel
