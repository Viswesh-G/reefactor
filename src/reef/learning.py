"""Leakage-aware multivariate windows and an autoregressive SST forecaster."""
import numpy as np
import xarray as xr
import torch
from torch import nn
from torch.utils.data import Dataset

from src.models import ConvLSTMCell
from .features import derive_features
from .pipeline import open_cube


class FeatureWindowDataset(Dataset):
    """Retrospective windows; future truth is used only as the supervised target.

    Slice both space and time BEFORE compute. Include missingness channels rather
    than confusing missing water/coast data with a real normalized zero.
    Publication latency still needs a provider-specific as-of archive for operations.
    """
    def __init__(self, root, features, lookback=14, horizon=7, split="train", train_fraction=0.7,
                 patch_size=32, seed=42, statistics=None):
        if split not in ("train", "val") or not 0 < train_fraction < 1:
            raise ValueError("Invalid split")
        if min(lookback, horizon, patch_size) < 1:
            raise ValueError("Window and patch sizes must be positive")
        if not features or len(set(features)) != len(features) or "degree_heating_weeks" in features:
            raise ValueError("Use unique input features; keep DHW as a reference product")
        self.ds = derive_features(open_cube(root))
        if "available_not_before" in self.ds.coords:
            raise ValueError("Monthly context needs an explicit causal as-of join before daily training")
        if set(features)-set(self.ds):
            raise ValueError("Requested input feature is absent")
        if np.any(np.diff(self.ds.time.values) != np.timedelta64(1, "D")):
            raise ValueError("Training requires contiguous UTC days")
        for name in features:
            if self.ds[name].dims == ("time",):
                self.ds[name] = xr.broadcast(self.ds[name], self.ds.sea_surface_temperature)[0].transpose("time", "lat", "lon")
            if self.ds[name].dims != ("time", "lat", "lon"):
                raise ValueError("Input fields must use time,lat,lon")
        self.features, self.lookback, self.horizon = features, lookback, horizon
        self.patch, self.seed, self.epoch = patch_size, seed, 0
        boundary = int(self.ds.sizes["time"]*train_fraction)
        # Entire windows lie on one side: there is no overlapping target split.
        self.offset, length = (0, boundary) if split == "train" else (boundary, self.ds.sizes["time"]-boundary)
        self.count = length-lookback-horizon+1
        if self.count <= 0:
            raise ValueError("Split too short for complete windows")
        if patch_size > min(self.ds.sizes["lat"], self.ds.sizes["lon"]):
            raise ValueError("patch_size exceeds spatial support")
        if statistics is None:
            training = self.ds[features].isel(time=slice(0, boundary))
            means, stds = training.mean(), training.std()
            self.statistics = {name: (float(means[name].compute()), float(stds[name].compute())) for name in features}
        else:
            self.statistics = statistics
        if any(not np.isfinite(mean+std) for mean, std in self.statistics.values()):
            raise ValueError("Feature statistics have no finite training support")
        self.target_stats = self.statistics.get("sea_surface_temperature")
        if self.target_stats is None:
            raise ValueError("SST must be one of the input features")

    def __len__(self):
        return self.count

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        row = int(rng.integers(0, self.ds.sizes["lat"]-self.patch+1))
        col = int(rng.integers(0, self.ds.sizes["lon"]-self.patch+1))
        begin = self.offset+index
        crop = self.ds.isel(lat=slice(row, row+self.patch), lon=slice(col, col+self.patch),
                            time=slice(begin, begin+self.lookback+self.horizon))
        inputs = []
        for name in self.features:
            mean, std = self.statistics[name]
            raw = crop[name].isel(time=slice(0, self.lookback)).values
            inputs.append(np.nan_to_num((raw-mean)/max(std, 1e-6)))
            inputs.append(np.isfinite(raw).astype("float32"))
        truth = crop.sea_surface_temperature.isel(time=slice(self.lookback, None)).values
        last = crop.sea_surface_temperature.isel(time=self.lookback-1).values
        mean, std = self.target_stats
        target = (truth-mean)/max(std, 1e-6)
        result = {"x": torch.tensor(np.stack(inputs, axis=1), dtype=torch.float32),
            "target": torch.tensor(np.nan_to_num(target), dtype=torch.float32),
            "target_valid": torch.tensor(np.isfinite(truth)),
            "last_sst": torch.tensor(last, dtype=torch.float32),
            "latitude": torch.tensor(crop.lat.values, dtype=torch.float64),
            "longitude": torch.tensor(crop.lon.values, dtype=torch.float64)}
        # A deliberately simple issuance-time persistence baseline for forcing.
        forcing_names = ["water_u", "water_v", "mixed_layer_depth", "net_heat_flux"]
        if set(forcing_names) <= set(crop):
            result["forcing"] = {name: torch.tensor(np.repeat(
                crop[name].isel(time=self.lookback-1).values[None], self.horizon, axis=0), dtype=torch.float32)
                for name in forcing_names}
        return result

    def close(self):
        self.ds.close()


class MultivariateSSTForecaster(nn.Module):
    def __init__(self, feature_count, hidden_channels=16, horizon=7, sst_channel=0):
        super().__init__()
        if min(feature_count, hidden_channels, horizon) <= 0 or not 0 <= sst_channel < feature_count*2:
            raise ValueError("Invalid model dimensions")
        self.encoder = ConvLSTMCell(feature_count*2, hidden_channels)
        self.decoder = ConvLSTMCell(1, hidden_channels)
        self.output = nn.Conv2d(hidden_channels, 1, 1)
        self.hidden, self.horizon, self.sst_channel = hidden_channels, horizon, sst_channel

    def forward(self, inputs):
        batch, _, _, height, width = inputs.shape
        state = (inputs.new_zeros(batch, self.hidden, height, width),
                 inputs.new_zeros(batch, self.hidden, height, width))
        for step in inputs.unbind(dim=1):
            state = self.encoder(step, state)
        current = inputs[:, -1, self.sst_channel:self.sst_channel+1]
        predictions = []
        for _ in range(self.horizon):
            state = self.decoder(current, state)
            current = self.output(state[0])
            predictions.append(current[:, 0])
        return torch.stack(predictions, dim=1)
