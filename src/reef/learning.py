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
                 patch_size=32, seed=42, statistics=None, labels_path=None):
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
            if self.ds[name].attrs.get("role") == "simulated_response":
                raise ValueError(f"{name} is a simulated biological response, not an admitted measured driver")
            if self.ds[name].dims == ("time",):
                self.ds[name] = xr.broadcast(self.ds[name], self.ds.sea_surface_temperature)[0].transpose("time", "lat", "lon")
            if self.ds[name].dims != ("time", "lat", "lon"):
                raise ValueError("Input fields must use time,lat,lon")
            self.ds[name] = self.ds[name].where(self.ds.model_domain)
            supported = np.isfinite(self.ds[name]).all("time")
            if bool((self.ds.model_domain & ~supported).any().compute()):
                raise ValueError(f"{name} lacks complete support on the fixed model domain")
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
        self.labels = {}
        if labels_path:
            import polars as pl
            labels = pl.scan_parquet(labels_path).filter(pl.col("target_valid")).select(
                "survey_date", "lat", "lon", "bleaching_percent").collect()
            dy, dx = float(self.ds.lat[1]-self.ds.lat[0]), float(self.ds.lon[1]-self.ds.lon[0])
            for survey in labels.iter_rows(named=True):
                row = int(np.floor((survey["lat"]-(float(self.ds.lat[0])-dy/2))/dy))
                col = int(np.floor((survey["lon"]-(float(self.ds.lon[0])-dx/2))/dx))
                if 0 <= row < self.ds.sizes["lat"] and 0 <= col < self.ds.sizes["lon"] and bool(self.ds.model_domain.values[row, col]):
                    self.labels.setdefault(str(survey["survey_date"]), []).append((row, col, survey["bleaching_percent"]))

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
        target_date = str(crop.time.values[-1].astype("datetime64[D]"))
        observations = [(r-row, c-col, y) for r, c, y in self.labels.get(target_date, [])
                        if row <= r < row+self.patch and col <= c < col+self.patch]
        # No aggregation across protocols, no invented negatives at unsurveyed cells.
        result["bleaching_sites"] = torch.tensor([(r, c) for r, c, _ in observations], dtype=torch.long).reshape(-1, 2)
        result["bleaching_target"] = torch.tensor([y for _, _, y in observations], dtype=torch.float32)
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
    def __init__(self, feature_count, hidden_channels=16, horizon=7, sst_channel=0,
                 embedding_channels=None, bleaching_head=False, readout_channels=None):
        super().__init__()
        if min(feature_count, hidden_channels, horizon) <= 0 or not 0 <= sst_channel < feature_count*2:
            raise ValueError("Invalid model dimensions")
        channels = feature_count*2 if embedding_channels is None else embedding_channels
        self.project = nn.Identity() if embedding_channels is None else nn.Conv2d(feature_count*2, channels, 1)
        self.encoder = ConvLSTMCell(channels, hidden_channels)
        self.decoder = ConvLSTMCell(1, hidden_channels)
        def head():
            if readout_channels is None:
                return nn.Conv2d(hidden_channels, 1, 1)
            return nn.Sequential(nn.Conv2d(hidden_channels, readout_channels, 1), nn.GELU(),
                                 nn.Conv2d(readout_channels, 1, 1))
        self.output = head()
        self.hidden, self.horizon, self.sst_channel = hidden_channels, horizon, sst_channel
        self.bleaching = head() if bleaching_head else None

    def forward(self, inputs, return_bleaching=False):
        batch, _, _, height, width = inputs.shape
        state = (inputs.new_zeros(batch, self.hidden, height, width),
                 inputs.new_zeros(batch, self.hidden, height, width))
        for step in inputs.unbind(dim=1):
            state = self.encoder(self.project(step), state)
        current = inputs[:, -1, self.sst_channel:self.sst_channel+1]
        predictions = []
        for _ in range(self.horizon):
            state = self.decoder(current, state)
            current = self.output(state[0])
            predictions.append(current[:, 0])
        sst = torch.stack(predictions, dim=1)
        if return_bleaching:
            if self.bleaching is None:
                raise ValueError("This model was built without a bleaching head")
            return {"sst": sst, "bleaching_percent": 100*torch.sigmoid(self.bleaching(state[0])[:, 0])}
        return sst


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_forecaster(feature_count, parameter_budget=1_000_000, horizon=7, sst_channel=0,
                     bleaching_head=True):
    """Spend the budget on active weights; every feature enters the same projection.

    Tiny models compress all features into a small latent space. A larger weight
    count is capacity, not evidence of accuracy or a reason to invent input fields.
    """
    if feature_count < 1 or parameter_budget < 1:
        raise ValueError("Feature count and parameter budget must be positive")
    candidates = []
    heads = 2 if bleaching_head else 1
    for embedding in range(1, 65):
        for hidden in range(1, 257):
            base = (2*feature_count+1)*embedding + 72*hidden**2 + 36*(embedding+1)*hidden + 8*hidden
            count = base + heads*(hidden+1)
            if count <= parameter_budget:
                candidates.append((count, hidden, embedding, 0))
            readout = min(256, (parameter_budget-base-heads)//(heads*(hidden+2)))
            if readout > 0:
                count = base + heads*(readout*(hidden+2)+1)
                candidates.append((count, hidden, embedding, readout))
    if not candidates:
        raise ValueError("Budget cannot fit all input features and recurrent/output layers")
    count, hidden, embedding, readout = max(candidates)
    model = MultivariateSSTForecaster(feature_count, hidden, horizon, sst_channel,
                embedding_channels=embedding, bleaching_head=bleaching_head, readout_channels=readout or None)
    if parameter_count(model) != count:
        raise RuntimeError("Architecture count differs from the budget calculation")
    model.architecture = {"trainable_parameters": count, "parameter_budget": parameter_budget,
                          "feature_count": feature_count, "hidden_channels": hidden,
                          "embedding_channels": embedding, "bleaching_head": bleaching_head}
    model.architecture["readout_channels"] = readout or None
    return model
