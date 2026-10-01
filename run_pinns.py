import torch
from src.config_validator import ConfigValidator
from src.models import SST_ConvLSTM
from src.dataset import ReefDataset
from src.train import train_pinn
from torch.utils.data import DataLoader


def main():
    print("Initializing Physics-Informed Neural Network (PINN) Training...")

    # Load and validate all settings from config.yaml
    validator = ConfigValidator("config.yaml")
    config    = validator.validate()

    zarr_path  = config["storage"]["processed_zarr_path"]
    ds_cfg     = config.get("dataset", {})
    train_cfg  = config.get("training", {})

    lookback     = ds_cfg.get("lookback", 14)
    horizon      = ds_cfg.get("horizon", 7)
    val_fraction = ds_cfg.get("val_fraction", 0.2)
    patch_size   = ds_cfg.get("patch_size", None)
    batch_size   = train_cfg.get("batch_size", 4)

    # --- Load Train and Validation Datasets ---
    train_loader = None
    val_loader   = None
    try:
        train_dataset = ReefDataset(
            zarr_path, lookback=lookback, horizon=horizon,
            split="train", val_fraction=val_fraction, patch_size=patch_size,
        )
        val_dataset = ReefDataset(
            zarr_path, lookback=lookback, horizon=horizon,
            split="val", val_fraction=val_fraction, patch_size=patch_size,
        )
        # num_workers=0 keeps things simple on Windows (avoids multiprocessing issues)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"  Training samples:   {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")

    except Exception as e:
        print(f"Could not load Zarr data ({e}).")
        print("Run `make ingest` first to build the data store, or check your OPeNDAP connection.")
        print("Falling back to smoke test with random tensors...\n")

    # --- Build the Model ---
    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=horizon)

    if train_loader is not None and len(train_loader) > 0:
        # --- Full Training Run ---
        model = train_pinn(model, train_loader, val_loader=val_loader, config=config)

    else:
        # --- Smoke Test (no Zarr data) ---
        # Runs a single forward pass and physics loss check to make sure the code works.
        print("Running smoke test with random tensors...")
        x     = torch.rand(2, lookback, 1, 16, 16)
        preds = model(x)
        print(f"  Forward pass OK. Output shape: {preds.shape}")

        from src.physics import PhyLoss
        phy   = PhyLoss(K=0.05)
        u     = torch.rand(2, horizon, 1, 16, 16) * 0.1
        v     = torch.rand(2, horizon, 1, 16, 16) * 0.1
        loss  = phy(preds, u, v)
        print(f"  Physics loss: {loss.item():.6f}")
        print("Smoke test passed. Run `make ingest` first to train on real data.")


if __name__ == "__main__":
    main()
