import argparse
import os
import torch
from src.config_validator import ConfigValidator
from src.models import SST_ConvLSTM
from src.dataset import ReefDataset
from src.train import train_pinn, resume_from_checkpoint, count_parameters
from src.reproducibility import set_seed, get_seed, get_device
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Train PhyConvLSTM-DHW")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from latest_checkpoint.pt instead of starting fresh."
    )
    args = parser.parse_args()

    print("Initializing Physics-Informed Neural Network (PINN) Training...")

    validator = ConfigValidator("config.yaml")
    config    = validator.validate()

    # Fix the random seed before building anything random (model init, shuffling).
    set_seed(get_seed(config))
    device = get_device(config)
    print(f"  Seed: {get_seed(config)} | Device: {device}")

    zarr_path  = config["storage"]["processed_zarr_path"]
    ds_cfg     = config.get("dataset", {})
    train_cfg  = config.get("training", {})

    lookback       = ds_cfg.get("lookback", 14)
    horizon        = ds_cfg.get("horizon", 7)
    val_fraction   = ds_cfg.get("val_fraction", 0.2)
    patch_size     = ds_cfg.get("patch_size", None)
    batch_size     = train_cfg.get("batch_size", 4)
    checkpoint_dir = config.get("storage", {}).get("checkpoint_dir", "data/checkpoints")

    # --- Build the Model ---
    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=horizon)
    print("\nModel architecture:")
    count_parameters(model)

    # --- Resume from checkpoint if requested ---
    if args.resume:
        ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        try:
            model, start_epoch, _ = resume_from_checkpoint(model, ckpt_path)
            print(f"  Resuming from epoch {start_epoch + 1}...")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            print("  Starting fresh training run instead.")

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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"  Training samples:   {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")

    except Exception as e:
        print(f"\nCould not load Zarr data ({e}).")
        print("Run `make ingest` first to build the data store.")
        print("Falling back to smoke test with random tensors...\n")

    if train_loader is not None and len(train_loader) > 0:
        model = train_pinn(model, train_loader, val_loader=val_loader, config=config)

    else:
        # Smoke test — verifies all modules connect correctly without real data
        print("Running smoke test with random tensors...")
        x     = torch.rand(2, lookback, 1, 16, 16)
        preds = model(x)
        print(f"  Forward pass OK. Output shape: {preds.shape}")

        from src.physics import PhyLoss
        phy  = PhyLoss(K=500.0)
        u    = torch.rand(2, horizon, 1, 16, 16) * 0.1
        v    = torch.rand(2, horizon, 1, 16, 16) * 0.1
        loss = phy(preds, u, v)
        print(f"  Physics loss: {loss.item():.6f}")
        print("Smoke test passed. Run `make ingest` first to train on real data.")


if __name__ == "__main__":
    main()
