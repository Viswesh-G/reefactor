"""
run_ablation.py — Physics Ablation Study Entry Point

Tests what specific physical term drives performance improvement:

    Full PINN:     dT/dt + u·∂T/∂x + v·∂T/∂y = K·∇²T  (K=0.05)
    Ablation:      dT/dt + u·∂T/∂x + v·∂T/∂y = 0       (K=0.0, diffusion only)

Both use the same Wang et al. 2021 dynamic gradient balancing. The difference
in validation loss isolates whether diffusion (K>0) is what helps.

Usage:
    python run_ablation.py
  or
    make ablation

Checkpoints:  data/checkpoints/ablation/ablation_best_model.pt
Training log: data/checkpoints/ablation/ablation_training_log.csv
"""
import os
from torch.utils.data import DataLoader

from src.config_validator import ConfigValidator
from src.models import SST_ConvLSTM
from src.dataset import ReefDataset
from src.ablation import train_ablation


def main():
    print("Initializing Physics Ablation Study (K=0: advection only, no diffusion)...")

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

    checkpoint_dir = os.path.join(
        config.get("storage", {}).get("checkpoint_dir", "data/checkpoints"),
        "ablation",
    )

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
        print(f"Could not load Zarr data ({e}).")
        print("Run `make ingest` first.")
        return

    # Same architecture — ONLY physics_K is changed to 0.0
    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=horizon)

    train_ablation(
        model,
        train_loader,
        val_loader=val_loader,
        config=config,
        ablation_K=0.0,          # Key: diffusion term disabled
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main()
