"""
run_baseline.py — Vanilla ConvLSTM Baseline (no physics loss)

Trains the same SST_ConvLSTM architecture as run_pinns.py but with the
physics weight forced to zero (lambda_phy = 0). This is Baseline #2 in the paper:

    "Pure ConvLSTM — same architecture, MSE loss only, no physics penalty"

Purpose: isolate the contribution of the physics constraint by comparing
PhyConvLSTM vs. this baseline on the 2023 held-out test year.

Usage:
    python run_baseline.py
  or
    make baseline

Checkpoints are saved to:
    data/checkpoints/baseline_best_model.pt
    data/checkpoints/baseline_training_log.csv
"""
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config_validator import ConfigValidator
from src.models import SST_ConvLSTM
from src.dataset import ReefDataset


def train_baseline(model, train_loader, val_loader, config, checkpoint_dir):
    """
    Trains the ConvLSTM using MSE loss ONLY (no physics penalty).
    Mirrors the structure of train_pinn() for a fair comparison.
    """
    train_cfg  = config.get("training", {})
    epochs     = train_cfg.get("epochs", 50)
    lr         = train_cfg.get("learning_rate", 1e-3)
    wd         = train_cfg.get("weight_decay", 1e-4)
    grad_clip  = train_cfg.get("gradient_clip", 1.0)

    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer    = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion    = nn.MSELoss()
    best_val     = float("inf")

    # CSV log — same format as PhyConvLSTM log for easy side-by-side comparison
    log_path   = os.path.join(checkpoint_dir, "baseline_training_log.csv")
    log_fields = ["epoch", "data_loss", "phy_loss", "lambda_phy", "val_loss"]
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    print("Starting Vanilla ConvLSTM Baseline Training (physics_K = 0.0)...")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Training log:         {log_path}")
    print(f"  Epochs: {epochs} | LR: {lr}")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x_sst, y_true_sst, _, _ = batch   # velocities ignored (no physics loss)
            optimizer.zero_grad()
            preds     = model(x_sst)
            loss      = criterion(preds, y_true_sst)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        n_train    = len(train_loader)
        avg_loss   = epoch_loss / n_train

        # --- Validation ---
        val_info = ""
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_sst, y_true_sst, _, _ = batch
                    preds      = model(x_sst)
                    val_total += criterion(preds, y_true_sst).item()

            val_loss = val_total / len(val_loader)
            val_info = f" | Val Loss: {val_loss:.4f}"

            if val_loss < best_val:
                best_val  = val_loss
                best_path = os.path.join(checkpoint_dir, "baseline_best_model.pt")
                torch.save(model.state_dict(), best_path)
                val_info += " ← best"

        # --- Checkpoint ---
        ckpt_path = os.path.join(checkpoint_dir, "baseline_latest_checkpoint.pt")
        torch.save({
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss":        best_val,
        }, ckpt_path)

        # --- CSV log row (phy_loss = 0, lambda_phy = 0 by design) ---
        log_writer.writerow({
            "epoch":      epoch + 1,
            "data_loss":  f"{avg_loss:.6f}",
            "phy_loss":   "0.000000",
            "lambda_phy": "0.000000",
            "val_loss":   f"{val_loss:.6f}" if not (val_loss != val_loss) else "nan",
        })
        log_file.flush()

        print(
            f"Epoch [{epoch+1:>3}/{epochs}] | "
            f"Data: {avg_loss:.4f} | Phy: 0.0000 | λ_phy: 0.0000"
            f"{val_info}"
        )

    log_file.close()
    print(f"Baseline Training Complete! Log saved to: {log_path}")
    return model


def main():
    print("Initializing Vanilla ConvLSTM Baseline Training...")

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

    # Checkpoints go to a separate subdirectory so they don't collide with PINN checkpoints
    checkpoint_dir = os.path.join(
        config.get("storage", {}).get("checkpoint_dir", "data/checkpoints"),
        "baseline",
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
        print("Run `make ingest` first to build the data store.")
        return

    # Same architecture as PhyConvLSTM — ensures a fair comparison
    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=horizon)

    train_baseline(model, train_loader, val_loader, config, checkpoint_dir)


if __name__ == "__main__":
    main()
