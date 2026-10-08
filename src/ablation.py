"""
src/ablation.py — Ablation Study Utilities

An ablation study isolates the exact contribution of a single component.
In REEFS_ML the core claim is: "the advection-diffusion physics loss improves
forecast skill". To prove this, we need two otherwise identical conditions:

    Condition A — PhyConvLSTM:    ConvLSTM + physics loss (physics_K > 0)
    Condition B — Ablation:       ConvLSTM + physics loss (physics_K = 0.0)

The ablation (B) is different from the Vanilla ConvLSTM baseline in run_baseline.py:

    Vanilla ConvLSTM:  λ_data loss only — no physics loss at all
    Ablation:          Same PINN training loop including dynamic λ_phy balancing,
                       but with K=0 so the diffusion term vanishes, leaving only
                       pure advection: dT/dt + u·∂T/∂x + v·∂T/∂y = 0

This is a stronger test: if the model still benefits from the physics loss even
when K=0 (pure advection, no diffusion), we can separate the contributions
of the two physical terms. If it doesn't, the gain comes from diffusion alone.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from src.models import SST_ConvLSTM
from src.physics import PhyLoss


def train_ablation(
    model,
    train_loader,
    val_loader=None,
    config=None,
    ablation_K: float = 0.0,
    checkpoint_dir: str = "data/checkpoints/ablation",
):
    """
    Trains the PhyConvLSTM with physics_K forced to a specified value.

    When ablation_K=0.0:  Only the advection term remains in the PDE loss.
                          dT/dt + u·∂T/∂x + v·∂T/∂y = 0
                          (Diffusion contribution is ablated)

    When ablation_K=0.05: Full advection-diffusion (same as PhyConvLSTM).

    The training loop is identical to train_pinn() — same EMA balancing,
    same gradient clipping — so results are directly comparable.

    Args:
        model:          SST_ConvLSTM instance.
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        config:         Full config dict (for epochs, lr, etc.).
        ablation_K:     Overrides physics_K. Set 0.0 to ablate diffusion.
        checkpoint_dir: Where to save ablation checkpoints.

    Returns:
        Trained model.
    """
    if config is not None:
        train_cfg  = config.get("training", {})
        epochs     = train_cfg.get("epochs", 50)
        lr         = train_cfg.get("learning_rate", 1e-3)
        wd         = train_cfg.get("weight_decay", 1e-4)
        alpha      = train_cfg.get("ema_alpha", 0.9)
        grad_clip  = train_cfg.get("gradient_clip", 1.0)
    else:
        epochs, lr, wd, alpha, grad_clip = 50, 1e-3, 1e-4, 0.9, 1.0

    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer         = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion_data    = nn.MSELoss()
    # ablation_K overrides whatever is in config
    criterion_physics = PhyLoss(K=ablation_K)

    lambda_data = 1.0
    lambda_phy  = 1.0
    best_val    = float("inf")

    log_path   = os.path.join(checkpoint_dir, "ablation_training_log.csv")
    log_fields = ["epoch", "data_loss", "phy_loss", "lambda_phy", "val_loss", "ablation_K"]
    log_file   = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    print(f"Starting Ablation Training (physics_K={ablation_K})...")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Log:            {log_path}")
    print(f"  Epochs: {epochs} | LR: {lr}")

    for epoch in range(epochs):
        model.train()
        epoch_data_loss = 0.0
        epoch_phy_loss  = 0.0

        for batch in train_loader:
            x_sst, y_true_sst, u_vel, v_vel = batch
            optimizer.zero_grad()

            preds     = model(x_sst)
            loss_data = criterion_data(preds, y_true_sst)
            loss_phy  = criterion_physics(preds, u_vel, v_vel)

            # Same Wang et al. 2021 dynamic balancing as the full PINN
            target_layer = model.final_conv.weight
            grad_data    = torch.autograd.grad(loss_data, target_layer, retain_graph=True)[0]
            grad_phy     = torch.autograd.grad(loss_phy,  target_layer, retain_graph=True)[0]

            max_grad_data  = torch.max(torch.abs(grad_data))
            mean_grad_phy  = torch.mean(torch.abs(grad_phy))
            lambda_hat     = max_grad_data / (mean_grad_phy + 1e-8)
            lambda_phy     = (1 - alpha) * lambda_phy + alpha * lambda_hat.item()

            loss_total = (lambda_data * loss_data) + (lambda_phy * loss_phy)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_data_loss += loss_data.item()
            epoch_phy_loss  += loss_phy.item()

        n_train       = len(train_loader)
        avg_data_loss = epoch_data_loss / n_train
        avg_phy_loss  = epoch_phy_loss  / n_train

        val_info = ""
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_sst, y_true_sst, _, _ = batch
                    preds      = model(x_sst)
                    val_total += criterion_data(preds, y_true_sst).item()
            val_loss = val_total / len(val_loader)
            val_info = f" | Val Loss: {val_loss:.4f}"

            if val_loss < best_val:
                best_val  = val_loss
                best_path = os.path.join(checkpoint_dir, "ablation_best_model.pt")
                torch.save(model.state_dict(), best_path)
                val_info += " ← best"

        # Latest checkpoint (resumable)
        torch.save({
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lambda_phy":           lambda_phy,
            "best_val_loss":        best_val,
            "ablation_K":           ablation_K,
        }, os.path.join(checkpoint_dir, "ablation_latest_checkpoint.pt"))

        log_writer.writerow({
            "epoch":      epoch + 1,
            "data_loss":  f"{avg_data_loss:.6f}",
            "phy_loss":   f"{avg_phy_loss:.6f}",
            "lambda_phy": f"{lambda_phy:.6f}",
            "val_loss":   f"{val_loss:.6f}" if not (val_loss != val_loss) else "nan",
            "ablation_K": ablation_K,
        })
        log_file.flush()

        print(
            f"Ablation Epoch [{epoch+1:>3}/{epochs}] | "
            f"Data: {avg_data_loss:.4f} | Phy(K={ablation_K}): {avg_phy_loss:.4f} | "
            f"λ_phy: {lambda_phy:.4f}"
            f"{val_info}"
        )

    log_file.close()
    print(f"Ablation Training Complete! Log: {log_path}")
    return model
