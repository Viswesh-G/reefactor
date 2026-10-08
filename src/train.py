import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import SST_ConvLSTM
from src.physics import PhyLoss

# Training a PINN involves balancing two objectives that can conflict:
#   1. Data Match (MSE Loss)   — fit the observed satellite SST measurements
#   2. Physics Rules (PhyLoss) — obey the Advection-Diffusion equation
#
# "Gradient Pathology" is when one loss dominates and the other gets ignored.
# We prevent this with Gradient-Based Loss Annealing (Wang et al., 2021):
#   - Inspect how loudly each loss shouts at the final layer's gradients
#   - Dynamically rescale the physics weight so both contribute equally
#   - Smooth the rescaling with an Exponential Moving Average to avoid jumps


def train_pinn(model, train_loader, val_loader=None, config=None, epochs=50, alpha=0.9):
    """
    Trains the ConvLSTM model using both Data Loss and Physics Loss.
    Saves a checkpoint after every epoch and keeps track of the best model by validation loss.

    Args:
        model:        The SST_ConvLSTM PyTorch model to train.
        train_loader: DataLoader providing training batches.
        val_loader:   DataLoader providing validation batches (optional, but recommended).
        config:       The full config dict from config.yaml (overrides epoch/alpha defaults).
        epochs:       Number of full passes through the training data.
        alpha:        EMA smoothing for the dynamic physics weight (0 = no smoothing, 1 = frozen).
    """

    # --- Pull hyperparameters from config (with sensible defaults) ---
    if config is not None:
        train_cfg      = config.get("training", {})
        epochs         = train_cfg.get("epochs", epochs)
        lr             = train_cfg.get("learning_rate", 1e-3)
        wd             = train_cfg.get("weight_decay", 1e-4)
        alpha          = train_cfg.get("ema_alpha", alpha)
        grad_clip      = train_cfg.get("gradient_clip", 1.0)
        physics_K      = train_cfg.get("physics_K", 0.05)
        checkpoint_dir = config.get("storage", {}).get("checkpoint_dir", "data/checkpoints")
    else:
        lr, wd, grad_clip, physics_K = 1e-3, 1e-4, 1.0, 0.05
        checkpoint_dir = "data/checkpoints"

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Setup Optimizer and Loss Functions ---
    # AdamW decouples weight decay from the gradient update, which improves generalization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    criterion_data    = nn.MSELoss()          # Standard pixel-wise squared error
    criterion_physics = PhyLoss(K=physics_K)  # Advection-Diffusion residual penalty

    # Physics weight starts at 1.0 and is dynamically adjusted each batch
    lambda_data = 1.0
    lambda_phy  = 1.0

    best_val_loss = float("inf")

    # --- CSV Training Log ---
    # Logs one row per epoch: useful for generating Figure 3 (λ_phy convergence)
    # and lead-time RMSE decay curves in the paper.
    log_path = os.path.join(checkpoint_dir, "training_log.csv")
    log_fields = ["epoch", "data_loss", "phy_loss", "lambda_phy", "val_loss"]
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    print("Starting Physics-Informed Training...")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Training log:         {log_path}")
    print(f"  Epochs: {epochs} | LR: {lr} | Batch size set by DataLoader")

    for epoch in range(epochs):

        # =============================================
        # TRAINING PHASE
        # =============================================
        model.train()
        epoch_data_loss = 0.0
        epoch_phy_loss  = 0.0

        for batch in train_loader:
            x_sst, y_true_sst, u_vel, v_vel = batch

            # Step 1: Clear gradients from the previous batch
            optimizer.zero_grad()

            # Step 2: Forward pass — make predictions for the next `horizon` days
            preds = model(x_sst)

            # Step 3: Calculate both losses independently
            loss_data = criterion_data(preds, y_true_sst)
            loss_phy  = criterion_physics(preds, u_vel, v_vel)

            # ==================================================
            # THE BALANCING ACT: Dynamic Gradient Annealing
            # ==================================================
            # We peek at the gradients each loss would produce on the final
            # convolutional layer. The ratio of their magnitudes tells us
            # how much to scale the physics loss so neither drowns out the other.
            target_layer = model.final_conv.weight

            # retain_graph=True lets us do multiple backward passes before optimizer.step()
            grad_data = torch.autograd.grad(loss_data, target_layer, retain_graph=True)[0]
            grad_phy  = torch.autograd.grad(loss_phy,  target_layer, retain_graph=True)[0]

            # "How loud is the data loss shouting vs. the physics loss?"
            max_grad_data  = torch.max(torch.abs(grad_data))
            mean_grad_phy  = torch.mean(torch.abs(grad_phy))
            lambda_hat     = max_grad_data / (mean_grad_phy + 1e-8)

            # Smooth the weight update with EMA to prevent wild oscillations
            lambda_phy = (1 - alpha) * lambda_phy + alpha * lambda_hat.item()
            # ==================================================

            # Step 4: Combine losses with the dynamically balanced weight
            loss_total = (lambda_data * loss_data) + (lambda_phy * loss_phy)

            # Step 5: Backprop the combined loss
            loss_total.backward()

            # Gradient clipping: prevents any single weight update from being too large
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            # Step 6: Update the model weights
            optimizer.step()

            epoch_data_loss += loss_data.item()
            epoch_phy_loss  += loss_phy.item()

        n_train = len(train_loader)

        # =============================================
        # VALIDATION PHASE
        # =============================================
        val_info  = ""
        val_loss  = float("nan")

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

            # Keep the best model weights separately — this is what run_eval.py loads
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                val_info += " ← best"

        # =============================================
        # CHECKPOINT: Save latest state every epoch
        # =============================================
        # Saving optimizer state lets us resume training later from exactly this point.
        ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        torch.save(
            {
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lambda_phy":           lambda_phy,
                "best_val_loss":        best_val_loss,
            },
            ckpt_path,
        )

        # --- Write CSV log row ---
        avg_data_loss = epoch_data_loss / n_train
        avg_phy_loss  = epoch_phy_loss  / n_train
        log_writer.writerow({
            "epoch":      epoch + 1,
            "data_loss":  f"{avg_data_loss:.6f}",
            "phy_loss":   f"{avg_phy_loss:.6f}",
            "lambda_phy": f"{lambda_phy:.6f}",
            "val_loss":   f"{val_loss:.6f}" if not (val_loss != val_loss) else "nan",
        })
        log_file.flush()  # Write immediately so progress is visible even if training crashes

        # Print epoch summary
        print(
            f"Epoch [{epoch+1:>3}/{epochs}] | "
            f"Data: {avg_data_loss:.4f} | "
            f"Phy: {avg_phy_loss:.4f} | "
            f"λ_phy: {lambda_phy:.4f}"
            f"{val_info}"
        )

    log_file.close()
    print(f"Training Complete! Log saved to: {log_path}")
    return model


def resume_from_checkpoint(model, checkpoint_path: str, device: str = "cpu"):
    """
    Loads a latest_checkpoint.pt and returns the model with restored weights,
    plus the epoch to resume from and the last lambda_phy value.

    Use this to continue a training run that was interrupted:

        model, start_epoch, lambda_phy = resume_from_checkpoint(model, ckpt_path)
        # Then call train_pinn with epochs reduced by start_epoch,
        # or manage the loop manually.

    Args:
        model:           An SST_ConvLSTM instance (same architecture as when saved).
        checkpoint_path: Path to latest_checkpoint.pt saved by train_pinn().
        device:          'cpu' or 'cuda' — where to load the weights.

    Returns:
        model:       Model with restored weights.
        start_epoch: The next epoch to train (i.e., checkpoint epoch + 1).
        lambda_phy:  The last physics weight value, for EMA continuity.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run `make train` to start training from scratch."
        )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt.get("epoch", 0)          # epoch already completed
    lambda_phy  = ckpt.get("lambda_phy", 1.0)
    best_val    = ckpt.get("best_val_loss", float("inf"))
    print(f"Resumed from checkpoint: epoch {start_epoch} | "
          f"λ_phy={lambda_phy:.4f} | best_val={best_val:.4f}")
    return model, start_epoch, lambda_phy


def count_parameters(model) -> int:
    """
    Returns the total number of trainable parameters in the model.
    Used in the paper's model description section.

    Example output:
        SST_ConvLSTM trainable parameters: 1,234,567
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total:,}")
    return total


# -------------------------------------------------------
# Smoke test: verifies model + physics setup is correct
# -------------------------------------------------------
if __name__ == "__main__":
    print("Testing the model and physics initialization...")

    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=3)

    # Dummy batch: (Batch=2, Time=5 past days, Channels=1, H=16, W=16)
    x_sst_dummy      = torch.rand(2, 5, 1, 16, 16)
    y_true_sst_dummy  = torch.rand(2, 3, 1, 16, 16)
    u_vel_dummy       = torch.rand(2, 3, 1, 16, 16) * 0.1
    v_vel_dummy       = torch.rand(2, 3, 1, 16, 16) * 0.1

    preds = model(x_sst_dummy)
    print(f"Prediction output shape: {preds.shape}")  # Expected: (2, 3, 1, 16, 16)

    physics_criterion = PhyLoss(K=0.05)
    p_loss = physics_criterion(preds, u_vel_dummy, v_vel_dummy)
    print(f"Initial Physics Loss: {p_loss.item():.6f}")

    print("All modules OK. Ready for real NOAA data.")
