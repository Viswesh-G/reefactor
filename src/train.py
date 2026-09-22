import torch
import torch.nn as nn
import torch.optim as optim
from src.models import SST_ConvLSTM
from src.physics import PhyLoss

# Training a PINN involves balancing two different objectives:
# 1. Data Match (MSE Loss)
# 2. Physics Rules (PhyLoss)
#
# An imbalance between these two can lead to "Gradient Pathology," where the 
# network focuses heavily on one and ignores the other.
# We address this using Gradient-Based Learning Rate Annealing. This method 
# dynamically scales the physics weight (lambda_phy) based on the gradient 
# magnitudes, ensuring both losses contribute evenly to weight updates.

def train_pinn(model, dataloader, epochs=50, alpha=0.9):
    """
    Trains the ConvLSTM model using both Data Loss and Physics Loss.
    
    Args:
        model: The SST_ConvLSTM PyTorch model.
        dataloader: PyTorch DataLoader providing batches of data.
        epochs: Number of times to loop through the entire dataset.
        alpha: Smoothing factor for the dynamic weight update (keeps things stable).
    """
    
    # ---------------------------------------------------------
    # 1. Setup Optimizers and Loss Functions
    # ---------------------------------------------------------
    # AdamW is a robust optimizer that updates the network weights
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Data Loss: Mean Squared Error (Standard way to compare two images)
    criterion_data = nn.MSELoss()
    
    # Physics Loss: Our custom Advection-Diffusion equation checker
    criterion_physics = PhyLoss(K=0.05) 
    
    # Initial weights for the two subjects (Data vs Physics)
    lambda_data = 1.0  # Data weight stays constant
    lambda_phy = 1.0   # Physics weight starts at 1, but we will dynamically change it!
    
    print("Starting Physics-Informed Training...")
    
    # ---------------------------------------------------------
    # 2. The Main Training Loop
    # ---------------------------------------------------------
    for epoch in range(epochs):
        epoch_data_loss = 0.0
        epoch_phy_loss = 0.0
        
        for batch in dataloader:
            # We assume the dataloader gives us 4 things:
            # x_sst: Past temperatures (input)
            # y_true_sst: Future temperatures (ground truth target)
            # u_vel, v_vel: Ocean currents (needed for physics loss)
            x_sst, y_true_sst, u_vel, v_vel = batch
            
            # Step 1: Clear out old memories of gradients from the previous batch
            optimizer.zero_grad()
            
            # Step 2: Forward Pass - Make predictions!
            preds = model(x_sst) 
            
            # Step 3: Calculate the individual "scores" (Losses)
            loss_data = criterion_data(preds, y_true_sst)
            loss_phy = criterion_physics(preds, u_vel, v_vel)
            
            # ==============================================================
            # THE MAGIC TRICK: Dynamic Loss Balancing
            # ==============================================================
            # We need to look inside the network's brain to see how loud the 
            # two losses are shouting. We look at the very last convolutional layer.
            target_layer = model.final_conv.weight
            
            # Get the gradients (the "shouting volume") of each loss independently.
            # retain_graph=True is needed because we are doing multiple backward passes.
            grad_data = torch.autograd.grad(loss_data, target_layer, retain_graph=True)[0]
            grad_phy = torch.autograd.grad(loss_phy, target_layer, retain_graph=True)[0]
            
            # Find the loudest shout from Data, and the average shout from Physics
            max_grad_data = torch.max(torch.abs(grad_data))
            mean_grad_phy = torch.mean(torch.abs(grad_phy))
            
            # Calculate the fair weight! 
            # We add a tiny number (1e-8) to prevent a "divide by zero" math error.
            lambda_hat = max_grad_data / (mean_grad_phy + 1e-8)
            
            # We don't want the weight to jump around wildly every batch.
            # So, we use an Exponential Moving Average (EMA) to smoothly transition it.
            lambda_phy = (1 - alpha) * lambda_phy + alpha * lambda_hat.item()
            
            # ==============================================================
            
            # Step 4: Combine the scores into one Final Total Loss using our new balanced weight
            loss_total = (lambda_data * loss_data) + (lambda_phy * loss_phy)
            
            # Step 5: Backward Pass - Send the combined feedback backward through the network
            loss_total.backward()
            
            # Optional but good practice: "Gradient Clipping" prevents feedback from exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step 6: Update the network's weights to be slightly smarter for next time
            optimizer.step()
            
            # Record keeping
            epoch_data_loss += loss_data.item()
            epoch_phy_loss += loss_phy.item()
            
        # Print a report card at the end of each epoch
        num_batches = len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Data Loss: {epoch_data_loss/num_batches:.4f} | "
              f"Physics Loss: {epoch_phy_loss/num_batches:.4f} | "
              f"Physics Weight (lambda_phy): {lambda_phy:.4f}")

    print("Training Complete!")
    return model

# ---------------------------------------------------------
# Mock execution block for testing the script structure
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Testing the model and physics initialization...")
    
    # 1. Create a dummy model
    # Predicting 1 channel (SST), 32 hidden channels, predicting 3 future steps
    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=3)
    
    # 2. Create dummy input data to simulate a batch
    # Shape: (Batch=2, Time=5 days, Channels=1, Height=16, Width=16)
    x_sst_dummy = torch.rand(2, 5, 1, 16, 16)
    y_true_sst_dummy = torch.rand(2, 3, 1, 16, 16)
    u_vel_dummy = torch.rand(2, 3, 1, 16, 16) * 0.1 # Small currents
    v_vel_dummy = torch.rand(2, 3, 1, 16, 16) * 0.1
    
    # 3. Test forward pass
    preds = model(x_sst_dummy)
    print(f"Prediction output shape: {preds.shape}") # Should be (2, 3, 1, 16, 16)
    
    # 4. Test Physics loss
    physics_criterion = PhyLoss(K=0.05)
    p_loss = physics_criterion(preds, u_vel_dummy, v_vel_dummy)
    print(f"Initial Physics Loss value: {p_loss.item():.6f}")
    
    print("All modules loaded successfully! Ready for real NOAA data.")
