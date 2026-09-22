import torch
from src.models import SST_ConvLSTM
from src.physics import PhyLoss
from src.dataset import ReefDataset
from src.train import train_pinn
from torch.utils.data import DataLoader

def main():
    print("Initializing Physics-Informed Neural Network (PINN) Training...")
    
    zarr_path = "data/processed/unified_gbr_canvas.zarr"
    
    # Try loading real data from the Zarr store produced by `make ingest`
    try:
        dataset = ReefDataset(zarr_path, lookback=14, horizon=7)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        print(f"Loaded {len(dataset)} samples from {zarr_path}")
    except Exception as e:
        print(f"Could not load Zarr data ({e}). Running with dummy tensors instead.")
        dataloader = None
    
    # Initialize the ConvLSTM model
    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=7)
    
    if dataloader is not None and len(dataloader) > 0:
        # Train on real data
        model = train_pinn(model, dataloader, epochs=2)
    else:
        # Fallback: quick smoke test with dummy data
        print("Running smoke test with random tensors...")
        x = torch.rand(2, 14, 1, 16, 16)
        preds = model(x)
        print(f"  Forward pass OK. Output shape: {preds.shape}")
        
        phy = PhyLoss(K=0.05)
        u = torch.rand(2, 7, 1, 16, 16) * 0.1
        v = torch.rand(2, 7, 1, 16, 16) * 0.1
        loss = phy(preds, u, v)
        print(f"  Physics loss: {loss.item():.6f}")
        print("Smoke test passed. Run `make ingest` first to train on real data.")

if __name__ == "__main__":
    main()
