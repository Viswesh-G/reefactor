"""Train the multivariate heat-budget PINN on a completed daily research run."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.reef.learning import FeatureWindowDataset, MultivariateSSTForecaster
from src.reef.physics import ForcedSurfaceBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True, help="New experiment directory; existing runs are protected")
    parser.add_argument("--features", nargs="+", default=["sea_surface_temperature", "salinity", "water_u", "water_v",
                        "mixed_layer_depth", "net_heat_flux", "chlorophyll", "oxygen"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--patch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.epochs < 1 or args.physics_weight < 0:
        parser.error("epochs must be positive; physics weight must be nonnegative")
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    # CPU and one DataLoader worker give a reproducible baseline on a small machine.
    torch.set_num_threads(1)
    common = dict(root=args.root, features=args.features, lookback=args.lookback, horizon=args.horizon,
                  patch_size=args.patch, seed=args.seed)
    train = FeatureWindowDataset(**common, split="train")
    val = FeatureWindowDataset(**common, split="val", statistics=train.statistics)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = MultivariateSSTForecaster(len(args.features), horizon=args.horizon,
                                      sst_channel=args.features.index("sea_surface_temperature")*2)
    physics = ForcedSurfaceBudget(learn_k=True)
    optimizer = torch.optim.AdamW(list(model.parameters())+list(physics.parameters()), lr=1e-3)
    mean, std = train.target_stats
    std = max(std, 1e-6)
    best = float("inf")
    history = []
    try:
        for epoch in range(args.epochs):
            train.set_epoch(epoch)
            model.train()
            total, samples = 0.0, 0
            for batch in DataLoader(train, batch_size=1, shuffle=False, num_workers=0):
                optimizer.zero_grad()
                prediction = model(batch["x"])
                mask = batch["target_valid"]
                if not bool(mask.any()):
                    continue
                data_loss = (prediction[mask]-batch["target"][mask]).square().mean()
                loss = data_loss
                if args.physics_weight:
                    if "forcing" not in batch:
                        raise ValueError("Heat PINN requires currents, mixed layer depth and net heat flux")
                    physical = prediction*std+mean
                    sequence = torch.cat([batch["last_sst"][:, None], physical], dim=1)
                    loss = loss + args.physics_weight*physics(sequence, batch["forcing"],
                                batch["latitude"][0], batch["longitude"][0])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(physics.parameters()), 1.0)
                optimizer.step()
                total, samples = total+float(loss.detach()), samples+1
            if not samples:
                raise ValueError("No valid training samples")
            model.eval()
            sse, count = 0.0, 0
            with torch.no_grad():
                for batch in DataLoader(val, batch_size=1, num_workers=0):
                    prediction = model(batch["x"])
                    mask = batch["target_valid"]
                    sse += float(((prediction[mask]-batch["target"][mask])*std).square().sum())
                    count += int(mask.sum())
            if not count:
                raise ValueError("No valid validation targets")
            record = {"epoch": epoch+1, "train_loss": total/samples, "val_rmse_celsius": (sse/count)**0.5}
            history.append(record)
            checkpoint = {"model": model.state_dict(), "physics": physics.state_dict(),
                "optimizer": optimizer.state_dict(), "statistics": train.statistics, "arguments": vars(args),
                "epoch": epoch+1, "torch_rng": torch.get_rng_state(), "evaluation_mode": "retrospective_persistent_forcing"}
            checkpoint["data_fingerprint"] = json.loads((Path(args.root)/"manifest.json").read_text(encoding="utf-8"))["fingerprint"]
            torch.save(checkpoint, output / "latest.pt")
            if record["val_rmse_celsius"] < best:
                best = record["val_rmse_celsius"]
                torch.save(checkpoint, output / "best.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps(record))
    finally:
        train.close()
        val.close()


if __name__ == "__main__":
    main()
