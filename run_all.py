"""
run_all.py — Full REEFS_ML Pipeline Orchestrator

Executes the entire research pipeline in strict order:
    1. Ingest      → stream NOAA + NESDIS satellite data → write Zarr
    2. Derive MMM  → recover the true NOAA CRW MMM climatology sidecar
    3. Train       → PhyConvLSTM (physics-constrained, learns K)
    4. Baseline    → Vanilla ConvLSTM (no physics)
    5. Ablation    → advection-only (no diffusion)
    6. NOAA DHW    → Operational algorithm baseline
    7. Eval        → PINN evaluation on the held-out validation period
    8. Early warn  → physics-residual early-warning precursor analysis
    9. Plot        → Generate all paper figures

Usage:
    python run_all.py              # Full pipeline
    python run_all.py --from train # Skip ingestion, start from training
    python run_all.py --from eval  # Skip training, start from evaluation

Flags:
    --from [ingest|derive_mmm|train|baseline|ablation|noaa|eval|early_warning|plot]
    --skip-ingest   Skip ingestion (if Zarr already exists)
    --skip-train    Skip training (if checkpoints already exist)
"""
import argparse
import os
import sys
import time


PHASE_ORDER = ["ingest", "derive_mmm", "train", "baseline", "ablation", "noaa", "eval", "early_warning", "plot"]


def banner(phase: str):
    width = 60
    print("\n" + "═" * width)
    print(f"  REEFS_ML │ Phase: {phase.upper()}")
    print("═" * width)


def run_phase(label: str, fn, skip_condition: bool = False, skip_reason: str = ""):
    banner(label)
    if skip_condition:
        print(f"  ⏭  Skipping — {skip_reason}")
        return True
    try:
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        print(f"\n  ✅ {label} completed in {elapsed/60:.1f} min")
        return True
    except Exception as e:
        print(f"\n  ❌ {label} FAILED: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# Phase runners
# ──────────────────────────────────────────────────────────────

def phase_ingest():
    from src.config_validator import ConfigValidator
    from src.data_pipeline import DataIngestionFSM
    validator    = ConfigValidator("config.yaml")
    valid_config = validator.validate()
    pipeline     = DataIngestionFSM(valid_config)
    pipeline.run()


def phase_derive_mmm():
    import run_derive_mmm
    run_derive_mmm.main()


def phase_train():
    import torch
    from src.config_validator import ConfigValidator
    from src.models import SST_ConvLSTM
    from src.dataset import ReefDataset
    from src.train import train_pinn
    from torch.utils.data import DataLoader

    validator = ConfigValidator("config.yaml")
    config    = validator.validate()

    ds_cfg       = config.get("dataset", {})
    train_cfg    = config.get("training", {})
    lookback     = ds_cfg.get("lookback", 14)
    horizon      = ds_cfg.get("horizon", 7)
    val_fraction = ds_cfg.get("val_fraction", 0.2)
    patch_size   = ds_cfg.get("patch_size", None)
    batch_size   = train_cfg.get("batch_size", 4)

    zarr_path = config["storage"]["processed_zarr_path"]
    train_dataset = ReefDataset(zarr_path, lookback=lookback, horizon=horizon,
                                split="train", val_fraction=val_fraction, patch_size=patch_size)
    val_dataset   = ReefDataset(zarr_path, lookback=lookback, horizon=horizon,
                                split="val",   val_fraction=val_fraction, patch_size=patch_size)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Training: {len(train_dataset)} samples | Validation: {len(val_dataset)} samples")
    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=horizon)
    train_pinn(model, train_loader, val_loader=val_loader, config=config)


def phase_baseline():
    import run_baseline
    run_baseline.main()


def phase_ablation():
    import run_ablation
    run_ablation.main()


def phase_noaa():
    import run_noaa_dhw_baseline
    run_noaa_dhw_baseline.main()


def phase_eval():
    import run_eval
    run_eval.main()


def phase_early_warning():
    import run_early_warning
    run_early_warning.main()


def phase_plot():
    import run_plot_figures
    run_plot_figures.main()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="REEFS_ML Full Pipeline Orchestrator")
    parser.add_argument("--from", dest="start_from", default="ingest",
                        choices=PHASE_ORDER,
                        help="Phase to start from (skips all earlier phases)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion even if Zarr doesn't exist (unsafe)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip PhyConvLSTM training")
    args = parser.parse_args()

    # One seed for the whole run so every phase is reproducible.
    from src.config_validator import ConfigValidator
    from src.reproducibility import set_seed, get_seed
    _cfg = ConfigValidator("config.yaml").validate()
    set_seed(get_seed(_cfg))

    start_idx = PHASE_ORDER.index(args.start_from)

    print("╔══════════════════════════════════════════════════════╗")
    print("║         REEFS_ML — Full Research Pipeline            ║")
    print("║  PhyConvLSTM for GBR Coral Bleaching Prediction      ║")
    print("╚══════════════════════════════════════════════════════╝")
    configured_zarr = _cfg["storage"]["processed_zarr_path"]
    configured_checkpoints = _cfg["storage"]["checkpoint_dir"]
    configured_mmm = _cfg["storage"]["crw_mmm_path"]
    pinn_ckpt = os.path.join(configured_checkpoints, "best_model.pt")
    print(f"\n  Starting from phase: {args.start_from.upper()}")
    print(f"  Zarr path:  {configured_zarr}")
    print(f"  PINN ckpt:  {pinn_ckpt}\n")

    phases = [
        ("ingest",   phase_ingest,
         start_idx > 0 or (args.skip_ingest and os.path.exists(configured_zarr)),
         f"Zarr already exists at {configured_zarr}" if os.path.exists(configured_zarr) else "Skipped by flag"),
        ("derive_mmm", phase_derive_mmm,
         start_idx > PHASE_ORDER.index("derive_mmm") or os.path.exists(configured_mmm),
         f"MMM sidecar already exists at {configured_mmm}"),
        ("train",    phase_train,
         start_idx > PHASE_ORDER.index("train") or args.skip_train,
         "checkpoint already exists" if os.path.exists(pinn_ckpt) else "Skipped by flag"),
        ("baseline", phase_baseline,
         start_idx > PHASE_ORDER.index("baseline"),
         "Skipped (start_from is later)"),
        ("ablation", phase_ablation,
         start_idx > PHASE_ORDER.index("ablation"),
         "Skipped (start_from is later)"),
        ("noaa",     phase_noaa,
         start_idx > PHASE_ORDER.index("noaa"),
         "Skipped (start_from is later)"),
        ("eval",     phase_eval,
         start_idx > PHASE_ORDER.index("eval"),
         "Skipped (start_from is later)"),
        ("early_warning", phase_early_warning,
         start_idx > PHASE_ORDER.index("early_warning"),
         "Skipped (start_from is later)"),
        ("plot",     phase_plot,
         False, ""),
    ]

    results = {}
    for label, fn, skip, skip_reason in phases:
        ok = run_phase(label, fn, skip_condition=skip, skip_reason=skip_reason)
        results[label] = "✅" if ok else "❌"
        if not ok:
            print(f"\n  Pipeline halted at phase: {label.upper()}")
            print("  Fix the error above and re-run with --from", label)
            break

    # Final summary
    print("\n" + "═" * 60)
    print("  PIPELINE SUMMARY")
    print("═" * 60)
    for label, status in results.items():
        print(f"  {status}  {label.capitalize()}")
    print("═" * 60)
    print("\n  Figures saved to: data/checkpoints/figures/")
    print("  Training logs:   data/checkpoints/training_log.csv")
    print("  Baseline log:    data/checkpoints/baseline/baseline_training_log.csv\n")


if __name__ == "__main__":
    main()
