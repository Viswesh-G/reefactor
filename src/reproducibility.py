"""
Reproducibility helpers shared by every entry script.

Two tiny functions, used at the top of each run_*.py:
  - set_seed(seed)     : make Python/NumPy/PyTorch deterministic
  - get_device(config) : pick cuda if available else cpu (from config, with fallback)

Keeping these in one place means every experiment uses the *same* random seed
and device, which is a basic requirement for a publishable ML result.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed every random number generator we rely on (one call, fully reproducible)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic kernels (may be slightly slower but makes runs repeatable).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: dict) -> str:
    """Return 'cuda' if requested and available, otherwise fall back to 'cpu'."""
    requested = str(config.get("pipeline", {}).get("run", {}).get("device", "cpu"))
    if requested == "cuda" and not torch.cuda.is_available():
        print("  Note: device 'cuda' requested but unavailable — using CPU.")
        return "cpu"
    return requested


def get_seed(config: dict, default: int = 42) -> int:
    """Read the configured random seed (single source of truth = config.yaml)."""
    return int(config.get("pipeline", {}).get("run", {}).get("seed", default))
