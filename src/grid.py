"""Small grid helpers shared by ingestion and climatology derivation."""

import numpy as np


def build_axis(start: float, stop: float, resolution: float) -> np.ndarray:
    """Build an inclusive, evenly spaced axis without floating-point drift."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    count = int(round((stop - start) / resolution))
    if count < 1 or not np.isclose(start + count * resolution, stop):
        raise ValueError(
            f"Grid bounds {start}..{stop} are not divisible by {resolution}"
        )
    return np.linspace(start, stop, count + 1, dtype=np.float64)
