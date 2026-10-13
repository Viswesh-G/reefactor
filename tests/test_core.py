"""Fast, dependency-light checks for the research pipeline's core contracts."""

import unittest

import numpy as np

from src.config_validator import ConfigValidator
from src.grid import build_axis


class CoreContractTests(unittest.TestCase):
    def test_config_is_valid_and_grid_is_inclusive(self):
        config = ConfigValidator("config.yaml").validate()
        axis = build_axis(-24.0, -10.0, 0.05)
        self.assertEqual(len(axis), 281)
        self.assertEqual(axis[0], -24.0)
        self.assertEqual(axis[-1], -10.0)
        self.assertEqual(config["temporal_canvas"]["start_date"], "2017-02-13")

    def test_dhw_uses_thresholded_full_hotspot(self):
        try:
            from src.metrics import compute_dhw_from_sst
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional scientific dependency unavailable: {exc}")
        sst = np.array([30.0, 31.0, 32.0, 30.5], dtype=np.float32)[:, None, None]
        dhw = compute_dhw_from_sst(sst, climatology_mean=30.0, window_days=84)
        expected = np.array([0.0, 1.0 / 7.0, 3.0 / 7.0, 3.0 / 7.0])
        np.testing.assert_allclose(dhw[:, 0, 0], expected)

    def test_physics_loss_is_finite_and_k_is_positive(self):
        try:
            import torch
            from src.physics import PhyLoss
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional ML dependency unavailable: {exc}")
        criterion = PhyLoss(K=500.0, dx=5500.0, dy=5500.0, dt=86400.0,
                            learn_K=True, K_min=10.0, K_max=5000.0)
        preds = torch.randn(2, 4, 1, 12, 12, requires_grad=True)
        velocity = torch.zeros(2, 4, 1, 12, 12)
        loss = criterion(preds, velocity, velocity)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(criterion.K_value(), 10.0)
        self.assertLessEqual(criterion.K_value(), 5000.0)


if __name__ == "__main__":
    unittest.main()
