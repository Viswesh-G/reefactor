"""
run_smoke_test.py -- End-to-end pipeline validation (no real data needed).

Tests every module with synthetic random tensors to catch shape bugs,
import errors, and gradient-flow problems before committing to a multi-hour
real-data training run.

Usage:
    python run_smoke_test.py
  or
    make smoke

What is tested:
    1. Config validation
    2. Model forward pass (correct output shape)
    3. Physics loss -- scalar K (fixed)
    4. Physics loss -- scalar K (learned)
    5. Physics loss -- spatial K (learned, per-pixel map)
    6. Replication-padding boundary fix (no NaN from padded derivatives)
    7. Training loop -- 2 batches, gradient flow, λ_phy update
    8. Metrics -- RMSE, ACC, NSE, DHW, bleaching alert
    9. Early-warning residual series
   10. Full-grid reconstruct_continuous

Exit code 0 = all tests pass. Non-zero = failure.
"""
import sys
import traceback

import numpy as np
import torch

PASS = "[PASS]"
FAIL = "[FAIL]"
results = {}


def run_test(name, fn):
    """Run fn(); record PASS/FAIL."""
    try:
        fn()
        results[name] = True
        print(f"  {PASS}  {name}")
    except Exception as e:
        results[name] = False
        print(f"  {FAIL}  {name}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Config validation
# ──────────────────────────────────────────────────────────────────────────────
def test_config():
    from src.config_validator import ConfigValidator
    cfg = ConfigValidator("config.yaml").validate()
    assert cfg["pipeline"]["name"], "Missing pipeline name"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Model forward pass
# ──────────────────────────────────────────────────────────────────────────────
BATCH, LOOKBACK, HORIZON, H, W = 2, 14, 7, 16, 16


def test_model_forward():
    from src.models import SST_ConvLSTM
    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=HORIZON)
    x = torch.rand(BATCH, LOOKBACK, 1, H, W)
    out = model(x)
    assert out.shape == (BATCH, HORIZON, 1, H, W), f"Bad shape: {out.shape}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Physics loss -- fixed scalar K
# ──────────────────────────────────────────────────────────────────────────────
def _dummy_uv():
    return (torch.rand(BATCH, HORIZON, 1, H, W) * 0.5,
            torch.rand(BATCH, HORIZON, 1, H, W) * 0.5)


def _dummy_preds():
    from src.models import SST_ConvLSTM
    model = SST_ConvLSTM(in_channels=1, hidden_dim=[32, 64], out_steps=HORIZON)
    x = torch.rand(BATCH, LOOKBACK, 1, H, W)
    return model(x)


def test_physics_fixed_K():
    from src.physics import PhyLoss
    phy = PhyLoss(K=500.0, learn_K=False)
    preds = _dummy_preds()
    u, v = _dummy_uv()
    loss = phy(preds, u, v)
    assert loss.item() >= 0, "Physics loss < 0 (impossible)"
    assert torch.isfinite(loss), "Physics loss is NaN/Inf"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Physics loss -- learned scalar K (gradient flows through log_K)
# ──────────────────────────────────────────────────────────────────────────────
def test_physics_learned_scalar_K():
    from src.physics import PhyLoss
    phy = PhyLoss(K=500.0, learn_K=True)
    optim_phy = torch.optim.Adam(phy.parameters(), lr=1e-3)

    preds = _dummy_preds()
    u, v = _dummy_uv()
    loss = phy(preds, u, v)
    loss.backward()
    assert phy.log_K.grad is not None, "No gradient on log_K"
    k_before = phy.K_value()
    optim_phy.step()
    k_after = phy.K_value()
    # K may or may not change (gradient might be tiny), but no error is the test
    assert 10.0 <= k_after <= 5000.0, f"K out of range: {k_after}"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Physics loss -- spatial K (per-pixel map)
# ──────────────────────────────────────────────────────────────────────────────
def test_physics_spatial_K():
    from src.physics import PhyLoss
    phy = PhyLoss(K=500.0, learn_K=True, K_spatial=True, grid_h=H, grid_w=W)
    assert hasattr(phy, "log_K_map"), "Missing log_K_map parameter"
    assert phy.log_K_map.shape == (1, 1, H, W)

    preds = _dummy_preds()
    u, v = _dummy_uv()
    loss = phy(preds, u, v)
    loss.backward()
    assert phy.log_K_map.grad is not None, "No gradient on log_K_map"
    assert torch.isfinite(loss), "Spatial-K physics loss is NaN/Inf"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Replication padding -- no NaN at boundaries
# ──────────────────────────────────────────────────────────────────────────────
def test_replication_padding_no_nan():
    from src.physics import PhyLoss
    phy = PhyLoss(K=500.0)
    T = torch.rand(1, 1, H, W)
    dx = phy._apply_kernel(T, phy.weight_dx)
    dy = phy._apply_kernel(T, phy.weight_dy)
    lap = phy._apply_kernel(T, phy.weight_laplace)
    for arr, name in [(dx, "dTdx"), (dy, "dTdy"), (lap, "laplacian")]:
        assert torch.isfinite(arr).all(), f"NaN/Inf in {name}"
        # With replication padding the border pixels should not be zero
        # (unless the input itself is constant), unlike zero-padding.
        assert arr.shape == (1, 1, H, W), f"Wrong shape for {name}: {arr.shape}"


# ──────────────────────────────────────────────────────────────────────────────
# 7. Training loop -- 2-batch smoke (gradient flow + λ_phy update)
# ──────────────────────────────────────────────────────────────────────────────
def test_training_loop():
    from src.models import SST_ConvLSTM
    from src.physics import PhyLoss
    import torch.nn as nn
    import torch.optim as optim

    model = SST_ConvLSTM(in_channels=1, hidden_dim=[8, 16], out_steps=HORIZON)
    phy   = PhyLoss(K=500.0, learn_K=True)
    opt   = optim.AdamW(list(model.parameters()) + list(phy.parameters()), lr=1e-3)
    crit  = nn.MSELoss()

    lambda_phy = 1.0
    alpha      = 0.9

    for _ in range(2):  # two mini-batches
        x   = torch.rand(BATCH, LOOKBACK, 1, H, W)
        y   = torch.rand(BATCH, HORIZON,  1, H, W)
        u, v = _dummy_uv()
        opt.zero_grad()

        preds     = model(x)
        loss_data = crit(preds, y)
        loss_phy  = phy(preds, u, v)

        target_layer = model.final_conv.weight
        g_d = torch.autograd.grad(loss_data, target_layer, retain_graph=True)[0]
        g_p = torch.autograd.grad(loss_phy,  target_layer, retain_graph=True)[0]

        lambda_hat = torch.max(torch.abs(g_d)) / (torch.mean(torch.abs(g_p)) + 1e-8)
        if torch.isfinite(lambda_hat):
            lambda_phy = alpha * lambda_phy + (1.0 - alpha) * lambda_hat.item()

        total = loss_data + lambda_phy * loss_phy
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    assert lambda_phy > 0, "lambda_phy should be positive"
    assert torch.isfinite(torch.tensor(lambda_phy)), "lambda_phy is NaN"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Metrics -- RMSE, ACC, NSE, DHW, bleaching alert
# ──────────────────────────────────────────────────────────────────────────────
def test_metrics():
    from src.metrics import (
        rmse, anomaly_correlation, nash_sutcliffe,
        compute_dhw_from_sst, bleaching_alert_agreement,
    )
    p = torch.rand(10, HORIZON, 1, H, W) * 5 + 25   # realistic SST range
    t = p + torch.rand_like(p) * 0.5                 # small noise

    r   = rmse(p, t)
    acc = anomaly_correlation(p, t)
    nse = nash_sutcliffe(p, t)
    assert 0 <= r <= 5,       f"Unreasonable RMSE: {r}"
    assert -1 <= acc <= 1,    f"ACC out of range: {acc}"
    assert nse <= 1.0,        f"NSE cannot exceed 1: {nse}"

    # DHW needs a continuous (T, H, W) array
    sst_seq = np.random.rand(100, H, W).astype(np.float32) * 5 + 25
    clim    = np.full((H, W), 27.0, dtype=np.float32)
    dhw     = compute_dhw_from_sst(sst_seq, clim)
    assert dhw.shape == sst_seq.shape, f"DHW shape mismatch: {dhw.shape}"
    assert dhw.min() >= 0, "DHW must be non-negative"

    alert = bleaching_alert_agreement(dhw, dhw + 0.1 * np.random.rand(*dhw.shape), 4.0)
    assert "f1" in alert, "Missing F1 key in alert dict"


# ──────────────────────────────────────────────────────────────────────────────
# 9. Early-warning residual series
# ──────────────────────────────────────────────────────────────────────────────
def test_early_warning_residual():
    from src.metrics import physics_residual_series
    T = 30
    preds = np.random.rand(T, H, W).astype(np.float64) * 5 + 25
    u     = np.random.rand(T, H, W).astype(np.float64) * 0.3
    v     = np.random.rand(T, H, W).astype(np.float64) * 0.3
    res   = physics_residual_series(preds, u, v, dx=5500.0, dy=5500.0, dt=86400.0)
    assert res.shape == (T - 1,), f"Bad residual shape: {res.shape}"
    assert np.isfinite(res).all(), "NaN in residual series"


# ──────────────────────────────────────────────────────────────────────────────
# 10. reconstruct_continuous
# ──────────────────────────────────────────────────────────────────────────────
def test_reconstruct_continuous():
    from src.metrics import reconstruct_continuous
    N = 20   # sliding windows
    H_out = HORIZON
    windows = torch.rand(N, H_out, 1, H, W)
    cont = reconstruct_continuous(windows)
    expected_T = N + H_out - 1
    assert cont.shape == (expected_T, 1, H, W), f"Bad shape: {cont.shape}"
    assert torch.isfinite(cont).all(), "NaN in continuous reconstruction"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("\n+----------------------------------------------------------+")
    print("|        REEFS_ML -- End-to-End Smoke Test                |")
    print("+----------------------------------------------------------+\n")

    tests = [
        ("1. Config validation",             test_config),
        ("2. Model forward pass",            test_model_forward),
        ("3. Physics loss (fixed K)",        test_physics_fixed_K),
        ("4. Physics loss (learned scalar K)", test_physics_learned_scalar_K),
        ("5. Physics loss (spatial K map)",  test_physics_spatial_K),
        ("6. Replication-padding (no NaN)",  test_replication_padding_no_nan),
        ("7. Training loop (2 batches)",     test_training_loop),
        ("8. Metrics (RMSE/ACC/NSE/DHW)",    test_metrics),
        ("9. Early-warning residual series", test_early_warning_residual),
        ("10. reconstruct_continuous",       test_reconstruct_continuous),
    ]
    for name, fn in tests:
        run_test(name, fn)

    total  = len(results)
    passed = sum(results.values())
    failed = total - passed

    print(f"\n{'=' * 58}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 58}\n")

    if failed:
        print("  [FAIL] Smoke test FAILED -- fix the errors above before training.\n")
        sys.exit(1)
    else:
        print("  [OK] All tests passed -- safe to run `make train`.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
