import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# The Physics Loss (PhyLoss) forces the neural network to obey physical rules
# (specifically the Advection-Diffusion Equation) instead of just fitting to data.
# - Advection: Heat moves with ocean currents (velocity u, v).
# - Diffusion: Heat spreads from hot to cold areas (controlled by K).
# We use Finite Difference Approximations (fixed convolution filters) to
# calculate the spatial derivatives across the grid.
#
# Boundary handling: replication padding is used instead of zero-padding.
# Zero-padding sets boundary edges to 0, creating artificial temperature
# gradients at the grid border. Replication padding copies the outermost
# pixel value outward, giving near-zero gradients at edges — much more
# physically realistic for an open-ocean bounding box.


class PhyLoss(nn.Module):
    """
    Discrete surface advection-diffusion residual penalty.

    SI units: preds in degC (or z-scored degC — residual scales by 1/σ,
    still a valid penalty), u/v in m/s, dx/dy in meters, dt in seconds,
    K in m^2/s (horizontal eddy diffusivity, ~1e2-1e3 at 5-km grids).

    Governing equation:
        dT/dt + u·∂T/∂x + v·∂T/∂y − K·∇²T = 0

    Spatial derivatives use 5-point central differences:
        ∂T/∂x  ≈ (T[i,j+1]−T[i,j−1]) / 2dx
        ∂T/∂y  ≈ (T[i+1,j]−T[i−1,j]) / 2dy
        ∇²T    ≈ (T[i+1,j]+T[i−1,j]−2T)/dx² + (T[i,j+1]+T[i,j−1]−2T)/dy²

    NOVEL DIFFERENTIATOR — spatially-varying K:
        When learn_K=True and K_spatial=True, we learn a per-pixel (H×W)
        eddy diffusivity map instead of a single scalar. This captures the
        GBR's spatially heterogeneous mixing: shelf-break jets, reef passes,
        and eddies all mix heat at different rates. The spatial K map is a
        physically interpretable result we can compare to observational
        estimates (e.g., Oke & Middleton 2001 for the GBR, ~50–2000 m²/s).
    """
    def __init__(
        self,
        K: float = 500.0,
        dx: float = 5500.0,
        dy: float = 5500.0,
        dt: float = 86400.0,
        learn_K: bool = False,
        K_min: float = 10.0,
        K_max: float = 5000.0,
        K_spatial: bool = False,
        grid_h: int = None,
        grid_w: int = None,
    ):
        """
        Args:
            K:         Initial (or fixed) eddy diffusivity in m²/s.
            dx, dy:    Grid spacing in meters (≈5500 m at 0.05° resolution).
            dt:        Time step in seconds (86400 s = 1 day).
            learn_K:   If True, K is a trainable parameter (scalar).
            K_min/max: Clamp bounds for learned K (m²/s).
            K_spatial: If True, learn a per-pixel (H×W) K map instead of
                       a scalar. Requires learn_K=True and grid_h, grid_w.
            grid_h/w:  Spatial dimensions of the GBR canvas (needed for K_spatial).
        """
        super().__init__()
        self.dx = float(dx)
        self.dy = float(dy)
        self.dt = float(dt)
        self.learn_K  = bool(learn_K)
        self.K_spatial = bool(K_spatial) and learn_K
        self.K_min, self.K_max = float(K_min), float(K_max)

        # --- Learnable diffusivity (log-space for positivity + wide dynamic range) ---
        if self.K_spatial:
            # Per-pixel log-K map: shape (1, 1, H, W) — broadcast over batch & time.
            # Initialised uniformly at log(K) so it starts at the scalar baseline.
            if grid_h is None or grid_w is None:
                raise ValueError("K_spatial=True requires grid_h and grid_w.")
            init_val = math.log(float(K)) * torch.ones(1, 1, grid_h, grid_w)
            self.log_K_map = nn.Parameter(init_val)
        elif learn_K:
            # Scalar learned K (original behaviour).
            self.log_K = nn.Parameter(torch.tensor(math.log(float(K))))
        else:
            # Fixed constant — not trained.
            self.register_buffer("K_const", torch.tensor(float(K)))

        # ---------------------------------------------------------
        # Finite-difference derivative kernels (fixed, non-trainable)
        # ---------------------------------------------------------
        # 1. ∂/∂x  — central difference along columns (longitude)
        kernel_dx = torch.tensor(
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32
        ) / (2.0 * dx)

        # 2. ∂/∂y  — central difference along rows (latitude)
        kernel_dy = torch.tensor(
            [[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32
        ) / (2.0 * dy)

        # 3. Laplacian — anisotropic 5-point stencil (handles dx ≠ dy)
        #    ∇²T ≈ (T_up + T_down − 2T)/dy² + (T_left + T_right − 2T)/dx²
        idx2 = 1.0 / (dx * dx)
        idy2 = 1.0 / (dy * dy)
        kernel_lap = torch.tensor(
            [[0.0, idy2, 0.0],
             [idx2, -2.0 * (idx2 + idy2), idx2],
             [0.0, idy2, 0.0]],
            dtype=torch.float32,
        )

        # register_buffer → moves to GPU automatically; excluded from optimizer
        self.register_buffer("weight_dx",      kernel_dx.view(1, 1, 3, 3))
        self.register_buffer("weight_dy",      kernel_dy.view(1, 1, 3, 3))
        self.register_buffer("weight_laplace", kernel_lap.view(1, 1, 3, 3))

    # ------------------------------------------------------------------
    def _K_effective(self) -> torch.Tensor:
        """Return the effective K for the current forward pass.

        Returns a scalar tensor (broadcast-compatible with any residual shape)
        when learn_K is scalar, or a (1,1,H,W) spatial map when K_spatial=True.
        In both cases the returned value is clamped to [K_min, K_max] m²/s.
        """
        if self.K_spatial:
            return torch.clamp(torch.exp(self.log_K_map), self.K_min, self.K_max)
        if self.learn_K:
            return torch.clamp(torch.exp(self.log_K), self.K_min, self.K_max)
        return self.K_const

    def K_value(self) -> float:
        """Scalar summary of the current K (m²/s) — used for logging.

        For the spatial map, returns the spatial mean (a meaningful single
        number for the training log without requiring array printing).
        """
        with torch.no_grad():
            k = self._K_effective()
            return float(k.mean())

    # ------------------------------------------------------------------
    def _apply_kernel(self, T: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply a 3×3 finite-difference kernel with REPLICATION padding.

        Replication padding copies the edge pixel outward rather than padding
        with zeros, eliminating the artificial temperature gradient that
        zero-padding would introduce at the bounding-box edges.

        Args:
            T:      Input grid, shape (B, 1, H, W).
            weight: Kernel weights,  shape (1, 1, 3, 3).

        Returns:
            Derivative field, shape (B, 1, H, W).
        """
        # F.pad with mode='replicate' adds one border pixel on each side.
        T_padded = F.pad(T, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(T_padded, weight, padding=0)

    # ------------------------------------------------------------------
    def forward(
        self,
        preds:  torch.Tensor,
        u_vel:  torch.Tensor,
        v_vel:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the mean squared advection-diffusion residual over all
        predicted time steps.

        Args:
            preds: Predicted SST sequence. Shape: (B, T, 1, H, W)
            u_vel: East-West ocean current.  Shape: (B, T, 1, H, W)
            v_vel: North-South ocean current. Shape: (B, T, 1, H, W)

        Returns:
            Scalar physics penalty (mean over time steps and spatial pixels).
        """
        b, t, c, h, w = preds.size()
        K_eff = self._K_effective()   # scalar or (1,1,H,W)
        physics_loss = torch.zeros(1, device=preds.device)

        for step in range(1, t):
            T_prev = preds[:, step - 1]   # (B, 1, H, W)  yesterday
            T_curr = preds[:, step]       # (B, 1, H, W)  today

            # Temporal derivative  ∂T/∂t  ≈  (T_curr − T_prev) / Δt
            dTdt = (T_curr - T_prev) / self.dt

            # Spatial derivatives via replication-padded central differences
            dTdx     = self._apply_kernel(T_prev, self.weight_dx)       # ∂T/∂x
            dTdy     = self._apply_kernel(T_prev, self.weight_dy)       # ∂T/∂y
            laplacian = self._apply_kernel(T_prev, self.weight_laplace) # ∇²T

            # Velocity fields for the transition  t−1 → t
            u = u_vel[:, step - 1]   # (B, 1, H, W)
            v = v_vel[:, step - 1]   # (B, 1, H, W)

            # Advection-diffusion residual:  ∂T/∂t + u·∂T/∂x + v·∂T/∂y − K·∇²T
            # In a perfectly physics-consistent forecast this equals 0.
            residual = dTdt + u * dTdx + v * dTdy - K_eff * laplacian

            # Accumulate the mean-squared residual (L2 penalty)
            physics_loss = physics_loss + torch.mean(residual ** 2)

        # Average over forecast steps (so length of horizon doesn't affect scale)
        return physics_loss / (t - 1)
