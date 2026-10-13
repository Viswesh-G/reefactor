import torch
import torch.nn as nn
import torch.nn.functional as F

# The Physics Loss (PhyLoss) forces the neural network to obey physical rules
# (specifically the Advection-Diffusion Equation) instead of just fitting to data.
# - Advection: Heat moves with ocean currents (velocity u, v).
# - Diffusion: Heat spreads from hot to cold areas (controlled by K).
# We use Finite Difference Approximations (fixed convolution filters) to 
# calculate the spatial derivatives across the grid.

class PhyLoss(nn.Module):
    """
    Discrete surface advection-diffusion residual penalty.

    Units are SI: preds in degC (or z-scored degC — residual scales by 1/std,
    still a valid penalty), u/v in m/s, dx/dy in meters, dt in seconds,
    K in m^2/s (horizontal eddy diffusivity, ~1e2-1e3 at 5km grids).

    dT/dt + u*dT/dx + v*dT/dy - K*lap(T) = 0, with central differences:
      dT/dx ~ (T[i,j+1]-T[i,j-1]) / 2dx
      lap   ~ (T[i+1,j]-2T+T[i-1,j])/dx^2 + (T[i,j+1]-2T+T[i,j-1])/dy^2
    """
    def __init__(self, K=500.0, dx=5500.0, dy=5500.0, dt=86400.0, learn_K=False,
                 K_min=10.0, K_max=5000.0):
        super().__init__()
        # K is the horizontal eddy diffusivity in m^2/s
        self.dx = float(dx)
        self.dy = float(dy)
        self.dt = float(dt)
        self.learn_K = bool(learn_K)
        self.K_min, self.K_max = float(K_min), float(K_max)
        if learn_K:
            import math as _math
            # learn in log-space for positivity + wide range
            self.log_K = nn.Parameter(torch.tensor(_math.log(float(K))))
        else:
            self.register_buffer('K_const', torch.tensor(float(K)))
        
        # ---------------------------------------------------------
        # Defining our "Calculus Filters" (Finite Difference Kernels)
        # ---------------------------------------------------------
        
        # 1. First Derivative w.r.t X (How much does temp change left-to-right?)
        # We look at the right pixel (+1) and left pixel (-1) to find the slope.
        kernel_dx = torch.tensor([[ 0,  0,  0],
                                  [-1,  0,  1],
                                  [ 0,  0,  0]], dtype=torch.float32) / (2 * dx)
                                  
        # 2. First Derivative w.r.t Y (How much does temp change top-to-bottom?)
        kernel_dy = torch.tensor([[ 0, -1,  0],
                                  [ 0,  0,  0],
                                  [ 0,  1,  0]], dtype=torch.float32) / (2 * dy)
                                  
        # 3. Laplacian (anisotropic 5-point, correct 1/dx^2 + 1/dy^2 scaling)
        # lap = (T[i+1,j]-2T+T[i-1,j])/dx^2 + (T[i,j+1]-2T+T[i,j-1])/dy^2
        idx2, idy2 = 1.0 / (dx * dx), 1.0 / (dy * dy)
        kernel_laplace = torch.tensor([[0.0, idy2, 0.0],
                                       [idx2, -2.0 * idx2 - 2.0 * idy2, idx2],
                                       [0.0, idy2, 0.0]], dtype=torch.float32)
                                       
        # We reshape these filters so PyTorch's F.conv2d function can use them.
        # Shape needs to be: (out_channels, in_channels, kernel_height, kernel_width)
        # We use register_buffer so these tensors are moved to the GPU automatically,
        # but the optimizer knows NOT to train/change them.
        self.register_buffer('weight_dx', kernel_dx.view(1, 1, 3, 3))
        self.register_buffer('weight_dy', kernel_dy.view(1, 1, 3, 3))
        self.register_buffer('weight_laplace', kernel_laplace.view(1, 1, 3, 3))

    def K_value(self) -> float:
        """Current effective diffusivity K in m^2/s (for logging / reporting)."""
        if self.learn_K:
            with torch.no_grad():
                return float(torch.clamp(torch.exp(self.log_K), self.K_min, self.K_max))
        return float(self.K_const)

    def forward(self, preds, u_vel, v_vel):
        """
        Calculates the physics penalty for a sequence of predictions.
        
        Args:
            preds: The predicted temperature grids. Shape: (Batch, Time, 1, Height, Width)
            u_vel: The ocean current moving East-West. Shape: (Batch, Time, 1, Height, Width)
            v_vel: The ocean current moving North-South. Shape: (Batch, Time, 1, Height, Width)
        """
        # Get dimensions
        b, t, c, h, w = preds.size()

        # Effective diffusivity is constant across the sequence — compute once.
        K_eff = (torch.clamp(torch.exp(self.log_K), self.K_min, self.K_max)
                 if self.learn_K else self.K_const)

        # Start with zero physics penalty
        physics_loss = 0.0

        # We calculate the physical rule between each consecutive time step
        for step in range(1, t):
            T_prev = preds[:, step-1] # Yesterday's temperature
            T_curr = preds[:, step]   # Today's temperature
            
            # 1. Temporal Derivative (How much did temp change over time?)
            dTdt = (T_curr - T_prev) / self.dt
            
            # 2. Spatial Derivatives (Applying our calculus filters)
            # padding=1 ensures our output image stays the same size as the input
            dTdx = F.conv2d(T_prev, self.weight_dx, padding=1)
            dTdy = F.conv2d(T_prev, self.weight_dy, padding=1)
            laplacian = F.conv2d(T_prev, self.weight_laplace, padding=1)
            
            # Get the current velocities
            u = u_vel[:, step-1]
            v = v_vel[:, step-1]

            # ==============================================================
            # THE ADVECTION-DIFFUSION EQUATION (SI units)
            # In a perfect physical world, this equation should perfectly equal ZERO.
            # If it doesn't equal zero, our network has made a physically impossible prediction!
            # residual = (Change over time) + (Heat blown by currents) - (Heat spreading out)
            # ==============================================================
            residual = dTdt + (u * dTdx) + (v * dTdy) - (K_eff * laplacian)
            
            # We want to penalize any deviation from zero, so we square it (Mean Squared Error)
            physics_loss += torch.mean(residual ** 2)
            
        # Average the penalty across all time steps
        return physics_loss / (t - 1)
