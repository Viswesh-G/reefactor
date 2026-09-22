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
    A custom loss function that penalizes the neural network if its temperature 
    predictions violate the physical laws of ocean heat movement.
    """
    def __init__(self, K=1e-3, dx=1.0, dy=1.0, dt=1.0):
        super().__init__()
        # K is the thermal diffusivity (how fast heat naturally spreads)
        self.K = K
        # dx, dy, dt are the real-world physical sizes of our pixels and time steps
        self.dt = dt
        self.dx = dx
        self.dy = dy
        
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
                                  
        # 3. Second Derivative (Laplacian) - Used to calculate diffusion (heat spreading)
        # It checks if the center pixel (-4) is hotter/colder than its 4 neighbors (+1)
        kernel_laplace = torch.tensor([[ 0,  1,  0],
                                       [ 1, -4,  1],
                                       [ 0,  1,  0]], dtype=torch.float32) / (dx * dy)
                                       
        # We reshape these filters so PyTorch's F.conv2d function can use them.
        # Shape needs to be: (out_channels, in_channels, kernel_height, kernel_width)
        # We use register_buffer so these tensors are moved to the GPU automatically,
        # but the optimizer knows NOT to train/change them.
        self.register_buffer('weight_dx', kernel_dx.view(1, 1, 3, 3))
        self.register_buffer('weight_dy', kernel_dy.view(1, 1, 3, 3))
        self.register_buffer('weight_laplace', kernel_laplace.view(1, 1, 3, 3))

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
            # THE ADVECTION-DIFFUSION EQUATION
            # In a perfect physical world, this equation should perfectly equal ZERO.
            # If it doesn't equal zero, our network has made a physically impossible prediction!
            # residual = (Change over time) + (Heat blown by currents) - (Heat spreading out)
            # ==============================================================
            residual = dTdt + (u * dTdx) + (v * dTdy) - (self.K * laplacian)
            
            # We want to penalize any deviation from zero, so we square it (Mean Squared Error)
            physics_loss += torch.mean(residual ** 2)
            
        # Average the penalty across all time steps
        return physics_loss / (t - 1)
