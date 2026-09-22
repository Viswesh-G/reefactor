import torch
import torch.nn as nn

# A ConvLSTM is used here instead of a standard LSTM because our data (SST) is 2D.
# If we flatten the 2D grid into a 1D vector, we lose spatial relationships.
# By using 2D convolutions inside the LSTM cell, the network can remember 
# patterns across both time and 2D space.

class ConvLSTMCell(nn.Module):
    """
    A single cell of a Convolutional LSTM. 
    This acts as the fundamental building block that steps through time one frame at a time.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # We need padding so that the output image stays the same size as the input image.
        # For a kernel of size 3, padding of 1 keeps the dimensions equal.
        padding = kernel_size // 2
        
        # In a standard LSTM, we calculate 4 gates: Input, Forget, Output, and Cell State.
        # Here, we do one big convolution that outputs 4 times the hidden channels, 
        # and then we will split it into the 4 gates.
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, state):
        """
        Forward pass for a single time step.
        
        Args:
            x (Tensor): The input image at this time step. Shape: (Batch, Channels, Height, Width)
            state (tuple): The previous hidden state and cell state. 
                           Each has shape: (Batch, Hidden_Channels, Height, Width)
        """
        # Unpack the previous states
        h, c = state
        
        # Concatenate the current input and the previous hidden state along the channel dimension.
        # This combines "what we are seeing now" (x) with "what we remember" (h).
        combined = torch.cat([x, h], dim=1)
        
        # Apply the convolution to the combined input
        gates = self.conv(combined)
        
        # Split the output into 4 chunks (one for each gate) along the channel dimension (dim=1)
        ingate, forgetgate, outgate, cellgate = torch.split(gates, self.hidden_channels, dim=1)
        
        # Apply activation functions:
        # Sigmoid gives values between 0 and 1 (acting like a valve/gate)
        # Tanh gives values between -1 and 1 (acting like the actual information)
        i = torch.sigmoid(ingate)      # What new information should we keep?
        f = torch.sigmoid(forgetgate)  # What old information should we forget?
        o = torch.sigmoid(outgate)     # What information should we output?
        g = torch.tanh(cellgate)       # New candidate memories
        
        # Update the cell state (long term memory)
        c_next = f * c + i * g
        
        # Update the hidden state (short term memory / current output)
        h_next = o * torch.tanh(c_next)
        
        # Return both so they can be passed to the next time step
        return h_next, c_next


class SST_ConvLSTM(nn.Module):
    """
    The main Network for Sea Surface Temperature (SST) forecasting.
    It chains multiple ConvLSTMCells together to process a sequence of past days
    and predict a sequence of future days.
    """
    def __init__(self, in_channels=1, hidden_dim=[32, 64], out_steps=7):
        super().__init__()
        
        # How many future days we want to predict
        self.out_steps = out_steps
        
        # We use a 2-layer ConvLSTM (like having 2 stacked neural networks for depth)
        self.cell1 = ConvLSTMCell(in_channels, hidden_dim[0])
        self.cell2 = ConvLSTMCell(hidden_dim[0], hidden_dim[1])
        
        # The final layer converts the thick hidden state (e.g., 64 channels) 
        # back down to our target: 1 channel (the single temperature value per pixel)
        self.final_conv = nn.Conv2d(hidden_dim[1], 1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for the whole sequence.
        
        Args:
            x (Tensor): The sequence of past SST maps. 
                        Shape required: (Batch, Time, Channels, Height, Width)
        """
        # Get dimensions from the input
        b, t_in, _, h, w = x.size()
        
        # Create starting memory states filled with zeros for both layers.
        # We have to explicitly put them on the same device (CPU/GPU) as the input.
        h1 = torch.zeros(b, self.cell1.hidden_channels, h, w, device=x.device)
        c1 = torch.zeros(b, self.cell1.hidden_channels, h, w, device=x.device)
        h2 = torch.zeros(b, self.cell2.hidden_channels, h, w, device=x.device)
        c2 = torch.zeros(b, self.cell2.hidden_channels, h, w, device=x.device)
        
        # ==========================================
        # PHASE 1: ENCODING (Reading the past)
        # ==========================================
        # We loop through each time step (day) in our input sequence.
        # We just update the memory, we don't care about predicting yet.
        for t in range(t_in):
            h1, c1 = self.cell1(x[:, t], (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))
            
        # ==========================================
        # PHASE 2: DECODING (Predicting the future)
        # ==========================================
        outputs = []
        
        # We feed the last observed day as the first input to the prediction phase
        current_input = x[:, -1, ...]  
        
        # Loop for however many days into the future we want to predict
        for _ in range(self.out_steps):
            # Pass through our LSTM cells
            h1, c1 = self.cell1(current_input, (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))
            
            # Predict the actual temperature image for this step
            pred = self.final_conv(h2)
            
            # Save the prediction to our list
            outputs.append(pred)
            
            # Use the prediction to drive the next prediction step
            current_input = pred  

        # Stack the list of predictions along the Time dimension (dim=1)
        # Output Shape will be: (Batch, Time_Out, 1, Height, Width)
        return torch.stack(outputs, dim=1)
