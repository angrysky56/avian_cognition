"""
Core BitNet Quantization Components and Utilities

This module provides fundamental building blocks for implementing BitNet-style 
1-bit quantization in PyTorch models, including the BitLinear layer, a 
quantized GRU cell, the NALU layer (often useful in numerically focused models),
and a utility function for converting standard Linear layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BitLinear(nn.Module):
    """
    Custom Linear layer implementing 1-bit weight quantization (BitNet style).
    
    During the forward pass, weights are binarized to +1/-1. A scaling factor,
    calculated from the full-precision weights, is applied to maintain output
    magnitude. Full-precision weights are retained internally for gradient 
    updates during training.

    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        weight (nn.Parameter): Trainable full-precision weights. 
                               Shape: [out_features, in_features].
        scale (torch.Tensor): Non-trainable buffer holding the scaling factor, 
                              recalculated each forward pass based on `weight`.
        bias (nn.Parameter, optional): Trainable bias term. Shape: [out_features].
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True, 
                 device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None) -> None:
        """
        Initialize the BitLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If True, adds a learnable bias to the output. Default: True.
            device (Optional[torch.device]): The desired device of the parameters. 
                                             If None, uses default device.
            dtype (Optional[torch.dtype]): The desired floating point type of the parameters. 
                                           If None, uses default dtype.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Full-precision weights stored as Parameter for training
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        # Scaling factor (calculated dynamically in forward pass) stored as buffer
        # Initialize scale to 1.0
        self.register_buffer('scale', torch.ones(1, device=device, dtype=dtype if dtype else torch.float32)) 
        
        # Optional bias term
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            # Register bias as None if not used
            self.register_parameter('bias', None) 
            
        # Initialize weights and bias using standard PyTorch practices
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize weights and bias using Kaiming uniform initialization."""
        # Initialize weights using Kaiming uniform for potentially better training stability
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 
        if self.bias is not None:
            # Calculate fan-in for bias initialization based on Kaiming uniform logic
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                 bound = 1 / math.sqrt(fan_in)
                 nn.init.uniform_(self.bias, -bound, bound)
            else:
                 # Handle case where fan_in is zero (e.g., in_features=0)
                 nn.init.zeros_(self.bias)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass using binarized weights and scaling.
        
        Args:
            input (torch.Tensor): Input tensor. 
                                  Shape: [*, in_features], where * means any number 
                                  of leading dimensions.
            
        Returns:
            torch.Tensor: Output tensor. Shape: [*, out_features].
        """
        # Ensure layer parameters are on the same device as the input
        if input.device != self.weight.device:
            input = input.to(self.weight.device)
            # Note: Parameters should ideally be moved to the correct device 
            # *before* the forward pass begins (e.g., model.to(device)). 
            # This check is a fallback.

        # --- BitNet Quantization Step ---
        # 1. Binarize weights: Use sign() to get +1, -1, or 0 (for zero weights).
        #    Using sign() is common in BitNet implementations.
        binary_weight = torch.sign(self.weight) 
        # Alternative: Binarize around 0 (STE - Straight-Through Estimator might be needed for training stability)
        # binary_weight = ((self.weight > 0).float() - 0.5) * 2 
        
        # 2. Calculate scaling factor (beta in some papers): Average absolute value of full-precision weights.
        #    This helps maintain the output magnitude similar to a full-precision layer.
        #    Recalculated each forward pass based on the *current* full-precision weights.
        #    Using detach() as scale calculation shouldn't contribute to gradients w.r.t scale itself.
        current_scale = self.weight.abs().mean().detach()
        # Update buffer if needed (e.g., for logging, though not strictly necessary if only used here)
        # self.scale.fill_(current_scale) 
        
        # 3. Apply scaling to binarized weights
        scaled_binary_weight = binary_weight * current_scale

        # 4. Perform linear operation using the scaled binary weights
        output = F.linear(input, scaled_binary_weight, self.bias)
        
        return output
    
    def extra_repr(self) -> str:
        """Return string representation of the layer's configuration."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class BitGRUCell(nn.Module):
    """
    Gated Recurrent Unit (GRU) cell implemented using BitLinear layers.
    
    Provides a recurrent cell suitable for sequence modeling tasks where memory 
    efficiency via 1-bit weights is desired.

    Attributes:
        input_size (int): The number of expected features in the input `x`.
        hidden_size (int): The number of features in the hidden state `h`.
        reset_gate (BitLinear): BitLinear layer for the reset gate.
        update_gate (BitLinear): BitLinear layer for the update gate.
        new_gate (BitLinear): BitLinear layer for the candidate hidden state.
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None) -> None:
        """
        Initialize the BitGRUCell.

        Args:
            input_size (int): Input feature dimension.
            hidden_size (int): Hidden state feature dimension.
            device (Optional[torch.device]): Device for parameters.
            dtype (Optional[torch.dtype]): Data type for parameters.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Linear layers for gates, input dimension is input_size + hidden_size
        gate_input_dim = input_size + hidden_size
        
        # Reset gate components (computes r_t)
        self.reset_gate = BitLinear(gate_input_dim, hidden_size, **factory_kwargs)
        
        # Update gate components (computes z_t)
        self.update_gate = BitLinear(gate_input_dim, hidden_size, **factory_kwargs)
        
        # Candidate hidden state components (computes n_t)
        self.new_gate = BitLinear(gate_input_dim, hidden_size, **factory_kwargs)
        
    def forward(self, 
                input: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform a single GRU cell time step using BitLinear layers.
        
        Args:
            input (torch.Tensor): Input tensor for the current time step.
                                  Shape: [batch_size, input_size].
            hidden (Optional[torch.Tensor]): Hidden state from the previous time step.
                                             Shape: [batch_size, hidden_size]. 
                                             Defaults to zeros if None.
            
        Returns:
            torch.Tensor: The updated hidden state for the current time step.
                          Shape: [batch_size, hidden_size].
        """
        batch_size = input.size(0)
        device = input.device # Use input's device

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=input.dtype)
        
        # Ensure hidden state is on the correct device
        hidden = hidden.to(device)

        # --- GRU Computations ---
        # Concatenate input and previous hidden state for gate inputs
        combined_input = torch.cat([input, hidden], dim=1) # Shape: [batch_size, input_size + hidden_size]
        
        # 1. Calculate reset gate (r_t)
        # r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
        r_t = torch.sigmoid(self.reset_gate(combined_input)) # Shape: [batch_size, hidden_size]
        
        # 2. Calculate update gate (z_t)
        # z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
        z_t = torch.sigmoid(self.update_gate(combined_input)) # Shape: [batch_size, hidden_size]
        
        # 3. Calculate candidate hidden state (n_t)
        # n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))
        # Combine input with the *reset* hidden state
        combined_input_reset = torch.cat([input, r_t * hidden], dim=1) # Shape: [batch_size, input_size + hidden_size]
        n_t = torch.tanh(self.new_gate(combined_input_reset)) # Shape: [batch_size, hidden_size]
        
        # 4. Calculate the new hidden state (h_t)
        # h_t = (1 - z_t) * n_t + z_t * h_{t-1}  <- Standard GRU update equation correction
        # The standard update is: h_t = (1 - z_t) * h_{t-1} + z_t * n_t
        # Let's use the standard one:
        h_new = (1 - z_t) * hidden + z_t * n_t # Shape: [batch_size, hidden_size]
        
        return h_new


class NALULayer(nn.Module):
    """
    Neural Arithmetic Logic Unit (NALU) layer.
    
    Designed to learn simple arithmetic functions (+, -, *, /) more effectively 
    and with better generalization than standard MLPs. It achieves this by using
    separate pathways for addition/subtraction and multiplication/division,
    combined via a learned gate. Multiplication is performed in log-space for
    numerical stability and to represent it as addition.

    Reference: https://arxiv.org/abs/1808.00508

    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        eps (float): Epsilon value added for numerical stability, especially for log.
        G (nn.Parameter): Weight matrix for the gate controlling add/multiply paths.
        W (nn.Parameter): Weight matrix for the additive path (learns linear transform).
        M (nn.Parameter): Weight matrix for the multiplicative path (learns linear transform in log space).
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 eps: float = 1e-7, # Epsilon for numerical stability, esp. log(0)
                 device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None) -> None:
        """
        Initialize the NALULayer.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            eps (float): Small epsilon value for numerical stability (log/sqrt). Default: 1e-7.
            device (Optional[torch.device]): Device for parameters.
            dtype (Optional[torch.dtype]): Data type for parameters.
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        # Epsilon should be positive
        self.eps = max(eps, torch.finfo(torch.float32).tiny) # Ensure eps is at least smallest representable positive float
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Gate parameters (controls mixing between addition and multiplication)
        self.G = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        # Parameters for the additive pathway (W_hat in paper)
        self.W = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        # Parameters for the multiplicative pathway (M_hat in paper, operates in log space)
        self.M = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        # NALU typically does not use a bias term
        self.register_parameter('bias', None) 
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize parameters, typically using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.G, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the NALU forward pass.
        
        Args:
            x (torch.Tensor): Input tensor. Shape: [batch_size, in_features].
            
        Returns:
            torch.Tensor: Output tensor. Shape: [batch_size, out_features].
        """
        # Ensure parameters are on the same device as input
        if x.device != self.G.device:
             x = x.to(self.G.device)
             # Again, ideally, model is moved to device beforehand.

        # --- NALU Computations ---
        # 1. Calculate the gate (g) - determines mix between add/multiply paths
        # g = sigmoid(G @ x)
        g = torch.sigmoid(F.linear(x, self.G)) # Shape: [batch_size, out_features]
        
        # 2. Calculate the additive path result (a)
        # a = W @ x
        a = F.linear(x, self.W) # Shape: [batch_size, out_features]
        
        # 3. Calculate the multiplicative path result (m)
        # Multiplication is done in log space: m = exp(M @ log(|x| + eps))
        # Taking log of absolute value + epsilon for stability
        log_input = torch.log(torch.abs(x) + self.eps) # Shape: [batch_size, in_features]
        # Linear transformation in log space
        m_log = F.linear(log_input, self.M) # Shape: [batch_size, out_features]
        # Convert back from log space using exp
        m = torch.exp(m_log) # Shape: [batch_size, out_features]
        
        # 4. Combine the additive and multiplicative paths using the gate
        # y = g * a + (1 - g) * m
        y = g * a + (1 - g) * m # Shape: [batch_size, out_features]
        
        return y


def convert_linear_to_bit_linear(module: nn.Module, 
                                device: Optional[torch.device] = None) -> nn.Module:
    """
    Recursively traverses a PyTorch module and replaces all instances of 
    `torch.nn.Linear` with `BitLinear` layers in-place.
    
    Args:
        module (nn.Module): The PyTorch module (e.g., a model or a sub-module) 
                            to convert. The conversion happens in-place.
        device (Optional[torch.device]): If provided, the newly created BitLinear 
                                         layers will be moved to this device.
        
    Returns:
        nn.Module: The modified module with nn.Linear layers replaced by BitLinear.
    """
    # Iterate over direct children modules, using list() to allow modification during iteration
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            # Create a new BitLinear layer with the same dimensions and bias setting
            bit_linear_layer = BitLinear(
                in_features=child.in_features, 
                out_features=child.out_features, 
                bias=(child.bias is not None),
                # Initialize directly on the target device if specified
                device=device, 
                # Attempt to match dtype, default to float32 if original dtype is complex/unavailable
                dtype=child.weight.dtype if child.weight.dtype.is_floating_point else torch.float32 
            )
            
            # Copy the weights and bias from the original nn.Linear layer
            # Use torch.no_grad() to avoid tracking these operations for autograd
            with torch.no_grad():
                bit_linear_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    bit_linear_layer.bias.copy_(child.bias)
                    
            # Replace the original nn.Linear layer with the new BitLinear layer
            setattr(module, name, bit_linear_layer)
            # print(f"Replaced '{name}' (nn.Linear) with BitLinear.") # Optional: for debugging
        else:
            # If the child is not nn.Linear, recursively call this function on the child
            convert_linear_to_bit_linear(child, device=device)
            
    # Return the modified module (though modification is in-place)
    return module