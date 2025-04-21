"""
Enhanced BitNet Quantization Components

This module provides implementations for BitNet-style quantization in PyTorch models,
supporting both original 1-bit quantization and the newer b1.58 ternary quantization,
along with utility functions for model conversion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Literal

class BitLinear(nn.Module):
    """
    Custom Linear layer implementing BitNet-style quantization.
    
    Supports both 1-bit (binary) and 1.58-bit (ternary) quantization modes.
    During the forward pass, weights are quantized to either {-1, +1} (binary)
    or {-1, 0, +1} (ternary) values. A scaling factor is applied to maintain
    output magnitude. Full-precision weights are retained for gradient updates.

    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        weight (nn.Parameter): Trainable full-precision weights.
        scale (torch.Tensor): Scaling factor for quantized weights.
        bias (nn.Parameter, optional): Trainable bias term.
        quant_mode (str): Quantization mode ('binary' or 'ternary').
        sparsity (float): Target sparsity for ternary mode (percent of weights as zero).
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True, 
                 quant_mode: Literal['binary', 'ternary'] = 'ternary',
                 sparsity: float = 0.5,
                 device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None) -> None:
        """
        Initialize the BitLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If True, adds a learnable bias to the output. Default: True.
            quant_mode (str): Quantization mode ('binary' for 1-bit, 'ternary' for 1.58-bit).
            sparsity (float): Target sparsity ratio for ternary mode (0.0-1.0, default 0.5).
            device (Optional[torch.device]): The desired device of the parameters.
            dtype (Optional[torch.dtype]): The desired floating point type of the parameters.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_mode = quant_mode
        self.sparsity = max(0.0, min(1.0, sparsity))  # Clamp to [0, 1]
        
        # Full-precision weights stored for training
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        # Scaling factor (calculated during forward pass)
        self.register_buffer('scale', torch.ones(1, device=device, dtype=dtype if dtype else torch.float32))
        
        # Optional bias term
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights and bias
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize weights and bias using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                nn.init.zeros_(self.bias)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass using quantized weights and scaling.
        
        Args:
            input (torch.Tensor): Input tensor with shape [*, in_features].
            
        Returns:
            torch.Tensor: Output tensor with shape [*, out_features].
        """
        # Ensure layer parameters are on the same device as the input
        if input.device != self.weight.device:
            input = input.to(self.weight.device)

        # Perform weight quantization based on the selected mode
        if self.quant_mode == 'binary':
            # Binary quantization: weights are -1 or +1
            binary_weight = torch.sign(self.weight)
            # Calculate scaling factor: mean absolute value
            current_scale = self.weight.abs().mean().detach()
            # Apply scaling
            scaled_binary_weight = binary_weight * current_scale
            
        elif self.quant_mode == 'ternary':
            # Ternary quantization: weights are -1, 0, or +1
            # Calculate threshold for sparsification (zeroing out weights)
            if self.sparsity > 0:
                # Use absolute magnitude sorting to determine threshold
                # Get the k-th largest absolute value where k determined by sparsity
                abs_weights = self.weight.abs().flatten()
                k = int(abs_weights.numel() * self.sparsity)
                if k < abs_weights.numel():
                    threshold = abs_weights.kthvalue(k).values
                else:
                    threshold = 0.0
            else:
                threshold = 0.0
                
            # Create ternary weights
            # Values below threshold become 0, rest use sign
            ternary_weight = torch.zeros_like(self.weight)
            mask = self.weight.abs() > threshold
            ternary_weight[mask] = torch.sign(self.weight[mask])
            
            # Calculate scaling factor using absmean on non-zero weights
            if mask.sum() > 0:
                current_scale = self.weight[mask].abs().mean().detach()
            else:
                current_scale = torch.tensor(1.0, device=self.weight.device)
                
            # Apply scaling
            scaled_binary_weight = ternary_weight * current_scale
            
        else:
            raise ValueError(f"Unknown quantization mode: {self.quant_mode}")

        # Perform linear operation with scaled quantized weights
        output = F.linear(input, scaled_binary_weight, self.bias)
        
        return output
    
    def extra_repr(self) -> str:
        """Return string representation of the layer's configuration."""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, quant_mode={self.quant_mode}, '
                f'sparsity={self.sparsity:.2f}')


class BitGRUCell(nn.Module):
    """
    Gated Recurrent Unit (GRU) cell implemented using BitLinear layers.
    
    Provides a recurrent cell for sequence modeling with BitNet quantization for
    memory and computational efficiency.

    Attributes:
        input_size (int): Number of expected features in the input.
        hidden_size (int): Number of features in the hidden state.
        reset_gate (BitLinear): Gate controlling reset of previous hidden state.
        update_gate (BitLinear): Gate controlling update proportions.
        new_gate (BitLinear): Candidate hidden state computation.
        quant_mode (str): Quantization mode used in BitLinear layers.
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 quant_mode: Literal['binary', 'ternary'] = 'ternary',
                 sparsity: float = 0.5,
                 device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None) -> None:
        """
        Initialize the BitGRUCell.

        Args:
            input_size (int): Input feature dimension.
            hidden_size (int): Hidden state feature dimension.
            quant_mode (str): Quantization mode ('binary' or 'ternary').
            sparsity (float): Target sparsity for ternary mode.
            device (Optional[torch.device]): Device for parameters.
            dtype (Optional[torch.dtype]): Data type for parameters.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.quant_mode = quant_mode
        factory_kwargs = {'device': device, 'dtype': dtype, 
                          'quant_mode': quant_mode, 'sparsity': sparsity}

        # Linear layers for gates
        gate_input_dim = input_size + hidden_size
        
        # Reset gate components
        self.reset_gate = BitLinear(gate_input_dim, hidden_size, **factory_kwargs)
        
        # Update gate components
        self.update_gate = BitLinear(gate_input_dim, hidden_size, **factory_kwargs)
        
        # Candidate hidden state components
        self.new_gate = BitLinear(gate_input_dim, hidden_size, **factory_kwargs)
        
    def forward(self, 
                input: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform a single GRU cell time step.
        
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, input_size].
            hidden (Optional[torch.Tensor]): Hidden state from previous step.
                Shape: [batch_size, hidden_size]. Defaults to zeros if None.
            
        Returns:
            torch.Tensor: Updated hidden state of shape [batch_size, hidden_size].
        """
        batch_size = input.size(0)
        device = input.device

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=input.dtype)
        
        # Ensure hidden state is on the correct device
        hidden = hidden.to(device)

        # Concatenate input and previous hidden state
        combined_input = torch.cat([input, hidden], dim=1)
        
        # Calculate reset gate
        r_t = torch.sigmoid(self.reset_gate(combined_input))
        
        # Calculate update gate
        z_t = torch.sigmoid(self.update_gate(combined_input))
        
        # Calculate candidate hidden state
        combined_input_reset = torch.cat([input, r_t * hidden], dim=1)
        n_t = torch.tanh(self.new_gate(combined_input_reset))
        
        # Calculate final hidden state
        h_new = (1 - z_t) * hidden + z_t * n_t
        
        return h_new


def convert_linear_to_bit_linear(module: nn.Module, 
                                quant_mode: str = 'ternary',
                                sparsity: float = 0.5,
                                device: Optional[torch.device] = None) -> nn.Module:
    """
    Recursively converts nn.Linear layers to BitLinear in a PyTorch module.
    
    Args:
        module (nn.Module): The PyTorch module to convert.
        quant_mode (str): Quantization mode ('binary' or 'ternary').
        sparsity (float): Target sparsity for ternary mode.
        device (Optional[torch.device]): Target device for new layers.
        
    Returns:
        nn.Module: The modified module with BitLinear layers.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            # Create a new BitLinear layer with the appropriate configuration
            bit_linear_layer = BitLinear(
                in_features=child.in_features, 
                out_features=child.out_features, 
                bias=(child.bias is not None),
                quant_mode=quant_mode,
                sparsity=sparsity,
                device=device, 
                dtype=child.weight.dtype if child.weight.dtype.is_floating_point else torch.float32
            )
            
            # Copy weights and bias
            with torch.no_grad():
                bit_linear_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    bit_linear_layer.bias.copy_(child.bias)
                    
            # Replace the original layer
            setattr(module, name, bit_linear_layer)
        else:
            # Recursively convert child modules
            convert_linear_to_bit_linear(child, quant_mode, sparsity, device)
            
    return module


def quantize_linear_layers(model: nn.Module,
                          quant_type: Literal['binary', 'ternary', 'b1.58'] = 'ternary',
                          sparsity: float = 0.5) -> nn.Module:
    """
    Convenience function to quantize all linear layers in a model.
    
    Args:
        model (nn.Module): The model to quantize.
        quant_type (str): Quantization type ('binary', 'ternary', or 'b1.58').
        sparsity (float): Target sparsity for ternary/b1.58 mode.
        
    Returns:
        nn.Module: The quantized model.
    """
    # Normalize quant_type
    if quant_type == 'b1.58':
        quant_mode = 'ternary'
    else:
        quant_mode = quant_type
        
    # Apply the conversion
    device = next(model.parameters()).device
    return convert_linear_to_bit_linear(
        model,
        quant_mode=quant_mode,
        sparsity=sparsity,
        device=device
    )
