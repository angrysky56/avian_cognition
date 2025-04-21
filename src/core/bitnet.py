"""
BitNet Quantization Implementation

This module implements 1-bit weight quantization for neural networks,
following the BitNet approach to extreme model compression while
maintaining performance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Module):
    """
    Linear layer with 1-bit weight quantization.
    
    Maintains full-precision weights during training but uses binary (+1/-1)
    weights during forward pass, enabling extreme memory efficiency with
    minimal performance degradation.
    
    Attributes:
        in_features: Input feature dimension
        out_features: Output feature dimension
        weight: Full-precision weight tensor (for gradient updates)
        scale: Scaling factor to preserve output magnitude
        bias: Optional bias parameter
    """
    
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Full-precision weights for gradient updates
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Scale factor to preserve output magnitude
        self.register_buffer('scale', torch.ones(1))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters following standard linear layer initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input):
        """
        Perform forward pass with binarized weights.
        
        Args:
            input: Input tensor of shape [batch_size, in_features]
            
        Returns:
            output: Output tensor of shape [batch_size, out_features]
        """
        # Move to input device if necessary
        if input.device != self.weight.device:
            self.to(input.device)
            
        # Binary quantization
        binary_weight = torch.sign(self.weight)
        
        # Scale factor (to preserve output magnitude)
        self.scale = self.weight.abs().mean()
        
        # Forward pass with binary weights
        return F.linear(input, binary_weight * self.scale, self.bias)
    
    def extra_repr(self):
        """String representation with dimensions."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class BitGRUCell(nn.Module):
    """
    GRU cell with BitLinear layers for extreme efficiency.
    
    Implements a recurrent cell with 1-bit weight quantization for all
    linear transformations, suitable for planning and sequential reasoning.
    
    Attributes:
        input_size: Input dimension
        hidden_size: Hidden state dimension
        reset_gate: BitLinear layer for reset gate
        update_gate: BitLinear layer for update gate
        new_gate: BitLinear layer for candidate activation
    """
    
    def __init__(self, input_size, hidden_size, device=None, dtype=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Reset gate
        self.reset_gate = BitLinear(input_size + hidden_size, hidden_size)
        
        # Update gate
        self.update_gate = BitLinear(input_size + hidden_size, hidden_size)
        
        # Candidate activation
        self.new_gate = BitLinear(input_size + hidden_size, hidden_size)
        
    def forward(self, input, hidden=None):
        """
        Perform GRU cell forward pass with BitLinear operations.
        
        Args:
            input: Input tensor of shape [batch_size, input_size]
            hidden: Previous hidden state [batch_size, hidden_size] or None
            
        Returns:
            h_new: Updated hidden state [batch_size, hidden_size]
        """
        # Move to input device if necessary
        if input.device != self.reset_gate.weight.device:
            self.to(input.device)
            
        if hidden is None:
            hidden = torch.zeros(input.size(0), self.hidden_size, device=input.device)
            
        # Combined input for gate computation
        combined = torch.cat([input, hidden], dim=1)
        
        # Compute gates
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        
        # Compute candidate activation
        combined_reset = torch.cat([input, r * hidden], dim=1)
        n = torch.tanh(self.new_gate(combined_reset))
        
        # Update hidden state
        h_new = (1 - z) * hidden + z * n
        
        return h_new


def convert_linear_to_bit_linear(module, device=None):
    """
    Recursively converts all nn.Linear layers in a module to BitLinear.
    
    Args:
        module: PyTorch module to convert
        device: Optional device to place BitLinear layers on
        
    Returns:
        module: Module with all linear layers converted to BitLinear
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            bit_linear = BitLinear(child.in_features, child.out_features, bias=child.bias is not None)
            
            # Copy weights and bias
            with torch.no_grad():
                bit_linear.weight.copy_(child.weight)
                if child.bias is not None:
                    bit_linear.bias.copy_(child.bias)
                    
            # Move to device if specified
            if device is not None:
                bit_linear = bit_linear.to(device)
                    
            # Replace module
            setattr(module, name, bit_linear)
        else:
            # Recursively convert children
            convert_linear_to_bit_linear(child, device=device)
            
    return module


class NALULayer(nn.Module):
    """
    Neural Arithmetic Logic Unit for precise numerical operations.
    
    Implements a differentiable module capable of learning to perform
    exact arithmetic operations with generalization beyond the training range.
    
    Attributes:
        in_features: Input feature dimension
        out_features: Output feature dimension
        G: Weight matrix for gating between addition and multiplication paths
        W: Weight matrix for addition path
        M: Weight matrix for multiplication path (in log space)
    """
    
    def __init__(self, in_features, out_features, eps=1e-7, device=None, dtype=None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        # Weight matrices for gating between add and multiply paths
        self.G = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.M = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters following NALU paper recommendations."""
        nn.init.kaiming_uniform_(self.G, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))
        
    def forward(self, x):
        """
        Perform NALU operation, combining addition and multiplication paths.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            y: Output tensor of shape [batch_size, out_features]
        """
        # Move to input device if necessary
        if x.device != self.G.device:
            self.to(x.device)
            
        # Gate for add/multiply operation selection
        g = torch.sigmoid(F.linear(x, self.G))
        
        # Addition sub-operation
        a = F.linear(x, self.W)
        
        # Multiplication sub-operation (in log space)
        m = torch.exp(F.linear(torch.log(torch.abs(x) + self.eps), self.M))
        
        # Combine operations
        y = g * a + (1 - g) * m
        
        return y