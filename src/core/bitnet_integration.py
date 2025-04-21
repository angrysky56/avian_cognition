"""
BitNet Integration Helper

This module provides utilities to integrate the BitNet repository with 
the Avian Cognition project, allowing the use of 1-bit weight quantization
in the cognitive modules.
"""

import os
import sys
import torch
import numpy as np

# Add BitNet repo to path if needed
BITNET_REPO_PATH = "/home/ty/Repositories/BitNet"
if os.path.exists(BITNET_REPO_PATH) and BITNET_REPO_PATH not in sys.path:
    sys.path.append(BITNET_REPO_PATH)

# Import BitNet functionality if available
try:
    import bitnet_cpp
    BITNET_AVAILABLE = True
    print("Successfully imported BitNet library")
except ImportError:
    BITNET_AVAILABLE = False
    print("Warning: BitNet library not found. Using fallback implementation.")

class BitNetWrapper:
    """
    Wrapper for BitNet models to use in Avian Cognition.
    
    This wrapper provides a consistent interface for using BitNet models
    within the Avian Cognitive Architecture, including fallbacks when
    the native BitNet implementation is not available.
    """
    
    def __init__(self, model_path=None, quant_type="i2_s"):
        """
        Initialize BitNet wrapper with optional model path.
        
        Args:
            model_path: Path to BitNet model (.gguf file)
            quant_type: Quantization type (i2_s or tl1)
        """
        self.model_path = model_path
        self.quant_type = quant_type
        self.model = None
        
        # Try to load BitNet model if available
        if BITNET_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                self.model = bitnet_cpp.BitNetModel(model_path)
                print(f"Successfully loaded BitNet model from {model_path}")
            except Exception as e:
                print(f"Error loading BitNet model: {e}")
        
    def quantize_module(self, module):
        """
        Apply BitNet quantization to a PyTorch module.
        
        Args:
            module: PyTorch module to quantize
            
        Returns:
            module: Quantized module
        """
        from src.core.bitnet import convert_linear_to_bit_linear
        
        # Use BitNet's native quantization if available
        if BITNET_AVAILABLE and self.model:
            # Implementation would depend on BitNet API
            # For now, we use our own BitLinear implementation
            return convert_linear_to_bit_linear(module)
        else:
            # Use our own implementation
            return convert_linear_to_bit_linear(module)
    
    def generate(self, prompt, max_tokens=100, temperature=0.7):
        """
        Generate text using the BitNet model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            text: Generated text
        """
        if not BITNET_AVAILABLE or not self.model:
            return "BitNet not available for text generation"
        
        try:
            return self.model.generate(
                prompt=prompt, 
                n_predict=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            return f"Error generating text: {e}"


def get_bitnet_model(model_name="BitNet-b1.58-2B-4T", local_path=None):
    """
    Get a BitNet model for use in Avian Cognition.
    
    Args:
        model_name: Name of BitNet model to load
        local_path: Optional local path to model
        
    Returns:
        model: BitNetWrapper instance
    """
    # First check if model exists locally
    if local_path and os.path.exists(local_path):
        return BitNetWrapper(model_path=local_path)
    
    # Check in BitNet repo
    bitnet_model_path = os.path.join(BITNET_REPO_PATH, "models", model_name, f"ggml-model-i2_s.gguf")
    if os.path.exists(bitnet_model_path):
        return BitNetWrapper(model_path=bitnet_model_path)
    
    # Return empty wrapper as fallback
    print(f"Warning: BitNet model {model_name} not found. Using fallback implementation.")
    return BitNetWrapper()


def apply_bitnet_quantization(model, bitnet_model=None):
    """
    Apply BitNet quantization to an Avian Mamba model.
    
    Args:
        model: AvianMambaModel to quantize
        bitnet_model: Optional BitNetWrapper to use
        
    Returns:
        model: Quantized model
    """
    if bitnet_model is None:
        bitnet_model = get_bitnet_model()
    
    # Apply quantization to cognitive modules
    if hasattr(model, 'metacognition_module'):
        model.metacognition_module = bitnet_model.quantize_module(model.metacognition_module)
        
    if hasattr(model, 'bayesian_module'):
        model.bayesian_module = bitnet_model.quantize_module(model.bayesian_module)
        
    if hasattr(model, 'planning_module'):
        model.planning_module = bitnet_model.quantize_module(model.planning_module)
        
    if hasattr(model, 'numerical_module'):
        model.numerical_module = bitnet_model.quantize_module(model.numerical_module)
    
    # Apply quantization to backbone (if possible)
    try:
        model.backbone = bitnet_model.quantize_module(model.backbone)
    except Exception as e:
        print(f"Warning: Could not quantize backbone: {e}")
    
    return model