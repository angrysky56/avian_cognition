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
    # Try importing from our own module first
    from src.core.bitnet_cpp import BitNetModel, check_bitnet_build
    BITNET_AVAILABLE = check_bitnet_build()
    if BITNET_AVAILABLE:
        print("Successfully imported BitNet library")
    else:
        print("Warning: BitNet binary not found. Using fallback implementation.")
        BITNET_AVAILABLE = False
except ImportError:
    # Fallback to importing directly
    try:
        import bitnet_cpp
        BITNET_AVAILABLE = True
        print("Successfully imported BitNet library")
    except ImportError:
        BITNET_AVAILABLE = False
        print("Warning: BitNet library not found. Using fallback implementation.")

class BitNetModel:
    """
    Integration with official BitNet models.
    
    This class provides a unified interface for using BitNet models,
    supporting both the native C++ implementation (bitnet.cpp) and
    HuggingFace transformers implementation when available.
    """
    
    def __init__(self, model_name_or_path, use_cpp=True, quant_type="i2_s"):
        """
        Initialize BitNet model integration.
        
        Args:
            model_name_or_path: HuggingFace model ID or path to local model
            use_cpp: Whether to use the C++ implementation (bitnet.cpp)
            quant_type: Quantization type for C++ implementation
        """
        self.model_name = model_name_or_path
        self.quant_type = quant_type
        self.use_cpp = use_cpp and BITNET_AVAILABLE
        
        self.model = None
        self.tokenizer = None
        self.cpp_model = None
        
        # Initialize the appropriate model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize either C++ or HuggingFace model based on configuration."""
        # Check for C++ implementation first if requested
        if self.use_cpp:
            try:
                # Try to find the model locally
                model_path = self._find_local_model()
                
                # Initialize C++ model if found
                if model_path:
                    if 'BitNetModel' in globals():
                        self.cpp_model = BitNetModel(model_path)
                    else:
                        self.cpp_model = bitnet_cpp.BitNetModel(model_path)
                    print(f"Successfully loaded BitNet C++ model from {model_path}")
                    return
                else:
                    print(f"Warning: Could not find local model at {model_path}")
            except Exception as e:
                print(f"Error initializing BitNet C++ model: {e}")
                self.use_cpp = False
        
        # Fall back to HuggingFace if C++ initialization failed or wasn't requested
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading BitNet model from HuggingFace: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print("Successfully loaded BitNet model via HuggingFace")
        except Exception as e:
            print(f"Error initializing HuggingFace BitNet model: {e}")
            if not self.cpp_model:
                print("Warning: Failed to initialize any BitNet model")
    
    def _find_local_model(self):
        """Find a local BitNet model file."""
        # Check if the path is a direct file path
        if os.path.isfile(self.model_name):
            return self.model_name
            
        # Check if it's a directory with a .gguf file
        if os.path.isdir(self.model_name):
            for file in os.listdir(self.model_name):
                if file.endswith(".gguf"):
                    return os.path.join(self.model_name, file)
        
        # Check in BitNet repo's models directory
        bitnet_model_path = os.path.join(BITNET_REPO_PATH, "models")
        if os.path.exists(bitnet_model_path):
            # Try exact name match
            if os.path.exists(os.path.join(bitnet_model_path, self.model_name)):
                model_dir = os.path.join(bitnet_model_path, self.model_name)
                # Look for .gguf file in model directory
                for file in os.listdir(model_dir):
                    if file.endswith(".gguf"):
                        return os.path.join(model_dir, file)
        
        # Check if it's a HuggingFace model ID that has been downloaded
        if "/" in self.model_name:
            model_id = self.model_name.split("/")[-1]
            model_path = os.path.join(BITNET_REPO_PATH, "models", model_id)
            if os.path.exists(model_path):
                for file in os.listdir(model_path):
                    if file.endswith(".gguf"):
                        return os.path.join(model_path, file)
        
        # If we get here, we couldn't find a local model
        return None
        
    def quantize_module(self, module):
        """
        Apply BitNet quantization to a PyTorch module.
        
        Args:
            module: PyTorch module to quantize
            
        Returns:
            module: Quantized module
        """
        from src.core.bitnet import convert_linear_to_bit_linear
        
        print(f"Quantizing module with BitNet b1.58 ternary weights")
        
        # Convert PyTorch module to use BitLinear layers
        quantized_module = convert_linear_to_bit_linear(module)
        
        return quantized_module
        
    def get_embedding_dim(self):
        """Get the embedding dimension of the model."""
        if self.model:
            # Try to get from HF model config
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'd_model'):
                return self.model.config.d_model
            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                return self.model.config.hidden_size
                
            # Fallback to estimating from parameters
            for name, param in self.model.named_parameters():
                if 'embed' in name.lower() and len(param.shape) == 2:
                    return param.shape[1]
        
        # Default value if we can't determine
        return 768  # Common embedding size
    
    def generate(self, prompt, max_tokens=100, temperature=0.7, do_sample=True, **kwargs):
        """
        Generate text using the BitNet model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (for HuggingFace)
            **kwargs: Additional arguments for the specific implementation
            
        Returns:
            text: Generated text
        """
        # Try C++ implementation first if available
        if self.cpp_model:
            try:
                return self.cpp_model.generate(
                    prompt=prompt, 
                    n_predict=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            except Exception as e:
                print(f"Error with C++ generation, falling back to HF: {e}")
                # Fall back to HuggingFace if C++ fails
        
        # Use HuggingFace implementation
        if self.model and self.tokenizer:
            try:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )
                
                # Decode and return
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                return f"Error generating with HuggingFace: {e}"
        
        # No implementation available
        return "BitNet not available for text generation"


def get_bitnet_model(model_name_or_path="microsoft/bitnet-b1.58-2B-4T", use_cpp=True):
    """
    Get a BitNet model for use in Avian Cognition.
    
    Args:
        model_name_or_path: HuggingFace model ID or path to local model
        use_cpp: Whether to use the C++ implementation if available
        
    Returns:
        model: BitNetModel instance
    """
    try:
        # Initialize model with proper fallbacks
        model = BitNetModel(model_name_or_path, use_cpp=use_cpp)
        return model
    except Exception as e:
        print(f"Error creating BitNet model: {e}")
        # Return empty model as fallback
        return BitNetModel("microsoft/bitnet-b1.58-2B-4T", use_cpp=False)


def apply_bitnet_quantization(model, bitnet_model=None):
    """
    Apply BitNet quantization to an Avian Mamba model.
    
    Args:
        model: AvianMambaModel to quantize
        bitnet_model: Optional BitNetModel instance to use
        
    Returns:
        model: Quantized model
    """
    if bitnet_model is None:
        bitnet_model = get_bitnet_model()
    
    print(f"Applying BitNet b1.58 ternary quantization to model")
    
    # Track original and quantized parameter count
    original_params = sum(p.numel() for p in model.parameters())
    
    # Apply quantization to cognitive modules
    modules_quantized = 0
    
    if hasattr(model, 'metacognition_module') and model.metacognition_module is not None:
        print("Quantizing metacognition module...")
        model.metacognition_module = bitnet_model.quantize_module(model.metacognition_module)
        modules_quantized += 1
        
    if hasattr(model, 'bayesian_module') and model.bayesian_module is not None:
        print("Quantizing Bayesian inference module...")
        model.bayesian_module = bitnet_model.quantize_module(model.bayesian_module)
        modules_quantized += 1
        
    if hasattr(model, 'planning_module') and model.planning_module is not None:
        print("Quantizing planning module...")
        model.planning_module = bitnet_model.quantize_module(model.planning_module)
        modules_quantized += 1
        
    if hasattr(model, 'numerical_module') and model.numerical_module is not None:
        print("Quantizing numerical module...")
        model.numerical_module = bitnet_model.quantize_module(model.numerical_module)
        modules_quantized += 1
    
    # Apply quantization to backbone (if possible)
    backbone_quantized = False
    try:
        if hasattr(model, 'backbone') and model.backbone is not None:
            print("Quantizing backbone model...")
            model.backbone = bitnet_model.quantize_module(model.backbone)
            backbone_quantized = True
    except Exception as e:
        print(f"Warning: Could not quantize backbone: {e}")
    
    # Calculate quantized parameters
    quantized_params = sum(p.numel() for p in model.parameters())
    
    print(f"Quantization complete:")
    print(f"  Modules quantized: {modules_quantized}")
    print(f"  Backbone quantized: {backbone_quantized}")
    print(f"  Original parameters: {original_params:,}")
    print(f"  Parameters after quantization: {quantized_params:,}")
    
    # Calculate estimated memory footprint
    fp32_size_mb = original_params * 4 / (1024 * 1024)
    b158_size_mb = quantized_params * 1.58 / 8 / (1024 * 1024)
    compression = fp32_size_mb / b158_size_mb if b158_size_mb > 0 else 0
    
    print(f"  Memory estimates:")
    print(f"    Original (FP32): {fp32_size_mb:.2f} MB")
    print(f"    Quantized (1.58-bit): {b158_size_mb:.2f} MB")
    print(f"    Compression ratio: {compression:.2f}x")
    
    return model


def estimate_model_size(model, with_quantization=True):
    """
    Estimate the model size with and without quantization.
    
    Args:
        model: The PyTorch model
        with_quantization: Whether to estimate with BitNet quantization
        
    Returns:
        size_info: Dictionary with size information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate memory in different formats
    fp32_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per parameter
    fp16_size_mb = total_params * 2 / (1024 * 1024)  # 2 bytes per parameter
    int8_size_mb = total_params * 1 / (1024 * 1024)  # 1 byte per parameter
    bit1_size_mb = total_params / 8 / (1024 * 1024)  # 1/8 byte per parameter
    
    # Create result dictionary
    result = {
        'total_parameters': total_params,
        'fp32_size_mb': fp32_size_mb,
        'fp16_size_mb': fp16_size_mb,
        'int8_size_mb': int8_size_mb,
        'bit1_size_mb': bit1_size_mb,
    }
    
    # Calculate compression ratios
    if with_quantization:
        result['compression_ratio_vs_fp32'] = fp32_size_mb / bit1_size_mb
        result['compression_ratio_vs_fp16'] = fp16_size_mb / bit1_size_mb
        result['compression_ratio_vs_int8'] = int8_size_mb / bit1_size_mb
    
    return result