#!/usr/bin/env python3
"""
Test BitNet Quantization

This script tests the BitNet quantization functionality by creating a
small model, applying quantization, and verifying the results.
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR  # Script is at project root
sys.path.append(PROJECT_ROOT)

# Import Avian Cognition modules
from src.core.mamba_integration import (
    AvianMambaConfig,
    AvianMambaModel,
    create_mini_model
)
from src.core.bitnet_integration import (
    apply_bitnet_quantization,
    get_bitnet_model,
    estimate_model_size
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test BitNet Quantization")
    parser.add_argument(
        "--model_size",
        type=str,
        default="mini",
        choices=["mini", "small"],
        help="Model size to test"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the quantized model to disk"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for saving outputs"
    )
    return parser.parse_args()


def create_model(args):
    """
    Create a model for testing.
    
    Args:
        args: Command line arguments
        
    Returns:
        model: Created model
    """
    print(f"Creating {args.model_size} model...")
    
    # Create model without quantization first
    config = AvianMambaConfig(
        vocab_size=10000,
        d_model=256 if args.model_size == "mini" else 768,
        n_layer=4 if args.model_size == "mini" else 24,
        ssm_d_state=8 if args.model_size == "mini" else 16,
        ssm_d_conv=2 if args.model_size == "mini" else 4,
        ssm_expand=2,
        enable_metacognition=True,
        enable_bayesian=True,
        enable_planning=True,
        enable_numerical=True,
        planning_steps=3 if args.model_size == "mini" else 5,
        quantize=False  # We'll apply quantization manually
    )
    
    model = AvianMambaModel(config).to(args.device)
    print(f"Created {args.model_size} model")
    
    return model


def test_model(model, args):
    """
    Test model functionality.
    
    Args:
        model: Model to test
        args: Command line arguments
        
    Returns:
        results: Test results
    """
    print("Testing model functionality...")
    
    # Create a dummy input
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=args.device)
    
    # Forward pass
    try:
        outputs = model(input_ids, return_dict=True)
        logits = outputs.logits
        print(f"Model forward pass successful: logits shape = {logits.shape}")
        success = True
    except Exception as e:
        print(f"Error in model forward pass: {e}")
        success = False
    
    # Test metacognition if available
    if hasattr(model, 'metacognition_module'):
        try:
            hidden_states = torch.randn(batch_size, model.config.d_model, device=args.device)
            confidence = model.metacognition_module(hidden_states)
            print(f"Metacognition module successful: confidence shape = {confidence.shape}")
        except Exception as e:
            print(f"Error in metacognition module: {e}")
            
    # Test bayesian module if available
    if hasattr(model, 'bayesian_module'):
        try:
            hidden_states = torch.randn(batch_size, model.config.d_model, device=args.device)
            belief_state, belief_embedding = model.bayesian_module(hidden_states)
            print(f"Bayesian module successful: belief state shape = {belief_state.shape}")
        except Exception as e:
            print(f"Error in bayesian module: {e}")
            
    # Test planning module if available
    if hasattr(model, 'planning_module'):
        try:
            hidden_states = torch.randn(batch_size, model.config.d_model, device=args.device)
            context_memory = torch.randn(batch_size, seq_len, model.config.d_model, device=args.device)
            plan_embedding, step_states, step_importances = model.planning_module(hidden_states, context_memory)
            print(f"Planning module successful: plan embedding shape = {plan_embedding.shape}")
        except Exception as e:
            print(f"Error in planning module: {e}")
            
    # Test numerical module if available
    if hasattr(model, 'numerical_module'):
        try:
            h1 = torch.randn(batch_size, model.config.d_model, device=args.device)
            h2 = torch.randn(batch_size, model.config.d_model, device=args.device)
            h_op = torch.randn(batch_size, model.config.d_model, device=args.device)
            result_hidden, op_weights = model.numerical_module(h1, h2, h_op)
            print(f"Numerical module successful: result shape = {result_hidden.shape}")
        except Exception as e:
            print(f"Error in numerical module: {e}")
    
    return {
        "success": success,
        "logits_shape": logits.shape if success else None
    }


def apply_quantization(model, args):
    """
    Apply BitNet quantization to the model.
    
    Args:
        model: Model to quantize
        args: Command line arguments
        
    Returns:
        model: Quantized model
    """
    print("Applying BitNet quantization...")
    
    # Get BitNet model
    bitnet_model = get_bitnet_model()
    
    # Get model size before quantization
    size_before = estimate_model_size(model, with_quantization=False)
    print(f"Model size before quantization: {size_before['fp32_size_mb']:.2f} MB (FP32)")
    
    # Apply quantization
    model = apply_bitnet_quantization(model, bitnet_model)
    
    # Get model size after quantization
    size_after = estimate_model_size(model, with_quantization=True)
    print(f"Model size after quantization: {size_after['bit1_size_mb']:.2f} MB (1-bit)")
    print(f"Compression ratio: {size_after['compression_ratio_vs_fp32']:.2f}x")
    
    # Update model config
    model.config.quantize = True
    
    return model


def save_model(model, args):
    """
    Save the model to disk.
    
    Args:
        model: Model to save
        args: Command line arguments
    """
    if not args.save_model:
        return
    
    print("Saving quantized model...")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"{args.model_size}_quantized")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    try:
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
        print(f"Model saved to {os.path.join(output_dir, 'model.pt')}")
    except Exception as e:
        print(f"Error saving model: {e}")


def main():
    """Main function."""
    args = parse_args()
    
    print("=== Testing BitNet Quantization ===")
    print(f"Model size: {args.model_size}")
    print(f"Device: {args.device}")
    print(f"Save model: {args.save_model}")
    print(f"Output directory: {args.output_dir}")
    print("===================================")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    model = create_model(args)
    
    # Test model before quantization
    print("\n--- Testing Model Before Quantization ---")
    results_before = test_model(model, args)
    
    # Apply quantization
    model = apply_quantization(model, args)
    
    # Test model after quantization
    print("\n--- Testing Model After Quantization ---")
    results_after = test_model(model, args)
    
    # Save model
    save_model(model, args)
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Model size: {args.model_size}")
    print(f"Before quantization: {'✓' if results_before['success'] else '✗'}")
    print(f"After quantization: {'✓' if results_after['success'] else '✗'}")
    
    if results_before['success'] and results_after['success']:
        print("BitNet quantization test passed!")
    else:
        print("BitNet quantization test failed")
        
    return 0 if results_before['success'] and results_after['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
