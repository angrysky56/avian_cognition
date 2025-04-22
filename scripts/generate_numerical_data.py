#!/usr/bin/env python3
"""
Numerical Data Generation Script

This script generates training, validation, and extrapolation data for the
NumericalModule. It creates hidden state representations for arithmetic operations
and expected results within different value ranges.

Usage:
    python scripts/generate_numerical_data.py \
        --output_dir data/numerical \
        --hidden_dim 512 \
        --num_train 5000 \
        --num_val 1000 \
        --num_extrapolation 1000
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate data for NumericalModule training"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/numerical",
        help="Directory to save generated data"
    )
    
    # Data configuration
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=5000,
        help="Number of training examples"
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=1000,
        help="Number of validation examples"
    )
    parser.add_argument(
        "--num_extrapolation",
        type=int,
        default=1000,
        help="Number of extrapolation examples"
    )
    parser.add_argument(
        "--train_range",
        type=int,
        nargs=2,
        default=[0, 100],
        help="Range of values for training data (min max)"
    )
    parser.add_argument(
        "--val_range",
        type=int,
        nargs=2,
        default=[0, 100],
        help="Range of values for validation data (min max)"
    )
    parser.add_argument(
        "--extrapolation_range",
        type=int,
        nargs=2,
        default=[100, 1000],
        help="Range of values for extrapolation data (min max)"
    )
    
    # Runtime configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for tensor operations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def generate_data(
    num_examples, 
    operations, 
    value_range, 
    hidden_dim, 
    device
):
    """
    Generate arithmetic examples with corresponding hidden states.
    
    Args:
        num_examples: Number of examples to generate
        operations: List of operations to include
        value_range: Range of values for operands (min, max)
        hidden_dim: Hidden dimension size
        device: Device to place tensors on
        
    Returns:
        data: Dictionary of generated data
    """
    min_val, max_val = value_range
    
    # Initialize storage
    h1_data = torch.zeros(num_examples, hidden_dim, device=device)
    h2_data = torch.zeros(num_examples, hidden_dim, device=device)
    h_op_data = torch.zeros(num_examples, hidden_dim, device=device)
    target_data = torch.zeros(num_examples, 1, device=device)
    
    # Track operation indexes and original operands
    operation_indices = []
    original_operands = []
    
    # Generate examples
    for i in range(num_examples):
        # Select random operation
        op_idx = np.random.randint(0, len(operations))
        operation = operations[op_idx]
        operation_indices.append(op_idx)
        
        # Generate operands
        a = np.random.randint(min_val, max_val)
        
        # For division, ensure we don't divide by zero
        if operation == 'divide':
            b = max(1, np.random.randint(min_val, max_val))
        else:
            b = np.random.randint(min_val, max_val)
            
        original_operands.append((a, b))
        
        # Compute result
        if operation == 'add':
            result = a + b
        elif operation == 'subtract':
            result = a - b
        elif operation == 'multiply':
            result = a * b
        elif operation == 'divide':
            result = a / b
        
        # Find normalization factor (maximum possible value in this range)
        if operation == 'add':
            max_result = 2 * max_val
        elif operation == 'subtract':
            max_result = max_val
        elif operation == 'multiply':
            max_result = max_val * max_val
        elif operation == 'divide':
            max_result = max_val  # Simplification, actual max could be larger
            
        # Use a scaling factor to normalize
        scaling_factor = 1.0 / max(1.0, max_result)
        
        # Improved distributed representation using logarithmic scale and periodic encoding
        
        # First value: Sign bit
        h1_data[i, 0] = float(np.sign(a))
        h2_data[i, 0] = float(np.sign(b))
        
        # Second value: Magnitude (log scale to handle large numbers better)
        h1_data[i, 1] = float(np.log1p(abs(a)))
        h2_data[i, 1] = float(np.log1p(abs(b)))
        
        # Distributed representation using periodic functions
        # This creates a more nuanced encoding that captures the continuous nature of numbers
        for j in range(2, min(10, hidden_dim)):
            freq = j - 1  # Frequency increases with dimension
            h1_data[i, j] = float(np.sin(a * freq * 0.01))
            h2_data[i, j] = float(np.sin(b * freq * 0.01))
            
            # Add cosine encoding to capture more information
            if j + 8 < hidden_dim:
                h1_data[i, j + 8] = float(np.cos(a * freq * 0.01))
                h2_data[i, j + 8] = float(np.cos(b * freq * 0.01))
        
        # Additional magnitude encoding spread across multiple dimensions
        if hidden_dim > 20:
            # Normalized value spread across a few dimensions
            for j in range(20, min(25, hidden_dim)):
                power = (j - 20) / 5.0  # Powers between 0 and 1
                h1_data[i, j] = float((abs(a) ** power) / (max_val ** power))
                h2_data[i, j] = float((abs(b) ** power) / (max_val ** power))
        
        # One-hot operation encoding
        h_op_data[i, op_idx] = 1.0
        
        # Store scaled result
        target_data[i, 0] = result * scaling_factor
    
    # Return data
    data = {
        'h1_data': h1_data,
        'h2_data': h2_data,
        'h_op_data': h_op_data,
        'target_data': target_data,
        'operations': operations,
        'operation_indices': operation_indices,
        'original_operands': original_operands,
        'scaling_factor': scaling_factor,
        'value_range': value_range
    }
    
    return data


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Define operations
    operations = ['add', 'subtract', 'multiply', 'divide']
    
    print("Generating numerical training data...")
    
    # Generate training data
    train_data = generate_data(
        args.num_train, operations, args.train_range, args.hidden_dim, args.device
    )
    
    # Generate validation data
    val_data = generate_data(
        args.num_val, operations, args.val_range, args.hidden_dim, args.device
    )
    
    # Generate extrapolation data
    extrapolation_data = generate_data(
        args.num_extrapolation, operations, args.extrapolation_range, args.hidden_dim, args.device
    )
    
    # Save data to file
    print(f"Saving data to {args.output_dir}...")
    
    train_path = os.path.join(args.output_dir, "train_data.pt")
    val_path = os.path.join(args.output_dir, "val_data.pt")
    extrapolation_path = os.path.join(args.output_dir, "extrapolation_data.pt")
    
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    torch.save(extrapolation_data, extrapolation_path)
    
    print(f"Generated and saved {args.num_train} training examples, {args.num_val} validation examples, and {args.num_extrapolation} extrapolation examples")


if __name__ == "__main__":
    main()
