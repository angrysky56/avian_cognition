#!/usr/bin/env python3
"""
Numerical Module Evaluation Script

This script evaluates a trained NumericalModule on various arithmetic operations,
with a focus on extrapolation capabilities and visualizing the results.

Usage:
    python scripts/evaluate_numerical_module.py \
        --model_path checkpoints/numerical/best_model.pt \
        --data_dir data/numerical \
        --output_dir results/numerical
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.numerical import NumericalModule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NumericalModule"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/numerical",
        help="Data directory containing test data"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/numerical",
        help="Directory to save evaluation results"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="Number of examples to generate for synthetic evaluation"
    )
    parser.add_argument(
        "--extrapolation_ranges",
        type=int,
        nargs="+",
        default=[100, 1000, 10000, 100000],
        help="Upper bounds for extrapolation ranges"
    )
    
    # Runtime configuration
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Infer device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def load_model(model_path, device):
    """
    Load a trained NumericalModule model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to place model on
        
    Returns:
        model: Loaded NumericalModule model
    """
    print(f"Loading model from {model_path}...")
    
    # Add necessary global for NumPy scalar types
    try:
        from numpy._core.multiarray import scalar
        import torch.serialization
        torch.serialization.add_safe_globals([scalar])
    except ImportError:
        print("WARNING: Could not import numpy._core.multiarray.scalar, attempting load with weights_only=False")
    
    # Load checkpoint with weights_only=False to handle NumPy arrays
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except (RuntimeError, TypeError) as e:
        print(f"Error with weights_only=False: {e}")
        print("Trying legacy loading method...")
        import pickle
        checkpoint = torch.load(model_path, map_location=device, pickle_module=pickle)
    
    # Extract configuration
    config = checkpoint.get('config', {})
    hidden_dim = config.get('hidden_dim', 512)
    num_dim = config.get('num_dim', 32)
    bit_linear = config.get('bit_linear', False)
    
    print(f"Model config: hidden_dim={hidden_dim}, num_dim={num_dim}, bit_linear={bit_linear}")
    
    # Create model
    model = NumericalModule(
        hidden_dim=hidden_dim,
        num_dim=num_dim,
        bit_linear=bit_linear
    ).to(device)
    
    # Check if state dict has wrapped model format (from NumericalModuleWithDecoder)
    state_dict = checkpoint['model_state_dict']
    
    # If state dict keys have 'numerical_module.' prefix, remove it
    if any(key.startswith('numerical_module.') for key in state_dict.keys()):
        print("Detected wrapped model state dict, extracting numerical module weights...")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('numerical_module.'):
                # Remove prefix to match the expected model keys
                new_key = key[len('numerical_module.'):]
                new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    # Load state dict with error handling
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    # Set to evaluation mode
    model.eval()
    
    print("Model loaded successfully")
    return model


def load_test_data(data_dir, device):
    """
    Load extrapolation data for testing.
    
    Args:
        data_dir: Directory containing data files
        device: Device to place data on
        
    Returns:
        test_data: Dictionary of test data
    """
    print(f"Loading test data from {data_dir}...")
    
    # Check if extrapolation data file exists
    extrapolation_path = os.path.join(data_dir, "extrapolation_data.pt")
    
    if not os.path.exists(extrapolation_path):
        raise FileNotFoundError(f"Extrapolation data not found at {extrapolation_path}")
    
    # Add necessary global for NumPy scalar types
    try:
        from numpy._core.multiarray import scalar
        import torch.serialization
        torch.serialization.add_safe_globals([scalar])
    except ImportError:
        print("WARNING: Could not import numpy._core.multiarray.scalar, attempting load with weights_only=False")
    
    # Load data with weights_only=False to handle NumPy arrays
    try:
        extrapolation_data = torch.load(extrapolation_path, weights_only=False)
    except (RuntimeError, TypeError) as e:
        print(f"Error with weights_only=False: {e}")
        print("Trying legacy loading method...")
        import pickle
        extrapolation_data = torch.load(extrapolation_path, pickle_module=pickle)
    
    # Move to device
    h1_data = extrapolation_data['h1_data'].to(device)
    h2_data = extrapolation_data['h2_data'].to(device)
    h_op_data = extrapolation_data['h_op_data'].to(device)
    target_data = extrapolation_data['target_data'].to(device)
    original_operands = extrapolation_data['original_operands']
    operations = extrapolation_data['operations']
    
    test_data = {
        'h1_data': h1_data,
        'h2_data': h2_data,
        'h_op_data': h_op_data,
        'target_data': target_data,
        'original_operands': original_operands,
        'operations': operations
    }
    
    print(f"Loaded {len(h1_data)} test examples")
    
    return test_data


def evaluate_on_test_data(model, test_data, batch_size, device):
    """
    Evaluate model on test data.
    
    Args:
        model: NumericalModule model
        test_data: Test data dictionary
        batch_size: Batch size for evaluation
        device: Device for computation
        
    Returns:
        results: Dictionary of evaluation results
    """
    print("Evaluating model on test data...")
    
    # Extract data tensors
    h1_data = test_data['h1_data']
    h2_data = test_data['h2_data']
    h_op_data = test_data['h_op_data']
    target_data = test_data['target_data']
    
    # Get dimensions
    num_examples = h1_data.size(0)
    num_batches = (num_examples + batch_size - 1) // batch_size
    
    # Track results
    all_predictions = []
    all_targets = []
    
    # Process data in batches without using DataLoader
    with torch.no_grad():
        for i in tqdm(range(0, num_examples, batch_size), desc="Evaluating"):
            # Extract batch
            end_idx = min(i + batch_size, num_examples)
            
            h1 = h1_data[i:end_idx].to(device)
            h2 = h2_data[i:end_idx].to(device)
            h_op = h_op_data[i:end_idx].to(device)
            targets = target_data[i:end_idx].to(device)
            
            # Forward pass
            result_hidden, op_weights = model(h1, h2, h_op)
            
            # Extract predictions from the first dimension
            # (consistent with the training script)
            predictions = result_hidden[:, 0:1]
            
            # Store predictions and targets
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate results
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Calculate metrics
    mse = nn.MSELoss()(all_predictions, all_targets).item()
    
    # Calculate absolute errors
    abs_errors = torch.abs(all_predictions - all_targets).squeeze()
    
    # Calculate accuracy with adaptive tolerance based on scale
    # Small numbers get absolute tolerance, large numbers get relative tolerance
    abs_tolerance = torch.tensor(0.1)  # Fixed tolerance for small numbers
    rel_tolerance = torch.abs(all_targets) * 0.05  # 5% relative tolerance for larger numbers
    
    # Use absolute tolerance for small numbers, relative tolerance for larger ones
    tolerance = torch.where(
        torch.abs(all_targets) < 10.0,
        abs_tolerance, 
        rel_tolerance.squeeze()
    )
    
    # Ensure minimum tolerance
    tolerance = torch.max(tolerance, torch.tensor(1e-2))
    
    # Calculate accuracy as percentage within tolerance
    within_tolerance = (abs_errors <= tolerance).float().mean().item()
    
    # Return results
    results = {
        'mse': mse,
        'accuracy': within_tolerance,
        'predictions': all_predictions,
        'targets': all_targets,
        'abs_errors': abs_errors
    }
    
    print(f"Test MSE: {mse:.6f}, Accuracy (5% tolerance): {within_tolerance:.4f}")
    
    return results


def create_synthetic_data(model, operation, value_range, num_examples, hidden_dim, device):
    """
    Create synthetic data for a specific operation and value range.
    
    Args:
        model: NumericalModule model
        operation: Mathematical operation ('add', 'subtract', 'multiply', 'divide')
        value_range: Range of values for operands (min_val, max_val)
        num_examples: Number of examples to generate
        hidden_dim: Hidden dimension size
        device: Device for computation
        
    Returns:
        data: Dictionary of synthetic data
    """
    min_val, max_val = value_range
    
    # Generate operand pairs with appropriate constraints
    operands = []
    results = []
    
    for _ in range(num_examples):
        # Generate operands within range
        a = np.random.randint(min_val, max_val)
        
        # For division, ensure clean division (when possible)
        if operation == 'divide':
            if a > 1 and np.random.random() < 0.8:  # 80% clean division
                # Need to ensure the range is valid (high > low)
                high_val = min(a, max_val // 2)
                if high_val > 1:
                    b = np.random.randint(1, high_val)
                    b = b * (a // b) if a // b > 0 else b  # Make a divisible by b
                else:
                    # Default to 1 if no valid range
                    b = 1
            else:
                # Avoid division by zero and ensure valid range
                if max_val > min_val:
                    b = max(1, np.random.randint(min_val, max_val))
                else:
                    b = 1
        else:
            # For other operations, ensure valid range
            if max_val > min_val:
                b = np.random.randint(min_val, max_val)
            else:
                b = min_val  # Default to min_val if no valid range
            
        # Compute result based on operation
        if operation == 'add':
            result = a + b
        elif operation == 'subtract':
            result = a - b
        elif operation == 'multiply':
            result = a * b
        elif operation == 'divide':
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        operands.append((a, b))
        results.append(result)
    
    # Convert to tensors
    results = torch.tensor(results, dtype=torch.float32, device=device).view(-1, 1)
    
    # Create hidden state representations for operands
    h1 = torch.zeros(num_examples, hidden_dim, device=device)
    h2 = torch.zeros(num_examples, hidden_dim, device=device)
    h_op = torch.zeros(num_examples, hidden_dim, device=device)
    
    # Simple numeric encoding (same as in training script)
    max_magnitude = max(max_val, max(abs(r) for r in results.view(-1).cpu().numpy()))
    scaling_factor = 1.0 / max(1.0, max_magnitude)
    
    for i, (a, b) in enumerate(operands):
        # Improved distributed representation using logarithmic scale and periodic encoding
        
        # First value: Sign bit
        h1[i, 0] = float(np.sign(a))
        h2[i, 0] = float(np.sign(b))
        
        # Second value: Magnitude (log scale to handle large numbers better)
        h1[i, 1] = float(np.log1p(abs(a)))
        h2[i, 1] = float(np.log1p(abs(b)))
        
        # Distributed representation using periodic functions
        # This creates a more nuanced encoding that captures the continuous nature of numbers
        for j in range(2, min(10, hidden_dim)):
            freq = j - 1  # Frequency increases with dimension
            h1[i, j] = float(np.sin(a * freq * 0.01))
            h2[i, j] = float(np.sin(b * freq * 0.01))
            
            # Add cosine encoding to capture more information
            if j + 8 < hidden_dim:
                h1[i, j + 8] = float(np.cos(a * freq * 0.01))
                h2[i, j + 8] = float(np.cos(b * freq * 0.01))
        
        # Additional magnitude encoding spread across multiple dimensions
        if hidden_dim > 20:
            # Normalized value spread across a few dimensions
            for j in range(20, min(25, hidden_dim)):
                power = (j - 20) / 5.0  # Powers between 0 and 1
                h1[i, j] = float((abs(a) ** power) / (max_val ** power))
                h2[i, j] = float((abs(b) ** power) / (max_val ** power))
    
    # Encode operation (one-hot)
    op_idx = ['add', 'subtract', 'multiply', 'divide'].index(operation)
    h_op[:, op_idx] = 1.0
    
    # Return data
    data = {
        'h1_data': h1,
        'h2_data': h2,
        'h_op_data': h_op,
        'target_data': results,
        'original_operands': operands,
        'operation': operation,
        'scaling_factor': scaling_factor
    }
    
    return data


def evaluate_extrapolation(model, operations, extrapolation_ranges, num_examples, device):
    """
    Evaluate model on extrapolation tasks across different value ranges.
    
    Args:
        model: NumericalModule model
        operations: List of operations to evaluate
        extrapolation_ranges: List of upper bounds for extrapolation ranges
        num_examples: Number of examples per range and operation
        device: Device for computation
        
    Returns:
        results: Dictionary of extrapolation results
    """
    print(f"Evaluating extrapolation performance across {len(extrapolation_ranges)} ranges...")
    
    # Extract model configuration
    hidden_dim = model.hidden_dim
    
    # Track results
    extrapolation_results = {}
    
    # Evaluate each operation at each range
    for operation in operations:
        operation_results = {}
        
        for max_val in extrapolation_ranges:
            # Define value range
            value_range = (0, max_val)
            
            # Create synthetic data
            data = create_synthetic_data(
                model, operation, value_range, num_examples, hidden_dim, device
            )
            
            # Evaluate on this data
            with torch.no_grad():
                # Forward pass
                result_hidden, op_weights = model(
                    data['h1_data'], data['h2_data'], data['h_op_data']
                )
                
                # Extract predictions from the first dimension
                predictions = result_hidden[:, 0:1] / data['scaling_factor']
                targets = data['target_data']
                
                # Calculate metrics
                mse = nn.MSELoss()(predictions, targets).item()
                
                # Calculate absolute errors
                abs_errors = torch.abs(predictions - targets).squeeze().cpu().numpy()
                
                # Calculate accuracy with 5% tolerance
                rel_tolerance = torch.abs(targets) * 0.05
                rel_tolerance = torch.max(rel_tolerance, torch.tensor(1e-2, device=device))
                within_tolerance = (torch.abs(predictions - targets) <= rel_tolerance).float().mean().item()
                
                # Store results
                range_results = {
                    'mse': mse,
                    'accuracy': within_tolerance,
                    'abs_errors': abs_errors.tolist(),
                    'percentile_25': np.percentile(abs_errors, 25),
                    'percentile_50': np.percentile(abs_errors, 50),
                    'percentile_75': np.percentile(abs_errors, 75),
                    'percentile_95': np.percentile(abs_errors, 95),
                    'range': value_range
                }
                
                operation_results[max_val] = range_results
                
                print(f"  {operation.capitalize()} Range 0-{max_val}: MSE={mse:.6f}, Accuracy={within_tolerance:.4f}")
        
        extrapolation_results[operation] = operation_results
    
    return extrapolation_results


def plot_extrapolation_results(extrapolation_results, operations, output_dir):
    """
    Create visualizations for extrapolation results.
    
    Args:
        extrapolation_results: Dictionary of extrapolation results
        operations: List of operations evaluated
        output_dir: Directory to save plots
    """
    print("Generating visualizations...")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Accuracy vs. Range
    plt.figure(figsize=(12, 8))
    
    for operation in operations:
        # Extract ranges and accuracies
        ranges = []
        accuracies = []
        
        for max_val, results in extrapolation_results[operation].items():
            ranges.append(max_val)
            accuracies.append(results['accuracy'])
        
        # Sort by range
        sorted_data = sorted(zip(ranges, accuracies))
        ranges, accuracies = zip(*sorted_data)
        
        # Plot
        plt.plot(ranges, accuracies, 'o-', label=operation.capitalize())
    
    plt.xlabel('Value Range Upper Bound', fontsize=12)
    plt.ylabel('Accuracy (5% tolerance)', fontsize=12)
    plt.title('Extrapolation Accuracy vs. Value Range', fontsize=14)
    plt.xscale('log')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "extrapolation_accuracy.png"))
    plt.close()
    
    # Plot 2: Median Error vs. Range
    plt.figure(figsize=(12, 8))
    
    for operation in operations:
        # Extract ranges and median errors
        ranges = []
        median_errors = []
        
        for max_val, results in extrapolation_results[operation].items():
            ranges.append(max_val)
            median_errors.append(results['percentile_50'])
        
        # Sort by range
        sorted_data = sorted(zip(ranges, median_errors))
        ranges, median_errors = zip(*sorted_data)
        
        # Plot
        plt.plot(ranges, median_errors, 'o-', label=operation.capitalize())
    
    plt.xlabel('Value Range Upper Bound', fontsize=12)
    plt.ylabel('Median Absolute Error', fontsize=12)
    plt.title('Extrapolation Median Error vs. Value Range', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "extrapolation_median_error.png"))
    plt.close()
    
    # Plot 3: Error Percentiles for each operation
    for operation in operations:
        plt.figure(figsize=(12, 8))
        
        # Extract ranges and error percentiles
        ranges = []
        p25_errors = []
        p50_errors = []
        p75_errors = []
        p95_errors = []
        
        for max_val, results in extrapolation_results[operation].items():
            ranges.append(max_val)
            p25_errors.append(results['percentile_25'])
            p50_errors.append(results['percentile_50'])
            p75_errors.append(results['percentile_75'])
            p95_errors.append(results['percentile_95'])
        
        # Sort by range
        sorted_indices = np.argsort(ranges)
        ranges = [ranges[i] for i in sorted_indices]
        p25_errors = [p25_errors[i] for i in sorted_indices]
        p50_errors = [p50_errors[i] for i in sorted_indices]
        p75_errors = [p75_errors[i] for i in sorted_indices]
        p95_errors = [p95_errors[i] for i in sorted_indices]
        
        # Plot
        plt.plot(ranges, p25_errors, 'o-', label='25th Percentile')
        plt.plot(ranges, p50_errors, 'o-', label='50th Percentile (Median)')
        plt.plot(ranges, p75_errors, 'o-', label='75th Percentile')
        plt.plot(ranges, p95_errors, 'o-', label='95th Percentile')
        
        plt.xlabel('Value Range Upper Bound', fontsize=12)
        plt.ylabel('Absolute Error', fontsize=12)
        plt.title(f'{operation.capitalize()} Error Percentiles', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{operation}_error_percentiles.png"))
        plt.close()
    
    # Plot 4: Extrapolation Factor Analysis
    plt.figure(figsize=(12, 8))
    
    for operation in operations:
        # Calculate extrapolation factors
        ranges = []
        extrapolation_factors = []
        
        # Sort ranges
        sorted_ranges = sorted(extrapolation_results[operation].keys())
        
        # Use the smallest range as the reference
        reference_range = sorted_ranges[0]
        reference_accuracy = extrapolation_results[operation][reference_range]['accuracy']
        
        for max_val in sorted_ranges:
            if max_val == reference_range:
                continue
                
            accuracy = extrapolation_results[operation][max_val]['accuracy']
            
            # Skip if reference accuracy is 0
            if reference_accuracy == 0:
                continue
                
            # Calculate extrapolation factor
            factor = accuracy / reference_accuracy
            
            ranges.append(max_val / reference_range)
            extrapolation_factors.append(factor)
        
        # Plot
        plt.plot(ranges, extrapolation_factors, 'o-', label=operation.capitalize())
    
    plt.xlabel('Extrapolation Range Factor', fontsize=12)
    plt.ylabel('Accuracy Retention Factor', fontsize=12)
    plt.title('Accuracy Retention vs. Extrapolation Range', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='gray', linestyle='--')
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "extrapolation_factor_analysis.png"))
    plt.close()
    
    print(f"Visualizations saved to {plots_dir}")


def save_results(results, extrapolation_results, operations, output_dir):
    """
    Save evaluation results to files.
    
    Args:
        results: Dictionary of standard evaluation results
        extrapolation_results: Dictionary of extrapolation results
        operations: List of operations evaluated
        output_dir: Directory to save results
    """
    print("Saving results...")
    
    # Save standard evaluation results
    results_path = os.path.join(output_dir, "evaluation_results.pt")
    try:
        # Try to save with pickle to ensure compatibility
        import pickle
        torch.save(results, results_path, pickle_module=pickle)
    except Exception as e:
        print(f"Error saving with pickle: {e}")
        # Fall back to standard save
        torch.save(results, results_path)
    
    # Save extrapolation results
    extrapolation_path = os.path.join(output_dir, "extrapolation_results.pt")
    try:
        # Try to save with pickle to ensure compatibility
        import pickle
        torch.save(extrapolation_results, extrapolation_path, pickle_module=pickle)
    except Exception as e:
        print(f"Error saving with pickle: {e}")
        # Fall back to standard save
        torch.save(extrapolation_results, extrapolation_path)
    
    # Create a summary report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("# NumericalModule Evaluation Report\n\n")
        
        # Standard evaluation
        f.write("## Standard Evaluation\n")
        f.write(f"MSE: {results['mse']:.6f}\n")
        f.write(f"Accuracy (5% tolerance): {results['accuracy']:.4f}\n\n")
        
        # Extrapolation evaluation
        f.write("## Extrapolation Evaluation\n\n")
        
        for operation in operations:
            f.write(f"### {operation.capitalize()}\n\n")
            f.write("| Range | MSE | Accuracy | Median Error |\n")
            f.write("|-------|-----|----------|-------------|\n")
            
            # Sort ranges
            sorted_ranges = sorted(extrapolation_results[operation].keys())
            
            for max_val in sorted_ranges:
                range_results = extrapolation_results[operation][max_val]
                f.write(f"| 0-{max_val} | {range_results['mse']:.6f} | {range_results['accuracy']:.4f} | {range_results['percentile_50']:.4f} |\n")
            
            f.write("\n")
    
    print(f"Results saved to {output_dir}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load model
    model = load_model(args.model_path, args.device)
    
    # Load test data
    test_data = load_test_data(args.data_dir, args.device)
    
    # Extract operations
    operations = test_data['operations']
    
    # Evaluate on test data
    results = evaluate_on_test_data(model, test_data, args.batch_size, args.device)
    
    # Evaluate extrapolation performance
    extrapolation_results = evaluate_extrapolation(
        model, operations, args.extrapolation_ranges, args.num_examples, args.device
    )
    
    # Create visualizations
    plot_extrapolation_results(extrapolation_results, operations, args.output_dir)
    
    # Save results
    save_results(results, extrapolation_results, operations, args.output_dir)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
