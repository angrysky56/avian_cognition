#!/usr/bin/env python3
"""
Numerical Module Evaluation Script (Fixed Version)

This script evaluates a trained NumericalModule on various arithmetic operations,
with a focus on extrapolation capabilities and visualizing the results.
It includes improvements for CUDA compatibility and error handling.

Usage:
    python scripts/evaluate_numerical_module_fixed.py \
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
import traceback

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
        default=32,  # Reduced from 128 to avoid CUDA issues
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,  # Set to 0 to avoid CUDA issues
        help="Number of worker processes for data loading"
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
    
    # Add necessary global for NumPy scalar types (if needed)
    try:
        from numpy._core.multiarray import scalar
        import torch.serialization
        torch.serialization.add_safe_globals([scalar])
    except ImportError:
        print("WARNING: Could not import numpy._core.multiarray.scalar, attempting load with weights_only=False")
    
    # Load checkpoint with error handling
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except (RuntimeError, TypeError) as e:
        print(f"Error with weights_only=False: {e}")
        print("Trying legacy loading method...")
        try:
            import pickle
            checkpoint = torch.load(model_path, map_location=device, pickle_module=pickle)
        except Exception as e2:
            print(f"Error with pickle loading: {e2}")
            # Last resort - try with weights_only=True
            print("Trying weights_only=True...")
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
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
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If no 'model_state_dict' key, assume the checkpoint is the state dict itself
        model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model.eval()
    
    print("Model loaded successfully")
    return model


def generate_test_data(operations, value_range, num_examples, hidden_dim, device):
    """
    Generate test data for evaluation.
    
    Args:
        operations: List of operations to test
        value_range: Range of values for operands (min_val, max_val)
        num_examples: Number of examples to generate
        hidden_dim: Hidden dimension size
        device: Device to place tensors on
        
    Returns:
        test_data: Dictionary of test data
    """
    print(f"Generating test data with {num_examples} examples per operation...")
    
    min_val, max_val = value_range
    total_examples = num_examples * len(operations)
    
    # Initialize storage tensors
    h1_data = torch.zeros(total_examples, hidden_dim, device=device)
    h2_data = torch.zeros(total_examples, hidden_dim, device=device)
    h_op_data = torch.zeros(total_examples, hidden_dim, device=device)
    target_data = torch.zeros(total_examples, 1, device=device)
    
    # Track original operands and operation labels
    original_operands = []
    operation_labels = []
    
    # Process each operation
    idx = 0
    for op_idx, operation in enumerate(operations):
        for _ in range(num_examples):
            # Generate operands
            a = np.random.randint(min_val, max_val)
            
            # For division, ensure we don't divide by zero
            if operation == 'divide':
                b = max(1, np.random.randint(min_val, max_val))
            else:
                b = np.random.randint(min_val, max_val)
                
            original_operands.append((a, b))
            operation_labels.append(operation)
            
            # Compute result
            if operation == 'add':
                result = a + b
            elif operation == 'subtract':
                result = a - b
            elif operation == 'multiply':
                result = a * b
            elif operation == 'divide':
                result = a / b
            
            # Find scaling factor (maximum possible value in this range)
            if operation == 'add':
                max_result = 2 * max_val
            elif operation == 'subtract':
                max_result = max_val
            elif operation == 'multiply':
                max_result = max_val * max_val
            elif operation == 'divide':
                max_result = max_val  # Simplification
                
            # Use a scaling factor to normalize
            scaling_factor = 1.0 / max(1.0, max_result)
            
            # Create hidden state representations (simple encoding)
            h1_data[idx, 0] = a * scaling_factor
            h1_data[idx, 1] = (a // 100) * scaling_factor
            h1_data[idx, 2] = ((a % 100) // 10) * scaling_factor
            h1_data[idx, 3] = (a % 10) * scaling_factor
            
            h2_data[idx, 0] = b * scaling_factor
            h2_data[idx, 1] = (b // 100) * scaling_factor
            h2_data[idx, 2] = ((b % 100) // 10) * scaling_factor
            h2_data[idx, 3] = (b % 10) * scaling_factor
            
            # One-hot operation encoding
            h_op_data[idx, op_idx] = 1.0
            
            # Store scaled result
            target_data[idx, 0] = result * scaling_factor
            
            idx += 1
    
    # Return data
    test_data = {
        'h1_data': h1_data,
        'h2_data': h2_data,
        'h_op_data': h_op_data,
        'target_data': target_data,
        'original_operands': original_operands,
        'operations': operation_labels,
        'scaling_factor': scaling_factor,
        'value_range': value_range
    }
    
    return test_data


def evaluate_model(model, h1, h2, h_op, targets, batch_size=32, device="cpu"):
    """
    Evaluate model performance on given data without using DataLoader.
    
    Args:
        model: NumericalModule model
        h1: First operand hidden states [num_examples, hidden_dim]
        h2: Second operand hidden states [num_examples, hidden_dim]
        h_op: Operation hidden states [num_examples, hidden_dim]
        targets: Target values [num_examples, 1]
        batch_size: Batch size for processing
        device: Device for computation
        
    Returns:
        results: Dictionary of evaluation results
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Move data to device if needed
    h1 = h1.to(device)
    h2 = h2.to(device)
    h_op = h_op.to(device)
    targets = targets.to(device)
    
    # Initialize storage for predictions
    num_examples = h1.size(0)
    all_predictions = []
    
    # Process in batches to avoid memory issues
    with torch.no_grad():
        for i in tqdm(range(0, num_examples, batch_size), desc="Evaluating"):
            # Get batch
            batch_h1 = h1[i:i+batch_size]
            batch_h2 = h2[i:i+batch_size]
            batch_h_op = h_op[i:i+batch_size]
            
            # Forward pass
            result_hidden, op_weights = model(batch_h1, batch_h2, batch_h_op)
            
            # Extract predictions (using the first dimension)
            predictions = result_hidden[:, 0:1]
            all_predictions.append(predictions.cpu())
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions)
    all_targets = targets.cpu()
    
    # Calculate metrics
    mse = nn.MSELoss()(all_predictions, all_targets).item()
    
    # Calculate absolute errors
    abs_errors = torch.abs(all_predictions - all_targets).squeeze()
    
    # Calculate accuracy with 5% tolerance
    rel_tolerance = torch.abs(all_targets) * 0.05
    rel_tolerance = torch.max(rel_tolerance, torch.tensor(1e-2))
    within_tolerance = (abs_errors <= rel_tolerance.squeeze()).float().mean().item()
    
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


def evaluate_per_operation(model, test_data, batch_size=32, device="cpu"):
    """
    Evaluate model performance separately for each operation.
    
    Args:
        model: NumericalModule model
        test_data: Test data dictionary
        batch_size: Batch size for processing
        device: Device for computation
        
    Returns:
        op_results: Dictionary of per-operation results
    """
    print("Evaluating model performance per operation...")
    
    # Group data by operation
    operations = ['add', 'subtract', 'multiply', 'divide']
    op_indices = {op: [] for op in operations}
    
    # Find indices for each operation
    for i, op in enumerate(test_data['operations']):
        if op in op_indices:
            op_indices[op].append(i)
            
    # Evaluate each operation separately
    op_results = {}
    for op in operations:
        if not op_indices[op]:
            print(f"No examples found for operation: {op}")
            continue
            
        # Get data for this operation
        indices = torch.tensor(op_indices[op])
        h1 = test_data['h1_data'][indices]
        h2 = test_data['h2_data'][indices]
        h_op = test_data['h_op_data'][indices]
        targets = test_data['target_data'][indices]
        
        # Evaluate
        print(f"\nEvaluating operation: {op}")
        results = evaluate_model(model, h1, h2, h_op, targets, batch_size, device)
        op_results[op] = results
        
        # Print some examples
        n_examples = min(5, len(indices))
        for i in range(n_examples):
            idx = indices[i].item()
            a, b = test_data['original_operands'][idx]
            pred = results['predictions'][i].item() / test_data['scaling_factor']
            targ = results['targets'][i].item() / test_data['scaling_factor']
            error = abs(pred - targ)
            print(f"  {a} {op} {b} = {pred:.2f} (correct: {targ:.2f}, error: {error:.2f})")
    
    # Calculate overall metrics
    all_preds = torch.cat([op_results[op]['predictions'] for op in operations if op in op_results])
    all_targets = torch.cat([op_results[op]['targets'] for op in operations if op in op_results])
    
    mse = nn.MSELoss()(all_preds, all_targets).item()
    
    abs_errors = torch.abs(all_preds - all_targets).squeeze()
    rel_tolerance = torch.abs(all_targets) * 0.05
    rel_tolerance = torch.max(rel_tolerance, torch.tensor(1e-2))
    within_tolerance = (abs_errors <= rel_tolerance.squeeze()).float().mean().item()
    
    # Add overall results
    op_results['overall'] = {
        'mse': mse,
        'accuracy': within_tolerance
    }
    
    # Print summary
    print("\nPer-Operation Performance:")
    for op in operations:
        if op in op_results:
            acc = op_results[op]['accuracy']
            print(f"  {op.capitalize()}: Accuracy = {acc:.4f}")
    print(f"  Overall: Accuracy = {within_tolerance:.4f}")
    
    return op_results


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
            
            # Generate test data for this operation and range
            test_data = generate_test_data(
                [operation], value_range, num_examples // len(operations), hidden_dim, device
            )
            
            # Evaluate model on this data
            print(f"\nEvaluating {operation} on range 0-{max_val}")
            results = evaluate_model(
                model, 
                test_data['h1_data'], 
                test_data['h2_data'], 
                test_data['h_op_data'], 
                test_data['target_data'],
                batch_size=32,
                device=device
            )
            
            # Extract metrics
            mse = results['mse']
            accuracy = results['accuracy']
            abs_errors = results['abs_errors'].numpy()
            
            # Store results
            range_results = {
                'mse': mse,
                'accuracy': accuracy,
                'abs_errors': abs_errors.tolist(),
                'percentile_25': np.percentile(abs_errors, 25),
                'percentile_50': np.percentile(abs_errors, 50),
                'percentile_75': np.percentile(abs_errors, 75),
                'percentile_95': np.percentile(abs_errors, 95),
                'range': value_range
            }
            
            operation_results[max_val] = range_results
            
            # Print some examples
            n_examples = min(5, len(test_data['original_operands']))
            for i in range(n_examples):
                a, b = test_data['original_operands'][i]
                pred = results['predictions'][i].item() / test_data['scaling_factor']
                targ = results['targets'][i].item() / test_data['scaling_factor']
                error = abs(pred - targ)
                print(f"  {a} {operation} {b} = {pred:.2f} (correct: {targ:.2f}, error: {error:.2f})")
            
            print(f"  {operation.capitalize()} Range 0-{max_val}: MSE={mse:.6f}, Accuracy={accuracy:.4f}")
        
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
        if operation not in extrapolation_results:
            continue
        
        # Extract ranges and accuracies
        ranges = []
        accuracies = []
        
        for max_val, results in extrapolation_results[operation].items():
            ranges.append(max_val)
            accuracies.append(results['accuracy'])
        
        # Sort by range
        sorted_data = sorted(zip(ranges, accuracies))
        ranges, accuracies = zip(*sorted_data) if sorted_data else ([], [])
        
        # Plot
        if ranges and accuracies:
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
        if operation not in extrapolation_results:
            continue
        
        # Extract ranges and median errors
        ranges = []
        median_errors = []
        
        for max_val, results in extrapolation_results[operation].items():
            ranges.append(max_val)
            median_errors.append(results['percentile_50'])
        
        # Sort by range
        sorted_data = sorted(zip(ranges, median_errors))
        ranges, median_errors = zip(*sorted_data) if sorted_data else ([], [])
        
        # Plot
        if ranges and median_errors:
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
        if operation not in extrapolation_results:
            continue
            
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
        ranges = [ranges[i] for i in sorted_indices] if ranges else []
        p25_errors = [p25_errors[i] for i in sorted_indices] if p25_errors else []
        p50_errors = [p50_errors[i] for i in sorted_indices] if p50_errors else []
        p75_errors = [p75_errors[i] for i in sorted_indices] if p75_errors else []
        p95_errors = [p95_errors[i] for i in sorted_indices] if p95_errors else []
        
        # Plot
        if ranges:
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
    
    # Create results path
    results_path = os.path.join(output_dir, "evaluation_results.pt")
    
    # Save with try/except for error handling
    try:
        torch.save(results, results_path)
    except Exception as e:
        print(f"Error saving results: {e}")
        # Try with pickle
        try:
            import pickle
            torch.save(results, results_path, pickle_module=pickle)
        except Exception as e2:
            print(f"Error saving with pickle: {e2}")
            # Last resort - use numpy
            np.save(os.path.join(output_dir, "evaluation_results.npy"), {
                'accuracy': results['overall']['accuracy'],
                'mse': results['overall']['mse']
            })
    
    # Save extrapolation results
    extrapolation_path = os.path.join(output_dir, "extrapolation_results.pt")
    
    # Save with try/except for error handling
    try:
        torch.save(extrapolation_results, extrapolation_path)
    except Exception as e:
        print(f"Error saving extrapolation results: {e}")
        # Try with pickle
        try:
            import pickle
            torch.save(extrapolation_results, extrapolation_path, pickle_module=pickle)
        except Exception as e2:
            print(f"Error saving with pickle: {e2}")
            # Skip saving if all methods fail
    
    # Create a text report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("# NumericalModule Evaluation Report\n\n")
        
        # Standard evaluation
        f.write("## Standard Evaluation\n")
        f.write(f"Overall Accuracy (5% tolerance): {results['overall']['accuracy']:.4f}\n")
        f.write(f"Overall MSE: {results['overall']['mse']:.6f}\n\n")
        
        # Per-operation performance
        f.write("## Per-Operation Performance\n\n")
        f.write("| Operation | Accuracy | MSE |\n")
        f.write("|-----------|----------|-----|\n")
        
        for op in operations:
            if op in results:
                acc = results[op]['accuracy']
                mse = results[op]['mse']
                f.write(f"| {op.capitalize()} | {acc:.4f} | {mse:.6f} |\n")
        
        # Extrapolation evaluation
        f.write("\n## Extrapolation Evaluation\n\n")
        
        for operation in operations:
            if operation not in extrapolation_results:
                continue
                
            f.write(f"### {operation.capitalize()}\n\n")
            f.write("| Range | Accuracy | MSE | Median Error |\n")
            f.write("|-------|----------|-----|-------------|\n")
            
            # Sort ranges
            sorted_ranges = sorted(extrapolation_results[operation].keys())
            
            for max_val in sorted_ranges:
                range_results = extrapolation_results[operation][max_val]
                f.write(f"| 0-{max_val} | {range_results['accuracy']:.4f} | {range_results['mse']:.6f} | {range_results['percentile_50']:.4f} |\n")
            
            f.write("\n")
    
    print(f"Results saved to {output_dir}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Load model
        model = load_model(args.model_path, args.device)
        
        # Define operations
        operations = ['add', 'subtract', 'multiply', 'divide']
        
        # Generate or load test data
        print("\n1. Evaluating standard performance:")
        test_data = generate_test_data(
            operations,
            value_range=(0, 100),  # In-distribution range
            num_examples=args.num_examples // len(operations),
            hidden_dim=model.hidden_dim,
            device='cpu'  # Generate on CPU then transfer as needed
        )
        
        # Evaluate per-operation
        results = evaluate_per_operation(
            model, 
            test_data, 
            batch_size=args.batch_size, 
            device=args.device
        )
        
        # Evaluate extrapolation
        print("\n2. Evaluating extrapolation performance:")
        extrapolation_results = evaluate_extrapolation(
            model,
            operations,
            args.extrapolation_ranges,
            args.num_examples,
            args.device
        )
        
        # Create visualizations
        plot_extrapolation_results(extrapolation_results, operations, args.output_dir)
        
        # Save results
        save_results(results, extrapolation_results, operations, args.output_dir)
        
        print("Evaluation complete!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Detailed traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
