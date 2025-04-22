#!/usr/bin/env python3
"""
Numerical Competence Module Evaluation Script

This script evaluates a trained Numerical Competence Module on validation
and extrapolation datasets, providing detailed accuracy metrics and visualizations.

Usage examples:
    # Evaluate trained model
    python examples/evaluate_numerical.py --checkpoint_path outputs/numerical/best_model.pt --test_data_path data/numerical/extrapolation_data.pt
    
    # Evaluate with visualizations
    python examples/evaluate_numerical.py --checkpoint_path outputs/numerical/best_model.pt --test_data_path data/numerical/extrapolation_data.pt --visualize
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.numerical import NumericalModule


class TemporaryDecoderHead(nn.Module):
    """Temporary decoder head for numerical module evaluation."""
    
    def __init__(self, hidden_dim):
        """Initialize decoder head."""
        super().__init__()
        self.decoder = nn.Linear(hidden_dim, 1)
        
    def forward(self, hidden):
        """Forward pass through decoder head."""
        return self.decoder(hidden)


class NumericalDataset(torch.utils.data.Dataset):
    """Dataset for loading numerical evaluation data."""
    
    def __init__(self, data_path, device='cpu'):
        """
        Initialize dataset by loading data from disk.
        
        Args:
            data_path: Path to numerical data
            device: Device to load data on
        """
        self.device = device
        self.load_real_data(data_path)
        
    def load_real_data(self, data_path):
        """Load data from disk."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        print(f"Loading data from {data_path}")
        data = torch.load(data_path, map_location=self.device, weights_only=False)
        
        # Extract data components
        self.h1_data = data.get('h1_data').to(self.device)
        self.h2_data = data.get('h2_data').to(self.device)
        self.h_op_data = data.get('h_op_data').to(self.device)
        self.target_data = data.get('target_data').to(self.device)
        self.original_operands = data.get('original_operands', [])
        self.operations = data.get('operations', ['add', 'subtract', 'multiply', 'divide'])
        self.hidden_dim = data.get('hidden_dim', self.h1_data.shape[1])
        
        # Update size based on loaded data
        self.size = len(self.h1_data)
        print(f"Loaded {self.size} examples with hidden dimension {self.hidden_dim}")
        
    def __len__(self):
        """Get dataset size."""
        return self.size
        
    def __getitem__(self, idx):
        """Get a single example."""
        return {
            'h1': self.h1_data[idx],
            'h2': self.h2_data[idx],
            'h_op': self.h_op_data[idx],
            'target': self.target_data[idx],
            'original_operands': self.original_operands[idx] if idx < len(self.original_operands) else None
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Numerical Competence Module")
    
    # Model configuration
    parser.add_argument("--checkpoint_path", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--test_data_path", type=str, required=True,
                      help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="outputs/numerical_evaluation",
                      help="Directory for evaluation outputs")
    
    # Evaluation configuration
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for evaluation")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualizations")
    parser.add_argument("--num_examples", type=int, default=10,
                      help="Number of examples to show in detail")
    
    # Runtime configuration
    parser.add_argument("--device", type=str, default=None,
                      help="Device to use (cuda, cpu)")
    
    # Parse and validate arguments
    args = parser.parse_args()
    
    # Infer device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def load_model(checkpoint_path, device):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load model on
        
    Returns:
        model: Loaded numerical module
        decoder_head: Loaded decoder head
        config: Model configuration
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract configuration
    config = checkpoint.get('config', {})
    hidden_dim = config.get('hidden_dim', 768)
    num_dim = config.get('num_dim', 32)
    quantize = config.get('quantize', False)
    
    # Create model
    model = NumericalModule(
        hidden_dim=hidden_dim,
        num_dim=num_dim,
        bit_linear=quantize
    )
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create and load decoder head
    decoder_head = TemporaryDecoderHead(hidden_dim).to(device)
    decoder_head.load_state_dict(checkpoint['decoder_state_dict'])
    decoder_head.eval()
    
    print(f"Model loaded successfully (hidden_dim={hidden_dim}, num_dim={num_dim}, quantize={quantize})")
    return model, decoder_head, config


def evaluate_model(model, decoder_head, test_loader, args):
    """
    Evaluate model on test data.
    
    Args:
        model: Numerical module
        decoder_head: Decoder head
        test_loader: DataLoader for test data
        args: Command line arguments
        
    Returns:
        metrics: Evaluation metrics
        examples: List of example predictions for visualization
    """
    print("Evaluating model...")
    
    # Initialize metrics
    total_samples = 0
    correct_samples = 0
    total_loss = 0
    
    # Track per-operation metrics
    operations = ['add', 'subtract', 'multiply', 'divide']
    op_correct = {op: 0 for op in operations}
    op_total = {op: 0 for op in operations}
    
    # Track errors
    all_errors = []
    relative_errors = []
    
    # Track example predictions for visualization
    example_preds = []
    
    # Loss function
    criterion = nn.MSELoss(reduction='sum')
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Extract data
            h1 = batch['h1']
            h2 = batch['h2']
            h_op = batch['h_op']
            targets = batch['target'].to(args.device).float()
            original_operands = batch.get('original_operands', None)
            
            # Forward pass through numerical module
            result_hidden, op_weights = model(h1, h2, h_op)
            
            # Forward pass through decoder head
            predictions = decoder_head(result_hidden)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            
            # Calculate accuracy with tolerance
            abs_errors = torch.abs(predictions - targets)
            rel_tolerance = torch.abs(targets) * 0.05  # 5% relative tolerance
            min_tolerance = torch.tensor(0.01, device=args.device)
            tolerance = torch.max(rel_tolerance, min_tolerance)
            
            correct_mask = (abs_errors <= tolerance).float()
            correct_samples += torch.sum(correct_mask).item()
            total_samples += len(h1)
            
            # Calculate relative errors
            rel_errors = abs_errors / (torch.abs(targets) + 1e-6)
            
            # Track errors
            all_errors.extend(abs_errors.cpu().numpy().flatten())
            relative_errors.extend(rel_errors.cpu().numpy().flatten())
            
            # Track per-operation metrics
            if original_operands:
                for i, (a, b, op) in enumerate(original_operands):
                    if op in op_total:
                        op_total[op] += 1
                        if correct_mask[i] > 0.5:
                            op_correct[op] += 1
            
            # Save examples for visualization
            if len(example_preds) < args.num_examples:
                for i in range(min(len(h1), args.num_examples - len(example_preds))):
                    if original_operands and i < len(original_operands):
                        a, b, op = original_operands[i]
                        target = targets[i].item()
                        pred = predictions[i].item()
                        error = abs_errors[i].item()
                        rel_error = rel_errors[i].item()
                        correct = correct_mask[i].item() > 0.5
                        
                        example_preds.append({
                            'a': a,
                            'b': b,
                            'op': op,
                            'target': target,
                            'prediction': pred,
                            'error': error,
                            'relative_error': rel_error,
                            'correct': correct
                        })
    
    # Calculate metrics
    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    
    # Calculate per-operation accuracy
    op_accuracy = {}
    for op in operations:
        op_accuracy[op] = op_correct[op] / op_total[op] if op_total[op] > 0 else 0
    
    # Calculate error statistics
    mean_error = np.mean(all_errors)
    median_error = np.median(all_errors)
    max_error = np.max(all_errors)
    
    mean_rel_error = np.mean(relative_errors)
    median_rel_error = np.median(relative_errors)
    max_rel_error = np.max(relative_errors)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'mean_error': mean_error,
        'median_error': median_error,
        'max_error': max_error,
        'mean_relative_error': mean_rel_error,
        'median_relative_error': median_rel_error,
        'max_relative_error': max_rel_error,
        'operation_accuracy': op_accuracy,
        'operation_counts': op_total
    }
    
    # Print metrics
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {avg_loss:.6f}")
    print(f"Mean Absolute Error: {mean_error:.4f}")
    print(f"Median Absolute Error: {median_error:.4f}")
    print(f"Maximum Absolute Error: {max_error:.4f}")
    print(f"Mean Relative Error: {mean_rel_error:.4f}")
    print(f"Median Relative Error: {median_rel_error:.4f}")
    print(f"Maximum Relative Error: {max_rel_error:.4f}")
    print("\nPer-Operation Accuracy:")
    for op in operations:
        print(f"  {op}: {op_accuracy[op]:.4f} ({op_total[op]} examples)")
    print("==========================\n")
    
    return metrics, example_preds, all_errors, relative_errors


def visualize_results(metrics, examples, all_errors, relative_errors, args):
    """
    Generate visualizations for evaluation results.
    
    Args:
        metrics: Evaluation metrics
        examples: List of example predictions
        all_errors: List of all absolute errors
        relative_errors: List of all relative errors
        args: Command line arguments
    """
    if not args.visualize:
        return
        
    print("Generating visualizations...")
    
    # Create plots directory
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Operation Accuracy Bar Chart
    plt.figure(figsize=(10, 6))
    operations = list(metrics['operation_accuracy'].keys())
    accuracies = [metrics['operation_accuracy'][op] for op in operations]
    counts = [metrics['operation_counts'][op] for op in operations]
    
    # Create bar chart
    bars = plt.bar(operations, accuracies, color='skyblue')
    
    # Add count labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={counts[i]}', ha='center', va='bottom')
    
    plt.title('Accuracy by Operation Type')
    plt.xlabel('Operation')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "operation_accuracy.png"))
    plt.close()
    
    # 2. Error Distribution Histogram
    plt.figure(figsize=(10, 6))
    
    # Absolute error histogram
    plt.subplot(1, 2, 1)
    plt.hist(all_errors, bins=50, alpha=0.7, color='blue')
    plt.axvline(metrics['mean_error'], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {metrics["mean_error"]:.3f}')
    plt.axvline(metrics['median_error'], color='green', linestyle='dashed', linewidth=2, label=f'Median: {metrics["median_error"]:.3f}')
    plt.title('Absolute Error Distribution')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Relative error histogram
    plt.subplot(1, 2, 2)
    plt.hist(relative_errors, bins=50, alpha=0.7, color='purple')
    plt.axvline(metrics['mean_relative_error'], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {metrics["mean_relative_error"]:.3f}')
    plt.axvline(metrics['median_relative_error'], color='green', linestyle='dashed', linewidth=2, label=f'Median: {metrics["median_relative_error"]:.3f}')
    plt.title('Relative Error Distribution')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "error_distribution.png"))
    plt.close()
    
    # 3. Example Predictions
    if examples:
        plt.figure(figsize=(12, len(examples) * 0.5))
        
        for i, example in enumerate(examples):
            a = example['a']
            b = example['b']
            op = example['op']
            target = example['target']
            pred = example['prediction']
            error = example['error']
            rel_error = example['relative_error']
            correct = example['correct']
            
            # Format the equation
            op_symbol = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}[op]
            equation = f"{a} {op_symbol} {b} = {target:.4f}"
            
            # Create a color bar with green for correct, red for incorrect
            color = 'green' if correct else 'red'
            
            # Plot the example
            plt.barh([i], [1], color=color, alpha=0.3)
            plt.text(0.01, i, equation, va='center', fontsize=10)
            plt.text(0.5, i, f"Prediction: {pred:.4f}", va='center', fontsize=10)
            plt.text(0.8, i, f"Error: {error:.4f} ({rel_error:.2%})", va='center', fontsize=10)
        
        plt.yticks([])
        plt.xlim(0, 1)
        plt.title('Example Predictions')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "example_predictions.png"))
        plt.close()
    
    # 4. Operation-specific error analysis
    plt.figure(figsize=(10, 8))
    
    # Create subplots for each operation
    for i, op in enumerate(operations):
        plt.subplot(2, 2, i+1)
        
        # Extract errors for this operation
        op_errors = []
        for example in examples:
            if example['op'] == op:
                op_errors.append(example['error'])
                
        if op_errors:
            plt.hist(op_errors, bins=20, alpha=0.7, color=f'C{i}')
            mean_error = np.mean(op_errors) if op_errors else 0
            plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_error:.3f}')
            plt.title(f'{op.capitalize()} Operation Errors')
            plt.xlabel('Absolute Error')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(alpha=0.3)
        else:
            plt.text(0.5, 0.5, f"No {op} examples", ha='center', va='center')
            plt.title(f'{op.capitalize()} Operation Errors')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "operation_errors.png"))
    plt.close()
    
    print(f"Visualizations saved to {plots_dir}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load model
    model, decoder_head, config = load_model(args.checkpoint_path, args.device)
    
    # Load test data
    try:
        test_dataset = NumericalDataset(args.test_data_path, args.device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False  # Disable pin_memory to avoid CUDA tensor issue
    )
    
    # Evaluate model
    metrics, examples, all_errors, relative_errors = evaluate_model(model, decoder_head, test_loader, args)
    
    # Generate visualizations
    visualize_results(metrics, examples, all_errors, relative_errors, args)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.pt")
    torch.save(metrics, metrics_path, weights_only=False)
    print(f"Metrics saved to {metrics_path}")
    
    # Save evaluation summary
    summary_path = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== Numerical Competence Module Evaluation ===\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Test Data: {args.test_data_path}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Date: {torch.datetime.datetime.now()}\n")
        f.write("\n=== Metrics ===\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Loss: {metrics['loss']:.6f}\n")
        f.write(f"Mean Absolute Error: {metrics['mean_error']:.4f}\n")
        f.write(f"Median Absolute Error: {metrics['median_error']:.4f}\n")
        f.write(f"Maximum Absolute Error: {metrics['max_error']:.4f}\n")
        f.write(f"Mean Relative Error: {metrics['mean_relative_error']:.4f}\n")
        f.write(f"Median Relative Error: {metrics['median_relative_error']:.4f}\n")
        f.write(f"Maximum Relative Error: {metrics['max_relative_error']:.4f}\n")
        f.write("\n=== Per-Operation Accuracy ===\n")
        for op, acc in metrics['operation_accuracy'].items():
            f.write(f"{op}: {acc:.4f} ({metrics['operation_counts'][op]} examples)\n")
        f.write("\n=== Example Predictions ===\n")
        for i, example in enumerate(examples):
            a = example['a']
            b = example['b']
            op = example['op']
            target = example['target']
            pred = example['prediction']
            error = example['error']
            rel_error = example['relative_error']
            correct = example['correct']
            
            op_symbol = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}[op]
            equation = f"{a} {op_symbol} {b} = {target:.4f}"
            
            f.write(f"Example {i+1}: {equation}\n")
            f.write(f"  Prediction: {pred:.4f}\n")
            f.write(f"  Error: {error:.4f} ({rel_error:.2%})\n")
            f.write(f"  Correct: {correct}\n")
            f.write("\n")
    
    print(f"Evaluation summary saved to {summary_path}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
