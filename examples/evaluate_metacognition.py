#!/usr/bin/env python
"""
Metacognition Module Evaluation Script

This script evaluates a trained metacognition module on test data,
calculating metrics like Expected Calibration Error (ECE) and 
generating reliability diagrams.
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.modules.metacognition import (
        MetacognitionModule,
        plot_reliability_diagram,
        expected_calibration_error
    )
except ImportError:
    print(f"Error: Failed to import MetacognitionModule from src.modules.metacognition")
    print(f"Ensure '{PROJECT_ROOT}' is in your Python path and the file exists.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Metacognition Module Evaluation")
    
    # Model/Checkpoint Args
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained metacognition module checkpoint")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Hidden dimension (if not specified in checkpoint)")
    parser.add_argument("--intermediate_dim", type=int, default=None,
                        help="Intermediate dimension (if not specified in checkpoint)")
    
    # Data Args
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to test data file")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for evaluation")
    
    # Output Args
    parser.add_argument("--output_dir", type=str, default="outputs/metacognition_eval",
                        help="Directory to save evaluation results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate and save visualizations")
    
    # Runtime Args
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation")
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device="cpu"):
    """Load model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict and metadata
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Regular checkpoint with metadata
        model_state_dict = checkpoint["model_state_dict"]
        hidden_dim = checkpoint.get("hidden_dim")
        intermediate_dim = checkpoint.get("intermediate_dim")
        config = checkpoint.get("config", {})
        
        # Try to extract dimensions from config if not directly available
        if hidden_dim is None and config:
            hidden_dim = config.get("hidden_dim")
        if intermediate_dim is None and config:
            intermediate_dim = config.get("intermediate_dim")
        
        epoch = checkpoint.get("epoch", "unknown")
        print(f"Checkpoint from epoch: {epoch}")
        
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) and k.isdigit() for k in checkpoint.keys()):
        # Looks like a raw state dict with numerical keys
        model_state_dict = checkpoint
        hidden_dim = None
        intermediate_dim = None
        print("Loaded raw state dictionary")
        
    else:
        # Assume it's a direct state dict
        model_state_dict = checkpoint
        hidden_dim = None
        intermediate_dim = None
        print("Loaded state dictionary")
    
    return model_state_dict, hidden_dim, intermediate_dim


def load_test_data(data_path, device="cpu"):
    """Load test data for evaluation."""
    print(f"Loading test data from {data_path}")
    
    test_data = torch.load(data_path, map_location=device)
    
    if isinstance(test_data, dict) and "hidden_states" in test_data and "correctness" in test_data:
        # Dataset saved as dictionary
        hidden_states = test_data["hidden_states"]
        correctness = test_data["correctness"]
        hidden_dim = hidden_states.shape[1]
        
        # Extract metadata if available
        model_info = test_data.get("model_info", "unknown")
        dataset = test_data.get("dataset", "unknown")
        
        print(f"Loaded {len(hidden_states)} examples with hidden_dim={hidden_dim}")
        print(f"Data from model: {model_info}, dataset: {dataset}")
        print(f"Percentage correct: {correctness.mean().item()*100:.2f}%")
        
    elif isinstance(test_data, tuple) and len(test_data) == 2:
        # Dataset saved as tuple (hidden_states, correctness)
        hidden_states, correctness = test_data
        hidden_dim = hidden_states.shape[1]
        
        print(f"Loaded {len(hidden_states)} examples with hidden_dim={hidden_dim}")
        print(f"Percentage correct: {correctness.mean().item()*100:.2f}%")
        
    else:
        raise ValueError(f"Unknown data format in {data_path}")
    
    # Ensure correctness has right shape
    if len(correctness.shape) == 1:
        correctness = correctness.unsqueeze(1)
    
    return hidden_states, correctness, hidden_dim


def create_model(hidden_dim, intermediate_dim=None, device="cpu"):
    """Create metacognition module for evaluation."""
    print(f"Creating MetacognitionModule with hidden_dim={hidden_dim}, intermediate_dim={intermediate_dim}")
    
    model = MetacognitionModule(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        bit_linear=False  # Use regular linear layers for evaluation
    )
    
    return model.to(device)


def evaluate_calibration(model, hidden_states, correctness, batch_size=128, device="cpu"):
    """Evaluate model calibration on test data."""
    model.eval()
    all_confidences = []
    
    print(f"Evaluating calibration on {len(hidden_states)} examples...")
    
    # Process in batches
    num_batches = (len(hidden_states) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(hidden_states))
            
            batch_hidden_states = hidden_states[start_idx:end_idx].to(device)
            
            # Get model predictions
            batch_confidences = model(batch_hidden_states)
            
            # Store confidences
            all_confidences.append(batch_confidences.cpu())
    
    # Concatenate all confidences
    all_confidences = torch.cat(all_confidences, dim=0)
    
    # Calculate metrics
    confidences_np = all_confidences.numpy().flatten()
    correctness_np = correctness.numpy().flatten()
    
    # Calculate Expected Calibration Error
    ece, bin_accuracies, bin_confidences, bin_counts = expected_calibration_error(
        confidences_np, correctness_np, n_bins=10
    )
    
    # Calculate additional metrics
    accuracy = correctness_np.mean()
    avg_confidence = confidences_np.mean()
    
    # Calculate Brier score (mean squared error)
    brier_score = np.mean((confidences_np - correctness_np) ** 2)
    
    # Calculate Area Under ROC Curve if sklearn is available
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(correctness_np, confidences_np)
    except ImportError:
        auroc = None
    
    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(confidences_np - correctness_np))
    
    print("\nCalibration Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Average Confidence: {avg_confidence:.4f}")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  Brier Score: {brier_score:.4f}")
    if auroc is not None:
        print(f"  AUROC: {auroc:.4f}")
    print(f"  Mean Absolute Error: {mae:.4f}")
    
    # Return metrics and data for plotting
    metrics = {
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "ece": ece,
        "brier_score": brier_score,
        "auroc": auroc,
        "mae": mae
    }
    
    bin_data = {
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts
    }
    
    return metrics, bin_data, all_confidences.numpy(), correctness_np


def generate_visualizations(metrics, bin_data, confidences, correctness, output_dir):
    """Generate and save visualizations for metacognition evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating visualizations in {output_dir}")
    
    # 1. Reliability Diagram
    print("  - Generating reliability diagram...")
    fig, _ = plot_reliability_diagram(
        confidences, correctness, n_bins=10,
        title=f"Reliability Diagram (ECE={metrics['ece']:.4f})"
    )
    
    if fig:
        fig.savefig(output_dir / "reliability_diagram.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    # 2. Confidence Distribution
    print("  - Generating confidence distribution plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Split by correctness
    correct_confidences = confidences[correctness == 1]
    incorrect_confidences = confidences[correctness == 0]
    
    # Plot histograms
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    
    if len(correct_confidences) > 0:
        ax.hist(correct_confidences, bins=bins, alpha=0.7, label="Correct Predictions", 
                color="green", density=True)
    
    if len(incorrect_confidences) > 0:
        ax.hist(incorrect_confidences, bins=bins, alpha=0.7, label="Incorrect Predictions", 
                color="red", density=True)
    
    ax.set_xlabel("Confidence", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("Confidence Distribution by Correctness", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    fig.savefig(output_dir / "confidence_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # 3. Bin Statistics
    print("  - Generating bin statistics plot...")
    bin_accuracies = bin_data["bin_accuracies"]
    bin_confidences = bin_data["bin_confidences"]
    bin_counts = bin_data["bin_counts"]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Define bin centers for x-axis
    n_bins = len(bin_counts)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot accuracy and confidence
    ax1.bar(bin_centers, bin_accuracies, width=1/n_bins, alpha=0.7, label="Accuracy", color="blue")
    ax1.plot(bin_centers, bin_confidences, 'ro-', label="Avg. Confidence", linewidth=2, markersize=8)
    ax1.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration", alpha=0.7)
    
    ax1.set_xlabel("Confidence Bin", fontsize=14)
    ax1.set_ylabel("Accuracy / Confidence", fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    
    # Plot sample counts on secondary y-axis
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, bin_counts, width=1/n_bins, alpha=0.2, label="Sample Count", color="gray")
    ax2.set_ylabel("Sample Count", fontsize=14)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)
    
    ax1.set_title("Bin Statistics: Accuracy, Confidence, and Sample Count", fontsize=16)
    
    fig.tight_layout()
    fig.savefig(output_dir / "bin_statistics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # 4. Calibration Error by Bin
    print("  - Generating calibration error plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate calibration error per bin
    cal_errors = np.abs(bin_accuracies - bin_confidences)
    
    # Plot calibration error
    bars = ax.bar(bin_centers, cal_errors, width=1/n_bins, alpha=0.7, color="purple")
    
    # Annotate with bin counts
    for i, (bar, count) in enumerate(zip(bars, bin_counts)):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f"n={count}", ha="center", va="bottom", fontsize=9)
    
    ax.set_xlabel("Confidence Bin", fontsize=14)
    ax.set_ylabel("Calibration Error (|Accuracy - Confidence|)", fontsize=14)
    ax.set_title(f"Calibration Error by Bin (ECE={metrics['ece']:.4f})", fontsize=16)
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print("Visualizations saved to directory:", output_dir)


def save_metrics(metrics, output_dir):
    """Save evaluation metrics to file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / "metrics.txt"
    
    with open(metrics_path, "w") as f:
        f.write("Metacognition Module Evaluation Metrics\n")
        f.write("=====================================\n\n")
        
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                f.write(f"{metric_name}: {metric_value:.6f}\n")
    
    print(f"Metrics saved to {metrics_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load checkpoint
        model_state_dict, ckpt_hidden_dim, ckpt_intermediate_dim = load_checkpoint(
            args.checkpoint_path, device=args.device
        )
        
        # Load test data
        hidden_states, correctness, data_hidden_dim = load_test_data(
            args.test_data_path, device="cpu"  # Keep data on CPU and process in batches
        )
        
        # Determine model dimensions
        hidden_dim = args.hidden_dim or ckpt_hidden_dim or data_hidden_dim
        if hidden_dim is None:
            raise ValueError("Could not determine hidden_dim. Please specify with --hidden_dim")
        
        intermediate_dim = args.intermediate_dim or ckpt_intermediate_dim
        
        # Check for dimension mismatch
        if data_hidden_dim != hidden_dim:
            print(f"Warning: Mismatch between data hidden_dim ({data_hidden_dim}) and model hidden_dim ({hidden_dim})")
            print("This will likely cause issues. Please ensure dimensions match.")
        
        # Create model
        model = create_model(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            device=args.device
        )
        
        # Load state dict
        model.load_state_dict(model_state_dict)
        
        # Evaluate model
        metrics, bin_data, confidences, correctness_np = evaluate_calibration(
            model=model,
            hidden_states=hidden_states,
            correctness=correctness,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Save metrics
        save_metrics(metrics, args.output_dir)
        
        # Generate visualizations if requested
        if args.visualize:
            generate_visualizations(
                metrics=metrics,
                bin_data=bin_data,
                confidences=confidences.flatten(),
                correctness=correctness_np,
                output_dir=args.output_dir
            )
        
        print(f"Evaluation completed successfully. Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
