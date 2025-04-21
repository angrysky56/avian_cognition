#!/usr/bin/env python
"""
Bayesian Module Evaluation Script

This script evaluates a trained Bayesian inference module on test data,
measuring how well it updates beliefs based on sequential evidence.
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
    from src.modules.bayesian import BayesianInferenceModule, kl_divergence_loss
except ImportError:
    print(f"Error: Failed to import BayesianInferenceModule from src.modules.bayesian")
    print(f"Ensure '{PROJECT_ROOT}' is in your Python path and the file exists.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bayesian Module Evaluation")
    
    # Model/Checkpoint Args
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained Bayesian module checkpoint")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Hidden dimension (if not specified in checkpoint)")
    parser.add_argument("--belief_dim", type=int, default=None,
                        help="Belief state dimension (if not specified in checkpoint)")
    parser.add_argument("--num_hypotheses", type=int, default=None,
                        help="Number of hypotheses (if not specified in checkpoint)")
    
    # Data Args
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to test data file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    
    # Output Args
    parser.add_argument("--output_dir", type=str, default="outputs/bayesian_eval",
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
        belief_dim = checkpoint.get("belief_dim")
        num_hypotheses = checkpoint.get("num_hypotheses")
        config = checkpoint.get("config", {})
        
        # Try to extract dimensions from config if not directly available
        if hidden_dim is None and config:
            hidden_dim = config.get("hidden_dim")
        if belief_dim is None and config:
            belief_dim = config.get("belief_dim")
        if num_hypotheses is None and config:
            num_hypotheses = config.get("num_hypotheses")
        
        epoch = checkpoint.get("epoch", "unknown")
        print(f"Checkpoint from epoch: {epoch}")
        
    else:
        # Assume it's a direct state dict
        model_state_dict = checkpoint
        hidden_dim = None
        belief_dim = None
        num_hypotheses = None
        print("Loaded state dictionary")
    
    return model_state_dict, hidden_dim, belief_dim, num_hypotheses


def load_test_data(data_path, device="cpu"):
    """Load test data for evaluation."""
    print(f"Loading test data from {data_path}")
    
    test_data = torch.load(data_path, map_location=device)
    
    if isinstance(test_data, dict):
        # Extract data and metadata
        sequences = test_data.get("sequences")
        posteriors = test_data.get("posteriors")
        hidden_dim = test_data.get("hidden_dim")
        num_hypotheses = test_data.get("num_hypotheses")
        
        # Extract metadata if available
        model_info = test_data.get("model_info", "unknown")
        dataset = test_data.get("dataset", "unknown")
        
        print(f"Loaded {len(sequences)} sequences with hidden_dim={hidden_dim}")
        print(f"Data from model: {model_info}, dataset: {dataset}")
    else:
        raise ValueError(f"Unknown data format in {data_path}")
    
    return sequences, posteriors, hidden_dim, num_hypotheses


def create_model(hidden_dim, belief_dim, num_hypotheses, device="cpu"):
    """Create Bayesian inference module for evaluation."""
    print(f"Creating BayesianInferenceModule with hidden_dim={hidden_dim}, "
          f"belief_dim={belief_dim}, num_hypotheses={num_hypotheses}")
    
    model = BayesianInferenceModule(
        hidden_dim=hidden_dim,
        belief_dim=belief_dim,
        num_hypotheses=num_hypotheses
    )
    
    return model.to(device)


def evaluate_belief_updating(model, sequences, posteriors, batch_size=32, device="cpu"):
    """Evaluate model's belief updating performance on test data."""
    model.eval()
    
    num_sequences = sequences.shape[0]
    seq_length = sequences.shape[1]
    num_hypotheses = posteriors.shape[2]
    
    # Metrics
    total_kl_div = 0.0
    total_accuracy = 0.0
    all_posteriors_model = []
    all_posteriors_true = []
    
    # Process in batches
    num_batches = (num_sequences + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_sequences)
            batch_size_actual = end_idx - start_idx
            
            batch_sequences = sequences[start_idx:end_idx].to(device)
            batch_posteriors = posteriors[start_idx:end_idx].to(device)
            
            # Process each sequence
            batch_posterior_model = torch.zeros(batch_size_actual, seq_length, num_hypotheses, device=device)
            
            for batch_idx in range(batch_size_actual):
                belief_state = None  # Start with no prior
                
                for seq_idx in range(seq_length):
                    # Get current evidence for this item in the batch
                    evidence = batch_sequences[batch_idx, seq_idx]
                    
                    # Update belief with evidence
                    belief_state, belief_embedding = model(evidence, belief_state)
                    
                    # Extract posterior from belief state
                    posterior_model = model.belief_to_posterior(belief_state)
                    
                    # Store the posterior
                    batch_posterior_model[batch_idx, seq_idx] = posterior_model
            
            # Compute KL divergence between model posteriors and true posteriors
            kl_div = 0.0
            for seq_idx in range(seq_length):
                kl_batch = kl_divergence_loss(
                    batch_posterior_model[:, seq_idx], 
                    batch_posteriors[:, seq_idx]
                )
                kl_div += kl_batch.item()
            
            kl_div /= seq_length
            total_kl_div += kl_div * batch_size_actual
            
            # Compute accuracy based on most probable hypothesis
            batch_accuracy = 0.0
            for seq_idx in range(seq_length):
                model_pred = batch_posterior_model[:, seq_idx].argmax(dim=1)
                true_pred = batch_posteriors[:, seq_idx].argmax(dim=1)
                batch_accuracy += (model_pred == true_pred).float().mean().item()
            
            batch_accuracy /= seq_length
            total_accuracy += batch_accuracy * batch_size_actual
            
            # Save posteriors for visualization
            all_posteriors_model.append(batch_posterior_model.cpu())
            all_posteriors_true.append(batch_posteriors.cpu())
    
    # Compute average metrics
    avg_kl_div = total_kl_div / num_sequences
    avg_accuracy = total_accuracy / num_sequences
    
    # Concat posteriors for all batches
    all_posteriors_model = torch.cat(all_posteriors_model, dim=0)
    all_posteriors_true = torch.cat(all_posteriors_true, dim=0)
    
    print("\nEvaluation Metrics:")
    print(f"  Average KL Divergence: {avg_kl_div:.4f}")
    print(f"  Average Accuracy: {avg_accuracy:.4f}")
    
    return {
        "kl_divergence": avg_kl_div,
        "accuracy": avg_accuracy,
        "posteriors_model": all_posteriors_model,
        "posteriors_true": all_posteriors_true
    }


def generate_visualizations(metrics, output_dir):
    """Generate and save visualizations of belief updating."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    posteriors_model = metrics["posteriors_model"]
    posteriors_true = metrics["posteriors_true"]
    
    num_sequences = posteriors_model.shape[0]
    seq_length = posteriors_model.shape[1]
    num_hypotheses = posteriors_model.shape[2]
    
    print(f"Generating visualizations in {output_dir}")
    
    # 1. Plot belief trajectories for a few example sequences
    num_examples = min(5, num_sequences)
    
    for i in range(num_examples):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot model posterior
        for h in range(num_hypotheses):
            axes[0].plot(
                range(seq_length), 
                posteriors_model[i, :, h], 
                marker='o',
                label=f"Hypothesis {h}"
            )
        
        axes[0].set_title("Model Posterior Probabilities")
        axes[0].set_xlabel("Evidence Step")
        axes[0].set_ylabel("Probability")
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot true posterior
        for h in range(num_hypotheses):
            axes[1].plot(
                range(seq_length), 
                posteriors_true[i, :, h], 
                marker='o',
                label=f"Hypothesis {h}"
            )
        
        axes[1].set_title("True Posterior Probabilities")
        axes[1].set_xlabel("Evidence Step")
        axes[1].set_ylabel("Probability")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"belief_trajectory_{i}.png", dpi=300)
        plt.close()
    
    # 2. Plot average KL divergence per evidence step
    kl_per_step = torch.zeros(seq_length)
    
    for i in range(seq_length):
        kl_per_step[i] = kl_divergence_loss(
            posteriors_model[:, i].flatten(0, 1), 
            posteriors_true[:, i].flatten(0, 1)
        ).item()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(seq_length), kl_per_step, marker='o', linewidth=2)
    ax.set_title("KL Divergence per Evidence Step")
    ax.set_xlabel("Evidence Step")
    ax.set_ylabel("KL Divergence")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "kl_divergence_per_step.png", dpi=300)
    plt.close()
    
    # 3. Plot accuracy per evidence step
    accuracy_per_step = torch.zeros(seq_length)
    
    for i in range(seq_length):
        model_pred = posteriors_model[:, i].argmax(dim=1)
        true_pred = posteriors_true[:, i].argmax(dim=1)
        accuracy_per_step[i] = (model_pred == true_pred).float().mean().item()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(seq_length), accuracy_per_step, marker='o', linewidth=2, color='green')
    ax.set_title("Accuracy per Evidence Step")
    ax.set_xlabel("Evidence Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_per_step.png", dpi=300)
    plt.close()
    
    print("Visualizations saved to directory:", output_dir)


def save_metrics(metrics, output_dir):
    """Save evaluation metrics to a file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / "metrics.txt"
    
    with open(metrics_path, "w") as f:
        f.write("Bayesian Module Evaluation Metrics\n")
        f.write("=====================================\n\n")
        
        f.write(f"KL Divergence: {metrics['kl_divergence']:.6f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.6f}\n")
    
    print(f"Metrics saved to {metrics_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load checkpoint
        model_state_dict, ckpt_hidden_dim, ckpt_belief_dim, ckpt_num_hypotheses = load_checkpoint(
            args.checkpoint_path, device=args.device
        )
        
        # Load test data
        sequences, posteriors, data_hidden_dim, data_num_hypotheses = load_test_data(
            args.test_data_path, device="cpu"  # Keep data on CPU and process in batches
        )
        
        # Determine model dimensions
        hidden_dim = args.hidden_dim or ckpt_hidden_dim or data_hidden_dim
        num_hypotheses = args.num_hypotheses or ckpt_num_hypotheses or data_num_hypotheses
        belief_dim = args.belief_dim or ckpt_belief_dim or (2 * num_hypotheses)  # Default if not specified
        
        if hidden_dim is None or num_hypotheses is None:
            raise ValueError("Could not determine hidden_dim or num_hypotheses. Please specify with args.")
        
        # Check for dimension mismatches
        if data_hidden_dim != hidden_dim:
            print(f"Warning: Mismatch between data hidden_dim ({data_hidden_dim}) and model hidden_dim ({hidden_dim})")
        
        # Create model
        model = create_model(
            hidden_dim=hidden_dim,
            belief_dim=belief_dim,
            num_hypotheses=num_hypotheses,
            device=args.device
        )
        
        # Load state dict
        model.load_state_dict(model_state_dict)
        
        # Evaluate model
        metrics = evaluate_belief_updating(
            model=model,
            sequences=sequences,
            posteriors=posteriors,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Save metrics
        save_metrics(metrics, args.output_dir)
        
        # Generate visualizations if requested
        if args.visualize:
            generate_visualizations(metrics, args.output_dir)
        
        print(f"Evaluation completed successfully. Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
