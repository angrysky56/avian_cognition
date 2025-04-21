"""
BitNet Testing Module

This module demonstrates the use of BitNet ternary quantization on 
neural network models for the Avian Cognition project.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

from src.core.bitnet import BitLinear, convert_linear_to_bit_linear

class SimpleNetwork(nn.Module):
    """
    A simple multi-layer network for testing BitNet quantization.
    """
    def __init__(self, input_dim=256, hidden_dims=[512, 256, 128], output_dim=10):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
            
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def create_test_networks(input_dim=256, hidden_dims=[512, 256, 128], output_dim=10):
    """
    Create two identical networks, one with standard Linear layers and one with BitLinear.
    """
    # Create standard network
    standard_net = SimpleNetwork(input_dim, hidden_dims, output_dim)
    
    # Create BitNet network (copy of standard network)
    bitnet = SimpleNetwork(input_dim, hidden_dims, output_dim)
    
    # Copy weights from standard to bitnet
    bitnet.load_state_dict(standard_net.state_dict())
    
    # Convert linear layers to BitLinear
    convert_linear_to_bit_linear(bitnet)
    
    return standard_net, bitnet


def analyze_weight_distribution(model):
    """Analyze weight distribution of BitLinear layers in the model."""
    stats = {
        "total_weights": 0,
        "linear_weights": 0,
        "ternary_counts": {-1: 0, 0: 0, 1: 0},
        "layer_stats": []
    }
    
    # Analyze each layer
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            # Get absolute mean and threshold for this layer
            abs_mean = torch.mean(torch.abs(module.weight), dim=1, keepdim=True)
            threshold = 0.5 * abs_mean
            
            # Count values after ternary quantization
            weight_abs = torch.abs(module.weight)
            zeros = (weight_abs <= threshold).sum().item()
            pos_ones = ((weight_abs > threshold) & (module.weight > 0)).sum().item()
            neg_ones = ((weight_abs > threshold) & (module.weight < 0)).sum().item()
            
            # Update counts
            stats["linear_weights"] += module.weight.numel()
            stats["ternary_counts"][-1] += neg_ones
            stats["ternary_counts"][0] += zeros
            stats["ternary_counts"][1] += pos_ones
            
            # Store layer-specific stats
            layer_stats = {
                "name": name,
                "shape": tuple(module.weight.shape),
                "params": module.weight.numel(),
                "distribution": {
                    -1: neg_ones / module.weight.numel() * 100,
                    0: zeros / module.weight.numel() * 100,
                    1: pos_ones / module.weight.numel() * 100
                }
            }
            stats["layer_stats"].append(layer_stats)
    
    # Calculate total weights
    stats["total_weights"] = sum(p.numel() for p in model.parameters())
    
    return stats


def compare_networks(standard_net, bitnet, input_dim=256, batch_size=32, num_runs=10):
    """
    Compare standard and BitNet networks for inference speed and output differences.
    """
    device = next(standard_net.parameters()).device
    
    # Create random input
    x = torch.randn(batch_size, input_dim, device=device)
    
    # Measure inference time for standard network
    standard_times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            standard_output = standard_net(x)
        standard_times.append((time.time() - start_time) * 1000)  # Convert to ms
    
    # Measure inference time for BitNet network
    bitnet_times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            bitnet_output = bitnet(x)
        bitnet_times.append((time.time() - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    standard_mean = np.mean(standard_times)
    standard_std = np.std(standard_times)
    bitnet_mean = np.mean(bitnet_times)
    bitnet_std = np.std(bitnet_times)
    
    # Compare outputs
    output_diff = torch.abs(standard_output - bitnet_output)
    mean_diff = output_diff.mean().item()
    max_diff = output_diff.max().item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        standard_output.flatten(), bitnet_output.flatten(), dim=0
    ).item()
    
    return {
        "timing": {
            "standard": {
                "mean": standard_mean,
                "std": standard_std
            },
            "bitnet": {
                "mean": bitnet_mean,
                "std": bitnet_std
            },
            "speedup": standard_mean / bitnet_mean
        },
        "output_diff": {
            "mean": mean_diff,
            "max": max_diff,
            "cosine_sim": cosine_sim
        }
    }


def visualize_results(stats, timing_results, output_dir="outputs/bitnet_experiment"):
    """
    Generate visualizations of BitNet performance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ternary Weight Distribution
    ternary_values = list(stats["ternary_counts"].keys())
    ternary_counts = list(stats["ternary_counts"].values())
    total_ternary_weights = sum(ternary_counts)
    ternary_percentages = [count / total_ternary_weights * 100 for count in ternary_counts]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ternary_values, ternary_percentages, color=['red', 'gray', 'blue'])
    
    # Add value labels on bars
    for bar, count, percentage in zip(bars, ternary_counts, ternary_percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count:,}\n({percentage:.1f}%)',
                ha='center', va='bottom')
    
    ax.set_xticks(ternary_values)
    ax.set_xticklabels(['-1', '0', '+1'])
    ax.set_title('BitNet b1.58 Ternary Weight Distribution')
    ax.set_ylabel('Percentage of weights')
    ax.set_ylim(0, max(ternary_percentages) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    fig.savefig(output_dir / "ternary_distribution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Inference Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Standard Linear', 'BitNet b1.58']
    mean_times = [
        timing_results['timing']['standard']['mean'], 
        timing_results['timing']['bitnet']['mean']
    ]
    std_times = [
        timing_results['timing']['standard']['std'], 
        timing_results['timing']['bitnet']['std']
    ]
    
    # Calculate speedup
    speedup = timing_results['timing']['speedup']
    
    bars = ax.bar(models, mean_times, yerr=std_times, capsize=10, color=['blue', 'orange'])
    
    # Add value labels on bars
    for i, (bar, time_ms) in enumerate(zip(bars, mean_times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time_ms:.2f} ms',
                ha='center', va='bottom')
    
    ax.set_title(f'Inference Time Comparison (BitNet is {speedup:.2f}x {"faster" if speedup > 1 else "slower"})')
    ax.set_ylabel('Time (ms)')
    ax.set_ylim(0, max(mean_times) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    fig.savefig(output_dir / "inference_time.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save results as text
    with open(output_dir / "results.txt", "w") as f:
        f.write("BitNet b1.58 Experiment Results\n")
        f.write("=============================\n\n")
        
        f.write("Weight Distribution:\n")
        f.write(f"  Total weights: {stats['linear_weights']:,}\n")
        f.write(f"  -1: {stats['ternary_counts'][-1]:,} ({ternary_percentages[0]:.2f}%)\n")
        f.write(f"   0: {stats['ternary_counts'][0]:,} ({ternary_percentages[1]:.2f}%)\n")
        f.write(f"  +1: {stats['ternary_counts'][1]:,} ({ternary_percentages[2]:.2f}%)\n\n")
        
        f.write("Inference Time:\n")
        f.write(f"  Standard: {mean_times[0]:.2f} ± {std_times[0]:.2f} ms\n")
        f.write(f"  BitNet: {mean_times[1]:.2f} ± {std_times[1]:.2f} ms\n")
        f.write(f"  Speedup: {speedup:.2f}x\n\n")
        
        f.write("Output Differences:\n")
        f.write(f"  Mean absolute difference: {timing_results['output_diff']['mean']:.6f}\n")
        f.write(f"  Max absolute difference: {timing_results['output_diff']['max']:.6f}\n")
        f.write(f"  Cosine similarity: {timing_results['output_diff']['cosine_sim']:.6f}\n")


def run_experiment(
    input_dim=256, 
    hidden_dims=[512, 256, 128], 
    output_dim=10,
    batch_size=32,
    num_runs=10,
    output_dir="outputs/bitnet_experiment",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run a complete BitNet experiment.
    """
    print(f"Running BitNet experiment on device: {device}")
    
    # Create networks
    print("Creating networks...")
    standard_net, bitnet = create_test_networks(input_dim, hidden_dims, output_dim)
    standard_net = standard_net.to(device)
    bitnet = bitnet.to(device)
    
    # Put networks in eval mode
    standard_net.eval()
    bitnet.eval()
    
    # Count parameters
    standard_params = sum(p.numel() for p in standard_net.parameters())
    bitnet_params = sum(p.numel() for p in bitnet.parameters())
    
    print(f"Standard network: {standard_params:,} parameters")
    print(f"BitNet network: {bitnet_params:,} parameters")
    
    # Analyze BitNet weight distribution
    print("Analyzing BitNet weight distribution...")
    stats = analyze_weight_distribution(bitnet)
    
    # Weight distribution summary
    total_ternary = sum(stats["ternary_counts"].values())
    print(f"Ternary weight distribution:")
    print(f"  -1: {stats['ternary_counts'][-1]:,} ({stats['ternary_counts'][-1]/total_ternary*100:.2f}%)")
    print(f"   0: {stats['ternary_counts'][0]:,} ({stats['ternary_counts'][0]/total_ternary*100:.2f}%)")
    print(f"  +1: {stats['ternary_counts'][1]:,} ({stats['ternary_counts'][1]/total_ternary*100:.2f}%)")
    
    # Compare networks
    print(f"Comparing networks (batch_size={batch_size}, runs={num_runs})...")
    results = compare_networks(standard_net, bitnet, input_dim, batch_size, num_runs)
    
    # Print comparison results
    print(f"Inference time:")
    print(f"  Standard: {results['timing']['standard']['mean']:.2f} ± {results['timing']['standard']['std']:.2f} ms")
    print(f"  BitNet: {results['timing']['bitnet']['mean']:.2f} ± {results['timing']['bitnet']['std']:.2f} ms")
    print(f"  Speedup: {results['timing']['speedup']:.2f}x")
    
    print(f"Output differences:")
    print(f"  Mean absolute difference: {results['output_diff']['mean']:.6f}")
    print(f"  Max absolute difference: {results['output_diff']['max']:.6f}")
    print(f"  Cosine similarity: {results['output_diff']['cosine_sim']:.6f}")
    
    # Visualize results
    print(f"Generating visualizations...")
    visualize_results(stats, results, output_dir)
    
    print(f"Experiment completed. Results saved to {output_dir}")
    
    return standard_net, bitnet, stats, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BitNet Experiment")
    
    # Model parameters
    parser.add_argument("--input_dim", type=int, default=256, help="Input dimension")
    parser.add_argument("--hidden_dims", type=str, default="512,256,128", help="Hidden dimensions (comma-separated)")
    parser.add_argument("--output_dim", type=int, default=10, help="Output dimension")
    
    # Experiment parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of inference runs")
    parser.add_argument("--output_dir", type=str, default="outputs/bitnet_experiment", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    # Scale parameter
    parser.add_argument("--scale", type=str, default="medium", choices=["small", "medium", "large", "xlarge"], 
                        help="Predefined network scale")
    
    args = parser.parse_args()
    
    # Define scales
    scales = {
        "small": {"input_dim": 128, "hidden_dims": [256, 128, 64], "output_dim": 10},
        "medium": {"input_dim": 256, "hidden_dims": [512, 256, 128], "output_dim": 10},
        "large": {"input_dim": 512, "hidden_dims": [1024, 512, 256], "output_dim": 10},
        "xlarge": {"input_dim": 1024, "hidden_dims": [2048, 1024, 512, 256], "output_dim": 10}
    }
    
    # Use scale if provided
    if args.scale in scales:
        args.input_dim = scales[args.scale]["input_dim"]
        args.hidden_dims = ",".join(str(x) for x in scales[args.scale]["hidden_dims"])
        args.output_dim = scales[args.scale]["output_dim"]
        args.output_dir = f"{args.output_dir}_{args.scale}"
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]
    
    # Run experiment
    run_experiment(
        input_dim=args.input_dim,
        hidden_dims=hidden_dims,
        output_dim=args.output_dim,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        output_dir=args.output_dir,
        device=args.device
    )
