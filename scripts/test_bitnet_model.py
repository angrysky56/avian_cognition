#!/usr/bin/env python
"""
BitNet Model Testing Script

This script tests the BitNet b1.58 ternary implementation on a larger pre-trained model.
It compares performance, memory usage, and output differences between:
1. Original full-precision model
2. Model converted to BitLinear with ternary weights
"""

import os
import sys
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.bitnet import BitLinear, convert_linear_to_bit_linear

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test BitNet b1.58 Implementation")
    
    # Model selection
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                        help="Hugging Face model to test (default: distilbert-base-uncased)")
    parser.add_argument("--model_type", type=str, default="base",
                        choices=["base", "sequence-classification"],
                        help="Type of model to load")
    
    # Test options
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of inference runs for timing")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length for input tokens")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="outputs/bitnet_test",
                        help="Directory to save test results")
    parser.add_argument("--visualize", action="store_true", 
                        help="Generate visualizations of weight distributions")
    
    # Runtime options
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for testing")
    
    return parser.parse_args()

def load_models(args):
    """Load original model and create a BitNet version."""
    print(f"Loading model: {args.model}")
    
    # Load tokenizer for both models
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load original model
    if args.model_type == "base":
        original_model = AutoModel.from_pretrained(args.model)
    else:
        original_model = AutoModelForSequenceClassification.from_pretrained(args.model)
    
    original_model = original_model.to(args.device)
    original_model.eval()
    
    # Create a deep copy for BitNet conversion
    # We need to copy to avoid modifying the original model
    import copy
    bitnet_model = copy.deepcopy(original_model)
    
    # Convert all Linear layers to BitLinear
    print("Converting to BitNet...")
    convert_linear_to_bit_linear(bitnet_model, device=args.device)
    bitnet_model.eval()
    
    return tokenizer, original_model, bitnet_model

def count_parameters(model):
    """Count number of parameters in model and memory usage."""
    param_count = sum(p.numel() for p in model.parameters())
    param_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)
    
    return param_count, param_size_mb

def analyze_weight_distribution(model):
    """Analyze weight distribution across model."""
    # Collect statistics
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

def measure_inference_time(model, inputs, num_runs=5):
    """Measure inference time over multiple runs."""
    times = []
    
    # Warmup run
    with torch.no_grad():
        _ = model(**inputs)
    
    # Timed runs
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "times": times
    }

def compare_outputs(original_outputs, bitnet_outputs):
    """Compare outputs between original and BitNet models."""
    # The structure of outputs depends on the model type, but this should work for most cases
    if isinstance(original_outputs, tuple) and isinstance(bitnet_outputs, tuple):
        # Compare first item in tuple (usually the main output)
        original_tensor = original_outputs[0]
        bitnet_tensor = bitnet_outputs[0]
    else:
        original_tensor = original_outputs
        bitnet_tensor = bitnet_outputs
    
    # Calculate differences
    if hasattr(original_tensor, "last_hidden_state"):
        # Compare hidden states for encoder models
        original_tensor = original_tensor.last_hidden_state
        bitnet_tensor = bitnet_tensor.last_hidden_state
    
    abs_diff = torch.abs(original_tensor - bitnet_tensor)
    
    metrics = {
        "mean_abs_diff": abs_diff.mean().item(),
        "max_abs_diff": abs_diff.max().item(),
        "cosine_sim": torch.nn.functional.cosine_similarity(
            original_tensor.flatten(), bitnet_tensor.flatten(), dim=0
        ).item()
    }
    
    return metrics

def generate_visualizations(stats, original_param_count, bitnet_param_count, 
                           timing_results, output_dir):
    """Generate visualizations of test results."""
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
    
    models = ['Original FP32', 'BitNet b1.58']
    mean_times = [timing_results['original']['mean'], timing_results['bitnet']['mean']]
    std_times = [timing_results['original']['std'], timing_results['bitnet']['std']]
    
    # Calculate speedup
    speedup = timing_results['original']['mean'] / timing_results['bitnet']['mean']
    
    bars = ax.bar(models, mean_times, yerr=std_times, capsize=10, color=['blue', 'orange'])
    
    # Add value labels on bars
    for i, (bar, time_ms) in enumerate(zip(bars, mean_times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time_ms:.2f} ms',
                ha='center', va='bottom')
    
    ax.set_title(f'Inference Time Comparison (BitNet is {speedup:.2f}x faster)')
    ax.set_ylabel('Time (ms)')
    ax.set_ylim(0, max(mean_times) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    fig.savefig(output_dir / "inference_time.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Memory Usage Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    original_size_mb = original_param_count * 4 / (1024 * 1024)  # FP32 (4 bytes per param)
    bitnet_size_mb = bitnet_param_count * 1.58 / 8 / (1024 * 1024)  # 1.58 bits per param
    
    # Memory compression ratio
    compression_ratio = original_size_mb / bitnet_size_mb
    
    sizes = [original_size_mb, bitnet_size_mb]
    bars = ax.bar(models, sizes, color=['blue', 'orange'])
    
    # Add value labels on bars
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{size:.2f} MB',
                ha='center', va='bottom')
    
    ax.set_title(f'Model Size Comparison (BitNet is {compression_ratio:.2f}x smaller)')
    ax.set_ylabel('Size (MB)')
    ax.set_ylim(0, max(sizes) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    fig.savefig(output_dir / "memory_usage.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main testing function."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    tokenizer, original_model, bitnet_model = load_models(args)
    
    # Count parameters
    original_params, original_size_mb = count_parameters(original_model)
    bitnet_params, bitnet_size_mb = count_parameters(bitnet_model)
    
    print(f"\nModel Parameters:")
    print(f"  Original: {original_params:,} parameters ({original_size_mb:.2f} MB)")
    print(f"  BitNet: {bitnet_params:,} parameters ({bitnet_size_mb:.2f} MB assuming float32)")
    print(f"  Compressed BitNet size: {bitnet_params * 1.58 / 8 / (1024 * 1024):.2f} MB (1.58 bits/weight)")
    print(f"  Compression ratio: {original_size_mb / (bitnet_params * 1.58 / 8 / (1024 * 1024)):.2f}x")
    
    # Analyze BitNet weight distribution
    print("\nAnalyzing BitNet weight distribution...")
    stats = analyze_weight_distribution(bitnet_model)
    
    total_ternary = sum(stats["ternary_counts"].values())
    print(f"  Total BitLinear weights: {stats['linear_weights']:,}")
    print(f"  Ternary weight distribution:")
    print(f"    -1: {stats['ternary_counts'][-1]:,} ({stats['ternary_counts'][-1]/total_ternary*100:.2f}%)")
    print(f"     0: {stats['ternary_counts'][0]:,} ({stats['ternary_counts'][0]/total_ternary*100:.2f}%)")
    print(f"    +1: {stats['ternary_counts'][1]:,} ({stats['ternary_counts'][1]/total_ternary*100:.2f}%)")
    
    # Prepare example inputs for inference
    print("\nPreparing inputs for inference testing...")
    example_texts = [
        "This is a test sentence for inference.",
        "Let's see how well the BitNet model performs.",
        "We want to measure inference speed and accuracy."
    ] * (args.batch_size // 3 + 1)  # Repeat to match batch size
    example_texts = example_texts[:args.batch_size]  # Truncate to exact batch size
    
    inputs = tokenizer(
        example_texts, 
        padding='max_length', 
        truncation=True, 
        max_length=args.seq_length, 
        return_tensors="pt"
    ).to(args.device)
    
    # Measure inference time
    print(f"\nMeasuring inference time ({args.num_runs} runs)...")
    timing_results = {
        'original': measure_inference_time(original_model, inputs, args.num_runs),
        'bitnet': measure_inference_time(bitnet_model, inputs, args.num_runs)
    }
    
    print(f"  Original model: {timing_results['original']['mean']:.2f} ± {timing_results['original']['std']:.2f} ms")
    print(f"  BitNet model: {timing_results['bitnet']['mean']:.2f} ± {timing_results['bitnet']['std']:.2f} ms")
    print(f"  Speedup: {timing_results['original']['mean'] / timing_results['bitnet']['mean']:.2f}x")
    
    # Compare outputs
    print("\nComparing outputs...")
    with torch.no_grad():
        original_outputs = original_model(**inputs)
        bitnet_outputs = bitnet_model(**inputs)
    
    output_diff = compare_outputs(original_outputs, bitnet_outputs)
    print(f"  Mean absolute difference: {output_diff['mean_abs_diff']:.6f}")
    print(f"  Max absolute difference: {output_diff['max_abs_diff']:.6f}")
    print(f"  Cosine similarity: {output_diff['cosine_sim']:.6f}")
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        generate_visualizations(
            stats, 
            original_params, 
            bitnet_params, 
            timing_results, 
            output_dir
        )
        print(f"  Visualizations saved to {output_dir}")
    
    # Save detailed results
    print("\nSaving detailed results...")
    import json
    
    results = {
        "model": args.model,
        "parameters": {
            "original": original_params,
            "bitnet": bitnet_params,
            "compression_ratio": original_size_mb / (bitnet_params * 1.58 / 8 / (1024 * 1024))
        },
        "weight_distribution": {
            "-1": stats['ternary_counts'][-1],
            "0": stats['ternary_counts'][0],
            "+1": stats['ternary_counts'][1]
        },
        "timing": {
            "original": {
                "mean_ms": timing_results['original']['mean'],
                "std_ms": timing_results['original']['std']
            },
            "bitnet": {
                "mean_ms": timing_results['bitnet']['mean'],
                "std_ms": timing_results['bitnet']['std']
            },
            "speedup": timing_results['original']['mean'] / timing_results['bitnet']['mean']
        },
        "output_comparison": output_diff
    }
    
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nTesting completed successfully!")
    print(f"All results saved to {output_dir}")
    

if __name__ == "__main__":
    main()
