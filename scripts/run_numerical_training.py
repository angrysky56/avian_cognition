#!/usr/bin/env python3
"""
Numerical Module Training Runner Script

This script automates the process of training and evaluating the numerical module
by running the fixed training and evaluation scripts in sequence.

Usage:
    python scripts/run_numerical_training.py
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run numerical module training and evaluation"
    )
    
    # Project directories
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/numerical",
        help="Data directory for numerical data"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints/numerical",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/numerical",
        help="Directory to save evaluation results"
    )
    
    # Model configuration
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--num_dim",
        type=int,
        default=32,
        help="Numerical representation dimension"
    )
    parser.add_argument(
        "--bit_linear",
        action="store_true",
        help="Use BitLinear quantization"
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
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
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    return args

def run_command(command, description):
    """Run a command and log the output."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(command)}")
    print(f"{'='*80}\n")
    
    # Run command and capture output
    start_time = time.time()
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        print(result.stdout)
        success = True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print(e.stdout)
        success = False
    
    # Print execution time
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nExecution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    return success

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Print setup info
    print(f"\n{'='*80}")
    print(f"NUMERICAL MODULE TRAINING PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"{'='*80}\n")
    
    # Step 1: Generate data
    generate_data_cmd = [
        "python", "scripts/generate_numerical_data.py",
        "--output_dir", args.data_dir,
        "--hidden_dim", str(args.hidden_dim),
        "--device", args.device
    ]
    generate_data_success = run_command(generate_data_cmd, "Generating numerical data")
    
    if not generate_data_success:
        print("Data generation failed. Aborting pipeline.")
        return
    
    # Step 2: Train model
    train_cmd = [
        "python", "scripts/train_numerical_module_fixed.py",
        "--data_dir", args.data_dir,
        "--model_dir", args.model_dir,
        "--hidden_dim", str(args.hidden_dim),
        "--num_dim", str(args.num_dim),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--device", args.device
    ]
    
    if args.bit_linear:
        train_cmd.append("--bit_linear")
    
    train_success = run_command(train_cmd, "Training numerical module")
    
    if not train_success:
        print("Training failed. Aborting pipeline.")
        return
    
    # Step 3: Evaluate model
    evaluate_cmd = [
        "python", "scripts/evaluate_numerical_module_fixed.py",
        "--model_path", f"{args.model_dir}/best_model.pt",
        "--data_dir", args.data_dir,
        "--output_dir", args.results_dir,
        "--batch_size", str(args.batch_size),
        "--device", args.device
    ]
    
    evaluate_success = run_command(evaluate_cmd, "Evaluating numerical module")
    
    if not evaluate_success:
        print("Evaluation failed.")
    
    # Print completion message
    print(f"\n{'='*80}")
    print(f"NUMERICAL MODULE TRAINING PIPELINE COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model checkpoints saved to: {args.model_dir}")
    print(f"Evaluation results saved to: {args.results_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
