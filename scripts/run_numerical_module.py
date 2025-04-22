#!/usr/bin/env python3
"""
Run Numerical Module Training and Evaluation

This script automates the process of training and evaluating the numerical module.
It uses the existing training and evaluation scripts correctly.

Usage:
    python scripts/run_numerical_module.py
"""

import os
import subprocess
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run numerical module training and evaluation")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda, cpu)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    
    return parser.parse_args()

def ensure_dir(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)

def run_command(command):
    """Run a shell command and print output."""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(command)}")
    print(f"{'='*80}\n")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for process to complete
    process.wait()
    
    return process.returncode

def main():
    args = parse_args()
    
    # Create necessary directories
    ensure_dir("data/numerical")
    ensure_dir("checkpoints/numerical")
    ensure_dir("results/numerical")
    
    # Step 1: Generate data
    print("\nStep 1: Generating numerical data")
    data_cmd = [
        "python", "scripts/generate_numerical_data.py",
        "--output_dir", "data/numerical",
        "--device", "cpu"  # Always use CPU for data generation
    ]
    
    if run_command(data_cmd) != 0:
        print("Error: Data generation failed")
        return
    
    # Step 2: Train numerical module
    print("\nStep 2: Training numerical module")
    train_cmd = [
        "python", "scripts/train_numerical_module.py",
        "--data_dir", "data/numerical",
        "--model_dir", "checkpoints/numerical",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--device", args.device
    ]
    
    if run_command(train_cmd) != 0:
        print("Error: Training failed")
        return
    
    # Step 3: Evaluate numerical module
    print("\nStep 3: Evaluating numerical module")
    eval_cmd = [
        "python", "scripts/evaluate_numerical_module.py",
        "--model_path", "checkpoints/numerical/best_model.pt",
        "--data_dir", "data/numerical",
        "--output_dir", "results/numerical",
        "--batch_size", str(args.batch_size),
        "--device", args.device
    ]
    
    if run_command(eval_cmd) != 0:
        print("Error: Evaluation failed")
        return
    
    print("\nNumerical module training and evaluation completed successfully")

if __name__ == "__main__":
    main()
