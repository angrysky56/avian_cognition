#!/usr/bin/env python3
"""
Numerical Module Training Script (Fixed Version)

This script trains the NumericalModule with improved CUDA handling and data processing.
It addresses issues with DataLoader workers and provides more robust error handling.

Usage:
    python scripts/train_numerical_module_fixed.py \
        --data_dir data/numerical \
        --model_dir checkpoints/numerical \
        --epochs 100 \
        --batch_size 32 \
        --lr 0.001
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.numerical import NumericalModule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the NumericalModule with improved CUDA handling"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/numerical",
        help="Data directory to store generated data"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints/numerical",
        help="Directory to save model checkpoints"
    )
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
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (reduced from 128 to avoid CUDA issues)"
    )
    parser.add_argument(
        "--lr",
        type=int,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping"
    )
    parser.add_argument(
        "--train_examples",
        type=int,
        default=5000,
        help="Number of training examples to generate"
    )
    parser.add_argument(
        "--val_examples",
        type=int,
        default=1000,
        help="Number of validation examples to generate"
    )
    
    # Runtime configuration
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,  # Use 0 to avoid CUDA issues
        help="Number of DataLoader workers"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Infer device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    return args


def setup_logging(model_dir, experiment_name):
    """
    Set up logging for the training process.
    
    Args:
        model_dir: Directory to save logs
        experiment_name: Name for the experiment
    
    Returns:
        logger: Configured logger
    """
    # Create log directory
    log_dir = os.path.join(model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(experiment_name)


def generate_data(num_examples, operations, value_range, hidden_dim, device):
    """
    Generate synthetic arithmetic data for training.
    
    Args:
        num_examples: Number of examples to generate
        operations: List of operations to include
        value_range: Range of values (min_val, max_val)
        hidden_dim: Hidden dimension size
        device: Device to place tensors on
        
    Returns:
        data: Dictionary of generated data
    """
    min_val, max_val = value_range
    examples_per_op = num_examples // len(operations)
    total_examples = examples_per_op * len(operations)
    
    # Initialize tensors
    h1_data = torch.zeros(total_examples, hidden_dim, device=device)
    h2_data = torch.zeros(total_examples, hidden_dim, device=device)
    h_op_data = torch.zeros(total_examples, hidden_dim, device=device)
    target_data = torch.zeros(total_examples, 1, device=device)
    
    # Track original operands and scaling factors
    original_operands = []
    scaling_factors = []
    operation_indices = []
    
    # Generate examples for each operation
    idx = 0
    for op_idx, operation in enumerate(operations):
        for _ in range(examples_per_op):
            # Generate operands with appropriate constraints
            a = np.random.randint(min_val, max_val)
            
            # For division, ensure clean division (when possible)
            if operation == 'divide':
                if a > 0 and np.random.random() < 0.8:  # 80% clean division
                    b = max(1, np.random.randint(1, min(a, max_val // 2)))
                    b = b * (a // b) if a // b > 0 else b  # Make a divisible by b
                else:
                    b = max(1, np.random.randint(min_val, max_val))  # Avoid division by zero
            else:
                b = np.random.randint(min_val, max_val)
                
            # Compute result based on operation
            if operation == 'add':
                result = a + b
                max_possible = 2 * max_val
            elif operation == 'subtract':
                result = a - b
                max_possible = max_val
            elif operation == 'multiply':
                result = a * b
                max_possible = max_val * max_val
            elif operation == 'divide':
                result = a / b
                max_possible = max_val
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
            # Calculate scaling factor for normalization
            scaling_factor = 1.0 / max(1.0, max_possible)
            scaling_factors.append(scaling_factor)
            
            # Save original operands
            original_operands.append((a, b))
            operation_indices.append(op_idx)
            
            # Create hidden state representations
            # Note: This is a fixed encoding scheme - in a real integration
            # these would be learned embeddings from the language model
            h1_data[idx, 0] = a * scaling_factor
            h1_data[idx, 1] = (a // 100) * scaling_factor
            h1_data[idx, 2] = ((a % 100) // 10) * scaling_factor
            h1_data[idx, 3] = (a % 10) * scaling_factor
            
            h2_data[idx, 0] = b * scaling_factor
            h2_data[idx, 1] = (b // 100) * scaling_factor
            h2_data[idx, 2] = ((b % 100) // 10) * scaling_factor
            h2_data[idx, 3] = (b % 10) * scaling_factor
            
            # One-hot encoding for operation
            h_op_data[idx, op_idx] = 1.0
            
            # Store result (scaled)
            target_data[idx, 0] = result * scaling_factor
            
            idx += 1
    
    # Return data
    data = {
        'h1_data': h1_data,
        'h2_data': h2_data,
        'h_op_data': h_op_data,
        'target_data': target_data,
        'original_operands': original_operands,
        'operations': operations,
        'operation_indices': operation_indices,
        'scaling_factors': scaling_factors,
        'value_range': value_range
    }
    
    return data


def create_model(hidden_dim, num_dim, bit_linear, device, add_decoder=True):
    """
    Create NumericalModule model with optional decoder for training.
    
    Args:
        hidden_dim: Hidden dimension size
        num_dim: Numerical representation dimension
        bit_linear: Whether to use BitLinear quantization
        device: Device to place model on
        add_decoder: Whether to add a decoder head for training
        
    Returns:
        model: NumericalModule model (with decoder if add_decoder=True)
    """
    print(f"Creating NumericalModule with hidden_dim={hidden_dim}, num_dim={num_dim}, bit_linear={bit_linear}")
    
    # Basic NumericalModule
    numerical_module = NumericalModule(
        hidden_dim=hidden_dim,
        num_dim=num_dim,
        bit_linear=bit_linear
    )
    
    # If no decoder needed, return module as is
    if not add_decoder:
        return numerical_module.to(device)
    
    # Create a wrapper model with decoder head
    class NumericalModuleWithDecoder(nn.Module):
        def __init__(self, numerical_module, hidden_dim):
            super().__init__()
            self.numerical_module = numerical_module
            # Simple decoder that extracts numerical value from hidden state
            self.decoder = nn.Linear(hidden_dim, 1)
            
        def forward(self, h1, h2, h_op):
            # Forward pass through numerical module
            result_hidden, op_weights = self.numerical_module(h1, h2, h_op)
            
            # Apply decoder to get scalar prediction
            prediction = self.decoder(result_hidden)
            
            return prediction, result_hidden, op_weights
    
    # Create and return the wrapped model
    model = NumericalModuleWithDecoder(numerical_module, hidden_dim).to(device)
    
    return model


def train_epoch(model, train_data, optimizer, criterion, batch_size, device, logger):
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        train_data: Training data dictionary
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        batch_size: Batch size for training
        device: Device for computation
        logger: Logger for recording progress
        
    Returns:
        metrics: Dictionary of training metrics
    """
    # Set model to training mode
    model.train()
    
    # Extract data
    h1 = train_data['h1_data']
    h2 = train_data['h2_data']
    h_op = train_data['h_op_data']
    targets = train_data['target_data']
    
    # Determine number of batches
    num_examples = h1.size(0)
    num_batches = (num_examples + batch_size - 1) // batch_size
    
    # Track metrics
    total_loss = 0
    correct = 0
    
    # Train in batches
    indices = torch.randperm(num_examples, device=device)
    
    for batch_idx in tqdm(range(num_batches), desc="Training"):
        # Get batch indices
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_examples)
        batch_indices = indices[batch_start:batch_end]
        
        # Get batch data
        batch_h1 = h1[batch_indices].to(device)
        batch_h2 = h2[batch_indices].to(device)
        batch_h_op = h_op[batch_indices].to(device)
        batch_targets = targets[batch_indices].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions, _, _ = model(batch_h1, batch_h2, batch_h_op)
        
        # Calculate loss
        loss = criterion(predictions, batch_targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Track loss
        batch_size_actual = batch_end - batch_start
        total_loss += loss.item() * batch_size_actual
        
        # Calculate accuracy (with tolerance)
        with torch.no_grad():
            abs_errors = torch.abs(predictions - batch_targets)
            rel_tolerance = torch.abs(batch_targets) * 0.05
            rel_tolerance = torch.max(rel_tolerance, torch.tensor(1e-2, device=device))
            batch_correct = (abs_errors <= rel_tolerance).float().sum().item()
            correct += batch_correct
    
    # Calculate metrics
    accuracy = correct / num_examples
    avg_loss = total_loss / num_examples
    
    logger.info(f"Training - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")
    
    return {'loss': avg_loss, 'accuracy': accuracy}


def validate(model, val_data, criterion, batch_size, device, logger):
    """
    Validate the model.
    
    Args:
        model: Model to validate
        val_data: Validation data dictionary
        criterion: Loss function
        batch_size: Batch size for validation
        device: Device for computation
        logger: Logger for recording progress
        
    Returns:
        metrics: Dictionary of validation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Extract data
    h1 = val_data['h1_data']
    h2 = val_data['h2_data']
    h_op = val_data['h_op_data']
    targets = val_data['target_data']
    operations = val_data['operations']
    operation_indices = val_data['operation_indices']
    
    # Determine number of batches
    num_examples = h1.size(0)
    num_batches = (num_examples + batch_size - 1) // batch_size
    
    # Track metrics
    total_loss = 0
    correct = 0
    
    # Track per-operation metrics
    op_correct = {op: 0 for op in operations}
    op_total = {op: 0 for op in operations}
    
    # Validate without gradient tracking
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Validating"):
            # Get batch indices
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, num_examples)
            batch_indices = torch.arange(batch_start, batch_end, device=device)
            
            # Get batch data
            batch_h1 = h1[batch_indices].to(device)
            batch_h2 = h2[batch_indices].to(device)
            batch_h_op = h_op[batch_indices].to(device)
            batch_targets = targets[batch_indices].to(device)
            
            # Forward pass
            predictions, _, _ = model(batch_h1, batch_h2, batch_h_op)
            
            # Calculate loss
            loss = criterion(predictions, batch_targets)
            
            # Track loss
            batch_size_actual = batch_end - batch_start
            total_loss += loss.item() * batch_size_actual
            
            # Calculate accuracy (with tolerance)
            abs_errors = torch.abs(predictions - batch_targets)
            rel_tolerance = torch.abs(batch_targets) * 0.05
            rel_tolerance = torch.max(rel_tolerance, torch.tensor(1e-2, device=device))
            batch_correct = (abs_errors <= rel_tolerance).float()
            correct += batch_correct.sum().item()
            
            # Track per-operation metrics
            for i, idx in enumerate(range(batch_start, batch_end)):
                if idx < len(operation_indices):
                    op_idx = operation_indices[idx]
                    if op_idx < len(operations):
                        op = operations[op_idx]
                        op_total[op] += 1
                        op_correct[op] += batch_correct[i].item()
    
    # Calculate metrics
    accuracy = correct / num_examples
    avg_loss = total_loss / num_examples
    
    # Calculate per-operation accuracy
    op_accuracy = {}
    for op in operations:
        if op_total[op] > 0:
            op_accuracy[op] = op_correct[op] / op_total[op]
        else:
            op_accuracy[op] = 0.0
    
    logger.info(f"Validation - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")
    logger.info("Per-Operation Validation Accuracy:")
    for op in operations:
        logger.info(f"  {op.capitalize()}: {op_accuracy[op]:.4f}")
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        **{f"{op}_accuracy": op_accuracy[op] for op in operations}
    }
    
    return metrics


def test_extrapolation(model, operations, extrapolation_range, num_examples, hidden_dim, batch_size, device, logger):
    """
    Test the model on extrapolation data.
    
    Args:
        model: Model to test
        operations: List of operations to test
        extrapolation_range: Range for extrapolation testing (min_val, max_val)
        num_examples: Number of examples to generate per operation
        hidden_dim: Hidden dimension size
        batch_size: Batch size for testing
        device: Device for computation
        logger: Logger for recording progress
        
    Returns:
        metrics: Dictionary of extrapolation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Generate extrapolation data
    extrapolation_data = generate_data(
        num_examples, 
        operations,
        extrapolation_range,
        hidden_dim,
        device='cpu'  # Generate on CPU for safety
    )
    
    # Extract data
    h1 = extrapolation_data['h1_data']
    h2 = extrapolation_data['h2_data']
    h_op = extrapolation_data['h_op_data']
    targets = extrapolation_data['target_data']
    operation_indices = extrapolation_data['operation_indices']
    original_operands = extrapolation_data['original_operands']
    
    # Determine number of batches
    num_examples_total = h1.size(0)
    num_batches = (num_examples_total + batch_size - 1) // batch_size
    
    # Track metrics
    correct = 0
    
    # Track per-operation metrics
    op_correct = {op: 0 for op in operations}
    op_total = {op: 0 for op in operations}
    op_examples = {op: [] for op in operations}
    
    # Test without gradient tracking
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Testing Extrapolation"):
            # Get batch indices
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, num_examples_total)
            batch_indices = torch.arange(batch_start, batch_end)
            
            # Get batch data
            batch_h1 = h1[batch_indices].to(device)
            batch_h2 = h2[batch_indices].to(device)
            batch_h_op = h_op[batch_indices].to(device)
            batch_targets = targets[batch_indices].to(device)
            
            # Forward pass
            predictions, _, _ = model(batch_h1, batch_h2, batch_h_op)
            
            # Calculate accuracy (with tolerance)
            abs_errors = torch.abs(predictions - batch_targets)
            rel_tolerance = torch.abs(batch_targets) * 0.05
            rel_tolerance = torch.max(rel_tolerance, torch.tensor(1e-2, device=device))
            batch_correct = (abs_errors <= rel_tolerance).float()
            correct += batch_correct.sum().item()
            
            # Track per-operation metrics and examples
            for i, idx in enumerate(range(batch_start, batch_end)):
                if idx < len(operation_indices):
                    op_idx = operation_indices[idx]
                    if op_idx < len(operations):
                        op = operations[op_idx]
                        op_total[op] += 1
                        op_correct[op] += batch_correct[i].item()
                        
                        # Track examples for logging
                        if len(op_examples[op]) < 5 and idx < len(original_operands):
                            a, b = original_operands[idx]
                            pred = predictions[i].item() / extrapolation_data['scaling_factors'][idx]
                            targ = batch_targets[i].item() / extrapolation_data['scaling_factors'][idx]
                            error = abs(pred - targ)
                            op_examples[op].append((a, b, pred, targ, error))
    
    # Calculate metrics
    accuracy = correct / num_examples_total
    
    # Calculate per-operation accuracy
    op_accuracy = {}
    for op in operations:
        if op_total[op] > 0:
            op_accuracy[op] = op_correct[op] / op_total[op]
        else:
            op_accuracy[op] = 0.0
    
    # Log example results
    for op in operations:
        if op_examples[op]:
            logger.info(f"Extrapolation examples ({op}):")
            for a, b, pred, targ, error in op_examples[op]:
                logger.info(f"  {a} {op} {b} = {pred:.2f} (correct: {targ:.2f}, error: {error:.2f})")
    
    logger.info(f"Extrapolation Accuracy: {accuracy:.4f}")
    logger.info("Per-Operation Extrapolation Accuracy:")
    for op in operations:
        logger.info(f"  {op.capitalize()}: {op_accuracy[op]:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        **{f"{op}_accuracy": op_accuracy[op] for op in operations}
    }
    
    return metrics


def plot_training_progress(train_metrics, val_metrics, extrapolation_metrics, operations, output_dir):
    """
    Create visualizations of training progress.
    
    Args:
        train_metrics: List of training metrics dictionaries
        val_metrics: List of validation metrics dictionaries
        extrapolation_metrics: List of extrapolation metrics dictionaries
        operations: List of operations
        output_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract metrics
    epochs = range(1, len(train_metrics) + 1)
    train_loss = [m['loss'] for m in train_metrics]
    train_acc = [m['accuracy'] for m in train_metrics]
    val_loss = [m['loss'] for m in val_metrics] if val_metrics else []
    val_acc = [m['accuracy'] for m in val_metrics] if val_metrics else []
    ext_acc = [m['accuracy'] for m in extrapolation_metrics] if extrapolation_metrics else []
    
    # Plot 1: Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "loss.png"))
    plt.close()
    
    # Plot 2: Training and Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    if val_acc:
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    if ext_acc:
        plt.plot(epochs, ext_acc, 'g-', label='Extrapolation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(plots_dir, "accuracy.png"))
    plt.close()
    
    # Plot 3: Per-Operation Accuracy
    plt.figure(figsize=(10, 6))
    for op in operations:
        op_val_acc = [m.get(f"{op}_accuracy", 0) for m in val_metrics] if val_metrics else []
        if op_val_acc:
            plt.plot(epochs, op_val_acc, '--', label=f'{op.capitalize()} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Per-Operation Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(plots_dir, "operation_accuracy.png"))
    plt.close()


def save_checkpoint(model, optimizer, epoch, metrics, model_dir, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        metrics: Current metrics
        model_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    # Create checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'hidden_dim': model.numerical_module.hidden_dim if hasattr(model, 'numerical_module') else model.hidden_dim,
            'num_dim': model.numerical_module.num_dim if hasattr(model, 'numerical_module') else model.num_dim,
            'bit_linear': model.numerical_module.arithmetic_units['add'].__class__.__name__ == 'BitLinear' if hasattr(model, 'numerical_module') else False
        }
    }
    
    # Regular checkpoint
    checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch}.pt")
    
    # Save with error handling
    try:
        torch.save(checkpoint, checkpoint_path)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        # Try with pickle
        try:
            import pickle
            torch.save(checkpoint, checkpoint_path, pickle_module=pickle)
        except Exception as e2:
            print(f"Error saving with pickle: {e2}")
            # Skip saving if all methods fail
    
    # Save best model if indicated
    if is_best:
        best_path = os.path.join(model_dir, "best_model.pt")
        try:
            torch.save(checkpoint, best_path)
            print(f"Saved new best model with validation accuracy: {metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"Error saving best model: {e}")
            # Try with pickle
            try:
                import pickle
                torch.save(checkpoint, best_path, pickle_module=pickle)
            except Exception as e2:
                print(f"Error saving with pickle: {e2}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"NumericalModule_{timestamp}"
    logger = setup_logging(args.model_dir, experiment_name)
    
    try:
        # Define operations
        operations = ['add', 'subtract', 'multiply', 'divide']
        
        # Generate training data
        logger.info("Generating training data...")
        train_data = generate_data(
            args.train_examples, 
            operations, 
            (0, 100),  # Training range
            args.hidden_dim, 
            'cpu'  # Generate on CPU for safety
        )
        
        # Generate validation data
        logger.info("Generating validation data...")
        val_data = generate_data(
            args.val_examples, 
            operations, 
            (0, 100),  # Validation range (same as training)
            args.hidden_dim, 
            'cpu'  # Generate on CPU for safety
        )
        
        # Create model with decoder for training
        model = create_model(
            args.hidden_dim, 
            args.num_dim, 
            args.bit_linear, 
            args.device,
            add_decoder=True  # Add decoder for training
        )
        
        # Create optimizer and loss function
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        criterion = nn.MSELoss()
        
        # Training setup
        best_val_accuracy = 0.0
        patience_counter = 0
        train_metrics_history = []
        val_metrics_history = []
        extrapolation_metrics_history = []
        
        # Training loop
        logger.info(f"Beginning training for {args.epochs} epochs...")
        
        for epoch in range(args.epochs):
            # Train for one epoch
            train_metrics = train_epoch(
                model, train_data, optimizer, criterion, 
                args.batch_size, args.device, logger
            )
            train_metrics_history.append(train_metrics)
            
            # Validate
            val_metrics = validate(
                model, val_data, criterion, 
                args.batch_size, args.device, logger
            )
            val_metrics_history.append(val_metrics)
            
            # Test extrapolation (every 5 epochs)
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                extrapolation_metrics = test_extrapolation(
                    model, operations, (100, 1000),  # Extrapolation range
                    args.val_examples, args.hidden_dim,
                    args.batch_size, args.device, logger
                )
                extrapolation_metrics_history.append(extrapolation_metrics)
            else:
                # Placeholder for plotting
                extrapolation_metrics_history.append({'accuracy': 0.0})
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    model, optimizer, epoch, val_metrics, 
                    args.model_dir, is_best=False
                )
            
            # Check for best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                patience_counter = 0
                
                # Save best model
                save_checkpoint(
                    model, optimizer, epoch, val_metrics, 
                    args.model_dir, is_best=True
                )
                
                logger.info(f"New best model with validation accuracy: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Early stopping patience: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Create final visualizations
        plot_training_progress(
            train_metrics_history,
            val_metrics_history,
            extrapolation_metrics_history,
            operations,
            args.model_dir
        )
        
        # Final evaluation
        logger.info("Training complete!")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
        
        # Save numerical module separately (without decoder)
        if hasattr(model, 'numerical_module'):
            numerical_module = model.numerical_module
            final_model_path = os.path.join(args.model_dir, "numerical_module_final.pt")
            try:
                torch.save(numerical_module.state_dict(), final_model_path)
                logger.info(f"Saved final numerical module to {final_model_path}")
            except Exception as e:
                logger.error(f"Error saving final model: {e}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
