#!/usr/bin/env python3
"""
Numerical Module Training Script

This script trains the NumericalModule using hidden states extracted from a pre-trained
language model. It implements a proper training pipeline with early stopping,
validation, and extrapolation testing.

Usage:
    python scripts/train_numerical_module.py \
        --data_dir data/numerical \
        --model_dir checkpoints/numerical \
        --epochs 100 \
        --batch_size 128 \
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
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# No patching - we'll handle device issues properly

from src.modules.numerical import NumericalModule
from training.numerical_trainer import NumericalCompetenceTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the NumericalModule using extracted hidden states"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/numerical",
        help="Data directory containing train_data.pt, val_data.pt, and extrapolation_data.pt"
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
        default=512,  # Match the dimension from data generation
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
        default=128,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Infer device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    return args


def load_data(data_dir):
    """
    Load training, validation, and extrapolation data.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        extrapolation_dataset: Extrapolation dataset
        operations: List of operations
    """
    print(f"Loading data from {data_dir}...")
    
    # Check if data files exist
    train_path = os.path.join(data_dir, "train_data.pt")
    val_path = os.path.join(data_dir, "val_data.pt")
    extrapolation_path = os.path.join(data_dir, "extrapolation_data.pt")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found at {val_path}")
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
        train_data = torch.load(train_path, weights_only=False)
        val_data = torch.load(val_path, weights_only=False)
        extrapolation_data = torch.load(extrapolation_path, weights_only=False)
    except (RuntimeError, TypeError) as e:
        print(f"Error with weights_only=False: {e}")
        print("Trying legacy loading method...")
        import pickle
        train_data = torch.load(train_path, pickle_module=pickle)
        val_data = torch.load(val_path, pickle_module=pickle)
        extrapolation_data = torch.load(extrapolation_path, pickle_module=pickle)
    
    # Extract components
    operations = train_data['operations']
    
    # For DataLoader compatibility, we'll keep tensors on CPU initially
    # We'll move them to the appropriate device right before use in the trainer
    train_dataset = TensorDataset(
        train_data['h1_data'],
        train_data['h2_data'],
        train_data['h_op_data'],
        train_data['target_data']
    )
    
    val_dataset = TensorDataset(
        val_data['h1_data'],
        val_data['h2_data'],
        val_data['h_op_data'],
        val_data['target_data']
    )
    
    extrapolation_dataset = TensorDataset(
        extrapolation_data['h1_data'],
        extrapolation_data['h2_data'],
        extrapolation_data['h_op_data'],
        extrapolation_data['target_data']
    )
    
    print(f"Loaded {len(train_dataset)} training examples, {len(val_dataset)} validation examples, and {len(extrapolation_dataset)} extrapolation examples")
    
    return train_dataset, val_dataset, extrapolation_dataset, operations


def create_dataloaders(train_dataset, val_dataset, extrapolation_dataset, args):
    """
    Create data loaders for training, validation, and extrapolation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        extrapolation_dataset: Extrapolation dataset
        args: Command line arguments
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        extrapolation_loader: Extrapolation data loader
    """
    # Set num_workers to 0 to avoid CUDA initialization issues
    # This is necessary when using CUDA with DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 workers to avoid CUDA issues
        pin_memory=False  # Disable pin_memory for CUDA tensors
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 workers to avoid CUDA issues
        pin_memory=False  # Disable pin_memory for CUDA tensors
    )
    
    extrapolation_loader = DataLoader(
        extrapolation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 workers to avoid CUDA issues
        pin_memory=False  # Disable pin_memory for CUDA tensors
    )
    
    return train_loader, val_loader, extrapolation_loader


def create_model(hidden_dim, num_dim, bit_linear, device):
    """
    Create NumericalModule model with an optional decoder head.
    
    Args:
        hidden_dim: Hidden dimension size
        num_dim: Numerical representation dimension
        bit_linear: Whether to use BitLinear quantization
        device: Device to place model on
        
    Returns:
        model: NumericalModule with decoder head for training
    """
    print(f"Creating NumericalModule with hidden_dim={hidden_dim}, num_dim={num_dim}, bit_linear={bit_linear}...")
    
    # Create the core numerical module
    numerical_module = NumericalModule(
        hidden_dim=hidden_dim,
        num_dim=num_dim,
        bit_linear=bit_linear
    )
    
    # Create a wrapper model with an improved decoder for training
    class NumericalModuleWithDecoder(nn.Module):
        def __init__(self, numerical_module):
            super().__init__()
            self.numerical_module = numerical_module
            hidden_dim = numerical_module.hidden_dim
            
            # Improved decoder architecture with multiple layers for better representation learning
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.Dropout(0.2),  # Add dropout for regularization
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1)
            )
            
        def forward(self, h1, h2, h_op):
            # Forward pass through numerical module
            result_hidden, op_weights = self.numerical_module(h1, h2, h_op)
            
            # Apply improved decoder to get scalar prediction
            # Handle the batch size=1 case for BatchNorm
            if result_hidden.size(0) == 1:
                # Skip BatchNorm layers for single samples
                unbatched = True
                x = result_hidden
                x = nn.functional.gelu(self.decoder[0](x))  # First Linear + GELU
                x = self.decoder[3](x)  # Skip BatchNorm, apply Dropout
                x = nn.functional.gelu(self.decoder[4](x))  # Second Linear + GELU
                prediction = self.decoder[6](x)  # Final Linear
            else:
                # Use the full sequential model for normal batches
                prediction = self.decoder(result_hidden)
            
            return prediction, result_hidden, op_weights
    
    # Create and return the model with decoder
    model = NumericalModuleWithDecoder(numerical_module).to(device)
    
    return model


def train_numerical_module(model, train_loader, val_loader, extrapolation_loader, operations, args):
    """
    Train the NumericalModule using the NumericalCompetenceTrainer.
    
    Args:
        model: NumericalModule model
        train_loader: Training data loader
        val_loader: Validation data loader
        extrapolation_loader: Extrapolation data loader
        operations: List of operations
        args: Command line arguments
        
    Returns:
        trainer: Trained NumericalCompetenceTrainer
    """
    # Create optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create a custom loss function for better handling of numerical operations
    class RelativeErrorLoss(nn.Module):
        def __init__(self, eps=1e-8):
            super().__init__()
            self.eps = eps
            # Also keep MSE for stability in early training
            self.mse = nn.MSELoss()
            
        def forward(self, pred, target):
            # Calculate mean squared error for stability
            mse_loss = self.mse(pred, target)
            
            # Calculate relative error (better for different scales)
            rel_error = torch.abs(pred - target) / (torch.abs(target) + self.eps)
            rel_loss = torch.mean(rel_error)
            
            # Combine losses with higher weight on relative error
            combined_loss = 0.3 * mse_loss + 0.7 * rel_loss
            
            return combined_loss
    
    criterion = RelativeErrorLoss()
    
    # Create trainer configuration
    config = {
        'hidden_dim': args.hidden_dim,
        'batch_size': args.batch_size,
        'plot_interval': 5,
        'checkpoint_interval': 5,
        'early_stopping_patience': args.patience
    }
    
    # Define curriculum learning stages for progressive number range training
    # This helps the model learn basic operations on small numbers first
    curriculum_stages = [
        {'epochs': 10, 'ranges': {'train': (0, 10), 'validation': (0, 10), 'extrapolation': (10, 20)}},
        {'epochs': 10, 'ranges': {'train': (0, 50), 'validation': (0, 50), 'extrapolation': (50, 100)}},
        {'epochs': args.epochs - 20, 'ranges': {'train': (0, 100), 'validation': (0, 100), 'extrapolation': (100, 1000)}}
    ]
    
    # Initial value ranges (will be updated during curriculum learning)
    value_ranges = curriculum_stages[0]['ranges']
    
    # Ensure model and criterion are on the correct device
    model = model.to(args.device)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(args.device)
    
    # Create trainer
    trainer = NumericalCompetenceTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        config=config,
        operations=operations,
        value_ranges=value_ranges,
        experiment_name="NumericalModule"
    )
    
    # Create a custom data wrapper for the trainer that adapts our TensorDataset format
    # to what the NumericalCompetenceTrainer expects
    class DatasetWrapper:
        def __init__(self, data_loader, device):
            self.data_loader = data_loader
            self.iterator = None
            self.device = device
            
        def __iter__(self):
            self.iterator = iter(self.data_loader)
            return self
            
        def __next__(self):
            try:
                batch = next(self.iterator)
                # Unpack batch and move to the correct device
                h1, h2, h_op, targets = [item.to(self.device) for item in batch]
                return {
                    'operands': (h1, h2, h_op),
                    'results': targets
                }
            except StopIteration:
                self.iterator = iter(self.data_loader)  # Reset for next epoch
                raise StopIteration
                
        def __len__(self):
            return len(self.data_loader)
    
    # Wrap data loaders with our custom wrapper and ensure everything moves to the right device
    train_data = DatasetWrapper(train_loader, args.device)
    val_data = DatasetWrapper(val_loader, args.device)
    
    # Early stopping is handled in the training loop with curriculum learning
    
    # Training loop with curriculum learning
    print(f"Beginning training with curriculum learning ({args.epochs} total epochs)...")
    
    current_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Implement curriculum learning through stages
    for stage_idx, stage in enumerate(curriculum_stages):
        stage_epochs = stage['epochs']
        stage_ranges = stage['ranges']
        
        print(f"\nCurriculum Stage {stage_idx+1}: Epochs {current_epoch}-{current_epoch+stage_epochs-1}")
        print(f"  Training Range: {stage_ranges['train']}")
        print(f"  Validation Range: {stage_ranges['validation']}")
        print(f"  Extrapolation Range: {stage_ranges['extrapolation']}")
        
        # Update trainer's value ranges for this stage
        trainer.value_ranges = stage_ranges
        
        # Train for the specified number of epochs in this stage
        for stage_epoch in range(stage_epochs):
            epoch = current_epoch + stage_epoch
            
            # Train for one epoch using the decoder to get predictions
            train_metrics = trainer.train_epoch(train_data)
            
            # Validate using the decoder to get predictions
            val_metrics = trainer.validate(val_data)
            
            # Progress to the next stage if accuracy is high enough
            if stage_idx < len(curriculum_stages) - 1 and val_metrics['validation_accuracy'] > 0.7:
                print(f"  Reached target accuracy ({val_metrics['validation_accuracy']:.4f}), advancing to next stage.")
                break
        
            # Extract metrics
            train_loss = train_metrics['loss']
            val_loss = val_metrics['validation_loss']
            val_accuracy = val_metrics['validation_accuracy']
            extrapolation_accuracy = val_metrics['extrapolation_accuracy']
            
            # Print epoch summary
            print(f"Epoch {epoch}/{args.epochs-1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}, Extrapolation Accuracy: {extrapolation_accuracy:.4f}")
            
            # Check early stopping (only within current curriculum stage)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(args.model_dir, "best_model.pt")
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'extrapolation_accuracy': extrapolation_accuracy,
                    'config': {
                        'hidden_dim': args.hidden_dim,
                        'num_dim': args.num_dim,
                        'bit_linear': args.bit_linear,
                        'curriculum_stage': stage_idx,
                        'value_ranges': stage_ranges
                    }
                }
                try:
                    # Try to save with pickle to ensure compatibility
                    import pickle
                    import torch  # Make sure torch is imported in this scope
                    torch.save(checkpoint_data, best_model_path, pickle_module=pickle)
                except Exception as e:
                    print(f"Error saving with pickle: {e}")
                    # Fall back to standard save
                    import torch  # Make sure torch is imported in this scope
                    torch.save(checkpoint_data, best_model_path)
                
                print(f"  Saved new best model with val_loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"  Early stopping patience: {patience_counter}/{args.patience}")
                
                # Only apply early stopping within the final curriculum stage
                if patience_counter >= args.patience and stage_idx == len(curriculum_stages) - 1:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    current_epoch = args.epochs  # Skip remaining epochs
                    break
            
            # Save regular checkpoint
            if (epoch + 1) % config['checkpoint_interval'] == 0:
                checkpoint_path = os.path.join(args.model_dir, f"checkpoint_epoch_{epoch}.pt")
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'curriculum_stage': stage_idx
                }
                try:
                    # Try to save with pickle to ensure compatibility
                    import pickle
                    import torch  # Make sure torch is imported in this scope
                    torch.save(checkpoint_data, checkpoint_path, pickle_module=pickle)
                except Exception as e:
                    print(f"Error saving with pickle: {e}")
                    # Fall back to standard save
                    import torch  # Make sure torch is imported in this scope
                    torch.save(checkpoint_data, checkpoint_path)
        
        # Update current epoch counter
        current_epoch += 1
        
        # Stop if we've reached the total number of epochs
        if current_epoch >= args.epochs:
            break
    
    # Generate final training summary
    trainer.plot_training_trajectory()
    
    # Load best model for final evaluation
    best_model_path = os.path.join(args.model_dir, "best_model.pt")
    try:
        # Add necessary global for NumPy scalar types
        try:
            from numpy._core.multiarray import scalar
            import torch.serialization
            torch.serialization.add_safe_globals([scalar])
        except ImportError:
            print("WARNING: Could not import numpy._core.multiarray.scalar")
        
        # Load checkpoint with correct device placement
        checkpoint = torch.load(best_model_path, map_location=args.device, weights_only=False)
    except (RuntimeError, TypeError) as e:
        print(f"Error with weights_only=False: {e}")
        print("Trying legacy loading method...")
        import pickle
        checkpoint = torch.load(best_model_path, map_location=args.device, pickle_module=pickle)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation using our real extrapolation data
    extrapolation_data = DatasetWrapper(extrapolation_loader, args.device)
    final_metrics = trainer.validate(extrapolation_data)
    print("\nFinal Evaluation:")
    print(f"  Validation Accuracy: {final_metrics['validation_accuracy']:.4f}")
    print(f"  Extrapolation Accuracy: {final_metrics['extrapolation_accuracy']:.4f}")
    
    # Analyze per-operation performance
    print("\nPer-Operation Performance:")
    for op in operations:
        val_acc = final_metrics.get(f"validation_{op}_accuracy", 0.0)
        ext_acc = final_metrics.get(f"extrapolation_{op}_accuracy", 0.0)
        print(f"  {op.capitalize()}: Val Acc = {val_acc:.4f}, Ext Acc = {ext_acc:.4f}")
    
    return trainer


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load data
    train_dataset, val_dataset, extrapolation_dataset, operations = load_data(args.data_dir)
    
    # Create data loaders
    train_loader, val_loader, extrapolation_loader = create_dataloaders(
        train_dataset, val_dataset, extrapolation_dataset, args
    )
    
    # Create model
    model = create_model(args.hidden_dim, args.num_dim, args.bit_linear, args.device)
    
    # Train model
    trainer = train_numerical_module(
        model, train_loader, val_loader, extrapolation_loader, operations, args
    )
    
    print("Training complete!")


if __name__ == "__main__":
    main()
