#!/usr/bin/env python3
"""
Numerical Competence Module Training Script

This script trains the Numerical Competence Module, enabling precise arithmetic
operations that generalize beyond the training distribution. It supports both synthetic
data generation and loading real data, as well as BitNet quantization for efficient
inference.

Usage examples:
    # Train with synthetic data
    python examples/train_numerical.py --hidden_dim 768 --epochs 50 --mode synthetic
    
    # Train with real data
    python examples/train_numerical.py --train_data_path data/numerical/train_data.pt --val_data_path data/numerical/val_data.pt
    
    # Train with BitNet quantization
    python examples/train_numerical.py --hidden_dim 768 --epochs 50 --quantize --mode synthetic
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
from training.numerical_trainer import NumericalCompetenceTrainer
from src.core.bitnet_integration import apply_bitnet_quantization, get_bitnet_model


class NumericalDataset(torch.utils.data.Dataset):
    """
    Dataset for numerical operations with real embeddings.
    
    This dataset handles both synthetic data generation and loading
    real data from pre-computed embeddings.
    """
    
    def __init__(self, hidden_dim=768, size=10000, operations=None, value_range=None, device='cpu', 
                 real_data_path=None, embedding_model=None):
        """
        Initialize dataset for numerical operations.
        
        Args:
            hidden_dim: Dimension of hidden representations
            size: Number of examples to generate (for synthetic mode)
            operations: List of operations to include (add, subtract, multiply, divide)
            value_range: Range of values for operands (min_val, max_val)
            device: Device to store tensors on
            real_data_path: Path to real data (if None, generate synthetic)
            embedding_model: Model for generating embeddings (for real data)
        """
        self.hidden_dim = hidden_dim
        self.device = device
        self.size = size
        self.operations = operations or ['add', 'subtract', 'multiply', 'divide']
        self.value_range = value_range or (0, 100)
        self.embedding_model = embedding_model
        
        if real_data_path and os.path.exists(real_data_path):
            print(f"Loading real data from {real_data_path}")
            self.load_real_data(real_data_path)
        else:
            print(f"Generating synthetic data with {size} examples")
            self.generate_synthetic_data()
            
    def generate_synthetic_data(self):
        """Generate synthetic data for numerical operations."""
        min_val, max_val = self.value_range
        
        # Initialize storage
        self.h1_list = []
        self.h2_list = []
        self.h_op_list = []
        self.targets = []
        self.original_operands = []
        
        # Generate examples
        for _ in range(self.size):
            # Select random operation
            operation = np.random.choice(self.operations)
            
            # Generate operands with appropriate constraints
            a = np.random.randint(min_val, max_val)
            
            # For division, ensure clean division when possible
            if operation == 'divide':
                if a > 1 and np.random.random() < 0.8:  # 80% clean division
                    high_val = min(a, self.value_range[1] // 2)
                    if high_val > 1:  # Ensure we have a valid range for randint
                        b = np.random.randint(1, high_val)
                        b = b * (a // b) if a // b > 0 else b  # Make a divisible by b
                    else:
                        b = 1  # Default to 1 if we can't find a valid divisor
                else:
                    b = max(1, np.random.randint(min_val, max_val))  # Avoid division by zero
            else:
                b = np.random.randint(min_val, max_val)
                
            # Compute result based on operation
            if operation == 'add':
                result = a + b
            elif operation == 'subtract':
                result = a - b
            elif operation == 'multiply':
                result = a * b
            elif operation == 'divide':
                result = a / b
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
            # Create hidden state representations with correct dimension
            h1 = torch.randn(self.hidden_dim, device=self.device)
            h2 = torch.randn(self.hidden_dim, device=self.device)
            h_op = torch.randn(self.hidden_dim, device=self.device)
            
            # Add some weak signal about the actual values to make training possible
            # In a real scenario, these would be learned embeddings from a pre-trained model
            max_magnitude = max(max_val, abs(result))
            scaling_factor = 1.0 / max(1.0, max_magnitude)
            
            # Add a small signal to a random subset of dimensions - safely handling dimensions
            signal_size = min(20, max(1, self.hidden_dim // 4))  # Use at most 1/4 of dimensions, at least 1
            if signal_size > 0:
                signal_dims = np.random.choice(self.hidden_dim, size=min(signal_size, self.hidden_dim), replace=False)
                if len(signal_dims) > 0:
                    h1[signal_dims[0]] += a * scaling_factor * 0.5
                    h2[signal_dims[0]] += b * scaling_factor * 0.5
            
            # Add a signal about the operation type
            op_idx = min(self.operations.index(operation), len(self.operations)-1)
            op_dim = np.random.randint(0, self.hidden_dim)  # Safer than choice with size param
            h_op[op_dim] += (op_idx + 1) * 0.5
            
            # Store the example
            self.h1_list.append(h1)
            self.h2_list.append(h2)
            self.h_op_list.append(h_op)
            self.targets.append(torch.tensor([result], dtype=torch.float32, device=self.device))
            self.original_operands.append((a, b, operation))
            
        # Convert lists to tensors
        self.h1_data = torch.stack(self.h1_list)
        self.h2_data = torch.stack(self.h2_list)
        self.h_op_data = torch.stack(self.h_op_list)
        self.target_data = torch.stack(self.targets)
            
    def load_real_data(self, data_path):
        """Load real data with pre-computed embeddings."""
        data = torch.load(data_path, map_location=self.device, weights_only=False)
        
        # Extract data components
        self.h1_data = data.get('h1_data').to(self.device)
        self.h2_data = data.get('h2_data').to(self.device)
        self.h_op_data = data.get('h_op_data').to(self.device)
        self.target_data = data.get('target_data').to(self.device)
        self.original_operands = data.get('original_operands', [])
        self.operations = data.get('operations', ['add', 'subtract', 'multiply', 'divide'])
        
        # Get hidden dimension from the data
        self.hidden_dim = data.get('hidden_dim', self.h1_data.shape[1])
        
        # Update size based on loaded data
        self.size = len(self.h1_data)
        
        print(f"Loaded dataset with {self.size} samples and hidden dimension {self.hidden_dim}")
        
    def save_data(self, output_path):
        """Save dataset to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for saving
        data = {
            'h1_data': self.h1_data.cpu(),
            'h2_data': self.h2_data.cpu(),
            'h_op_data': self.h_op_data.cpu(),
            'target_data': self.target_data.cpu(),
            'original_operands': self.original_operands,
            'operations': self.operations,
            'value_range': self.value_range,
            'hidden_dim': self.hidden_dim
        }
        
        # Save data
        torch.save(data, output_path)
        print(f"Saved data to {output_path}")
        
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
    parser = argparse.ArgumentParser(description="Train Numerical Competence Module")
    
    # Data configuration
    parser.add_argument("--train_data_path", type=str, default=None,
                      help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default=None,
                      help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="outputs/numerical",
                      help="Directory for outputs")
    parser.add_argument("--mode", type=str, default="synthetic",
                      choices=["synthetic", "real"],
                      help="Data generation mode")
    
    # Model configuration
    parser.add_argument("--hidden_dim", type=int, default=768,
                      help="Hidden dimension size")
    parser.add_argument("--num_dim", type=int, default=32,
                      help="Numerical representation dimension")
    parser.add_argument("--quantize", action="store_true",
                      help="Use BitNet quantization")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                      help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw",
                      choices=["adam", "adamw", "sgd"],
                      help="Optimizer to use")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                      help="Patience for early stopping")
    parser.add_argument("--save_interval", type=int, default=5,
                      help="Epoch interval for saving checkpoints")
    
    # Runtime configuration
    parser.add_argument("--device", type=str, default=None,
                      help="Device to use (cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--data_parallel", action="store_true",
                      help="Use DataParallel for multi-GPU training")
    
    # Parse and validate arguments
    args = parser.parse_args()
    
    # Infer device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def create_model(args):
    """Create and initialize numerical module."""
    print(f"Creating numerical module with hidden_dim={args.hidden_dim}, num_dim={args.num_dim}")
    
    # Create model with BitNet quantization if specified
    model = NumericalModule(
        hidden_dim=args.hidden_dim,
        num_dim=args.num_dim,
        bit_linear=args.quantize  # Use BitLinear layers if quantization is enabled
    )
    
    # Move model to device
    model = model.to(args.device)
    
    # Apply DataParallel if enabled and multiple GPUs available
    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        
    return model


def create_optimizer(model, args):
    """Create optimizer based on arguments."""
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def create_datasets(args):
    """Create training and validation datasets."""
    # Default configurations
    operations = ['add', 'subtract', 'multiply', 'divide']
    train_range = (0, 100)
    val_range = (0, 100)
    extrapolation_range = (100, 1000)
    
    # Training dataset
    if args.train_data_path and os.path.exists(args.train_data_path):
        # Load real training data
        train_dataset = NumericalDataset(
            hidden_dim=args.hidden_dim,
            device=args.device,
            real_data_path=args.train_data_path
        )
        
        # Update hidden_dim to match the loaded data
        if hasattr(train_dataset, 'hidden_dim'):
            args.hidden_dim = train_dataset.hidden_dim
    else:
        # Generate synthetic training data
        train_dataset = NumericalDataset(
            hidden_dim=args.hidden_dim,
            size=10000,  # Default size for synthetic data
            operations=operations,
            value_range=train_range,
            device=args.device
        )
        
        # Save synthetic data if in synthetic mode
        if args.mode == "synthetic":
            os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
            train_dataset.save_data(os.path.join(args.output_dir, "data", "train_data.pt"))
    
    # Validation dataset
    if args.val_data_path and os.path.exists(args.val_data_path):
        # Load real validation data
        val_dataset = NumericalDataset(
            hidden_dim=args.hidden_dim,  # Use updated hidden_dim
            device=args.device,
            real_data_path=args.val_data_path
        )
    else:
        # Generate synthetic validation data
        val_dataset = NumericalDataset(
            hidden_dim=args.hidden_dim,  # Use updated hidden_dim
            size=2000,  # Default size for synthetic data
            operations=operations,
            value_range=val_range,
            device=args.device
        )
        
        # Save synthetic data if in synthetic mode
        if args.mode == "synthetic":
            os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
            val_dataset.save_data(os.path.join(args.output_dir, "data", "val_data.pt"))
    
    # Check for real extrapolation data
    extrapolation_data_path = os.path.join(args.output_dir, "data", "extrapolation_data.pt")
    if os.path.exists(args.train_data_path.replace("train_data.pt", "extrapolation_data.pt")):
        # Use real extrapolation data if available
        extrapolation_data_path = args.train_data_path.replace("train_data.pt", "extrapolation_data.pt")
        extrapolation_dataset = NumericalDataset(
            hidden_dim=args.hidden_dim,  # Use updated hidden_dim
            device=args.device,
            real_data_path=extrapolation_data_path
        )
    else:
        # Generate synthetic extrapolation data matching the hidden dimension of training data
        extrapolation_dataset = NumericalDataset(
            hidden_dim=args.hidden_dim,  # Use updated hidden_dim
            size=2000,  # Default size for synthetic data
            operations=operations,
            value_range=extrapolation_range,
            device=args.device
        )
        
        # Save extrapolation data
        os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
        extrapolation_dataset.save_data(os.path.join(args.output_dir, "data", "extrapolation_data.pt"))
    
    return train_dataset, val_dataset, extrapolation_dataset


def train_model(model, datasets, args):
    """Train the numerical competence module."""
    # Unpack datasets
    train_dataset, val_dataset, extrapolation_dataset = datasets
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for simple dataset
        pin_memory=False  # Disable pin_memory to avoid CUDA tensor issue
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False  # Disable pin_memory to avoid CUDA tensor issue
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, args)
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Create temp decoder head for proper decoding during training
    class TemporaryDecoderHead(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.decoder = nn.Linear(hidden_dim, 1)
            
        def forward(self, hidden):
            return self.decoder(hidden)
    
    # Add temporary decoder head
    decoder_head = TemporaryDecoderHead(args.hidden_dim).to(args.device)
    decoder_optimizer = torch.optim.AdamW(decoder_head.parameters(), lr=args.learning_rate)
    
    # Create training configuration
    config = {
        "hidden_dim": args.hidden_dim,
        "num_dim": args.num_dim,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "early_stopping_patience": args.early_stopping_patience,
        "save_interval": args.save_interval,
        "operations": ['add', 'subtract', 'multiply', 'divide'],
        "value_ranges": {
            "train": train_dataset.value_range,
            "validation": val_dataset.value_range,
            "extrapolation": extrapolation_dataset.value_range
        },
        "quantize": args.quantize,
        "plot_interval": 5
    }
    
    # Print training configuration
    print("\n=== Training Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=============================\n")
    
    # Initialize training history
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    extrapolation_acc_history = []
    
    # Track best model for early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    def evaluate(dataset, loader_name):
        """Evaluate model on a dataset."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Create loader if provided a dataset
        if isinstance(dataset, torch.utils.data.Dataset):
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False  # Disable pin_memory to avoid CUDA tensor issue
            )
        else:
            loader = dataset  # Assume dataset is already a DataLoader
            
        with torch.no_grad():
            for batch in loader:
                # Extract data
                h1 = batch['h1']
                h2 = batch['h2']
                h_op = batch['h_op']
                targets = batch['target'].to(args.device).float()
                
                # Forward pass through numerical module
                result_hidden, op_weights = model(h1, h2, h_op)
                
                # Forward pass through temporary decoder
                predictions = decoder_head(result_hidden)
                
                # Calculate loss
                loss = criterion(predictions, targets)
                total_loss += loss.item() * len(h1)
                
                # Calculate accuracy with tolerance
                abs_error = torch.abs(predictions - targets)
                rel_tolerance = torch.abs(targets) * 0.05  # 5% relative tolerance
                min_tolerance = torch.tensor(0.01, device=args.device)
                tolerance = torch.max(rel_tolerance, min_tolerance)
                
                correct += torch.sum((abs_error <= tolerance).float()).item()
                total += len(h1)
        
        # Calculate metrics
        avg_loss = total_loss / total if total > 0 else float('inf')
        accuracy = correct / total if total > 0 else 0
        
        print(f"{loader_name} - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy
    
    # Training loop
    print("\n=== Beginning Training ===")
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Initialize progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            # Extract data
            h1 = batch['h1']
            h2 = batch['h2']
            h_op = batch['h_op']
            targets = batch['target'].to(args.device).float()
            
            # Reset gradients
            optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # Forward pass through numerical module
            result_hidden, op_weights = model(h1, h2, h_op)
            
            # Forward pass through temporary decoder
            predictions = decoder_head(result_hidden)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            decoder_optimizer.step()
            
            # Calculate accuracy with tolerance
            abs_error = torch.abs(predictions - targets)
            rel_tolerance = torch.abs(targets) * 0.05  # 5% relative tolerance
            min_tolerance = torch.tensor(0.01, device=args.device)
            tolerance = torch.max(rel_tolerance, min_tolerance)
            
            batch_correct = torch.sum((abs_error <= tolerance).float()).item()
            
            # Update metrics
            train_loss += loss.item() * len(h1)
            train_correct += batch_correct
            train_total += len(h1)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': batch_correct / len(h1) if len(h1) > 0 else 0
            })
        
        # Calculate epoch metrics
        train_epoch_loss = train_loss / train_total if train_total > 0 else float('inf')
        train_epoch_acc = train_correct / train_total if train_total > 0 else 0
        
        # Evaluate on validation set
        val_epoch_loss, val_epoch_acc = evaluate(val_dataset, "Validation")
        
        # Evaluate on extrapolation set
        _, extrapolation_epoch_acc = evaluate(extrapolation_dataset, "Extrapolation")
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_epoch_loss:.6f}, Train Acc: {train_epoch_acc:.4f}, Val Loss: {val_epoch_loss:.6f}, Val Acc: {val_epoch_acc:.4f}, Extrapolation Acc: {extrapolation_epoch_acc:.4f}")
        
        # Update history
        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(val_epoch_loss)
        train_acc_history.append(train_epoch_acc)
        val_acc_history.append(val_epoch_acc)
        extrapolation_acc_history.append(extrapolation_epoch_acc)
        
        # Check for early stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            early_stopping_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'decoder_state_dict': decoder_head.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'loss': best_val_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"), weights_only=False)
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")
            
            if early_stopping_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save checkpoint at regular intervals
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'decoder_state_dict': decoder_head.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'loss': val_epoch_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"), weights_only=False)
            print(f"Saved checkpoint for epoch {epoch+1}")
            
        # Generate plots at regular intervals
        if (epoch + 1) % config["plot_interval"] == 0:
            # Create directory for plots
            os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
            
            # Create accuracy plot
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, 'b-', label='Training')
            plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, 'g-', label='Validation')
            plt.plot(range(1, len(extrapolation_acc_history) + 1), extrapolation_acc_history, 'r-', label='Extrapolation')
            plt.title('Numerical Competence Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(args.output_dir, "plots", f"accuracy_epoch_{epoch+1}.png"))
            plt.close()
            
            # Create loss plot
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'b-', label='Training')
            plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, 'g-', label='Validation')
            plt.title('Numerical Competence Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(args.output_dir, "plots", f"loss_epoch_{epoch+1}.png"))
            plt.close()
    
    # Generate final plots
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    
    # Create accuracy plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, 'b-', label='Training')
    plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, 'g-', label='Validation')
    plt.plot(range(1, len(extrapolation_acc_history) + 1), extrapolation_acc_history, 'r-', label='Extrapolation')
    plt.title('Numerical Competence Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "plots", "accuracy_final.png"))
    plt.close()
    
    # Create loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'b-', label='Training')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, 'g-', label='Validation')
    plt.title('Numerical Competence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "plots", "loss_final.png"))
    plt.close()
    
    # Print final performance
    print("\n=== Final Performance ===")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Final Validation Accuracy: {val_acc_history[-1]:.4f}")
    print(f"Final Extrapolation Accuracy: {extrapolation_acc_history[-1]:.4f}")
    print("=========================\n")
    
    # Return training history
    return {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'extrapolation_acc': extrapolation_acc_history
    }


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create datasets first to detect hidden dimension
    datasets = create_datasets(args)
    
    # Get hidden dimension from data if available
    train_dataset, _, _ = datasets
    if hasattr(train_dataset, 'hidden_dim'):
        detected_hidden_dim = train_dataset.hidden_dim
        if detected_hidden_dim != args.hidden_dim:
            print(f"Note: Updating hidden dimension from {args.hidden_dim} to {detected_hidden_dim} to match data")
            args.hidden_dim = detected_hidden_dim
    
    # Create model with updated hidden_dim
    model = create_model(args)
    
    # Train model
    history = train_model(model, datasets, args)
    
    # Print completion message
    print(f"Training complete. Model and outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
