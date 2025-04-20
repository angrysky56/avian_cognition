"""
Metacognition Training Example

This script demonstrates how to train the metacognition module using
the specialized training protocol, showcasing the emergence of calibrated
self-awareness in the avian cognitive architecture.
"""

import os
import sys
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.metacognition import MetacognitionModule
from src.core.bitnet import BitLinear
from training.metacognition_trainer import MetacognitionTrainer


class SyntheticMetacognitionDataset(Dataset):
    """
    Generates synthetic data for metacognition training.
    
    Creates pairs of hidden state representations and binary correctness
    indicators, simulating the inputs needed to train a metacognition
    module to predict confidence.
    
    The synthetic hidden states contain latent information about their
    correctness through conditional patterns, allowing the metacognition
    module to learn the relationship between representation structure
    and correctness probability.
    
    Attributes:
        hidden_dim: Dimension of hidden state representations
        size: Number of examples in dataset
        correctness_patterns: Patterns associated with correctness
        error_patterns: Patterns associated with errors
        hidden_states: Generated hidden state representations
        correctness: Generated correctness indicators
    """
    
    def __init__(self, hidden_dim=768, size=10000, pattern_strength=0.5, seed=42):
        """
        Initialize synthetic metacognition dataset.
        
        Args:
            hidden_dim: Dimension of hidden state representations
            size: Number of examples in dataset
            pattern_strength: Strength of correctness/error patterns
            seed: Random seed for reproducibility
        """
        self.hidden_dim = hidden_dim
        self.size = size
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate correctness patterns (features that indicate correctness)
        self.correctness_patterns = torch.randn(5, hidden_dim)
        self.error_patterns = torch.randn(5, hidden_dim)
        
        # Generate synthetic data
        self._generate_data(pattern_strength)
        
    def _generate_data(self, pattern_strength):
        """
        Generate synthetic hidden states and correctness indicators.
        
        Creates hidden states with latent patterns that correlate with
        correctness, allowing the metacognition module to learn to
        predict the probability of correctness from the hidden state.
        
        Args:
            pattern_strength: Strength of correctness/error patterns
        """
        # Initialize storage
        self.hidden_states = torch.zeros(self.size, self.hidden_dim)
        self.correctness = torch.zeros(self.size, 1)
        
        # Generate each example
        for i in range(self.size):
            # Determine if this example is correct (50% probability)
            is_correct = np.random.rand() > 0.5
            
            # Base representation (random noise)
            hidden_state = torch.randn(self.hidden_dim)
            
            if is_correct:
                # Add correctness patterns with random strength
                for pattern in self.correctness_patterns:
                    if np.random.rand() > 0.5:  # Randomly apply patterns
                        strength = pattern_strength * np.random.rand()
                        hidden_state += strength * pattern
            else:
                # Add error patterns with random strength
                for pattern in self.error_patterns:
                    if np.random.rand() > 0.5:  # Randomly apply patterns
                        strength = pattern_strength * np.random.rand()
                        hidden_state += strength * pattern
            
            # Store example
            self.hidden_states[i] = hidden_state
            self.correctness[i] = 1.0 if is_correct else 0.0
            
    def __len__(self):
        """Return the size of the dataset."""
        return self.size
        
    def __getitem__(self, idx):
        """
        Retrieve a single example.
        
        Args:
            idx: Index of example to retrieve
            
        Returns:
            sample: Dictionary containing hidden state and correctness
        """
        return {
            'hidden_states': self.hidden_states[idx],
            'correctness': self.correctness[idx]
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Metacognition Training Example")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for representations"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply BitNet quantization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    return parser.parse_args()


def create_model(args):
    """
    Create metacognition module for training.
    
    Args:
        args: Command line arguments
        
    Returns:
        model: Metacognition module
    """
    print(f"Creating metacognition module with hidden_dim={args.hidden_dim}")
    
    # Create metacognition module
    model = MetacognitionModule(
        hidden_dim=args.hidden_dim,
        bit_linear=args.quantize
    )
    
    return model.to(args.device)


def create_datasets(args):
    """
    Create training and validation datasets.
    
    Args:
        args: Command line arguments
        
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    print("Creating synthetic datasets for metacognition training")
    
    # Create training dataset
    train_dataset = SyntheticMetacognitionDataset(
        hidden_dim=args.hidden_dim,
        size=10000,
        pattern_strength=0.5,
        seed=42
    )
    
    # Create validation dataset (different seed)
    val_dataset = SyntheticMetacognitionDataset(
        hidden_dim=args.hidden_dim,
        size=2000,
        pattern_strength=0.5,
        seed=43
    )
    
    return train_dataset, val_dataset


def main():
    """Main training function."""
    args = parse_args()
    
    # Print configuration
    print("=== Metacognition Training Configuration ===")
    print(f"Device: {args.device}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"BitNet quantization: {args.quantize}")
    print("=============================================")
    
    # Create model
    model = create_model(args)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(args)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Define loss function
    criterion = torch.nn.BCELoss()
    
    # Create trainer configuration
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'checkpoint_interval': 5,
        'plot_interval': 2
    }
    
    # Create trainer
    trainer = MetacognitionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        config=config,
        confidence_threshold=0.5,
        experiment_name="MetacognitionExample"
    )
    
    # Train model
    print("\nBeginning training process...")
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs
    )
    
    # Generate final trajectory plot
    trainer.plot_training_trajectory()
    
    print("\nTraining complete!")
    print(f"Final validation metrics: {training_history['val_metrics'][-1]}")
    print(f"Checkpoints saved to: {trainer.log_dir}/checkpoints")
    print(f"Visualizations saved to: {trainer.log_dir}/plots")
    

if __name__ == "__main__":
    main()
