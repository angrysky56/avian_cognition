"""
Metacognition Module Training Script

This script trains the MetacognitionModule. 
Currently, it uses synthetic data for structural testing and demonstration.

**IMPORTANT:** For practical use, replace the SyntheticMetacognitionDataset 
with a dataset derived from a primary task (e.g., language modeling). This dataset 
should contain pairs of hidden states from the main model and binary labels 
indicating whether the main model's prediction for that state was correct.
"""

import os
import sys
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path for imports
# Ensure this path is correct relative to where you run the script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Attempt imports, provide guidance if they fail
try:
    from src.modules.metacognition import MetacognitionModule
except ImportError:
    print(f"Error: Failed to import MetacognitionModule from src.modules.metacognition")
    print(f"Ensure '{PROJECT_ROOT}' is in your Python path and the file exists.")
    sys.exit(1)

try:
    from src.core.bitnet import BitLinear # Not directly used here, but confirms availability
except ImportError:
     print(f"Warning: Failed to import BitLinear from src.core.bitnet.")
     # Continue, as MetacognitionModule might handle the fallback internally

try:
    # Assuming the trainer class exists here
    from training.metacognition_trainer import MetacognitionTrainer 
except ImportError:
     print(f"Error: Failed to import MetacognitionTrainer from training.metacognition_trainer")
     print(f"Please ensure this custom trainer class exists and '{PROJECT_ROOT}' is in the Python path.")
     sys.exit(1)


class SyntheticMetacognitionDataset(Dataset):
    """
    Generates synthetic data for **debugging/demonstrating** metacognition training.

    Creates hidden states with artificial patterns correlated with correctness.
    **This is NOT suitable for training a production-ready module.**
    Replace with real data (model hidden states + correctness labels) for actual use.
    
    Attributes:
        hidden_dim (int): Dimension of hidden state representations.
        size (int): Number of examples in the dataset.
        hidden_states (torch.Tensor): Generated hidden state representations.
        correctness (torch.Tensor): Generated binary correctness indicators (0 or 1).
    """
    
    def __init__(self, hidden_dim: int = 768, size: int = 10000, pattern_strength: float = 0.5, seed: int = 42):
        """
        Initialize synthetic metacognition dataset.
        """
        self.hidden_dim = hidden_dim
        self.size = size
        self.pattern_strength = pattern_strength
        self.seed = seed
        
        # Generate data on initialization
        self._generate_data()
        
    def _generate_data(self):
        """Generates synthetic hidden states and correctness indicators."""
        print(f"Generating {self.size} synthetic samples with hidden_dim={self.hidden_dim}, seed={self.seed}...")
        # Set random seed for reproducibility within data generation
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Define some arbitrary patterns
        # Using more patterns might make the task slightly more complex
        num_patterns = 10 
        correctness_patterns = torch.randn(num_patterns, self.hidden_dim)
        error_patterns = torch.randn(num_patterns, self.hidden_dim)
        
        # Initialize storage
        self.hidden_states = torch.zeros(self.size, self.hidden_dim)
        self.correctness = torch.zeros(self.size, 1)
        
        # Generate each example
        for i in range(self.size):
            # Determine if this example is correct (50% probability)
            is_correct = np.random.rand() > 0.5
            
            # Base representation (random noise)
            # Normalize base noise slightly
            hidden_state = torch.randn(self.hidden_dim) * 0.5 
            
            # Select patterns based on correctness
            patterns_to_use = self.correctness_patterns if is_correct else self.error_patterns
            
            # Add selected patterns with random strength and selection
            num_active_patterns = np.random.randint(1, num_patterns // 2 + 1) # Apply 1 to N/2 patterns
            active_indices = np.random.choice(num_patterns, num_active_patterns, replace=False)
            
            for idx in active_indices:
                # Vary strength more significantly
                strength = self.pattern_strength * (0.5 + np.random.rand() * 1.5) # Strength variation
                hidden_state += strength * patterns_to_use[idx]
            
            # Normalize the final hidden state (optional, but can help training)
            hidden_state = hidden_state / torch.norm(hidden_state)

            # Store example
            self.hidden_states[i] = hidden_state
            self.correctness[i] = 1.0 if is_correct else 0.0
            
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size
        
    def __getitem__(self, idx: int) -> dict:
        """Retrieve a single example."""
        if idx >= self.size:
            raise IndexError("Index out of bounds")
        return {
            # Ensure data types are consistent (e.g., float32)
            'hidden_states': self.hidden_states[idx].float(), 
            'correctness': self.correctness[idx].float()
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Metacognition Module Training")
    # Model/Data Args
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for MetacognitionModule input")
    parser.add_argument("--intermediate_dim", type=int, default=None, help="Intermediate dimension (default: hidden_dim // 2)")
    parser.add_argument("--quantize", action="store_true", help="Use BitLinear layers instead of nn.Linear")
    # Training Args
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs") # Increased max epochs
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training") # Slightly larger bs
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam optimizer") # Often lower LR is better
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    # Early Stopping Args
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Epochs to wait for improvement before stopping (0 to disable)")
    parser.add_argument("--early_stopping_metric", type=str, default="val_loss", help="Metric to monitor for early stopping (e.g., 'val_loss', 'val_ece')")
    parser.add_argument("--early_stopping_mode", type=str, default="min", choices=["min", "max"], help="Mode for early stopping ('min' for loss/ECE, 'max' for accuracy)")
    # Runtime Args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count(), 4), help="Number of workers for DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment_name", type=str, default="MetacognitionSynthetic", help="Name for logging directory")

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU.
    # Potentially add torch.backends.cudnn settings if needed
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def create_model(args):
    """Creates the MetacognitionModule instance."""
    print(f"Creating MetacognitionModule (hidden_dim={args.hidden_dim}, "
          f"intermediate_dim={args.intermediate_dim or args.hidden_dim // 2}, "
          f"BitLinear={args.quantize})")
    
    model = MetacognitionModule(
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim, # Pass None or value
        bit_linear=args.quantize
    )
    return model.to(args.device)


def create_datasets(args):
    """Creates synthetic training and validation datasets."""
    print("Creating synthetic datasets (FOR DEBUGGING/DEMO ONLY)...")
    print("Replace with real data loading for actual training.")
    
    train_dataset = SyntheticMetacognitionDataset(
        hidden_dim=args.hidden_dim,
        size=10000, # Example size
        pattern_strength=0.6, # Slightly stronger pattern?
        seed=args.seed # Use main seed for train set
    )
    
    val_dataset = SyntheticMetacognitionDataset(
        hidden_dim=args.hidden_dim,
        size=2000, # Example size
        pattern_strength=0.6,
        seed=args.seed + 1 # Use different seed for validation set
    )
    
    return train_dataset, val_dataset


def main():
    """Main training execution function."""
    args = parse_args()
    set_seed(args.seed) # Set seed early

    print("=== Metacognition Training Configuration ===")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=============================================")
    
    # --- Setup ---
    model = create_model(args)
    train_dataset, val_dataset = create_datasets(args)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False # Pin memory for faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2, # Can often use larger batch size for validation
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )
    
    # Create Optimizer
    if args.optimizer == "adamw":
         optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else: # adam
         optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) # Adam doesn't use weight_decay arg directly

    # Loss Function (criterion) - Binary Cross Entropy is suitable here
    criterion = torch.nn.BCELoss() 
    
    # --- Trainer Initialization ---
    # Assuming MetacognitionTrainer handles the training loop, evaluation,
    # logging, checkpointing (saving the *best* model based on validation metric), 
    # and early stopping logic.
    trainer_config = {
        # Pass essential parameters the trainer might need
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        # Early stopping config (assuming trainer uses these names)
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_metric': args.early_stopping_metric,
        'early_stopping_mode': args.early_stopping_mode,
        # Optional: intervals for checkpointing/logging if configurable
        'checkpoint_interval': 1, # Checkpoint every epoch? Or based on improvement
        'log_interval': 50, # Log every 50 batches?
        # Add any other config MetacognitionTrainer expects
    }

    print("\nInitializing MetacognitionTrainer...")
    print("NOTE: Assumes MetacognitionTrainer implements training loop, evaluation (acc, ECE),")
    print("      logging, checkpointing (best model based on val metric), and early stopping.")

    trainer = MetacognitionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        config=trainer_config, # Pass the structured config
        # Provide a name for logging outputs (TensorBoard, files, etc.)
        experiment_name=args.experiment_name 
        # confidence_threshold=0.5, # This seems less relevant for training itself
    )
    
    # --- Training ---
    print(f"\nStarting training for {args.epochs} epochs (with early stopping patience {args.early_stopping_patience})...")
    
    try:
        # Assuming train returns history including best metrics
        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            # epochs argument might be handled by trainer using config['epochs']
        )
        
        # --- Results ---
        print("\nTraining finished!")
        
        best_epoch = training_history.get('best_epoch', 'N/A')
        best_metrics = training_history.get('best_val_metrics', {})
        
        print(f"Best model checkpoint saved from epoch: {best_epoch}")
        print(f"Best validation metrics ({args.early_stopping_metric}):")
        for metric, value in best_metrics.items():
             # Format floats nicely
             if isinstance(value, float):
                 print(f"  {metric}: {value:.4f}")
             else:
                 print(f"  {metric}: {value}")

        # Optional: Generate final plot if trainer supports it
        if hasattr(trainer, 'plot_training_trajectory'):
            print("Generating final training trajectory plot...")
            trainer.plot_training_trajectory()

        print(f"\nCheckpoints saved to: {os.path.join(trainer.log_dir, 'checkpoints')}")
        print(f"Visualizations/Logs saved to: {trainer.log_dir}")

    except Exception as e:
         print(f"\nAn error occurred during training: {e}")
         import traceback
         traceback.print_exc()
         sys.exit(1) # Exit with error code

if __name__ == "__main__":
    main()