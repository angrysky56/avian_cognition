"""
Metacognition Module Training Script

This script trains the MetacognitionModule to predict whether the main model's prediction
for a given hidden state is correct, helping the model develop calibrated confidence.

This module is trained using real data from a language model's hidden states and
correctness labels (whether the prediction was correct or not).
"""

import os
import sys
import torch
import argparse
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# Optional imports for datasets
try:
    from datasets import load_dataset
    has_hf_datasets = True
except ImportError:
    has_hf_datasets = False
    print("Warning: HuggingFace datasets not found. Install with pip install datasets")

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('metacognition_training')

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


class MetacognitionDataset(Dataset):
    """
    Dataset for metacognition training using real hidden states and correctness labels.
    
    This dataset loads pre-processed data containing language model hidden states
    and binary labels indicating whether the model's prediction was correct.
    
    Attributes:
        data_path (str): Path to the data files
        hidden_states (torch.Tensor): Hidden state representations from the model
        correctness (torch.Tensor): Binary correctness indicators (0 or 1)
    """
    
    def __init__(self, data_path: str, hidden_dim: int = 768, device="cpu"):
        """
        Initialize metacognition dataset from real model outputs.
        
        Args:
            data_path: Path to the directory or file containing the data
            hidden_dim: Dimension of hidden state representations
            device: Device to load tensors to
        """
        self.data_path = data_path
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Loads hidden states and correctness labels from files."""
        logger.info(f"Loading metacognition data from {self.data_path}")
        
        data_path = Path(self.data_path)
        
        # Check if path exists
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")
            
        # There are multiple ways to load the data depending on format:
        # 1. Loading from .pt files (PyTorch tensors)
        if data_path.is_file() and data_path.suffix == '.pt':
            logger.info("Loading data from PyTorch .pt file")
            data = torch.load(data_path, map_location=self.device)
            self.hidden_states = data.get('hidden_states')
            self.correctness = data.get('correctness')
        
        # 2. Loading from HuggingFace dataset
        elif has_hf_datasets and (data_path.is_dir() or data_path.as_posix().startswith('hf://')):
            logger.info("Loading data from HuggingFace dataset")
            dataset = load_dataset(self.data_path)
            
            # This assumes the dataset has columns named 'hidden_states' and 'correctness'
            # Adjust column names if your dataset uses different names
            hidden_states = []
            correctness = []
            
            # Get the split (train/validation)
            split = 'train' if 'train' in dataset else list(dataset.keys())[0]
            
            # Load the data
            for item in dataset[split]:
                hidden_states.append(torch.tensor(item['hidden_states']))
                correctness.append(torch.tensor([item['correctness']]))
            
            self.hidden_states = torch.stack(hidden_states)
            self.correctness = torch.stack(correctness)
            
        # 3. Custom loading logic can be added here for other formats
        else:
            raise ValueError(f"Unsupported data format for {self.data_path}. "
                           "Please provide a .pt file or HuggingFace dataset")
            
        # Verify data loaded correctly
        if self.hidden_states is None or self.correctness is None:
            raise ValueError("Failed to load hidden_states or correctness from the data source")
            
        # Verify shapes
        if len(self.hidden_states) != len(self.correctness):
            raise ValueError(f"Mismatch between hidden_states length ({len(self.hidden_states)}) "
                           f"and correctness length ({len(self.correctness)})")
                           
        if self.hidden_states.shape[1] != self.hidden_dim:
            logger.warning(f"Hidden state dimension in data ({self.hidden_states.shape[1]}) "
                          f"does not match expected dimension ({self.hidden_dim})")
        
        logger.info(f"Loaded {len(self.hidden_states)} examples with hidden_dim={self.hidden_states.shape[1]}")
        
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.hidden_states)
        
    def __getitem__(self, idx: int) -> dict:
        """Retrieve a single example."""
        if idx >= len(self.hidden_states):
            raise IndexError("Index out of bounds")
        return {
            'hidden_states': self.hidden_states[idx].float(), 
            'correctness': self.correctness[idx].float()
        }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Metacognition Module Training")
    
    # Model Args
    parser.add_argument("--hidden_dim", type=int, default=768, 
                        help="Hidden dimension for MetacognitionModule input")
    parser.add_argument("--intermediate_dim", type=int, default=None, 
                        help="Intermediate dimension (default: hidden_dim // 2)")
    parser.add_argument("--quantize", action="store_true", 
                        help="Use BitLinear layers instead of nn.Linear")
    
    # Data Args
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to training data (file or directory)")
    parser.add_argument("--val_data_path", type=str, required=True,
                        help="Path to validation data (file or directory)")
    parser.add_argument("--data_format", type=str, default="pt",
                        choices=["pt", "hf", "custom"],
                        help="Format of data files (pt: PyTorch, hf: HuggingFace, custom)")
    
    # Training Args
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate for optimizer")
    parser.add_argument("--optimizer", type=str, default="adamw", 
                        choices=["adam", "adamw"], 
                        help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay for AdamW optimizer")
    
    # Early Stopping Args
    parser.add_argument("--early_stopping_patience", type=int, default=5, 
                        help="Epochs to wait for improvement before stopping (0 to disable)")
    parser.add_argument("--early_stopping_metric", type=str, default="val_loss", 
                        help="Metric to monitor for early stopping (e.g., 'val_loss', 'val_ece')")
    parser.add_argument("--early_stopping_mode", type=str, default="min", 
                        choices=["min", "max"], 
                        help="Mode for early stopping ('min' for loss/ECE, 'max' for accuracy)")
    
    # Checkpoint Args
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/metacognition",
                        help="Directory to save model checkpoints")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    # Runtime Args
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, 
                        default=min(os.cpu_count() or 1, 4), 
                        help="Number of workers for DataLoader")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for logging directory (default: auto-generated from timestamp)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log training metrics every N batches")

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
    """Creates training and validation datasets using real model data."""
    logger.info("Loading real model data for metacognition training...")
    
    try:
        # Create train dataset
        train_dataset = MetacognitionDataset(
            data_path=args.train_data_path,
            hidden_dim=args.hidden_dim,
            device=args.device if args.device == "cpu" else "cpu"  # Load to CPU first, then move to GPU in batches
        )
        
        # Create validation dataset
        val_dataset = MetacognitionDataset(
            data_path=args.val_data_path,
            hidden_dim=args.hidden_dim,
            device=args.device if args.device == "cpu" else "cpu"
        )
        
        logger.info(f"Successfully loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise


def main():
    """Main training execution function."""
    args = parse_args()
    set_seed(args.seed)  # Set seed early

    # Create experiment name if not provided
    if args.experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"Metacognition_{timestamp}"

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Metacognition Training Configuration ===")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=============================================")
    
    # --- Setup ---
    logger.info("Creating metacognition model...")
    model = create_model(args)
    
    # Load checkpoint if provided
    if args.load_checkpoint:
        checkpoint_path = Path(args.load_checkpoint)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            logger.warning(f"Checkpoint {checkpoint_path} not found, starting from scratch")
    
    logger.info("Loading datasets...")
    try:
        train_dataset, val_dataset = create_datasets(args)
    
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,  # Larger batch size for validation
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False
        )
    except Exception as e:
        logger.error(f"Failed to create datasets: {e}")
        raise
    
    # Create Optimizer
    logger.info(f"Creating {args.optimizer} optimizer with lr={args.learning_rate}")
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:  # adam
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Loss Function (criterion) - Binary Cross Entropy is suitable for confidence prediction
    criterion = torch.nn.BCELoss()
    
    # --- Trainer Initialization ---
    trainer_config = {
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_metric': args.early_stopping_metric,
        'early_stopping_mode': args.early_stopping_mode,
        'checkpoint_interval': 1,
        'log_interval': args.log_interval,
        'hidden_dim': args.hidden_dim
    }

    logger.info("Initializing MetacognitionTrainer...")
    
    trainer = MetacognitionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        config=trainer_config,
        experiment_name=args.experiment_name
    )
    
    # --- Training ---
    logger.info(f"Starting training for {args.epochs} epochs (with early stopping patience {args.early_stopping_patience})...")
    
    try:
        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs  # Explicitly pass epochs
        )
        
        # --- Results ---
        logger.info("Training finished!")
        
        best_epoch = training_history.get('best_epoch', 'N/A')
        best_metrics = training_history.get('best_val_metrics', {})
        
        logger.info(f"Best model checkpoint saved from epoch: {best_epoch}")
        logger.info(f"Best validation metrics ({args.early_stopping_metric}):")
        for metric, value in best_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

        # Generate final plot if available
        if hasattr(trainer, 'plot_training_trajectory'):
            logger.info("Generating final training trajectory plot...")
            try:
                trainer.plot_training_trajectory()
                logger.info(f"Plot saved to {trainer.log_dir}")
            except Exception as plot_error:
                logger.warning(f"Failed to generate plot: {plot_error}")

        logger.info(f"Checkpoints saved to: {os.path.join(trainer.log_dir, 'checkpoints')}")
        logger.info(f"Logs saved to: {trainer.log_dir}")
        
        # Save final path to best model for easy reference
        best_model_path = os.path.join(trainer.log_dir, 'checkpoints', 'best_model.pt')
        with open(os.path.join(trainer.log_dir, 'best_model_path.txt'), 'w') as f:
            f.write(best_model_path)
        logger.info(f"Best model path saved to {os.path.join(trainer.log_dir, 'best_model_path.txt')}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save checkpoint on interrupt
        trainer.save_checkpoint(is_best=False, filename="interrupt_checkpoint.pt")
        logger.info(f"Interrupt checkpoint saved to {os.path.join(trainer.log_dir, 'checkpoints', 'interrupt_checkpoint.pt')}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  # Exit with error code

if __name__ == "__main__":
    main()