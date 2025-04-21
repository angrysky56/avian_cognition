"""
Numerical Module Training Script (Placeholder)

This script provides a basic framework for training the NumericalModule.
It uses synthetic data representing simple arithmetic problems (e.g., a + b = c) 
and attempts to train the module to predict the scalar result 'c'.

**IMPORTANT:** This requires adding a temporary decoding head to the module 
during training to predict the scalar result. It also uses highly simplified 
input representations (random vectors). For practical use, replace the synthetic 
data with realistic inputs (e.g., hidden states corresponding to numbers and 
operations from a larger model) and potentially use a more sophisticated 
loss or task formulation.
"""

import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn # Needed for nn.MSELoss and nn.Linear
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm # Progress bars
import time
import random

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Attempt imports
try:
    from src.modules.numerical import NumericalModule
except ImportError:
    print(f"Error: Failed to import NumericalModule from src.modules.numerical")
    print(f"Ensure '{PROJECT_ROOT}' is in your Python path and the file exists.")
    sys.exit(1)

try:
    from src.core.bitnet import BitLinear, NALULayer # Check availability
except ImportError:
     print(f"Warning: Failed to import BitLinear/NALULayer from src.core.bitnet.")

# Placeholder Trainer Class (Includes early stopping and best model saving)
class PlaceholderTrainer:
    def __init__(self, model, decoder_head, optimizer, criterion, device, config, experiment_name="NumericalExperiment"):
        self.model = model
        self.decoder_head = decoder_head # Specific to this training setup
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.experiment_name = experiment_name
        self.log_dir = os.path.join("logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_metric = float('inf') if config.get('early_stopping_mode', 'min') == 'min' else float('-inf')
        self.epochs_no_improve = 0
        self.best_epoch = -1
        # Store module state and decoder head state separately if needed
        self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': []} 

    def _save_checkpoint(self, epoch, is_best=False):
        filename = f"checkpoint_epoch_{epoch+1}.pth"
        if is_best:
             filename = "checkpoint_best.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        # Save both the main module and the temporary decoder head
        state = {
            'model_state_dict': self.model.state_dict(),
            'decoder_head_state_dict': self.decoder_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_metric': self.best_metric
        }
        torch.save(state, filepath)

    def _train_epoch(self, train_loader):
        self.model.train()
        self.decoder_head.train() # Train decoder head as well
        total_loss = 0.0
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Assuming batch yields dict {'h1': tensor, 'h2': tensor, 'h_op': tensor, 'target_result': tensor}
            h1 = batch['h1'].to(self.device)
            h2 = batch['h2'].to(self.device)
            h_op = batch['h_op'].to(self.device)
            target_result = batch['target_result'].to(self.device) # Scalar target

            self.optimizer.zero_grad()
            
            # --- Model Forward Pass ---
            # Get the hidden representation of the result from the numerical module
            result_hidden, op_weights = self.model(h1, h2, h_op)
            
            # --- Decoding and Loss ---
            # Use the temporary decoder head to predict the scalar result
            predicted_result = self.decoder_head(result_hidden) # Shape: [batch_size, 1]
            
            # Calculate loss between predicted scalar and target scalar
            loss = self.criterion(predicted_result, target_result) 
            # --- End Loss ---

            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader):
        self.model.eval()
        self.decoder_head.eval() # Eval decoder head
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in pbar:
                h1 = batch['h1'].to(self.device)
                h2 = batch['h2'].to(self.device)
                h_op = batch['h_op'].to(self.device)
                target_result = batch['target_result'].to(self.device)

                result_hidden, _ = self.model(h1, h2, h_op)
                predicted_result = self.decoder_head(result_hidden)
                
                loss = self.criterion(predicted_result, target_result)
                total_loss += loss.item()
                
                all_preds.append(predicted_result.cpu())
                all_targets.append(target_result.cpu())

        avg_loss = total_loss / len(val_loader)
        
        # --- Calculate additional validation metrics ---
        preds_cat = torch.cat(all_preds, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        # Example: Mean Absolute Error (MAE)
        mae = F.l1_loss(preds_cat, targets_cat).item()
        metrics = {'val_mae': mae} # Define relevant metrics
        # ---------------------------------------------

        return avg_loss, metrics

    def train(self, train_loader, val_loader):
        # (Training loop identical to other trainers - see Bayesian or Planning)
        epochs = self.config.get('epochs', 10)
        patience = self.config.get('early_stopping_patience', 3)
        metric_name = self.config.get('early_stopping_metric', 'val_loss') # Could use val_mae
        metric_mode = self.config.get('early_stopping_mode', 'min')
        print(f"Starting training for {epochs} epochs...")
        print(f"Early stopping: Monitor='{metric_name}', Patience={patience}, Mode='{metric_mode}'")

        for epoch in range(epochs):
            start_time = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate_epoch(val_loader)
            epoch_time = time.time() - start_time

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)

            print(f"Epoch {epoch+1}/{epochs} [{epoch_time:.2f}s] - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", end="")
            for name, value in val_metrics.items():
                 print(f", {name}: {value:.4f}", end="")
            print()

            # Early Stopping & Best Model Check
            current_metric = val_loss if metric_name == 'val_loss' else val_metrics.get(metric_name, None)
            if current_metric is None and metric_name != 'val_loss':
                 print(f"Warning: Early stopping metric '{metric_name}' not found. Using val_loss.")
                 current_metric = val_loss
                 metric_name = 'val_loss'
                 metric_mode = 'min'
                 
            is_better = False
            if metric_mode == 'min':
                 if current_metric < self.best_metric: is_better = True
            else: # mode == 'max'
                 if current_metric > self.best_metric: is_better = True
            
            if is_better:
                print(f"  Validation metric ({metric_name}) improved ({self.best_metric:.4f} -> {current_metric:.4f}). Saving best model...")
                self.best_metric = current_metric
                self.epochs_no_improve = 0
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True) # Saves model + decoder head state
            else:
                self.epochs_no_improve += 1
                print(f"  Validation metric ({metric_name}) did not improve for {self.epochs_no_improve} epoch(s).")

            if patience > 0 and self.epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
        
        print(f"Training finished. Best model from epoch {self.best_epoch+1} saved.")
        best_model_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pth")
        if os.path.exists(best_model_path):
             print(f"Loading best model weights from {best_model_path}")
             checkpoint = torch.load(best_model_path, map_location=self.device)
             self.model.load_state_dict(checkpoint['model_state_dict'])
             self.decoder_head.load_state_dict(checkpoint['decoder_head_state_dict'])
             
        final_history = {
             'train_loss': self.history['train_loss'],
             'val_loss': self.history['val_loss'],
             'val_metrics': self.history['val_metrics'],
             'best_epoch': self.best_epoch + 1,
             'best_val_metrics': self.history['val_metrics'][self.best_epoch] if self.best_epoch >= 0 else None
        }
        return final_history


class SyntheticNumericalDataset(Dataset):
    """
    Generates synthetic data for **debugging/demonstrating** numerical training.

    Creates random hidden state vectors for operands/operation and a scalar 
    target result based on a simple arithmetic operation.
    **Uses random vectors for hidden states - replace with meaningful representations.**
    """
    def __init__(self, 
                 num_samples: int = 10000, 
                 hidden_dim: int = 256, 
                 value_range: tuple = (0, 100), # Range for operands
                 operations: list = ['add', 'subtract', 'multiply'], # Exclude divide for simplicity?
                 seed: int = 42):
        """Initialize the synthetic dataset."""
        self.num_samples = num_samples
        self.hidden_dim = hidden_dim
        self.value_range = value_range
        self.operations = operations
        self.num_ops = len(operations)
        self.seed = seed
        
        print(f"Generating {num_samples} synthetic numerical samples...")
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed) # For random.choice

        self.h1_states = torch.randn(num_samples, hidden_dim)
        self.h2_states = torch.randn(num_samples, hidden_dim)
        self.h_op_states = torch.randn(num_samples, hidden_dim)
        self.target_results = torch.zeros(num_samples, 1)

        min_val, max_val = value_range
        for i in range(num_samples):
            a = random.uniform(min_val, max_val)
            b = random.uniform(min_val, max_val)
            op_name = random.choice(self.operations)
            op_idx = self.operations.index(op_name)

            if op_name == 'add': result = a + b
            elif op_name == 'subtract': result = a - b
            elif op_name == 'multiply': result = a * b
            elif op_name == 'divide': result = a / (b + 1e-6) # Avoid division by zero
            else: result = 0.0 # Should not happen

            # Store target result
            self.target_results[i] = result
            
            # --- Placeholder: Inject minimal info into random hidden states ---
            # This is highly artificial and should be replaced by representations
            # learned or extracted from a real model/task.
            self.h1_states[i, 0] = a / max_val if max_val != 0 else a # Normalize roughly
            self.h2_states[i, 0] = b / max_val if max_val != 0 else b
            # Use one-hot like encoding for operation in h_op
            self.h_op_states[i, op_idx] = 1.0 
            self.h_op_states[i, self.num_ops:] *= 0.1 # Dampen other dimensions
            # ------------------------------------------------------------------

        print("Synthetic data generation complete.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        if idx >= self.num_samples:
            raise IndexError("Index out of bounds")
        return {
            'h1': self.h1_states[idx].float(),
            'h2': self.h2_states[idx].float(),
            'h_op': self.h_op_states[idx].float(),
            'target_result': self.target_results[idx].float()
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Numerical Module Training (Placeholder)")
    # Model/Data Args
    parser.add_argument("--hidden_dim", type=int, default=256, help="Input hidden dimension")
    parser.add_argument("--num_dim", type=int, default=32, help="Internal numerical dimension")
    parser.add_argument("--quantize", action="store_true", help="Use BitLinear layers")
    # Training Args
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    # Early Stopping Args
    parser.add_argument("--early_stopping_patience", type=int, default=7, help="Patience for early stopping") # Might need more patience
    parser.add_argument("--early_stopping_metric", type=str, default="val_mae", help="Metric for early stopping (e.g., val_mae)")
    parser.add_argument("--early_stopping_mode", type=str, default="min", choices=["min", "max"], help="Early stopping mode")
    # Runtime Args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count(), 2), help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=789, help="Random seed")
    parser.add_argument("--experiment_name", type=str, default="NumericalSynthetic", help="Logging directory name")
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_model_and_head(args):
    """Creates the NumericalModule and a temporary decoder head."""
    print(f"Creating NumericalModule (hidden_dim={args.hidden_dim}, num_dim={args.num_dim}, "
          f"BitLinear={args.quantize})")
    model = NumericalModule(
        hidden_dim=args.hidden_dim,
        num_dim=args.num_dim,
        bit_linear=args.quantize
    )
    # --- Temporary Decoder Head ---
    # Add a simple linear head to decode the module's output hidden state 
    # into a scalar prediction for training loss calculation.
    # This head is *only* used for this specific training setup.
    decoder_head = nn.Linear(args.hidden_dim, 1) 
    print("Created temporary nn.Linear decoder head (hidden_dim -> 1) for training.")
    # -----------------------------
    return model.to(args.device), decoder_head.to(args.device)

def create_datasets(args):
    """Creates synthetic training and validation datasets."""
    print("Creating synthetic Numerical datasets (FOR DEBUGGING/DEMO ONLY)...")
    print("Replace with real data loading using meaningful input representations.")
    train_dataset = SyntheticNumericalDataset(
        num_samples=20000, # Needs more data potentially
        hidden_dim=args.hidden_dim,
        seed=args.seed
    )
    val_dataset = SyntheticNumericalDataset(
        num_samples=4000,
        hidden_dim=args.hidden_dim,
        seed=args.seed + 1
    )
    return train_dataset, val_dataset

def main():
    args = parse_args()
    set_seed(args.seed)

    print("=== Numerical Module Training Configuration ===")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("===============================================")
    
    # Setup
    model, decoder_head = create_model_and_head(args)
    train_dataset, val_dataset = create_datasets(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Optimizer - Include parameters from both model and decoder head
    optimizer_params = list(model.parameters()) + list(decoder_head.parameters())
    if args.optimizer == "adamw":
         optimizer = torch.optim.AdamW(optimizer_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
         optimizer = torch.optim.Adam(optimizer_params, lr=args.learning_rate)
         
    # Loss Function (MSE for predicting scalar result)
    criterion = nn.MSELoss() 
    print("Using MSE loss for predicting scalar numerical result.")

    # Trainer
    trainer_config = {
        'epochs': args.epochs, 'learning_rate': args.learning_rate,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_metric': args.early_stopping_metric,
        'early_stopping_mode': args.early_stopping_mode,
    }
    print("\nInitializing PlaceholderTrainer for Numerical Module...")
    trainer = PlaceholderTrainer(
        model=model, 
        decoder_head=decoder_head, # Pass the decoder head
        optimizer=optimizer, 
        criterion=criterion, 
        device=args.device, 
        config=trainer_config, 
        experiment_name=args.experiment_name
    )
    
    # Training
    print(f"\nStarting training...")
    try:
        training_history = trainer.train(train_loader, val_loader)
        
        # Results
        print("\nTraining finished!")
        best_epoch = training_history.get('best_epoch', 'N/A')
        best_metrics = training_history.get('best_val_metrics', {})
        print(f"Best model checkpoint saved from epoch: {best_epoch}")
        print(f"Best validation metrics ({args.early_stopping_metric}):")
        for metric, value in best_metrics.items():
             print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
        print(f"\nCheckpoints saved to: {trainer.checkpoint_dir}")
        print("NOTE: Checkpoint includes both NumericalModule and temporary decoder head state.")

    except Exception as e:
         print(f"\nAn error occurred during training: {e}")
         import traceback
         traceback.print_exc()
         sys.exit(1)

if __name__ == "__main__":
    main()