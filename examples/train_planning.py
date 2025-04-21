"""
Planning Module Training Script (Placeholder)

This script provides a basic framework for training the PlanningModule.
It currently uses synthetic data where the goal is (arbitrarily) to predict 
a target embedding based on input context.

**IMPORTANT:** For practical use, replace the SyntheticPlanningDataset with a 
dataset relevant to planning (e.g., goal-oriented dialogues, procedural text)
and define a suitable task, loss function (e.g., predicting next step, plan 
quality score), and evaluation metrics.
"""

import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm # Progress bars
import time

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Attempt imports
try:
    from src.modules.planning import PlanningModule
except ImportError:
    print(f"Error: Failed to import PlanningModule from src.modules.planning")
    print(f"Ensure '{PROJECT_ROOT}' is in your Python path and the file exists.")
    sys.exit(1)

try:
    from src.core.bitnet import BitLinear, BitGRUCell # Check availability
except ImportError:
     print(f"Warning: Failed to import BitLinear/BitGRUCell from src.core.bitnet.")

# Placeholder Trainer Class (Includes early stopping and best model saving)
class PlaceholderTrainer:
    def __init__(self, model, optimizer, criterion, device, config, experiment_name="PlanningExperiment"):
        self.model = model
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
        self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

    def _save_checkpoint(self, epoch, is_best=False):
        filename = f"checkpoint_epoch_{epoch+1}.pth"
        if is_best:
             filename = "checkpoint_best.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), filepath)

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Assuming batch yields dict {'context_state': tensor, 'context_memory': tensor, 'target_embedding': tensor}
            context_state = batch['context_state'].to(self.device)
            context_memory = batch['context_memory'].to(self.device)
            target_embedding = batch['target_embedding'].to(self.device) # Placeholder target

            self.optimizer.zero_grad()
            
            # Model Forward Pass
            plan_embedding, step_states, step_importances = self.model(context_state, context_memory)
            
            # Loss Calculation (Placeholder: MSE between output plan_embedding and dummy target)
            # **REPLACE THIS WITH A MEANINGFUL LOSS FOR YOUR PLANNING TASK**
            loss = self.criterion(plan_embedding, target_embedding) 
            # --- End Placeholder Loss ---

            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        # Add containers for specific validation metrics if needed
        
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in pbar:
                context_state = batch['context_state'].to(self.device)
                context_memory = batch['context_memory'].to(self.device)
                target_embedding = batch['target_embedding'].to(self.device)

                plan_embedding, _, _ = self.model(context_state, context_memory)
                
                # Loss Calculation (Placeholder - Match Training)
                loss = self.criterion(plan_embedding, target_embedding)
                # --- End Placeholder Loss ---
                
                total_loss += loss.item()
                # Collect predictions/targets if calculating other metrics

        avg_loss = total_loss / len(val_loader)
        
        # --- Calculate additional validation metrics ---
        # Example: Cosine similarity if target is an embedding
        # Placeholder: just use loss for now
        metrics = {'val_cosine_sim': 0.0} # Replace with actual calculation if needed
        # ---------------------------------------------

        return avg_loss, metrics

    def train(self, train_loader, val_loader):
        epochs = self.config.get('epochs', 10)
        patience = self.config.get('early_stopping_patience', 3)
        metric_name = self.config.get('early_stopping_metric', 'val_loss')
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
                self._save_checkpoint(epoch, is_best=True)
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
             self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
             
        final_history = {
             'train_loss': self.history['train_loss'],
             'val_loss': self.history['val_loss'],
             'val_metrics': self.history['val_metrics'],
             'best_epoch': self.best_epoch + 1,
             'best_val_metrics': self.history['val_metrics'][self.best_epoch] if self.best_epoch >= 0 else None
        }
        return final_history


class SyntheticPlanningDataset(Dataset):
    """
    Generates synthetic data for **debugging/demonstrating** planning training.

    Creates random context states, context memories, and target embeddings.
    **Replace with real data loading for actual training.** Needs task definition.
    """
    def __init__(self, 
                 num_samples: int = 5000, 
                 hidden_dim: int = 256, 
                 seq_len: int = 10, 
                 seed: int = 42):
        """Initialize the synthetic dataset."""
        self.num_samples = num_samples
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.seed = seed
        
        print(f"Generating {num_samples} synthetic planning samples...")
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Generate random data as placeholders
        self.context_states = torch.randn(num_samples, hidden_dim)
        self.context_memories = torch.randn(num_samples, seq_len, hidden_dim)
        # Placeholder target: another random vector of the same dimension as the model's output embedding
        self.target_embeddings = torch.randn(num_samples, hidden_dim) 
        print("Synthetic data generation complete.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        if idx >= self.num_samples:
            raise IndexError("Index out of bounds")
        return {
            'context_state': self.context_states[idx].float(),
            'context_memory': self.context_memories[idx].float(),
            'target_embedding': self.target_embeddings[idx].float() # Placeholder target
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Planning Module Training (Placeholder)")
    # Model/Data Args
    parser.add_argument("--hidden_dim", type=int, default=256, help="Input hidden dimension")
    parser.add_argument("--plan_dim", type=int, default=256, help="Internal planning dimension")
    parser.add_argument("--plan_steps", type=int, default=5, help="Number of planning steps")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length for synthetic context memory")
    parser.add_argument("--quantize", action="store_true", help="Use BitLinear/BitGRUCell layers")
    # Training Args
    parser.add_argument("--epochs", type=int, default=40, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size") # Planning might need smaller bs
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    # Early Stopping Args
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--early_stopping_metric", type=str, default="val_loss", help="Metric for early stopping")
    parser.add_argument("--early_stopping_mode", type=str, default="min", choices=["min", "max"], help="Early stopping mode")
    # Runtime Args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count(), 2), help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=456, help="Random seed")
    parser.add_argument("--experiment_name", type=str, default="PlanningSynthetic", help="Logging directory name")
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_model(args):
    """Creates the PlanningModule."""
    print(f"Creating PlanningModule (hidden_dim={args.hidden_dim}, plan_dim={args.plan_dim}, "
          f"steps={args.plan_steps}, BitLinear={args.quantize})")
    model = PlanningModule(
        hidden_dim=args.hidden_dim,
        plan_dim=args.plan_dim,
        plan_steps=args.plan_steps,
        bit_linear=args.quantize
    )
    return model.to(args.device)

def create_datasets(args):
    """Creates synthetic training and validation datasets."""
    print("Creating synthetic Planning datasets (FOR DEBUGGING/DEMO ONLY)...")
    print("Replace with real data loading and task definition.")
    train_dataset = SyntheticPlanningDataset(
        num_samples=5000,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        seed=args.seed
    )
    val_dataset = SyntheticPlanningDataset(
        num_samples=1000,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        seed=args.seed + 1
    )
    return train_dataset, val_dataset

def main():
    args = parse_args()
    set_seed(args.seed)

    print("=== Planning Module Training Configuration ===")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("==============================================")
    
    # Setup
    model = create_model(args)
    train_dataset, val_dataset = create_datasets(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Optimizer
    if args.optimizer == "adamw":
         optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
         optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
         
    # Loss Function (Placeholder: MSE on final embedding)
    criterion = nn.MSELoss() 
    print("WARNING: Using placeholder MSE loss on plan embedding. Define task-specific loss.")

    # Trainer
    trainer_config = {
        'epochs': args.epochs, 'learning_rate': args.learning_rate,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_metric': args.early_stopping_metric,
        'early_stopping_mode': args.early_stopping_mode,
    }
    print("\nInitializing PlaceholderTrainer for Planning Module...")
    trainer = PlaceholderTrainer(
        model=model, optimizer=optimizer, criterion=criterion, 
        device=args.device, config=trainer_config, experiment_name=args.experiment_name
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

    except Exception as e:
         print(f"\nAn error occurred during training: {e}")
         import traceback
         traceback.print_exc()
         sys.exit(1)

if __name__ == "__main__":
    main()