"""
Bayesian Inference Module Training Script (Placeholder)

This script provides a basic framework for training the BayesianInferenceModule.
It currently uses synthetic sequential data.

**IMPORTANT:** For practical use, replace the SyntheticBayesianDataset 
with a dataset derived from a relevant task involving sequential evidence 
and probabilistic reasoning. The loss function (KL Divergence) assumes the goal 
is to match target probability distributions.
"""

import os
import sys
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm # For progress bars
import time

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Attempt imports
try:
    from src.modules.bayesian import BayesianInferenceModule, kl_divergence_loss, generate_bayesian_training_data
except ImportError:
    print(f"Error: Failed to import modules from src.modules.bayesian")
    print(f"Ensure '{PROJECT_ROOT}' is in your Python path and the files exist.")
    sys.exit(1)
    
try:
    # Check if BitLinear is available (though not strictly required by the Bayesian module itself unless enabled)
    from src.core.bitnet import BitLinear 
except ImportError:
     print(f"Warning: Failed to import BitLinear from src.core.bitnet.")

# Placeholder Trainer Class (Replace with a more robust implementation later)
# Includes basic early stopping and best model saving logic
class PlaceholderTrainer:
    def __init__(self, model, optimizer, criterion, device, config, experiment_name="BayesianExperiment"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config # Expects dict with 'early_stopping_...' keys etc.
        self.experiment_name = experiment_name
        self.log_dir = os.path.join("logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_metric = float('inf') if config.get('early_stopping_mode', 'min') == 'min' else float('-inf')
        self.epochs_no_improve = 0
        self.best_epoch = -1
        self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': []} # Add other metrics as needed

    def _save_checkpoint(self, epoch, is_best=False):
        filename = f"checkpoint_epoch_{epoch+1}.pth"
        if is_best:
             filename = "checkpoint_best.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        # print(f"Saved checkpoint: {filepath}")

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Move batch to device
            # Assuming batch yields dict {'evidence': tensor, 'posteriors': tensor}
            evidence_seq = batch['evidence'].to(self.device) # Shape: [batch, seq_len, hidden_dim]
            target_posteriors = batch['posteriors'].to(self.device) # Shape: [batch, seq_len, num_hypotheses]
            
            # Transpose evidence for module: [seq_len, batch, hidden_dim]
            evidence_seq = evidence_seq.transpose(0, 1) 
            # Transpose target: [seq_len, batch, num_hypotheses]
            target_posteriors = target_posteriors.transpose(0, 1)
            
            self.optimizer.zero_grad()
            
            # --- Model Forward Pass ---
            # Get final belief state and trajectory
            # NOTE: Training might focus on the final belief or the entire trajectory
            final_belief, belief_trajectory = self.model.infer_posterior(evidence_seq)
            # Shape belief_trajectory: [seq_len, batch, belief_dim]

            # --- Loss Calculation ---
            # Example: Use KL loss between *final* belief state (needs decoding) and final target posterior
            # This requires adapting the belief state (e.g., using a linear layer + softmax) 
            # to match the target distribution dimensions (num_hypotheses).
            # For this placeholder, let's use a simple MSE on the raw belief state vs a dummy target.
            # **REPLACE THIS WITH A MEANINGFUL LOSS CALCULATION**
            # Example: If final_belief needs projecting to num_hypotheses for KLDivLoss
            # if not hasattr(self.model, 'belief_to_hypotheses'):
            #     self.model.belief_to_hypotheses = nn.Linear(self.model.belief_dim, target_posteriors.shape[-1]).to(self.device)
            # predicted_log_probs = F.log_softmax(self.model.belief_to_hypotheses(final_belief), dim=-1)
            # loss = F.kl_div(predicted_log_probs, target_posteriors[-1], reduction='batchmean', log_target=False)

            # Using placeholder MSE loss on last belief state vs mean of target posteriors
            # This is arbitrary and just for making the loop run.
            dummy_target = target_posteriors[-1].mean(dim=-1, keepdim=True).expand_as(final_belief) * 0.1 # Arbitrary target
            loss = F.mse_loss(final_belief, dummy_target) 
            # --- End Placeholder Loss ---

            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        # Add other metrics containers if needed (e.g., accuracy, KL div)
        all_final_beliefs = []
        all_final_targets = []
        
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in pbar:
                evidence_seq = batch['evidence'].to(self.device).transpose(0, 1)
                target_posteriors = batch['posteriors'].to(self.device).transpose(0, 1)

                final_belief, _ = self.model.infer_posterior(evidence_seq)
                
                # --- Loss Calculation (Placeholder - Match Training) ---
                dummy_target = target_posteriors[-1].mean(dim=-1, keepdim=True).expand_as(final_belief) * 0.1
                loss = F.mse_loss(final_belief, dummy_target)
                # --- End Placeholder Loss ---
                
                total_loss += loss.item()
                # Collect outputs/targets for other metrics if needed
                all_final_beliefs.append(final_belief.cpu())
                all_final_targets.append(target_posteriors[-1].cpu()) # Example: store last target

        avg_loss = total_loss / len(val_loader)
        
        # --- Calculate additional validation metrics ---
        # Example: Calculate overall KL divergence if loss was KLDiv
        # Or calculate accuracy if hypotheses prediction was involved.
        # Placeholder metric: Mean absolute error on the dummy task
        final_beliefs_cat = torch.cat(all_final_beliefs, dim=0)
        final_targets_cat = torch.cat(all_final_targets, dim=0)
        dummy_targets_cat = final_targets_cat.mean(dim=-1, keepdim=True).expand_as(final_beliefs_cat) * 0.1
        mae = F.l1_loss(final_beliefs_cat, dummy_targets_cat).item()
        metrics = {'val_mae': mae} # Example metric
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
            print() # Newline

            # --- Early Stopping & Best Model Check ---
            current_metric = val_loss if metric_name == 'val_loss' else val_metrics.get(metric_name, None)
            if current_metric is None and metric_name != 'val_loss':
                 print(f"Warning: Early stopping metric '{metric_name}' not found in validation metrics. Using val_loss.")
                 current_metric = val_loss
                 metric_name = 'val_loss' # Fallback
                 metric_mode = 'min'

            is_better = False
            if metric_mode == 'min':
                if current_metric < self.best_metric:
                    is_better = True
            else: # mode == 'max'
                if current_metric > self.best_metric:
                    is_better = True
            
            if is_better:
                print(f"  Validation metric ({metric_name}) improved ({self.best_metric:.4f} -> {current_metric:.4f}). Saving best model...")
                self.best_metric = current_metric
                self.epochs_no_improve = 0
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_no_improve += 1
                print(f"  Validation metric ({metric_name}) did not improve for {self.epochs_no_improve} epoch(s).")

            # Save checkpoint periodically if needed (optional)
            # self._save_checkpoint(epoch, is_best=False) 

            if patience > 0 and self.epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
        
        print(f"Training finished. Best model from epoch {self.best_epoch+1} saved.")
        # Add logic to load best model state back into self.model if desired
        best_model_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pth")
        if os.path.exists(best_model_path):
             print(f"Loading best model weights from {best_model_path}")
             self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        # Return history including best metrics achieved
        final_history = {
             'train_loss': self.history['train_loss'],
             'val_loss': self.history['val_loss'],
             'val_metrics': self.history['val_metrics'],
             'best_epoch': self.best_epoch + 1, # 1-based index
             'best_val_metrics': self.history['val_metrics'][self.best_epoch] if self.best_epoch >= 0 else None
        }
        return final_history


class SyntheticBayesianDataset(Dataset):
    """
    Generates synthetic sequential data for **debugging/demonstrating** Bayesian training.

    Uses `generate_bayesian_training_data` internally. 
    **Replace with real data loading for actual training.**
    
    Attributes:
        evidence (torch.Tensor): Evidence sequences. Shape [size, seq_len, feature_dim].
        posteriors (torch.Tensor): Target posterior probabilities. Shape [size, seq_len, num_hypotheses].
    """
    def __init__(self, 
                 num_samples: int = 1000, 
                 num_hypotheses: int = 3, 
                 sequence_length: int = 10, 
                 feature_dim: int = 256, # Match model's hidden_dim
                 seed: int = 42):
        """Initialize the synthetic dataset."""
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_hypotheses = num_hypotheses
        self.seed = seed
        
        print(f"Generating {num_samples} synthetic Bayesian sequences...")
        # Use the imported generator function
        self.evidence, self.posteriors = generate_bayesian_training_data(
            num_samples=self.num_samples,
            num_hypotheses=self.num_hypotheses,
            sequence_length=self.sequence_length,
            output_feature_dim=self.feature_dim, # Ensure output matches model input dim
            device='cpu' # Generate data on CPU, move to GPU in DataLoader/training loop
        )
        print("Synthetic data generation complete.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        if idx >= self.num_samples:
            raise IndexError("Index out of bounds")
        return {
            'evidence': self.evidence[idx].float(), 
            'posteriors': self.posteriors[idx].float()
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian Module Training (Placeholder)")
    # Model/Data Args
    parser.add_argument("--hidden_dim", type=int, default=256, help="Input hidden dimension (from main model)")
    parser.add_argument("--belief_dim", type=int, default=128, help="Internal dimension of the belief state")
    parser.add_argument("--num_hypotheses", type=int, default=3, help="Number of hypotheses for synthetic data/task")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length for synthetic data/task")
    parser.add_argument("--quantize", action="store_true", help="Use BitLinear layers")
    # Training Args
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    # Early Stopping Args
    parser.add_argument("--early_stopping_patience", type=int, default=4, help="Patience for early stopping (0 to disable)")
    parser.add_argument("--early_stopping_metric", type=str, default="val_loss", help="Metric for early stopping") # Could be val_mae if defined well
    parser.add_argument("--early_stopping_mode", type=str, default="min", choices=["min", "max"], help="Early stopping mode (min/max)")
    # Runtime Args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count(), 2), help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--experiment_name", type=str, default="BayesianSynthetic", help="Logging directory name")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_model(args):
    """Creates the BayesianInferenceModule."""
    print(f"Creating BayesianInferenceModule (hidden_dim={args.hidden_dim}, "
          f"belief_dim={args.belief_dim}, BitLinear={args.quantize})")
    model = BayesianInferenceModule(
        hidden_dim=args.hidden_dim,
        belief_dim=args.belief_dim,
        bit_linear=args.quantize
    )
     # **TODO**: Add a projection head if needed for the loss function
     # e.g., if using KL divergence loss, add a head to map belief_dim -> num_hypotheses
     # model.belief_to_hypotheses = nn.Linear(args.belief_dim, args.num_hypotheses)
    return model.to(args.device)


def create_datasets(args):
    """Creates synthetic training and validation datasets."""
    print("Creating synthetic Bayesian datasets (FOR DEBUGGING/DEMO ONLY)...")
    print("Replace with real data loading for actual training.")
    train_dataset = SyntheticBayesianDataset(
        num_samples=5000, # Smaller size for faster demo training
        num_hypotheses=args.num_hypotheses,
        sequence_length=args.seq_len,
        feature_dim=args.hidden_dim, # Match model input
        seed=args.seed
    )
    val_dataset = SyntheticBayesianDataset(
        num_samples=1000,
        num_hypotheses=args.num_hypotheses,
        sequence_length=args.seq_len,
        feature_dim=args.hidden_dim,
        seed=args.seed + 1
    )
    return train_dataset, val_dataset

def main():
    args = parse_args()
    set_seed(args.seed)

    print("=== Bayesian Module Training Configuration ===")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("==============================================")
    
    # --- Setup ---
    model = create_model(args)
    train_dataset, val_dataset = create_datasets(args)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Optimizer
    if args.optimizer == "adamw":
         optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
         optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
         
    # --- Loss Function ---
    # **IMPORTANT**: Replace placeholder criterion in trainer with appropriate loss
    # If model output is adapted to predict log-probs for hypotheses:
    # criterion = nn.KLDivLoss(reduction='batchmean', log_target=False)
    # If using MSE on belief state (less ideal):
    criterion = nn.MSELoss() 
    print("WARNING: Using placeholder MSE loss in trainer. Replace with appropriate criterion (e.g., KLDivLoss) and ensure model output matches target format.")
    # ---------------------

    # --- Trainer Initialization ---
    trainer_config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_metric': args.early_stopping_metric,
        'early_stopping_mode': args.early_stopping_mode,
    }
    print("\nInitializing PlaceholderTrainer for Bayesian Module...")

    trainer = PlaceholderTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion, # Pass the chosen criterion
        device=args.device,
        config=trainer_config,
        experiment_name=args.experiment_name
    )
    
    # --- Training ---
    print(f"\nStarting training...")
    try:
        training_history = trainer.train(train_loader, val_loader)
        
        # --- Results ---
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