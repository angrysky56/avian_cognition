"""
Metacognition Module Training Protocol

This module implements the specialized neural cultivation process for the
metacognition circuit, enabling the emergence of calibrated self-awareness
through systematic epistemological feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from .base_trainer import CognitiveTrainer
from ..src.modules.metacognition import expected_calibration_error, plot_reliability_diagram


class MetacognitionTrainer(CognitiveTrainer):
    """
    Specialized evolutionary orchestrator for metacognitive awareness.
    
    Cultivates the system's capacity for calibrated self-evaluation through
    a carefully designed developmental protocol that aligns confidence
    with empirical correctness, mirroring the certainty-encoding neurons
    found in the avian pallium.
    
    The training rhythm alternates between prediction and introspection,
    gradually refining the system's capacity to know what it knows—and
    equally important, to recognize what it does not.
    
    Attributes:
        model: Metacognition module or integrated architecture
        backbone: Optional underlying model for generating representations
        confidence_threshold: Level at which decisions are modulated
        calibration_bins: Granularity of calibration assessment
    """
    
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        config,
        backbone=None,
        confidence_threshold=0.5,
        calibration_bins=10,
        experiment_name="MetacognitionTraining"
    ):
        """
        Initialize the metacognition training protocol.
        
        Args:
            model: Neural architecture to evolve
            optimizer: Parameter update mechanism
            criterion: Loss evaluation function
            device: Computational substrate
            config: Hyperparameter configuration
            backbone: Optional base model for generating representations
            confidence_threshold: Confidence level for decision making
            calibration_bins: Number of bins for calibration analysis
            experiment_name: Identifier for this evolutionary sequence
        """
        super().__init__(model, optimizer, criterion, device, config, experiment_name)
        
        self.backbone = backbone
        self.confidence_threshold = confidence_threshold
        self.calibration_bins = calibration_bins
        
        # Track calibration metrics
        self.ece_history = []
        self.accuracy_history = []
        
    def _extract_metacognition_module(self, model):
        """
        Isolate metacognitive circuitry from integrated architecture.
        
        Many architectural variants may contain metacognition components,
        this function navigates the architectural hierarchy to locate
        the specific circuitry responsible for self-evaluation.
        
        Args:
            model: Integrated neural architecture
            
        Returns:
            metacognition_module: Isolated metacognitive circuit
        """
        # Direct access if model is the metacognition module itself
        if hasattr(model, 'confidence_head'):
            return model
            
        # Extract from AvianMamba architecture
        if hasattr(model, 'metacognition_module'):
            return model.metacognition_module
            
        # Find module by name in children
        for name, child in model.named_children():
            if 'meta' in name.lower() and hasattr(child, 'confidence_head'):
                return child
                
        # Default to the provided model
        return model
        
    def train_epoch(self, train_loader):
        """
        Conduct a single evolutionary cycle for metacognitive calibration.
        
        This cycle orchestrates the recursive feedback loop between
        prediction and introspection, gradually aligning confidence
        with empirical accuracy through systematic parameter refinement.
        
        Args:
            train_loader: Iterator through training examples
            
        Returns:
            metrics: Quantified aspects of metacognitive development
        """
        # Set to training mode
        self.model.train()
        if self.backbone is not None:
            self.backbone.eval()  # Freeze backbone during metacognition training
            
        # Isolate metacognition module if needed
        metacognition_module = self._extract_metacognition_module(self.model)
        
        # Tracking metrics
        total_loss = 0
        total_samples = 0
        all_confidences = []
        all_correctness = []
        
        # Evolutionary iteration
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {self.epoch}")):
            # Extract data components
            hidden_states, targets = batch['hidden_states'].to(self.device), batch['correctness'].to(self.device)
            
            # Generate hidden representations if using backbone
            if self.backbone is not None and not isinstance(hidden_states, torch.Tensor):
                with torch.no_grad():
                    inputs = hidden_states.to(self.device)
                    hidden_states = self.backbone(inputs, output_hidden_states=True).hidden_states[-1]
            
            # Reset gradient pathways
            self.optimizer.zero_grad()
            
            # Forward pass through metacognition circuit
            confidences = metacognition_module(hidden_states)
            
            # Calculate alignment loss
            loss = self.criterion(confidences, targets.float())
            
            # Backward propagation of alignment signal
            loss.backward()
            
            # Update parameters along improvement gradient
            self.optimizer.step()
            
            # Accumulate metrics
            batch_size = hidden_states.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Record confidences and correctness for calibration analysis
            all_confidences.append(confidences.detach().cpu())
            all_correctness.append(targets.detach().cpu())
            
            # Update step counter
            self.global_step += 1
            
        # Calculate epoch-level metrics
        avg_loss = total_loss / total_samples
        
        # Compute calibration metrics
        all_confidences = torch.cat(all_confidences).numpy()
        all_correctness = torch.cat(all_correctness).numpy()
        
        ece, _, _, _ = expected_calibration_error(
            all_confidences, all_correctness, n_bins=self.calibration_bins
        )
        
        # Calculate accuracy using confidence threshold
        predictions = (all_confidences >= self.confidence_threshold).astype(np.int32)
        accuracy = np.mean(predictions == all_correctness)
        
        # Record calibration history
        self.ece_history.append(ece)
        self.accuracy_history.append(accuracy)
        
        # Generate calibration visualization
        if self.epoch % self.config.get('plot_interval', 5) == 0:
            self._plot_calibration(all_confidences, all_correctness)
            
        # Return consolidated metrics
        metrics = {
            'loss': avg_loss,
            'ece': ece,
            'accuracy': accuracy
        }
        
        return metrics
        
    def validate(self, val_loader):
        """
        Evaluate current metacognitive calibration on validation ecosystem.
        
        Args:
            val_loader: Iterator through validation examples
            
        Returns:
            metrics: Quantified aspects of metacognitive calibration
        """
        # Set to evaluation mode
        self.model.eval()
        if self.backbone is not None:
            self.backbone.eval()
            
        # Isolate metacognition module if needed
        metacognition_module = self._extract_metacognition_module(self.model)
        
        # Tracking metrics
        total_loss = 0
        total_samples = 0
        all_confidences = []
        all_correctness = []
        
        # Evaluation without gradient tracking
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                # Extract data components
                hidden_states, targets = batch['hidden_states'].to(self.device), batch['correctness'].to(self.device)
                
                # Generate hidden representations if using backbone
                if self.backbone is not None and not isinstance(hidden_states, torch.Tensor):
                    inputs = hidden_states.to(self.device)
                    hidden_states = self.backbone(inputs, output_hidden_states=True).hidden_states[-1]
                
                # Forward pass through metacognition circuit
                confidences = metacognition_module(hidden_states)
                
                # Calculate alignment loss
                loss = self.criterion(confidences, targets.float())
                
                # Accumulate metrics
                batch_size = hidden_states.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Record confidences and correctness for calibration analysis
                all_confidences.append(confidences.cpu())
                all_correctness.append(targets.cpu())
        
        # Calculate validation metrics
        avg_loss = total_loss / total_samples
        
        # Compute calibration metrics
        all_confidences = torch.cat(all_confidences).numpy()
        all_correctness = torch.cat(all_correctness).numpy()
        
        ece, bin_accuracies, bin_confidences, bin_counts = expected_calibration_error(
            all_confidences, all_correctness, n_bins=self.calibration_bins
        )
        
        # Calculate accuracy using confidence threshold
        predictions = (all_confidences >= self.confidence_threshold).astype(np.int32)
        accuracy = np.mean(predictions == all_correctness)
        
        # Generate validation calibration visualization
        self._plot_calibration(
            all_confidences, all_correctness, 
            title=f"Validation Calibration (Epoch {self.epoch})",
            filename=f"val_calibration_epoch_{self.epoch}.png"
        )
        
        # Return consolidated metrics
        metrics = {
            'loss': avg_loss,
            'ece': ece,
            'accuracy': accuracy
        }
        
        return metrics
        
    def _plot_calibration(self, confidences, correctness, title=None, filename=None):
        """
        Visualize the alignment between confidence and accuracy.
        
        Creates a reliability diagram showing how well the model's confidence
        estimates align with its empirical accuracy, providing a visual
        interpretation of its metacognitive calibration.
        
        Args:
            confidences: Model confidence predictions
            correctness: Binary correctness indicators
            title: Optional title for the plot
            filename: Optional filename for saving the visualization
        """
        # Create reliability diagram
        fig, _ = plot_reliability_diagram(
            confidences, correctness, n_bins=self.calibration_bins
        )
        
        # Set title if provided
        if title is not None:
            fig.suptitle(title)
            
        # Save figure if filename provided
        if filename is not None:
            save_path = f"{self.log_dir}/plots/{filename}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
        else:
            # Save with default name
            epoch_str = f"epoch_{self.epoch}"
            save_path = f"{self.log_dir}/plots/calibration_{epoch_str}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            
        # Log figure to tensorboard
        self.writer.add_figure(
            f"calibration/reliability_diagram", 
            fig, 
            global_step=self.epoch
        )
        
        # Close figure to prevent memory leaks
        plt.close(fig)
        
    def plot_training_trajectory(self):
        """
        Visualize the evolutionary trajectory of metacognitive development.
        
        Creates plots showing the progression of calibration error and
        accuracy throughout the training process, providing insight into
        the system's metacognitive development.
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot ECE history
        epochs = range(len(self.ece_history))
        ax1.plot(epochs, self.ece_history, 'b-', marker='o')
        ax1.set_title('Expected Calibration Error')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('ECE')
        ax1.grid(True)
        
        # Plot accuracy history
        ax2.plot(epochs, self.accuracy_history, 'g-', marker='o')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        # Save figure
        save_path = f"{self.log_dir}/plots/training_trajectory.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        
        # Log figure to tensorboard
        self.writer.add_figure(
            "training/trajectory", 
            fig, 
            global_step=self.epoch
        )
        
        # Close figure to prevent memory leaks
        plt.close(fig)
