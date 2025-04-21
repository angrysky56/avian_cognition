"""
Bayesian Inference Module Training Protocol

This module implements the specialized neural cultivation process for the
Bayesian inference circuit, enabling probabilistic reasoning through 
sequential belief updating across multiple time scales.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from .base_trainer import CognitiveTrainer
from src.modules.bayesian import generate_bayesian_training_data


class BayesianInferenceTrainer(CognitiveTrainer):
    """
    Specialized evolutionary orchestrator for probabilistic reasoning.
    
    Cultivates the system's capacity for Bayesian inference through a
    temporal training protocol that shapes the module's belief-updating
    mechanisms, mirroring the evidence accumulation processes observed
    in corvids and kea parrots.
    
    The training rhythm centers on sequential evidence integration tasks,
    where the module must learn to update its beliefs in accordance with
    Bayesian principles, balancing prior distributions with new evidence
    while accounting for uncertainty.
    
    Attributes:
        model: Bayesian inference module or integrated architecture
        backbone: Optional underlying model for generating representations
        kl_weight: Weight for KL divergence loss component
        sequence_lengths: Range of sequence lengths to train on
    """
    
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        config,
        backbone=None,
        kl_weight=1.0,
        sequence_lengths=(5, 10),
        experiment_name="BayesianTraining"
    ):
        """
        Initialize the Bayesian training protocol.
        
        Args:
            model: Neural architecture to evolve
            optimizer: Parameter update mechanism
            criterion: Loss evaluation function
            device: Computational substrate
            config: Hyperparameter configuration
            backbone: Optional base model for generating representations
            kl_weight: Weight for KL divergence loss component
            sequence_lengths: Range of sequence lengths for training tasks
            experiment_name: Identifier for this evolutionary sequence
        """
        super().__init__(model, optimizer, criterion, device, config, experiment_name)
        
        self.backbone = backbone
        self.kl_weight = kl_weight
        self.sequence_lengths = sequence_lengths
        
        # Track belief updating metrics
        self.kl_divergence_history = []
        self.accuracy_history = []
        
    def _extract_bayesian_module(self, model):
        """
        Isolate Bayesian circuitry from integrated architecture.
        
        Many architectural variants may contain Bayesian components,
        this function navigates the architectural hierarchy to locate
        the specific circuitry responsible for belief updating.
        
        Args:
            model: Integrated neural architecture
            
        Returns:
            bayesian_module: Isolated Bayesian circuit
        """
        # Direct access if model is the Bayesian module itself
        if hasattr(model, 'update_belief'):
            return model
            
        # Extract from AvianMamba architecture
        if hasattr(model, 'bayesian_module'):
            return model.bayesian_module
            
        # Find module by name in children
        for name, child in model.named_children():
            if 'bayes' in name.lower() and hasattr(child, 'update_belief'):
                return child
                
        # Default to the provided model
        return model
        
    def _generate_training_batch(self, batch_size, seq_len, num_hypotheses=3):
        """
        Create synthetic Bayesian reasoning tasks.
        
        Generates sequences of evidence and corresponding ground-truth
        posterior probabilities for training the belief updating mechanism.
        
        Args:
            batch_size: Number of tasks to generate
            seq_len: Length of evidence sequences
            num_hypotheses: Number of hypotheses to distinguish
            
        Returns:
            evidence_sequences: Generated evidence sequences
            posterior_probs: Ground-truth posterior probabilities
        """
        return generate_bayesian_training_data(
            num_samples=batch_size,
            num_hypotheses=num_hypotheses,
            sequence_length=seq_len,
            device=self.device
        )
        
    def train_epoch(self, train_loader=None):
        """
        Conduct a single evolutionary cycle for belief updating calibration.
        
        This cycle orchestrates the refinement of belief updating mechanisms
        through sequential evidence integration tasks, developing the module's
        capacity to perform Bayesian inference.
        
        Args:
            train_loader: Optional iterator through training examples
                         (if None, generate synthetic data)
            
        Returns:
            metrics: Quantified aspects of Bayesian reasoning development
        """
        # Set to training mode
        self.model.train()
        if self.backbone is not None:
            self.backbone.eval()  # Freeze backbone during Bayesian training
            
        # Isolate Bayesian module if needed
        bayesian_module = self._extract_bayesian_module(self.model)
        
        # Tracking metrics
        total_loss = 0
        total_kl_div = 0
        total_samples = 0
        total_correct = 0
        
        # Determine number of batches
        if train_loader is not None:
            num_batches = len(train_loader)
        else:
            # Default number of synthetic batches
            num_batches = self.config.get('synthetic_batches', 100)
            
        # Evolutionary iteration
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {self.epoch}"):
            # Either use provided data or generate synthetic tasks
            if train_loader is not None:
                batch = next(iter(train_loader))
                evidence_sequences = batch['evidence'].to(self.device)
                posterior_probs = batch['posterior'].to(self.device)
            else:
                # Randomly select sequence length from range
                seq_len = np.random.randint(
                    self.sequence_lengths[0],
                    self.sequence_lengths[1] + 1
                )
                
                # Generate synthetic Bayesian tasks
                batch_size = self.config.get('batch_size', 32)
                num_hypotheses = self.config.get('num_hypotheses', 3)
                evidence_sequences, posterior_probs = self._generate_training_batch(
                    batch_size, seq_len, num_hypotheses
                )
            
            # Get dimensions
            seq_len, batch_size, feature_dim = evidence_sequences.shape
            _, _, num_hypotheses = posterior_probs.shape
            
            # Reset gradient pathways
            self.optimizer.zero_grad()
            
            # Initialize belief state
            belief_state = None
            
            # Accumulated loss for sequence
            sequence_loss = 0
            
            # Process sequence step by step
            for t in range(seq_len):
                # Current evidence
                evidence_t = evidence_sequences[t]
                
                # Ground truth posterior
                true_posterior_t = posterior_probs[t]
                
                # Update belief state using Bayesian module
                belief_state, belief_embedding = bayesian_module(evidence_t, belief_state)
                
                # Convert belief state to probability distribution
                if hasattr(bayesian_module, 'belief_activation') and isinstance(bayesian_module.belief_activation, nn.Tanh):
                    # If using tanh activation, transform to [0,1]
                    belief_probs = (belief_state + 1) / 2
                else:
                    # Otherwise use softmax to normalize
                    belief_probs = F.softmax(belief_state[:, :num_hypotheses], dim=1)
                
                # Calculate KL divergence from true posterior
                kl_div = F.kl_div(
                    torch.log(belief_probs + 1e-10),  # Add small epsilon to avoid log(0)
                    true_posterior_t,
                    reduction='batchmean'
                )
                
                # Add to sequence loss with weight
                sequence_loss += self.kl_weight * kl_div
                
                # Track KL divergence
                total_kl_div += kl_div.item() * batch_size
            
            # Backward propagation of alignment signal
            sequence_loss.backward()
            
            # Update parameters along improvement gradient
            self.optimizer.step()
            
            # Track loss
            total_loss += sequence_loss.item() * batch_size
            total_samples += batch_size
            
            # Evaluate accuracy (highest probability hypothesis matches true hypothesis)
            final_beliefs = belief_probs.argmax(dim=1)
            true_hypotheses = true_posterior_t.argmax(dim=1)
            correct = (final_beliefs == true_hypotheses).sum().item()
            total_correct += correct
            
            # Update step counter
            self.global_step += 1
            
        # Calculate epoch-level metrics
        avg_loss = total_loss / total_samples
        avg_kl_div = total_kl_div / total_samples
        accuracy = total_correct / total_samples
        
        # Record metric history
        self.kl_divergence_history.append(avg_kl_div)
        self.accuracy_history.append(accuracy)
        
        # Generate belief updating visualization
        if self.epoch % self.config.get('plot_interval', 5) == 0:
            self._plot_belief_updating()
            
        # Return consolidated metrics
        metrics = {
            'loss': avg_loss,
            'kl_divergence': avg_kl_div,
            'accuracy': accuracy
        }
        
        return metrics
        
    def validate(self, val_loader=None):
        """
        Evaluate current belief updating calibration on validation ecosystem.
        
        Args:
            val_loader: Optional iterator through validation examples
                       (if None, generate synthetic data)
            
        Returns:
            metrics: Quantified aspects of Bayesian reasoning calibration
        """
        # Set to evaluation mode
        self.model.eval()
        if self.backbone is not None:
            self.backbone.eval()
            
        # Isolate Bayesian module if needed
        bayesian_module = self._extract_bayesian_module(self.model)
        
        # Tracking metrics
        total_loss = 0
        total_kl_div = 0
        total_samples = 0
        total_correct = 0
        
        # Store belief trajectories for visualization
        belief_trajectories = []
        ground_truth_trajectories = []
        
        # Determine number of batches
        if val_loader is not None:
            num_batches = min(len(val_loader), self.config.get('val_batches', 10))
        else:
            # Default number of synthetic batches
            num_batches = self.config.get('val_batches', 10)
            
        # Evaluation without gradient tracking
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Validation"):
                # Either use provided data or generate synthetic tasks
                if val_loader is not None:
                    batch = next(iter(val_loader))
                    evidence_sequences = batch['evidence'].to(self.device)
                    posterior_probs = batch['posterior'].to(self.device)
                else:
                    # Fix sequence length for consistent validation
                    seq_len = self.sequence_lengths[1]  # Use max length
                    
                    # Generate synthetic Bayesian tasks
                    batch_size = self.config.get('batch_size', 32)
                    num_hypotheses = self.config.get('num_hypotheses', 3)
                    evidence_sequences, posterior_probs = self._generate_training_batch(
                        batch_size, seq_len, num_hypotheses
                    )
                
                # Get dimensions
                seq_len, batch_size, feature_dim = evidence_sequences.shape
                _, _, num_hypotheses = posterior_probs.shape
                
                # Initialize belief state
                belief_state = None
                
                # Save belief trajectories for first sample
                sample_beliefs = []
                sample_ground_truth = []
                
                # Accumulated loss for sequence
                sequence_loss = 0
                
                # Process sequence step by step
                for t in range(seq_len):
                    # Current evidence
                    evidence_t = evidence_sequences[t]
                    
                    # Ground truth posterior
                    true_posterior_t = posterior_probs[t]
                    
                    # Update belief state using Bayesian module
                    belief_state, belief_embedding = bayesian_module(evidence_t, belief_state)
                    
                    # Convert belief state to probability distribution
                    if hasattr(bayesian_module, 'belief_activation') and isinstance(bayesian_module.belief_activation, nn.Tanh):
                        # If using tanh activation, transform to [0,1]
                        belief_probs = (belief_state + 1) / 2
                    else:
                        # Otherwise use softmax to normalize
                        belief_probs = F.softmax(belief_state[:, :num_hypotheses], dim=1)
                    
                    # Calculate KL divergence from true posterior
                    kl_div = F.kl_div(
                        torch.log(belief_probs + 1e-10),
                        true_posterior_t,
                        reduction='batchmean'
                    )
                    
                    # Add to sequence loss
                    sequence_loss += self.kl_weight * kl_div
                    
                    # Track KL divergence
                    total_kl_div += kl_div.item() * batch_size
                    
                    # Save trajectory for first sample
                    if batch_idx == 0:
                        sample_beliefs.append(belief_probs[0].cpu().numpy())
                        sample_ground_truth.append(true_posterior_t[0].cpu().numpy())
                
                # Save trajectories
                if batch_idx == 0:
                    belief_trajectories = sample_beliefs
                    ground_truth_trajectories = sample_ground_truth
                
                # Track loss
                total_loss += sequence_loss.item() * batch_size
                total_samples += batch_size
                
                # Evaluate accuracy (highest probability hypothesis matches true hypothesis)
                final_beliefs = belief_probs.argmax(dim=1)
                true_hypotheses = true_posterior_t.argmax(dim=1)
                correct = (final_beliefs == true_hypotheses).sum().item()
                total_correct += correct
        
        # Calculate validation metrics
        avg_loss = total_loss / total_samples
        avg_kl_div = total_kl_div / total_samples
        accuracy = total_correct / total_samples
        
        # Generate validation belief trajectory visualization
        if len(belief_trajectories) > 0:
            self._plot_belief_trajectory(
                belief_trajectories,
                ground_truth_trajectories,
                title=f"Validation Belief Trajectory (Epoch {self.epoch})",
                filename=f"val_belief_trajectory_epoch_{self.epoch}.png"
            )
        
        # Return consolidated metrics
        metrics = {
            'loss': avg_loss,
            'kl_divergence': avg_kl_div,
            'accuracy': accuracy
        }
        
        return metrics
        
    def _plot_belief_updating(self):
        """
        Visualize the belief updating process on a test problem.
        
        Creates a demonstration of the Bayesian module's belief updating
        mechanism on a single test problem, showing how beliefs evolve
        as new evidence arrives.
        """
        # Set to evaluation mode
        self.model.eval()
        
        # Isolate Bayesian module
        bayesian_module = self._extract_bayesian_module(self.model)
        
        # Create test problem
        seq_len = 10  # Fixed length for visualization
        num_hypotheses = 3
        evidence_sequences, posterior_probs = self._generate_training_batch(
            batch_size=1, seq_len=seq_len, num_hypotheses=num_hypotheses
        )
        
        # Track belief evolution
        belief_trajectory = []
        ground_truth_trajectory = []
        
        # Process sequence without gradients
        with torch.no_grad():
            belief_state = None
            
            for t in range(seq_len):
                # Current evidence
                evidence_t = evidence_sequences[t]
                
                # Ground truth posterior
                true_posterior_t = posterior_probs[t]
                
                # Update belief state
                belief_state, _ = bayesian_module(evidence_t, belief_state)
                
                # Convert to probabilities
                if hasattr(bayesian_module, 'belief_activation') and isinstance(bayesian_module.belief_activation, nn.Tanh):
                    belief_probs = (belief_state + 1) / 2
                else:
                    belief_probs = F.softmax(belief_state[:, :num_hypotheses], dim=1)
                
                # Record beliefs
                belief_trajectory.append(belief_probs[0].cpu().numpy())
                ground_truth_trajectory.append(true_posterior_t[0].cpu().numpy())
        
        # Create visualization
        self._plot_belief_trajectory(
            belief_trajectory,
            ground_truth_trajectory,
            title=f"Belief Updating (Epoch {self.epoch})",
            filename=f"belief_trajectory_epoch_{self.epoch}.png"
        )
        
    def _plot_belief_trajectory(self, belief_trajectory, ground_truth_trajectory, title=None, filename=None):
        """
        Visualize belief evolution compared to ground truth.
        
        Creates a plot showing how the module's belief state evolves over
        time compared to the ground truth Bayesian posterior, illustrating
        its belief updating capabilities.
        
        Args:
            belief_trajectory: Module's belief probabilities over time
            ground_truth_trajectory: Ground truth posteriors over time
            title: Optional title for the plot
            filename: Optional filename for saving the visualization
        """
        # Convert to numpy if needed
        if isinstance(belief_trajectory[0], torch.Tensor):
            belief_trajectory = [b.cpu().numpy() for b in belief_trajectory]
            
        if isinstance(ground_truth_trajectory[0], torch.Tensor):
            ground_truth_trajectory = [g.cpu().numpy() for g in ground_truth_trajectory]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get number of hypotheses and time steps
        num_hypotheses = belief_trajectory[0].shape[0]
        seq_len = len(belief_trajectory)
        
        # Plot beliefs for each hypothesis
        time_steps = list(range(seq_len))
        
        # Colors for hypotheses
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        for h in range(min(num_hypotheses, len(colors))):
            # Extract trajectory for hypothesis h
            h_beliefs = [beliefs[h] for beliefs in belief_trajectory]
            h_ground_truth = [gt[h] for gt in ground_truth_trajectory]
            
            # Plot belief trajectory
            ax.plot(time_steps, h_beliefs, f'{colors[h]}-', marker='o', linewidth=2, 
                   label=f'Belief H{h}')
            
            # Plot ground truth with dashed line
            ax.plot(time_steps, h_ground_truth, f'{colors[h]}--', marker='x', alpha=0.5,
                   label=f'True H{h}')
        
        # Customize plot
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Probability')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Belief Trajectory vs Ground Truth')
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()
        
        # Save figure
        if filename:
            save_path = f"{self.log_dir}/plots/{filename}"
        else:
            save_path = f"{self.log_dir}/plots/belief_trajectory_{self.epoch}.png"
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        
        # Log to tensorboard
        self.writer.add_figure(
            "belief_updating/trajectory",
            fig,
            global_step=self.epoch
        )
        
        # Close figure to prevent memory leaks
        plt.close(fig)
        
    def plot_training_trajectory(self):
        """
        Visualize the evolutionary trajectory of Bayesian reasoning development.
        
        Creates plots showing the progression of KL divergence and accuracy
        throughout the training process, providing insight into the system's
        Bayesian inference development.
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot KL divergence history
        epochs = range(len(self.kl_divergence_history))
        ax1.plot(epochs, self.kl_divergence_history, 'b-', marker='o')
        ax1.set_title('KL Divergence from Bayes-Optimal')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('KL Divergence')
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
