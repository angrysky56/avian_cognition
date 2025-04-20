"""
Bayesian Inference Module

This module implements a slow-timescale recurrent network for belief state tracking
and probabilistic reasoning, inspired by evidence integration in statistical corvids
and kea parrots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.bitnet import BitLinear


class BayesianInferenceModule(nn.Module):
    """
    Belief state tracker implementing approximate Bayesian inference.
    
    Maintains and updates probability distributions over latent variables
    as new evidence arrives, mimicking the way birds integrate information
    over time to make probabilistic judgments.
    
    Attributes:
        hidden_dim: Dimension of hidden state from main model
        belief_dim: Dimension of belief state representation
        evidence_encoder: Transforms token evidence into belief space
        prior_transform: Transforms previous belief (prior) for update
        update_gate: Controls information flow between prior and new evidence
        output_proj: Projects belief state back to hidden dimension
    """
    
    def __init__(self, hidden_dim, belief_dim=None, bit_linear=True):
        """
        Initialize Bayesian inference module with BitLinear quantization.
        
        Args:
            hidden_dim: Dimension of the input hidden state
            belief_dim: Dimension of belief state (defaults to hidden_dim)
            bit_linear: Whether to use BitLinear quantization
        """
        super().__init__()
        
        # Set dimensions
        self.hidden_dim = hidden_dim
        self.belief_dim = belief_dim if belief_dim is not None else hidden_dim
        
        # Evidence encoding
        self.evidence_encoder = BitLinear(hidden_dim, self.belief_dim) if bit_linear else nn.Linear(hidden_dim, self.belief_dim)
        
        # Prior belief transformation
        self.prior_transform = BitLinear(self.belief_dim, self.belief_dim) if bit_linear else nn.Linear(self.belief_dim, self.belief_dim)
        
        # Belief update mechanism (Bayes-inspired gating)
        self.update_gate = nn.Sequential(
            BitLinear(self.belief_dim * 2, self.belief_dim) if bit_linear else nn.Linear(self.belief_dim * 2, self.belief_dim),
            nn.Sigmoid()
        )
        
        # Output projection to hidden dimension
        self.output_proj = BitLinear(self.belief_dim, hidden_dim) if bit_linear else nn.Linear(self.belief_dim, hidden_dim)
        
        # Activation functions
        self.belief_activation = nn.Tanh()  # Constrains belief values
        
    def forward(self, hidden_state, belief_state=None):
        """
        Updates belief state based on new evidence.
        
        Args:
            hidden_state: Current token's hidden state representation
                          [batch_size, hidden_dim]
            belief_state: Previous belief state (or None for initial state)
                          [batch_size, belief_dim]
            
        Returns:
            new_belief: Updated belief state [batch_size, belief_dim]
            belief_embedding: Projection of belief state to hidden dimension
                             [batch_size, hidden_dim]
        """
        batch_size = hidden_state.shape[0]
        
        # Initialize belief state if None
        if belief_state is None:
            belief_state = torch.zeros(batch_size, self.belief_dim, device=hidden_state.device)
        
        # Transform prior belief
        prior_belief = self.prior_transform(belief_state)
        
        # Encode current evidence
        evidence = self.evidence_encoder(hidden_state)
        
        # Compute update gate based on prior and evidence
        gate_input = torch.cat([prior_belief, evidence], dim=-1)
        update_gate = self.update_gate(gate_input)
        
        # Update belief (approximating Bayes rule)
        new_belief = update_gate * evidence + (1 - update_gate) * prior_belief
        new_belief = self.belief_activation(new_belief)  # Apply constraint
        
        # Project to hidden dimension for integration with main model
        belief_embedding = self.output_proj(new_belief)
        
        return new_belief, belief_embedding
    
    def infer_posterior(self, evidence_sequence, initial_belief=None):
        """
        Processes a sequence of evidence to produce a final posterior belief.
        
        Args:
            evidence_sequence: Sequence of evidence representations
                              [seq_len, batch_size, hidden_dim]
            initial_belief: Optional initial belief state
                           [batch_size, belief_dim]
            
        Returns:
            final_belief: Final posterior belief state after seeing all evidence
                         [batch_size, belief_dim]
            belief_trajectory: Sequence of belief states throughout processing
                              [seq_len, batch_size, belief_dim]
        """
        seq_len, batch_size, _ = evidence_sequence.shape
        device = evidence_sequence.device
        
        # Initialize belief
        if initial_belief is None:
            belief = torch.zeros(batch_size, self.belief_dim, device=device)
        else:
            belief = initial_belief
            
        # Process evidence sequentially
        belief_trajectory = []
        for t in range(seq_len):
            belief, _ = self.forward(evidence_sequence[t], belief)
            belief_trajectory.append(belief)
            
        # Stack trajectory
        belief_trajectory = torch.stack(belief_trajectory, dim=0)
        
        return belief, belief_trajectory


class SpatialBayesianLayer(nn.Module):
    """
    Specialized Bayesian layer for spatial probabilistic reasoning.
    
    Implements belief tracking over spatial locations, similar to how birds
    might maintain probabilistic maps of reward likelihoods across locations.
    
    Attributes:
        spatial_dim: Number of spatial locations to track
        feature_dim: Feature dimension for each location
        belief_update: Update network for spatial belief
    """
    
    def __init__(self, spatial_dim, feature_dim, bit_linear=True):
        """
        Initialize spatial Bayesian layer with BitLinear quantization.
        
        Args:
            spatial_dim: Number of spatial locations to track
            feature_dim: Feature dimension for each location
            bit_linear: Whether to use BitLinear quantization
        """
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        
        # Belief update network
        self.belief_update = nn.Sequential(
            BitLinear(feature_dim * 2, feature_dim) if bit_linear else nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            BitLinear(feature_dim, feature_dim) if bit_linear else nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, evidence, prior_belief=None):
        """
        Updates spatial belief distribution based on new evidence.
        
        Args:
            evidence: Evidence features for each location
                    [batch_size, spatial_dim, feature_dim]
            prior_belief: Prior belief distribution over locations
                         [batch_size, spatial_dim, feature_dim]
            
        Returns:
            posterior_belief: Updated belief distribution over locations
                             [batch_size, spatial_dim, feature_dim]
        """
        batch_size = evidence.shape[0]
        
        # Initialize uniform prior if None
        if prior_belief is None:
            prior_belief = torch.ones(batch_size, self.spatial_dim, self.feature_dim, device=evidence.device)
            prior_belief = prior_belief / self.spatial_dim  # Normalize
        
        # Concatenate prior and evidence for each location
        update_input = torch.cat([prior_belief, evidence], dim=-1)
        
        # Compute likelihood ratio
        likelihood = self.belief_update(update_input)
        
        # Apply Bayes rule (element-wise multiplication)
        posterior_unnorm = prior_belief * likelihood
        
        # Normalize (sum to 1 across spatial dimension)
        normalizer = posterior_unnorm.sum(dim=1, keepdim=True).clamp(min=1e-10)
        posterior_belief = posterior_unnorm / normalizer
        
        return posterior_belief


def kl_divergence_loss(predicted_distribution, target_distribution):
    """
    Kullback-Leibler divergence loss for training Bayesian modules.
    
    Measures how much the predicted distribution differs from the target
    distribution, providing a training signal for Bayesian inference.
    
    Args:
        predicted_distribution: Predicted probability distribution
                               [batch_size, num_classes]
        target_distribution: Target probability distribution
                           [batch_size, num_classes]
        
    Returns:
        kl_div: KL divergence loss (lower is better)
    """
    # Ensure proper probability distributions
    predicted = F.softmax(predicted_distribution, dim=-1)
    target = F.softmax(target_distribution, dim=-1)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    predicted = predicted + epsilon
    target = target + epsilon
    
    # Normalize
    predicted = predicted / predicted.sum(dim=-1, keepdim=True)
    target = target / target.sum(dim=-1, keepdim=True)
    
    # KL divergence
    kl_div = (target * torch.log(target / predicted)).sum(dim=-1).mean()
    
    return kl_div


def generate_bayesian_training_data(num_samples, num_hypotheses, sequence_length, device='cpu'):
    """
    Generates synthetic data for training Bayesian inference modules.
    
    Creates sequences of evidence with corresponding ground-truth posterior
    probabilities for supervised training of belief updating.
    
    Args:
        num_samples: Number of training sequences
        num_hypotheses: Number of hypotheses to distinguish between
        sequence_length: Length of evidence sequences
        device: Device for tensor allocation
        
    Returns:
        evidence_sequences: Synthetic evidence sequences
                           [num_samples, sequence_length, feature_dim]
        posterior_probs: Ground-truth posterior probabilities after each evidence
                        [num_samples, sequence_length, num_hypotheses]
    """
    # Feature dimension for evidence (arbitrary choice)
    feature_dim = num_hypotheses * 2
    
    # Generate random prior probabilities
    prior_probs = torch.rand(num_samples, num_hypotheses, device=device)
    prior_probs = prior_probs / prior_probs.sum(dim=1, keepdim=True)  # Normalize
    
    # Initialize outputs
    evidence_sequences = torch.zeros(num_samples, sequence_length, feature_dim, device=device)
    posterior_probs = torch.zeros(num_samples, sequence_length, num_hypotheses, device=device)
    
    # Generate evidence sequences and compute posteriors
    for i in range(num_samples):
        # Choose true hypothesis
        true_hypothesis = torch.randint(0, num_hypotheses, (1,)).item()
        
        # Current belief starts with prior
        current_belief = prior_probs[i].clone()
        
        for t in range(sequence_length):
            # Generate evidence that slightly favors true hypothesis
            evidence = torch.randn(feature_dim, device=device) * 0.5  # Random noise
            
            # Add signal favoring true hypothesis
            signal_strength = 0.5 + 0.5 * torch.rand(1, device=device).item()  # Random strength
            evidence[true_hypothesis:true_hypothesis+num_hypotheses] += signal_strength
            
            # Store evidence
            evidence_sequences[i, t] = evidence
            
            # Update belief using Bayes rule
            likelihoods = torch.zeros(num_hypotheses, device=device)
            for h in range(num_hypotheses):
                # Simple likelihood model: how much evidence favors this hypothesis
                likelihoods[h] = torch.exp(evidence[h:h+num_hypotheses].sum() / num_hypotheses)
            
            # Bayesian update
            posterior_unnorm = current_belief * likelihoods
            current_belief = posterior_unnorm / posterior_unnorm.sum()  # Normalize
            
            # Store posterior
            posterior_probs[i, t] = current_belief
            
    return evidence_sequences, posterior_probs
