"""
Bayesian Inference Module

This module implements neural components for belief state tracking and 
probabilistic reasoning, inspired by evidence integration behaviors observed 
in birds like statistical corvids and kea parrots.

It includes a general recurrent belief updater and a specialized layer for
spatial probability tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # Keep for potential use in data generation or analysis

# Assuming BitLinear is correctly defined in src.core.bitnet
try:
    from src.core.bitnet import BitLinear
except ImportError:
    print("Warning: BitLinear not found. Falling back to nn.Linear for Bayesian module.")
    BitLinear = nn.Linear


class BayesianInferenceModule(nn.Module):
    """
    Recurrent belief state tracker approximating Bayesian inference.
    
    Uses a GRU-like mechanism to maintain and update a belief state vector based 
    on sequential evidence (represented by hidden states). The belief state is 
    an abstract representation and not necessarily a normalized probability 
    distribution, depending on the activation used (default Tanh).
    
    Attributes:
        hidden_dim (int): Dimension of input hidden state from main model.
        belief_dim (int): Dimension of the internal belief state representation.
        evidence_encoder (nn.Module): Transforms input hidden state into belief space.
        prior_transform (nn.Module): Transforms previous belief state (prior) for update.
        update_gate (nn.Module): Gating mechanism controlling information flow.
        output_proj (nn.Module): Projects final belief state back to hidden_dim.
        belief_activation (nn.Module): Activation applied to the updated belief state.
    """
    
    def __init__(self, hidden_dim, belief_dim=None, bit_linear=True):
        """
        Initialize Bayesian inference module.
        
        Args:
            hidden_dim (int): Dimension of the input hidden state.
            belief_dim (int, optional): Dimension of belief state. Defaults to hidden_dim.
            bit_linear (bool): Whether to use BitLinear layers.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        # Default belief_dim to hidden_dim if not specified
        self.belief_dim = belief_dim if belief_dim is not None else hidden_dim
        LinearLayer = BitLinear if bit_linear else nn.Linear
        
        # Evidence encoding layer
        self.evidence_encoder = LinearLayer(hidden_dim, self.belief_dim)
        
        # Prior belief transformation layer
        self.prior_transform = LinearLayer(self.belief_dim, self.belief_dim)
        
        # Belief update gating mechanism (inspired by GRU gates)
        # Takes concatenated [prior_belief, evidence] as input
        self.update_gate = nn.Sequential(
            LinearLayer(self.belief_dim * 2, self.belief_dim),
            nn.Sigmoid() # Gate values between 0 and 1
        )
        
        # Output projection layer
        self.output_proj = LinearLayer(self.belief_dim, hidden_dim)
        
        # Activation function for the belief state (constrains its values)
        # Tanh produces values in [-1, 1], not probabilities.
        # Use Softmax or other appropriate activation if probabilities are needed.
        self.belief_activation = nn.Tanh()  
        
    def forward(self, hidden_state, belief_state=None):
        """
        Performs a single update step of the belief state based on new evidence.
        
        Args:
            hidden_state (torch.Tensor): Current hidden state representation from backbone.
                                         Shape: [batch_size, hidden_dim].
            belief_state (torch.Tensor, optional): Previous belief state. 
                                                   Shape: [batch_size, belief_dim]. 
                                                   Defaults to zeros if None.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - new_belief (torch.Tensor): Updated belief state vector.
                                            Shape: [batch_size, belief_dim].
                - belief_embedding (torch.Tensor): Projection of the new belief state 
                                                  back to the main hidden dimension.
                                                  Shape: [batch_size, hidden_dim].
        """
        batch_size = hidden_state.shape[0]
        device = hidden_state.device # Ensure tensors are on the same device

        # Initialize belief state with zeros if it's the first step
        if belief_state is None:
            belief_state = torch.zeros(batch_size, self.belief_dim, device=device)
        
        # Ensure belief_state is on the correct device
        belief_state = belief_state.to(device)

        # 1. Transform the prior belief state
        prior_belief_transformed = self.prior_transform(belief_state)
        
        # 2. Encode the current evidence (hidden_state) into belief space
        evidence_encoded = self.evidence_encoder(hidden_state)
        
        # 3. Compute the update gate value
        gate_input = torch.cat([prior_belief_transformed, evidence_encoded], dim=-1)
        update_gate_value = self.update_gate(gate_input) # Shape: [batch_size, belief_dim]
        
        # 4. Update the belief state using the gate
        # Combines new evidence and transformed prior based on the gate
        new_belief_unactivated = update_gate_value * evidence_encoded + \
                                (1 - update_gate_value) * prior_belief_transformed
                                
        # 5. Apply activation to the updated belief state
        new_belief = self.belief_activation(new_belief_unactivated)
        
        # 6. Project the new belief state back to the hidden dimension
        belief_embedding = self.output_proj(new_belief)
        
        return new_belief, belief_embedding
    
    @torch.no_grad() # Inference typically doesn't require gradients here
    def infer_posterior(self, evidence_sequence, initial_belief=None):
        """
        Processes a sequence of evidence to compute the final posterior belief state.
        
        Args:
            evidence_sequence (torch.Tensor): Sequence of hidden state representations.
                                             Shape: [seq_len, batch_size, hidden_dim].
            initial_belief (torch.Tensor, optional): Initial belief state before processing sequence.
                                                     Shape: [batch_size, belief_dim]. 
                                                     Defaults to zeros if None.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - final_belief (torch.Tensor): Final belief state after processing all evidence.
                                              Shape: [batch_size, belief_dim].
                - belief_trajectory (torch.Tensor): Sequence of belief states at each step.
                                                   Shape: [seq_len, batch_size, belief_dim].
        """
        seq_len, batch_size, hidden_dim_seq = evidence_sequence.shape
        device = evidence_sequence.device

        if hidden_dim_seq != self.hidden_dim:
             raise ValueError(f"Input evidence_sequence hidden_dim ({hidden_dim_seq}) "
                              f"does not match module hidden_dim ({self.hidden_dim})")

        # Initialize belief state
        belief = initial_belief if initial_belief is not None else \
                 torch.zeros(batch_size, self.belief_dim, device=device)
        belief = belief.to(device) # Ensure it's on the right device

        # Store the trajectory of belief states
        belief_trajectory_list = []
        
        # Process evidence sequentially
        for t in range(seq_len):
            # Note: self.forward expects input shape [batch_size, hidden_dim]
            current_evidence = evidence_sequence[t] # Shape: [batch_size, hidden_dim]
            # Update belief state using the forward method
            belief, _ = self.forward(current_evidence, belief) # Ignore belief_embedding here
            belief_trajectory_list.append(belief.clone()) # Store a clone

        # Stack the list of tensors into a single tensor for the trajectory
        belief_trajectory = torch.stack(belief_trajectory_list, dim=0) # Shape: [seq_len, batch_size, belief_dim]
        
        # The final belief is the last state in the sequence
        final_belief = belief
        
        return final_belief, belief_trajectory


class SpatialBayesianLayer(nn.Module):
    """
    Specialized Bayesian layer for spatial probabilistic reasoning.
    
    Maintains and updates a belief distribution over a fixed set of spatial 
    locations, potentially representing reward likelihoods or object locations.
    Uses a more explicit multiplicative update rule followed by normalization.
    
    Attributes:
        spatial_dim (int): Number of spatial locations to track.
        feature_dim (int): Feature dimension associated with each location's belief.
        belief_update (nn.Module): Network to compute likelihood factors.
    """
    
    def __init__(self, spatial_dim, feature_dim, bit_linear=True):
        """
        Initialize spatial Bayesian layer.
        
        Args:
            spatial_dim (int): Number of spatial locations.
            feature_dim (int): Dimension of features per location.
            bit_linear (bool): Whether to use BitLinear layers.
        """
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        LinearLayer = BitLinear if bit_linear else nn.Linear
        
        # Network to compute likelihood update factors based on prior and evidence
        # Takes concatenated [prior_belief_feature, evidence_feature] per location
        self.belief_update = nn.Sequential(
            LinearLayer(feature_dim * 2, feature_dim),
            nn.ReLU(), # Or another suitable activation
            LinearLayer(feature_dim, feature_dim),
            nn.Sigmoid() # Output likelihood factors between 0 and 1
        )
        
    def forward(self, evidence_features, prior_belief=None):
        """
        Updates the spatial belief distribution based on new evidence features.
        
        Args:
            evidence_features (torch.Tensor): Evidence features for each location.
                                              Shape: [batch_size, spatial_dim, feature_dim].
            prior_belief (torch.Tensor, optional): Prior belief distribution over locations.
                                                  Shape: [batch_size, spatial_dim, feature_dim].
                                                  Defaults to uniform if None.
            
        Returns:
            torch.Tensor: Updated posterior belief distribution over locations.
                          Shape: [batch_size, spatial_dim, feature_dim]. 
                          Normalized across the spatial dimension.
        """
        batch_size = evidence_features.shape[0]
        device = evidence_features.device

        # Initialize uniform prior belief if not provided
        if prior_belief is None:
            # Uniform probability across spatial locations for each feature dimension
            prior_belief = torch.ones(batch_size, self.spatial_dim, self.feature_dim, device=device)
            prior_belief = prior_belief / self.spatial_dim 
        
        prior_belief = prior_belief.to(device) # Ensure device match

        # Concatenate prior belief and current evidence for each location
        # Shape: [batch_size, spatial_dim, feature_dim * 2]
        update_input = torch.cat([prior_belief, evidence_features], dim=-1)
        
        # Compute likelihood factors using the belief update network
        # Shape: [batch_size, spatial_dim, feature_dim]
        likelihood_factors = self.belief_update(update_input)
        
        # Apply Bayes rule approximation: posterior ~ prior * likelihood
        # Element-wise multiplication
        posterior_unnorm = prior_belief * likelihood_factors
        
        # Normalize the posterior distribution across the spatial dimension (dim=1)
        # Ensure numerical stability by clamping the normalizer sum
        normalizer = posterior_unnorm.sum(dim=1, keepdim=True).clamp(min=1e-10)
        posterior_belief = posterior_unnorm / normalizer
        
        return posterior_belief


def kl_divergence_loss(predicted_distribution, target_distribution, reduction='mean'):
    """
    Calculates Kullback-Leibler (KL) divergence loss.
    
    Useful for training models where the output should match a target probability 
    distribution (e.g., training the Bayesian module to match true posteriors).
    
    Args:
        predicted_distribution (torch.Tensor): Logits or unnormalized scores from the model.
                                              Shape: [batch_size, num_classes].
        target_distribution (torch.Tensor): Target probability distribution. 
                                           Shape: [batch_size, num_classes]. 
                                           Should sum to 1 along the last dimension.
        reduction (str): Specifies the reduction to apply to the output: 
                         'none' | 'mean' | 'sum'. Default: 'mean'.

    Returns:
        torch.Tensor: The calculated KL divergence loss (scalar if reduction is 'mean' or 'sum').
    """
    # Apply log_softmax to model predictions to get log probabilities
    log_pred = F.log_softmax(predicted_distribution, dim=-1)
    
    # Ensure target is a valid probability distribution (sums to 1)
    # Clamp target to avoid log(0) issues if target contains zeros
    target = target_distribution.clamp(min=1e-8)
    # No need to re-normalize target if it's already guaranteed to be a distribution

    # Calculate KL divergence: sum(target * (log(target) - log(predicted)))
    # Note: F.kl_div expects log probabilities as input (log_pred) and probabilities as target.
    # It computes sum(target * (log(target) - log_pred))
    # Ensure target and log_pred have the same shape.
    
    # F.kl_div(input, target) computes D_KL(target || input)
    # 'input' should be log-probabilities, 'target' should be probabilities
    kl_div = F.kl_div(log_pred, target, reduction=reduction, log_target=False) 
    # Using log_target=False because our target is probabilities.
    # If target was log-probabilities, set log_target=True.
    
    return kl_div


def generate_bayesian_training_data(num_samples: int, 
                                    num_hypotheses: int, 
                                    sequence_length: int, 
                                    output_feature_dim: int = None, 
                                    device: str ='cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic sequential evidence data for training Bayesian modules.
    
    Creates sequences where evidence subtly favors a randomly chosen 'true' 
    hypothesis. Calculates the ground-truth posterior probabilities at each 
    step using a simplified explicit Bayes update rule.

    **WARNING:** The default evidence generation creates features of dimension 
    `num_hypotheses * 2`. If using this data directly with a module expecting a 
    different input dimension (like the main model's `hidden_dim`), you MUST 
    either adapt this function (using `output_feature_dim`) or adapt the data 
    preprocessing/module input layer.

    Args:
        num_samples (int): Number of training sequences to generate.
        num_hypotheses (int): Number of distinct hypotheses the evidence relates to.
        sequence_length (int): Length of each evidence sequence.
        output_feature_dim (int, optional): If provided, the generated evidence features 
                                           will be padded with zeros or truncated to match 
                                           this dimension. Defaults to None (using 
                                           `num_hypotheses * 2`).
        device (str): Device ('cpu' or 'cuda') for tensor allocation.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - evidence_sequences (torch.Tensor): Synthetic evidence sequences.
                Shape: [num_samples, sequence_length, feature_dim], where 
                feature_dim is `output_feature_dim` if specified, otherwise `num_hypotheses * 2`.
            - posterior_probs (torch.Tensor): Ground-truth posterior probabilities 
                                             after each evidence step.
                Shape: [num_samples, sequence_length, num_hypotheses].
    """
    # Internal feature dimension for generating evidence structure
    internal_feature_dim = num_hypotheses * 2
    
    # Determine the final output dimension for evidence
    if output_feature_dim is None:
        final_feature_dim = internal_feature_dim
    else:
        final_feature_dim = output_feature_dim
        if final_feature_dim < internal_feature_dim:
             print(f"Warning: output_feature_dim ({final_feature_dim}) is less than internal "
                   f"feature dim ({internal_feature_dim}). Evidence will be truncated.")
        
    # Generate random initial prior probabilities for each sample
    # Shape: [num_samples, num_hypotheses]
    prior_probs = torch.rand(num_samples, num_hypotheses, device=device)
    prior_probs = prior_probs / prior_probs.sum(dim=1, keepdim=True) # Normalize priors

    # Initialize tensors to store the results
    evidence_sequences = torch.zeros(num_samples, sequence_length, final_feature_dim, device=device)
    posterior_probs = torch.zeros(num_samples, sequence_length, num_hypotheses, device=device)
    
    # Generate sequences and compute posteriors sample by sample
    for i in range(num_samples):
        # Randomly select the 'true' hypothesis for this sequence
        true_hypothesis_idx = torch.randint(0, num_hypotheses, (1,), device=device).item()
        
        # Initialize the belief state with the prior for this sample
        current_belief = prior_probs[i].clone() # Shape: [num_hypotheses]
        
        for t in range(sequence_length):
            # 1. Generate base evidence (random noise)
            # Using internal dimension for structured generation
            base_evidence = torch.randn(internal_feature_dim, device=device) * 0.5 
            
            # 2. Add a signal favoring the true hypothesis
            # Random signal strength for variability
            signal_strength = 0.5 + 0.5 * torch.rand(1, device=device).item() 
            # Add signal to specific dimensions corresponding to the true hypothesis
            # (This is a simple, arbitrary way to encode the signal)
            base_evidence[true_hypothesis_idx : true_hypothesis_idx + num_hypotheses] += signal_strength
            
            # 3. Pad or truncate evidence to match final_feature_dim
            if final_feature_dim == internal_feature_dim:
                final_evidence = base_evidence
            elif final_feature_dim > internal_feature_dim:
                # Pad with zeros
                padding = torch.zeros(final_feature_dim - internal_feature_dim, device=device)
                final_evidence = torch.cat((base_evidence, padding))
            else: # final_feature_dim < internal_feature_dim
                # Truncate
                final_evidence = base_evidence[:final_feature_dim]

            # Store the final evidence vector for this time step
            evidence_sequences[i, t] = final_evidence
            
            # 4. Update the ground-truth belief using a simple Bayes rule
            #    This requires defining a simple likelihood model P(evidence|hypothesis)
            
            # Simple likelihood model (example): Assume evidence dimensions relate to hypotheses
            # Higher sum in relevant dimensions means higher likelihood for that hypothesis.
            likelihoods = torch.zeros(num_hypotheses, device=device)
            for h in range(num_hypotheses):
                # Calculate likelihood based on the sum of relevant evidence dimensions
                # This is a very basic model and can be made more sophisticated.
                # Using the original base_evidence before padding/truncation for likelihood calc.
                likelihood_score = base_evidence[h : h + num_hypotheses].sum()
                # Convert score to probability-like value (e.g., via exp)
                likelihoods[h] = torch.exp(likelihood_score / num_hypotheses) # Average effect

            # Bayesian update: posterior ~ prior * likelihood
            posterior_unnorm = current_belief * likelihoods
            # Normalize to get the new posterior probability distribution
            current_belief = posterior_unnorm / posterior_unnorm.sum().clamp(min=1e-10) 
            
            # Store the calculated posterior probability for this time step
            posterior_probs[i, t] = current_belief
            
    return evidence_sequences, posterior_probs