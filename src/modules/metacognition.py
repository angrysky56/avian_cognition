"""
Metacognition Module

This module implements a neural component for self-monitoring and estimating
calibrated confidence (uncertainty) based on a model's internal state, 
inspired by certainty-encoding mechanisms observed in the avian pallium.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Attempt to import BitNet components, fall back to standard PyTorch layers
try:
    from src.core.bitnet import BitLinear
    print("Successfully imported BitLinear for Metacognition module.")
except ImportError:
    print("Warning: BitLinear not found. Falling back to nn.Linear for Metacognition module.")
    BitLinear = nn.Linear

# Optional imports for evaluation functions
try:
    import numpy as np
except ImportError:
    np = None # Flag that numpy is unavailable

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle # Used in some potential plotting variants
except ImportError:
    plt = None # Flag that matplotlib is unavailable


class MetacognitionModule(nn.Module):
    """
    Neural circuit for estimating task-specific confidence (uncertainty).
    
    Takes a hidden state representation from a primary model (e.g., the final 
    hidden state before prediction) and outputs a scalar value between 0 and 1, 
    intended to represent the model's confidence in its own prediction for 
    that specific input. Requires training against actual correctness labels 
    to become well-calibrated.
    
    Attributes:
        hidden_dim (int): Dimension of the input hidden state.
        intermediate_dim (int): Dimension of the intermediate representation.
        hidden_transform (nn.Module): Transforms input hidden state.
        confidence_head (nn.Module): Predicts the confidence logit.
        activation (nn.Module): Non-linear activation function.
        sigmoid (nn.Module): Sigmoid activation to output confidence in [0, 1].
    """
    
    def __init__(self, hidden_dim, intermediate_dim=None, bit_linear=True):
        """
        Initialize the metacognition module.
        
        Args:
            hidden_dim (int): Dimension of the input hidden state.
            intermediate_dim (int, optional): Dimension for the intermediate layer. 
                                              Defaults to hidden_dim // 2.
            bit_linear (bool): Whether to use BitLinear layers.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        # Set intermediate dimension, default to half of hidden_dim
        self.intermediate_dim = intermediate_dim if intermediate_dim is not None else hidden_dim // 2
        # Ensure intermediate_dim is at least 1
        if self.intermediate_dim < 1:
             print(f"Warning: Calculated intermediate_dim ({self.intermediate_dim}) is less than 1. Setting to 1.")
             self.intermediate_dim = 1
             
        LinearLayer = BitLinear if bit_linear else nn.Linear
        
        # Transformation layer from input hidden state
        self.hidden_transform = LinearLayer(hidden_dim, self.intermediate_dim)
        
        # Final layer predicting a single confidence logit
        self.confidence_head = LinearLayer(self.intermediate_dim, 1)
        
        # Activation functions
        self.activation = nn.GELU() # GELU is common, could use ReLU etc.
        self.sigmoid = nn.Sigmoid() # To squash the output logit to [0, 1] range
        
    def forward(self, hidden_state):
        """
        Predicts a confidence score based on the input hidden state.
        
        Args:
            hidden_state (torch.Tensor): The input hidden state, typically from the 
                                         main model before the final prediction layer.
                                         Shape: [batch_size, hidden_dim].
            
        Returns:
            torch.Tensor: A scalar confidence score for each input in the batch.
                          Shape: [batch_size, 1], Values: [0, 1].
        """
        # Device placement is handled by PyTorch based on input tensor's device
        
        # 1. Transform the hidden state and apply activation
        # Shape: [batch_size, intermediate_dim]
        intermediate_repr = self.activation(self.hidden_transform(hidden_state))
        
        # 2. Predict the confidence logit from the intermediate representation
        # Shape: [batch_size, 1]
        confidence_logit = self.confidence_head(intermediate_repr)
        
        # 3. Apply sigmoid to get the final confidence score between 0 and 1
        # Shape: [batch_size, 1]
        confidence = self.sigmoid(confidence_logit)
        
        return confidence


# --- Evaluation and Loss Functions ---

def confidence_calibration_loss(confidence, correctness):
    """
    Calculates Binary Cross-Entropy loss for training confidence prediction.
    
    This loss encourages the predicted `confidence` score to match the actual 
    `correctness` (probability of being correct, represented as 0 or 1).
    
    Args:
        confidence (torch.Tensor): Predicted confidence scores (output of the module).
                                   Shape: [batch_size, 1] or [batch_size]. Values: [0, 1].
        correctness (torch.Tensor): Binary labels indicating if the main model's 
                                    prediction was correct (1) or incorrect (0).
                                    Shape: [batch_size, 1] or [batch_size].
        
    Returns:
        torch.Tensor: The scalar BCE loss value (lower means better calibration).
    """
    # Ensure correctness is float and has the same shape as confidence
    correctness = correctness.view_as(confidence).float()
    
    # Calculate Binary Cross Entropy loss
    # F.binary_cross_entropy expects inputs in [0, 1] range, which sigmoid ensures for confidence.
    loss = F.binary_cross_entropy(confidence, correctness) 
    
    return loss


def expected_calibration_error(confidence, correctness, n_bins=15):
    """
    Calculates the Expected Calibration Error (ECE).
    
    ECE measures the discrepancy between predicted confidence scores and the 
    actual accuracy of predictions within those confidence ranges (bins). 
    A lower ECE indicates better calibration (confidence reflects accuracy).
    
    Args:
        confidence (Union[torch.Tensor, np.ndarray]): Predicted confidence scores.
                                                       Shape: [num_samples] or [num_samples, 1]. Values: [0, 1].
        correctness (Union[torch.Tensor, np.ndarray]): Binary correctness labels (0 or 1).
                                                        Shape: [num_samples] or [num_samples, 1].
        n_bins (int): Number of bins to divide the [0, 1] confidence range into.
        
    Returns:
        Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
            - ece (float): The calculated Expected Calibration Error.
            - bin_accuracies (np.ndarray): Accuracy within each confidence bin. Shape: [n_bins].
            - bin_confidences (np.ndarray): Average confidence within each bin. Shape: [n_bins].
            - bin_counts (np.ndarray): Number of samples falling into each bin. Shape: [n_bins].
    """
    if np is None:
        raise ImportError("Numpy is required to calculate Expected Calibration Error.")

    # Ensure inputs are numpy arrays on the CPU
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.detach().cpu().numpy().flatten()
    elif not isinstance(confidence, np.ndarray):
        confidence = np.array(confidence).flatten()
        
    if isinstance(correctness, torch.Tensor):
        correctness = correctness.detach().cpu().numpy().flatten()
    elif not isinstance(correctness, np.ndarray):
        correctness = np.array(correctness).flatten()
        
    if confidence.shape != correctness.shape:
        raise ValueError(f"Confidence shape {confidence.shape} must match correctness shape {correctness.shape}")

    num_samples = len(confidence)
    if num_samples == 0:
        return 0.0, np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)

    # Define the edges of the confidence bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Calculate statistics for each bin
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    
    for bin_idx, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find samples whose confidence falls into the current bin
        # Include lower boundary for the first bin, upper boundary for others
        if bin_idx == 0:
            in_bin = confidence <= upper
        else:
            in_bin = (confidence > lower) & (confidence <= upper)
            
        bin_counts[bin_idx] = np.sum(in_bin)
        
        if bin_counts[bin_idx] > 0:
            # Calculate accuracy (mean correctness) for samples in the bin
            bin_accuracies[bin_idx] = np.mean(correctness[in_bin])
            # Calculate average confidence for samples in the bin
            bin_confidences[bin_idx] = np.mean(confidence[in_bin])
            
    # Calculate ECE: weighted average of the absolute difference between accuracy and confidence
    # Weight by the proportion of samples in each bin
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / num_samples
    
    return ece, bin_accuracies, bin_confidences, bin_counts


def plot_reliability_diagram(confidence, correctness, n_bins=15, title=None, ax=None):
    """
    Generates a reliability diagram (calibration curve) plot.
    
    Visualizes model calibration by plotting accuracy as a function of predicted 
    confidence. A perfectly calibrated model's curve lies along the diagonal.
    Also displays the Expected Calibration Error (ECE).
    
    Args:
        confidence (Union[torch.Tensor, np.ndarray]): Predicted confidence scores.
        correctness (Union[torch.Tensor, np.ndarray]): Binary correctness labels (0 or 1).
        n_bins (int): Number of confidence bins for the plot.
        title (str, optional): Title for the plot. Defaults to 'Reliability Diagram'.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. If None, 
                                             a new figure and axis are created.
        
    Returns:
        Tuple[matplotlib.figure.Figure, float]: 
            - fig: The Matplotlib figure containing the plot (or None if matplotlib unavailable).
            - ece: The calculated Expected Calibration Error.
    """
    if plt is None:
        print("Warning: Matplotlib is required for plotting reliability diagrams.")
        # Still calculate ECE even if plotting is unavailable
        try:
            ece, _, _, _ = expected_calibration_error(confidence, correctness, n_bins)
            return None, ece
        except ImportError:
            return None, float('nan') # Cannot calculate ECE without numpy either

    # Calculate calibration metrics needed for the plot
    try:
        ece, bin_accuracies, bin_confidences, bin_counts = expected_calibration_error(
            confidence, correctness, n_bins
        )
    except ImportError: # Handle missing numpy
        print("Warning: Numpy required for ECE calculation.")
        return None, float('nan')

    # Create a new figure and axis if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure # Get figure from existing axis
    
    # Define bin properties
    bin_width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1, n_bins, endpoint=False) + bin_width / 2
    
    # Filter out bins with zero counts for plotting clarity
    valid_bins = bin_counts > 0
    
    # Plot the accuracy bars for bins with samples
    ax.bar(bin_centers[valid_bins], bin_accuracies[valid_bins], 
           width=bin_width, alpha=0.7, color='blue', 
           label='Accuracy', edgecolor='black')
           
    # Plot the average confidence within each bin (optional, can show bias)
    # ax.plot(bin_centers[valid_bins], bin_confidences[valid_bins], marker='o', linestyle='-', color='red', label='Avg. Confidence')

    # Plot the ideal calibration line (diagonal)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    # Add ECE value text to the plot
    ax.text(0.05, 0.95, f'ECE = {ece:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    
    # Configure plot appearance
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    plot_title = title if title is not None else 'Reliability Diagram'
    ax.set_title(plot_title, fontsize=16)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Add count annotations above bars (optional, can clutter plot)
    # for i, count in enumerate(bin_counts):
    #     if count > 0:
    #         ax.text(bin_centers[i], bin_accuracies[i] + 0.02, f'n={count}', ha='center', va='bottom', fontsize=8)
            
    return fig, ece