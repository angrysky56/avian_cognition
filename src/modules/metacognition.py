"""
Metacognition Module

This module implements a neural circuit for self-monitoring and calibrated
uncertainty estimation, inspired by certainty-encoding neurons in the corvid pallium.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.calibration import calibration_curve

from src.core.bitnet import BitLinear


class MetacognitionModule(nn.Module):
    """
    Neural circuit for self-monitoring and uncertainty estimation.
    
    Predicts the model's confidence in its outputs based on internal state.
    Inspired by certainty-encoding neurons in the corvid pallium that provide
    birds with awareness of their own knowledge states.
    
    Attributes:
        hidden_dim: Dimension of hidden representation
        hidden_transform: Neural projection for confidence feature extraction
        confidence_head: Output projection for confidence score
        activation: Non-linear activation function
        sigmoid: Sigmoid activation for confidence normalization
    """
    
    def __init__(self, hidden_dim, intermediate_dim=None, bit_linear=True):
        """
        Initialize metacognition module with BitLinear quantization.
        
        Args:
            hidden_dim: Dimension of the input hidden state
            intermediate_dim: Dimension of intermediate representation (defaults to hidden_dim//2)
            bit_linear: Whether to use BitLinear quantization
        """
        super().__init__()
        
        # Set dimensions
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim if intermediate_dim is not None else hidden_dim // 2
        
        # Hidden state transformation
        self.hidden_transform = BitLinear(hidden_dim, self.intermediate_dim) if bit_linear else nn.Linear(hidden_dim, self.intermediate_dim)
        
        # Confidence prediction head
        self.confidence_head = BitLinear(self.intermediate_dim, 1) if bit_linear else nn.Linear(self.intermediate_dim, 1)
        
        # Activation functions
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, hidden_state):
        """
        Produces a calibrated confidence estimate from model's hidden state.
        
        Args:
            hidden_state: The final hidden state from the model backbone
                          [batch_size, hidden_dim]
            
        Returns:
            confidence: A scalar confidence score (0-1) for each input
                        [batch_size, 1]
        """
        # Extract confidence features
        x = self.activation(self.hidden_transform(hidden_state))
        
        # Predict confidence logit
        logit = self.confidence_head(x)
        
        # Normalize to 0-1 range
        confidence = self.sigmoid(logit)
        
        return confidence


def confidence_calibration_loss(confidence, correctness):
    """
    Binary cross-entropy loss for calibrating confidence predictions.
    
    Trains the metacognition module to produce confidence scores that
    match the actual probability of correctness.
    
    Args:
        confidence: Predicted confidence scores (0-1) [batch_size, 1]
        correctness: Binary labels indicating correct (1) or incorrect (0) [batch_size, 1]
        
    Returns:
        loss: Calibration loss (lower means better calibration)
    """
    return F.binary_cross_entropy(confidence, correctness.float())


def expected_calibration_error(confidence, correctness, n_bins=10):
    """
    Calculates Expected Calibration Error (ECE) for reliability evaluation.
    
    ECE measures the difference between predicted confidence and actual accuracy
    across confidence bins. Lower values indicate better calibration.
    
    Args:
        confidence: Predicted confidence scores (0-1) [batch_size, 1]
        correctness: Binary labels indicating correct (1) or incorrect (0) [batch_size, 1]
        n_bins: Number of confidence bins for ECE calculation
        
    Returns:
        ece: Expected Calibration Error (lower is better)
        bin_accuracies: Accuracy in each confidence bin
        bin_confidences: Average confidence in each bin
        bin_counts: Number of examples in each bin
    """
    # Ensure tensors are on CPU and in numpy
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.detach().cpu().numpy().flatten()
    if isinstance(correctness, torch.Tensor):
        correctness = correctness.detach().cpu().numpy().flatten()
        
    # Define bin edges
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Calculate bin statistics
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find examples in this bin
        in_bin = np.logical_and(confidence > bin_lower, confidence <= bin_upper)
        bin_counts[bin_idx] = np.sum(in_bin)
        
        if bin_counts[bin_idx] > 0:
            bin_accuracies[bin_idx] = np.mean(correctness[in_bin])
            bin_confidences[bin_idx] = np.mean(confidence[in_bin])
            
    # Calculate ECE
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / np.sum(bin_counts)
    
    return ece, bin_accuracies, bin_confidences, bin_counts


def plot_reliability_diagram(confidence, correctness, n_bins=10, ax=None):
    """
    Generates a reliability diagram to visualize calibration quality.
    
    A reliability diagram plots predicted confidence against observed accuracy.
    Perfect calibration follows the diagonal.
    
    Args:
        confidence: Predicted confidence scores (0-1)
        correctness: Binary labels indicating correct (1) or incorrect (0)
        n_bins: Number of confidence bins
        ax: Optional matplotlib axis for plotting
        
    Returns:
        fig: Matplotlib figure with reliability diagram
        ece: Expected Calibration Error
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("Matplotlib required for plotting reliability diagrams.")
        return None, None
    
    # Calculate calibration metrics
    ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
        confidence, correctness, n_bins
    )
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure
    
    # Plot reliability diagram
    bin_width = 1.0 / n_bins
    
    # Plot accuracy bars
    for i, (acc, conf, count) in enumerate(zip(bin_accs, bin_confs, bin_counts)):
        if count > 0:
            x = i * bin_width + bin_width / 2
            ax.bar(x, acc, width=bin_width, alpha=0.8, color='b', edgecolor='black')
    
    # Plot confidence
    bin_centers = np.linspace(0, 1, n_bins, endpoint=False) + bin_width / 2
    ax.plot(bin_centers, bin_confs, marker='o', linestyle='-', color='r', label='Confidence')
    
    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    
    # Annotate with ECE
    ax.text(0.05, 0.95, f'ECE = {ece:.4f}', transform=ax.transAxes,
            fontsize=14, verticalalignment='top')
    
    # Configure plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Reliability Diagram (Calibration Curve)', fontsize=16)
    ax.legend(loc='lower right')
    ax.grid(True)
    
    return fig, ece
