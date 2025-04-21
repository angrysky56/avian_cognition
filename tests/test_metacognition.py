"""
Unit tests for the Metacognition module.

These tests verify the basic functionality of the metacognition module,
including forward pass correctness and training capability.
"""

import os
import sys
import unittest
import torch
import numpy as np

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.modules.metacognition import (
    MetacognitionModule,
    confidence_calibration_loss,
    expected_calibration_error
)


class TestMetacognitionModule(unittest.TestCase):
    """Tests for the MetacognitionModule."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_dim = 64
        self.batch_size = 8
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create a small module for testing
        self.module = MetacognitionModule(
            hidden_dim=self.hidden_dim,
            intermediate_dim=32,
            bit_linear=False  # Use standard linear layers for testing
        ).to(self.device)
        
        # Create sample hidden states and correctness labels
        self.hidden_states = torch.randn(
            self.batch_size, self.hidden_dim, device=self.device
        )
        self.correctness = torch.randint(
            0, 2, (self.batch_size, 1), device=self.device, dtype=torch.float32
        )
    
    def test_module_initialization(self):
        """Test that the module initializes correctly."""
        # Check dimensions
        self.assertEqual(self.module.hidden_dim, self.hidden_dim)
        self.assertEqual(self.module.intermediate_dim, 32)
        
        # Check layer dimensions
        self.assertEqual(self.module.hidden_transform.in_features, self.hidden_dim)
        self.assertEqual(self.module.hidden_transform.out_features, 32)
        self.assertEqual(self.module.confidence_head.in_features, 32)
        self.assertEqual(self.module.confidence_head.out_features, 1)
    
    def test_forward_pass(self):
        """Test the forward pass of the metacognition module."""
        # Run forward pass
        confidence = self.module(self.hidden_states)
        
        # Check output shape
        self.assertEqual(confidence.shape, (self.batch_size, 1))
        
        # Check output range
        self.assertTrue(torch.all(confidence >= 0))
        self.assertTrue(torch.all(confidence <= 1))
    
    def test_forward_pass_with_gradients(self):
        """Test the forward pass with gradient tracking."""
        # Ensure gradients are being tracked
        self.hidden_states.requires_grad = True
        
        # Run forward pass
        confidence = self.module(self.hidden_states)
        
        # Compute loss
        loss = confidence_calibration_loss(confidence, self.correctness)
        
        # Check that loss is a scalar
        self.assertEqual(loss.numel(), 1)
        
        # Check that gradients can flow
        loss.backward()
        
        # Check that gradients are not None
        for param in self.module.parameters():
            self.assertIsNotNone(param.grad)
        
        # Check that hidden_states has gradients
        self.assertIsNotNone(self.hidden_states.grad)
    
    def test_training_loop(self):
        """Test that the module can be trained."""
        # Create optimizer
        optimizer = torch.optim.Adam(self.module.parameters(), lr=0.01)
        
        # Initial predictions and loss
        with torch.no_grad():
            initial_confidence = self.module(self.hidden_states)
            initial_loss = confidence_calibration_loss(initial_confidence, self.correctness)
        
        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            confidence = self.module(self.hidden_states)
            loss = confidence_calibration_loss(confidence, self.correctness)
            loss.backward()
            optimizer.step()
        
        # Final predictions and loss
        with torch.no_grad():
            final_confidence = self.module(self.hidden_states)
            final_loss = confidence_calibration_loss(final_confidence, self.correctness)
        
        # Check that loss decreased
        self.assertLessEqual(final_loss.item(), initial_loss.item())
    
    def test_bit_linear_variant(self):
        """Test the BitLinear variant of the module."""
        try:
            # Attempt to create module with BitLinear
            bit_module = MetacognitionModule(
                hidden_dim=self.hidden_dim,
                intermediate_dim=32,
                bit_linear=True  # Use BitLinear if available
            ).to(self.device)
            
            # Run forward pass
            confidence = bit_module(self.hidden_states)
            
            # Check output shape
            self.assertEqual(confidence.shape, (self.batch_size, 1))
            
            # Check output range
            self.assertTrue(torch.all(confidence >= 0))
            self.assertTrue(torch.all(confidence <= 1))
            
        except ImportError:
            # Skip test if BitLinear is not available
            self.skipTest("BitLinear not available. Skipping test_bit_linear_variant.")
    
    def test_calibration_error_function(self):
        """Test the Expected Calibration Error function."""
        # Create toy confidence scores and correctness labels
        confidences = np.array([0.2, 0.3, 0.7, 0.8, 0.9])
        correctness = np.array([0, 0, 1, 1, 1])
        
        # Calculate ECE
        ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
            confidences, correctness, n_bins=2
        )
        
        # Check that ECE is a scalar
        self.assertEqual(np.isscalar(ece), True)
        
        # Check that ECE is non-negative
        self.assertGreaterEqual(ece, 0)
        
        # Check that bin counts sum to the number of samples
        self.assertEqual(np.sum(bin_counts), len(confidences))
        
        # Perfect calibration case
        perfect_confidences = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        perfect_ece, _, _, _ = expected_calibration_error(
            perfect_confidences, correctness, n_bins=2
        )
        
        # Check that perfect calibration gives ECE = 0
        self.assertAlmostEqual(perfect_ece, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
