"""
Fixed Numerical Trainer for Avian Cognition project.

This module implements an extended version of the NumericalCompetenceTrainer
that fixes device handling and gradient tracking issues.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from training.numerical_trainer import NumericalCompetenceTrainer

class FixedNumericalTrainer(NumericalCompetenceTrainer):
    """
    An extended version of NumericalCompetenceTrainer that fixes device handling
    for synthetic data generation.
    """
    
    def _evaluate_accuracy(self, predictions, targets, tolerance=0.05):
        """
        Evaluate numerical accuracy with appropriate tolerance.
        
        Fixed version that properly detaches tensors before conversion to numpy.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            tolerance: Relative tolerance for correctness
            
        Returns:
            accuracy: Proportion of correct predictions
            errors: Absolute errors for each prediction
        """
        # Ensure tensor format
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions, dtype=torch.float32, device=self.device)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
            
        # Calculate absolute errors
        abs_errors = torch.abs(predictions - targets)
        
        # Calculate relative tolerance for each target
        rel_tolerance = torch.abs(targets) * tolerance
        rel_tolerance = torch.max(rel_tolerance, torch.tensor(1e-2, device=self.device))  # Minimum tolerance
        
        # Check if within tolerance
        within_tolerance = abs_errors <= rel_tolerance
        
        # Calculate accuracy - detach to avoid gradient tracking
        accuracy = within_tolerance.float().mean().detach().item()
        
        # Detach before converting to numpy to avoid the runtime error
        return accuracy, abs_errors.detach().cpu().numpy()
    
    def _generate_batch(self, batch_size, operation, value_range):
        """
        Generate synthetic arithmetic problems with proper device handling.
        
        This fixed version ensures all tensors are properly created on the
        correct device to avoid device mismatch errors.
        
        Args:
            batch_size: Number of problems to generate
            operation: Mathematical operation ('add', 'subtract', etc.)
            value_range: Range of values for operands (min_val, max_val)
            
        Returns:
            operands: Pairs of operand values
            results: Expected results for each operation
            hidden_states: Encoded hidden state representations
        """
        min_val, max_val = value_range
        hidden_dim = self.config.get('hidden_dim', 256)
        
        # Generate operand pairs with appropriate constraints
        operands = []
        results = []
        
        for _ in range(batch_size):
            # Generate operands within range
            a = np.random.randint(min_val, max_val)
            
            # For division, ensure clean division (when possible)
            if operation == 'divide':
                if a > 1 and np.random.random() < 0.8:  # 80% clean division
                    # Ensure valid range for randint (low < high)
                    high_val = min(a, max_val // 2)
                    if high_val > 1:
                        b = np.random.randint(1, high_val)
                        b = b * (a // b) if a // b > 0 else b  # Make a divisible by b
                    else:
                        b = 1  # Default to 1 if no valid range
                else:
                    b = max(1, np.random.randint(min_val, max_val))  # Avoid division by zero
            else:
                b = np.random.randint(min_val, max_val)
                
            # Compute result based on operation
            if operation == 'add':
                result = a + b
            elif operation == 'subtract':
                result = a - b
            elif operation == 'multiply':
                result = a * b
            elif operation == 'divide':
                result = a / b
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
            operands.append((a, b))
            results.append(result)
        
        # Convert to tensors
        results = torch.tensor(results, dtype=torch.float32, device=self.device).view(batch_size, 1)
        
        # Create hidden state representations for operands
        h1 = torch.zeros(batch_size, hidden_dim, device=self.device)
        h2 = torch.zeros(batch_size, hidden_dim, device=self.device)
        h_op = torch.zeros(batch_size, hidden_dim, device=self.device)
        
        # Simple numeric encoding (in practice, this would be learned)
        max_magnitude = max(max_val, max(abs(r) for r in results.view(-1).cpu().numpy()))
        scaling_factor = float(1.0 / max(1.0, max_magnitude))  # Convert to Python float to avoid numpy types
        
        for i, (a, b) in enumerate(operands):
            # Convert values to Python floats to avoid numpy types which can cause device mismatch
            h1[i, 0] = float(a) * scaling_factor
            h1[i, 1] = float(a // 100) * scaling_factor  # Hundreds place
            h1[i, 2] = float((a % 100) // 10) * scaling_factor  # Tens place
            h1[i, 3] = float(a % 10) * scaling_factor  # Ones place
            
            h2[i, 0] = float(b) * scaling_factor
            h2[i, 1] = float(b // 100) * scaling_factor  # Hundreds place
            h2[i, 2] = float((b % 100) // 10) * scaling_factor  # Tens place
            h2[i, 3] = float(b % 10) * scaling_factor  # Ones place
            
        # Encode operation
        op_idx = self.operations.index(operation)
        h_op[:, op_idx] = 1.0
        
        return (h1, h2, h_op), results, operands
    
    def train_epoch(self, train_loader=None):
        """
        Conduct a single evolutionary cycle for numerical competence.
        
        Fixed version that properly handles custom dataset wrapper.
        
        Args:
            train_loader: Optional iterator through training examples
                         (if None, generate synthetic data)
            
        Returns:
            metrics: Quantified aspects of numerical competence
        """
        # Set to training mode
        self.model.train()
        if self.backbone is not None:
            self.backbone.eval()  # Freeze backbone during numerical training
            
        # Isolate numerical module if needed
        numerical_module = self._extract_numerical_module(self.model)
        
        # Tracking metrics
        total_loss = 0
        total_samples = 0
        operation_correct = {op: 0 for op in self.operations}
        operation_total = {op: 0 for op in self.operations}
        
        # Determine number of batches
        if train_loader is not None:
            num_batches = len(train_loader)
        else:
            # Default number of synthetic batches
            num_batches = self.config.get('synthetic_batches', 100)
            
        # Batch size
        batch_size = self.config.get('batch_size', 32)
        
        # Training value range
        train_range = self.value_ranges['train']
        
        # Evolutionary iteration
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {self.epoch}"):
            # Select operation to train on (for synthetic data)
            operation = np.random.choice(self.operations)
            
            # Either use provided data or generate synthetic problems
            if train_loader is not None:
                try:
                    batch = next(iter(train_loader))
                    h1, h2, h_op = batch['operands']
                    targets = batch['results']
                    # Default operands for tracking
                    operands = [(0, 0)] * len(h1)  # Placeholder
                except Exception as e:
                    print(f"Error loading batch: {e}")
                    print("Falling back to synthetic data generation")
                    (h1, h2, h_op), targets, operands = self._generate_batch(
                        batch_size, operation, train_range
                    )
            else:
                # Generate synthetic batch
                (h1, h2, h_op), targets, operands = self._generate_batch(
                    batch_size, operation, train_range
                )
            
            # Reset gradient pathways
            self.optimizer.zero_grad()
            
            # Forward pass through numerical module
            result_hidden, op_weights = numerical_module(h1, h2, h_op)
            
            # Decode result (simplified extraction from first dimension)
            max_magnitude = max(abs(t.item()) for t in targets.view(-1))
            scaling_factor = 1.0 / max(1.0, max_magnitude)
            
            predictions = result_hidden[:, 0] / scaling_factor
            
            # Calculate loss
            loss = self.criterion(predictions.view(-1, 1), targets)
            
            # Backward propagation of alignment signal
            loss.backward()
            
            # Update parameters along improvement gradient
            self.optimizer.step()
            
            # Accumulate metrics
            current_batch_size = h1.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size
            
            # Determine operation for metrics collection
            if train_loader is not None:
                # Try to determine operation from one-hot encoding in h_op
                if h_op.size(1) >= len(self.operations):
                    op_idx = torch.argmax(h_op[0]).item()
                    if op_idx < len(self.operations):
                        operation = self.operations[op_idx]
            
            # Evaluate accuracy for this operation
            accuracy, _ = self._evaluate_accuracy(predictions.view(-1, 1), targets)
            operation_correct[operation] += accuracy * current_batch_size
            operation_total[operation] += current_batch_size
            
            # Update step counter
            self.global_step += 1
            
        # Calculate epoch-level metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Calculate per-operation accuracy
        operation_accuracy = {}
        for op in self.operations:
            if operation_total[op] > 0:
                operation_accuracy[op] = operation_correct[op] / operation_total[op]
            else:
                operation_accuracy[op] = 0.0
                
        # Calculate overall accuracy
        overall_accuracy = sum(operation_correct.values()) / sum(operation_total.values()) if sum(operation_total.values()) > 0 else 0.0
        
        # Record accuracies
        self.train_accuracy_history.append(overall_accuracy)
        for op in self.operations:
            if operation_total[op] > 0:
                self.operation_accuracy[op].append(operation_accuracy[op])
        
        # Generate visualizations
        if self.epoch % self.config.get('plot_interval', 5) == 0:
            self._plot_operation_accuracy()
            
        # Return consolidated metrics
        metrics = {
            'loss': avg_loss,
            'accuracy': overall_accuracy,
            **{f"{op}_accuracy": operation_accuracy[op] for op in self.operations}
        }
        
        return metrics
        
    def validate(self, val_loader=None):
        """
        Evaluate numerical competence on validation and extrapolation tasks.
        
        This method is modified to properly use the DatasetWrapper if provided,
        while still supporting the original synthetic data generation if needed.
        
        Args:
            val_loader: Optional iterator through validation examples
                       (if None, generate synthetic data)
            
        Returns:
            metrics: Quantified aspects of numerical competence
        """
        # Set to evaluation mode
        self.model.eval()
        if self.backbone is not None:
            self.backbone.eval()
            
        # Isolate numerical module if needed
        numerical_module = self._extract_numerical_module(self.model)
        
        # Tracking metrics
        validation_metrics = {}
        
        # Define test scenarios
        scenarios = [
            ('validation', self.value_ranges['validation']),
            ('extrapolation', self.value_ranges['extrapolation'])
        ]
        
        # Batch size
        batch_size = self.config.get('batch_size', 32)
        
        # Number of test batches per operation
        test_batches = self.config.get('val_batches_per_op', 5)
        
        # Evaluation without gradient tracking
        with torch.no_grad():
            # Test each scenario
            for scenario_name, value_range in scenarios:
                # Initialize metrics for this scenario
                scenario_loss = 0
                scenario_samples = 0
                scenario_operation_correct = {op: 0 for op in self.operations}
                scenario_operation_total = {op: 0 for op in self.operations}
                
                # Handle validation with real data if provided
                if val_loader is not None and scenario_name == 'validation':
                    # Real data validation
                    try:
                        for batch_idx in range(len(val_loader)):
                            try:
                                batch = next(iter(val_loader))
                                h1, h2, h_op = batch['operands']
                                targets = batch['results']
                                
                                # Forward pass
                                result_hidden, op_weights = numerical_module(h1, h2, h_op)
                                
                                # Decode result (simplified extraction from first dimension)
                                max_magnitude = max(abs(t.item()) for t in targets.view(-1))
                                scaling_factor = 1.0 / max(1.0, max_magnitude)
                                
                                predictions = result_hidden[:, 0] / scaling_factor
                                
                                # Calculate loss
                                loss = self.criterion(predictions.view(-1, 1), targets)
                                
                                # Accumulate metrics
                                current_batch_size = h1.size(0)
                                scenario_loss += loss.item() * current_batch_size
                                scenario_samples += current_batch_size
                                
                                # Get the operation type and calculate metrics for it
                                # Assume all examples in the batch are the same operation
                                try:
                                    operation_idx = torch.argmax(h_op[0]).item()
                                    if operation_idx < len(self.operations):
                                        operation = self.operations[operation_idx]
                                    else:
                                        # Default to first operation if index is out of bounds
                                        operation = self.operations[0]
                                except Exception as e:
                                    # Default to first operation if there's any issue
                                    operation = self.operations[0]
                                
                                # Evaluate accuracy for this operation
                                accuracy, _ = self._evaluate_accuracy(predictions.view(-1, 1), targets)
                                scenario_operation_correct[operation] += accuracy * current_batch_size
                                scenario_operation_total[operation] += current_batch_size
                            except StopIteration:
                                # Re-initialize iterator
                                val_loader = iter(val_loader)
                                continue
                    except Exception as e:
                        print(f"Error during validation: {e}")
                        print("Falling back to synthetic data for validation")
                        # Fall back to synthetic data
                        for operation in self.operations:
                            for _ in range(test_batches):
                                # Generate synthetic batch
                                (h1, h2, h_op), targets, operands = self._generate_batch(
                                    batch_size, operation, value_range
                                )
                                
                                # Forward pass through numerical module
                                result_hidden, op_weights = numerical_module(h1, h2, h_op)
                                
                                # Decode result (simplified extraction from first dimension)
                                max_magnitude = max(abs(t.item()) for t in targets.view(-1))
                                scaling_factor = 1.0 / max(1.0, max_magnitude)
                                
                                predictions = result_hidden[:, 0] / scaling_factor
                                
                                # Calculate loss
                                loss = self.criterion(predictions.view(-1, 1), targets)
                                
                                # Accumulate metrics
                                scenario_loss += loss.item() * batch_size
                                scenario_samples += batch_size
                                
                                # Evaluate accuracy for this operation
                                accuracy, _ = self._evaluate_accuracy(predictions.view(-1, 1), targets)
                                scenario_operation_correct[operation] += accuracy * batch_size
                                scenario_operation_total[operation] += batch_size
                else:
                    # Use synthetic data for this scenario
                    # Test each operation
                    for operation in self.operations:
                        for batch_i in range(test_batches):
                            # Generate synthetic batch
                            (h1, h2, h_op), targets, operands = self._generate_batch(
                                batch_size, operation, value_range
                            )
                            
                            # Forward pass through numerical module
                            result_hidden, op_weights = numerical_module(h1, h2, h_op)
                            
                            # Decode result (simplified extraction from first dimension)
                            max_magnitude = max(abs(t.item()) for t in targets.view(-1))
                            scaling_factor = 1.0 / max(1.0, max_magnitude)
                            
                            predictions = result_hidden[:, 0] / scaling_factor
                            
                            # Calculate loss
                            loss = self.criterion(predictions.view(-1, 1), targets)
                            
                            # Accumulate metrics
                            scenario_loss += loss.item() * batch_size
                            scenario_samples += batch_size
                            
                            # Evaluate accuracy for this operation
                            accuracy, _ = self._evaluate_accuracy(predictions.view(-1, 1), targets)
                            scenario_operation_correct[operation] += accuracy * batch_size
                            scenario_operation_total[operation] += batch_size
                            
                            # Log some examples (first batch only)
                            if batch_i == 0 and scenario_name == 'extrapolation':
                                examples = []
                                for i in range(min(5, batch_size)):
                                    if operands:
                                        a, b = operands[i]
                                        pred = predictions[i].item()
                                        targ = targets[i].item()
                                        error = abs(pred - targ)
                                        examples.append((a, b, pred, targ, error))
                                
                                if examples:
                                    self.logger.info(f"Extrapolation examples ({operation}):")
                                    for a, b, pred, targ, error in examples:
                                        self.logger.info(f"  {a} {operation} {b} = {pred:.2f} (correct: {targ:.2f}, error: {error:.2f})")
                
                # Calculate scenario metrics
                scenario_avg_loss = scenario_loss / scenario_samples if scenario_samples > 0 else 0.0
                
                # Calculate per-operation accuracy
                scenario_operation_accuracy = {}
                for op in self.operations:
                    if scenario_operation_total[op] > 0:
                        scenario_operation_accuracy[op] = scenario_operation_correct[op] / scenario_operation_total[op]
                    else:
                        scenario_operation_accuracy[op] = 0.0
                        
                # Calculate overall accuracy
                scenario_overall_accuracy = sum(scenario_operation_correct.values()) / sum(scenario_operation_total.values()) if sum(scenario_operation_total.values()) > 0 else 0.0
                
                # Record in metrics
                validation_metrics[f"{scenario_name}_loss"] = scenario_avg_loss
                validation_metrics[f"{scenario_name}_accuracy"] = scenario_overall_accuracy
                
                for op in self.operations:
                    validation_metrics[f"{scenario_name}_{op}_accuracy"] = scenario_operation_accuracy[op]
                    
                # Record in history
                if scenario_name == 'validation':
                    self.validation_accuracy_history.append(scenario_overall_accuracy)
                elif scenario_name == 'extrapolation':
                    self.extrapolation_accuracy_history.append(scenario_overall_accuracy)
        
        # Generate extrapolation visualization
        if self.epoch % self.config.get('plot_interval', 5) == 0:
            self._plot_extrapolation_performance()
            
        return validation_metrics
