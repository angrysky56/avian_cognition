"""
Numerical Competence Module Training Protocol

This module implements the specialized neural cultivation process for the
numerical competence circuit, enabling precise arithmetic operations that
generalize beyond training distributions through structural bias induction.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from .base_trainer import CognitiveTrainer


class NumericalCompetenceTrainer(CognitiveTrainer):
    """
    Specialized evolutionary orchestrator for mathematical abstraction.
    
    Cultivates the system's capacity for precise numerical operations through
    a developmental protocol that transcends pattern-matching approximation,
    inducing structural biases that facilitate algebraic generalization beyond
    training distributions.
    
    The training rhythm emphasizes compositional arithmetic understanding,
    scaffolding symbolic manipulation capabilities analogous to the counting
    and arithmetic abilities observed in corvids and parrots.
    
    Attributes:
        model: Numerical competence module or integrated architecture
        backbone: Optional underlying model for generating representations
        operations: Mathematical operations to train on
        value_ranges: Ranges for training and extrapolation validation
        extrapolation_factor: Target generalization factor beyond training
    """
    
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        config,
        backbone=None,
        operations=None,
        value_ranges=None,
        extrapolation_factor=10,
        experiment_name="NumericalTraining"
    ):
        """
        Initialize the numerical training protocol.
        
        Args:
            model: Neural architecture to evolve
            optimizer: Parameter update mechanism
            criterion: Loss evaluation function
            device: Computational substrate
            config: Hyperparameter configuration
            backbone: Optional base model for generating representations
            operations: Mathematical operations to train on
            value_ranges: Ranges for training and extrapolation validation
            extrapolation_factor: Target generalization factor
            experiment_name: Identifier for this evolutionary sequence
        """
        super().__init__(model, optimizer, criterion, device, config, experiment_name)
        
        self.backbone = backbone
        
        # Set default operations if not provided
        self.operations = operations if operations is not None else ['add', 'subtract', 'multiply', 'divide']
        
        # Set default value ranges if not provided
        self.value_ranges = value_ranges if value_ranges is not None else {
            'train': (0, 100),
            'validation': (0, 100),
            'extrapolation': (100, 1000)
        }
        
        self.extrapolation_factor = extrapolation_factor
        
        # Track performance metrics
        self.train_accuracy_history = []
        self.validation_accuracy_history = []
        self.extrapolation_accuracy_history = []
        self.operation_accuracy = {op: [] for op in self.operations}
        
    def _extract_numerical_module(self, model):
        """
        Isolate numerical circuitry from integrated architecture.
        
        Many architectural variants may contain numerical components,
        this function navigates the architectural hierarchy to locate
        the specific circuitry responsible for mathematical operations.
        
        Args:
            model: Integrated neural architecture
            
        Returns:
            numerical_module: Isolated numerical circuit
        """
        # Direct access if model is the numerical module itself
        if hasattr(model, 'arithmetic_units'):
            return model
            
        # Extract from AvianMamba architecture
        if hasattr(model, 'numerical_module'):
            return model.numerical_module
            
        # Find module by name in children
        for name, child in model.named_children():
            if 'num' in name.lower() and hasattr(child, 'arithmetic_units'):
                return child
                
        # Default to the provided model
        return model
        
    def _generate_batch(self, batch_size, operation, value_range):
        """
        Generate synthetic arithmetic problems.
        
        Creates pairs of operands and expected results for training
        the numerical module on specific arithmetic operations.
        
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
        scaling_factor = 1.0 / max(1.0, max_magnitude)
        
        for i, (a, b) in enumerate(operands):
            # Convert to Python floats first to avoid numpy dtype issues with CUDA tensors
            h1[i, 0] = float(a) * float(scaling_factor)
            h1[i, 1] = float(a // 100) * float(scaling_factor)  # Hundreds place
            h1[i, 2] = float((a % 100) // 10) * float(scaling_factor)  # Tens place
            h1[i, 3] = float(a % 10) * float(scaling_factor)  # Ones place
            
            h2[i, 0] = float(b) * float(scaling_factor)
            h2[i, 1] = float(b // 100) * float(scaling_factor)  # Hundreds place
            h2[i, 2] = float((b % 100) // 10) * float(scaling_factor)  # Tens place
            h2[i, 3] = float(b % 10) * float(scaling_factor)  # Ones place
            
        # Encode operation
        op_idx = self.operations.index(operation)
        h_op[:, op_idx] = 1.0
        
        return (h1, h2, h_op), results, operands
        
    def _evaluate_accuracy(self, predictions, targets, tolerance=0.05):
        """
        Evaluate numerical accuracy with appropriate tolerance.
        
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
        
    def train_epoch(self, train_loader=None):
        """
        Conduct a single evolutionary cycle for numerical competence.
        
        This cycle orchestrates the refinement of arithmetic operations
        across a spectrum of mathematical functions, developing the module's
        capacity for precise calculation and algebraic generalization.
        
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
            # Select operation to train on
            operation = np.random.choice(self.operations)
            
            # Either use provided data or generate synthetic problems
            if train_loader is not None:
                try:
                    batch = next(iter(train_loader))
                    h1, h2, h_op = batch['operands']
                    targets = batch['results']
                    operands = batch.get('original_operands', None)
                except Exception as e:
                    print(f"Error loading batch from data loader: {e}")
                    print("Falling back to synthetic data generation")
                    # Generate synthetic batch as fallback
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
            
            # Forward pass through numerical module with decoder
            if hasattr(self.model, 'numerical_module'):
                # Model with decoder head
                predictions, result_hidden, op_weights = self.model(h1, h2, h_op)
            else:
                # Original model without decoder head
                result_hidden, op_weights = numerical_module(h1, h2, h_op)
                
                # Fall back to the original approach if necessary
                max_magnitude = max(abs(t.item()) for t in targets.view(-1))
                scaling_factor = 1.0 / max(1.0, max_magnitude)
                predictions = result_hidden[:, 0:1] / scaling_factor
            
            # Calculate loss
            loss = self.criterion(predictions.view(-1, 1), targets)
            
            # Backward propagation of alignment signal
            loss.backward()
            
            # Update parameters along improvement gradient
            self.optimizer.step()
            
            # Accumulate metrics
            batch_size = h1.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Evaluate accuracy for this operation
            accuracy, _ = self._evaluate_accuracy(predictions.view(-1, 1), targets)
            operation_correct[operation] += accuracy * batch_size
            operation_total[operation] += batch_size
            
            # Update step counter
            self.global_step += 1
            
        # Calculate epoch-level metrics
        avg_loss = total_loss / total_samples
        
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
        
        Tests both in-distribution performance and generalization beyond
        the training range, assessing the emergence of true mathematical
        abstraction versus pattern matching.
        
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
                    # Try to use provided validation data
                    try:
                        for batch_idx in range(len(val_loader)):
                            try:
                                batch = next(iter(val_loader))
                                h1, h2, h_op = batch['operands']
                                targets = batch['results']
                                
                                # Forward pass with decoder
                                if hasattr(self.model, 'numerical_module'):
                                    # Model with decoder head
                                    predictions, result_hidden, op_weights = self.model(h1, h2, h_op)
                                else:
                                    # Original model without decoder head
                                    result_hidden, op_weights = numerical_module(h1, h2, h_op)
                                    
                                    # Fall back to the original approach if necessary
                                    max_magnitude = max(abs(t.item()) for t in targets.view(-1))
                                    scaling_factor = 1.0 / max(1.0, max_magnitude)
                                    predictions = result_hidden[:, 0:1] / scaling_factor
                                
                                # Calculate loss
                                loss = self.criterion(predictions.view(-1, 1), targets)
                                
                                # Accumulate metrics
                                current_batch_size = h1.size(0)
                                scenario_loss += loss.item() * current_batch_size
                                scenario_samples += current_batch_size
                                
                                # Try to determine operation from h_op one-hot encoding
                                try:
                                    operation_idx = torch.argmax(h_op[0]).item()
                                    if operation_idx < len(self.operations):
                                        operation = self.operations[operation_idx]
                                    else:
                                        # Default to first operation if index is out of bounds
                                        operation = self.operations[0]
                                except:
                                    # Default to first operation if there's any issue
                                    operation = self.operations[0]
                                
                                # Evaluate accuracy for this operation
                                accuracy, _ = self._evaluate_accuracy(predictions.view(-1, 1), targets)
                                scenario_operation_correct[operation] += accuracy * current_batch_size
                                scenario_operation_total[operation] += current_batch_size
                            except StopIteration:
                                # Re-initialize iterator if needed
                                break
                    except Exception as e:
                        print(f"Error during validation with real data: {e}")
                        print("Falling back to synthetic data for validation")
                        # Continue with synthetic data generation below
                        use_synthetic = True
                    else:
                        # Skip synthetic data generation if real data was used successfully
                        use_synthetic = False
                else:
                    # No real data provided, use synthetic
                    use_synthetic = True
                    
                # Generate synthetic data if needed
                if scenario_name == 'extrapolation' or use_synthetic:
                    # Test each operation with synthetic data
                    for operation in self.operations:
                        for batch_i in range(test_batches):
                            # Generate synthetic batch
                            (h1, h2, h_op), targets, operands = self._generate_batch(
                                batch_size, operation, value_range
                            )
                            
                            # Forward pass with decoder
                            if hasattr(self.model, 'numerical_module'):
                                # Model with decoder head
                                predictions, result_hidden, op_weights = self.model(h1, h2, h_op)
                            else:
                                # Original model without decoder head
                                result_hidden, op_weights = numerical_module(h1, h2, h_op)
                                
                                # Fall back to the original approach if necessary
                                max_magnitude = max(abs(t.item()) for t in targets.view(-1))
                                scaling_factor = 1.0 / max(1.0, max_magnitude)
                                predictions = result_hidden[:, 0:1] / scaling_factor
                            
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
        
    def _plot_operation_accuracy(self):
        """
        Visualize accuracy across different arithmetic operations.
        
        Creates a plot showing how well the numerical module performs
        on different operations, highlighting its strengths and weaknesses.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        epochs = range(len(self.train_accuracy_history))
        
        # Plot overall accuracy
        ax.plot(epochs, self.train_accuracy_history, 'k-', linewidth=2, label='Overall')
        
        # Plot per-operation accuracy
        for op in self.operations:
            if len(self.operation_accuracy[op]) > 0:
                # Pad with zeros for any missing epochs
                padded_accuracy = [0] * len(epochs)
                for i, acc in enumerate(self.operation_accuracy[op]):
                    if i < len(padded_accuracy):
                        padded_accuracy[i] = acc
                
                ax.plot(epochs, padded_accuracy, '--', label=op.capitalize())
        
        # Configure plot
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Numerical Operation Accuracy')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Save figure
        save_path = f"{self.log_dir}/plots/operation_accuracy_epoch_{self.epoch}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        
        # Log to tensorboard
        self.writer.add_figure(
            "numerical/operation_accuracy",
            fig,
            global_step=self.epoch
        )
        
        # Close figure to prevent memory leaks
        plt.close(fig)
        
    def _plot_extrapolation_performance(self):
        """
        Visualize extrapolation performance compared to in-distribution accuracy.
        
        Creates a plot showing how well the numerical module generalizes
        beyond its training distribution, a key indicator of true
        mathematical abstraction versus pattern matching.
        """
        if len(self.validation_accuracy_history) == 0 or len(self.extrapolation_accuracy_history) == 0:
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        epochs = range(len(self.validation_accuracy_history))
        
        # Plot validation and extrapolation accuracy
        ax.plot(epochs, self.validation_accuracy_history, 'b-', linewidth=2, label='Validation (In-Distribution)')
        ax.plot(epochs, self.extrapolation_accuracy_history, 'r-', linewidth=2, label='Extrapolation')
        
        # Configure plot
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Numerical Extrapolation Performance')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculate extrapolation gap
        if epochs:
            val_acc = self.validation_accuracy_history[-1]
            ext_acc = self.extrapolation_accuracy_history[-1]
            gap = val_acc - ext_acc
            
            ax.text(0.02, 0.02, f"Current gap: {gap:.3f}", transform=ax.transAxes,
                   fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        # Save figure
        save_path = f"{self.log_dir}/plots/extrapolation_epoch_{self.epoch}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        
        # Log to tensorboard
        self.writer.add_figure(
            "numerical/extrapolation",
            fig,
            global_step=self.epoch
        )
        
        # Close figure to prevent memory leaks
        plt.close(fig)
        
    def plot_training_trajectory(self):
        """
        Visualize the evolutionary trajectory of numerical competence.
        
        Creates a comprehensive view of the system's development across
        different operations and generalization scenarios.
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Operation accuracy
        ax1 = axes[0]
        epochs = range(len(self.train_accuracy_history))
        
        # Overall accuracy
        ax1.plot(epochs, self.train_accuracy_history, 'k-', linewidth=2, label='Overall')
        
        # Per-operation accuracy
        for op in self.operations:
            if len(self.operation_accuracy[op]) > 0:
                # Pad with zeros for any missing epochs
                padded_accuracy = [0] * len(epochs)
                for i, acc in enumerate(self.operation_accuracy[op]):
                    if i < len(padded_accuracy):
                        padded_accuracy[i] = acc
                
                ax1.plot(epochs, padded_accuracy, '--', label=op.capitalize())
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Operation Accuracy')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Extrapolation performance
        ax2 = axes[1]
        
        if len(self.validation_accuracy_history) > 0 and len(self.extrapolation_accuracy_history) > 0:
            epochs_val = range(len(self.validation_accuracy_history))
            
            # Validation and extrapolation accuracy
            ax2.plot(epochs_val, self.validation_accuracy_history, 'b-', linewidth=2, label='Validation (In-Distribution)')
            ax2.plot(epochs_val, self.extrapolation_accuracy_history, 'r-', linewidth=2, label='Extrapolation')
            
            # Calculate extrapolation ratio
            if epochs_val:
                val_acc = self.validation_accuracy_history[-1]
                ext_acc = self.extrapolation_accuracy_history[-1]
                
                if val_acc > 0:
                    ratio = ext_acc / val_acc
                    ax2.text(0.02, 0.02, f"Extrapolation ratio: {ratio:.3f}", transform=ax2.transAxes,
                           fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Extrapolation Performance')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure
        save_path = f"{self.log_dir}/plots/numerical_training_trajectory.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        
        # Log to tensorboard
        self.writer.add_figure(
            "training/numerical_trajectory",
            fig,
            global_step=self.epoch
        )
        
        # Close figure to prevent memory leaks
        plt.close(fig)
