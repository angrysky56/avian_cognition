"""
Data Utilities for Cognitive Module Training

This module provides specialized data generation and processing utilities
for training the avian cognitive architecture, creating semantic substrates
that enable the emergence of sophisticated cognition.
"""

import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader


class MetacognitionDataGenerator:
    """
    Data generator for metacognitive awareness cultivation.
    
    Creates structured training scenarios where hidden state representations
    contain latent indicators of correctness, allowing the metacognition
    module to develop calibrated confidence estimation.
    
    The generator embeds pattern-based signatures of correctness and error
    within the representations, simulating the latent features that would
    emerge in real model activations as indicators of knowledge certainty.
    
    Attributes:
        hidden_dim: Dimension of hidden state representations
        pattern_strength: Intensity of correctness/error patterns
        correctness_patterns: Tensor patterns associated with correctness
        error_patterns: Tensor patterns associated with errors
    """
    
    def __init__(self, hidden_dim=256, pattern_strength=0.5, num_patterns=5, seed=42):
        """
        Initialize metacognition data generator.
        
        Args:
            hidden_dim: Dimension of hidden state representations
            pattern_strength: Intensity of correctness/error patterns
            num_patterns: Number of distinct patterns for each class
            seed: Random seed for reproducibility
        """
        self.hidden_dim = hidden_dim
        self.pattern_strength = pattern_strength
        self.num_patterns = num_patterns
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate pattern signatures
        self.correctness_patterns = [
            torch.randn(hidden_dim) for _ in range(num_patterns)
        ]
        self.error_patterns = [
            torch.randn(hidden_dim) for _ in range(num_patterns)
        ]
        
    def generate_batch(self, batch_size, device='cpu'):
        """
        Generate a batch of hidden states with correctness labels.
        
        Args:
            batch_size: Number of examples to generate
            device: Device to place tensors on
            
        Returns:
            hidden_states: Generated hidden state representations
            correctness: Binary correctness indicators
        """
        # Initialize storage
        hidden_states = torch.zeros(batch_size, self.hidden_dim, device=device)
        correctness = torch.zeros(batch_size, 1, device=device)
        
        # Generate each example
        for i in range(batch_size):
            # Determine if this example is correct (50% probability)
            is_correct = random.random() > 0.5
            
            # Base representation (random noise)
            hidden_state = torch.randn(self.hidden_dim, device=device)
            
            if is_correct:
                # Add correctness patterns with random strength
                for pattern in self.correctness_patterns:
                    if random.random() > 0.5:  # Randomly apply patterns
                        strength = self.pattern_strength * random.random()
                        hidden_state += strength * pattern.to(device)
            else:
                # Add error patterns with random strength
                for pattern in self.error_patterns:
                    if random.random() > 0.5:  # Randomly apply patterns
                        strength = self.pattern_strength * random.random()
                        hidden_state += strength * pattern.to(device)
            
            # Store example
            hidden_states[i] = hidden_state
            correctness[i] = 1.0 if is_correct else 0.0
            
        return hidden_states, correctness
        
    def generate_calibration_sweep(self, num_points=100, device='cpu'):
        """
        Generate examples with varying degrees of certainty.
        
        Creates a spectrum of examples with different pattern strengths,
        resulting in a range of certainty levels for calibration testing.
        
        Args:
            num_points: Number of points in the certainty spectrum
            device: Device to place tensors on
            
        Returns:
            hidden_states: Generated hidden state representations
            correctness: Binary correctness indicators
            true_probs: Ground truth probability of correctness
        """
        # Initialize storage
        hidden_states = torch.zeros(num_points, self.hidden_dim, device=device)
        correctness = torch.zeros(num_points, 1, device=device)
        true_probs = torch.zeros(num_points, 1, device=device)
        
        # Generate examples with varying certainty
        for i in range(num_points):
            # Set certainty level (0.5 to 1.0)
            certainty = 0.5 + 0.5 * (i / (num_points - 1))
            true_probs[i] = certainty
            
            # Determine if this example is correct based on certainty
            is_correct = random.random() < certainty
            
            # Base representation (random noise)
            hidden_state = torch.randn(self.hidden_dim, device=device)
            
            # Pattern strength based on certainty
            pattern_strength = 2.0 * (certainty - 0.5) * self.pattern_strength
            
            if is_correct:
                # Add strong correctness pattern
                pattern = random.choice(self.correctness_patterns).to(device)
                hidden_state += pattern_strength * pattern
            else:
                # Add strong error pattern
                pattern = random.choice(self.error_patterns).to(device)
                hidden_state += pattern_strength * pattern
            
            # Store example
            hidden_states[i] = hidden_state
            correctness[i] = 1.0 if is_correct else 0.0
            
        return hidden_states, correctness, true_probs


class MetacognitionDataset(Dataset):
    """
    Dataset for metacognition training.
    
    Provides access to synthetic hidden states and correctness labels
    for training the metacognition module to predict confidence.
    
    Attributes:
        generator: MetacognitionDataGenerator instance
        size: Number of examples in dataset
        hidden_states: Pregenerated hidden state representations
        correctness: Pregenerated correctness indicators
    """
    
    def __init__(self, hidden_dim=256, size=10000, pattern_strength=0.5, seed=42, pregenerate=True, device='cpu'):
        """
        Initialize metacognition dataset.
        
        Args:
            hidden_dim: Dimension of hidden state representations
            size: Number of examples in dataset
            pattern_strength: Intensity of correctness/error patterns
            seed: Random seed for reproducibility
            pregenerate: Whether to pregenerate all examples
            device: Device to place tensors on
        """
        self.generator = MetacognitionDataGenerator(
            hidden_dim=hidden_dim,
            pattern_strength=pattern_strength,
            seed=seed
        )
        self.size = size
        self.device = device
        
        # Pregenerate data if requested
        if pregenerate:
            self.hidden_states, self.correctness = self.generator.generate_batch(size, device)
        else:
            self.hidden_states = None
            self.correctness = None
            
    def __len__(self):
        """Return the size of the dataset."""
        return self.size
        
    def __getitem__(self, idx):
        """
        Retrieve a single example.
        
        Args:
            idx: Index of example to retrieve
            
        Returns:
            sample: Dictionary containing hidden state and correctness
        """
        if self.hidden_states is not None:
            # Return pregenerated example
            return {
                'hidden_states': self.hidden_states[idx],
                'correctness': self.correctness[idx]
            }
        else:
            # Generate on-the-fly
            hidden_state, correctness = self.generator.generate_batch(1, self.device)
            return {
                'hidden_states': hidden_state[0],
                'correctness': correctness[0]
            }


class BayesianDataGenerator:
    """
    Data generator for Bayesian inference cultivation.
    
    Creates structured sequential evidence scenarios where latent
    variables must be inferred from observations, training the
    Bayesian module to perform probabilistic belief updating.
    
    The generator creates scenarios with hidden states and sequential
    evidence, requiring Bayesian inference to determine the correct
    posterior probabilities over hypotheses.
    
    Attributes:
        hidden_dim: Dimension of hidden state representations
        num_hypotheses: Number of hypotheses to distinguish
        evid_strength: Strength of evidence (higher = more decisive)
        hypotheses: Latent representations for each hypothesis
    """
    
    def __init__(self, hidden_dim=256, num_hypotheses=3, evid_strength=0.7, seed=42):
        """
        Initialize Bayesian data generator.
        
        Args:
            hidden_dim: Dimension of hidden state representations
            num_hypotheses: Number of hypotheses to distinguish
            evid_strength: Strength of evidence (higher = more decisive)
            seed: Random seed for reproducibility
        """
        self.hidden_dim = hidden_dim
        self.num_hypotheses = num_hypotheses
        self.evid_strength = evid_strength
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate hypothesis representations
        self.hypotheses = [
            torch.randn(hidden_dim) for _ in range(num_hypotheses)
        ]
        
    def generate_evidence_sequence(self, seq_len, true_hypothesis=None, device='cpu'):
        """
        Generate a sequence of evidence for a hypothesis.
        
        Creates a sequence of observations that provide evidence about
        the true hypothesis, requiring Bayesian integration to infer.
        
        Args:
            seq_len: Length of evidence sequence
            true_hypothesis: Index of true hypothesis (or None for random)
            device: Device to place tensors on
            
        Returns:
            evidence_sequence: Sequence of evidence observations
            posterior_probs: Ground truth posterior probabilities
            true_hyp_idx: Index of true hypothesis
        """
        # Select true hypothesis if not provided
        if true_hypothesis is None:
            true_hyp_idx = random.randrange(self.num_hypotheses)
        else:
            true_hyp_idx = true_hypothesis
            
        # Get true hypothesis representation
        true_hyp = self.hypotheses[true_hyp_idx].to(device)
        
        # Initialize uniform prior
        prior = torch.ones(self.num_hypotheses, device=device) / self.num_hypotheses
        
        # Initialize storage
        evidence_sequence = torch.zeros(seq_len, self.hidden_dim, device=device)
        posterior_probs = torch.zeros(seq_len, self.num_hypotheses, device=device)
        
        # Generate evidence sequence
        for t in range(seq_len):
            # Generate evidence (noisy version of true hypothesis)
            noise = torch.randn(self.hidden_dim, device=device)
            evidence = self.evid_strength * true_hyp + (1 - self.evid_strength) * noise
            evidence = torch.nn.functional.normalize(evidence, dim=0)
            
            # Calculate likelihoods for each hypothesis
            likelihoods = torch.zeros(self.num_hypotheses, device=device)
            
            for h in range(self.num_hypotheses):
                # Likelihood is based on similarity to hypothesis
                h_vec = self.hypotheses[h].to(device)
                similarity = torch.nn.functional.cosine_similarity(evidence, h_vec, dim=0)
                
                # Transform to likelihood (higher similarity = higher likelihood)
                likelihood = torch.exp(similarity * 2)  # Scale factor for contrast
                likelihoods[h] = likelihood
                
            # Normalize likelihoods
            likelihoods = likelihoods / likelihoods.sum()
            
            # Apply Bayes' rule
            posterior_unnorm = prior * likelihoods
            posterior = posterior_unnorm / posterior_unnorm.sum()
            
            # Store evidence and posterior
            evidence_sequence[t] = evidence
            posterior_probs[t] = posterior
            
            # Update prior for next step
            prior = posterior
            
        return evidence_sequence, posterior_probs, true_hyp_idx
        
    def generate_batch(self, batch_size, seq_len, device='cpu'):
        """
        Generate a batch of evidence sequences.
        
        Args:
            batch_size: Number of sequences to generate
            seq_len: Length of each evidence sequence
            device: Device to place tensors on
            
        Returns:
            evidence_sequences: Batch of evidence sequences
            posterior_probs: Batch of ground truth posteriors
            true_hyp_indices: Indices of true hypotheses
        """
        # Initialize storage
        evidence_sequences = torch.zeros(seq_len, batch_size, self.hidden_dim, device=device)
        posterior_probs = torch.zeros(seq_len, batch_size, self.num_hypotheses, device=device)
        true_hyp_indices = []
        
        # Generate each sequence
        for i in range(batch_size):
            # Generate sequence
            evidence, posteriors, true_hyp = self.generate_evidence_sequence(
                seq_len, device=device
            )
            
            # Store in batch tensors
            evidence_sequences[:, i, :] = evidence
            posterior_probs[:, i, :] = posteriors
            true_hyp_indices.append(true_hyp)
            
        return evidence_sequences, posterior_probs, true_hyp_indices


class BayesianDataset(Dataset):
    """
    Dataset for Bayesian inference training.
    
    Provides access to synthetic evidence sequences and posterior
    probabilities for training the Bayesian module on belief updating.
    
    Attributes:
        generator: BayesianDataGenerator instance
        size: Number of examples in dataset
        seq_len: Length of evidence sequences
        evidence_sequences: Pregenerated evidence sequences
        posterior_probs: Pregenerated posterior probabilities
    """
    
    def __init__(self, hidden_dim=256, num_hypotheses=3, evid_strength=0.7,
                size=1000, seq_len=10, seed=42, pregenerate=True, device='cpu'):
        """
        Initialize Bayesian dataset.
        
        Args:
            hidden_dim: Dimension of hidden state representations
            num_hypotheses: Number of hypotheses to distinguish
            evid_strength: Strength of evidence (higher = more decisive)
            size: Number of examples in dataset
            seq_len: Length of evidence sequences
            seed: Random seed for reproducibility
            pregenerate: Whether to pregenerate all examples
            device: Device to place tensors on
        """
        self.generator = BayesianDataGenerator(
            hidden_dim=hidden_dim,
            num_hypotheses=num_hypotheses,
            evid_strength=evid_strength,
            seed=seed
        )
        self.size = size
        self.seq_len = seq_len
        self.device = device
        
        # Pregenerate data if requested
        if pregenerate:
            self.evidence_sequences, self.posterior_probs, self.true_hyp_indices = self.generator.generate_batch(
                size, seq_len, device
            )
        else:
            self.evidence_sequences = None
            self.posterior_probs = None
            self.true_hyp_indices = None
            
    def __len__(self):
        """Return the size of the dataset."""
        return self.size
        
    def __getitem__(self, idx):
        """
        Retrieve a single example.
        
        Args:
            idx: Index of example to retrieve
            
        Returns:
            sample: Dictionary containing evidence and posteriors
        """
        if self.evidence_sequences is not None:
            # Return pregenerated example
            return {
                'evidence': self.evidence_sequences[:, idx, :],
                'posterior': self.posterior_probs[:, idx, :],
                'true_hypothesis': self.true_hyp_indices[idx]
            }
        else:
            # Generate on-the-fly
            evidence, posteriors, true_hyp = self.generator.generate_evidence_sequence(
                self.seq_len, device=self.device
            )
            return {
                'evidence': evidence,
                'posterior': posteriors,
                'true_hypothesis': true_hyp
            }


class PlanningDataGenerator:
    """
    Data generator for planning capability cultivation.
    
    Creates structured reasoning tasks with multi-step solutions,
    training the planning module to develop sequential reasoning
    and causal understanding capabilities.
    
    The generator creates scenarios with problem states, ground-truth
    reasoning steps, and final solutions, enabling the planning module
    to learn how to break down complex problems.
    
    Attributes:
        hidden_dim: Dimension of hidden state representations
        num_steps: Maximum number of reasoning steps
        task_types: Types of reasoning tasks to generate
        transform_strength: Intensity of transformations
    """
    
    def __init__(self, hidden_dim=256, num_steps=5, transform_strength=0.5, seed=42):
        """
        Initialize planning data generator.
        
        Args:
            hidden_dim: Dimension of hidden state representations
            num_steps: Maximum number of reasoning steps
            transform_strength: Intensity of transformations
            seed: Random seed for reproducibility
        """
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.transform_strength = transform_strength
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Define abstract reasoning patterns
        self.task_types = [
            'sequential_transformation',  # Apply transforms in sequence
            'recursive_pattern',          # Apply same transform repeatedly
            'hierarchical_grouping',      # Group and then transform
            'conditional_sequence',       # If-then branching logic
            'compositional_integration'   # Combine partial results
        ]
        
    def generate_task(self, steps, device='cpu'):
        """
        Generate a multi-step reasoning task.
        
        Creates a problem representation with ground-truth reasoning
        steps and solution, requiring sequential planning to solve.
        
        Args:
            steps: Number of reasoning steps
            device: Device to place tensors on
            
        Returns:
            problem_state: Problem context representation
            step_sequence: Ground-truth reasoning steps
            solution: Correct final solution
            context: Optional context sequence for attention
        """
        # Lambda for creating a random unit vector
        random_unit_vector = lambda: torch.nn.functional.normalize(
            torch.randn(self.hidden_dim, device=device), dim=0
        )
        
        # Create problem representation
        problem_state = torch.zeros(self.hidden_dim, device=device)
        
        # Create context sequence (for attention)
        seq_len = 5  # Fixed length for simplicity
        context = torch.zeros(seq_len, self.hidden_dim, device=device)
        
        # Initialize storage for ground-truth steps and solution
        step_sequence = torch.zeros(self.num_steps, self.hidden_dim, device=device)
        
        # Select random task type
        task_type = random.choice(self.task_types)
        
        # Encode task type in problem state
        task_idx = self.task_types.index(task_type)
        problem_state[task_idx] = 1.0
        
        # Create reference vectors for this problem
        base_vector = random_unit_vector()
        transform_vectors = [random_unit_vector() for _ in range(self.num_steps)]
        
        # Initialize context with task-relevant information
        for j in range(seq_len):
            # Encode task-specific context
            if j == 0:
                context[j] = base_vector
            elif j < seq_len - 1:
                context[j] = transform_vectors[min(j-1, self.num_steps-1)]
        
        # Generate ground-truth reasoning steps based on task type
        if task_type == 'sequential_transformation':
            # Apply sequence of different transformations
            current_state = base_vector
            for step in range(steps):
                # Apply transform
                current_state = (1 - self.transform_strength) * current_state + self.transform_strength * transform_vectors[step]
                current_state = torch.nn.functional.normalize(current_state, dim=0)
                
                # Store step
                step_sequence[step] = current_state
            
            # Final solution is the last state
            solution = current_state
            
        elif task_type == 'recursive_pattern':
            # Apply same transformation repeatedly
            transform = transform_vectors[0]
            current_state = base_vector
            
            for step in range(steps):
                # Apply transform
                current_state = (1 - self.transform_strength) * current_state + self.transform_strength * transform
                current_state = torch.nn.functional.normalize(current_state, dim=0)
                
                # Store step
                step_sequence[step] = current_state
            
            # Final solution is the last state
            solution = current_state
            
        elif task_type == 'hierarchical_grouping':
            # Group then transform
            groups = min(steps - 1, 2)  # At least 1 transform after grouping
            
            # Grouping steps
            current_state = base_vector
            for step in range(groups):
                # Combine with transform
                current_state = (1 - self.transform_strength) * current_state + self.transform_strength * transform_vectors[step]
                current_state = torch.nn.functional.normalize(current_state, dim=0)
                
                # Store step
                step_sequence[step] = current_state
            
            # Transformation steps
            for step in range(groups, steps):
                # Apply final transform
                current_state = (1 - self.transform_strength) * current_state + self.transform_strength * transform_vectors[groups]
                current_state = torch.nn.functional.normalize(current_state, dim=0)
                
                # Store step
                step_sequence[step] = current_state
            
            # Final solution is the last state
            solution = current_state
            
        elif task_type == 'conditional_sequence':
            # If-then branching logic based on initial condition
            branch_condition = (base_vector[0] > 0)  # Arbitrary condition
            
            current_state = base_vector
            for step in range(steps):
                # Apply different transform based on branch
                if branch_condition:
                    transform_idx = min(step, len(transform_vectors) - 1)
                else:
                    transform_idx = min(steps - 1 - step, len(transform_vectors) - 1)
                    
                transform = transform_vectors[transform_idx]
                
                # Apply transform
                current_state = (1 - self.transform_strength) * current_state + self.transform_strength * transform
                current_state = torch.nn.functional.normalize(current_state, dim=0)
                
                # Store step
                step_sequence[step] = current_state
            
            # Final solution is the last state
            solution = current_state
            
        elif task_type == 'compositional_integration':
            # Combine partial results
            partial_results = [base_vector]
            
            # Generate partial results
            for step in range(steps - 1):
                partial = (1 - self.transform_strength) * base_vector + self.transform_strength * transform_vectors[step]
                partial = torch.nn.functional.normalize(partial, dim=0)
                partial_results.append(partial)
                
                # Store step
                step_sequence[step] = partial
            
            # Final step integrates all partial results
            integration = sum(partial_results) / len(partial_results)
            integration = torch.nn.functional.normalize(integration, dim=0)
            
            # Store final step
            step_sequence[steps - 1] = integration
            
            # Final solution
            solution = integration
            
        # Pad remaining steps if needed
        if steps < self.num_steps:
            for step in range(steps, self.num_steps):
                step_sequence[step] = solution
        
        return problem_state, step_sequence, solution, context
        
    def generate_batch(self, batch_size, steps, device='cpu'):
        """
        Generate a batch of multi-step reasoning tasks.
        
        Args:
            batch_size: Number of tasks to generate
            steps: Number of reasoning steps per task
            device: Device to place tensors on
            
        Returns:
            problem_states: Batch of problem representations
            step_sequences: Batch of ground-truth reasoning steps
            solutions: Batch of correct final solutions
            contexts: Batch of context sequences for attention
        """
        # Initialize storage
        problem_states = torch.zeros(batch_size, self.hidden_dim, device=device)
        step_sequences = torch.zeros(self.num_steps, batch_size, self.hidden_dim, device=device)
        solutions = torch.zeros(batch_size, self.hidden_dim, device=device)
        contexts = torch.zeros(batch_size, 5, self.hidden_dim, device=device)  # Fixed context length of 5
        
        # Generate each task
        for i in range(batch_size):
            problem_state, step_sequence, solution, context = self.generate_task(steps, device)
            
            # Store in batch tensors
            problem_states[i] = problem_state
            step_sequences[:, i, :] = step_sequence
            solutions[i] = solution
            contexts[i] = context
            
        return problem_states, step_sequences, solutions, contexts


class PlanningDataset(Dataset):
    """
    Dataset for planning capability training.
    
    Provides access to synthetic reasoning tasks with multi-step
    solutions for training the planning module on sequential reasoning.
    
    Attributes:
        generator: PlanningDataGenerator instance
        size: Number of examples in dataset
        steps: Number of reasoning steps per task
        problem_states: Pregenerated problem representations
        step_sequences: Pregenerated ground-truth reasoning steps
        solutions: Pregenerated correct final solutions
    """
    
    def __init__(self, hidden_dim=256, num_steps=5, transform_strength=0.5,
                size=1000, steps=None, seed=42, pregenerate=True, device='cpu'):
        """
        Initialize planning dataset.
        
        Args:
            hidden_dim: Dimension of hidden state representations
            num_steps: Maximum number of reasoning steps
            transform_strength: Intensity of transformations
            size: Number of examples in dataset
            steps: Number of reasoning steps per task (or None for max)
            seed: Random seed for reproducibility
            pregenerate: Whether to pregenerate all examples
            device: Device to place tensors on
        """
        self.generator = PlanningDataGenerator(
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            transform_strength=transform_strength,
            seed=seed
        )
        self.size = size
        self.steps = steps if steps is not None else num_steps
        self.device = device
        
        # Pregenerate data if requested
        if pregenerate:
            self.problem_states, self.step_sequences, self.solutions, self.contexts = self.generator.generate_batch(
                size, self.steps, device
            )
        else:
            self.problem_states = None
            self.step_sequences = None
            self.solutions = None
            self.contexts = None
            
    def __len__(self):
        """Return the size of the dataset."""
        return self.size
        
    def __getitem__(self, idx):
        """
        Retrieve a single example.
        
        Args:
            idx: Index of example to retrieve
            
        Returns:
            sample: Dictionary containing problem state and steps
        """
        if self.problem_states is not None:
            # Return pregenerated example
            return {
                'problem_states': self.problem_states[idx],
                'step_sequences': self.step_sequences[:, idx, :],
                'solutions': self.solutions[idx],
                'contexts': self.contexts[idx]
            }
        else:
            # Generate on-the-fly
            problem_state, step_sequence, solution, context = self.generator.generate_task(
                self.steps, device=self.device
            )
            return {
                'problem_states': problem_state,
                'step_sequences': step_sequence,
                'solutions': solution,
                'contexts': context
            }


class NumericalDataGenerator:
    """
    Data generator for numerical competence cultivation.
    
    Creates structured arithmetic problems with ground-truth results,
    training the numerical module to perform precise calculations and
    develop mathematical abstraction that generalizes beyond training.
    
    The generator creates problems with operands, operations, and
    expected results, enabling the numerical module to learn exact
    arithmetic operations.
    
    Attributes:
        hidden_dim: Dimension of hidden state representations
        operations: Mathematical operations to generate
        value_ranges: Ranges for operand values
        encoding_factor: Scaling factor for numeric encoding
    """
    
    def __init__(self, hidden_dim=256, operations=None, value_ranges=None, seed=42):
        """
        Initialize numerical data generator.
        
        Args:
            hidden_dim: Dimension of hidden state representations
            operations: Mathematical operations to generate
            value_ranges: Ranges for operand values
            seed: Random seed for reproducibility
        """
        self.hidden_dim = hidden_dim
        
        # Set default operations if not provided
        self.operations = operations if operations is not None else ['add', 'subtract', 'multiply', 'divide']
        
        # Set default value ranges if not provided
        self.value_ranges = value_ranges if value_ranges is not None else {
            'train': (0, 100),
            'validation': (0, 100),
            'extrapolation': (100, 1000)
        }
        
        # Encoding factor (adjusted based on max value)
        self.encoding_factor = 0.01  # Will be adjusted dynamically
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def _encode_number(self, value, device='cpu'):
        """
        Encode a numeric value into a hidden state representation.
        
        Args:
            value: Numeric value to encode
            device: Device to place tensor on
            
        Returns:
            encoded: Hidden state representation of value
        """
        # Create hidden state
        encoded = torch.zeros(self.hidden_dim, device=device)
        
        # Simple numeric encoding in first few dimensions
        encoded[0] = value * self.encoding_factor
        
        # Encode digits in separate dimensions
        digit_values = [
            (value // 100) % 10,  # Hundreds place
            (value // 10) % 10,   # Tens place
            value % 10            # Ones place
        ]
        
        for i, digit in enumerate(digit_values):
            encoded[i + 1] = digit * 0.1  # Scale to [0, 1]
            
        # Add additional encoding for magnitude
        if value > 0:
            magnitude = np.log10(max(1, value))
            encoded[4] = magnitude * 0.1  # Scale to reasonable range
        
        return encoded
        
    def _encode_operation(self, operation, device='cpu'):
        """
        Encode an operation into a hidden state representation.
        
        Args:
            operation: Operation name
            device: Device to place tensor on
            
        Returns:
            encoded: Hidden state representation of operation
        """
        # Create hidden state
        encoded = torch.zeros(self.hidden_dim, device=device)
        
        # Encode operation as one-hot
        if operation in self.operations:
            op_idx = self.operations.index(operation)
            encoded[op_idx] = 1.0
            
        return encoded
        
    def generate_problem(self, operation=None, value_range=None, device='cpu'):
        """
        Generate a single arithmetic problem.
        
        Creates operands, operation, and expected result for an
        arithmetic problem, with hidden state representations.
        
        Args:
            operation: Specific operation or None for random
            value_range: Range for operand values
            device: Device to place tensors on
            
        Returns:
            h1: Hidden state for first operand
            h2: Hidden state for second operand
            h_op: Hidden state for operation
            result: Expected result
            operands: Original operand values
        """
        # Select operation if not provided
        if operation is None:
            operation = random.choice(self.operations)
            
        # Select value range if not provided
        if value_range is None:
            value_range = self.value_ranges['train']
            
        min_val, max_val = value_range
        
        # Generate operands with appropriate constraints
        if operation == 'divide':
            # For division, ensure clean division when possible
            a = random.randint(min_val, max_val)
            
            if a > 0 and random.random() < 0.8:  # 80% clean division
                b = random.randint(1, min(a, max_val // 2))
                b = b * (a // b) if a // b > 0 else b  # Make a divisible by b
            else:
                b = max(1, random.randint(min_val, max_val))  # Avoid division by zero
        else:
            a = random.randint(min_val, max_val)
            b = random.randint(min_val, max_val)
            
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
            
        # Adjust encoding factor based on result magnitude
        max_magnitude = max(abs(a), abs(b), abs(result))
        self.encoding_factor = 1.0 / max(1.0, max_magnitude)
        
        # Encode operands and operation
        h1 = self._encode_number(a, device)
        h2 = self._encode_number(b, device)
        h_op = self._encode_operation(operation, device)
        
        return h1, h2, h_op, torch.tensor([result], device=device), (a, b)
        
    def generate_batch(self, batch_size, operation=None, value_range=None, device='cpu'):
        """
        Generate a batch of arithmetic problems.
        
        Args:
            batch_size: Number of problems to generate
            operation: Specific operation or None for random
            value_range: Range for operand values
            device: Device to place tensors on
            
        Returns:
            h1_batch: Batch of hidden states for first operands
            h2_batch: Batch of hidden states for second operands
            h_op_batch: Batch of hidden states for operations
            results: Batch of expected results
            operands: Original operand values
        """
        # Initialize storage
        h1_batch = torch.zeros(batch_size, self.hidden_dim, device=device)
        h2_batch = torch.zeros(batch_size, self.hidden_dim, device=device)
        h_op_batch = torch.zeros(batch_size, self.hidden_dim, device=device)
        results = torch.zeros(batch_size, 1, device=device)
        operands_list = []
        
        # Select operation if not provided
        if operation is None:
            # Generate mixed operations
            for i in range(batch_size):
                op = random.choice(self.operations)
                h1, h2, h_op, result, ops = self.generate_problem(op, value_range, device)
                
                h1_batch[i] = h1
                h2_batch[i] = h2
                h_op_batch[i] = h_op
                results[i] = result
                operands_list.append(ops)
        else:
            # Generate single operation
            for i in range(batch_size):
                h1, h2, h_op, result, ops = self.generate_problem(operation, value_range, device)
                
                h1_batch[i] = h1
                h2_batch[i] = h2
                h_op_batch[i] = h_op
                results[i] = result
                operands_list.append(ops)
                
        return h1_batch, h2_batch, h_op_batch, results, operands_list


class NumericalDataset(Dataset):
    """
    Dataset for numerical competence training.
    
    Provides access to synthetic arithmetic problems with expected
    results for training the numerical module on exact calculation.
    
    Attributes:
        generator: NumericalDataGenerator instance
        size: Number of examples in dataset
        operation: Specific operation or None for mixed
        value_range: Range for operand values
        h1_batch: Pregenerated hidden states for first operands
        h2_batch: Pregenerated hidden states for second operands
        h_op_batch: Pregenerated hidden states for operations
        results: Pregenerated expected results
    """
    
    def __init__(self, hidden_dim=256, operations=None, value_ranges=None,
                size=1000, operation=None, value_range=None, seed=42,
                pregenerate=True, device='cpu'):
        """
        Initialize numerical dataset.
        
        Args:
            hidden_dim: Dimension of hidden state representations
            operations: Mathematical operations to generate
            value_ranges: Ranges for operand values
            size: Number of examples in dataset
            operation: Specific operation or None for mixed
            value_range: Range for operand values
            seed: Random seed for reproducibility
            pregenerate: Whether to pregenerate all examples
            device: Device to place tensors on
        """
        self.generator = NumericalDataGenerator(
            hidden_dim=hidden_dim,
            operations=operations,
            value_ranges=value_ranges,
            seed=seed
        )
        self.size = size
        self.operation = operation
        
        # Select value range if not provided
        if value_range is None:
            self.value_range = self.generator.value_ranges['train']
        else:
            self.value_range = value_range
            
        self.device = device
        
        # Pregenerate data if requested
        if pregenerate:
            self.h1_batch, self.h2_batch, self.h_op_batch, self.results, self.operands_list = self.generator.generate_batch(
                size, operation, self.value_range, device
            )
        else:
            self.h1_batch = None
            self.h2_batch = None
            self.h_op_batch = None
            self.results = None
            self.operands_list = None
            
    def __len__(self):
        """Return the size of the dataset."""
        return self.size
        
    def __getitem__(self, idx):
        """
        Retrieve a single example.
        
        Args:
            idx: Index of example to retrieve
            
        Returns:
            sample: Dictionary containing operands and result
        """
        if self.h1_batch is not None:
            # Return pregenerated example
            return {
                'operands': (self.h1_batch[idx], self.h2_batch[idx], self.h_op_batch[idx]),
                'results': self.results[idx],
                'original_operands': self.operands_list[idx] if self.operands_list else None
            }
        else:
            # Generate on-the-fly
            h1, h2, h_op, result, operands = self.generator.generate_problem(
                self.operation, self.value_range, device=self.device
            )
            return {
                'operands': (h1, h2, h_op),
                'results': result,
                'original_operands': operands
            }


# Utility functions for creating data loaders

def create_metacognition_dataloaders(hidden_dim=256, train_size=10000, val_size=2000,
                                    batch_size=32, pattern_strength=0.5, seed=42,
                                    num_workers=2, device='cpu'):
    """
    Create data loaders for metacognition training.
    
    Args:
        hidden_dim: Dimension of hidden state representations
        train_size: Size of training dataset
        val_size: Size of validation dataset
        batch_size: Batch size for data loaders
        pattern_strength: Intensity of correctness/error patterns
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        device: Device to place tensors on
        
    Returns:
        train_loader: Data loader for training
        val_loader: Data loader for validation
    """
    # Create datasets
    train_dataset = MetacognitionDataset(
        hidden_dim=hidden_dim,
        size=train_size,
        pattern_strength=pattern_strength,
        seed=seed,
        pregenerate=True,
        device=device
    )
    
    val_dataset = MetacognitionDataset(
        hidden_dim=hidden_dim,
        size=val_size,
        pattern_strength=pattern_strength,
        seed=seed + 1,  # Different seed for validation
        pregenerate=True,
        device=device
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def create_bayesian_dataloaders(hidden_dim=256, num_hypotheses=3, evid_strength=0.7,
                               train_size=1000, val_size=200, seq_len=10,
                               batch_size=32, seed=42, num_workers=2, device='cpu'):
    """
    Create data loaders for Bayesian inference training.
    
    Args:
        hidden_dim: Dimension of hidden state representations
        num_hypotheses: Number of hypotheses to distinguish
        evid_strength: Strength of evidence (higher = more decisive)
        train_size: Size of training dataset
        val_size: Size of validation dataset
        seq_len: Length of evidence sequences
        batch_size: Batch size for data loaders
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        device: Device to place tensors on
        
    Returns:
        train_loader: Data loader for training
        val_loader: Data loader for validation
    """
    # Create datasets
    train_dataset = BayesianDataset(
        hidden_dim=hidden_dim,
        num_hypotheses=num_hypotheses,
        evid_strength=evid_strength,
        size=train_size,
        seq_len=seq_len,
        seed=seed,
        pregenerate=True,
        device=device
    )
    
    val_dataset = BayesianDataset(
        hidden_dim=hidden_dim,
        num_hypotheses=num_hypotheses,
        evid_strength=evid_strength,
        size=val_size,
        seq_len=seq_len,
        seed=seed + 1,  # Different seed for validation
        pregenerate=True,
        device=device
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def create_planning_dataloaders(hidden_dim=256, num_steps=5, transform_strength=0.5,
                              train_size=1000, val_size=200, steps=None,
                              batch_size=32, seed=42, num_workers=2, device='cpu'):
    """
    Create data loaders for planning capability training.
    
    Args:
        hidden_dim: Dimension of hidden state representations
        num_steps: Maximum number of reasoning steps
        transform_strength: Intensity of transformations
        train_size: Size of training dataset
        val_size: Size of validation dataset
        steps: Number of reasoning steps per task (or None for max)
        batch_size: Batch size for data loaders
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        device: Device to place tensors on
        
    Returns:
        train_loader: Data loader for training
        val_loader: Data loader for validation
    """
    # Create datasets
    train_dataset = PlanningDataset(
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        transform_strength=transform_strength,
        size=train_size,
        steps=steps,
        seed=seed,
        pregenerate=True,
        device=device
    )
    
    val_dataset = PlanningDataset(
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        transform_strength=transform_strength,
        size=val_size,
        steps=steps,
        seed=seed + 1,  # Different seed for validation
        pregenerate=True,
        device=device
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def create_numerical_dataloaders(hidden_dim=256, operations=None, value_ranges=None,
                               train_size=1000, val_size=200, extra_size=200,
                               operation=None, batch_size=32, seed=42,
                               num_workers=2, device='cpu'):
    """
    Create data loaders for numerical competence training.
    
    Args:
        hidden_dim: Dimension of hidden state representations
        operations: Mathematical operations to generate
        value_ranges: Ranges for operand values
        train_size: Size of training dataset
        val_size: Size of validation dataset
        extra_size: Size of extrapolation dataset
        operation: Specific operation or None for mixed
        batch_size: Batch size for data loaders
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        device: Device to place tensors on
        
    Returns:
        train_loader: Data loader for training
        val_loader: Data loader for validation
        extra_loader: Data loader for extrapolation testing
    """
    # Set default value ranges if not provided
    if value_ranges is None:
        value_ranges = {
            'train': (0, 100),
            'validation': (0, 100),
            'extrapolation': (100, 1000)
        }
    
    # Create datasets
    train_dataset = NumericalDataset(
        hidden_dim=hidden_dim,
        operations=operations,
        value_ranges=value_ranges,
        size=train_size,
        operation=operation,
        value_range=value_ranges['train'],
        seed=seed,
        pregenerate=True,
        device=device
    )
    
    val_dataset = NumericalDataset(
        hidden_dim=hidden_dim,
        operations=operations,
        value_ranges=value_ranges,
        size=val_size,
        operation=operation,
        value_range=value_ranges['validation'],
        seed=seed + 1,  # Different seed for validation
        pregenerate=True,
        device=device
    )
    
    extra_dataset = NumericalDataset(
        hidden_dim=hidden_dim,
        operations=operations,
        value_ranges=value_ranges,
        size=extra_size,
        operation=operation,
        value_range=value_ranges['extrapolation'],
        seed=seed + 2,  # Different seed for extrapolation
        pregenerate=True,
        device=device
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    extra_loader = DataLoader(
        extra_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, extra_loader
