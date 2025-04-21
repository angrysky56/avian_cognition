"""
Numerical Competence Module

This module implements specialized neural circuits for precise numerical operations
and quantitative reasoning, inspired by the counting and arithmetic abilities
observed in corvids and parrots.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.bitnet import BitLinear, NALULayer


class NumericalModule(nn.Module):
    """
    Neural arithmetic processor for quantitative cognition.
    
    Implements specialized neural circuits for number representation and
    numerical operations, enabling precise calculation and generalization
    beyond the training distribution.
    
    The architecture transcends pattern-matching approximations through
    explicitly encoded arithmetic operations, mirroring how corvids demonstrate
    exact counting rather than magnitude estimation.
    
    Attributes:
        hidden_dim: Dimension of hidden representation
        num_dim: Dimension of numerical representation
        num_extractor: Extracts numeric values from hidden states
        num_encoder: Encodes numeric values into hidden states
        arithmetic_units: Neural arithmetic units for various operations
        op_classifier: Classifies operation type from context
    """
    
    def __init__(self, hidden_dim, num_dim=32, bit_linear=True):
        """
        Initialize numerical module with BitLinear quantization.
        
        Args:
            hidden_dim: Dimension of the hidden state
            num_dim: Dimension of numerical representation
            bit_linear: Whether to use BitLinear quantization
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_dim = num_dim
        
        # Number extraction (hidden state → numeric representation)
        self.num_extractor = nn.Sequential(
            BitLinear(hidden_dim, num_dim * 2) if bit_linear else nn.Linear(hidden_dim, num_dim * 2),
            nn.GELU(),
            BitLinear(num_dim * 2, num_dim) if bit_linear else nn.Linear(num_dim * 2, num_dim)
        )
        
        # Number encoding (numeric representation → hidden state)
        self.num_encoder = nn.Sequential(
            BitLinear(num_dim, num_dim * 2) if bit_linear else nn.Linear(num_dim, num_dim * 2),
            nn.GELU(),
            BitLinear(num_dim * 2, hidden_dim) if bit_linear else nn.Linear(num_dim * 2, hidden_dim)
        )
        
        # Neural arithmetic units
        self.arithmetic_units = nn.ModuleDict({
            'add': NALULayer(num_dim * 2, num_dim),
            'subtract': NALULayer(num_dim * 2, num_dim),
            'multiply': NALULayer(num_dim * 2, num_dim),
            'divide': NALULayer(num_dim * 2, num_dim),
            'greater': BitLinear(num_dim * 2, 1) if bit_linear else nn.Linear(num_dim * 2, 1),
            'equal': BitLinear(num_dim * 2, 1) if bit_linear else nn.Linear(num_dim * 2, 1)
        })
        
        # Operation classifier
        self.op_classifier = nn.Sequential(
            BitLinear(hidden_dim, hidden_dim // 2) if bit_linear else nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            BitLinear(hidden_dim // 2, len(self.arithmetic_units)) if bit_linear else nn.Linear(hidden_dim // 2, len(self.arithmetic_units))
        )
        
    def extract_number(self, hidden_state):
        """
        Extracts numerical value from hidden state representation.
        
        Args:
            hidden_state: Hidden state representation
                         [batch_size, hidden_dim]
            
        Returns:
            num_representation: Numerical representation
                               [batch_size, num_dim]
        """
        return self.num_extractor(hidden_state)
    
    def encode_number(self, num_representation):
        """
        Encodes numerical representation into hidden state.
        
        Args:
            num_representation: Numerical representation
                               [batch_size, num_dim]
            
        Returns:
            hidden_state: Hidden state representation
                         [batch_size, hidden_dim]
        """
        return self.num_encoder(num_representation)
    
    def classify_operation(self, op_hidden):
        """
        Classifies arithmetic operation from context.
        
        Args:
            op_hidden: Hidden state representing operation
                      [batch_size, hidden_dim]
            
        Returns:
            op_logits: Operation classification logits
                      [batch_size, num_operations]
            op_weights: Softmax weights for operations
                       [batch_size, num_operations]
        """
        op_logits = self.op_classifier(op_hidden)
        op_weights = F.softmax(op_logits, dim=-1)
        return op_logits, op_weights
    
    def forward(self, x1, x2, op=None):
        """
        Performs arithmetic operation on two numeric inputs.
        
        Args:
            x1: Hidden state representing first operand
                [batch_size, hidden_dim]
            x2: Hidden state representing second operand
                [batch_size, hidden_dim]
            op: Optional hidden state representing operation
                [batch_size, hidden_dim]
            
        Returns:
            result_hidden: Hidden state representation of result
                          [batch_size, hidden_dim]
            op_weights: Weights for different operations
                       [batch_size, num_operations]
        """
        # Extract numeric values from hidden states
        n1 = self.extract_number(x1)
        n2 = self.extract_number(x2)
        
        # Concatenate operands for binary operations
        cat_operands = torch.cat([n1, n2], dim=-1)
        
        # Determine operation probabilities
        if op is not None:
            _, op_weights = self.classify_operation(op)
        else:
            # Default to uniform weights if no operation specified
            op_weights = torch.ones(x1.shape[0], len(self.arithmetic_units), device=x1.device)
            op_weights = op_weights / len(self.arithmetic_units)
        
        # Perform all operations
        results = {}
        for i, (op_name, op_unit) in enumerate(self.arithmetic_units.items()):
            if op_name in ['greater', 'equal']:
                # Comparison operations output a sigmoid value
                results[op_name] = torch.sigmoid(op_unit(cat_operands))
            else:
                # Arithmetic operations output a numeric representation
                results[op_name] = op_unit(cat_operands)
        
        # Combine results based on operation weights
        weighted_results = []
        for i, (op_name, result) in enumerate(results.items()):
            if op_name in ['greater', 'equal']:
                # Expand comparison result to match numeric dimension
                expanded_result = result.expand(-1, self.num_dim)
                weighted_results.append(op_weights[:, i:i+1] * expanded_result)
            else:
                weighted_results.append(op_weights[:, i:i+1] * result)
        
        # Sum weighted results
        numeric_result = sum(weighted_results)
        
        # Encode result back to hidden representation
        result_hidden = self.encode_number(numeric_result)
        
        return result_hidden, op_weights


class CountingModule(nn.Module):
    """
    Specialized counting processor for accumulating quantities.
    
    Implements neural mechanisms for precise counting and tracking
    quantities across a sequence, inspired by corvids' ability to
    match requested call numbers and track objects.
    
    Attributes:
        hidden_dim: Dimension of hidden state
        count_dim: Dimension of count representation
        count_detector: Detects countable entities in input
        count_incrementer: Increments count state
        count_encoder: Encodes count into hidden representation
    """
    
    def __init__(self, hidden_dim, count_dim=16, bit_linear=True):
        """
        Initialize counting module with BitLinear quantization.
        
        Args:
            hidden_dim: Dimension of hidden state
            count_dim: Dimension of count representation
            bit_linear: Whether to use BitLinear quantization
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.count_dim = count_dim
        
        # Count detector (determines if input should be counted)
        self.count_detector = nn.Sequential(
            BitLinear(hidden_dim, hidden_dim // 2) if bit_linear else nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            BitLinear(hidden_dim // 2, 1) if bit_linear else nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Count incrementer (NALU-based to ensure exact +1 operations)
        self.count_incrementer = nn.Sequential(
            BitLinear(count_dim, count_dim) if bit_linear else nn.Linear(count_dim, count_dim),
            nn.GELU(),
            NALULayer(count_dim, count_dim)
        )
        
        # Count encoder (count → hidden representation)
        self.count_encoder = nn.Sequential(
            BitLinear(count_dim, hidden_dim // 2) if bit_linear else nn.Linear(count_dim, hidden_dim // 2),
            nn.GELU(),
            BitLinear(hidden_dim // 2, hidden_dim) if bit_linear else nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Count initializer (for zero state)
        self.register_buffer('zero_count', torch.zeros(1, count_dim))
        
    def detect_countable(self, hidden_state):
        """
        Determines if input contains a countable entity.
        
        Args:
            hidden_state: Hidden state representation
                         [batch_size, hidden_dim]
            
        Returns:
            count_signal: Signal indicating countable entity (0-1)
                         [batch_size, 1]
        """
        return self.count_detector(hidden_state)
    
    def increment_count(self, count_state, increment_signal):
        """
        Increments count state based on detection signal.
        
        Args:
            count_state: Current count state representation
                        [batch_size, count_dim]
            increment_signal: Signal for incrementation (0-1)
                            [batch_size, 1]
            
        Returns:
            new_count: Updated count state
                      [batch_size, count_dim]
        """
        # Generate increment vector (learned +1 operation)
        increment = self.count_incrementer(count_state)
        
        # Apply increment proportionally to signal
        new_count = count_state + increment_signal * (increment - count_state)
        
        return new_count
    
    def encode_count(self, count_state):
        """
        Encodes count state into hidden representation.
        
        Args:
            count_state: Count state representation
                        [batch_size, count_dim]
            
        Returns:
            count_hidden: Hidden representation of count
                         [batch_size, hidden_dim]
        """
        return self.count_encoder(count_state)
    
    def forward(self, hidden_sequence, initial_count=None):
        """
        Counts entities across a sequence of hidden states.
        
        Args:
            hidden_sequence: Sequence of hidden states
                           [seq_len, batch_size, hidden_dim]
            initial_count: Optional initial count state
                          [batch_size, count_dim]
            
        Returns:
            final_count_hidden: Hidden representation of final count
                               [batch_size, hidden_dim]
            count_trajectory: Count states across sequence
                             [seq_len, batch_size, count_dim]
        """
        seq_len, batch_size, _ = hidden_sequence.shape
        device = hidden_sequence.device
        
        # Initialize count state
        if initial_count is None:
            count_state = self.zero_count.expand(batch_size, -1).to(device)
        else:
            count_state = initial_count
            
        # Process sequence
        count_trajectory = []
        for t in range(seq_len):
            # Detect countable entities
            count_signal = self.detect_countable(hidden_sequence[t])
            
            # Update count
            count_state = self.increment_count(count_state, count_signal)
            count_trajectory.append(count_state)
            
        # Stack trajectory
        count_trajectory = torch.stack(count_trajectory)
        
        # Encode final count
        final_count_hidden = self.encode_count(count_state)
        
        return final_count_hidden, count_trajectory


class NumberMapping(nn.Module):
    """
    Bidirectional mapping between symbolic and numeric representations.
    
    Implements neural mechanisms for translating between token-level
    representations (e.g., "seven") and quantitative representations,
    inspired by birds' ability to associate symbols with quantities.
    
    Attributes:
        vocab_size: Size of token vocabulary
        num_dim: Dimension of numerical representation
        token_to_num: Maps token embeddings to numerical representations
        num_to_logits: Maps numerical representations to token logits
    """
    
    def __init__(self, vocab_size, hidden_dim, num_dim=16, max_num=100, bit_linear=True):
        """
        Initialize number mapping module with BitLinear quantization.
        
        Args:
            vocab_size: Size of token vocabulary
            hidden_dim: Dimension of hidden state
            num_dim: Dimension of numerical representation
            max_num: Maximum representable number
            bit_linear: Whether to use BitLinear quantization
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_dim = num_dim
        self.max_num = max_num
        
        # Token to number mapping
        self.token_to_num = nn.Sequential(
            BitLinear(hidden_dim, hidden_dim // 2) if bit_linear else nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            BitLinear(hidden_dim // 2, num_dim) if bit_linear else nn.Linear(hidden_dim // 2, num_dim)
        )
        
        # Number to token mapping
        self.num_to_logits = nn.Sequential(
            BitLinear(num_dim, hidden_dim // 2) if bit_linear else nn.Linear(num_dim, hidden_dim // 2),
            nn.GELU(),
            BitLinear(hidden_dim // 2, vocab_size) if bit_linear else nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        # Initialize number prototypes for calibration
        self.register_buffer('number_prototypes', self._initialize_prototypes())
        
    def _initialize_prototypes(self):
        """
        Initializes prototype vectors for specific numbers.
        
        Returns:
            prototypes: Tensor of prototype vectors for numbers 0 to max_num
                       [max_num+1, num_dim]
        """
        prototypes = torch.zeros(self.max_num + 1, self.num_dim)
        
        # Generate structured encoding for each number
        for i in range(self.max_num + 1):
            # Convert number to binary representation
            binary = [(i >> bit) & 1 for bit in range(min(self.num_dim, 10))]
            
            # Fill remaining dimensions with structured pattern
            for j in range(min(self.num_dim, 10), self.num_dim):
                # Use periodic functions to create structure
                binary.append(math.sin(math.pi * i / (2 ** (j % 5))))
                
            # Convert to tensor
            binary_tensor = torch.tensor(binary)
            
            # Normalize
            if binary_tensor.norm() > 0:
                binary_tensor = binary_tensor / binary_tensor.norm()
                
            prototypes[i, :len(binary)] = binary_tensor
            
        return prototypes
        
    def token_to_number(self, token_hidden):
        """
        Maps token hidden state to numerical representation.
        
        Args:
            token_hidden: Hidden state of token
                         [batch_size, hidden_dim]
            
        Returns:
            num_representation: Numerical representation
                               [batch_size, num_dim]
            estimated_value: Estimated numeric value
                            [batch_size]
        """
        # Map to numerical representation
        num_representation = self.token_to_num(token_hidden)
        
        # Find closest prototype to estimate value
        similarities = F.cosine_similarity(
            num_representation.unsqueeze(1),  # [batch_size, 1, num_dim]
            self.number_prototypes.unsqueeze(0),  # [1, max_num+1, num_dim]
            dim=2
        )
        
        # Get most similar prototype
        estimated_index = torch.argmax(similarities, dim=1)
        
        return num_representation, estimated_index
        
    def number_to_token(self, num_representation):
        """
        Maps numerical representation to token logits.
        
        Args:
            num_representation: Numerical representation
                               [batch_size, num_dim]
            
        Returns:
            token_logits: Logits over vocabulary
                         [batch_size, vocab_size]
        """
        return self.num_to_logits(num_representation)
    
    def calibrate_prototypes(self, token_hiddens, values):
        """
        Calibrates number prototypes based on known examples.
        
        Args:
            token_hiddens: Hidden states of number tokens
                          [num_examples, hidden_dim]
            values: Corresponding numeric values
                   [num_examples]
            
        Returns:
            loss: Calibration loss
        """
        # Get numerical representations
        num_representations = self.token_to_num(token_hiddens)
        
        # Update prototypes for seen values
        for i, value in enumerate(values):
            if value <= self.max_num:
                # Update prototype with moving average
                self.number_prototypes[value] = 0.9 * self.number_prototypes[value] + 0.1 * num_representations[i]
                # Normalize
                self.number_prototypes[value] = self.number_prototypes[value] / self.number_prototypes[value].norm()
                
        # Compute calibration loss (cosine similarity)
        loss = 0
        for i, value in enumerate(values):
            if value <= self.max_num:
                similarity = F.cosine_similarity(
                    num_representations[i:i+1],
                    self.number_prototypes[value:value+1],
                    dim=1
                )
                loss += 1 - similarity
                
        return loss


def numerical_extrapolation_test(numerical_module, value_range, operation='add', device='cpu'):
    """
    Tests the numerical module's ability to extrapolate beyond training range.
    
    Args:
        numerical_module: Numerical module to test
        value_range: Range of values to test (e.g., (0, 1000))
        operation: Arithmetic operation to test
        device: Device to run test on
        
    Returns:
        accuracy: Accuracy on test cases
        error_stats: Statistics on errors
    """
    min_val, max_val = value_range
    test_size = 100
    
    # Generate test cases
    test_cases = []
    for _ in range(test_size):
        a = torch.randint(min_val, max_val, (1,)).item()
        b = torch.randint(min_val, max_val, (1,)).item()
        
        if operation == 'add':
            result = a + b
        elif operation == 'subtract':
            result = a - b
        elif operation == 'multiply':
            result = a * b
        elif operation == 'divide':
            if b == 0:
                b = 1  # Avoid division by zero
            result = a / b
            
        test_cases.append((a, b, result))
    
    # Create hidden state representations
    # This is a simplified version - in practice, you'd use the actual model's embedding
    def create_hidden(value, dim=numerical_module.hidden_dim):
        h = torch.zeros(1, dim, device=device)
        h[0, 0] = value / max_val  # Simple numerical encoding
        return h
    
    correct = 0
    errors = []
    
    # Run test cases
    for a, b, expected in test_cases:
        # Create hidden states
        h_a = create_hidden(a)
        h_b = create_hidden(b)
        
        # Create operation hidden state
        h_op = torch.zeros(1, numerical_module.hidden_dim, device=device)
        op_idx = ['add', 'subtract', 'multiply', 'divide'].index(operation)
        h_op[0, op_idx] = 1.0
        
        # Run through module
        result_hidden, _ = numerical_module(h_a, h_b, h_op)
        
        # Extract result
        # Again, simplified - in practice you'd use the model's number extraction mechanism
        predicted = result_hidden[0, 0].item() * max_val
        
        # Check correctness (with tolerance)
        tolerance = max(0.05 * abs(expected), 1.0)  # 5% or 1, whichever is larger
        if abs(predicted - expected) <= tolerance:
            correct += 1
        else:
            errors.append(abs(predicted - expected))
    
    # Calculate statistics
    accuracy = correct / test_size
    error_stats = {
        'mean': sum(errors) / len(errors) if errors else 0,
        'median': sorted(errors)[len(errors) // 2] if errors else 0,
        'max': max(errors) if errors else 0
    }
    
    return accuracy, error_stats
