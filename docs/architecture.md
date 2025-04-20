# System Architecture: The Avian Cognitive Framework

## Architectural Overview

The Avian Cognitive Architecture represents a novel synthesis of state-space modeling and extreme quantization, augmented with specialized modules inspired by avian neurobiology. The system integrates a Mamba-SSM backbone with four cognitive modules, all implemented with BitNet 1-bit quantization to achieve extraordinary computational efficiency.

```
┌──────────────────────────────────────────────────────────┐
│                    Avian Cognitive System                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐                                      │
│  │  Metacognition │                                      │
│  │     Module     │◄─────┐                               │
│  └────────────────┘      │                               │
│         ▲                │                               │
│         │                │                               │
│         │                │                               │
│  ┌──────┴───────┐  ┌─────┴──────┐  ┌─────────────────┐  │
│  │              │  │            │  │                 │  │
│  │   Mamba-SSM  │◄─┤  Planning  │◄─┤    Bayesian     │  │
│  │   Backbone   │  │   Module   │  │ Inference Module│  │
│  │              │──►            │  │                 │  │
│  └──────┬───────┘  └────────────┘  └─────────────────┘  │
│         │                                                │
│         │                                                │
│         ▼                                                │
│  ┌────────────────┐                                      │
│  │   Numerical    │                                      │
│  │ Competence     │                                      │
│  │    Module      │                                      │
│  └────────────────┘                                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Mamba-SSM Backbone

The foundation of our architecture is the Mamba Selective State Space Model, which replaces attention mechanisms with a more efficient recurrent approach to sequence modeling.

#### Key Properties:
- **Linear-Time Complexity**: O(n) scaling with sequence length versus O(n²) for attention-based models
- **Selective State Propagation**: Dynamically modulates information flow through the state, preserving relevant context
- **Structured Sequence Processing**: Combines SSM layers for long-range dependencies with local convolution for token-level processing
- **BitNet Quantization**: All weight matrices implemented with 1-bit representation (±1 values)

#### Implementation Variants:
- **Mamba-mini**: ~60M parameters (12 layers, 512 hidden size)
- **Mamba-small**: ~130M parameters (24 layers, 768 hidden size)
- **Mamba-medium**: ~370M parameters (48 layers, 1024 hidden size)

### 2. Metacognition Module

A specialized neural circuit that monitors the model's own certainty and knowledge state, analogous to the certainty-encoding neurons observed in the corvid pallium.

#### Architecture:
```python
class MetacognitionModule:
    """
    Neural circuit for self-monitoring and uncertainty estimation.
    
    Implements calibrated confidence prediction based on internal model state.
    """
    def __init__(self, hidden_dim, bit_linear=True):
        self.confidence_network = BitLinear(hidden_dim, 1) if bit_linear else nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, h_final):
        """
        Produces a calibrated confidence score from the model's final hidden state.
        
        Args:
            h_final: The final hidden state representation from the Mamba backbone
            
        Returns:
            confidence: A calibrated probability (0-1) representing model certainty
        """
        logit = self.confidence_network(h_final)
        confidence = self.sigmoid(logit)
        return confidence
```

#### Training Objective:
The metacognition module is trained with a calibration loss (binary cross-entropy between predicted confidence and actual correctness), ensuring that confidence scores align with empirical accuracy.

### 3. Bayesian Inference Module

A slow-timescale recurrent network that maintains and updates belief states as new evidence arrives, implementing probabilistic reasoning capabilities similar to those observed in statistical crows.

#### Architecture:
```python
class BayesianModule:
    """
    Belief state tracker implementing approximate Bayesian inference.
    
    Maintains a running belief representation that gets updated with each
    new token, modeling sequential probabilistic inference.
    """
    def __init__(self, hidden_dim, belief_dim, bit_linear=True):
        # Belief state update components
        self.prior_transform = BitLinear(belief_dim, belief_dim) if bit_linear else nn.Linear(belief_dim, belief_dim)
        self.evidence_encoder = BitLinear(hidden_dim, belief_dim) if bit_linear else nn.Linear(hidden_dim, belief_dim)
        self.posterior_gate = BitLinear(belief_dim * 2, belief_dim) if bit_linear else nn.Linear(belief_dim * 2, belief_dim)
        self.sigmoid = nn.Sigmoid()
        
    def update_belief(self, b_prev, h_t):
        """
        Updates belief state using new evidence (Bayes-like update).
        
        Args:
            b_prev: Previous belief state (prior)
            h_t: Current hidden state (evidence)
            
        Returns:
            b_t: Updated belief state (posterior)
        """
        # Transform prior belief
        b_prior = self.prior_transform(b_prev)
        
        # Encode new evidence
        e_t = self.evidence_encoder(h_t)
        
        # Compute update gate
        update_input = torch.cat([b_prior, e_t], dim=-1)
        update_gate = self.sigmoid(self.posterior_gate(update_input))
        
        # Gated update (approximating Bayes rule)
        b_t = (1 - update_gate) * b_prior + update_gate * e_t
        
        return b_t
```

#### Integration with Mamba:
The Bayesian module runs in parallel with the Mamba backbone, with its belief state influencing the Mamba hidden state at each step through gated interaction.

### 4. Planning and Tool-Use Module

A specialized reasoning controller that can generate multi-step plans and simulate outcomes before producing a final answer, mimicking the planning capabilities observed in tool-using corvids.

#### Architecture:
```python
class PlanningModule:
    """
    Internal reasoning controller for multi-step planning.
    
    Generates a sequence of reasoning steps and potential action plans
    before producing a final solution.
    """
    def __init__(self, hidden_dim, plan_dim, num_steps=5, bit_linear=True):
        self.hidden_dim = hidden_dim
        self.plan_dim = plan_dim
        self.num_steps = num_steps
        
        # Plan generation components
        self.init_transform = BitLinear(hidden_dim, plan_dim) if bit_linear else nn.Linear(hidden_dim, plan_dim)
        self.step_rnn = BitGRUCell(plan_dim, plan_dim) if bit_linear else nn.GRUCell(plan_dim, plan_dim)
        
        # Plan aggregation
        self.plan_aggregator = BitLinear(plan_dim * num_steps, plan_dim) if bit_linear else nn.Linear(plan_dim * num_steps, plan_dim)
        
    def generate_plan(self, h_context):
        """
        Generates a multi-step reasoning plan from the context representation.
        
        Args:
            h_context: Hidden state representation of the problem context
            
        Returns:
            plan_final: Final aggregated plan representation
            plan_steps: Individual plan step representations
        """
        # Initialize plan state
        p_t = self.init_transform(h_context)
        
        # Generate reasoning steps
        plan_steps = []
        for _ in range(self.num_steps):
            p_t = self.step_rnn(p_t)
            plan_steps.append(p_t)
            
        # Aggregate plan
        plan_concat = torch.cat(plan_steps, dim=-1)
        plan_final = self.plan_aggregator(plan_concat)
        
        return plan_final, plan_steps
```

#### Integration with Mamba:
The planning module activates after the input has been encoded, generating a plan representation that influences the final output through concatenation or additive interaction.

### 5. Numerical Competence Module

A specialized neural arithmetic unit that enables precise numerical processing and calculation, inspired by the counting abilities observed in corvids.

#### Architecture:
```python
class NumericalModule:
    """
    Neural arithmetic processor for exact numerical computation.
    
    Implements specialized neural circuits for arithmetic operations
    with generalization beyond the training distribution.
    """
    def __init__(self, hidden_dim, numeric_dim, bit_linear=True):
        self.hidden_to_numeric = BitLinear(hidden_dim, numeric_dim) if bit_linear else nn.Linear(hidden_dim, numeric_dim)
        
        # Neural Arithmetic Logic Units for different operations
        self.add_nalu = NALULayer(numeric_dim, numeric_dim, operation='add')
        self.subtract_nalu = NALULayer(numeric_dim, numeric_dim, operation='subtract')
        self.multiply_nalu = NALULayer(numeric_dim, numeric_dim, operation='multiply')
        
        # Operation selector
        self.op_selector = BitLinear(hidden_dim, 3) if bit_linear else nn.Linear(hidden_dim, 3)
        self.softmax = nn.Softmax(dim=-1)
        
        # Result transformation
        self.numeric_to_hidden = BitLinear(numeric_dim, hidden_dim) if bit_linear else nn.Linear(numeric_dim, hidden_dim)
        
    def forward(self, h_operands, h_op):
        """
        Performs a numeric operation on encoded operands.
        
        Args:
            h_operands: Hidden states corresponding to numeric operands
            h_op: Hidden state corresponding to operation token
            
        Returns:
            h_result: Hidden state encoding of the result
        """
        # Decode operation
        op_logits = self.op_selector(h_op)
        op_weights = self.softmax(op_logits)
        
        # Convert to numeric representation
        numeric_operands = [self.hidden_to_numeric(h) for h in h_operands]
        
        # Apply operations
        add_result = self.add_nalu(*numeric_operands)
        sub_result = self.subtract_nalu(*numeric_operands)
        mul_result = self.multiply_nalu(*numeric_operands)
        
        # Weighted combination based on detected operation
        numeric_result = (
            op_weights[0] * add_result +
            op_weights[1] * sub_result +
            op_weights[2] * mul_result
        )
        
        # Convert back to hidden representation
        h_result = self.numeric_to_hidden(numeric_result)
        
        return h_result
```

#### Integration with Mamba:
The numerical module activates when the model encounters arithmetic expressions, replacing the default processing with exact calculation and injecting the result back into the hidden state stream.

## System Integration

The complete system operates through coordinated interaction between these components:

1. **Input Processing**: The Mamba backbone processes the input sequence token by token, with the Bayesian module updating its belief state in parallel.

2. **Reasoning and Planning**: Upon reaching the end of the input, the planning module activates to generate a multi-step reasoning plan.

3. **Numerical Processing**: Throughout processing, when arithmetic expressions are encountered, the numerical module is triggered to perform exact calculations.

4. **Output Generation**: The final output is produced based on the Mamba hidden state, influenced by the planning module's reasoning steps.

5. **Confidence Estimation**: The metacognition module examines the final state and produces a calibrated confidence score for the answer.

All components are implemented with BitNet 1-bit quantization, enabling the entire system to operate with extraordinary memory and computational efficiency.

## Memory and Computational Efficiency

The architecture achieves remarkable efficiency through several mechanisms:

- **BitNet Quantization**: 1-bit weight representation reduces memory footprint by 32× compared to FP32 models.

- **Linear-Time Complexity**: Mamba's O(n) scaling with sequence length enables processing of long contexts with minimal computational growth.

- **Selective State Propagation**: Information-dependent gating in the SSM ensures only relevant context is preserved, reducing redundant computation.

- **Modular Activation**: Cognitive modules are triggered only when relevant, reducing baseline computational overhead.

With these efficiency mechanisms, the system can run on minimal hardware configurations:

- **Mamba-mini (60M parameters)**: Runs on CPU or low-end GPU (4GB VRAM)
- **Mamba-small (130M parameters)**: Fits on mid-range GPU (8GB VRAM)
- **Mamba-medium (370M parameters)**: Requires high-end consumer GPU (12GB VRAM)