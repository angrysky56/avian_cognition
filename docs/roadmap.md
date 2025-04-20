# Implementation Roadmap: Avian Cognitive Architecture

This document outlines the phased approach to implementing the Avian Cognitive Architecture, a synthesis of Mamba-SSM and BitNet with avian-inspired cognitive modules.

## Phase 1: Core Mamba Model Setup

**Objective**: Establish the backbone model—a Mamba state-space sequence model—and verify its performance and efficiency before integrating cognitive modules.

### Stage 1.1: Environment Setup

```python
# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install transformers accelerate bitsandbytes
pip install pytorch-lightning
```

### Stage 1.2: Model Initialization

```python
class MambaConfig:
    """
    Configuration for the Mamba-SSM backbone.
    
    Defines the architectural hyperparameters for the selective state-space model.
    """
    def __init__(
        self,
        vocab_size=50277,
        d_model=768,
        n_layer=24,
        ssm_state_size=16,
        ssm_dt_min=0.001,
        ssm_dt_max=0.1,
        ssm_d_conv=4,
        ssm_expand=2,
        dt_init="random",
        dt_scale=1.0,
        pad_vocab_size_multiple=8,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.ssm_state_size = ssm_state_size
        self.ssm_dt_min = ssm_dt_min
        self.ssm_dt_max = ssm_dt_max
        self.ssm_d_conv = ssm_d_conv
        self.ssm_expand = ssm_expand
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
```

### Stage 1.3: Loading Pretrained Weights

```python
# Load the pretrained model from Hugging Face
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "state-spaces/mamba-370m",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

### Stage 1.4: BitNet Quantization

```python
class BitLinear(nn.Module):
    """
    Linear layer with 1-bit weight quantization, following BitNet approach.
    
    Maintains full-precision weights during training but uses binary (+1/-1)
    weights during forward pass.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Full-precision weights for gradient updates
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('scale', torch.ones(1))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input):
        # Binary quantization
        binary_weight = torch.sign(self.weight)
        
        # Scale factor (to preserve output magnitude)
        self.scale = self.weight.abs().mean()
        
        # Forward pass with binary weights
        return F.linear(input, binary_weight * self.scale, self.bias)
```

### Stage 1.5: Validation of Baseline Performance

```python
# Evaluate perplexity on validation set
def evaluate_perplexity(model, val_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input_ids'].to(model.device)
            targets = batch['labels'].to(model.device)
            
            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            
            total_loss += loss.item() * inputs.size(1)
            total_tokens += inputs.size(1)
    
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity
```

## Phase 2: Cognitive Module Integration

**Objective**: Incrementally implement the four avian-inspired cognitive modules on top of the Mamba-BitNet core.

### Phase 2a: Metacognition Module Integration

```python
class MetacognitionModule(nn.Module):
    """
    Neural circuit for calibrated uncertainty estimation.
    
    Predicts the model's confidence in its outputs based on internal state.
    """
    def __init__(self, hidden_dim, bit_linear=True):
        super().__init__()
        
        # Confidence prediction network
        self.hidden_transform = BitLinear(hidden_dim, hidden_dim // 2) if bit_linear else nn.Linear(hidden_dim, hidden_dim // 2)
        self.confidence_head = BitLinear(hidden_dim // 2, 1) if bit_linear else nn.Linear(hidden_dim // 2, 1)
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, hidden_state):
        """
        Produces a calibrated confidence estimate from model's hidden state.
        
        Args:
            hidden_state: The final hidden state from the Mamba backbone
            
        Returns:
            confidence: A scalar confidence score (0-1)
        """
        x = self.activation(self.hidden_transform(hidden_state))
        logit = self.confidence_head(x)
        confidence = self.sigmoid(logit)
        
        return confidence
        
# Training objective for metacognition
def confidence_calibration_loss(confidence, correctness):
    """
    Binary cross-entropy loss for calibrating confidence predictions.
    
    Args:
        confidence: Predicted confidence scores (0-1)
        correctness: Binary labels indicating whether predictions were correct (1) or not (0)
        
    Returns:
        loss: Calibration loss (lower means better calibration)
    """
    return F.binary_cross_entropy(confidence, correctness)
```

### Phase 2b: Bayesian Inference Module Integration

```python
class BayesianInferenceModule(nn.Module):
    """
    Belief state tracker implementing approximate Bayesian inference.
    
    Maintains and updates probability distributions over latent variables
    as new evidence arrives.
    """
    def __init__(self, hidden_dim, belief_dim, bit_linear=True):
        super().__init__()
        
        # Evidence encoding
        self.evidence_encoder = BitLinear(hidden_dim, belief_dim) if bit_linear else nn.Linear(hidden_dim, belief_dim)
        
        # Belief update mechanism
        self.update_gate = nn.Sequential(
            BitLinear(hidden_dim + belief_dim, belief_dim) if bit_linear else nn.Linear(hidden_dim + belief_dim, belief_dim),
            nn.Sigmoid()
        )
        
        # Prior transform
        self.prior_transform = BitLinear(belief_dim, belief_dim) if bit_linear else nn.Linear(belief_dim, belief_dim)
        
        # Output projection
        self.output_proj = BitLinear(belief_dim, hidden_dim) if bit_linear else nn.Linear(belief_dim, hidden_dim)
        
    def forward(self, hidden_state, belief_state=None):
        """
        Updates belief state based on new evidence.
        
        Args:
            hidden_state: Current token's hidden state representation
            belief_state: Previous belief state (or None for initial state)
            
        Returns:
            new_belief: Updated belief state
            belief_embedding: Projection of belief state into hidden dimension
        """
        batch_size = hidden_state.shape[0]
        
        # Initialize belief state if None
        if belief_state is None:
            belief_state = torch.zeros(batch_size, self.belief_dim, device=hidden_state.device)
        
        # Transform prior belief
        prior_belief = self.prior_transform(belief_state)
        
        # Encode current evidence
        evidence = self.evidence_encoder(hidden_state)
        
        # Compute update gate
        gate_input = torch.cat([hidden_state, prior_belief], dim=-1)
        update_gate = self.update_gate(gate_input)
        
        # Update belief (approximating Bayes rule)
        new_belief = update_gate * evidence + (1 - update_gate) * prior_belief
        
        # Project to hidden dimension for integration with main model
        belief_embedding = self.output_proj(new_belief)
        
        return new_belief, belief_embedding
```

### Phase 2c: Planning/Tool-Use Module Integration

```python
class PlanningModule(nn.Module):
    """
    Multi-step reasoning controller for plan generation.
    
    Produces a structured reasoning plan before final answer generation.
    """
    def __init__(self, hidden_dim, plan_steps=5, plan_dim=None, bit_linear=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.plan_steps = plan_steps
        self.plan_dim = plan_dim if plan_dim is not None else hidden_dim
        
        # Context comprehension
        self.context_encoder = BitLinear(hidden_dim, self.plan_dim) if bit_linear else nn.Linear(hidden_dim, self.plan_dim)
        
        # Planning recurrence
        self.plan_cell = BitLinearGRUCell(self.plan_dim, self.plan_dim) if bit_linear else nn.GRUCell(self.plan_dim, self.plan_dim)
        
        # Plan aggregation
        self.plan_aggregator = BitLinear(self.plan_dim * plan_steps, self.plan_dim) if bit_linear else nn.Linear(self.plan_dim * plan_steps, self.plan_dim)
        
        # Final projection
        self.output_proj = BitLinear(self.plan_dim, hidden_dim) if bit_linear else nn.Linear(self.plan_dim, hidden_dim)
        
    def forward(self, context_state):
        """
        Generates a multi-step reasoning plan from context representation.
        
        Args:
            context_state: Hidden state representing problem context
            
        Returns:
            plan_embedding: Final aggregated plan representation
            plan_steps: List of individual reasoning steps
        """
        # Encode context
        plan_state = self.context_encoder(context_state)
        
        # Generate reasoning steps
        step_states = []
        for _ in range(self.plan_steps):
            plan_state = self.plan_cell(plan_state)
            step_states.append(plan_state)
            
        # Aggregate plan
        plan_concat = torch.cat(step_states, dim=-1)
        aggregated_plan = self.plan_aggregator(plan_concat)
        
        # Project to hidden dimension
        plan_embedding = self.output_proj(aggregated_plan)
        
        return plan_embedding, step_states
```

### Phase 2d: Numerical Competence Module Integration

```python
class NALULayer(nn.Module):
    """
    Neural Arithmetic Logic Unit for precise numerical operations.
    
    Implements a differentiable module that can learn to perform exact
    arithmetic operations and generalize beyond the training range.
    """
    def __init__(self, in_features, out_features, eps=1e-7):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        # Weight matrices for gating between add and multiply paths
        self.G = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.M = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.G, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))
        
    def forward(self, x):
        # Gate for add/multiply operation selection
        g = torch.sigmoid(F.linear(x, self.G))
        
        # Addition sub-operation
        a = F.linear(x, self.W)
        
        # Multiplication sub-operation (in log space)
        m = torch.exp(F.linear(torch.log(torch.abs(x) + self.eps), self.M))
        
        # Combine operations
        y = g * a + (1 - g) * m
        
        return y
        
class NumericalModule(nn.Module):
    """
    Specialized module for arithmetic operations and numerical reasoning.
    
    Extracts numeric values from hidden states and performs precise calculations.
    """
    def __init__(self, hidden_dim, num_dim=32, bit_linear=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_dim = num_dim
        
        # Number extraction
        self.num_extractor = BitLinear(hidden_dim, num_dim) if bit_linear else nn.Linear(hidden_dim, num_dim)
        
        # Arithmetic operations
        self.add_nalu = NALULayer(num_dim * 2, num_dim)
        self.sub_nalu = NALULayer(num_dim * 2, num_dim)
        self.mul_nalu = NALULayer(num_dim * 2, num_dim)
        
        # Operation selection
        self.op_classifier = BitLinear(hidden_dim, 3) if bit_linear else nn.Linear(hidden_dim, 3)
        
        # Result encoding
        self.result_encoder = BitLinear(num_dim, hidden_dim) if bit_linear else nn.Linear(num_dim, hidden_dim)
        
    def forward(self, x1, x2, op=None):
        """
        Performs arithmetic operation on two numeric inputs.
        
        Args:
            x1: Hidden state representing first operand
            x2: Hidden state representing second operand
            op: Optional hidden state representing operation (if None, infer from inputs)
            
        Returns:
            result_embedding: Hidden state representation of calculation result
        """
        # Extract numeric values
        n1 = self.num_extractor(x1)
        n2 = self.num_extractor(x2)
        
        # Concatenate operands
        cat_operands = torch.cat([n1, n2], dim=-1)
        
        # Perform all operations
        add_result = self.add_nalu(cat_operands)
        sub_result = self.sub_nalu(cat_operands)
        mul_result = self.mul_nalu(cat_operands)
        
        if op is not None:
            # Classify operation type
            op_logits = self.op_classifier(op)
            op_weights = F.softmax(op_logits, dim=-1)
            
            # Weight results by operation probabilities
            result = (
                op_weights[:, 0:1] * add_result +
                op_weights[:, 1:2] * sub_result +
                op_weights[:, 2:3] * mul_result
            )
        else:
            # Default to addition if no operation specified
            result = add_result
            
        # Encode result back to hidden dimension
        result_embedding = self.result_encoder(result)
        
        return result_embedding
```

## Phase 3: BitNet Quantization

**Objective**: Ensure the entire system operates with 1-bit weight representation.

### Stage 3.1: BitNet Conversion

```python
def convert_to_bitnet(model):
    """
    Converts all linear layers in a model to BitLinear layers.
    
    Quantizes weights to 1-bit representation while preserving model behavior.
    
    Args:
        model: PyTorch model with nn.Linear layers
        
    Returns:
        model: Model with all linear layers converted to BitLinear
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create BitLinear with same dimensions
            bit_linear = BitLinear(module.in_features, module.out_features, bias=module.bias is not None)
            
            # Copy weights and bias
            with torch.no_grad():
                bit_linear.weight.copy_(module.weight)
                if module.bias is not None:
                    bit_linear.bias.copy_(module.bias)
                    
            # Replace module
            setattr(model, name, bit_linear)
        else:
            # Recursively convert submodules
            convert_to_bitnet(module)
            
    return model
```

### Stage 3.2: Calibration and Fine-tuning

```python
def calibrate_bitnet_model(model, calibration_loader, epochs=1, lr=1e-5):
    """
    Calibrates a BitNet model after quantization to recover accuracy.
    
    Performs a short fine-tuning pass to adjust biases and scaling factors.
    
    Args:
        model: BitNet model
        calibration_loader: DataLoader with calibration samples
        epochs: Number of calibration epochs
        lr: Learning rate for calibration
        
    Returns:
        model: Calibrated BitNet model
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        
        for batch in calibration_loader:
            inputs = batch['input_ids'].to(model.device)
            targets = batch['labels'].to(model.device)
            
            # Forward pass
            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model
```

## Phase 4: Training Protocols and Fine-Tuning

**Objective**: Train the integrated model on specialized datasets to develop its cognitive capabilities.

### Stage 4.1: Module-Specific Training

```python
def train_metacognition(model, metacog_module, train_loader, val_loader, epochs=5, lr=1e-5):
    """
    Trains the metacognition module on a confidence calibration task.
    
    Args:
        model: Base model (Mamba backbone)
        metacog_module: Metacognition module to train
        train_loader: DataLoader with QA pairs and correctness labels
        val_loader: Validation DataLoader
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        metacog_module: Trained metacognition module
    """
    optimizer = torch.optim.AdamW(metacog_module.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training loop
        model.eval()  # Freeze backbone
        metacog_module.train()
        
        for batch in train_loader:
            inputs = batch['input_ids'].to(model.device)
            correctness = batch['correctness'].to(model.device)
            
            # Get model hidden states
            with torch.no_grad():
                outputs = model(inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1][:, -1, :]  # Last token, last layer
            
            # Predict confidence
            confidence = metacog_module(hidden_states)
            
            # Calibration loss
            loss = confidence_calibration_loss(confidence, correctness)
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Validation
        metacog_module.eval()
        val_ece = evaluate_calibration(model, metacog_module, val_loader)
        
    return metacog_module
```

### Stage 4.2: Multi-task Fine-tuning

```python
class AvianCognitiveModel(nn.Module):
    """
    Complete avian-inspired cognitive architecture.
    
    Integrates Mamba backbone with four cognitive modules for metacognition,
    Bayesian inference, planning, and numerical processing.
    """
    def __init__(self, mamba_config, bit_linear=True):
        super().__init__()
        
        # Mamba backbone
        self.backbone = MambaModel(mamba_config)
        
        # Cognitive modules
        self.metacog_module = MetacognitionModule(mamba_config.d_model, bit_linear)
        self.bayes_module = BayesianInferenceModule(mamba_config.d_model, mamba_config.d_model // 2, bit_linear)
        self.planning_module = PlanningModule(mamba_config.d_model, bit_linear=bit_linear)
        self.numeric_module = NumericalModule(mamba_config.d_model, bit_linear=bit_linear)
        
        # Output head
        self.lm_head = BitLinear(mamba_config.d_model, mamba_config.vocab_size) if bit_linear else nn.Linear(mamba_config.d_model, mamba_config.vocab_size)
        
    def forward(self, input_ids, attention_mask=None, labels=None, return_confidence=False):
        # Process input through Mamba backbone with Bayesian updates
        batch_size, seq_len = input_ids.shape
        hidden_states = []
        belief_state = None
        
        for t in range(seq_len):
            # Get token embedding
            token_ids = input_ids[:, t:t+1]
            token_embedding = self.backbone.token_embedding(token_ids).squeeze(1)
            
            # Bayesian update
            belief_state, belief_embedding = self.bayes_module(token_embedding, belief_state)
            
            # Inject belief into token representation
            augmented_embedding = token_embedding + belief_embedding
            
            # Process through Mamba layers
            hidden_state = self.backbone.process_token(augmented_embedding)
            hidden_states.append(hidden_state)
            
        # Stack hidden states
        hidden_states = torch.stack(hidden_states, dim=1)
        
        # Planning module for reasoning
        final_hidden = hidden_states[:, -1, :]
        plan_embedding, _ = self.planning_module(final_hidden)
        
        # Combine plan with final hidden state
        output_hidden = final_hidden + plan_embedding
        
        # Language modeling head
        logits = self.lm_head(output_hidden)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
        # Metacognition (confidence estimation)
        confidence = None
        if return_confidence:
            confidence = self.metacog_module(output_hidden)
            
        return MambaOutputWithConfidence(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            confidence=confidence
        )
        
def multitask_training(model, task_loaders, epochs=3, lr=5e-6):
    """
    Multi-task training for the integrated avian cognitive model.
    
    Args:
        model: AvianCognitiveModel instance
        task_loaders: Dict mapping task names to DataLoaders
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        model: Trained model
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        
        # Create iterators for each task
        iterators = {task: iter(loader) for task, loader in task_loaders.items()}
        active_tasks = set(task_loaders.keys())
        
        # Training loop
        while active_tasks:
            for task in list(active_tasks):
                try:
                    batch = next(iterators[task])
                except StopIteration:
                    active_tasks.remove(task)
                    continue
                    
                # Forward pass (task-specific)
                if task == 'language_modeling':
                    outputs = model(batch['input_ids'], labels=batch['labels'])
                    loss = outputs.loss
                elif task == 'metacognition':
                    outputs = model(batch['input_ids'], return_confidence=True)
                    confidence = outputs.confidence
                    loss = confidence_calibration_loss(confidence, batch['correctness'])
                elif task == 'bayesian':
                    # Custom loss for Bayesian inference
                    # ...
                    pass
                elif task == 'planning':
                    # Custom loss for planning
                    # ...
                    pass
                elif task == 'numeric':
                    # Custom loss for numeric reasoning
                    # ...
                    pass
                    
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
    return model
```

## Phase 5: Evaluation and Benchmarking

**Objective**: Rigorously evaluate the model's performance on cognitive tasks.

### Stage 5.1: Metacognition Evaluation

```python
def evaluate_calibration(model, val_loader):
    """
    Evaluates the calibration of the model's confidence predictions.
    
    Calculates Expected Calibration Error (ECE) and generates reliability diagram.
    
    Args:
        model: Avian cognitive model
        val_loader: Validation DataLoader with correctness labels
        
    Returns:
        ece: Expected Calibration Error (lower is better)
    """
    model.eval()
    confidences = []
    correctnesses = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input_ids'].to(model.device)
            correct = batch['correctness'].to(model.device)
            
            outputs = model(inputs, return_confidence=True)
            confidence = outputs.confidence
            
            confidences.append(confidence)
            correctnesses.append(correct)
            
    confidences = torch.cat(confidences).cpu().numpy()
    correctnesses = torch.cat(correctnesses).cpu().numpy()
    
    # Calculate ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries) - 1
    
    ece = 0
    for bin_idx in range(n_bins):
        bin_mask = bin_indices == bin_idx
        if np.any(bin_mask):
            bin_conf = confidences[bin_mask].mean()
            bin_acc = correctnesses[bin_mask].mean()
            bin_count = np.sum(bin_mask)
            
            ece += (bin_count / len(confidences)) * abs(bin_acc - bin_conf)
            
    return ece
```

### Stage 5.2: Bayesian Inference Evaluation

```python
def evaluate_bayesian_inference(model, test_problems):
    """
    Evaluates the model's Bayesian inference capabilities.
    
    Tests ability to update beliefs given sequential evidence and compare
    to analytical Bayesian posterior.
    
    Args:
        model: Avian cognitive model
        test_problems: List of test problems with analytical solutions
        
    Returns:
        kl_divergence: Average KL divergence between model posterior and true posterior
    """
    model.eval()
    kl_divergences = []
    
    for problem in test_problems:
        inputs = problem['input_ids'].to(model.device)
        true_posterior = problem['posterior'].to(model.device)
        
        # Process sequentially to update belief
        belief_state = None
        for t in range(inputs.shape[1]):
            token_ids = inputs[:, t:t+1]
            token_embedding = model.backbone.token_embedding(token_ids).squeeze(1)
            belief_state, _ = model.bayes_module(token_embedding, belief_state)
            
        # Get model's posterior belief
        model_posterior = F.softmax(belief_state, dim=-1)
        
        # Calculate KL divergence
        kl_div = F.kl_div(
            model_posterior.log(),
            true_posterior,
            reduction='batchmean'
        )
        
        kl_divergences.append(kl_div.item())
        
    return np.mean(kl_divergences)
```

### Stage 5.3: Planning and Reasoning Evaluation

```python
def evaluate_multistep_reasoning(model, gsm8k_loader):
    """
    Evaluates the model's performance on multi-step math reasoning problems.
    
    Tests ability to break down complex problems into sequential steps.
    
    Args:
        model: Avian cognitive model
        gsm8k_loader: DataLoader with GSM8K math word problems
        
    Returns:
        accuracy: Proportion of correctly solved problems
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in gsm8k_loader:
            inputs = batch['input_ids'].to(model.device)
            targets = batch['answer'].to(model.device)
            
            # Generate answer with planning
            outputs = model.generate(
                inputs,
                max_length=512,
                num_return_sequences=1,
                output_hidden_states=True
            )
            
            # Extract answers from generation
            predicted_answers = extract_answers(outputs)
            
            # Compare with targets
            batch_correct = (predicted_answers == targets).sum().item()
            
            correct += batch_correct
            total += len(targets)
            
    accuracy = correct / total
    return accuracy
```

### Stage 5.4: Numerical Competence Evaluation

```python
def evaluate_arithmetic(model, arithmetic_problems):
    """
    Evaluates the model's arithmetic capabilities.
    
    Tests ability to perform exact calculations, including extrapolation
    beyond training range.
    
    Args:
        model: Avian cognitive model
        arithmetic_problems: Test set of arithmetic problems
        
    Returns:
        accuracy: Proportion of correctly solved arithmetic problems
        extrapolation_accuracy: Accuracy on problems beyond training range
    """
    model.eval()
    correct = 0
    extrapolation_correct = 0
    total = 0
    extrapolation_total = 0
    
    with torch.no_grad():
        for problem in arithmetic_problems:
            inputs = problem['input_ids'].to(model.device)
            expected = problem['result'].to(model.device)
            is_extrapolation = problem['is_extrapolation']
            
            # Extract operands and operation
            op1_embedding = model.get_operand_embedding(inputs, problem['op1_position'])
            op2_embedding = model.get_operand_embedding(inputs, problem['op2_position'])
            op_embedding = model.get_operand_embedding(inputs, problem['op_position'])
            
            # Perform calculation
            result_embedding = model.numeric_module(op1_embedding, op2_embedding, op_embedding)
            
            # Decode result
            result = model.decode_numeric(result_embedding)
            
            # Check correctness
            is_correct = torch.isclose(result, expected, rtol=1e-3)
            
            correct += is_correct.item()
            total += 1
            
            if is_extrapolation:
                extrapolation_correct += is_correct.item()
                extrapolation_total += 1
                
    accuracy = correct / total
    extrapolation_accuracy = extrapolation_correct / extrapolation_total if extrapolation_total > 0 else 0
    
    return accuracy, extrapolation_accuracy
```

## Phase 6: Deployment and Open-Source Release

**Objective**: Prepare the model for release to the research community and optimize for resource-constrained deployment.

### Stage 6.1: Optimized Inference

```python
def optimize_for_cpu_inference(model, save_path):
    """
    Optimizes the BitNet model for efficient CPU inference.
    
    Exports model in a format suitable for bitnet.cpp or other efficient
    1-bit inference engines.
    
    Args:
        model: Trained BitNet model
        save_path: Path to save optimized model
        
    Returns:
        inference_stats: Dictionary with memory usage and throughput metrics
    """
    # Convert to inference mode
    model.eval()
    
    # Pack binary weights into efficient format
    packed_model = pack_binary_weights(model)
    
    # Export model configuration
    config = {
        'd_model': model.config.d_model,
        'n_layer': model.config.n_layer,
        'vocab_size': model.config.vocab_size,
        'has_metacognition': True,
        'has_bayesian': True,
        'has_planning': True,
        'has_numeric': True
    }
    
    # Save packed model and configuration
    torch.save({
        'packed_weights': packed_model,
        'config': config,
        'bias': extract_biases(model)
    }, save_path)
    
    # Measure memory usage
    memory_usage = calculate_memory_footprint(packed_model)
    
    # Benchmark inference speed
    throughput = benchmark_inference_speed(packed_model)
    
    return {
        'memory_mb': memory_usage,
        'tokens_per_second': throughput
    }
```

### Stage 6.2: Open-Source Documentation

```python
def generate_model_card(model, eval_results, save_path):
    """
    Generates a comprehensive model card for the open-source release.
    
    Creates documentation of the model's capabilities, performance,
    and resource requirements.
    
    Args:
        model: Trained avian cognitive model
        eval_results: Dictionary of evaluation metrics
        save_path: Path to save model card
    """
    model_card = f"""
    # Avian Cognitive Model: {model.config.model_type}
    
    ## Model Description
    
    A lightweight, efficient cognitive architecture inspired by avian neurobiology,
    combining Mamba state-space modeling with BitNet 1-bit quantization. The model
    features specialized modules for metacognition, Bayesian inference, planning,
    and numerical processing.
    
    ## Performance Metrics
    
    - **Language Modeling**: Perplexity = {eval_results['perplexity']:.2f}
    - **Uncertainty Calibration**: ECE = {eval_results['ece']:.4f}
    - **Bayesian Inference**: KL Divergence = {eval_results['kl_div']:.4f}
    - **Multi-step Reasoning**: GSM8K Accuracy = {eval_results['gsm8k_acc']*100:.1f}%
    - **Arithmetic**: Accuracy = {eval_results['arithmetic_acc']*100:.1f}%
    
    ## Resource Requirements
    
    - **Memory**: {eval_results['memory_mb']:.1f} MB
    - **Inference Speed**: {eval_results['tokens_per_second']:.1f} tokens/sec on CPU
    
    ## Limitations and Biases
    
    [...]
    
    ## Usage
    
    ```python
    from avian_cognition import AvianCognitiveModel
    
    model = AvianCognitiveModel.from_pretrained("angrysky56/avian-mamba-small")
    
    # Generate text with confidence estimation
    outputs = model.generate(
        "What is the sum of 345 and 789?",
        return_confidence=True
    )
    
    print(f"Answer: {outputs.text}")
    print(f"Confidence: {outputs.confidence:.2f}")
    ```
    """
    
    with open(save_path, 'w') as f:
        f.write(model_card)
```

This roadmap provides a comprehensive guide to implementing the Avian Cognitive Architecture, from initial setup through to evaluation and deployment. Each phase builds upon the previous one, creating a sophisticated AI system inspired by the remarkable cognitive capabilities of birds.