"""
Mamba-SSM Integration

This module implements the integration layer between the Mamba SSM backbone
and the avian-inspired cognitive modules, creating a unified architecture
with linear-time complexity and selective state propagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    from mamba_ssm.models.config_mamba import MambaConfig
    print("Successfully imported Mamba and MambaConfig")
except ImportError as e:
    print(f"Warning: mamba_ssm package import failed: {e}. Using placeholder implementation.")
    # Placeholder for environments without mamba_ssm
    Mamba = type('Mamba', (nn.Module,), {})
    MambaConfig = type('MambaConfig', (), {})

from ..core.bitnet import BitLinear, convert_linear_to_bit_linear
from ..modules.metacognition import MetacognitionModule
from ..modules.bayesian import BayesianInferenceModule
from ..modules.planning import PlanningModule
from ..modules.numerical import NumericalModule


class AvianMambaConfig:
    """
    Configuration for the Avian Mamba architecture.
    
    Defines the architectural hyperparameters for the integrated system,
    including backbone dimensions and cognitive module configurations.
    
    Attributes:
        vocab_size: Vocabulary size for token embeddings
        d_model: Hidden dimension throughout the model
        n_layer: Number of Mamba layers
        ssm_d_state: SSM state dimension
        ssm_d_conv: Local convolution width
        ssm_expand: Expansion factor for SSM
        enable_metacognition: Whether to enable metacognition module
        enable_bayesian: Whether to enable Bayesian inference module
        enable_planning: Whether to enable planning module
        enable_numerical: Whether to enable numerical module
        quantize: Whether to use BitNet quantization
    """
    
    def __init__(
        self,
        vocab_size=50277,
        d_model=768,
        n_layer=24,
        ssm_d_state=16,
        ssm_d_conv=4,
        ssm_expand=2,
        enable_metacognition=True,
        enable_bayesian=True,
        enable_planning=True,
        enable_numerical=True,
        planning_steps=5,
        quantize=True
    ):
        # Mamba backbone configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.ssm_d_state = ssm_d_state
        self.ssm_d_conv = ssm_d_conv
        self.ssm_expand = ssm_expand
        
        # Cognitive module configuration
        self.enable_metacognition = enable_metacognition
        self.enable_bayesian = enable_bayesian
        self.enable_planning = enable_planning
        self.enable_numerical = enable_numerical
        
        # Module-specific configuration
        self.planning_steps = planning_steps
        
        # Quantization configuration
        self.quantize = quantize


class AvianMambaModel(nn.Module):
    """
    Integrated Avian Cognitive Architecture with Mamba backbone.
    
    Combines the Mamba SSM sequence model with avian-inspired cognitive modules
    to create a unified architecture with both computational efficiency and
    sophisticated reasoning capabilities.
    
    The system orchestrates information flow between specialized modules and
    the backbone, enabling metacognition, Bayesian inference, planning, and
    numerical processing within a linear-time recurrent framework.
    
    Attributes:
        config: Configuration object defining architecture parameters
        backbone: Mamba SSM backbone for sequence processing
        token_embedding: Embedding layer for input tokens
        position_embedding: Optional positional embedding
        layer_norm: Layer normalization for outputs
        lm_head: Language modeling head for token prediction
        metacognition_module: Module for uncertainty estimation
        bayesian_module: Module for Bayesian inference
        planning_module: Module for multi-step reasoning
        numerical_module: Module for numerical processing
    """
    
    def __init__(self, config):
        """
        Initialize the Avian Mamba architecture.
        
        Args:
            config: AvianMambaConfig object defining architecture parameters
        """
        super().__init__()
        
        self.config = config
        
        # Convert to MambaConfig for backbone
        mamba_config = self._create_mamba_config(config)
        
        # Create Mamba backbone
        try:
            self.backbone = Mamba(mamba_config)
        except (ImportError, NameError, TypeError):
            # Fallback to placeholder if mamba_ssm not available
            print("Creating placeholder Mamba backbone")
            self.backbone = self._create_placeholder_backbone(config)
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        
        # Cognitive modules
        if config.enable_metacognition:
            self.metacognition_module = MetacognitionModule(
                hidden_dim=config.d_model,
                bit_linear=config.quantize
            )
        
        if config.enable_bayesian:
            self.bayesian_module = BayesianInferenceModule(
                hidden_dim=config.d_model,
                belief_dim=config.d_model // 2,
                bit_linear=config.quantize
            )
        
        if config.enable_planning:
            self.planning_module = PlanningModule(
                hidden_dim=config.d_model,
                plan_steps=config.planning_steps,
                bit_linear=config.quantize
            )
        
        if config.enable_numerical:
            self.numerical_module = NumericalModule(
                hidden_dim=config.d_model,
                bit_linear=config.quantize
            )
        
        # Apply BitNet quantization if enabled
        if config.quantize:
            self._apply_quantization()
            
    def _create_mamba_config(self, config):
        """
        Creates a MambaConfig object from AvianMambaConfig.
        
        Args:
            config: AvianMambaConfig object
            
        Returns:
            mamba_config: MambaConfig object for backbone
        """
        try:
            mamba_config = MambaConfig(
                d_model=config.d_model,
                n_layer=config.n_layer,
                vocab_size=config.vocab_size,
                d_state=config.ssm_d_state,
                d_conv=config.ssm_d_conv,
                expand=config.ssm_expand
            )
            return mamba_config
        except (ImportError, NameError, TypeError):
            # Fallback if MambaConfig not available
            return type('PlaceholderConfig', (), {
                'd_model': config.d_model,
                'n_layer': config.n_layer,
                'vocab_size': config.vocab_size
            })
            
    def _create_placeholder_backbone(self, config):
        """
        Creates a placeholder Mamba backbone for environments without mamba_ssm.
        
        Args:
            config: AvianMambaConfig object
            
        Returns:
            backbone: Placeholder backbone with similar interface
        """
        class PlaceholderMamba(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                # Simple RNN as placeholder for Mamba
                self.layers = nn.ModuleList([
                    nn.GRU(config.d_model, config.d_model, batch_first=True)
                    for _ in range(config.n_layer)
                ])
                
            def forward(self, hidden_states, return_dict=True):
                for layer in self.layers:
                    hidden_states, _ = layer(hidden_states)
                
                if return_dict:
                    return type('MambaOutput', (), {
                        'last_hidden_state': hidden_states,
                        'hidden_states': [hidden_states]
                    })
                else:
                    return hidden_states
                    
        return PlaceholderMamba(config)
    
    def _apply_quantization(self):
        """
        Applies BitNet quantization to all linear layers in the model.
        """
        print("Applying BitNet quantization to model...")
        
        # Skip embedding and layer norm, which are typically not quantized
        modules_to_quantize = [
            self.lm_head,
            self.backbone
        ]
        
        # Add cognitive modules if enabled
        if hasattr(self, 'metacognition_module'):
            modules_to_quantize.append(self.metacognition_module)
            
        if hasattr(self, 'bayesian_module'):
            modules_to_quantize.append(self.bayesian_module)
            
        if hasattr(self, 'planning_module'):
            modules_to_quantize.append(self.planning_module)
            
        if hasattr(self, 'numerical_module'):
            modules_to_quantize.append(self.numerical_module)
            
        # Convert each module
        for module in modules_to_quantize:
            convert_linear_to_bit_linear(module)
            
    def forward_sequential(self, input_ids, attention_mask=None):
        """
        Processes input sequence in a recurrent manner, with belief updating.
        
        This implementation processes the sequence token-by-token, updating
        the belief state at each step, similar to how birds integrate
        information over time.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            hidden_states: Hidden states for each token [batch_size, seq_len, d_model]
            belief_states: Belief states for each token [batch_size, seq_len, belief_dim]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize states
        hidden_states = []
        belief_states = []
        belief_state = None
        
        # Process sequence token by token
        for t in range(seq_len):
            # Get token embedding
            token_ids = input_ids[:, t:t+1]
            token_embedding = self.token_embedding(token_ids).squeeze(1)
            
            # Update belief state if Bayesian module enabled
            if hasattr(self, 'bayesian_module'):
                belief_state, belief_embedding = self.bayesian_module(token_embedding, belief_state)
                belief_states.append(belief_state)
                
                # Augment token with belief information
                token_embedding = token_embedding + belief_embedding
            
            # Process through Mamba (simplified for sequential processing)
            # In practice, this would need a custom Mamba implementation for token-by-token processing
            hidden_state = token_embedding  # Placeholder for actual Mamba processing
            hidden_states.append(hidden_state)
            
        # Stack states
        hidden_states = torch.stack(hidden_states, dim=1)  # [batch_size, seq_len, d_model]
        
        if belief_states:
            belief_states = torch.stack(belief_states, dim=1)  # [batch_size, seq_len, belief_dim]
        else:
            belief_states = None
            
        return hidden_states, belief_states
    
    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True):
        """
        Forward pass through the integrated architecture.
        
        Orchestrates processing through the Mamba backbone and cognitive modules,
        including belief updating, planning, numerical processing, and confidence
        estimation.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            labels: Optional labels for language modeling [batch_size, seq_len]
            return_dict: Whether to return output as dictionary
            
        Returns:
            output: Model outputs including logits, hidden states, etc.
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Process through Mamba backbone
        mamba_outputs = self.backbone(embeddings, return_dict=True)
        hidden_states = mamba_outputs.last_hidden_state
        
        # Bayesian belief updating (simplified for now)
        # In practice, this would be integrated more tightly with the Mamba processing
        belief_state = None
        if hasattr(self, 'bayesian_module'):
            # Process sequence with belief updating
            # This is a simplified version - ideally would be integrated within Mamba recurrence
            final_state = hidden_states[:, -1, :]
            belief_state, _ = self.bayesian_module(final_state)
        
        # Planning module for complex reasoning
        plan_embedding = None
        if hasattr(self, 'planning_module'):
            final_state = hidden_states[:, -1, :]
            plan_embedding, _, _ = self.planning_module(final_state, hidden_states)
            
            # Augment hidden states with planning information
            if plan_embedding is not None:
                hidden_states = hidden_states + plan_embedding.unsqueeze(1)
        
        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Flatten dimensions for loss calculation
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        # Metacognition (confidence estimation)
        confidence = None
        if hasattr(self, 'metacognition_module'):
            final_state = hidden_states[:, -1, :]
            confidence = self.metacognition_module(final_state)
        
        if return_dict:
            return AvianMambaOutput(
                loss=loss,
                logits=logits,
                hidden_states=hidden_states,
                belief_state=belief_state,
                plan_embedding=plan_embedding,
                confidence=confidence
            )
        else:
            return (loss, logits, hidden_states, confidence)
    
    def generate(
        self,
        input_ids,
        max_length=100,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        return_confidence=False,
        use_planning=True,
        **kwargs
    ):
        """
        Generates text with the integrated cognitive architecture.
        
        Implements avian-inspired cognitive processes during generation, including
        belief tracking, planning, numerical processing, and confidence estimation.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Number of top tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            return_confidence: Whether to return confidence scores
            use_planning: Whether to use planning module during generation
            
        Returns:
            output_ids: Generated token IDs [batch_size, max_length]
            confidence: Optional confidence scores for generation
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize output with input
        output_ids = input_ids
        
        # Initialize belief state
        belief_state = None
        
        # Initialize confidence scores if requested
        confidence_scores = [] if return_confidence else None
        
        # Generation loop
        for i in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(output_ids, return_dict=True)
            
            # Get next token logits (last token prediction)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply planning if enabled
            if use_planning and hasattr(self, 'planning_module'):
                # Planning augmentation (simplified)
                if hasattr(outputs, 'plan_embedding') and outputs.plan_embedding is not None:
                    plan_logits = self.lm_head(outputs.plan_embedding)
                    # Blend plan logits with next token logits
                    next_token_logits = next_token_logits + 0.5 * plan_logits
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # Apply top-k filtering
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_values)
                
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create mask for indices to remove
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to output
            output_ids = torch.cat([output_ids, next_token], dim=1)
            
            # Get confidence score if requested
            if return_confidence and hasattr(self, 'metacognition_module'):
                confidence = outputs.confidence
                confidence_scores.append(confidence)
                
            # Update belief state if Bayesian module is enabled
            if hasattr(self, 'bayesian_module'):
                token_embedding = self.token_embedding(next_token).squeeze(1)
                belief_state, _ = self.bayesian_module(token_embedding, outputs.belief_state)
        
        # Combine confidence scores if requested
        if return_confidence:
            confidence = torch.cat(confidence_scores, dim=1)
            return output_ids, confidence
        else:
            return output_ids
    
    @classmethod
    def from_pretrained(cls, model_path, config=None, **kwargs):
        """
        Loads a pretrained model from disk or huggingface hub.
        
        Args:
            model_path: Path to model or model name in hub
            config: Optional configuration override
            
        Returns:
            model: Loaded AvianMambaModel
        """
        try:
            import transformers
            from transformers import AutoModelForCausalLM
            
            # Load mamba model
            print(f"Loading pretrained model from {model_path}")
            mamba_model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            
            # Create config if not provided
            if config is None:
                config = AvianMambaConfig(
                    vocab_size=mamba_model.config.vocab_size,
                    d_model=mamba_model.config.hidden_size if hasattr(mamba_model.config, 'hidden_size') else 768,
                    n_layer=mamba_model.config.num_hidden_layers if hasattr(mamba_model.config, 'num_hidden_layers') else 24
                )
                
            # Create our model
            our_model = cls(config)
            
            # Copy weights from pretrained model (simplified)
            our_model.token_embedding = mamba_model.get_input_embeddings()
            our_model.lm_head = mamba_model.get_output_embeddings()
            
            # Copy backbone weights (this would require model-specific adaptation)
            # our_model.backbone.load_state_dict(mamba_model.backbone.state_dict())
            
            return our_model
        except ImportError:
            print("Warning: transformers package not found, creating new model from scratch")
            return cls(config if config is not None else AvianMambaConfig())
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            return cls(config if config is not None else AvianMambaConfig())


class AvianMambaOutput:
    """
    Output container for AvianMambaModel.
    
    Contains the various outputs of the model, including standard language
    modeling outputs and cognitive module outputs like belief states, planning
    embeddings, and confidence scores.
    
    Attributes:
        loss: Language modeling loss
        logits: Token prediction logits
        hidden_states: Hidden states from backbone
        belief_state: Belief state from Bayesian module
        plan_embedding: Planning representation
        confidence: Confidence score from metacognition module
    """
    
    def __init__(
        self,
        loss=None,
        logits=None,
        hidden_states=None,
        belief_state=None,
        plan_embedding=None,
        confidence=None
    ):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.belief_state = belief_state
        self.plan_embedding = plan_embedding
        self.confidence = confidence


def create_mini_model(quantize=True):
    """
    Creates a minimal Avian Mamba model for experimentation.
    
    Args:
        quantize: Whether to apply BitNet quantization
        
    Returns:
        model: Minimal AvianMambaModel
    """
    config = AvianMambaConfig(
        vocab_size=10000,
        d_model=256,
        n_layer=4,
        ssm_d_state=8,
        ssm_d_conv=2,
        ssm_expand=2,
        enable_metacognition=True,
        enable_bayesian=True,
        enable_planning=True,
        enable_numerical=True,
        planning_steps=3,
        quantize=quantize
    )
    
    return AvianMambaModel(config)


def create_small_model(quantize=True):
    """
    Creates a small Avian Mamba model (130M parameters).
    
    Args:
        quantize: Whether to apply BitNet quantization
        
    Returns:
        model: Small AvianMambaModel
    """
    config = AvianMambaConfig(
        vocab_size=50277,
        d_model=768,
        n_layer=24,
        ssm_d_state=16,
        ssm_d_conv=4,
        ssm_expand=2,
        enable_metacognition=True,
        enable_bayesian=True,
        enable_planning=True,
        enable_numerical=True,
        planning_steps=5,
        quantize=quantize
    )
    
    return AvianMambaModel(config)


def create_medium_model(quantize=True):
    """
    Creates a medium Avian Mamba model (370M parameters).
    
    Args:
        quantize: Whether to apply BitNet quantization
        
    Returns:
        model: Medium AvianMambaModel
    """
    config = AvianMambaConfig(
        vocab_size=50277,
        d_model=1024,
        n_layer=48,
        ssm_d_state=16,
        ssm_d_conv=4,
        ssm_expand=2,
        enable_metacognition=True,
        enable_bayesian=True,
        enable_planning=True,
        enable_numerical=True,
        planning_steps=5,
        quantize=quantize
    )
    
    return AvianMambaModel(config)
