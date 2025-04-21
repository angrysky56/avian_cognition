"""
Planning and Tool-Use Module

This module implements a sequential reasoning controller for multi-step planning
and simulated action orchestration, inspired by the tool-use capabilities and
prospective cognition observed in corvids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.bitnet import BitLinear, BitGRUCell


class PlanningModule(nn.Module):
    """
    Neural reasoning orchestrator for multi-step cognitive projection.
    
    Generates structured action sequences and simulates hypothetical futures,
    mirroring corvids' ability to plan tool use before encountering problems.
    
    The planning mechanism unfolds as a recursive thought process where each
    step builds upon previous cognitive projections, ultimately crystallizing
    into a cohesive action strategy.
    
    Attributes:
        hidden_dim: Dimension of problem representation
        plan_dim: Dimension of planning state representation
        plan_steps: Number of reasoning steps to simulate
        context_encoder: Transforms problem context into planning space
        plan_cell: Recurrent cell for step-by-step reasoning
        step_attention: Attends to relevant context for each step
        plan_aggregator: Synthesizes reasoning steps into unified plan
        output_proj: Projects planning representation to hidden dimension
    """
    
    def __init__(self, hidden_dim, plan_dim=None, plan_steps=5, bit_linear=True):
        """
        Initialize planning module with BitLinear quantization.
        
        Args:
            hidden_dim: Dimension of the problem representation
            plan_dim: Dimension of planning state (defaults to hidden_dim)
            plan_steps: Number of reasoning steps to generate
            bit_linear: Whether to use BitLinear quantization
        """
        super().__init__()
        
        # Set dimensions
        self.hidden_dim = hidden_dim
        self.plan_dim = plan_dim if plan_dim is not None else hidden_dim
        self.plan_steps = plan_steps
        
        # Context comprehension
        self.context_encoder = BitLinear(hidden_dim, self.plan_dim) if bit_linear else nn.Linear(hidden_dim, self.plan_dim)
        
        # Planning recurrence (step-by-step reasoning)
        self.plan_cell = BitGRUCell(self.plan_dim, self.plan_dim) if bit_linear else nn.GRUCell(self.plan_dim, self.plan_dim)
        
        # Step-wise context attention
        self.step_attention = PlanningAttention(self.plan_dim, hidden_dim, bit_linear=bit_linear)
        
        # Plan aggregation (unifying steps into coherent plan)
        self.plan_aggregator = BitLinear(self.plan_dim * plan_steps, self.plan_dim) if bit_linear else nn.Linear(self.plan_dim * plan_steps, self.plan_dim)
        
        # Final projection to hidden dimension
        self.output_proj = BitLinear(self.plan_dim, hidden_dim) if bit_linear else nn.Linear(self.plan_dim, hidden_dim)
        
        # Step-wise cognitive gates
        self.step_gates = nn.ModuleList([
            BitLinear(self.plan_dim, 1) if bit_linear else nn.Linear(self.plan_dim, 1)
            for _ in range(plan_steps)
        ])
        
        # Activation functions
        self.activation = nn.Tanh()
        self.gate_activation = nn.Sigmoid()
        
    def forward(self, context_state, context_memory=None):
        """
        Generates a multi-step reasoning plan from problem context.
        
        Args:
            context_state: Hidden state representing problem context
                          [batch_size, hidden_dim]
            context_memory: Optional memory of context sequence for attention
                           [batch_size, seq_len, hidden_dim]
            
        Returns:
            plan_embedding: Final aggregated plan representation
                           [batch_size, hidden_dim]
            step_states: Individual reasoning step representations
                        [plan_steps, batch_size, plan_dim]
            step_importances: Importance weights for each reasoning step
                             [batch_size, plan_steps]
        """
        batch_size = context_state.shape[0]
        device = context_state.device
        
        # Encode context
        plan_state = self.activation(self.context_encoder(context_state))
        
        # Generate reasoning steps
        step_states = []
        step_importances = []
        
        for i in range(self.plan_steps):
            # Attend to relevant context if available
            if context_memory is not None:
                context_vector = self.step_attention(plan_state, context_memory)
                # Blend context with current state
                plan_state = plan_state + context_vector
            
            # Update planning state
            plan_state = self.plan_cell(plan_state)
            step_states.append(plan_state)
            
            # Compute step importance (how much this step contributes)
            importance = self.gate_activation(self.step_gates[i](plan_state))
            step_importances.append(importance)
        
        # Stack step states and importances
        step_states = torch.stack(step_states)  # [plan_steps, batch_size, plan_dim]
        step_importances = torch.cat(step_importances, dim=1)  # [batch_size, plan_steps]
        
        # Normalize step importances
        step_importances = F.softmax(step_importances, dim=1)
        
        # Transpose step_states for batch-first concatenation
        states_for_concat = step_states.transpose(0, 1).contiguous()  # [batch_size, plan_steps, plan_dim]
        states_for_concat = states_for_concat.view(batch_size, -1)  # [batch_size, plan_steps*plan_dim]
        
        # Aggregate plan with learned weights
        aggregated_plan = self.plan_aggregator(states_for_concat)
        
        # Project to hidden dimension
        plan_embedding = self.output_proj(aggregated_plan)
        
        return plan_embedding, step_states, step_importances
    
    def decode_plan_steps(self, step_states, vocabulary, tokenizer=None):
        """
        Decodes planning steps into human-readable reasoning.
        
        Converts internal planning state representations into natural language
        descriptions of each reasoning step, facilitating interpretability.
        
        Args:
            step_states: Planning step state representations
                        [plan_steps, batch_size, plan_dim]
            vocabulary: Output vocabulary size
            tokenizer: Optional tokenizer for decoding
            
        Returns:
            step_texts: Text descriptions of planning steps
                       [batch_size, plan_steps]
        """
        if not hasattr(self, 'step_decoder'):
            # Lazily initialize step decoder if needed
            self.step_decoder = nn.Linear(self.plan_dim, vocabulary)
            
        plan_steps, batch_size, _ = step_states.shape
        
        # Reshape for batch processing
        flat_states = step_states.view(-1, self.plan_dim)  # [plan_steps*batch_size, plan_dim]
        
        # Decode to vocabulary space
        logits = self.step_decoder(flat_states)  # [plan_steps*batch_size, vocabulary]
        
        # Get most likely tokens
        token_ids = torch.argmax(logits, dim=-1)  # [plan_steps*batch_size]
        token_ids = token_ids.view(plan_steps, batch_size)  # [plan_steps, batch_size]
        
        # Convert to text if tokenizer provided
        if tokenizer is not None:
            step_texts = []
            for i in range(batch_size):
                batch_texts = []
                for j in range(plan_steps):
                    text = tokenizer.decode(token_ids[j, i].item())
                    batch_texts.append(text)
                step_texts.append(batch_texts)
            return step_texts
        
        # Otherwise return token IDs
        return token_ids.transpose(0, 1)  # [batch_size, plan_steps]


class PlanningAttention(nn.Module):
    """
    Selective context attention mechanism for focused reasoning.
    
    Enables the planning module to attend to specific aspects of problem context
    that are relevant for each reasoning step, similar to how corvids focus on
    relevant environmental features during tool selection.
    
    Attributes:
        query_projection: Transforms planning state into attention query
        key_projection: Transforms context elements into attention keys
        value_projection: Transforms context elements into attention values
        attention_combine: Combines attended values with query
    """
    
    def __init__(self, query_dim, key_dim, bit_linear=True):
        """
        Initialize planning attention with BitLinear quantization.
        
        Args:
            query_dim: Dimension of planning state query
            key_dim: Dimension of context memory keys
            bit_linear: Whether to use BitLinear quantization
        """
        super().__init__()
        
        # Projection dimensions
        self.attention_dim = min(query_dim, key_dim)
        
        # Query projection (planning state)
        self.query_projection = BitLinear(query_dim, self.attention_dim) if bit_linear else nn.Linear(query_dim, self.attention_dim)
        
        # Key projection (context memory)
        self.key_projection = BitLinear(key_dim, self.attention_dim) if bit_linear else nn.Linear(key_dim, self.attention_dim)
        
        # Value projection (context memory)
        self.value_projection = BitLinear(key_dim, query_dim) if bit_linear else nn.Linear(key_dim, query_dim)
        
        # Output combination
        self.attention_combine = BitLinear(query_dim * 2, query_dim) if bit_linear else nn.Linear(query_dim * 2, query_dim)
        
    def forward(self, query, context_memory):
        """
        Performs attention-based context retrieval.
        
        Args:
            query: Planning state to use as attention query
                  [batch_size, query_dim]
            context_memory: Context sequence to attend to
                          [batch_size, seq_len, key_dim]
            
        Returns:
            context_vector: Attended context representation
                           [batch_size, query_dim]
        """
        batch_size, seq_len, _ = context_memory.shape
        
        # Project query
        q = self.query_projection(query)  # [batch_size, attention_dim]
        
        # Project context keys and values
        k = self.key_projection(context_memory)  # [batch_size, seq_len, attention_dim]
        v = self.value_projection(context_memory)  # [batch_size, seq_len, query_dim]
        
        # Reshape query for attention
        q = q.unsqueeze(1)  # [batch_size, 1, attention_dim]
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2))  # [batch_size, 1, seq_len]
        scores = scores / (self.attention_dim ** 0.5)  # Scale by sqrt(d_k)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, 1, seq_len]
        
        # Get weighted context vector
        context_vector = torch.bmm(attention_weights, v)  # [batch_size, 1, query_dim]
        context_vector = context_vector.squeeze(1)  # [batch_size, query_dim]
        
        # Combine with original query
        combined = torch.cat([query, context_vector], dim=-1)  # [batch_size, query_dim*2]
        output = self.attention_combine(combined)  # [batch_size, query_dim]
        
        return output


class ToolUseModule(nn.Module):
    """
    Tool invocation interface for external action sequences.
    
    Enables the planning module to make use of external tools and operations,
    inspired by corvids' ability to manipulate tools for specific purposes.
    
    Attributes:
        hidden_dim: Dimension of hidden representations
        tool_embedding: Embedding for available tools
        tool_selector: Selects appropriate tool for current state
        parameter_generator: Generates parameters for selected tool
    """
    
    def __init__(self, hidden_dim, num_tools, max_params=3, bit_linear=True):
        """
        Initialize tool use module with BitLinear quantization.
        
        Args:
            hidden_dim: Dimension of hidden state representations
            num_tools: Number of available tools
            max_params: Maximum number of parameters per tool
            bit_linear: Whether to use BitLinear quantization
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_tools = num_tools
        self.max_params = max_params
        
        # Tool embeddings
        self.tool_embedding = nn.Embedding(num_tools, hidden_dim)
        
        # Tool selection network
        self.tool_selector = nn.Sequential(
            BitLinear(hidden_dim, hidden_dim) if bit_linear else nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            BitLinear(hidden_dim, num_tools) if bit_linear else nn.Linear(hidden_dim, num_tools)
        )
        
        # Parameter generation network
        self.parameter_generator = nn.ModuleList([
            nn.Sequential(
                BitLinear(hidden_dim * 2, hidden_dim) if bit_linear else nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                BitLinear(hidden_dim, hidden_dim) if bit_linear else nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(max_params)
        ])
        
    def forward(self, hidden_state):
        """
        Selects tool and generates parameters for tool invocation.
        
        Args:
            hidden_state: Hidden state representation
                         [batch_size, hidden_dim]
            
        Returns:
            tool_logits: Tool selection logits
                        [batch_size, num_tools]
            param_vectors: Parameter vector representations
                          [batch_size, max_params, hidden_dim]
        """
        batch_size = hidden_state.shape[0]
        
        # Select tool
        tool_logits = self.tool_selector(hidden_state)  # [batch_size, num_tools]
        
        # Get tool embedding for most likely tool
        tool_probs = F.softmax(tool_logits, dim=-1)  # [batch_size, num_tools]
        expected_tool_embedding = torch.matmul(tool_probs, self.tool_embedding.weight)  # [batch_size, hidden_dim]
        
        # Generate parameters
        param_vectors = []
        for i in range(self.max_params):
            # Combine hidden state with tool embedding
            combined = torch.cat([hidden_state, expected_tool_embedding], dim=-1)  # [batch_size, hidden_dim*2]
            
            # Generate parameter
            param_vector = self.parameter_generator[i](combined)  # [batch_size, hidden_dim]
            param_vectors.append(param_vector)
            
        # Stack parameters
        param_vectors = torch.stack(param_vectors, dim=1)  # [batch_size, max_params, hidden_dim]
        
        return tool_logits, param_vectors
    
    def invoke_tool(self, tool_logits, param_vectors, tool_registry):
        """
        Invokes selected tool with generated parameters.
        
        Args:
            tool_logits: Tool selection logits
                        [batch_size, num_tools]
            param_vectors: Parameter vector representations
                          [batch_size, max_params, hidden_dim]
            tool_registry: Dictionary mapping tool indices to callables
            
        Returns:
            results: Tool execution results
                    [batch_size]
        """
        batch_size = tool_logits.shape[0]
        results = []
        
        # For each item in batch
        for i in range(batch_size):
            # Get selected tool
            tool_idx = torch.argmax(tool_logits[i]).item()
            tool_fn = tool_registry.get(tool_idx)
            
            if tool_fn is None:
                results.append(None)
                continue
                
            # Get parameters for this tool
            params = param_vectors[i]  # [max_params, hidden_dim]
            
            # Invoke tool (implementation depends on tool_registry)
            result = tool_fn(params)
            results.append(result)
            
        return results
