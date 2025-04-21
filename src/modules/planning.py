"""
Planning and Tool-Use Module

This module implements components for sequential reasoning, planning, and 
simulated interaction with external tools or actions, inspired by the 
prospective cognition and tool manipulation capabilities observed in corvids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Attempt to import BitNet components, fall back to standard PyTorch layers
try:
    from src.core.bitnet import BitLinear, BitGRUCell
    print("Successfully imported BitLinear and BitGRUCell for Planning module.")
except ImportError:
    print("Warning: BitLinear or BitGRUCell not found. Falling back to nn.Linear and nn.GRUCell for Planning module.")
    BitLinear = nn.Linear
    # Provide a basic GRUCell fallback if BitGRUCell is missing
    if 'BitGRUCell' not in globals():
        BitGRUCell = nn.GRUCell


class PlanningAttention(nn.Module):
    """
    Selective context attention mechanism for focused planning steps.
    
    Allows the planning process to dynamically attend to relevant parts of the 
    input context memory at each reasoning step, weighting information based on 
    the current planning state (query). Uses standard scaled dot-product attention.
    
    Attributes:
        query_projection (nn.Module): Transforms planning state (query).
        key_projection (nn.Module): Transforms context memory elements (keys).
        value_projection (nn.Module): Transforms context memory elements (values).
        attention_combine (nn.Module): Combines attended values with the query.
    """
    
    def __init__(self, query_dim, key_dim, bit_linear=True):
        """
        Initialize planning attention layer.
        
        Args:
            query_dim (int): Dimension of the planning state query.
            key_dim (int): Dimension of the context memory keys/values (hidden_dim).
            bit_linear (bool): Whether to use BitLinear layers.
        """
        super().__init__()
        
        LinearLayer = BitLinear if bit_linear else nn.Linear
        # Attention dimension often set to query_dim or key_dim, or a fraction
        self.attention_dim = query_dim # Can be configured differently
        
        # Linear projections for Query, Key, Value
        self.query_projection = LinearLayer(query_dim, self.attention_dim, bias=False)
        self.key_projection = LinearLayer(key_dim, self.attention_dim, bias=False)
        self.value_projection = LinearLayer(key_dim, query_dim, bias=False) # Value projected back to query_dim
        
        # Output layer to combine query and attended context
        # Takes concatenated [query, context_vector]
        self.attention_combine = LinearLayer(query_dim * 2, query_dim) 
        
    def forward(self, query, context_memory):
        """
        Performs attention calculation.
        
        Args:
            query (torch.Tensor): Current planning state (the query).
                                  Shape: [batch_size, query_dim].
            context_memory (torch.Tensor): Context sequence to attend over.
                                           Shape: [batch_size, seq_len, key_dim].
            
        Returns:
            torch.Tensor: An updated query vector incorporating attended context.
                          Shape: [batch_size, query_dim].
        """
        batch_size, seq_len, key_dim_mem = context_memory.shape
        query_dim_in = query.shape[-1]
        device = query.device
        
        # Ensure context memory is on the right device
        context_memory = context_memory.to(device)

        # Project query, keys, and values
        q = self.query_projection(query)    # [batch_size, attention_dim]
        k = self.key_projection(context_memory) # [batch_size, seq_len, attention_dim]
        v = self.value_projection(context_memory) # [batch_size, seq_len, query_dim]
        
        # Prepare query for batch matrix multiplication
        q_unsq = q.unsqueeze(1)  # [batch_size, 1, attention_dim]
        
        # Calculate attention scores: (Query * Key^T) / sqrt(d_k)
        # Result shape: [batch_size, 1, seq_len]
        scores = torch.bmm(q_unsq, k.transpose(1, 2)) 
        scores = scores / (self.attention_dim ** 0.5) 
        
        # Apply softmax to get attention weights
        # Shape: [batch_size, 1, seq_len]
        attention_weights = F.softmax(scores, dim=-1) 
        
        # Calculate weighted sum of values (context vector)
        # Result shape: [batch_size, 1, query_dim]
        context_vector = torch.bmm(attention_weights, v) 
        context_vector = context_vector.squeeze(1) # Shape: [batch_size, query_dim]
        
        # Combine the original query with the attended context vector
        combined_output = torch.cat([query, context_vector], dim=-1) # Shape: [batch_size, query_dim*2]
        
        # Project the combined vector back to the query dimension
        output = self.attention_combine(combined_output) # Shape: [batch_size, query_dim]
        
        return output


class PlanningModule(nn.Module):
    """
    Neural reasoning orchestrator for multi-step cognitive planning.
    
    Generates a sequence of internal planning states using a recurrent cell, 
    optionally attending to context memory at each step. Calculates step 
    importances and aggregates the steps into a final plan embedding. Requires 
    training to produce meaningful plans.
    
    Attributes:
        hidden_dim (int): Dimension of input problem representation (from backbone).
        plan_dim (int): Dimension of the internal planning state representation.
        plan_steps (int): Number of fixed reasoning steps to simulate.
        context_encoder (nn.Module): Encodes input context into planning space.
        plan_cell (nn.Module): Recurrent cell (BitGRUCell or GRUCell) for step updates.
        step_attention (nn.Module): Attention mechanism over context memory.
        plan_aggregator (nn.Module): Aggregates step states into a final plan.
        output_proj (nn.Module): Projects final plan back to hidden_dim.
        step_gates (nn.ModuleList): Gates to determine importance of each step.
    """
    
    def __init__(self, hidden_dim, plan_dim=None, plan_steps=5, bit_linear=True):
        """
        Initialize the planning module.
        
        Args:
            hidden_dim (int): Dimension of the input context hidden state.
            plan_dim (int, optional): Dimension of internal planning states. Defaults to hidden_dim.
            plan_steps (int): Number of reasoning steps.
            bit_linear (bool): Whether to use BitLinear/BitGRUCell layers.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.plan_dim = plan_dim if plan_dim is not None else hidden_dim
        self.plan_steps = plan_steps
        LinearLayer = BitLinear if bit_linear else nn.Linear
        RecurrentCell = BitGRUCell if bit_linear else nn.GRUCell
        
        # Layer to encode the initial context state into the planning dimension
        self.context_encoder = LinearLayer(hidden_dim, self.plan_dim)
        
        # Recurrent cell for evolving the planning state over steps
        # Input size is plan_dim because it receives the output of the previous step
        # (or the encoded context for the first step)
        self.plan_cell = RecurrentCell(self.plan_dim, self.plan_dim)
        
        # Attention mechanism to incorporate context memory at each step
        self.step_attention = PlanningAttention(self.plan_dim, hidden_dim, bit_linear=bit_linear)
        
        # Layer to aggregate the sequence of plan step states into a single vector
        # Takes flattened [batch_size, plan_steps * plan_dim] input
        self.plan_aggregator = LinearLayer(self.plan_dim * self.plan_steps, self.plan_dim)
        
        # Layer to project the final aggregated plan back to the main hidden dimension
        self.output_proj = LinearLayer(self.plan_dim, hidden_dim)
        
        # Individual gates to calculate importance for each planning step
        self.step_gates = nn.ModuleList([
            LinearLayer(self.plan_dim, 1) # Output a single scalar logit per step
            for _ in range(self.plan_steps)
        ])
        
        # Activation functions
        self.activation = nn.Tanh() # Activation for initial encoding and possibly steps
        self.gate_activation = nn.Sigmoid() # For importance gates (before softmax normalization)
        
    def forward(self, context_state, context_memory=None):
        """
        Generates a plan based on the input context.
        
        Args:
            context_state (torch.Tensor): Hidden state representing the initial problem context.
                                          Shape: [batch_size, hidden_dim].
            context_memory (torch.Tensor, optional): Sequence of hidden states representing 
                                                    context history for attention.
                                                    Shape: [batch_size, seq_len, hidden_dim].
                                                    Defaults to None (attention is skipped).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - plan_embedding (torch.Tensor): Final aggregated plan projected to hidden_dim.
                                                 Shape: [batch_size, hidden_dim].
                - step_states (torch.Tensor): Sequence of internal planning states.
                                              Shape: [plan_steps, batch_size, plan_dim].
                - step_importances (torch.Tensor): Normalized importance weights for each step.
                                                  Shape: [batch_size, plan_steps].
        """
        batch_size = context_state.shape[0]
        device = context_state.device
        
        # 1. Encode the initial context into the planning dimension
        # This becomes the initial hidden state for the recurrent planning cell
        plan_state = self.activation(self.context_encoder(context_state)) # Shape: [batch_size, plan_dim]
        plan_state = plan_state.to(device) # Ensure initial state is on correct device

        # Lists to store states and importances generated at each step
        step_states_list = []
        step_importances_list = []
        
        # 2. Iteratively generate planning steps
        for i in range(self.plan_steps):
            # 2a. Attend to context memory if available
            current_step_input = plan_state # Input to GRU cell is the state from previous step
            if context_memory is not None:
                # Use current plan_state as query to attend over context_memory
                context_vector = self.step_attention(plan_state, context_memory)
                # Integrate attended context (e.g., add or concatenate before GRU)
                # Here, we simply add it, modifying the input to the GRU cell for this step.
                # Note: The GRU input size must match plan_dim. Adding is okay.
                # If concatenating, the GRU input size would need to be adjusted.
                current_step_input = current_step_input + context_vector # Add context influence

            # 2b. Update planning state using the recurrent cell
            plan_state = self.plan_cell(current_step_input, plan_state) # h_t = GRU(input_t, h_{t-1})
            step_states_list.append(plan_state)
            
            # 2c. Calculate the importance score for this step
            # Pass the *output* state of this step through its specific gate
            importance_logit = self.step_gates[i](plan_state) # Shape: [batch_size, 1]
            # Sigmoid not strictly needed here as Softmax follows, but can help stabilize
            # importance_score = self.gate_activation(importance_logit) 
            step_importances_list.append(importance_logit) # Store logits
        
        # 3. Post-process steps and importances
        # Stack the list of states into a single tensor
        step_states = torch.stack(step_states_list, dim=0) # Shape: [plan_steps, batch_size, plan_dim]
        
        # Concatenate importance logits and normalize using Softmax
        step_importances_logits = torch.cat(step_importances_list, dim=1) # Shape: [batch_size, plan_steps]
        step_importances = F.softmax(step_importances_logits, dim=1) # Normalize across steps

        # 4. Aggregate the sequence of plan states
        # Prepare states for the aggregation layer (needs batch-first)
        # Shape: [batch_size, plan_steps, plan_dim]
        states_for_aggregation = step_states.transpose(0, 1).contiguous() 
        # Flatten the steps and plan_dim dimensions
        # Shape: [batch_size, plan_steps * plan_dim]
        flat_aggregated_input = states_for_aggregation.view(batch_size, -1) 
        
        # Apply the aggregation layer
        # Shape: [batch_size, plan_dim]
        aggregated_plan = self.activation(self.plan_aggregator(flat_aggregated_input)) # Added activation
        
        # 5. Project the final aggregated plan back to the main hidden dimension
        # Shape: [batch_size, hidden_dim]
        plan_embedding = self.output_proj(aggregated_plan)
        
        return plan_embedding, step_states, step_importances
    
    @torch.no_grad()
    def decode_plan_steps(self, step_states, vocabulary_size, tokenizer=None):
        """
        Decodes internal planning step states into token IDs or text.
        
        **NOTE:** This requires a trained decoder layer (`step_decoder`) to produce
        meaningful results. Without training, the output tokens will be arbitrary.

        Args:
            step_states (torch.Tensor): Planning step state representations.
                                        Shape: [plan_steps, batch_size, plan_dim].
            vocabulary_size (int): Size of the target vocabulary for decoding.
            tokenizer (object, optional): Tokenizer with a `decode` method. Defaults to None.
            
        Returns:
            Union[torch.Tensor, List[List[str]]]: 
                - If tokenizer is None: Tensor of predicted token IDs. 
                                        Shape: [batch_size, plan_steps].
                - If tokenizer is provided: List of lists containing decoded text for each step.
                                           Shape: [batch_size, plan_steps].
        """
        plan_steps, batch_size, _ = step_states.shape
        device = step_states.device

        # Lazily initialize the step decoder if it doesn't exist
        if not hasattr(self, 'step_decoder') or \
           self.step_decoder.out_features != vocabulary_size or \
           self.step_decoder.in_features != self.plan_dim:
            print(f"Initializing step_decoder (Linear {self.plan_dim} -> {vocabulary_size}). "
                  "NOTE: This decoder requires training for meaningful output.")
            # Use standard nn.Linear for decoding, regardless of bit_linear flag for the module
            self.step_decoder = nn.Linear(self.plan_dim, vocabulary_size).to(device)
            
        # Reshape states for batch processing by the decoder
        # Shape: [plan_steps * batch_size, plan_dim]
        flat_states = step_states.view(-1, self.plan_dim) 
        
        # Decode states to vocabulary logits
        # Shape: [plan_steps * batch_size, vocabulary_size]
        logits = self.step_decoder(flat_states) 
        
        # Get the token ID with the highest logit for each step state
        # Shape: [plan_steps * batch_size]
        token_ids = torch.argmax(logits, dim=-1) 
        # Reshape back to [plan_steps, batch_size]
        token_ids = token_ids.view(plan_steps, batch_size) 
        
        # Transpose to get [batch_size, plan_steps]
        token_ids_batch_first = token_ids.transpose(0, 1) 

        # Convert to text if a tokenizer is provided
        if tokenizer is not None and hasattr(tokenizer, 'decode'):
            step_texts = []
            for i in range(batch_size):
                batch_texts = [tokenizer.decode(token_ids_batch_first[i, j].item()) for j in range(plan_steps)]
                step_texts.append(batch_texts)
            return step_texts
        else:
            # Otherwise, return the tensor of token IDs
            return token_ids_batch_first


class ToolUseModule(nn.Module):
    """
    Interface module for selecting and parameterizing external tools.
    
    Predicts which tool to use based on a hidden state and generates abstract
    parameter representations for the selected tool. Does not execute tools; 
    requires an external mechanism (e.g., tool_registry and invocation logic) 
    and training to be functional.
    
    Attributes:
        hidden_dim (int): Dimension of input hidden state.
        num_tools (int): Number of available external tools.
        max_params (int): Maximum number of parameters any tool might require.
        tool_embedding (nn.Embedding): Embeddings for each tool.
        tool_selector (nn.Module): Network to predict tool selection logits.
        parameter_generator (nn.ModuleList): Networks to generate parameter vectors.
    """
    
    def __init__(self, hidden_dim, num_tools, max_params=3, bit_linear=True):
        """
        Initialize the tool use interface module.
        
        Args:
            hidden_dim (int): Dimension of the input hidden state.
            num_tools (int): Number of external tools available.
            max_params (int): Maximum number of parameters needed by any tool.
            bit_linear (bool): Whether to use BitLinear layers.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_tools = num_tools
        self.max_params = max_params
        LinearLayer = BitLinear if bit_linear else nn.Linear
        
        # Learnable embeddings for each tool
        self.tool_embedding = nn.Embedding(num_tools, hidden_dim)
        
        # Network to select the appropriate tool based on the hidden state
        self.tool_selector = nn.Sequential(
            LinearLayer(hidden_dim, hidden_dim),
            nn.ReLU(),
            LinearLayer(hidden_dim, num_tools) # Output logits over tools
        )
        
        # Networks to generate parameter vectors (one generator per potential parameter)
        # Input to each generator is concatenated [hidden_state, expected_tool_embedding]
        self.parameter_generator = nn.ModuleList([
            nn.Sequential(
                LinearLayer(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                LinearLayer(hidden_dim, hidden_dim) # Output parameter embedding
            )
            for _ in range(max_params)
        ])
        
    def forward(self, hidden_state):
        """
        Predicts tool selection logits and generates parameter vectors.
        
        Args:
            hidden_state (torch.Tensor): Input hidden state representation.
                                          Shape: [batch_size, hidden_dim].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - tool_logits (torch.Tensor): Logits for tool selection.
                                             Shape: [batch_size, num_tools].
                - param_vectors (torch.Tensor): Generated parameter representations.
                                               Shape: [batch_size, max_params, hidden_dim].
        """
        batch_size = hidden_state.shape[0]
        device = hidden_state.device
        
        # 1. Predict tool selection logits based on the input state
        tool_logits = self.tool_selector(hidden_state) # Shape: [batch_size, num_tools]
        
        # 2. Calculate expected tool embedding based on selection probabilities
        # This provides context about the likely chosen tool to the parameter generators
        tool_probs = F.softmax(tool_logits, dim=-1) # Shape: [batch_size, num_tools]
        # Weighted average of tool embeddings based on predicted probabilities
        # Shape: [batch_size, hidden_dim]
        expected_tool_embedding = torch.matmul(tool_probs, self.tool_embedding.weight) 
        
        # 3. Generate parameter vectors
        param_vectors_list = []
        # Input for parameter generators: combines current state and expected tool info
        param_gen_input = torch.cat([hidden_state, expected_tool_embedding], dim=-1) # Shape: [batch_size, hidden_dim*2]
        
        for i in range(self.max_params):
            # Generate the i-th parameter vector
            param_vector = self.parameter_generator[i](param_gen_input) # Shape: [batch_size, hidden_dim]
            param_vectors_list.append(param_vector)
            
        # Stack the generated parameter vectors
        # Shape: [batch_size, max_params, hidden_dim]
        param_vectors = torch.stack(param_vectors_list, dim=1) 
        
        return tool_logits, param_vectors
    
    @staticmethod
    def invoke_tool(tool_logits, param_vectors, tool_registry):
        """
        Example static method showing how tool invocation might work.
        
        **NOTE:** This is conceptual. Actual invocation requires a `tool_registry` 
        mapping indices to callable functions and defining how parameter vectors 
        are translated into concrete arguments for those functions. Requires training
        of the main module to produce useful logits and parameter vectors.

        Args:
            tool_logits (torch.Tensor): Tool selection logits from forward pass.
                                        Shape: [batch_size, num_tools].
            param_vectors (torch.Tensor): Parameter vectors from forward pass.
                                          Shape: [batch_size, max_params, hidden_dim].
            tool_registry (dict): A dictionary mapping tool indices (int) to 
                                  callable tool functions.
            
        Returns:
            list: A list containing the results of tool execution for each item 
                  in the batch (or None if tool not found/callable).
        """
        batch_size = tool_logits.shape[0]
        results = []
        
        # Process each item in the batch
        for i in range(batch_size):
            # Determine the most likely tool index for this batch item
            selected_tool_idx = torch.argmax(tool_logits[i]).item()
            
            # Retrieve the corresponding tool function from the registry
            tool_function = tool_registry.get(selected_tool_idx)
            
            if tool_function is None or not callable(tool_function):
                print(f"Warning: Tool index {selected_tool_idx} not found or not callable in registry.")
                results.append(None) # Indicate tool failure or absence
                continue
                
            # Get the generated parameter vectors for this batch item
            params_for_tool = param_vectors[i] # Shape: [max_params, hidden_dim]
            
            # --- Placeholder for Parameter Translation ---
            # This is the complex part: How do abstract `param_vectors` map to 
            # concrete arguments needed by `tool_function`? This depends heavily 
            # on the specific tools and would likely involve further processing 
            # or dedicated heads trained for each tool's argument types.
            # For this example, we'll just pass the raw vectors.
            concrete_args = params_for_tool 
            # -------------------------------------------

            try:
                # Invoke the tool function with the (translated) parameters
                result = tool_function(concrete_args) 
                results.append(result)
            except Exception as e:
                print(f"Error invoking tool index {selected_tool_idx}: {e}")
                results.append(f"Error: Tool {selected_tool_idx} failed.") # Indicate execution error
                
        return results