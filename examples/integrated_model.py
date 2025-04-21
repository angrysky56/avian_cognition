"""
Integrated Avian Cognitive Architecture

This script demonstrates the creation and **structural testing** of a complete 
Avian Cognitive Architecture, integrating the Mamba-SSM backbone with 
specialized cognitive modules.

**NOTE:** This revised script removes flawed demonstration logic that relied on
untrained modules and unrealistic input/output assumptions. True functional
testing requires training the individual modules with appropriate datasets and
defining realistic interaction protocols. The demonstrations below primarily
verify structural integrity and component existence.
"""

import os
import sys
import torch
import argparse
import numpy as np
# from tqdm import tqdm # No longer needed for simplified demos
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mamba_integration import (
    # AvianMambaConfig, # Config might be accessed via model.config
    AvianMambaModel,
    create_mini_model,
    create_small_model
)
from src.modules.metacognition import (
    MetacognitionModule, 
    plot_reliability_diagram, 
    expected_calibration_error # Keep for structure
)
from src.modules.bayesian import (
    BayesianInferenceModule, 
    generate_bayesian_training_data, # Keep for structure, needs adaptation
    kl_divergence_loss # Keep for potential future use
)
from src.modules.planning import PlanningModule # Keep PlanningModule
from src.modules.numerical import NumericalModule # Keep NumericalModule
# numerical_extrapolation_test is removed as it relied on flawed demo logic

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Integrated Avian Cognitive Architecture")
    parser.add_argument(
        "--model_size",
        type=str,
        default="mini",
        choices=["mini", "small"],
        help="Model size to create"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply BitNet quantization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations (where applicable)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for saving outputs"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "demonstration"],
        help="Running mode"
    )
    # Add argument for potentially loading trained checkpoints later
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing trained module checkpoints (optional)"
    )
    return parser.parse_args()


def load_checkpoints(model, checkpoint_dir, device):
    """Load trained checkpoints for individual modules if available."""
    if checkpoint_dir is None:
        print("No checkpoint directory provided, using initialized modules.")
        return

    module_map = {
        'metacognition_module.pth': 'metacognition_module',
        'bayesian_module.pth': 'bayesian_module',
        'planning_module.pth': 'planning_module',
        'numerical_module.pth': 'numerical_module',
        # Add mappings for other potential sub-modules if needed
    }

    print(f"Attempting to load checkpoints from: {checkpoint_dir}")
    loaded_any = False
    for filename, module_name in module_map.items():
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        if hasattr(model, module_name) and os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                # Adapt keys if necessary (e.g., if saved directly from module vs part of larger model)
                # Example: Might need to remove a prefix like 'module.' if saved from DataParallel
                # Or might need to add a prefix if loading into a specific attribute
                getattr(model, module_name).load_state_dict(state_dict)
                print(f"  Successfully loaded checkpoint for {module_name}")
                loaded_any = True
            except Exception as e:
                print(f"  Failed to load checkpoint for {module_name}: {e}")
        elif hasattr(model, module_name):
             print(f"  Checkpoint not found for {module_name} at {checkpoint_path}")

    if not loaded_any:
         print("  No module checkpoints were found or loaded.")


def create_model(args):
    """
    Create integrated avian cognitive model.
    
    Args:
        args: Command line arguments
        
    Returns:
        model: Integrated AvianMambaModel
    """
    print(f"Creating {args.model_size} model with quantization={args.quantize}")
    
    if args.model_size == "mini":
        model = create_mini_model(quantize=args.quantize)
    elif args.model_size == "small":
        model = create_small_model(quantize=args.quantize)
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
        
    model.to(args.device)
    
    # Attempt to load checkpoints for modules after model creation
    load_checkpoints(model, args.checkpoint_dir, args.device)
    
    return model


def print_model_info(model):
    """
    Print detailed information about the integrated model.
    (No changes needed in this function)
    """
    print("\n=== Avian Cognitive Architecture ===")
    
    # Core architecture
    print("\nCore Architecture:")
    print(f"  Model type: {model.__class__.__name__}")
    if hasattr(model, 'config'):
        print(f"  Hidden dimension: {model.config.d_model}")
        print(f"  Number of layers: {model.config.n_layer}")
        print(f"  Vocabulary size: {model.config.vocab_size}")
        # Check if quantize attribute exists before accessing
        quantize_flag = getattr(model.config, 'quantize', False) 
        print(f"  Quantization: {quantize_flag}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    # Ensure parameters have numel attribute (handle potential non-tensor params if any)
    total_params = sum(p.numel() for p in model.parameters() if hasattr(p, 'numel'))
    fp32_size_mb = total_params * 4 / (1024 * 1024) if total_params > 0 else 0
    
    print("\nModel Size:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  FP32 size: {fp32_size_mb:.2f} MB")
    
    quantize_flag = hasattr(model, 'config') and getattr(model.config, 'quantize', False)
    if quantize_flag and total_params > 0:
        bitnet_size_mb = total_params / 8 / (1024 * 1024) 
        compression = fp32_size_mb / bitnet_size_mb if bitnet_size_mb > 0 else float('inf')
        print(f"  BitNet (1-bit) size: {bitnet_size_mb:.2f} MB")
        print(f"  Compression ratio: {compression:.2f}x")
    
    # Cognitive modules
    print("\nCognitive Modules:")
    module_names = ['metacognition_module', 'bayesian_module', 'planning_module', 'numerical_module']
    for name in module_names:
        display_name = name.replace('_module', '').replace('_', ' ').capitalize()
        if hasattr(model, name) and getattr(model, name) is not None:
            module = getattr(model, name)
            params = sum(p.numel() for p in module.parameters() if hasattr(p, 'numel'))
            print(f"  {display_name}: ✓ ({params:,} parameters)")
        else:
            print(f"  {display_name}: ✗")
            
    print("=====================================")


def demonstrate_metacognition(model, args):
    """
    Demonstrate metacognitive capabilities (structure check).
    
    NOTE: This uses synthetic data and requires a TRAINED metacognition module
    (loaded via checkpoint) for meaningful results. Otherwise, it just checks
    if the module runs and produces outputs of the correct shape.
    """
    print("\n--- Demonstrating Metacognitive Awareness (Structure Check) ---")
    
    if not hasattr(model, 'metacognition_module') or model.metacognition_module is None:
        print("Model does not have a metacognition module.")
        return
        
    metacog_module = model.metacognition_module
    
    # Generate synthetic data (placeholder for real model states and correctness)
    batch_size = 100 # Reduced size for faster structural check
    hidden_dim = model.config.d_model
    print(f"Generating {batch_size} synthetic hidden states...")
    # Use random data as a placeholder for real hidden states
    hidden_states = torch.randn(batch_size, hidden_dim, device=args.device)
    # Create random placeholder correctness labels
    correctness = torch.randint(0, 2, (batch_size, 1), device=args.device).float() 
    
    try:
        with torch.no_grad():
            confidence = metacog_module(hidden_states)
        
        # Ensure output shape is correct
        if confidence.shape != (batch_size, 1):
             raise ValueError(f"Unexpected confidence shape: {confidence.shape}")
        if not ((confidence >= 0) & (confidence <= 1)).all():
             print("Warning: Confidence scores outside [0, 1] range detected.")

        print("Metacognition module forward pass successful.")

        # Calculate metrics (will be meaningless without training)
        confidence_np = confidence.cpu().numpy().flatten()
        correctness_np = correctness.cpu().numpy().flatten()
        
        ece, bin_acc, bin_conf, bin_counts = expected_calibration_error(
            confidence_np, correctness_np, n_bins=10
        )
        print(f"  Calculated ECE: {ece:.4f} (NOTE: Meaningless without training)")
        
        # Generate visualization if requested (will show calibration of random outputs)
        if args.visualize:
            print("Generating metacognition visualization (based on synthetic data)...")
            os.makedirs(f"{args.output_dir}/metacognition", exist_ok=True)
            
            fig, _ = plot_reliability_diagram(
                confidence_np, correctness_np, n_bins=10
            )
            if fig:
                fig.suptitle("Metacognitive Calibration (Untrained/Synthetic Data)", fontsize=14)
                fig.savefig(f"{args.output_dir}/metacognition/reliability_diagram.png")
                print(f"  Saved reliability diagram to {args.output_dir}/metacognition/reliability_diagram.png")
                plt.close(fig)
            else:
                 print("  Failed to generate reliability diagram (matplotlib issue?).")

            # Confidence distribution plot (less critical) might be added here if desired

    except Exception as e:
        print(f"Error during Metacognition demonstration: {e}")


def demonstrate_bayesian_inference(model, args):
    """
    Demonstrate Bayesian inference capabilities (structure check).

    NOTE: This uses synthetic data generation which needs adaptation to output
    vectors of the correct hidden_dim. It requires a TRAINED Bayesian module
    (loaded via checkpoint) for meaningful results. Otherwise, it just checks
    if the module processes a sequence without crashing.
    """
    print("\n--- Demonstrating Bayesian Inference (Structure Check) ---")
    
    if not hasattr(model, 'bayesian_module') or model.bayesian_module is None:
        print("Model does not have a Bayesian inference module.")
        return
        
    bayesian_module = model.bayesian_module
    hidden_dim = model.config.d_model
    num_hypotheses = 3 # Example value
    seq_len = 5 # Reduced length for faster check
    batch_size = 1 # Keep batch size 1 for simplicity of demo indexing

    print("Generating synthetic Bayesian inference task...")
    # NOTE: generate_bayesian_training_data needs adaptation.
    # It currently outputs feature_dim = num_hypotheses * 2, not hidden_dim.
    # For structural testing, we might need to manually create placeholder data
    # or assume the generator will be fixed later. Let's create placeholders.
    
    # Placeholder evidence sequence with the CORRECT dimension
    evidence_sequences = torch.randn(seq_len, batch_size, hidden_dim, device=args.device)
    # Placeholder ground truth (matching batch_size=1)
    ground_truth_probs = torch.rand(batch_size, seq_len, num_hypotheses, device=args.device)
    ground_truth_probs = ground_truth_probs / ground_truth_probs.sum(dim=-1, keepdim=True) # Normalize

    print("Processing evidence sequence (structure check)...")
    
    belief_states = []
    # belief_probs = [] # Decoding is problematic without training, focus on structure

    try:
        with torch.no_grad():
            belief_state = None # Start with no prior
            for t in range(seq_len):
                # Get current evidence for the batch
                evidence_t = evidence_sequences[t] # Shape: [batch_size, hidden_dim]
                
                # Update belief - check if forward pass runs
                belief_state, belief_embedding = bayesian_module(evidence_t, belief_state)
                
                # Basic checks on output shapes
                if belief_state.shape != (batch_size, bayesian_module.belief_dim):
                     raise ValueError(f"Unexpected belief_state shape: {belief_state.shape}")
                if belief_embedding.shape != (batch_size, hidden_dim):
                     raise ValueError(f"Unexpected belief_embedding shape: {belief_embedding.shape}")
                
                belief_states.append(belief_state.cpu().clone())
                
                # NOTE: Skipping belief_probs calculation as the decoding method
                # (Tanh->linear or Softmax on slice) is unreliable without training.

        print("Bayesian module forward pass successful for sequence.")

        # Analyze belief updating (structurally possible, but values meaningless)
        print("\nBelief updating trajectory (structure check - values are untrained):")
        for t in range(seq_len):
            # Access corrected ground_truth index: sample 0, time t
            gt_posterior = ground_truth_probs[0, t, :num_hypotheses].cpu().numpy()
            # Placeholder for model belief as we skipped reliable calculation
            model_posterior_placeholder = belief_states[t][0, :num_hypotheses].numpy() # Just take a slice

            print(f"  Step {t}:")
            print(f"    Ground truth (placeholder): {gt_posterior.round(3)}")
            print(f"    Model belief state slice (untrained): {model_posterior_placeholder.round(3)}")
            # KL divergence calculation removed as model_posterior is unreliable

        # Visualization generation removed as it relies on meaningful probabilities.
        if args.visualize:
             print("Skipping Bayesian visualization as module is untrained.")

    except Exception as e:
         print(f"Error during Bayesian Inference demonstration: {e}")


def demonstrate_planning(model, args):
    """
    Demonstrate Planning capabilities (structure check).

    NOTE: This uses random data as placeholders for realistic context states 
    and memory. It requires a TRAINED planning module for meaningful results.
    It primarily checks if the module runs, performs attention (if memory given),
    and produces outputs (plan embedding, step states, importances) of the
    correct shape. Visualization of step states via PCA remains as a
    structural check.
    """
    print("\n--- Demonstrating Planning & Reasoning (Structure Check) ---")
    
    if not hasattr(model, 'planning_module') or model.planning_module is None:
        print("Model does not have a planning module.")
        return
        
    planning_module = model.planning_module
    hidden_dim = model.config.d_model
    batch_size = 1 # Keep batch size 1 for simplicity
    seq_len = 5 # Example sequence length for context memory

    print("Generating synthetic planning problem (using random placeholders)...")
    # Use random tensors as placeholders for real context/memory
    context_state = torch.randn(batch_size, hidden_dim, device=args.device)
    context_memory = torch.randn(batch_size, seq_len, hidden_dim, device=args.device)

    print("Generating reasoning plan (structure check)...")
    
    try:
        with torch.no_grad():
            plan_embedding, step_states, step_importances = planning_module(
                context_state, context_memory
            )

        # Check output shapes
        if plan_embedding.shape != (batch_size, hidden_dim):
            raise ValueError(f"Unexpected plan_embedding shape: {plan_embedding.shape}")
        if step_states.shape != (planning_module.plan_steps, batch_size, planning_module.plan_dim):
             raise ValueError(f"Unexpected step_states shape: {step_states.shape}")
        if step_importances.shape != (batch_size, planning_module.plan_steps):
             raise ValueError(f"Unexpected step_importances shape: {step_importances.shape}")
        
        print("Planning module forward pass successful.")
        
        # Analyze planning steps structurally
        print("\nPlanning steps importance (untrained values):")
        num_steps = step_states.shape[0]
        for i in range(num_steps):
            importance = step_importances[0, i].item()
            print(f"  Step {i}: {importance:.4f}")

        # Use PCA to visualize planning states in 2D (structural check)
        if args.visualize:
            print("Generating planning visualization (PCA of untrained states)...")
            os.makedirs(f"{args.output_dir}/planning", exist_ok=True)
            try:
                from sklearn.decomposition import PCA
                
                steps = step_states.permute(1, 0, 2)[0].cpu().numpy() # [num_steps, plan_dim]
                
                # Handle case where plan_dim < 2 for PCA
                n_components = min(2, steps.shape[1])
                if n_components < 1:
                    print("  Skipping PCA: Plan dimension is less than 1.")
                    return
                    
                pca = PCA(n_components=n_components)
                steps_2d = pca.fit_transform(steps)
                
                print(f"  PCA variance explained (top {n_components}): {pca.explained_variance_ratio_.sum():.2f}")

                if n_components == 2:
                    # Plotting only makes sense for 2D PCA
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.plot(steps_2d[:, 0], steps_2d[:, 1], 'b-', alpha=0.3)
                    
                    importances = step_importances[0].cpu().numpy()
                    # Normalize importances for color mapping if needed
                    norm_importances = (importances - importances.min()) / (importances.max() - importances.min() + 1e-6)

                    for i, (x, y) in enumerate(steps_2d):
                        size = 100 * norm_importances[i] + 50 # Use normalized importance for size
                        ax.scatter(x, y, s=size, alpha=0.7, 
                                   color=plt.cm.viridis(norm_importances[i])) # Use normalized for color
                        ax.text(x, y, f"{i}", fontsize=12, ha='center', va='center')
                        
                    ax.set_title("Planning Steps in Latent Space (PCA, Untrained)")
                    ax.grid(True, alpha=0.3)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
                    sm.set_array(importances) # Use original importances for color bar range
                    cbar = plt.colorbar(sm, ax=ax)
                    cbar.set_label('Step Importance (Untrained)')
                    
                    fig.savefig(f"{args.output_dir}/planning/planning_steps.png")
                    print(f"  Saved PCA plot to {args.output_dir}/planning/planning_steps.png")
                    plt.close(fig)
                else:
                    print("  Skipping PCA plot generation as dimension is not 2.")

                # Importance bar plot (still relevant structurally)
                fig, ax = plt.subplots(figsize=(10, 6))
                importances = step_importances[0].cpu().numpy()
                norm_importances = (importances - importances.min()) / (importances.max() - importances.min() + 1e-6)
                ax.bar(range(num_steps), importances, color=plt.cm.viridis(norm_importances)) # Use normalized for color
                ax.set_xlabel('Planning Step')
                ax.set_ylabel('Importance (Untrained)')
                ax.set_title('Planning Step Importance Distribution (Untrained)')
                ax.set_xticks(range(num_steps))
                ax.grid(True, alpha=0.3)
                
                fig.savefig(f"{args.output_dir}/planning/step_importance.png")
                print(f"  Saved importance plot to {args.output_dir}/planning/step_importance.png")
                plt.close(fig)

            except ImportError:
                 print("  Skipping PCA visualization: scikit-learn not installed.")
            except Exception as e:
                print(f"  Error generating planning visualizations: {e}")

    except Exception as e:
        print(f"Error during Planning demonstration: {e}")


def demonstrate_numerical(model, args):
    """
    Demonstrate Numerical processing capabilities (structure check).

    NOTE: The previous demonstration logic was flawed as it used unrealistic
    inputs and outputs for an untrained module. This function now only checks
    if the module exists and can perform a forward pass with placeholder data.
    Meaningful testing requires training and realistic data handling.
    """
    print("\n--- Demonstrating Numerical Processing (Structure Check) ---")
    
    if not hasattr(model, 'numerical_module') or model.numerical_module is None:
        print("Model does not have a numerical module.")
        return

    numerical_module = model.numerical_module
    hidden_dim = model.config.d_model
    batch_size = 1

    print("Checking numerical module forward pass with placeholder data...")

    try:
        # Create placeholder hidden states for operands and operation context
        h1 = torch.randn(batch_size, hidden_dim, device=args.device)
        h2 = torch.randn(batch_size, hidden_dim, device=args.device)
        h_op = torch.randn(batch_size, hidden_dim, device=args.device) # Placeholder for operation context

        with torch.no_grad():
            # Check if the forward pass runs without errors
            result_hidden, op_weights = numerical_module(h1, h2, h_op)

        # Check output shapes
        if result_hidden.shape != (batch_size, hidden_dim):
             raise ValueError(f"Unexpected result_hidden shape: {result_hidden.shape}")
        if op_weights.shape != (batch_size, len(numerical_module.arithmetic_units)):
             raise ValueError(f"Unexpected op_weights shape: {op_weights.shape}")

        print("Numerical module forward pass successful.")
        print("NOTE: Actual arithmetic testing requires a trained module, realistic")
        print("      input representations, and a proper output decoding mechanism.")
        
        # Visualization generation removed as it was based on flawed demo logic.
        if args.visualize:
            print("Skipping numerical visualization as it requires a trained module and evaluation setup.")

    except Exception as e:
        print(f"Error during Numerical demonstration: {e}")


def interactive_mode(model, args):
    """
    Run interactive session with the integrated model.
    Focuses on structural checks and placeholder generation.
    """
    print("\n=== Interactive Mode ===")
    print("Enter commands to perform structural checks on cognitive modules.")
    print("Type 'exit' to quit, 'info' for model information, or 'help' for commands.")
    
    # Placeholder tokenizer info - replace with actual tokenizer when available
    vocab_size = model.config.vocab_size if hasattr(model, 'config') else 10000
    print(f"Note: Using placeholder vocab size: {vocab_size}. Generation needs a real tokenizer.")

    commands = {
        'exit': 'Exit the interactive session',
        'info': 'Display model information',
        'meta': 'Run structural check for metacognition module',
        'bayes': 'Run structural check for Bayesian inference module',
        'plan': 'Run structural check for planning module',
        'math': 'Run structural check for numerical module',
        'generate': 'Generate text (placeholder - needs trained model & tokenizer)',
        'help': 'Show available commands'
    }
    
    handlers = {
        'info': lambda: print_model_info(model),
        'meta': lambda: demonstrate_metacognition(model, args),
        'bayes': lambda: demonstrate_bayesian_inference(model, args),
        'plan': lambda: demonstrate_planning(model, args),
        'math': lambda: demonstrate_numerical(model, args),
        'help': lambda: print("\nAvailable commands:\n" + "\n".join([f"  {cmd}: {desc}" for cmd, desc in commands.items()]))
    }
    
    while True:
        try:
            user_input = input("\nEnter command> ").strip().lower()
            
            if user_input == 'exit':
                print("Exiting interactive session.")
                break
                
            if user_input in handlers:
                handlers[user_input]()
                continue
                
            # Handle text generation placeholder
            # NOTE: This requires a trained model and a proper tokenizer
            if user_input.startswith('generate '):
                prompt = user_input[len('generate '):].strip()
                if not prompt:
                     print("Usage: generate <your prompt text>")
                     continue
                
                print(f"Attempting generation for: '{prompt}' (Requires trained model & tokenizer)")
                
                # Placeholder tokenization - replace with actual tokenizer
                print("Warning: Using placeholder tokenization!")
                # Create random input_ids as placeholder
                input_ids = torch.randint(1, vocab_size, (1, min(10, len(prompt.split()))), device=args.device) 
                
                try:
                    with torch.no_grad():
                         # Assume model has a generate method similar to Hugging Face Transformers
                         # The actual signature might vary based on mamba_integration specifics
                        outputs = model.generate(
                            input_ids=input_ids,
                            max_length=30 # Generate a short sequence
                            # Add other generation parameters as needed (e.g., temperature)
                            # return_confidence=True # Check if model.generate supports this
                        )
                    
                    # Placeholder decoding - replace with actual tokenizer
                    print("Warning: Using placeholder decoding!")
                    if isinstance(outputs, torch.Tensor):
                        output_tokens = outputs[0].cpu().tolist()
                        print(f"Generated token IDs (placeholder): {output_tokens}")
                        # Try to decode using simple character mapping as fallback placeholder
                        try:
                             decoded_text = "".join([chr(max(32, min(t, 126))) for t in output_tokens]) # Map to printable ASCII
                             print(f"Decoded text (placeholder): {decoded_text}")
                        except:
                             pass # Ignore decoding errors for placeholder
                    else:
                        # Handle other output types if model.generate returns differently
                        print(f"Generated output (raw): {outputs}")
                        
                except AttributeError:
                     print("Error: Model does not have a 'generate' method.")
                except Exception as e:
                    print(f"Error during text generation: {e}")
                
                continue # Added continue here
                
            # Default message for unrecognized input
            print(f"Unknown command: '{user_input}'. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            import traceback
            print(f"\nAn unexpected error occurred: {e}")
            print("Traceback:")
            traceback.print_exc()
            # Optionally continue or break depending on desired robustness
            # break


def main():
    """Main function."""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Initializing Avian Cognitive Architecture ===")
    print(f"Device: {args.device}")
    print(f"Model size: {args.model_size}")
    print(f"Quantization: {args.quantize}")
    print(f"Visualization: {args.visualize}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir or 'None'}")
    print(f"Mode: {args.mode}")
    print("==============================================")
    
    # Create model (and potentially load checkpoints)
    model = create_model(args)
    
    # Print model information
    print_model_info(model)
    
    # Run in selected mode
    if args.mode == "interactive":
        interactive_mode(model, args)
    elif args.mode == "demonstration":
        print("\nRunning structural demonstrations for all modules...")
        demonstrate_metacognition(model, args)
        demonstrate_bayesian_inference(model, args)
        demonstrate_planning(model, args)
        demonstrate_numerical(model, args)
        print("\nAll demonstrations completed.")
    else:
         print(f"Unknown mode: {args.mode}")

    print("\nExecution finished.")


if __name__ == "__main__":
    main()