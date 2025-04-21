"""
Integrated Avian Cognitive Architecture

This script demonstrates the creation and usage of a complete Avian Cognitive
Architecture, integrating the Mamba-SSM backbone with all four specialized
cognitive modules in a unified computational framework.
"""

import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mamba_integration import (
    AvianMambaConfig,
    AvianMambaModel,
    create_mini_model,
    create_small_model
)
from src.modules.metacognition import MetacognitionModule, plot_reliability_diagram
from src.modules.bayesian import BayesianInferenceModule, generate_bayesian_training_data
from src.modules.planning import PlanningModule
from src.modules.numerical import NumericalModule, numerical_extrapolation_test


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
        help="Generate visualizations"
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
    return parser.parse_args()


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
        
    return model.to(args.device)


def print_model_info(model):
    """
    Print detailed information about the integrated model.
    
    Args:
        model: Integrated AvianMambaModel
    """
    print("\n=== Avian Cognitive Architecture ===")
    
    # Core architecture
    print("\nCore Architecture:")
    print(f"  Model type: {model.__class__.__name__}")
    if hasattr(model, 'config'):
        print(f"  Hidden dimension: {model.config.d_model}")
        print(f"  Number of layers: {model.config.n_layer}")
        print(f"  Vocabulary size: {model.config.vocab_size}")
        print(f"  Quantization: {model.config.quantize}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    fp32_size_mb = total_params * 4 / (1024 * 1024)
    bitnet_size_mb = total_params / 8 / (1024 * 1024)
    
    print("\nModel Size:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  FP32 size: {fp32_size_mb:.2f} MB")
    
    if hasattr(model, 'config') and hasattr(model.config, 'quantize') and model.config.quantize:
        print(f"  BitNet (1-bit) size: {bitnet_size_mb:.2f} MB")
        print(f"  Compression ratio: {fp32_size_mb / bitnet_size_mb:.2f}x")
    
    # Cognitive modules
    print("\nCognitive Modules:")
    
    if hasattr(model, 'metacognition_module'):
        meta_module = model.metacognition_module
        meta_params = sum(p.numel() for p in meta_module.parameters())
        print(f"  Metacognition: ✓ ({meta_params:,} parameters)")
    else:
        print(f"  Metacognition: ✗")
        
    if hasattr(model, 'bayesian_module'):
        bayes_module = model.bayesian_module
        bayes_params = sum(p.numel() for p in bayes_module.parameters())
        print(f"  Bayesian Inference: ✓ ({bayes_params:,} parameters)")
    else:
        print(f"  Bayesian Inference: ✗")
        
    if hasattr(model, 'planning_module'):
        plan_module = model.planning_module
        plan_params = sum(p.numel() for p in plan_module.parameters())
        print(f"  Planning: ✓ ({plan_params:,} parameters)")
    else:
        print(f"  Planning: ✗")
        
    if hasattr(model, 'numerical_module'):
        num_module = model.numerical_module
        num_params = sum(p.numel() for p in num_module.parameters())
        print(f"  Numerical: ✓ ({num_params:,} parameters)")
    else:
        print(f"  Numerical: ✗")
    
    print("=====================================")


def demonstrate_metacognition(model, args):
    """
    Demonstrate metacognitive capabilities with synthetic examples.
    
    Args:
        model: Integrated model with metacognition module
        args: Command line arguments
    """
    print("\n--- Demonstrating Metacognitive Awareness ---")
    
    # Check if model has metacognition module
    if not hasattr(model, 'metacognition_module'):
        print("Model does not have a metacognition module")
        return
        
    # Extract metacognition module
    metacog_module = model.metacognition_module
    
    # Generate synthetic examples with varying degrees of uncertainty
    batch_size = 500
    hidden_dim = model.config.d_model if hasattr(model.config, 'd_model') else 256
    
    print(f"Generating {batch_size} synthetic examples with varying certainty...")
    
    # Create mock hidden states with different patterns
    # - "Certain correct" examples have strong correct signal
    # - "Certain incorrect" examples have strong incorrect signal
    # - "Uncertain" examples have weak or mixed signals
    
    # Base representations
    hidden_states = torch.randn(batch_size, hidden_dim, device=args.device)
    
    # Add certainty signals to first dimension (arbitrary pattern)
    certainty_level = torch.linspace(0, 1, batch_size).to(args.device)
    hidden_states[:, 0] = certainty_level  # More certain examples have higher values
    
    # Add correctness signals to second dimension (arbitrary pattern)
    correctness = torch.zeros(batch_size, 1, device=args.device)
    for i in range(batch_size):
        # Examples with certainty > 0.7 are mostly correct
        # Examples with certainty < 0.3 are mostly incorrect
        # Examples in between are mixed
        if certainty_level[i] > 0.7:
            correct_prob = 0.9  # High certainty, high probability of correctness
        elif certainty_level[i] < 0.3:
            correct_prob = 0.1  # High certainty, high probability of incorrectness
        else:
            correct_prob = 0.5  # Uncertain, random correctness
            
        # Set correctness
        if torch.rand(1).item() < correct_prob:
            correctness[i] = 1.0
            hidden_states[i, 1] = 0.8  # Correct signal
        else:
            correctness[i] = 0.0
            hidden_states[i, 1] = -0.8  # Incorrect signal
    
    # Process through metacognition module
    with torch.no_grad():
        confidence = metacog_module(hidden_states)
    
    # Analyze confidence vs. correctness
    confidence_np = confidence.cpu().numpy().flatten()
    correctness_np = correctness.cpu().numpy().flatten()
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, brier_score_loss
    
    auc = roc_auc_score(correctness_np, confidence_np)
    brier = brier_score_loss(correctness_np, confidence_np)
    
    print(f"Metacognition metrics:")
    print(f"  ROC AUC: {auc:.4f} (higher is better)")
    print(f"  Brier score: {brier:.4f} (lower is better)")
    
    # Calculate calibration
    from src.modules.metacognition import expected_calibration_error
    
    ece, bin_acc, bin_conf, bin_counts = expected_calibration_error(
        confidence_np, correctness_np, n_bins=10
    )
    
    print(f"  Expected Calibration Error: {ece:.4f} (lower is better)")
    
    # Generate visualization
    if args.visualize:
        print("Generating metacognition visualization...")
        
        # Create output directory
        os.makedirs(f"{args.output_dir}/metacognition", exist_ok=True)
        
        # Create reliability diagram
        fig, _ = plot_reliability_diagram(
            confidence_np, correctness_np, n_bins=10
        )
        
        fig.suptitle("Metacognitive Calibration", fontsize=16)
        
        # Save figure
        fig.savefig(f"{args.output_dir}/metacognition/reliability_diagram.png")
        print(f"  Saved to {args.output_dir}/metacognition/reliability_diagram.png")
        
        plt.close(fig)
        
        # Create confidence distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot confidence distribution for correct vs incorrect
        correct_conf = confidence_np[correctness_np == 1]
        incorrect_conf = confidence_np[correctness_np == 0]
        
        ax.hist(correct_conf, bins=20, alpha=0.5, label='Correct examples', color='green')
        ax.hist(incorrect_conf, bins=20, alpha=0.5, label='Incorrect examples', color='red')
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution by Correctness')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        fig.savefig(f"{args.output_dir}/metacognition/confidence_distribution.png")
        print(f"  Saved to {args.output_dir}/metacognition/confidence_distribution.png")
        
        plt.close(fig)


def demonstrate_bayesian_inference(model, args):
    """
    Demonstrate Bayesian inference capabilities with synthetic examples.
    
    Args:
        model: Integrated model with Bayesian module
        args: Command line arguments
    """
    print("\n--- Demonstrating Bayesian Inference ---")
    
    # Check if model has Bayesian module
    if not hasattr(model, 'bayesian_module'):
        print("Model does not have a Bayesian inference module")
        return
        
    # Extract Bayesian module
    bayesian_module = model.bayesian_module
    
    # Generate synthetic Bayesian task
    print("Generating synthetic Bayesian inference task...")
    
    # Create a sequential evidence task
    num_hypotheses = 3
    seq_len = 10
    
    evidence_sequences, ground_truth = generate_bayesian_training_data(
        num_samples=1,
        num_hypotheses=num_hypotheses,
        sequence_length=seq_len,
        device=args.device
    )
    
    # Reshape evidence to match bayesian module input dimensions
    # The bayesian module expects [batch_size, hidden_dim] but gets [batch_size, feature_dim]
    hidden_dim = model.config.d_model if hasattr(model.config, 'd_model') else 256
    
    # Create properly shaped evidence tensors
    shaped_evidence = []
    for t in range(seq_len):
        # Get current evidence and convert to proper shape
        original_evidence = evidence_sequences[0, t]  # [batch_size=1, feature_dim]
        
        # Create a properly shaped hidden state
        reshaped_evidence = torch.zeros(1, hidden_dim, device=args.device)
        
        # Copy values from original evidence to the beginning of the reshaped tensor
        feature_dim = original_evidence.size(-1)
        reshaped_evidence[0, :feature_dim] = original_evidence
        
        shaped_evidence.append(reshaped_evidence)
    
    # Replace original evidence sequences with correctly shaped ones
    evidence_sequences = torch.stack(shaped_evidence)  # [seq_len, batch_size, hidden_dim]
    
    # Process sequence through Bayesian module
    print("Processing evidence sequence...")
    
    # Track belief evolution
    belief_states = []
    belief_probs = []
    
    with torch.no_grad():
        belief_state = None
        
        for t in range(seq_len):
            # Get current evidence
            evidence_t = evidence_sequences[t]
            
            # Update belief
            belief_state, belief_embedding = bayesian_module(evidence_t, belief_state)
            
            # Store belief state
            belief_states.append(belief_state.cpu().clone())
            
            # Convert to probabilities 
            if hasattr(bayesian_module, 'belief_activation') and isinstance(bayesian_module.belief_activation, torch.nn.Tanh):
                probs = (belief_state + 1) / 2
            else:
                probs = torch.softmax(belief_state[:, :num_hypotheses], dim=1)
                
            belief_probs.append(probs.cpu().clone())
    
    # Analyze belief updating
    print("\nBelief updating trajectory:")
    
    for t in range(seq_len):
        # Ground truth posterior
        gt_posterior = ground_truth[t, 0, :num_hypotheses].cpu().numpy()
        
        # Model posterior
        model_posterior = belief_probs[t][0, :num_hypotheses].numpy()
        
        # Calculate KL divergence
        epsilon = 1e-10  # Small value to avoid numerical issues
        kl_div = np.sum(gt_posterior * np.log((gt_posterior + epsilon) / (model_posterior + epsilon)))
        
        print(f"  Step {t}:")
        print(f"    Ground truth: {gt_posterior.round(3)}")
        print(f"    Model belief: {model_posterior.round(3)}")
        print(f"    KL divergence: {kl_div:.4f}")
    
    # Generate visualization
    if args.visualize:
        print("Generating Bayesian inference visualization...")
        
        # Create output directory
        os.makedirs(f"{args.output_dir}/bayesian", exist_ok=True)
        
        # Create belief trajectory plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert to numpy
        gt_trajectory = np.array([ground_truth[t, 0, :num_hypotheses].cpu().numpy() for t in range(seq_len)])
        model_trajectory = np.array([belief_probs[t][0, :num_hypotheses].numpy() for t in range(seq_len)])
        
        # Time steps
        time_steps = list(range(seq_len))
        
        # Colors for hypotheses
        colors = ['b', 'g', 'r', 'c', 'm']
        
        # Plot trajectories for each hypothesis
        for h in range(num_hypotheses):
            # Ground truth
            ax.plot(time_steps, gt_trajectory[:, h], f'{colors[h]}--', marker='x', linewidth=2,
                   label=f'Ground truth H{h}')
                   
            # Model belief
            ax.plot(time_steps, model_trajectory[:, h], f'{colors[h]}-', marker='o', linewidth=2,
                   label=f'Model belief H{h}')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Probability')
        ax.set_title('Bayesian Belief Updating')
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()
        
        # Save figure
        fig.savefig(f"{args.output_dir}/bayesian/belief_trajectory.png")
        print(f"  Saved to {args.output_dir}/bayesian/belief_trajectory.png")
        
        plt.close(fig)
        
        # Create KL divergence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate KL divergence at each step
        kl_divergences = []
        for t in range(seq_len):
            gt = gt_trajectory[t]
            model = model_trajectory[t]
            epsilon = 1e-10
            kl_div = np.sum(gt * np.log((gt + epsilon) / (model + epsilon)))
            kl_divergences.append(kl_div)
        
        ax.plot(time_steps, kl_divergences, 'b-', marker='o', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('KL Divergence')
        ax.set_title('Divergence from Bayes-Optimal')
        ax.grid(True)
        
        # Save figure
        fig.savefig(f"{args.output_dir}/bayesian/kl_divergence.png")
        print(f"  Saved to {args.output_dir}/bayesian/kl_divergence.png")
        
        plt.close(fig)


def demonstrate_planning(model, args):
    """
    Demonstrate planning capabilities with synthetic reasoning task.
    
    Args:
        model: Integrated model with planning module
        args: Command line arguments
    """
    print("\n--- Demonstrating Planning & Reasoning ---")
    
    # Check if model has planning module
    if not hasattr(model, 'planning_module'):
        print("Model does not have a planning module")
        return
        
    # Extract planning module
    planning_module = model.planning_module
    
    # Generate synthetic planning problem
    print("Generating synthetic planning problem...")
    
    # Create a problem representation
    batch_size = 1
    hidden_dim = model.config.d_model if hasattr(model.config, 'd_model') else 256
    
    # Create a synthetic multi-step problem representation
    # In a real implementation, this would come from encoding a natural language problem
    problem_state = torch.zeros(batch_size, hidden_dim, device=args.device)
    
    # Encode problem type in particular dimensions (arbitrary pattern)
    problem_state[0, 0] = 0.9  # Indicates reasoning problem
    problem_state[0, 1] = 0.7  # Indicates medium difficulty
    problem_state[0, 2] = 0.5  # Indicates multi-step nature
    
    # Add some context representation
    seq_len = 5
    context_memory = torch.randn(batch_size, seq_len, hidden_dim, device=args.device)
    
    # Process through planning module
    print("Generating reasoning plan...")
    
    with torch.no_grad():
        plan_embedding, step_states, step_importances = planning_module(
            problem_state, context_memory
        )
    
    # Analyze planning steps
    print("\nPlanning steps importance:")
    
    # Get number of planning steps
    num_steps = step_states.shape[0]
    
    for i in range(num_steps):
        importance = step_importances[0, i].item()
        print(f"  Step {i}: {importance:.4f}")
    
    # Use PCA to visualize planning states in 2D
    try:
        from sklearn.decomposition import PCA
        
        # Extract step representations
        steps = step_states.permute(1, 0, 2)[0].cpu().numpy()  # [num_steps, hidden_dim]
        
        # Apply PCA
        pca = PCA(n_components=2)
        steps_2d = pca.fit_transform(steps)
        
        print(f"\nPCA variance explained: {pca.explained_variance_ratio_.sum():.2f}")
        
        # Generate visualization
        if args.visualize:
            print("Generating planning visualization...")
            
            # Create output directory
            os.makedirs(f"{args.output_dir}/planning", exist_ok=True)
            
            # Create planning steps plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot steps as a path
            ax.plot(steps_2d[:, 0], steps_2d[:, 1], 'b-', alpha=0.3)
            
            # Plot individual steps
            importances = step_importances[0].cpu().numpy()
            for i, (x, y) in enumerate(steps_2d):
                size = 100 * importances[i] + 50
                ax.scatter(x, y, s=size, alpha=0.7, 
                           color=plt.cm.viridis(importances[i]))
                ax.text(x, y, f"{i}", fontsize=12, ha='center', va='center')
                
            ax.set_title("Planning Steps in Latent Space")
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add color bar for importance
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
            sm.set_array(importances)
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Step Importance')
            
            # Save figure
            fig.savefig(f"{args.output_dir}/planning/planning_steps.png")
            print(f"  Saved to {args.output_dir}/planning/planning_steps.png")
            
            plt.close(fig)
            
            # Create step importance plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.bar(range(num_steps), importances, color=plt.cm.viridis(importances))
            ax.set_xlabel('Planning Step')
            ax.set_ylabel('Importance')
            ax.set_title('Planning Step Importance Distribution')
            ax.set_xticks(range(num_steps))
            ax.grid(True, alpha=0.3)
            
            # Save figure
            fig.savefig(f"{args.output_dir}/planning/step_importance.png")
            print(f"  Saved to {args.output_dir}/planning/step_importance.png")
            
            plt.close(fig)
            
    except Exception as e:
        print(f"Error generating planning visualizations: {e}")


def demonstrate_numerical(model, args):
    """
    Demonstrate numerical processing capabilities with arithmetic tasks.
    
    Args:
        model: Integrated model with numerical module
        args: Command line arguments
    """
    print("\n--- Demonstrating Numerical Processing ---")
    
    # Check if model has numerical module
    if not hasattr(model, 'numerical_module'):
        print("Model does not have a numerical module")
        return
        
    # Extract numerical module
    numerical_module = model.numerical_module
    
    # Generate simple arithmetic problems
    print("Testing basic arithmetic operations...")
    
    # Create hidden states for operands and operations
    batch_size = 1
    hidden_dim = model.config.d_model if hasattr(model.config, 'd_model') else 256
    
    # Test operations
    operations = ['add', 'subtract', 'multiply', 'divide']
    operand_pairs = [(5, 3), (12, 4), (25, 5), (100, 10)]
    
    results = {}
    
    with torch.no_grad():
        for op_name, (a, b) in zip(operations, operand_pairs):
            # Create hidden states (simplified representation)
            h1 = torch.zeros(batch_size, hidden_dim, device=args.device)
            h2 = torch.zeros(batch_size, hidden_dim, device=args.device)
            h_op = torch.zeros(batch_size, hidden_dim, device=args.device)
            
            # Encode numbers in first dimension (simplistic)
            h1[:, 0] = a / 100.0
            h2[:, 0] = b / 100.0
            
            # Encode operation
            op_idx = operations.index(op_name)
            h_op[:, op_idx] = 1.0
            
            # Process through numerical module
            result_hidden, op_weights = numerical_module(h1, h2, h_op)
            
            # Simple decoding (just retrieve from first dimension)
            result_value = result_hidden[0, 0].item() * 100.0
            
            # Calculate correct result safely without using eval
            if op_name == 'add':
                correct_result = a + b
            elif op_name == 'subtract':
                correct_result = a - b
            elif op_name == 'multiply':
                correct_result = a * b
            elif op_name == 'divide':
                correct_result = a / b if b != 0 else float('inf')
            else:
                correct_result = 0.0
                
            # Store result
            results[op_name] = {
                'operands': (a, b),
                'model_result': result_value,
                'correct_result': correct_result,
                'op_weights': op_weights[0].cpu().numpy()
            }
    
    # Print results
    print("\nArithmetic results:")
    
    for op_name, data in results.items():
        a, b = data['operands']
        model_result = data['model_result']
        correct_result = data['correct_result']
        error = abs(model_result - correct_result)
        
        print(f"  {a} {op_name} {b} = {model_result:.2f} (correct: {correct_result}, error: {error:.2f})")
    
    # Test extrapolation
    print("\nTesting numerical extrapolation...")
    
    # Define benchmark ranges
    value_ranges = [
        (0, 100),      # Training range
        (100, 1000),   # Small extrapolation
        (1000, 10000)  # Large extrapolation
    ]
    
    extrapolation_results = {}
    
    # Skip if error exceeds threshold
    skip_extrapolation = False
    for op_name, data in results.items():
        if abs(data['model_result'] - data['correct_result']) > 10:
            skip_extrapolation = True
    
    if skip_extrapolation:
        print("  Skipping extrapolation tests due to high base error")
    else:
        # Mock module adaptation for testing
        numerical_module.hidden_dim = hidden_dim
        
        # Test extrapolation
        for op_name in ['add', 'multiply']:
            extrapolation_results[op_name] = []
            
            for value_range in value_ranges:
                accuracy, error_stats = numerical_extrapolation_test(
                    numerical_module,
                    value_range=value_range,
                    operation=op_name,
                    device=args.device
                )
                
                extrapolation_results[op_name].append({
                    'range': value_range,
                    'accuracy': accuracy,
                    'error_stats': error_stats
                })
                
                print(f"  {op_name.capitalize()} {value_range}: Accuracy = {accuracy:.2f}, Mean error = {error_stats['mean']:.2f}")
    
    # Generate visualization
    if args.visualize and not skip_extrapolation:
        print("Generating numerical visualization...")
        
        # Create output directory
        os.makedirs(f"{args.output_dir}/numerical", exist_ok=True)
        
        # Create operation recognition plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        op_names = list(results.keys())
        x = np.arange(len(op_names))
        
        # For each operation, get the weight for that operation
        op_weights = np.zeros((len(op_names), len(op_names)))
        for i, op_name in enumerate(op_names):
            op_weights[i] = results[op_name]['op_weights']
        
        # Create grouped bar chart
        bar_width = 0.15
        for i in range(len(op_names)):
            offset = (i - len(op_names)/2 + 0.5) * bar_width
            ax.bar(x + offset, op_weights[:, i], width=bar_width, 
                   label=f'{op_names[i]} weight')
        
        ax.set_xlabel('Operation')
        ax.set_ylabel('Operation Weight')
        ax.set_title('Numerical Operation Recognition')
        ax.set_xticks(x)
        ax.set_xticklabels(op_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        fig.savefig(f"{args.output_dir}/numerical/operation_recognition.png")
        print(f"  Saved to {args.output_dir}/numerical/operation_recognition.png")
        
        plt.close(fig)
        
        # Create extrapolation plot
        if extrapolation_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare data
            range_labels = [f"{r[0]}-{r[1]}" for r in value_ranges]
            x = np.arange(len(range_labels))
            
            # Plot accuracy for each operation
            bar_width = 0.35
            for i, op_name in enumerate(['add', 'multiply']):
                accuracies = [result['accuracy'] for result in extrapolation_results[op_name]]
                ax.bar(x + (i - 0.5 + 0.5) * bar_width, accuracies, width=bar_width,
                       label=f'{op_name.capitalize()}')
            
            ax.set_xlabel('Value Range')
            ax.set_ylabel('Accuracy')
            ax.set_title('Numerical Extrapolation Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(range_labels)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save figure
            fig.savefig(f"{args.output_dir}/numerical/extrapolation.png")
            print(f"  Saved to {args.output_dir}/numerical/extrapolation.png")
            
            plt.close(fig)


def interactive_mode(model, args):
    """
    Run interactive session with the integrated model.
    
    Args:
        model: Integrated AvianMambaModel
        args: Command line arguments
    """
    print("\n=== Interactive Mode ===")
    print("Enter queries to test the avian cognitive architecture.")
    print("Type 'exit' to quit, 'info' for model information, or 'help' for commands.")
    
    # Create simple tokenizer (placeholder)
    vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 10000
    
    # Define available commands
    commands = {
        'exit': 'Exit the interactive session',
        'info': 'Display model information',
        'meta': 'Test metacognition module',
        'bayes': 'Test Bayesian inference module',
        'plan': 'Test planning module',
        'math': 'Test numerical module',
        'generate': 'Generate text with the model',
        'help': 'Show available commands'
    }
    
    # Command handler mappings
    handlers = {
        'info': lambda: print_model_info(model),
        'meta': lambda: demonstrate_metacognition(model, args),
        'bayes': lambda: demonstrate_bayesian_inference(model, args),
        'plan': lambda: demonstrate_planning(model, args),
        'math': lambda: demonstrate_numerical(model, args),
        'help': lambda: print("\n".join([f"{cmd}: {desc}" for cmd, desc in commands.items()]))
    }
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\nEnter query> ").strip()
            
            # Check for exit command
            if user_input.lower() == 'exit':
                print("Exiting interactive session")
                break
                
            # Check for other commands
            if user_input.lower() in handlers:
                handlers[user_input.lower()]()
                continue
                
            # Handle text generation
            if user_input.lower().startswith('generate:'):
                prompt = user_input[9:].strip()
                print(f"Generating text for: '{prompt}'")
                
                # Create input tokens (placeholder)
                input_ids = torch.randint(0, vocab_size, (1, 5), device=args.device)
                
                try:
                    # Generate text
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_length=20,
                        return_confidence=True
                    )
                    
                    # Print outputs
                    if isinstance(outputs, tuple):
                        output_ids, confidence = outputs
                        print(f"Generated tokens: {output_ids}")
                        print(f"Confidence: {confidence.mean().item():.4f}")
                    else:
                        print(f"Generated tokens: {outputs}")
                        
                except Exception as e:
                    print(f"Error in text generation: {e}")
                
                continue
                
            # Default: process as regular input
            print("Processing query (placeholder functionality)...")
            print("Use 'help' to see available commands")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=== Avian Cognitive Architecture ===")
    print(f"Device: {args.device}")
    print(f"Model size: {args.model_size}")
    print(f"Quantization: {args.quantize}")
    print(f"Visualization: {args.visualize}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print("====================================")
    
    # Create model
    model = create_model(args)
    
    # Print model information
    print_model_info(model)
    
    # Run in selected mode
    if args.mode == "interactive":
        interactive_mode(model, args)
    else:
        # Run all demonstrations
        demonstrate_metacognition(model, args)
        demonstrate_bayesian_inference(model, args)
        demonstrate_planning(model, args)
        demonstrate_numerical(model, args)
    
    print("\nExecution completed successfully")


if __name__ == "__main__":
    main()
