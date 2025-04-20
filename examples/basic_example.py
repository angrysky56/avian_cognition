"""
Basic Example of Avian Cognitive Architecture

This script demonstrates a minimal working example of the Avian-inspired
cognitive architecture, showcasing the integration of the Mamba-SSM backbone
with the four specialized cognitive modules.
"""

import sys
import os
import torch
import argparse
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mamba_integration import (
    create_mini_model,
    create_small_model,
    create_medium_model
)
from src.modules.metacognition import plot_reliability_diagram
from src.modules.bayesian import generate_bayesian_training_data
from src.modules.numerical import numerical_extrapolation_test


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Avian Cognitive Architecture Example")
    parser.add_argument(
        "--model_size",
        type=str,
        default="mini",
        choices=["mini", "small", "medium"],
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
    return parser.parse_args()


def create_model(args):
    """Create model based on command line arguments."""
    print(f"Creating {args.model_size} model with quantization={args.quantize}")
    
    if args.model_size == "mini":
        model = create_mini_model(quantize=args.quantize)
    elif args.model_size == "small":
        model = create_small_model(quantize=args.quantize)
    elif args.model_size == "medium":
        model = create_medium_model(quantize=args.quantize)
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
        
    return model.to(args.device)


def test_metacognition(model, args):
    """Test metacognition module with synthetic data."""
    print("\n--- Testing Metacognition Module ---")
    
    # Generate synthetic data (correct and incorrect predictions)
    batch_size = 100
    
    # Mock confidence levels from 0.1 to 0.9
    confidence = torch.linspace(0.1, 0.9, batch_size).unsqueeze(1).to(args.device)
    
    # Generate correctness with probabilities matching confidence
    # (i.e., if confidence is 0.7, 70% chance of being correct)
    correctness = torch.zeros_like(confidence)
    for i in range(batch_size):
        if torch.rand(1).item() < confidence[i].item():
            correctness[i] = 1.0
            
    # Evaluate calibration
    print(f"Generating calibration evaluation...")
    
    from src.modules.metacognition import expected_calibration_error
    ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
        confidence=confidence.cpu(),
        correctness=correctness.cpu(),
        n_bins=10
    )
    
    print(f"Expected Calibration Error: {ece:.4f}")
    
    # Create reliability diagram if requested
    if args.visualize:
        print("Generating reliability diagram...")
        fig, _ = plot_reliability_diagram(
            confidence=confidence.cpu(),
            correctness=correctness.cpu(),
            n_bins=10
        )
        
        fig.savefig("metacognition_reliability.png")
        print("Saved reliability diagram to metacognition_reliability.png")


def test_bayesian_inference(model, args):
    """Test Bayesian inference module with synthetic data."""
    print("\n--- Testing Bayesian Inference Module ---")
    
    # Generate synthetic Bayesian inference tasks
    num_samples = 10
    num_hypotheses = 3
    sequence_length = 5
    
    print(f"Generating {num_samples} Bayesian inference tasks...")
    evidence_sequences, posterior_probs = generate_bayesian_training_data(
        num_samples=num_samples,
        num_hypotheses=num_hypotheses,
        sequence_length=sequence_length,
        device=args.device
    )
    
    # Print example
    print(f"\nExample Bayesian inference task:")
    for t in range(sequence_length):
        evidence = evidence_sequences[0, t]
        posterior = posterior_probs[0, t]
        
        print(f"Step {t}:")
        print(f"  Evidence: {evidence[:5]}...")
        print(f"  Posterior: {posterior}")
        
    # Visualize belief updating if requested
    if args.visualize:
        print("Generating belief trajectory visualization...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for h in range(num_hypotheses):
            # Extract belief trajectory for hypothesis h
            trajectory = [posterior_probs[0, t, h].item() for t in range(sequence_length)]
            ax.plot(range(sequence_length), trajectory, marker='o', label=f"Hypothesis {h}")
            
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Belief Probability")
        ax.set_title("Bayesian Belief Updating")
        ax.grid(True)
        ax.legend()
        
        fig.savefig("bayesian_belief_trajectory.png")
        print("Saved belief trajectory to bayesian_belief_trajectory.png")


def test_planning(model, args):
    """Test planning module with synthetic reasoning task."""
    print("\n--- Testing Planning Module ---")
    
    # Create synthetic planning problem
    batch_size = 1
    hidden_dim = model.config.d_model if hasattr(model, 'config') else 256
    
    # Mock problem representation
    problem_state = torch.randn(batch_size, hidden_dim).to(args.device)
    
    # Mock sequence context
    seq_len = 10
    context_memory = torch.randn(batch_size, seq_len, hidden_dim).to(args.device)
    
    # Invoke planning module directly if available
    if hasattr(model, 'planning_module'):
        print("Invoking planning module...")
        
        plan_embedding, step_states, step_importances = model.planning_module(
            problem_state, context_memory
        )
        
        # Print planning steps importance
        print(f"\nPlanning step importances:")
        for i, importance in enumerate(step_importances[0].cpu().detach().numpy()):
            print(f"  Step {i}: {importance:.4f}")
            
        # Visualize planning steps if requested
        if args.visualize:
            print("Generating planning steps visualization...")
            
            # Convert step states to 2D using PCA for visualization
            from sklearn.decomposition import PCA
            
            # Extract step representations
            steps = step_states.permute(1, 0, 2)[0].cpu().detach().numpy()  # [num_steps, hidden_dim]
            
            # Apply PCA
            pca = PCA(n_components=2)
            steps_2d = pca.fit_transform(steps)
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot steps as a path
            ax.plot(steps_2d[:, 0], steps_2d[:, 1], 'b-', alpha=0.3)
            
            # Plot individual steps
            for i, (x, y) in enumerate(steps_2d):
                ax.scatter(x, y, s=100 * step_importances[0, i].item() + 50, alpha=0.7)
                ax.text(x, y, f"{i}", fontsize=12, ha='center', va='center')
                
            ax.set_title("Planning Steps in Latent Space")
            ax.grid(True)
            ax.set_xticks([])
            ax.set_yticks([])
            
            fig.savefig("planning_steps.png")
            print("Saved planning visualization to planning_steps.png")
    else:
        print("Planning module not available in this model configuration")


def test_numerical(model, args):
    """Test numerical module with arithmetic operations."""
    print("\n--- Testing Numerical Module ---")
    
    # Create synthetic numerical problems
    batch_size = 1
    hidden_dim = model.config.d_model if hasattr(model, 'config') else 256
    
    # Mock number representations
    num1 = 23
    num2 = 45
    
    # Create hidden states (simplified)
    h1 = torch.zeros(batch_size, hidden_dim).to(args.device)
    h2 = torch.zeros(batch_size, hidden_dim).to(args.device)
    h_op = torch.zeros(batch_size, hidden_dim).to(args.device)
    
    # Encode numbers in the first dimension (simplified)
    h1[:, 0] = num1 / 100.0
    h2[:, 0] = num2 / 100.0
    
    # Encode operation (0 = add)
    h_op[:, 0] = 1.0
    
    # Invoke numerical module directly if available
    if hasattr(model, 'numerical_module'):
        print(f"Testing numerical operations for {num1} and {num2}...")
        
        # Process through numerical module
        result_hidden, op_weights = model.numerical_module(h1, h2, h_op)
        
        # Decode result (simplified)
        result = result_hidden[0, 0].item() * 100
        
        print(f"  Operation weights: {op_weights[0].cpu().detach().numpy().round(3)}")
        print(f"  Result (decoded): {result:.2f}")
        
        # Test extrapolation if requested
        if args.visualize:
            print("\nTesting numerical extrapolation...")
            
            # Mock improved module for testing
            model.numerical_module.hidden_dim = hidden_dim
            
            # Test extrapolation on addition
            accuracy, error_stats = numerical_extrapolation_test(
                model.numerical_module,
                value_range=(0, 1000),
                operation='add',
                device=args.device
            )
            
            print(f"Addition extrapolation accuracy: {accuracy:.2f}")
            print(f"Error statistics: {error_stats}")
            
            # Generate visualization of extrapolation performance
            print("Generating numerical extrapolation visualization...")
            
            # Test on different ranges
            ranges = [(0, 100), (100, 1000), (1000, 10000), (10000, 100000)]
            add_accuracies = []
            mul_accuracies = []
            
            for value_range in ranges:
                # Addition
                acc_add, _ = numerical_extrapolation_test(
                    model.numerical_module,
                    value_range=value_range,
                    operation='add',
                    device=args.device
                )
                add_accuracies.append(acc_add)
                
                # Multiplication
                acc_mul, _ = numerical_extrapolation_test(
                    model.numerical_module,
                    value_range=value_range,
                    operation='multiply',
                    device=args.device
                )
                mul_accuracies.append(acc_mul)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            range_labels = [f"{r[0]}-{r[1]}" for r in ranges]
            x = range(len(ranges))
            
            ax.bar([i - 0.2 for i in x], add_accuracies, width=0.4, label='Addition')
            ax.bar([i + 0.2 for i in x], mul_accuracies, width=0.4, label='Multiplication')
            
            ax.set_xticks(x)
            ax.set_xticklabels(range_labels)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Value Range')
            ax.set_title('Numerical Module Extrapolation Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.savefig("numerical_extrapolation.png")
            print("Saved numerical extrapolation visualization to numerical_extrapolation.png")
    else:
        print("Numerical module not available in this model configuration")


def run_integrated_example(model, args):
    """Run integrated example using all cognitive modules."""
    print("\n--- Running Integrated Cognitive Example ---")
    
    # Mock token IDs (simplified)
    batch_size = 1
    seq_len = 5
    vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 10000
    
    # Create random token IDs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(args.device)
    
    print(f"Running forward pass with input shape: {input_ids.shape}")
    
    # Forward pass through the integrated model
    outputs = model(input_ids, return_dict=True)
    
    # Print output information
    print(f"\nOutput shapes:")
    print(f"  Logits: {outputs.logits.shape}")
    print(f"  Hidden states: {outputs.hidden_states.shape}")
    
    if outputs.confidence is not None:
        print(f"  Confidence: {outputs.confidence.shape}, value: {outputs.confidence.item():.4f}")
    
    if outputs.belief_state is not None:
        print(f"  Belief state: {outputs.belief_state.shape}")
    
    if outputs.plan_embedding is not None:
        print(f"  Plan embedding: {outputs.plan_embedding.shape}")
    
    # Generate text (mock example)
    max_length = 10
    print(f"\nGenerating text with max_length={max_length}")
    
    try:
        if hasattr(model, 'generate'):
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.7,
                return_confidence=True
            )
            
            if isinstance(output_ids, tuple):
                output_ids, confidence = output_ids
                print(f"  Generated shape: {output_ids.shape}")
                print(f"  Confidence shape: {confidence.shape}")
            else:
                print(f"  Generated shape: {output_ids.shape}")
        else:
            print("  Generate method not available")
    except Exception as e:
        print(f"  Generation example failed: {e}")


def print_model_info(model):
    """Print information about the model."""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    if hasattr(model, 'config') and hasattr(model.config, 'quantize') and model.config.quantize:
        # For BitNet quantized model
        quantized_size_mb = total_params / 8 / (1024 * 1024)  # 1-bit
        print(f"\nModel size (BitNet quantized): {quantized_size_mb:.2f} MB")
    
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Model size (float32): {model_size_mb:.2f} MB")
    
    # Print cognitive module availability
    print("\nCognitive modules:")
    print(f"  Metacognition: {'✓' if hasattr(model, 'metacognition_module') else '✗'}")
    print(f"  Bayesian inference: {'✓' if hasattr(model, 'bayesian_module') else '✗'}")
    print(f"  Planning: {'✓' if hasattr(model, 'planning_module') else '✗'}")
    print(f"  Numerical: {'✓' if hasattr(model, 'numerical_module') else '✗'}")


def main():
    """Main function."""
    args = parse_args()
    
    # Print config
    print(f"Device: {args.device}")
    print(f"Model size: {args.model_size}")
    print(f"Quantization: {args.quantize}")
    print(f"Visualization: {args.visualize}")
    
    # Create model
    model = create_model(args)
    
    # Print model information
    print_model_info(model)
    
    # Run individual module tests
    test_metacognition(model, args)
    test_bayesian_inference(model, args)
    test_planning(model, args)
    test_numerical(model, args)
    
    # Run integrated example
    run_integrated_example(model, args)
    
    print("\nExample completed successfully")


if __name__ == "__main__":
    main()
