#!/usr/bin/env python
"""
Generate Bayesian Inference Training Data

This script generates sequential evidence data for training the BayesianInferenceModule.
It creates scenarios where evidence gradually favors certain hypotheses, with the
ground truth posterior probabilities calculated and saved along with the data.

The script supports multiple data generation modes:
1. Synthetic scenarios (coin flips, dice rolls, etc.)
2. Model-based evidence (using pretrained language model hidden states)
3. Toy Gaussian scenarios for classification
"""

import os
import sys
import torch
import argparse
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bayesian_data_gen')

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Try to import required modules
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    has_transformers = True
except ImportError:
    has_transformers = False
    logger.warning("HuggingFace transformers not installed. Some features won't be available.")

try:
    from src.core.mamba_integration import create_mini_model, create_small_model
    has_mamba = True
except ImportError:
    has_mamba = False
    logger.warning("Mamba modules not found. Will attempt to use HuggingFace models instead.")

try:
    from src.modules.bayesian import generate_bayesian_training_data
    from src.modules.bayesian import BayesianInferenceModule
except ImportError:
    logger.error("Failed to import from src.modules.bayesian. Make sure the module exists.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate Bayesian inference training data")
    
    # Data generation mode
    parser.add_argument("--mode", type=str, default="synthetic",
                        choices=["synthetic", "model", "toy_gaussian"],
                        help="Data generation mode")
    
    # Model-specific arguments (for mode='model')
    parser.add_argument("--model_type", type=str, default="hf",
                        choices=["mamba", "hf"],
                        help="Type of model to use (only for mode='model')")
    parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/pythia-70m",
                        help="Model name or path for HuggingFace model")
    parser.add_argument("--mamba_size", type=str, default="mini",
                        choices=["mini", "small"],
                        help="Size of Mamba model to use")
    
    # Data source arguments
    parser.add_argument("--text_dataset", type=str, default=None,
                        help="Text dataset for model-based generation")
    parser.add_argument("--topics", type=str, nargs="+", default=["science", "history", "politics"],
                        help="List of topics for evidence generation")
    
    # Parameters for synthetic data
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of samples to generate")
    parser.add_argument("--num_hypotheses", type=int, default=3,
                        help="Number of hypotheses to generate evidence for")
    parser.add_argument("--sequence_length", type=int, default=5,
                        help="Length of evidence sequences")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for evidence and belief states")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="data/bayesian",
                        help="Directory to save generated data")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of data to use for training vs validation")
    
    # Runtime arguments
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for running the model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (fewer samples, more logging)")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_evidence_data(args):
    """
    Generate synthetic evidence data for Bayesian inference.
    
    Uses the built-in generate_bayesian_training_data function to create
    synthetic evidence sequences and corresponding ground-truth posteriors.
    
    Args:
        args: Command-line arguments containing data generation parameters
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Evidence sequences and posterior probabilities
    """
    logger.info("Generating synthetic Bayesian evidence data")
    
    if args.debug:
        args.num_samples = min(100, args.num_samples)
        logger.debug(f"Debug mode: limiting to {args.num_samples} samples")
    
    # Generate synthetic data using the utility function
    evidence_sequences, posterior_probs = generate_bayesian_training_data(
        num_samples=args.num_samples,
        num_hypotheses=args.num_hypotheses,
        sequence_length=args.sequence_length,
        output_feature_dim=args.hidden_dim,
        device=args.device
    )
    
    logger.info(f"Generated {args.num_samples} synthetic evidence sequences")
    logger.info(f"Evidence shape: {evidence_sequences.shape}")
    logger.info(f"Posterior shape: {posterior_probs.shape}")
    
    return evidence_sequences, posterior_probs


def generate_toy_gaussian_dataset(args):
    """
    Generate a toy Gaussian dataset for Bayesian inference.
    
    Creates a simple 2D Gaussian classification problem with two classes,
    where evidence points are drawn from class-specific distributions.
    The posterior probability is calculated using Bayes' rule.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Evidence sequences and posterior probabilities
    """
    logger.info("Generating toy Gaussian dataset for Bayesian inference")
    
    if args.debug:
        args.num_samples = min(100, args.num_samples)
        logger.debug(f"Debug mode: limiting to {args.num_samples} samples")
    
    # For a 2D Gaussian dataset
    num_classes = args.num_hypotheses
    num_samples = args.num_samples
    sequence_length = args.sequence_length
    
    # Define class means (centers of Gaussians)
    means = torch.randn(num_classes, 2, device=args.device) * 3.0
    
    # Evidence sequences: [num_samples, sequence_length, hidden_dim]
    evidence_sequences = torch.zeros(num_samples, sequence_length, args.hidden_dim, device=args.device)
    
    # Posterior probabilities: [num_samples, sequence_length, num_classes]
    posterior_probs = torch.zeros(num_samples, sequence_length, num_classes, device=args.device)
    
    # Process each sample
    for i in range(num_samples):
        # Randomly select the true class for this sequence
        true_class = torch.randint(0, num_classes, (1,), device=args.device).item()
        
        # Initialize uniform prior
        prior = torch.ones(num_classes, device=args.device) / num_classes
        
        # Generate evidence sequence
        for t in range(sequence_length):
            # Generate a 2D evidence point from the true class distribution
            # Add noise for potential overlap between classes
            point = means[true_class] + torch.randn(2, device=args.device) * 1.0
            
            # Embed the 2D point into hidden_dim
            evidence = torch.zeros(args.hidden_dim, device=args.device)
            evidence[0:2] = point  # First two dims are the actual point
            
            # Add random noise to the remaining dimensions
            if args.hidden_dim > 2:
                evidence[2:] = torch.randn(args.hidden_dim - 2, device=args.device) * 0.1
            
            # Store the evidence
            evidence_sequences[i, t] = evidence
            
            # Calculate likelihoods P(evidence|class) using multivariate Gaussian
            likelihoods = torch.zeros(num_classes, device=args.device)
            for c in range(num_classes):
                # Distance from point to this class mean
                dist = torch.norm(point - means[c])
                # Convert distance to likelihood (using a Gaussian kernel)
                likelihood = torch.exp(-0.5 * dist ** 2)
                likelihoods[c] = likelihood
            
            # Apply Bayes' rule: posterior ∝ prior * likelihood
            posterior_unnormalized = prior * likelihoods
            posterior = posterior_unnormalized / posterior_unnormalized.sum()
            
            # Store the posterior
            posterior_probs[i, t] = posterior
            
            # Update prior for next step
            prior = posterior
    
    logger.info(f"Generated {num_samples} toy Gaussian evidence sequences")
    logger.info(f"Evidence shape: {evidence_sequences.shape}")
    logger.info(f"Posterior shape: {posterior_probs.shape}")
    
    # Visualize the first few sequences if in debug mode
    if args.debug:
        try:
            import matplotlib.pyplot as plt
            
            # Plot the first 5 sequences (or fewer if fewer exist)
            num_to_plot = min(5, num_samples)
            
            plt.figure(figsize=(12, 10))
            
            # Define colors for classes
            colors = ['red', 'blue', 'green', 'purple', 'orange'][:num_classes]
            
            # Plot class means
            for c in range(num_classes):
                plt.scatter(means[c, 0].cpu(), means[c, 1].cpu(), s=100, c=colors[c], 
                            marker='*', label=f'Class {c} Mean')
            
            # Plot the first few sequences
            for i in range(num_to_plot):
                # Extract 2D points from evidence
                points = evidence_sequences[i, :, 0:2].cpu().numpy()
                
                # Plot the trajectory
                plt.plot(points[:, 0], points[:, 1], 'k-', alpha=0.3)
                
                # Plot points with color based on true posterior
                for t in range(sequence_length):
                    post = posterior_probs[i, t].cpu().numpy()
                    # Mix colors based on posterior probabilities
                    color_mix = np.zeros(3)
                    for c in range(num_classes):
                        color_idx = colors[c]
                        if color_idx == 'red':
                            rgb = [1, 0, 0]
                        elif color_idx == 'blue':
                            rgb = [0, 0, 1]
                        elif color_idx == 'green':
                            rgb = [0, 1, 0]
                        elif color_idx == 'purple':
                            rgb = [0.5, 0, 0.5]
                        elif color_idx == 'orange':
                            rgb = [1, 0.5, 0]
                        color_mix += np.array(rgb) * post[c]
                    
                    plt.scatter(points[t, 0], points[t, 1], 
                              c=[color_mix], s=50, alpha=0.7)
            
            plt.title('Toy Gaussian Dataset for Bayesian Inference')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend()
            plt.tight_layout()
            
            # Save the plot
            os.makedirs(args.output_dir, exist_ok=True)
            plt.savefig(os.path.join(args.output_dir, 'toy_gaussian_visualization.png'))
            plt.close()
            
            logger.info(f"Saved visualization to {os.path.join(args.output_dir, 'toy_gaussian_visualization.png')}")
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
        except Exception as e:
            logger.warning(f"Error during visualization: {e}")
    
    return evidence_sequences, posterior_probs


def generate_model_based_evidence(args):
    """
    Generate evidence using hidden states from a pretrained model.
    
    Uses a language model to process a dataset, capturing hidden states and
    using entropy or logit distributions as proxies for belief uncertainty.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Evidence sequences and posterior probabilities
    """
    logger.info("Generating model-based evidence for Bayesian inference")
    
    if not has_transformers and args.model_type == "hf":
        raise ImportError("HuggingFace transformers required for model-based evidence generation")
    
    if not has_mamba and args.model_type == "mamba":
        raise ImportError("Mamba modules required for Mamba-based evidence generation")
    
    if args.debug:
        args.num_samples = min(100, args.num_samples)
        logger.debug(f"Debug mode: limiting to {args.num_samples} samples")
    
    # Load model
    logger.info("Loading model...")
    if args.model_type == "hf":
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    elif args.model_type == "mamba":
        if args.mamba_size == "mini":
            model = create_mini_model(quantize=False)
        else:  # small
            model = create_small_model(quantize=False)
        # Get tokenizer from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = model.to(args.device)
    model.eval()
    
    # Create synthetic hypothesis scenarios
    scenarios = []
    for i in range(min(args.num_samples, 100)):  # Generate up to 100 scenarios (reuse if needed)
        # Randomly select a set of hypotheses for this scenario
        num_classes = args.num_hypotheses
        hypotheses = []
        
        # Randomly select a topic
        topic = np.random.choice(args.topics)
        
        # Generate hypotheses for this topic
        if topic == "science":
            options = [
                "The experiment will succeed",
                "The experiment will fail",
                "The experiment will have mixed results",
                "The experiment will have unexpected outcomes"
            ]
        elif topic == "history":
            options = [
                "The artifact is genuine",
                "The artifact is a forgery",
                "The artifact has been modified",
                "The artifact has been misidentified"
            ]
        elif topic == "politics":
            options = [
                "Candidate A will win the election",
                "Candidate B will win the election",
                "The election will end in a tie",
                "The election will be contested"
            ]
        else:
            # Generic hypotheses
            options = [f"Hypothesis {j}" for j in range(num_classes)]
        
        # Select hypotheses (ensuring we have enough)
        available_options = options[:num_classes]
        if len(available_options) < num_classes:
            # Add generic hypotheses if needed
            available_options.extend([f"Additional Hypothesis {j}" for j in range(num_classes - len(available_options))])
        
        # Randomly select the true hypothesis
        true_hypothesis_idx = np.random.randint(0, num_classes)
        
        # Create scenario
        scenario = {
            "topic": topic,
            "hypotheses": available_options,
            "true_hypothesis": true_hypothesis_idx
        }
        scenarios.append(scenario)
    
    # Repeat scenarios if necessary
    if args.num_samples > len(scenarios):
        scenarios = [scenarios[i % len(scenarios)] for i in range(args.num_samples)]
    
    # Create evidence sequences and posterior distributions
    evidence_sequences = torch.zeros(args.num_samples, args.sequence_length, args.hidden_dim, device=args.device)
    posterior_probs = torch.zeros(args.num_samples, args.sequence_length, args.num_hypotheses, device=args.device)
    
    # Process each scenario
    for i, scenario in enumerate(tqdm(scenarios, desc="Generating evidence")):
        # Initialize uniform prior
        prior = torch.ones(args.num_hypotheses, device=args.device) / args.num_hypotheses
        
        for t in range(args.sequence_length):
            # Generate evidence text based on the scenario
            # For simplicity, create a sentence mentioning the true hypothesis with some noise
            true_idx = scenario["true_hypothesis"]
            
            # Generate an evidence sentence
            confidence = min(0.5 + 0.1 * t, 0.9)  # Increase confidence over time
            
            if scenario["topic"] == "science":
                if np.random.rand() < confidence:
                    # Evidence supporting true hypothesis
                    evidence_text = f"The latest measurement shows results consistent with {scenario['hypotheses'][true_idx].lower()}."
                else:
                    # Ambiguous or opposing evidence
                    other_idx = np.random.choice([j for j in range(args.num_hypotheses) if j != true_idx])
                    evidence_text = f"Some data points suggest {scenario['hypotheses'][other_idx].lower()}, but the evidence is not conclusive."
            elif scenario["topic"] == "history":
                if np.random.rand() < confidence:
                    evidence_text = f"Carbon dating and material analysis suggests {scenario['hypotheses'][true_idx].lower()}."
                else:
                    other_idx = np.random.choice([j for j in range(args.num_hypotheses) if j != true_idx])
                    evidence_text = f"Initial examination indicates {scenario['hypotheses'][other_idx].lower()}, but further tests are needed."
            elif scenario["topic"] == "politics":
                if np.random.rand() < confidence:
                    evidence_text = f"Recent polling data suggests {scenario['hypotheses'][true_idx].lower()}."
                else:
                    other_idx = np.random.choice([j for j in range(args.num_hypotheses) if j != true_idx])
                    evidence_text = f"Some analyses predict {scenario['hypotheses'][other_idx].lower()}, but the race remains close."
            else:
                # Generic evidence
                if np.random.rand() < confidence:
                    evidence_text = f"Evidence suggests {scenario['hypotheses'][true_idx].lower()}."
                else:
                    other_idx = np.random.choice([j for j in range(args.num_hypotheses) if j != true_idx])
                    evidence_text = f"Some indicators point to {scenario['hypotheses'][other_idx].lower()}, but it's uncertain."
            
            # Process the evidence text through the model
            with torch.no_grad():
                inputs = tokenizer(evidence_text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                
                if args.model_type == "hf":
                    outputs = model(**inputs, output_hidden_states=True)
                    # Get hidden state from the last layer
                    hidden_states = outputs.hidden_states[-1]
                    # Take the mean across tokens
                    evidence_hidden = hidden_states.mean(dim=1)
                    
                    # Get token probabilities for uncertainty calculation
                    logits = outputs.logits
                    probs = torch.softmax(logits[:, -1], dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    
                elif args.model_type == "mamba":
                    # This assumes the Mamba model has a method to access hidden states
                    # Exact implementation depends on your Mamba wrapper
                    outputs = model.forward(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                    # Extract hidden states - implementation depends on your model's output structure
                    if isinstance(outputs, tuple) and len(outputs) > 1:
                        hidden_states = outputs[1]  # Assuming second element is hidden_states
                    elif hasattr(outputs, 'hidden_states'):
                        hidden_states = outputs.hidden_states
                    else:
                        raise ValueError("Could not extract hidden states from Mamba model")
                    
                    # Take the mean across tokens
                    evidence_hidden = hidden_states.mean(dim=1)
                    
                    # Calculate entropy from logits if available
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        probs = torch.softmax(logits[:, -1], dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    else:
                        # Use a placeholder entropy
                        entropy = torch.tensor([0.5], device=args.device) 
            
            # Resize evidence hidden to match required hidden_dim
            if evidence_hidden.shape[1] != args.hidden_dim:
                if evidence_hidden.shape[1] > args.hidden_dim:
                    # Truncate
                    evidence_hidden = evidence_hidden[:, :args.hidden_dim]
                else:
                    # Pad with zeros
                    padding = torch.zeros(1, args.hidden_dim - evidence_hidden.shape[1], device=args.device)
                    evidence_hidden = torch.cat([evidence_hidden, padding], dim=1)
            
            # Store the evidence
            evidence_sequences[i, t] = evidence_hidden
            
            # Calculate likelihoods for each hypothesis
            likelihoods = torch.ones(args.num_hypotheses, device=args.device)
            
            # True hypothesis gets higher likelihood
            for h in range(args.num_hypotheses):
                if h == true_idx:
                    # True hypothesis: likelihood increases with step (more evidence)
                    # Higher confidence = higher likelihood ratio
                    likelihoods[h] = 1.0 + confidence * (1.0 + t * 0.2)
                else:
                    # Other hypotheses: lower likelihood, depends on confidence
                    likelihoods[h] = 1.0 + (1.0 - confidence) * 0.5
            
            # Add some noise to likelihoods
            likelihoods += torch.randn_like(likelihoods) * 0.1
            likelihoods = likelihoods.clamp(min=0.1)  # Ensure positive
            
            # Apply Bayes' rule
            posterior = prior * likelihoods
            posterior = posterior / posterior.sum()  # Normalize
            
            # Store the posterior
            posterior_probs[i, t] = posterior
            
            # Update prior for next step
            prior = posterior
    
    logger.info(f"Generated {args.num_samples} model-based evidence sequences")
    logger.info(f"Evidence shape: {evidence_sequences.shape}")
    logger.info(f"Posterior shape: {posterior_probs.shape}")
    
    return evidence_sequences, posterior_probs


def save_data(evidence_sequences, posterior_probs, args):
    """
    Save the generated data to disk.
    
    Args:
        evidence_sequences: Tensor of evidence sequences
        posterior_probs: Tensor of posterior probabilities
        args: Command-line arguments
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into train and validation sets
    num_samples = len(evidence_sequences)
    indices = torch.randperm(num_samples)
    
    train_size = int(num_samples * args.train_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_evidence = evidence_sequences[train_indices]
    train_posteriors = posterior_probs[train_indices]
    
    val_evidence = evidence_sequences[val_indices]
    val_posteriors = posterior_probs[val_indices]
    
    # Create data dictionaries
    train_data = {
        'evidence_sequences': train_evidence,
        'posterior_probs': train_posteriors,
        'mode': args.mode,
        'num_hypotheses': args.num_hypotheses,
        'hidden_dim': args.hidden_dim,
        'sequence_length': args.sequence_length
    }
    
    val_data = {
        'evidence_sequences': val_evidence,
        'posterior_probs': val_posteriors,
        'mode': args.mode,
        'num_hypotheses': args.num_hypotheses,
        'hidden_dim': args.hidden_dim,
        'sequence_length': args.sequence_length
    }
    
    # Add model info if model-based
    if args.mode == 'model':
        if args.model_type == 'hf':
            model_info = args.model_name_or_path
        else:
            model_info = f"mamba-{args.mamba_size}"
        
        train_data['model_info'] = model_info
        val_data['model_info'] = model_info
    
    # Save data
    train_path = output_dir / "train_bayesian_data.pt"
    val_path = output_dir / "val_bayesian_data.pt"
    
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    
    # Save paths to a reference file
    with open(output_dir / "data_paths.txt", "w") as f:
        f.write(f"Training data: {train_path.absolute()}\n")
        f.write(f"Validation data: {val_path.absolute()}\n")
        f.write(f"Number of training samples: {len(train_evidence)}\n")
        f.write(f"Number of validation samples: {len(val_evidence)}\n")
        f.write(f"Hidden dimension: {args.hidden_dim}\n")
        f.write(f"Number of hypotheses: {args.num_hypotheses}\n")
        f.write(f"Sequence length: {args.sequence_length}\n")
        f.write(f"Generation mode: {args.mode}\n")
    
    logger.info(f"Saved {len(train_evidence)} training samples to {train_path}")
    logger.info(f"Saved {len(val_evidence)} validation samples to {val_path}")
    logger.info(f"Saved data paths to {output_dir / 'data_paths.txt'}")


def test_bayesian_module(evidence_sequences, posterior_probs, args):
    """
    Test a randomly initialized BayesianInferenceModule on the generated data.
    
    This is just a sanity check to ensure the data format is compatible with the module.
    
    Args:
        evidence_sequences: Generated evidence sequences
        posterior_probs: Generated posterior probabilities
        args: Command-line arguments
    """
    logger.info("Testing BayesianInferenceModule on generated data...")
    
    # Create a BayesianInferenceModule
    module = BayesianInferenceModule(
        hidden_dim=args.hidden_dim,
        belief_dim=args.num_hypotheses,
        bit_linear=False  # Use standard linear layers for testing
    ).to(args.device)
    
    # Test on a small subset of the data
    num_test = min(5, len(evidence_sequences))
    test_evidence = evidence_sequences[:num_test]
    test_posterior = posterior_probs[:num_test]
    
    # Process the evidence sequences
    with torch.no_grad():
        for i in range(num_test):
            # Get a single sequence
            sequence = test_evidence[i].unsqueeze(1)  # [seq_len, 1, hidden_dim]
            true_posteriors = test_posterior[i]  # [seq_len, num_hypotheses]
            
            # Process through the module
            final_belief, belief_trajectory = module.infer_posterior(sequence)
            
            logger.info(f"Sample {i+1}/{num_test}:")
            logger.info(f"  Belief trajectory shape: {belief_trajectory.shape}")
            logger.info(f"  True posterior at final step: {true_posteriors[-1].cpu().numpy()}")
            logger.info(f"  Module belief at final step: {belief_trajectory[-1, 0].cpu().numpy()}")
    
    logger.info("BayesianInferenceModule test completed")


def main():
    """Main execution function."""
    args = parse_args()
    set_seed(args.seed)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    try:
        # Generate data based on selected mode
        if args.mode == 'synthetic':
            evidence_sequences, posterior_probs = generate_synthetic_evidence_data(args)
        elif args.mode == 'model':
            evidence_sequences, posterior_probs = generate_model_based_evidence(args)
        elif args.mode == 'toy_gaussian':
            evidence_sequences, posterior_probs = generate_toy_gaussian_dataset(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        # Test the BayesianInferenceModule
        test_bayesian_module(evidence_sequences, posterior_probs, args)
        
        # Save the data
        save_data(evidence_sequences, posterior_probs, args)
        
        logger.info("Data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
