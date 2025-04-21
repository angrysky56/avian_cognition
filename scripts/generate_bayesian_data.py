#!/usr/bin/env python
"""
Generate Bayesian inference training data using HuggingFace datasets.

This script extracts sequential evidence patterns from real datasets and 
creates training data for the Bayesian inference module with proper 
posterior distributions.
"""

import os
import sys
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bayesian_data_gen')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    HAS_HF = True
except ImportError:
    logger.warning("HuggingFace libraries not found. Install with: pip install datasets transformers")
    HAS_HF = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate Bayesian training data")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="multi_nli",
                       help="HuggingFace dataset name to use")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use (train, validation, test)")
    parser.add_argument("--subset", type=str, default=None,
                       help="Dataset subset/config name if applicable")
    
    # Model for embedding extraction
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="HuggingFace model for extracting embeddings")
    
    # Processing options
    parser.add_argument("--seq_length", type=int, default=5,
                       help="Length of evidence sequences to generate")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of samples to generate (0 for all)")
    parser.add_argument("--hidden_dim", type=int, default=768,
                       help="Hidden dimension for extracted states")
    parser.add_argument("--num_hypotheses", type=int, default=3,
                       help="Number of hypotheses in Bayesian inference tasks")
    
    # Output options
    parser.add_argument("--output_dir", type=str, 
                       default="/home/ty/Repositories/ai_workspace/avian_cognition/data/bayesian",
                       help="Directory to save generated data")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of data to use for training vs validation")
    
    # Runtime options
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for embedding extraction")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (fewer samples, more logging)")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_model_and_tokenizer(args):
    """Load model and tokenizer for embedding extraction."""
    logger.info(f"Loading model: {args.model}")
    
    # Use BERT base instead of DeBERTa to avoid tokenizer compatibility issues
    # If the user specified model fails, we'll fall back to BERT
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = AutoModel.from_pretrained(args.model).to(args.device)
    except Exception as e:
        logger.warning(f"Failed to load specified model: {e}")
        logger.info("Falling back to BERT-base-uncased which has better tokenizer compatibility")
        
        fallback_model = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(fallback_model, use_fast=True)
        model = AutoModel.from_pretrained(fallback_model).to(args.device)
    
    model.eval()
    
    return model, tokenizer


def load_dataset_data(args):
    """Load dataset from HuggingFace."""
    logger.info(f"Loading dataset: {args.dataset}, split: {args.split}")
    
    if args.subset:
        dataset = load_dataset(args.dataset, args.subset, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    
    # For debugging, limit to a small sample
    if args.debug:
        dataset = dataset.select(range(min(100, len(dataset))))
    
    # Limit to requested number of samples
    if args.num_samples > 0 and args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))
    
    logger.info(f"Loaded {len(dataset)} examples from dataset")
    return dataset


def create_nli_sequential_data(model, tokenizer, dataset, args):
    """
    Create sequential Bayesian data from NLI dataset.
    
    For NLI datasets, each premise-hypothesis pair becomes a step in
    sequential inference. We create sequences where multiple premises
    provide evidence about a hypothesis.
    """
    logger.info("Creating sequential Bayesian data from NLI dataset...")
    
    # Group examples by label to create sequences with mixed evidence
    examples_by_label = defaultdict(list)
    for i, example in enumerate(dataset):
        if 'label' in example:
            # Store index as Python int to avoid numpy.int64 issues later
            examples_by_label[example['label']].append(int(i))
    
    # Create sequences and extract embeddings
    all_sequences = []
    all_evidence = []
    all_posteriors = []
    
    # Process in batches to extract embeddings efficiently
    for batch_start in tqdm(range(0, args.num_samples, args.batch_size)):
        batch_end = min(batch_start + args.batch_size, args.num_samples)
        batch_size = batch_end - batch_start
        
        # Create a batch of sequences
        batch_texts = []
        batch_labels = []
        
        for seq_idx in range(batch_start, batch_end):
            # Randomly choose a target label/hypothesis
            target_label = np.random.randint(0, 3)  # Assuming 3 labels: entailment, neutral, contradiction
            
            # Create a sequence with mixed evidence
            seq_indices = []
            # Add mostly evidence supporting the target hypothesis
            for _ in range(args.seq_length):
                if np.random.random() < 0.7:  # 70% chance to add supporting evidence
                    # Convert numpy.int64 to Python int
                    idx = int(np.random.choice(examples_by_label[target_label]))
                else:  # 30% chance to add evidence for another label
                    other_label = np.random.choice([l for l in examples_by_label.keys() if l != target_label])
                    # Convert numpy.int64 to Python int
                    idx = int(np.random.choice(examples_by_label[other_label]))
                seq_indices.append(idx)
            
            # Extract premise texts for the sequence - all indices should be Python ints now
            sequence_texts = [dataset[idx]['premise'] for idx in seq_indices]
            batch_texts.extend(sequence_texts)
            batch_labels.append(target_label)
        
        # Get embeddings for all texts in the batch
        with torch.no_grad():
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                              max_length=128, return_tensors="pt").to(args.device)
            outputs = model(**inputs)
            # Use CLS token embedding as the representation
            embeddings = outputs.last_hidden_state[:, 0].cpu()
        
        # Reshape embeddings back into sequences
        embeddings = embeddings.reshape(batch_size, args.seq_length, -1)
        
        # Create posterior distributions (probability distributions over hypotheses)
        # Start with uniform prior
        posteriors = torch.zeros(batch_size, args.seq_length, args.num_hypotheses)
        
        # Update posterior after each piece of evidence (simplified Bayesian update)
        for i in range(batch_size):
            posterior = torch.ones(args.num_hypotheses) / args.num_hypotheses  # Uniform prior
            
            for j in range(args.seq_length):
                # Compute likelihoods (simplified example)
                # In a real scenario, these would be learned or derived from data
                likelihood = torch.ones(args.num_hypotheses) * 0.1  # Small baseline probability
                likelihood[batch_labels[i]] = 0.7  # Higher probability for true hypothesis
                
                # Bayes' rule (simplified)
                unnormalized = posterior * likelihood
                posterior = unnormalized / unnormalized.sum()
                
                # Store the posterior for this step
                posteriors[i, j] = posterior
        
        # Add to collected data
        all_sequences.append(embeddings)
        all_posteriors.append(posteriors)
    
    # Concatenate all batches
    all_sequences = torch.cat(all_sequences, dim=0)
    all_posteriors = torch.cat(all_posteriors, dim=0)
    
    logger.info(f"Created {len(all_sequences)} sequences with shape {all_sequences.shape}")
    
    return {
        'sequences': all_sequences,
        'posteriors': all_posteriors,
        'hidden_dim': all_sequences.shape[-1],
        'num_hypotheses': args.num_hypotheses,
        'model_info': args.model
    }


def save_data(data, args):
    """Save generated data to output directory."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into train and validation sets
    num_samples = len(data['sequences'])
    indices = torch.randperm(num_samples)
    
    train_size = int(num_samples * args.train_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create training data
    train_data = {
        'sequences': data['sequences'][train_indices],
        'posteriors': data['posteriors'][train_indices],
        'hidden_dim': data['hidden_dim'],
        'num_hypotheses': data['num_hypotheses'],
        'model_info': data['model_info'],
        'dataset': args.dataset
    }
    
    # Create validation data
    val_data = {
        'sequences': data['sequences'][val_indices],
        'posteriors': data['posteriors'][val_indices],
        'hidden_dim': data['hidden_dim'],
        'num_hypotheses': data['num_hypotheses'],
        'model_info': data['model_info'],
        'dataset': args.dataset
    }
    
    # Save data
    train_path = output_dir / "train_bayesian_data.pt"
    val_path = output_dir / "val_bayesian_data.pt"
    
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    
    # Save paths to a reference file
    with open(output_dir / "data_paths.txt", "w") as f:
        f.write(f"Training data: {train_path.absolute()}\n")
        f.write(f"Validation data: {val_path.absolute()}\n")
        f.write(f"Number of training samples: {len(train_indices)}\n")
        f.write(f"Number of validation samples: {len(val_indices)}\n")
        f.write(f"Hidden dimension: {data['hidden_dim']}\n")
        f.write(f"Number of hypotheses: {data['num_hypotheses']}\n")
        f.write(f"Model: {data['model_info']}\n")
        f.write(f"Dataset: {args.dataset}\n")
    
    logger.info(f"Saved {len(train_indices)} training samples to {train_path}")
    logger.info(f"Saved {len(val_indices)} validation samples to {val_path}")
    logger.info(f"Saved data reference info to {output_dir / 'data_paths.txt'}")


def main():
    """Main execution function."""
    args = parse_args()
    set_seed(args.seed)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    try:
        if not HAS_HF:
            logger.error("HuggingFace libraries are required but not installed.")
            logger.error("Install with: pip install datasets transformers")
            sys.exit(1)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args)
        
        # Load dataset
        dataset = load_dataset_data(args)
        
        # Process dataset based on its type
        if "nli" in args.dataset.lower():
            data = create_nli_sequential_data(model, tokenizer, dataset, args)
        else:
            logger.error(f"Unsupported dataset type: {args.dataset}")
            logger.error("Currently supported datasets: NLI datasets (multi_nli, mnli, etc.)")
            sys.exit(1)
        
        # Save data
        save_data(data, args)
        
        logger.info("Data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()