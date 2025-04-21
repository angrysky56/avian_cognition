#!/usr/bin/env python
"""
Generate Metacognition Training Data

This script uses a pretrained Mamba model (or other language model) to generate 
hidden states and correctness labels for training the metacognition module.

The script:
1. Loads a pretrained model
2. Processes text data (e.g., WikiText, C4)
3. For each token prediction, extracts hidden states and whether prediction was correct
4. Saves these pairs as training data for the metacognition module
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('metacognition_data_gen')

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Try to import required modules
try:
    from datasets import load_dataset
    has_datasets = True
except ImportError:
    has_datasets = False
    logger.warning("HuggingFace datasets not installed. Install with: pip install datasets")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    has_transformers = True
except ImportError:
    has_transformers = False
    logger.warning("HuggingFace transformers not installed. Install with: pip install transformers")

# Attempt to import mamba-related modules
try:
    from src.core.mamba_integration import create_mini_model, create_small_model
    has_mamba = True
except ImportError:
    has_mamba = False
    logger.warning("Mamba modules not found. Will attempt to use HuggingFace models instead.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate metacognition training data")
    
    # Data source arguments
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["wikitext", "c4", "custom"],
                        help="Dataset to use for generating data")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-v1",
                        help="Dataset configuration name")
    parser.add_argument("--dataset_split", type=str, default="train[:10%]",
                        help="Dataset split to use (default: small portion of training set)")
    parser.add_argument("--custom_data_path", type=str, default=None,
                        help="Path to custom text data (only used with --dataset=custom)")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="mamba",
                        choices=["mamba", "hf"],
                        help="Type of model to use")
    parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/pythia-70m",
                        help="Model name or path for HuggingFace model (only used with --model_type=hf)")
    parser.add_argument("--mamba_size", type=str, default="mini",
                        choices=["mini", "small"],
                        help="Size of Mamba model to use (only used with --model_type=mamba)")
    
    # Processing arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for processing")
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride for sliding window processing")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of samples to generate (0 for all)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="data/metacognition",
                        help="Directory to save generated data")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio of data to use for training vs validation")
    
    # Runtime arguments
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for running the model")
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


def load_model(args):
    """Load language model based on specified arguments."""
    if args.model_type == "mamba":
        if not has_mamba:
            raise ImportError("Mamba modules not found. Cannot use model_type='mamba'.")
        
        logger.info(f"Loading Mamba-{args.mamba_size} model...")
        if args.mamba_size == "mini":
            model = create_mini_model(quantize=False)
        else:  # small
            model = create_small_model(quantize=False)
        
        model = model.to(args.device)
        
        # Get tokenizer from huggingface - assuming compatible tokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return model, tokenizer
    
    elif args.model_type == "hf":
        if not has_transformers:
            raise ImportError("HuggingFace transformers not found. Cannot use model_type='hf'.")
        
        logger.info(f"Loading HuggingFace model: {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        model = model.to(args.device)
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return model, tokenizer
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def load_dataset_data(args):
    """Load text dataset for processing."""
    if not has_datasets:
        raise ImportError("HuggingFace datasets not installed. Cannot load dataset.")
    
    if args.dataset == "wikitext":
        logger.info(f"Loading WikiText dataset: {args.dataset_config}, split: {args.dataset_split}")
        dataset = load_dataset("wikitext", args.dataset_config, split=args.dataset_split)
    
    elif args.dataset == "c4":
        logger.info(f"Loading C4 dataset, split: {args.dataset_split}")
        dataset = load_dataset("c4", "en", split=args.dataset_split)
    
    elif args.dataset == "custom":
        if args.custom_data_path is None:
            raise ValueError("Must provide --custom_data_path when using --dataset=custom")
        
        logger.info(f"Loading custom dataset from: {args.custom_data_path}")
        # Implementation depends on the format of the custom data
        # For now, assume it's a text file with one document per line
        
        from datasets import Dataset as HFDataset
        
        with open(args.custom_data_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        dataset = HFDataset.from_dict({"text": texts})
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # For debugging, limit to a small sample
    if args.debug:
        dataset = dataset.select(range(min(100, len(dataset))))
    
    # Limit to requested number of samples
    if args.num_samples > 0 and args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))
    
    logger.info(f"Loaded {len(dataset)} examples from dataset")
    return dataset


def extract_hidden_states_and_correctness(model, tokenizer, dataset, args):
    """
    Extract hidden states and correctness from model predictions.
    
    This function:
    1. Tokenizes text examples
    2. Passes them through the model
    3. For each token, extracts:
       - The hidden state before prediction
       - Whether the prediction was correct (compared to actual next token)
    """
    hidden_states_list = []
    correctness_list = []
    
    # Processing function depends on model type
    if args.model_type == "mamba":
        # For Mamba models, we need to handle extraction differently
        # This is placeholder logic - adjust based on actual Mamba implementation
        
        def process_batch(batch_texts):
            # Convert batch to tensors
            encodings = tokenizer(batch_texts, padding=True, truncation=True, 
                                  max_length=args.max_length, return_tensors="pt")
            input_ids = encodings.input_ids.to(args.device)
            attention_mask = encodings.attention_mask.to(args.device)
            
            # Prepare target tokens (shifted input_ids)
            # For each sequence, target tokens are input tokens shifted one position right
            target_ids = input_ids.clone()
            target_ids[:, :-1] = input_ids[:, 1:]  # Shift left
            
            # Process through model
            with torch.no_grad():
                # Run forward pass
                outputs = model(input_ids, attention_mask=attention_mask, 
                               output_hidden_states=True)
                
                # Get hidden states (usually the last layer)
                # This format depends on the specific Mamba implementation
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    if isinstance(outputs[1], tuple):  # If hidden_states is a tuple
                        hidden_states = outputs[1][-1]  # Last layer
                    else:
                        hidden_states = outputs[1]  # Assuming outputs[1] is hidden_states
                elif hasattr(outputs, 'hidden_states'):
                    hidden_states = outputs.hidden_states[-1]  # Last layer
                else:
                    raise ValueError("Could not extract hidden states from model outputs")
                
                # Get logits
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0]  # Assume first element is logits
                
                # For each position, determine if prediction was correct
                predictions = logits.argmax(dim=-1)
                
                # For each sequence in the batch
                for i in range(input_ids.size(0)):
                    seq_length = attention_mask[i].sum().item()
                    
                    # For each token position (except the last one, which has no target)
                    for j in range(seq_length - 1):
                        # Get hidden state for this position
                        state = hidden_states[i, j].cpu()
                        
                        # Check if prediction was correct
                        pred = predictions[i, j].item()
                        target = target_ids[i, j].item()
                        correct = (pred == target)
                        
                        # Add to lists
                        hidden_states_list.append(state)
                        correctness_list.append(float(correct))
            
            return len(hidden_states_list)
    
    elif args.model_type == "hf":
        # For HuggingFace models
        
        def process_batch(batch_texts):
            # Convert batch to tensors
            encodings = tokenizer(batch_texts, padding=True, truncation=True, 
                                  max_length=args.max_length, return_tensors="pt")
            input_ids = encodings.input_ids.to(args.device)
            attention_mask = encodings.attention_mask.to(args.device)
            
            # Prepare target tokens (shifted input_ids)
            target_ids = input_ids.clone()
            target_ids[:, :-1] = input_ids[:, 1:]  # Shift left
            
            # Process through model
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, 
                               output_hidden_states=True)
                
                # Get hidden states (last layer)
                hidden_states = outputs.hidden_states[-1]
                
                # Get logits
                logits = outputs.logits
                
                # For each position, determine if prediction was correct
                predictions = logits.argmax(dim=-1)
                
                # For each sequence in the batch
                for i in range(input_ids.size(0)):
                    seq_length = attention_mask[i].sum().item()
                    
                    # For each token position (except the last one, which has no target)
                    for j in range(seq_length - 1):
                        # Get hidden state for this position
                        state = hidden_states[i, j].cpu()
                        
                        # Check if prediction was correct
                        pred = predictions[i, j].item()
                        target = target_ids[i, j].item()
                        correct = (pred == target)
                        
                        # Add to lists
                        hidden_states_list.append(state)
                        correctness_list.append(float(correct))
            
            return len(hidden_states_list)
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Create dataloader for processing batches
    dataloader = DataLoader(
        dataset["text"] if "text" in dataset.column_names else dataset,
        batch_size=args.batch_size,
        collate_fn=lambda x: x  # Just return the batch as is
    )
    
    # Process batches
    total_samples = 0
    pbar = tqdm(dataloader, desc="Processing")
    for batch in pbar:
        num_processed = process_batch(batch)
        total_samples += num_processed
        pbar.set_postfix({"samples": total_samples})
        
        # For debugging, process fewer batches
        if args.debug and total_samples >= 1000:
            logger.info("Debug mode: stopping after reaching 1000 samples")
            break
        
        # Stop if reached requested number of samples
        if args.num_samples > 0 and total_samples >= args.num_samples:
            logger.info(f"Reached requested {args.num_samples} samples")
            break
    
    # Convert lists to tensors
    hidden_states_tensor = torch.stack(hidden_states_list)
    correctness_tensor = torch.tensor(correctness_list).unsqueeze(1)  # Add dimension for consistency
    
    logger.info(f"Extracted {len(hidden_states_list)} samples with hidden_dim={hidden_states_tensor.shape[1]}")
    logger.info(f"Percentage correct: {correctness_tensor.mean().item()*100:.2f}%")
    
    return hidden_states_tensor, correctness_tensor


def save_data(hidden_states, correctness, args):
    """Save extracted data to output directory."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into train and validation sets
    num_samples = len(hidden_states)
    indices = torch.randperm(num_samples)
    
    train_size = int(num_samples * args.train_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_hidden_states = hidden_states[train_indices]
    train_correctness = correctness[train_indices]
    
    val_hidden_states = hidden_states[val_indices]
    val_correctness = correctness[val_indices]
    
    # Create data dictionaries
    train_data = {
        'hidden_states': train_hidden_states,
        'correctness': train_correctness,
        'model_type': args.model_type,
        'model_info': args.model_name_or_path if args.model_type == 'hf' else f"mamba-{args.mamba_size}",
        'dataset': args.dataset,
        'hidden_dim': train_hidden_states.shape[1]
    }
    
    val_data = {
        'hidden_states': val_hidden_states,
        'correctness': val_correctness,
        'model_type': args.model_type,
        'model_info': args.model_name_or_path if args.model_type == 'hf' else f"mamba-{args.mamba_size}",
        'dataset': args.dataset,
        'hidden_dim': val_hidden_states.shape[1]
    }
    
    # Save data
    train_path = output_dir / "train_metacognition_data.pt"
    val_path = output_dir / "val_metacognition_data.pt"
    
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    
    # Save paths to a reference file
    with open(output_dir / "data_paths.txt", "w") as f:
        f.write(f"Training data: {train_path.absolute()}\n")
        f.write(f"Validation data: {val_path.absolute()}\n")
        f.write(f"Number of training samples: {len(train_hidden_states)}\n")
        f.write(f"Number of validation samples: {len(val_hidden_states)}\n")
        f.write(f"Hidden dimension: {train_hidden_states.shape[1]}\n")
        f.write(f"Percentage correct (train): {train_correctness.mean().item()*100:.2f}%\n")
        f.write(f"Percentage correct (val): {val_correctness.mean().item()*100:.2f}%\n")
    
    logger.info(f"Saved {len(train_hidden_states)} training samples to {train_path}")
    logger.info(f"Saved {len(val_hidden_states)} validation samples to {val_path}")
    logger.info(f"Saved data paths to {output_dir / 'data_paths.txt'}")


def main():
    """Main execution function."""
    args = parse_args()
    set_seed(args.seed)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    try:
        # Load model
        model, tokenizer = load_model(args)
        
        # Load dataset
        dataset = load_dataset_data(args)
        
        # Extract hidden states and correctness
        hidden_states, correctness = extract_hidden_states_and_correctness(
            model, tokenizer, dataset, args
        )
        
        # Save data
        save_data(hidden_states, correctness, args)
        
        logger.info("Data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
