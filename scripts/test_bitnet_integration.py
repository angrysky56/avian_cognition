#!/usr/bin/env python
"""
BitNet Model Integration Tester

This script tests the integration of pre-existing BitNet models with our avian cognition
architecture. It demonstrates loading and utilizing the BitNet models directly rather
than reimplementing BitNet from scratch.
"""

import os
import sys
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import BitNet integration from our project
from src.core.bitnet_integration import BitNetModel

# Try to import transformers for comparison
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers package not found. Will only test BitNet C++ integration.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BitNet Model Integration Test")
    
    # BitNet model selection
    parser.add_argument("--model", type=str, default="microsoft/bitnet-b1.58-2B-4T",
                        help="HuggingFace BitNet model ID or path")
    
    # Test options
    parser.add_argument("--test_prompts", type=str, default=None,
                        help="Path to a text file with test prompts, one per line")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of inference runs for timing")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--compare_hf", action="store_true",
                        help="Compare with HuggingFace implementation (slower)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="outputs/bitnet_integration",
                        help="Directory to save test results")
    
    return parser.parse_args()

def load_test_prompts(args):
    """Load test prompts from file or use defaults."""
    default_prompts = [
        "Explain how avian cognition works in simple terms.",
        "What are the key differences between birds and mammals in terms of neural architecture?",
        "Describe the process of Bayesian inference in a biological context."
    ]
    
    if args.test_prompts and os.path.exists(args.test_prompts):
        print(f"Loading test prompts from {args.test_prompts}")
        with open(args.test_prompts, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        if not prompts:
            print("Warning: No valid prompts found in file. Using defaults.")
            prompts = default_prompts
    else:
        print("Using default test prompts")
        prompts = default_prompts
    
    return prompts

def test_bitnet_cpp_integration(model_id, prompts, max_tokens, num_runs):
    """Test BitNet C++ integration using our wrapper."""
    try:
        print(f"Loading BitNet model: {model_id}")
        model = BitNetModel(model_id)
        
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Time the generation
            times = []
            outputs = []
            
            # Run multiple times for timing
            for run in range(num_runs):
                start_time = time.time()
                output = model.generate(prompt, max_tokens=max_tokens)
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)  # Convert to ms
                outputs.append(output)
            
            # Calculate metrics
            avg_time = np.mean(times)
            std_time = np.std(times)
            tokens_per_sec = max_tokens / (avg_time / 1000)
            
            print(f"  Generation Time: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"  Tokens/second: {tokens_per_sec:.2f}")
            print(f"  Output (Run 1): {outputs[0][:100]}...")
            
            results.append({
                "prompt": prompt,
                "output": outputs[0],
                "avg_time_ms": avg_time,
                "std_time_ms": std_time,
                "tokens_per_sec": tokens_per_sec
            })
        
        return {
            "model": model_id,
            "backend": "BitNet C++",
            "results": results
        }
        
    except Exception as e:
        print(f"Error testing BitNet C++ integration: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_hf_bitnet(model_id, prompts, max_tokens, num_runs):
    """Test BitNet using HuggingFace transformers."""
    if not HAS_TRANSFORMERS:
        print("HuggingFace transformers not available. Skipping HF comparison.")
        return None
    
    try:
        print(f"Loading BitNet model via HuggingFace: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        device = next(model.parameters()).device
        print(f"Model loaded on device: {device}")
        
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Prepare input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Time the generation
            times = []
            output_texts = []
            
            # Run multiple times for timing
            for run in range(num_runs):
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                end_time = time.time()
                
                # Decode output
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                times.append((end_time - start_time) * 1000)  # Convert to ms
                output_texts.append(output_text)
            
            # Calculate metrics
            avg_time = np.mean(times)
            std_time = np.std(times)
            tokens_per_sec = max_tokens / (avg_time / 1000) if avg_time > 0 else 0
            
            print(f"  Generation Time: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"  Tokens/second: {tokens_per_sec:.2f}")
            print(f"  Output (Run 1): {output_texts[0][:100]}...")
            
            results.append({
                "prompt": prompt,
                "output": output_texts[0],
                "avg_time_ms": avg_time,
                "std_time_ms": std_time,
                "tokens_per_sec": tokens_per_sec
            })
        
        return {
            "model": model_id,
            "backend": "HuggingFace",
            "results": results
        }
        
    except Exception as e:
        print(f"Error testing HuggingFace BitNet: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_results(all_results, output_dir):
    """Save test results to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results in JSON format
    import json
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = output_dir / f"bitnet_results_{timestamp}.json"
    
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Create a summary text file
    summary_path = output_dir / f"bitnet_summary_{timestamp}.txt"
    
    with open(summary_path, "w") as f:
        f.write("BitNet Integration Test Summary\n")
        f.write("=============================\n\n")
        
        for result in all_results:
            if result is None:
                continue
                
            f.write(f"Model: {result['model']}\n")
            f.write(f"Backend: {result['backend']}\n")
            f.write("\nPerformance Summary:\n")
            
            avg_tokens_per_sec = np.mean([r['tokens_per_sec'] for r in result['results']])
            f.write(f"  Average Tokens/second: {avg_tokens_per_sec:.2f}\n")
            
            avg_time = np.mean([r['avg_time_ms'] for r in result['results']])
            f.write(f"  Average Generation Time: {avg_time:.2f} ms\n\n")
            
            f.write("Sample Output:\n")
            sample_result = result['results'][0]
            f.write(f"  Prompt: {sample_result['prompt'][:100]}...\n")
            f.write(f"  Output: {sample_result['output'][:500]}...\n\n")
            
            f.write("-" * 50 + "\n\n")
    
    print(f"Summary saved to {summary_path}")
    
    # Create simple report in Markdown
    report_path = output_dir / f"bitnet_report_{timestamp}.md"
    
    with open(report_path, "w") as f:
        f.write("# BitNet Integration Test Report\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Performance table
        f.write("## Performance Summary\n\n")
        f.write("| Model | Backend | Tokens/sec | Avg Time (ms) |\n")
        f.write("|-------|---------|------------|---------------|\n")
        
        for result in all_results:
            if result is None:
                continue
                
            avg_tokens_per_sec = np.mean([r['tokens_per_sec'] for r in result['results']])
            avg_time = np.mean([r['avg_time_ms'] for r in result['results']])
            
            f.write(f"| {result['model'].split('/')[-1]} | {result['backend']} | ")
            f.write(f"{avg_tokens_per_sec:.2f} | {avg_time:.2f} |\n")
        
        f.write("\n## Sample Outputs\n\n")
        
        for result in all_results:
            if result is None:
                continue
                
            f.write(f"### Model: {result['model'].split('/')[-1]} ({result['backend']})\n\n")
            
            sample_result = result['results'][0]
            f.write(f"**Prompt:** {sample_result['prompt']}\n\n")
            f.write(f"**Output:**\n\n{sample_result['output'][:500]}...\n\n")
            f.write("---\n\n")
    
    print(f"Report saved to {report_path}")
    
def main():
    """Main testing function."""
    args = parse_args()
    
    # Load test prompts
    prompts = load_test_prompts(args)
    
    all_results = []
    
    # Test BitNet C++ integration
    print("\n=== Testing BitNet C++ Integration ===\n")
    cpp_results = test_bitnet_cpp_integration(
        args.model, 
        prompts, 
        args.max_tokens, 
        args.num_runs
    )
    if cpp_results:
        all_results.append(cpp_results)
    
    # Test HuggingFace BitNet integration
    if args.compare_hf:
        print("\n=== Testing HuggingFace BitNet Integration ===\n")
        hf_results = test_hf_bitnet(
            args.model, 
            prompts, 
            args.max_tokens, 
            args.num_runs
        )
        if hf_results:
            all_results.append(hf_results)
    
    # Save results
    if all_results:
        save_results(all_results, args.output_dir)
        print("\nTesting completed successfully!")
    else:
        print("\nNo successful test results to save.")
    

if __name__ == "__main__":
    main()
