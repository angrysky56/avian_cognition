"""
BitNet C++ Python Wrapper

This module provides Python bindings for the BitNet C++ command-line tools, 
enabling interaction with pre-compiled BitNet binaries for inference and 
quantization from Python.

It allows loading BitNet models in GGUF format, running inference via llama-cli,
and quantizing GGUF models via llama-quantize.
"""

import os
import sys
import torch
import numpy as np
import subprocess
from pathlib import Path
import ctypes
from typing import List, Dict, Optional, Union, Tuple

# --- Configuration ---
# Attempt to get BITNET_ROOT from environment variable, otherwise use default
DEFAULT_BITNET_ROOT = "/home/ty/Repositories/BitNet" 
BITNET_ROOT = os.environ.get("BITNET_ROOT", DEFAULT_BITNET_ROOT)

# Check if the determined BITNET_ROOT exists
if not os.path.isdir(BITNET_ROOT):
    # Try the default path if the environment variable path didn't exist
    if BITNET_ROOT != DEFAULT_BITNET_ROOT and os.path.isdir(DEFAULT_BITNET_ROOT):
        print(f"Warning: BITNET_ROOT environment variable path '{BITNET_ROOT}' not found.")
        print(f"Falling back to default path: '{DEFAULT_BITNET_ROOT}'")
        BITNET_ROOT = DEFAULT_BITNET_ROOT
    elif not os.path.isdir(DEFAULT_BITNET_ROOT):
         raise ImportError(
            f"BitNet repository not found at environment path ('{BITNET_ROOT}') "
            f"or default path ('{DEFAULT_BITNET_ROOT}'). "
            f"Please set the BITNET_ROOT environment variable or place the repo at the default location."
         )

# Add BitNet repo root to system path if not already present
if BITNET_ROOT not in sys.path:
    sys.path.append(BITNET_ROOT)
# --- End Configuration ---


class BitNetModel:
    """
    Python wrapper for interacting with BitNet C++ binaries (llama-cli, llama-quantize).
    
    Enables loading GGUF models, running inference via llama-cli subprocess, 
    and quantizing GGUF models via llama-quantize subprocess.
    """
    
    def __init__(self, model_path: str, threads: int = 4, ctx_size: int = 2048):
        """
        Initialize BitNet model wrapper using a GGUF file path.
        
        Args:
            model_path (str): Path to the BitNet model (.gguf file). This model
                              must be compatible with the llama-cli binary.
            threads (int): Number of threads for the C++ inference binary to use.
            ctx_size (int): Context size for the C++ inference binary.
            
        Raises:
            FileNotFoundError: If the model file or required BitNet binaries 
                               (llama-cli, llama-quantize) are not found.
        """
        self.model_path = model_path
        self.threads = threads
        self.ctx_size = ctx_size
        
        # Verify model path exists
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"BitNet model file not found at: {self.model_path}")
        
        # Locate required binaries
        self.cli_binary_path = os.path.join(BITNET_ROOT, "build", "bin", "llama-cli")
        self.quantize_binary_path = os.path.join(BITNET_ROOT, "build", "bin", "llama-quantize")
        
        if not os.path.isfile(self.cli_binary_path):
            raise FileNotFoundError(
                f"BitNet inference binary (llama-cli) not found at: {self.cli_binary_path}. "
                f"Ensure BitNet is compiled (e.g., run setup_env.py in {BITNET_ROOT})."
            )
        if not os.path.isfile(self.quantize_binary_path):
             # Warn instead of error, as quantization might not always be needed
            print(f"Warning: BitNet quantization binary (llama-quantize) not found at: "
                  f"{self.quantize_binary_path}. Quantization methods will fail.")
            # raise FileNotFoundError(
            #     f"BitNet quantization binary (llama-quantize) not found at: {self.quantize_binary_path}. "
            #     f"Ensure BitNet is compiled."
            # )
            
        print(f"Initialized BitNetModel wrapper for GGUF model: {model_path}")
        print(f"Using llama-cli binary: {self.cli_binary_path}")
    
    def generate(self, 
                 prompt: str, 
                 n_predict: int = 128, 
                 temperature: float = 0.7, # Default temp often lower for BitNet
                 top_k: int = 40,
                 top_p: float = 0.9,
                 repeat_penalty: float = 1.1,
                 conversation: bool = False) -> str:
        """
        Generate text using the BitNet C++ llama-cli binary via subprocess.
        
        Args:
            prompt (str): Input prompt for text generation.
            n_predict (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature. Lower values (~0.2-0.7) often work well.
            top_k (int): Top-k sampling parameter.
            top_p (float): Top-p (nucleus) sampling parameter.
            repeat_penalty (float): Penalty for repeating tokens.
            conversation (bool): Whether to use llama-cli's conversation mode.
            
        Returns:
            str: Generated text (stdout from llama-cli).
                 Returns an error message string if inference fails.
        """
        # Prepare command arguments for llama-cli
        cmd = [
            self.cli_binary_path,
            '-m', self.model_path,
            '-n', str(n_predict),
            '-t', str(self.threads),
            '-p', prompt,
            '-c', str(self.ctx_size), # Context size
            '--temp', str(temperature),
            '--top-k', str(top_k),
            '--top-p', str(top_p),
            '--repeat-penalty', str(repeat_penalty),
            '-ngl', '0', # Number of GPU layers (set to 0 for CPU, adjust if llama.cpp supports GPU for BitNet)
            '-b', '1', # Batch size (llama-cli might have limitations)
            '--no-mmap', # Often recommended for stability
            # Add other relevant llama-cli parameters if needed
        ]
        
        if conversation:
            # Check llama-cli documentation for the correct conversation flag if needed
            # cmd.append("--interactive") or similar might be required
            print("Warning: Conversation mode flag needs verification for llama-cli.")
            # cmd.append("-cnv") # Assuming -cnv is correct, verify this
        
        print(f"Running llama-cli command: {' '.join(cmd)}")
        
        # Run inference via subprocess
        try:
            # Use Popen for potentially better handling of large outputs later if needed
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                error_message = (
                    f"Error running BitNet inference (llama-cli exited with code {process.returncode}).\n"
                    f"Stderr:\n{stderr}"
                )
                print(error_message)
                return f"Error: Inference failed. Check console logs." # Return error string

            # Typically, llama-cli prints the prompt and then the generation to stdout.
            # We might want to parse this to return only the generated part.
            # Simple approach: Return everything after the prompt.
            # More robust: Need knowledge of llama-cli's exact output format.
            # For now, return the full stdout. A common pattern is the output starts after the prompt ends.
            # Let's assume the output follows the prompt directly.
            if prompt in stdout:
                # Find the end of the last occurrence of the prompt
                last_prompt_end = stdout.rfind(prompt) + len(prompt)
                generated_text = stdout[last_prompt_end:].strip()
            else:
                # If prompt not found (unexpected), return full stdout with a warning
                print("Warning: Prompt not found in llama-cli output. Returning full stdout.")
                generated_text = stdout.strip()
                
            return generated_text
            
        except FileNotFoundError:
             error_message = f"Error: llama-cli binary not found at {self.cli_binary_path}."
             print(error_message)
             return error_message
        except Exception as e:
            error_message = f"An unexpected error occurred during BitNet inference: {e}"
            print(error_message)
            # Include stderr if available from Popen context
            if 'stderr' in locals() and stderr:
                 print(f"Stderr:\n{stderr}")
            return f"Error: {e}"
            
    def quantize_gguf_model(self, src_gguf_path: str, quant_type: str = "i2_s") -> str:
        """
        Quantize an existing GGUF model (e.g., FP32 or FP16) to a BitNet 
        quantization type (e.g., i2_s) using the llama-quantize binary.
        
        Args:
            src_gguf_path (str): Path to the source GGUF model (e.g., f32 gguf).
            quant_type (str): BitNet quantization type (e.g., "i2_s"). 
                              Refer to BitNet/llama.cpp documentation for valid types.
            
        Returns:
            str: Path to the newly created quantized GGUF model.
            
        Raises:
            FileNotFoundError: If the source model or llama-quantize binary is not found.
            RuntimeError: If the quantization process fails.
        """
        # Check if quantize binary exists
        if not os.path.isfile(self.quantize_binary_path):
            raise FileNotFoundError(
                f"Quantize binary (llama-quantize) not found at: {self.quantize_binary_path}. "
                "Cannot quantize model."
            )
            
        # Check if source GGUF exists
        if not os.path.isfile(src_gguf_path):
            raise FileNotFoundError(f"Source GGUF model not found at: {src_gguf_path}")
            
        # Construct destination path
        dest_gguf_path = src_gguf_path.replace(".gguf", f"-{quant_type}.gguf")
        if dest_gguf_path == src_gguf_path: # Avoid overwriting source if naming fails
             dest_gguf_path = f"{src_gguf_path}.{quant_type}.gguf"

        print(f"Quantizing GGUF model: {src_gguf_path}")
        print(f"  Quantization type: {quant_type}")
        print(f"  Output path: {dest_gguf_path}")
        
        # Prepare quantization command
        # NOTE: Verify the exact arguments required by the BitNet/llama.cpp llama-quantize version
        # It might need the quantization type as a number or specific string.
        # Assuming the format: llama-quantize <infile> <outfile> <quant_type> <nthreads>
        cmd = [
            self.quantize_binary_path,
            src_gguf_path,
            dest_gguf_path,
            quant_type.upper(), # Common GGUF convention, verify for BitNet's version
            str(self.threads) # Use specified threads for quantization
        ]
        
        print(f"Running llama-quantize command: {' '.join(cmd)}")
        
        # Run quantization via subprocess
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                 raise RuntimeError(
                    f"GGUF model quantization failed (exit code {process.returncode}).\n"
                    f"Stderr:\n{stderr}"
                 )
                 
            print(f"Quantization successful. Quantized model saved to: {dest_gguf_path}")
            return dest_gguf_path
            
        except FileNotFoundError:
             raise FileNotFoundError(f"Error: llama-quantize binary not found at {self.quantize_binary_path}.")
        except Exception as e:
            # Include stderr if available
            stderr_info = f"\nStderr:\n{stderr}" if 'stderr' in locals() and stderr else ""
            raise RuntimeError(f"An unexpected error occurred during GGUF quantization: {e}{stderr_info}")
            
    @staticmethod
    def convert_pytorch_to_gguf(model: torch.nn.Module, 
                              output_path: str,
                              tokenizer=None) -> str:
        """
        Placeholder for PyTorch to GGUF conversion.

        **CRITICAL:** This function is a placeholder. Actual conversion requires
        using the specific conversion scripts provided by the BitNet or adapted
        llama.cpp repositories (e.g., `convert-hf-to-gguf-bitnet.py`). 
        Directly saving a PyTorch model state_dict is NOT sufficient for GGUF.
        
        Args:
            model: The PyTorch model instance (e.g., AvianMambaModel).
            output_path: Desired path for the output GGUF file.
            tokenizer: The tokenizer associated with the model (needed for GGUF metadata).
            
        Returns:
            str: The provided output_path (as conversion is not performed here).
            
        Raises:
            NotImplementedError: To emphasize that this is a placeholder.
        """
        error_message = (
            "Error: PyTorch to GGUF conversion is NOT implemented in this wrapper. "
            "This complex process requires dedicated scripts from the BitNet/llama.cpp "
            "repositories (like 'convert-hf-to-gguf-bitnet.py' or similar) which handle "
            "weight extraction, tensor mapping, and GGUF format specification. "
            "Please use those external scripts for conversion."
        )
        print(error_message)
        raise NotImplementedError(error_message)
        # Keep the function signature but don't implement conversion here.
        # return output_path # Returning path is misleading
        
    @staticmethod
    def convert_linear_to_bit_linear(module: torch.nn.Module) -> torch.nn.Module:
        """
        Recursively convert PyTorch nn.Linear modules to BitLinear equivalents 
        within a given module.
        
        This relies on an external `convert_linear_to_bit_linear` function,
        presumably located in `src.core.bitnet`, which performs the actual
        in-memory PyTorch layer replacement.

        Args:
            module (torch.nn.Module): The PyTorch module (or model) to convert.
            
        Returns:
            torch.nn.Module: The module with Linear layers replaced by BitLinear.
        """
        try:
            # Assumes this import works and the function exists
            from src.core.bitnet import convert_linear_to_bit_linear 
            print("Attempting in-memory conversion of nn.Linear to BitLinear...")
            return convert_linear_to_bit_linear(module)
        except ImportError:
             print("Error: Could not import 'convert_linear_to_bit_linear' from 'src.core.bitnet'.")
             print("       In-memory conversion requires this function to be implemented.")
             raise # Re-raise the import error
        except Exception as e:
             print(f"Error during BitLinear conversion: {e}")
             raise # Re-raise other exceptions


# --- Helper Functions ---
def check_bitnet_build() -> bool:
    """
    Check if the BitNet C++ inference binary (llama-cli) exists.
    
    Returns:
        bool: True if llama-cli is found in the expected build location, False otherwise.
    """
    cli_binary_path = os.path.join(BITNET_ROOT, "build", "bin", "llama-cli")
    quantize_binary_path = os.path.join(BITNET_ROOT, "build", "bin", "llama-quantize")
    
    cli_exists = os.path.isfile(cli_binary_path)
    quantize_exists = os.path.isfile(quantize_binary_path)
    
    if not cli_exists:
        print(f"Build Check Failed: llama-cli not found at {cli_binary_path}")
    if not quantize_exists:
         print(f"Build Check Warning: llama-quantize not found at {quantize_binary_path}")
         
    return cli_exists # Primarily check for the inference binary


def build_bitnet(model_type: str = "1bitLLM/bitnet_b1_58-large", 
                quant_type: str = "i2_s") -> bool:
    """
    Attempt to build the BitNet C++ tools by running the setup_env.py script.
    
    Args:
        model_type (str): Hugging Face model repo ID (e.g., "1bitLLM/bitnet_b1_58-large") 
                          to potentially download during setup.
        quant_type (str): Default quantization type to configure during setup.
        
    Returns:
        bool: True if the setup script runs successfully AND llama-cli exists afterwards, 
              False otherwise.
    """
    setup_script_path = os.path.join(BITNET_ROOT, "setup_env.py")
    if not os.path.isfile(setup_script_path):
        print(f"Error: setup_env.py not found in BITNET_ROOT ({BITNET_ROOT}). Cannot build.")
        return False
        
    try:
        print(f"Running BitNet setup script: {setup_script_path}")
        # Ensure using the correct python executable
        python_executable = sys.executable 
        cmd = [
            python_executable,
            setup_script_path,
            "--hf-repo", model_type, 
            "--quant-type", quant_type
            # Add other necessary flags for setup_env.py if required
        ]
        
        print(f"Build command: {' '.join(cmd)}")
        # Run the build script from within the BITNET_ROOT directory
        result = subprocess.run(
            cmd,
            check=True, # Raise exception on non-zero exit code
            cwd=BITNET_ROOT, # Execute in the BitNet repo directory
            capture_output=True, # Capture output
            text=True,
            encoding='utf-8'
        )
        print("BitNet setup script completed successfully.")
        print("stdout:\n", result.stdout)
        if result.stderr:
             print("stderr:\n", result.stderr) # Show stderr even on success, might contain warnings
        # Verify build by checking for the binary again
        return check_bitnet_build()

    except FileNotFoundError:
         print(f"Error: Python executable '{python_executable}' not found?")
         return False
    except subprocess.CalledProcessError as e:
        print(f"Error building BitNet using setup_env.py (exit code {e.returncode}):")
        print(f"Command: {' '.join(e.cmd)}")
        print("stdout:\n", e.stdout)
        print("stderr:\n", e.stderr)
        return False
    except Exception as e:
         print(f"An unexpected error occurred during BitNet build: {e}")
         return False


# --- Auto-build attempt ---
# Optionally attempt to build if check fails on import
# Set AUTO_BUILD_BITNET = True to enable this
AUTO_BUILD_BITNET = False 

if not check_bitnet_build():
    print("BitNet C++ tools not found.")
    if AUTO_BUILD_BITNET:
        print("Attempting to build BitNet automatically...")
        if build_bitnet():
            print("BitNet built successfully.")
        else:
            print("Failed to build BitNet. Please build it manually in the BitNet repository.")
    else:
        print("Automatic build disabled. Please build BitNet manually in the BitNet repository")
        print(f"Expected location: {os.path.join(BITNET_ROOT, 'build', 'bin')}")
# --- End Auto-build ---


# Export public API components
__all__ = ['BitNetModel', 'check_bitnet_build', 'build_bitnet', 'BITNET_ROOT']