#!/usr/bin/env python3
"""
BitNet Setup Script

This script sets up BitNet integration for the Avian Cognition project.
It ensures that the BitNet repository is properly installed, builds the
necessary components, and prepares the Python integration.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR  # Script is at project root
BITNET_REPO_PATH = "/home/ty/Repositories/BitNet"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
CORE_DIR = os.path.join(SRC_DIR, "core")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BitNet Setup for Avian Cognition")
    parser.add_argument(
        "--model",
        type=str,
        default="BitNet-b1.58-2B-4T",
        help="BitNet model to use"
    )
    parser.add_argument(
        "--force_build",
        action="store_true",
        help="Force rebuild of BitNet even if already built"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="i2_s",
        choices=["i2_s", "tl1", "tl2"],
        help="Quantization type to use"
    )
    return parser.parse_args()


def check_bitnet_repo():
    """
    Check if BitNet repository exists and is properly set up.
    
    Returns:
        bool: Whether BitNet repo exists
    """
    if not os.path.exists(BITNET_REPO_PATH):
        print(f"Error: BitNet repository not found at {BITNET_REPO_PATH}")
        print("Please clone the BitNet repository first:")
        print("  git clone https://github.com/microsoft/BitNet.git /home/ty/Repositories/BitNet")
        return False
    
    # Check if repository has essential files
    required_files = [
        "setup_env.py",
        "run_inference.py",
        "CMakeLists.txt"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(BITNET_REPO_PATH, file)):
            print(f"Error: BitNet repository missing required file: {file}")
            print("Please ensure you have cloned the correct repository")
            return False
    
    return True


def build_bitnet(args):
    """
    Build BitNet from source.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Whether build was successful
    """
    print(f"Building BitNet with model {args.model} and quantization {args.quantize}...")
    
    # Check if BitNet is already built
    binary_path = os.path.join(BITNET_REPO_PATH, "build", "bin", "llama-cli")
    if os.path.exists(binary_path) and not args.force_build:
        print(f"BitNet already built at {binary_path}")
        return True
    
    # Build BitNet
    build_cmd = [
        sys.executable,
        os.path.join(BITNET_REPO_PATH, "setup_env.py"),
        "-md", os.path.join(BITNET_REPO_PATH, "models", args.model),
        "-q", args.quantize
    ]
    
    try:
        subprocess.run(build_cmd, check=True, cwd=BITNET_REPO_PATH)
        
        # Verify build success
        if os.path.exists(binary_path):
            print(f"BitNet successfully built at {binary_path}")
            return True
        else:
            print(f"Error: BitNet build failed, binary not found at {binary_path}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error building BitNet: {e}")
        return False


def check_bitnet_wrapper():
    """
    Check if BitNet wrapper is installed.
    
    Returns:
        bool: Whether wrapper is installed
    """
    print("Checking BitNet wrapper...")
    
    # Check if wrapper is already installed
    wrapper_path = os.path.join(CORE_DIR, "bitnet_cpp.py")
    
    if os.path.exists(wrapper_path):
        print(f"BitNet wrapper found at {wrapper_path}")
        return True
    else:
        print(f"Error: BitNet wrapper not found at {wrapper_path}")
        return False


def check_bitnet_integration():
    """
    Check if BitNet integration is installed.
    
    Returns:
        bool: Whether integration is installed
    """
    print("Checking BitNet integration...")
    
    # Check if integration is already installed
    integration_path = os.path.join(CORE_DIR, "bitnet_integration.py")
    
    if os.path.exists(integration_path):
        print(f"BitNet integration found at {integration_path}")
        return True
    else:
        print(f"Error: BitNet integration not found at {integration_path}")
        return False


def test_integration():
    """
    Test BitNet integration.
    
    Returns:
        bool: Whether test was successful
    """
    print("Testing BitNet integration...")
    
    test_script = """
import os
import sys

# Add project root to path
sys.path.append('{}')

# Try importing BitNet
try:
    from src.core.bitnet_cpp import BitNetModel
    print("Successfully imported BitNetModel")
    
    # Check if BitNet is built
    if not os.path.exists('/home/ty/Repositories/BitNet/build/bin/llama-cli'):
        print("Warning: BitNet binary not found")
    else:
        print("BitNet binary found and ready to use")
        
except ImportError as e:
    print(f"Error importing BitNetModel: {{e}}")
    sys.exit(1)
    
# Try importing integration
try:
    from src.core.bitnet_integration import get_bitnet_model, apply_bitnet_quantization
    print("Successfully imported BitNet integration")
    
    # Try getting a model
    model = get_bitnet_model()
    print(f"Model path: {{getattr(model, 'model_path', None)}}")
    
except ImportError as e:
    print(f"Error importing BitNet integration: {{e}}")
    sys.exit(1)

print("BitNet integration test passed!")
    """.format(PROJECT_ROOT)
    
    test_script_path = os.path.join(PROJECT_ROOT, "test_bitnet_integration.py")
    
    # Write test script
    with open(test_script_path, "w") as f:
        f.write(test_script)
    
    # Run test script
    try:
        result = subprocess.run(
            [sys.executable, test_script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        # Check if test passed
        if "BitNet integration test passed!" in result.stdout:
            print("BitNet integration test passed!")
            return True
        else:
            print("BitNet integration test failed")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error testing BitNet integration: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    print("=== BitNet Setup for Avian Cognition ===")
    print(f"BitNet repository: {BITNET_REPO_PATH}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantize}")
    print(f"Force build: {args.force_build}")
    print("========================================")
    
    # Check BitNet repository
    if not check_bitnet_repo():
        print("BitNet repository check failed. Please fix the issues and try again.")
        return 1
    
    # Build BitNet
    if not build_bitnet(args):
        print("BitNet build failed. Please fix the issues and try again.")
        return 1
    
    # Check wrapper
    if not check_bitnet_wrapper():
        print("BitNet wrapper not found. Please ensure it's properly installed.")
        return 1
    
    # Check integration
    if not check_bitnet_integration():
        print("BitNet integration not found. Please ensure it's properly installed.")
        return 1
    
    # Test integration
    if not test_integration():
        print("BitNet integration test failed. Please fix the issues and try again.")
        return 1
    
    print("\n=== BitNet Setup Complete ===")
    print("BitNet is now integrated with the Avian Cognition project.")
    print("You can now run the integrated model with BitNet quantization:")
    print(f"  python examples/integrated_model.py --quantize --model_size mini")
    print("=========================================================")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
