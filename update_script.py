#!/usr/bin/env python3
"""
Update BitNet Integration

This script updates the BitNet integration in the Avian Cognition project
by copying the new implementation files to the appropriate locations.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
CORE_DIR = os.path.join(SRC_DIR, "core")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Update BitNet Integration")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backups of existing files"
    )
    return parser.parse_args()


def create_backup(file_path):
    """
    Create a backup of a file.
    
    Args:
        file_path: Path to file to backup
    """
    if os.path.exists(file_path):
        backup_path = f"{file_path}.bak"
        print(f"Creating backup: {backup_path}")
        shutil.copy2(file_path, backup_path)


def update_bitnet_integration():
    """
    Update BitNet integration files.
    
    Args:
        args: Command line arguments
    """
    args = parse_args()
    
    # Create src/core directory if it doesn't exist
    os.makedirs(CORE_DIR, exist_ok=True)
    
    # Define files to update
    files_to_update = [
        {
            "source": os.path.join(SCRIPT_DIR, "bitnet_cpp.py"),
            "destination": os.path.join(CORE_DIR, "bitnet_cpp.py"),
            "description": "BitNet C++ wrapper"
        },
        {
            "source": os.path.join(SCRIPT_DIR, "bitnet_integration.py"),
            "destination": os.path.join(CORE_DIR, "bitnet_integration.py"),
            "description": "BitNet integration helper"
        },
        {
            "source": os.path.join(SCRIPT_DIR, "test_bitnet_quantization.py"),
            "destination": os.path.join(PROJECT_ROOT, "test_bitnet_quantization.py"),
            "description": "BitNet quantization test script"
        },
        {
            "source": os.path.join(SCRIPT_DIR, "setup_bitnet.py"),
            "destination": os.path.join(PROJECT_ROOT, "setup_bitnet.py"),
            "description": "BitNet setup script"
        },
        {
            "source": os.path.join(SCRIPT_DIR, "updated_readme.md"),
            "destination": os.path.join(PROJECT_ROOT, "README.md"),
            "description": "Updated README.md"
        }
    ]
    
    # Update files
    for file_info in files_to_update:
        source = file_info["source"]
        destination = file_info["destination"]
        description = file_info["description"]
        
        if not os.path.exists(source):
            print(f"Error: Source file not found: {source}")
            continue
        
        # Create backup if requested and file exists
        if args.backup and os.path.exists(destination):
            create_backup(destination)
        
        # Copy file
        try:
            shutil.copy2(source, destination)
            print(f"Updated {description}: {destination}")
        except Exception as e:
            print(f"Error updating {description}: {e}")


def make_executable(file_path):
    """
    Make a file executable.
    
    Args:
        file_path: Path to file to make executable
    """
    if os.path.exists(file_path):
        try:
            # Set executable bit (chmod +x)
            mode = os.stat(file_path).st_mode
            mode |= 0o111  # Add executable bit for user, group, and others
            os.chmod(file_path, mode)
            print(f"Made executable: {file_path}")
        except Exception as e:
            print(f"Error making executable: {e}")


def main():
    """Main function."""
    print("=== Updating BitNet Integration ===")
    
    # Update BitNet integration files
    update_bitnet_integration()
    
    # Make scripts executable
    make_executable(os.path.join(PROJECT_ROOT, "setup_bitnet.py"))
    make_executable(os.path.join(PROJECT_ROOT, "test_bitnet_quantization.py"))
    
    print("\n=== Update Complete ===")
    print("BitNet integration has been updated.")
    print("Next steps:")
    print("1. Run the BitNet setup script:")
    print("   python setup_bitnet.py")
    print("2. Test the BitNet quantization:")
    print("   python test_bitnet_quantization.py")
    print("3. Run the integrated model with quantization:")
    print("   python examples/integrated_model.py --quantize")
    print("============================")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
