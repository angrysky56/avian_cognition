
import os
import sys

# Add project root to path
sys.path.append('/home/ty/Repositories/ai_workspace/avian_cognition')

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
    print(f"Error importing BitNetModel: {e}")
    sys.exit(1)
    
# Try importing integration
try:
    from src.core.bitnet_integration import get_bitnet_model, apply_bitnet_quantization
    print("Successfully imported BitNet integration")
    
    # Try getting a model
    model = get_bitnet_model()
    print(f"Model path: {getattr(model, 'model_path', None)}")
    
except ImportError as e:
    print(f"Error importing BitNet integration: {e}")
    sys.exit(1)

print("BitNet integration test passed!")
    