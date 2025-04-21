
#!/usr/bin/env python
# test_mamba_import.py - Test if mamba_ssm can be imported correctly

try:
    from mamba_ssm import Mamba, MambaConfig
    print("SUCCESS: mamba_ssm successfully imported.")
    print(f"Mamba: {Mamba}")
    print(f"MambaConfig: {MambaConfig}")
except ImportError as e:
    print(f"FAIL: Could not import mamba_ssm: {e}")
    
print("\nPython path:")
import sys
for p in sys.path:
    print(f"  {p}")

print("\nChecking for mamba_ssm package location:")
import importlib.util
spec = importlib.util.find_spec("mamba_ssm")
if spec is not None:
    print(f"mamba_ssm found at: {spec.origin}")
else:
    print("mamba_ssm not found in sys.path")
