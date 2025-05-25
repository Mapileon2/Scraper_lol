"""
This module provides a workaround for PyTorch versions that don't have the torch.compiler module.
It creates a dummy compiler module to prevent import errors.
"""
import torch

# Check if torch.compiler exists, if not create a dummy one
if not hasattr(torch, 'compiler'):
    import sys
    import types
    
    # Create a dummy compiler module
    compiler_module = types.ModuleType('torch.compiler')
    
    # Add a dummy disable decorator
    def dummy_disable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    compiler_module.disable = dummy_disable
    
    # Add the dummy module to torch
    torch.compiler = compiler_module
    
    # Also add to sys.modules to prevent re-import issues
    sys.modules['torch.compiler'] = compiler_module
