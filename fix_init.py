#!/usr/bin/env python3
"""
Fix LayerNorm initialization issue.
LayerNorm weights should be initialized to 1.0, not 0.0
"""
import torch
import torch.nn as nn


def fix_layernorm_init(model):
    """
    Fix LayerNorm initialization.
    By default, LayerNorm weights should be 1.0 and bias 0.0.
    """
    fixed_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            # Check if weights are zero
            if module.weight is not None:
                if module.weight.std() < 1e-6:
                    # Fix: initialize to ones
                    nn.init.ones_(module.weight)
                    fixed_count += 1
            
            # Check if bias exists and initialize to zero
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    return fixed_count


if __name__ == "__main__":
    from models.hybrid_model import HybridConvNeXtV2
    
    print("Testing LayerNorm Fix")
    print("="*60)
    
    model = HybridConvNeXtV2(num_classes=8, pretrained=False)
    
    # Check before
    print("\nBefore fix:")
    ln_count = 0
    zero_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            ln_count += 1
            if module.weight.std() < 1e-6:
                zero_count += 1
                if zero_count <= 5:
                    print(f"  {name}: std={module.weight.std():.6f}")
    
    print(f"\nTotal LayerNorms: {ln_count}")
    print(f"Zero-initialized: {zero_count}")
    
    # Fix
    fixed = fix_layernorm_init(model)
    print(f"\nFixed {fixed} LayerNorm modules")
    
    # Check after
    print("\nAfter fix:")
    zero_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            if module.weight.std() < 1e-6:
                zero_count += 1
    
    print(f"Zero-initialized: {zero_count}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        y = model(x)
    
    print(f"Output: {y.shape}")
    print(f"Output stats: mean={y.mean():.4f}, std={y.std():.4f}")
    
    if torch.isnan(y).any():
        print("✗ Output contains NaN")
    else:
        print("✓ Output is valid")