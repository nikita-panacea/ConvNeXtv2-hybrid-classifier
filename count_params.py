#!/usr/bin/env python3
"""
Detailed parameter breakdown for the hybrid model.
Helps identify where parameters are coming from.
"""
import torch
# from models.hybrid_model import HybridConvNeXtV2
from models.hybrid_model_adaptive import HybridConvNeXtV2Adaptive


def count_params_recursive(module, name="", indent=0):
    """Recursively count parameters in a module"""
    own_params = sum(p.numel() for p in module.parameters(recurse=False))
    
    prefix = "  " * indent
    if own_params > 0:
        print(f"{prefix}{name}: {own_params:,} params")
    
    total = own_params
    for child_name, child in module.named_children():
        child_total = count_params_recursive(child, child_name, indent+1)
        total += child_total
    
    if indent == 0 or (hasattr(module, '__len__') and len(list(module.children())) > 0):
        if name and total > 0:
            print(f"{prefix}{name} TOTAL: {total:,} params")
    
    return total


def detailed_breakdown():
    """Detailed parameter breakdown by component"""
    print("="*60)
    print("DETAILED PARAMETER BREAKDOWN")
    print("="*60)
    
    # model = HybridConvNeXtV2(num_classes=8, pretrained=False)
    model = HybridConvNeXtV2Adaptive(num_classes=8, pretrained=False)
    
    # Count by major component
    components = {
        "stem": model.stem,
        "stage1": model.stage1,
        "down1": model.down1,
        "stage2": model.stage2,
        "down2": model.down2,
        "stage3": model.stage3,
        "down3": model.down3,
        "stage4": model.stage4,
        "norm": model.norm,
        "head": model.head,
    }
    
    print("\nComponent-wise breakdown:")
    print("-" * 60)
    
    total = 0
    for name, component in components.items():
        params = sum(p.numel() for p in component.parameters())
        total += params
        pct = 100.0 * params / 21.92e6
        print(f"{name:15s}: {params:12,} params ({pct:5.2f}% of target)")
    
    print("-" * 60)
    print(f"{'TOTAL':15s}: {total:12,} params")
    print(f"{'TARGET':15s}: {21920000:12,} params")
    diff = total - 21920000
    print(f"{'DIFFERENCE':15s}: {diff:12,} params ({100*diff/21920000:+.2f}%)")
    
    # Detailed breakdown of attention stages
    print("\n" + "="*60)
    print("ATTENTION STAGE BREAKDOWN")
    print("="*60)
    
    print("\nStage 3 (9 blocks, dim=384):")
    if len(model.stage3) > 0:
        block = model.stage3[0]
        block_params = sum(p.numel() for p in block.parameters())
        print(f"  Per block: {block_params:,} params")
        print(f"  Total: {block_params * 9:,} params")
        
        # Break down one block
        print("\n  Single block breakdown:")
        for name, module in block.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"    {name:20s}: {params:,} params")
    
    print("\nStage 4 (12 blocks, dim=768):")
    if len(model.stage4) > 0:
        block = model.stage4[0]
        block_params = sum(p.numel() for p in block.parameters())
        print(f"  Per block: {block_params:,} params")
        print(f"  Total: {block_params * 12:,} params")
        
        # Break down one block
        print("\n  Single block breakdown:")
        for name, module in block.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"    {name:20s}: {params:,} params")
    
    # Separable attention detail
    print("\n" + "="*60)
    print("SEPARABLE ATTENTION ANALYSIS")
    print("="*60)
    
    if len(model.stage3) > 0:
        attn = model.stage3[0].attn
        print("\nStage 3 attention (dim=384):")
        for name, param in attn.named_parameters():
            print(f"  {name:30s}: {list(param.shape)} = {param.numel():,} params")
        total_attn = sum(p.numel() for p in attn.parameters())
        print(f"  {'TOTAL':30s}: {total_attn:,} params")
    
    if len(model.stage4) > 0:
        attn = model.stage4[0].attn
        print("\nStage 4 attention (dim=768):")
        for name, param in attn.named_parameters():
            print(f"  {name:30s}: {list(param.shape)} = {param.numel():,} params")
        total_attn = sum(p.numel() for p in attn.parameters())
        print(f"  {'TOTAL':30s}: {total_attn:,} params")
    
    # MLP detail
    print("\n" + "="*60)
    print("MLP ANALYSIS")
    print("="*60)
    
    if len(model.stage3) > 0:
        mlp = model.stage3[0].mlp
        print("\nStage 3 MLP (dim=384):")
        for name, param in mlp.named_parameters():
            layer_name = name.split('.')[0]
            print(f"  {name:30s}: {list(param.shape)} = {param.numel():,} params")
        total_mlp = sum(p.numel() for p in mlp.parameters())
        print(f"  {'TOTAL':30s}: {total_mlp:,} params")
    
    if len(model.stage4) > 0:
        mlp = model.stage4[0].mlp
        print("\nStage 4 MLP (dim=768):")
        for name, param in mlp.named_parameters():
            print(f"  {name:30s}: {list(param.shape)} = {param.numel():,} params")
        total_mlp = sum(p.numel() for p in mlp.parameters())
        print(f"  {'TOTAL':30s}: {total_mlp:,} params")


def estimate_target_params():
    """Estimate what parameters should be based on architecture"""
    print("\n" + "="*60)
    print("TARGET PARAMETER ESTIMATION")
    print("="*60)
    
    # ConvNeXt Tiny stages 1-2 params (from official model)
    convnext_params = 28_589_128  # Full Tiny model
    # Stages 3-4 contribute roughly: 28.6M - 4M (stages 1-2) ≈ 24M
    # So stages 1-2 + stem ≈ 4-5M
    stages_1_2_estimate = 4_000_000
    
    # Attention parameters per block
    # dim=384: context(384), key(384*384), value(384*384), out(384*384) = 384 + 3*147456 = 442,752
    # dim=768: context(768), key(768*768), value(768*768), out(768*768) = 768 + 3*589824 = 1,770,240
    
    attn_384_per_block = 384 + 3 * (384 * 384)
    attn_768_per_block = 768 + 3 * (768 * 768)
    
    # MLP parameters per block (with 0.5x expansion)
    # dim=384: 384->192: 384*192 + 192 = 73,920; 192->384: 192*384 + 384 = 74,112
    # dim=768: 768->384: 768*384 + 384 = 295,296; 384->768: 384*768 + 768 = 295,680
    
    mlp_384_per_block = (384 * 192 + 192) + (192 * 384 + 384)
    mlp_768_per_block = (768 * 384 + 384) + (384 * 768 + 768)
    
    # LayerNorm parameters (negligible but included)
    ln_384 = 384 * 2  # weight + bias
    ln_768 = 768 * 2
    
    # Per transformer block
    block_384 = attn_384_per_block + mlp_384_per_block + 2 * ln_384
    block_768 = attn_768_per_block + mlp_768_per_block + 2 * ln_768
    
    # Stage totals
    stage3_params = 9 * block_384
    stage4_params = 12 * block_768
    
    # Head
    head_params = 768 * 8 + 8
    
    # Final norm
    final_norm_params = 768 * 2
    
    # Downsampling layers (small)
    downsample_estimate = 200_000
    
    print("\nEstimated parameters:")
    print(f"  ConvNeXt stages 1-2: {stages_1_2_estimate:,}")
    print(f"  Downsample layers:   {downsample_estimate:,}")
    print(f"  Stage 3 (9 blocks):  {stage3_params:,}")
    print(f"  Stage 4 (12 blocks): {stage4_params:,}")
    print(f"  Final norm:          {final_norm_params:,}")
    print(f"  Head:                {head_params:,}")
    print(f"  " + "-"*40)
    
    total_estimate = (stages_1_2_estimate + downsample_estimate + 
                     stage3_params + stage4_params + 
                     final_norm_params + head_params)
    
    print(f"  TOTAL:               {total_estimate:,}")
    print(f"  TARGET:              {21_920_000:,}")
    print(f"  DIFFERENCE:          {total_estimate - 21_920_000:,}")
    
    print("\n  Per-block breakdown:")
    print(f"    Stage 3 block (384): {block_384:,} params")
    print(f"      - Attention:       {attn_384_per_block:,}")
    print(f"      - MLP:             {mlp_384_per_block:,}")
    print(f"      - LayerNorms:      {2*ln_384:,}")
    
    print(f"    Stage 4 block (768): {block_768:,} params")
    print(f"      - Attention:       {attn_768_per_block:,}")
    print(f"      - MLP:             {mlp_768_per_block:,}")
    print(f"      - LayerNorms:      {2*ln_768:,}")


def main():
    detailed_breakdown()
    estimate_target_params()
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("\nIf parameters are still too high, reduce MLP expansion further.")
    print("Current: 0.5x expansion (dim -> dim/2 -> dim)")
    print("Consider: Remove MLP entirely or use identity mapping")


if __name__ == "__main__":
    main()