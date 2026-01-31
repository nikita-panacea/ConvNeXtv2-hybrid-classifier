# debug_params.py
"""
Debug script for analyzing model parameters in detail.

This script provides detailed analysis of parameter distribution
across model components to help identify configuration issues.

Usage:
    python debug_params.py
    python debug_params.py --config tiny_minimal
"""

import sys
import os
import argparse
sys.path.append(os.getcwd())

import torch
from models.hybrid_model import HybridConvNeXtV2, BACKBONE_CONFIGS


def print_model_info(backbone='tiny', attn_type='minimal', mlp_ratio=1.125, pretrained=False):
    """Print detailed model information."""
    print("=" * 80)
    print(f"HYBRID CONVNEXTV2 DEBUG INFO")
    print(f"Configuration: backbone={backbone}, attn_type={attn_type}, mlp_ratio={mlp_ratio}")
    print("=" * 80)
    
    print("\nInstantiating model...")
    model = HybridConvNeXtV2(
        backbone_variant=backbone,
        attn_type=attn_type,
        mlp_ratio=mlp_ratio,
        num_classes=8,
        pretrained=pretrained,
        pretrained_weights='ft_1k' if pretrained else 'none',
    )
    
    # Total parameters
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_all = sum(p.numel() for p in model.parameters())
    
    print(f"\nTotal trainable parameters: {total_trainable/1e6:.3f}M")
    print(f"Total all parameters: {total_all/1e6:.3f}M")
    print(f"Target (paper): 21.92M")
    print(f"Difference: {abs(total_trainable/1e6 - 21.92):.3f}M")
    
    # Parameter counts by top-level module
    by_prefix = {}
    for name, p in model.named_parameters():
        prefix = name.split('.')[0]
        by_prefix.setdefault(prefix, 0)
        by_prefix[prefix] += p.numel()
    
    print("\n" + "-" * 60)
    print("Parameter counts by top-level module:")
    print("-" * 60)
    for k, v in sorted(by_prefix.items(), key=lambda x: -x[1]):
        pct = 100 * v / total_trainable
        print(f"  {k:30s} : {v/1e6:8.3f}M ({pct:5.1f}%)")
    
    # Stage breakdown
    print("\n" + "-" * 60)
    print("Stage breakdown:")
    print("-" * 60)
    
    stage_groups = {
        'ConvNeXtV2 (stem)': ['stem'],
        'ConvNeXtV2 (stage1)': ['stage1'],
        'ConvNeXtV2 (down1)': ['down1'],
        'ConvNeXtV2 (stage2)': ['stage2'],
        'Downsample (down2)': ['down2'],
        'Transformer (stage3)': ['stage3'],
        'Downsample (down3)': ['down3'],
        'Transformer (stage4)': ['stage4'],
        'Head (norm+linear)': ['norm', 'head'],
    }
    
    for group_name, prefixes in stage_groups.items():
        group_params = sum(by_prefix.get(p, 0) for p in prefixes)
        pct = 100 * group_params / total_trainable
        print(f"  {group_name:30s} : {group_params/1e6:8.3f}M ({pct:5.1f}%)")
    
    # Summary
    convnext_params = by_prefix.get('stem', 0) + by_prefix.get('stage1', 0) + \
                      by_prefix.get('down1', 0) + by_prefix.get('stage2', 0)
    transformer_params = by_prefix.get('stage3', 0) + by_prefix.get('stage4', 0)
    
    print("\n" + "-" * 60)
    print("Summary:")
    print("-" * 60)
    print(f"  ConvNeXtV2 stages (1-2):     {convnext_params/1e6:8.3f}M")
    print(f"  Transformer stages (3-4):   {transformer_params/1e6:8.3f}M")
    print(f"  Other:                      {(total_trainable - convnext_params - transformer_params)/1e6:8.3f}M")
    
    # List largest parameter tensors
    large = [(name, p.numel()) for name, p in model.named_parameters()]
    large.sort(key=lambda x: -x[1])
    
    print("\n" + "-" * 60)
    print("Top 20 largest parameter tensors:")
    print("-" * 60)
    for name, cnt in large[:20]:
        print(f"  {name:60s} : {cnt/1e6:.3f}M")
    
    # Check backbone factory
    print("\n" + "-" * 60)
    print("Backbone configuration:")
    print("-" * 60)
    config = BACKBONE_CONFIGS[backbone]
    print(f"  Depths: {config['depths']}")
    print(f"  Dims:   {config['dims']}")
    print(f"  Stage 3 dim: {config['dims'][2]}")
    print(f"  Stage 4 dim: {config['dims'][3]}")
    
    # MLP hidden dims
    mlp_hidden_3 = int(config['dims'][2] * mlp_ratio)
    mlp_hidden_4 = int(config['dims'][3] * mlp_ratio)
    print(f"\n  MLP hidden dim (stage 3): {mlp_hidden_3} (dim={config['dims'][2]}, ratio={mlp_ratio})")
    print(f"  MLP hidden dim (stage 4): {mlp_hidden_4} (dim={config['dims'][3]}, ratio={mlp_ratio})")
    
    return model


def compare_configurations():
    """Compare different configurations to find best match."""
    print("\n" + "=" * 80)
    print("CONFIGURATION COMPARISON")
    print("=" * 80)
    
    configs = [
        # (backbone, attn_type, mlp_ratio)
        ('tiny', 'minimal', 1.0),
        ('tiny', 'minimal', 1.125),
        ('tiny', 'minimal', 1.5),
        ('tiny', 'minimal', 2.0),
        ('tiny', 'paper', 0.5),
        ('tiny', 'paper', 1.0),
        ('nano', 'minimal', 1.0),
        ('nano', 'minimal', 1.5),
        ('nano', 'minimal', 2.0),
        ('nano', 'paper', 0.5),
        ('nano', 'paper', 1.0),
        ('pico', 'minimal', 1.5),
        ('pico', 'minimal', 2.0),
        ('pico', 'minimal', 2.5),
        ('pico', 'paper', 1.0),
    ]
    
    print(f"\n{'Backbone':<10}{'Attn':<12}{'MLP':<8}{'Params(M)':<12}{'Diff(M)':<10}{'Match':<10}")
    print("-" * 70)
    
    results = []
    for backbone, attn_type, mlp_ratio in configs:
        try:
            model = HybridConvNeXtV2(
                backbone_variant=backbone,
                attn_type=attn_type,
                mlp_ratio=mlp_ratio,
                num_classes=8,
                pretrained=False,
            )
            total = sum(p.numel() for p in model.parameters()) / 1e6
            diff = abs(total - 21.92)
            
            match = "🎯" if diff < 0.1 else ("✅" if diff < 0.5 else ("⚠️" if diff < 1.0 else ""))
            
            print(f"{backbone:<10}{attn_type:<12}{mlp_ratio:<8.2f}{total:<12.3f}{diff:<10.3f}{match}")
            results.append((backbone, attn_type, mlp_ratio, total, diff))
            del model
            
        except Exception as e:
            print(f"{backbone:<10}{attn_type:<12}{mlp_ratio:<8.2f}ERROR: {e}")
    
    # Best match
    results.sort(key=lambda x: x[4])
    if results:
        best = results[0]
        print(f"\n🏆 Best match: backbone={best[0]}, attn_type={best[1]}, mlp_ratio={best[2]}")
        print(f"   Parameters: {best[3]:.3f}M (diff from 21.92M: {best[4]:.3f}M)")


def main():
    parser = argparse.ArgumentParser(description="Debug model parameters")
    parser.add_argument("--backbone", type=str, default="tiny")
    parser.add_argument("--attn_type", type=str, default="minimal")
    parser.add_argument("--mlp_ratio", type=float, default=1.125)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare multiple configurations")
    args = parser.parse_args()
    
    if args.compare:
        compare_configurations()
    else:
        print_model_info(
            backbone=args.backbone,
            attn_type=args.attn_type,
            mlp_ratio=args.mlp_ratio,
            pretrained=args.pretrained,
        )


if __name__ == '__main__':
    main()
