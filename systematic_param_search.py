#!/usr/bin/env python3
"""
Systematic search to find the EXACT configuration that produces 21.92M parameters.

Since the paper does NOT explicitly state the dimensions, we must test all
reasonable combinations of:
- ConvNeXtV2 variant dims (for stages 1-2)
- Attention block dims (for stages 3-4)
- MLP expansion ratios

Goal: Find configuration closest to 21.92M parameters
"""

import torch
import torch.nn as nn
from itertools import product
import sys
sys.path.append('.')

from models.separable_attention import SeparableSelfAttention
from models.convnextv2 import ConvNeXtV2
from models.utils import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SeparableSelfAttention(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FlexibleHybridModel(nn.Module):
    """Hybrid model with flexible dimension configuration"""
    
    def __init__(self, dims, mlp_ratio=0.5, num_classes=8):
        """
        Args:
            dims: List of 4 dimensions [d1, d2, d3, d4] for each stage
            mlp_ratio: MLP expansion ratio in transformer blocks
            num_classes: Number of output classes
        """
        super().__init__()
        
        assert len(dims) == 4, "Must provide 4 dimensions"
        
        # Depths are fixed from paper: [3, 3, 9, 12]
        depths = [3, 3, 9, 12]
        
        # Create ConvNeXt backbone with only stages 1-2
        # We use depths [3, 3, 0, 0] since stages 3-4 are attention
        backbone = ConvNeXtV2(
            in_chans=3,
            num_classes=1000,
            depths=[depths[0], depths[1], 0, 0],
            dims=dims,
            drop_path_rate=0.0,
        )
        
        # Extract components
        self.stem = backbone.downsample_layers[0]
        self.stage1 = backbone.stages[0]
        self.down1 = backbone.downsample_layers[1]
        self.stage2 = backbone.stages[1]
        self.down2 = backbone.downsample_layers[2]
        self.down3 = backbone.downsample_layers[3]
        
        del backbone
        
        # Stage 3: 9 attention blocks
        self.stage3 = nn.ModuleList([
            TransformerBlock(dims[2], mlp_ratio)
            for _ in range(depths[2])
        ])
        
        # Stage 4: 12 attention blocks
        self.stage4 = nn.ModuleList([
            TransformerBlock(dims[3], mlp_ratio)
            for _ in range(depths[3])
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dims[3], eps=1e-6)
        self.head = nn.Linear(dims[3], num_classes)
    
    def forward(self, x):
        # Stage 1
        x = self.stem(x)
        x = self.stage1(x)
        
        # Stage 2
        x = self.down1(x)
        x = self.stage2(x)
        
        # Stage 3
        x = self.down2(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for block in self.stage3:
            x = block(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # Stage 4
        x = self.down3(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for block in self.stage4:
            x = block(x)
        
        # Classification
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.head(x)
        return x
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return total


def test_configuration(dims, mlp_ratio, target_params=21.92e6):
    """Test a specific configuration"""
    try:
        model = FlexibleHybridModel(dims=dims, mlp_ratio=mlp_ratio)
        params = model.count_parameters()
        diff = abs(params - target_params)
        diff_pct = (diff / target_params) * 100
        
        del model
        torch.cuda.empty_cache()
        
        return {
            'dims': dims,
            'mlp_ratio': mlp_ratio,
            'params': params,
            'diff': diff,
            'diff_pct': diff_pct
        }
    except Exception as e:
        return None


def systematic_search():
    """Systematically search all reasonable configurations"""
    
    print("="*80)
    print("SYSTEMATIC CONFIGURATION SEARCH")
    print("="*80)
    print("\nPaper specifications:")
    print("  - Depths: [3, 3, 9, 12] (explicitly stated)")
    print("  - Target params: 21.92M (from Table 3)")
    print("  - Dimensions: NOT EXPLICITLY STATED")
    print("\nSearching all ConvNeXtV2 variant dimensions...")
    print("="*80)
    
    # All ConvNeXtV2 variants with their dims
    variants = {
        'Atto':  [40, 80, 160, 320],
        'Femto': [48, 96, 192, 384],
        'Pico':  [64, 128, 256, 512],
        'Nano':  [80, 160, 320, 640],
        'Tiny':  [96, 192, 384, 768],
        'Base':  [128, 256, 512, 1024],
    }
    
    # MLP ratios to test
    mlp_ratios = [0.25, 0.5, 1.0, 2.0, 0.4]
    
    target_params = 21.92e6
    results = []
    
    print(f"\nTesting {len(variants)} variants × {len(mlp_ratios)} MLP ratios = {len(variants)*len(mlp_ratios)} configurations\n")
    
    for variant_name, dims in variants.items():
        print(f"\n{'='*60}")
        print(f"Testing {variant_name} variant: dims={dims}")
        print(f"{'='*60}")
        
        for mlp_ratio in mlp_ratios:
            print(f"\n  MLP ratio: {mlp_ratio}")
            
            result = test_configuration(dims, mlp_ratio, target_params)
            
            if result:
                results.append({
                    'variant': variant_name,
                    **result
                })
                
                print(f"    Params:     {result['params']:>12,} ({result['params']/1e6:.2f}M)")
                print(f"    Difference: {result['diff']:>12,.0f} ({result['diff_pct']:.2f}%)")
                
                if result['diff_pct'] < 1:
                    print(f"    ★★★ EXCELLENT MATCH! (<1%)")
                elif result['diff_pct'] < 5:
                    print(f"    ★★ GOOD MATCH! (<5%)")
    
    # Sort by difference
    results.sort(key=lambda x: x['diff'])
    
    # Print top results
    print("\n" + "="*80)
    print("TOP 10 CLOSEST CONFIGURATIONS")
    print("="*80)
    print(f"\n{'Rank':<6} {'Variant':<8} {'Dims':<25} {'MLP':<6} {'Params':<15} {'Diff%':<8}")
    print("-"*80)
    
    for i, result in enumerate(results[:10], 1):
        dims_str = str(result['dims'])
        print(f"{i:<6} {result['variant']:<8} {dims_str:<25} "
              f"{result['mlp_ratio']:<6} {result['params']/1e6:<14.2f}M {result['diff_pct']:<7.2f}%")
    
    # Analyze best result
    best = results[0]
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"\nVariant:    {best['variant']}")
    print(f"Dimensions: {best['dims']}")
    print(f"MLP ratio:  {best['mlp_ratio']}")
    print(f"Parameters: {best['params']:,} ({best['params']/1e6:.2f}M)")
    print(f"Target:     {target_params:,.0f} (21.92M)")
    print(f"Difference: {best['diff_pct']:.2f}%")
    
    if best['diff_pct'] < 1:
        print("\n✓✓✓ EXCELLENT MATCH - Use this configuration!")
    elif best['diff_pct'] < 5:
        print("\n✓✓ GOOD MATCH - This is likely correct")
    else:
        print("\n⚠ Warning: No exact match found")
        print("The paper may use custom dimensions not in standard ConvNeXtV2 variants")
    
    # Check if multiple variants are close
    close_matches = [r for r in results if r['diff_pct'] < 5]
    
    if len(close_matches) > 1:
        print(f"\n⚠ Note: {len(close_matches)} configurations within 5% of target:")
        for match in close_matches:
            print(f"  - {match['variant']}: dims={match['dims']}, MLP={match['mlp_ratio']} "
                  f"({match['diff_pct']:.2f}%)")
    
    # Additional analysis
    print("\n" + "="*80)
    print("ANALYSIS BY VARIANT")
    print("="*80)
    
    for variant_name in variants.keys():
        variant_results = [r for r in results if r['variant'] == variant_name]
        if variant_results:
            best_for_variant = min(variant_results, key=lambda x: x['diff'])
            print(f"\n{variant_name}:")
            print(f"  Best MLP ratio: {best_for_variant['mlp_ratio']}")
            print(f"  Params: {best_for_variant['params']/1e6:.2f}M")
            print(f"  Difference: {best_for_variant['diff_pct']:.2f}%")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    print(f"\nBased on parameter count analysis:")
    print(f"  Use ConvNeXtV2 {best['variant']} variant")
    print(f"  Dimensions: {best['dims']}")
    print(f"  MLP ratio: {best['mlp_ratio']}")
    
    print(f"\nCorresponding pretrained weights:")
    weight_urls = {
        'Atto': 'convnextv2_atto_1k_224_ema.pt',
        'Femto': 'convnextv2_femto_1k_224_ema.pt',
        'Pico': 'convnextv2_pico_1k_224_ema.pt',
        'Nano': 'convnextv2_nano_1k_224_ema.pt',
        'Tiny': 'convnextv2_tiny_1k_224_ema.pt',
        'Base': 'convnextv2_base_1k_224_ema.pt',
    }
    
    if best['variant'] in weight_urls:
        print(f"  URL: https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/{weight_urls[best['variant']]}")
    
    print(f"\nUpdate model code to use:")
    print(f"  dims = {best['dims']}")
    print(f"  mlp_ratio = {best['mlp_ratio']}")
    
    return results


if __name__ == "__main__":
    results = systematic_search()
    
    print("\n" + "="*80)
    print("SEARCH COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Update hybrid_model.py with the recommended configuration")
    print("2. Run test_model.py to verify parameter count")
    print("3. Start training with train.py")
    print("\n" + "="*80)