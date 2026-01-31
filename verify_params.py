# verify_params.py
"""
Quick script to verify model parameters for different configurations.

Usage:
    python verify_params.py
    python verify_params.py --backbone tiny --attn_type minimal --mlp_ratio 1.125
"""

import argparse
import torch
from models.hybrid_model import HybridConvNeXtV2, print_model_summary, BACKBONE_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Verify model parameters")
    parser.add_argument("--backbone", type=str, default="tiny",
                       choices=list(BACKBONE_CONFIGS.keys()),
                       help="Backbone variant")
    parser.add_argument("--attn_type", type=str, default="minimal",
                       choices=["minimal", "paper", "reduced", "mobilevit"],
                       help="Attention type")
    parser.add_argument("--mlp_ratio", type=float, default=1.125,
                       help="MLP expansion ratio")
    parser.add_argument("--pretrained", action="store_true",
                       help="Load pretrained weights")
    parser.add_argument("--all", action="store_true",
                       help="Test all common configurations")
    args = parser.parse_args()
    
    print("=" * 70)
    print("HYBRID CONVNEXTV2 PARAMETER VERIFICATION")
    print("=" * 70)
    print(f"\nPaper target: 21.92M parameters")
    
    if args.all:
        # Test multiple configurations
        configs = [
            ('tiny', 'minimal', 1.125),
            ('tiny', 'minimal', 1.0),
            ('tiny', 'paper', 1.0),
            ('nano', 'minimal', 1.125),
            ('nano', 'paper', 1.0),
            ('pico', 'minimal', 2.0),
            ('pico', 'paper', 1.0),
        ]
        
        print("\nConfiguration comparison:")
        print("-" * 70)
        print(f"{'Backbone':<10}{'Attention':<12}{'MLP Ratio':<12}{'Params (M)':<14}{'Diff from 21.92M':<18}")
        print("-" * 70)
        
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
                marker = "✓" if diff < 0.5 else ""
                print(f"{backbone:<10}{attn_type:<12}{mlp_ratio:<12.3f}{total:<14.3f}{diff:<14.3f} {marker}")
                del model
            except Exception as e:
                print(f"{backbone:<10}{attn_type:<12}{mlp_ratio:<12.3f}ERROR: {e}")
        
    else:
        # Test single configuration
        print(f"\nConfiguration: backbone={args.backbone}, attn_type={args.attn_type}, "
              f"mlp_ratio={args.mlp_ratio}, pretrained={args.pretrained}")
        
        model = HybridConvNeXtV2(
            backbone_variant=args.backbone,
            attn_type=args.attn_type,
            mlp_ratio=args.mlp_ratio,
            num_classes=8,
            pretrained=args.pretrained,
            pretrained_weights='ft_1k',
        )
        
        print_model_summary(model)
        
        # Verify no old stage naming errors
        for name, p in model.named_parameters():
            if "stages.2" in name or "stages.3" in name:
                print(f"ERROR: Found old-style parameter name: {name}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {y.shape}")
        print("Forward pass successful!")
        
        # Parameter breakdown
        params = model.count_parameters()
        print(f"\nTotal parameters: {params['total']/1e6:.3f}M")
        print(f"Difference from paper target (21.92M): {abs(params['total']/1e6 - 21.92):.3f}M")


if __name__ == '__main__':
    main()
