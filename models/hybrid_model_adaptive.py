# models/hybrid_model_adaptive.py
"""
Adaptive hybrid model that automatically tunes to 21.92M parameters
by using minimal MLP expansion or identity MLP if needed.
"""
import torch
import torch.nn as nn
from models.separable_attention import SeparableSelfAttention
from models.convnextv2 import ConvNeXtV2


class MinimalTransformerBlock(nn.Module):
    """
    Minimal transformer block with identity MLP.
    Only attention + norms, no MLP (or minimal MLP).
    """
    def __init__(self, dim, use_mlp=True, mlp_ratio=0.25):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SeparableSelfAttention(dim)
        
        if use_mlp:
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            mlp_dim = max(int(dim * mlp_ratio), 1)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, dim),
            )
        else:
            self.norm2 = None
            self.mlp = None

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        return x


class HybridConvNeXtV2Adaptive(nn.Module):
    """
    Adaptive version that tunes MLP size to hit 21.92M target.
    Tests different configurations to find the right balance.
    """

    def __init__(self, num_classes=8, pretrained=True, target_params=21.92e6):
        super().__init__()
        
        self.target_params = target_params

        # Create backbone
        backbone = ConvNeXtV2(
            in_chans=3,
            num_classes=1000,
            depths=[3, 3, 0, 0],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.0,
        )

        if pretrained:
            try:
                ckpt = torch.hub.load_state_dict_from_url(
                    "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt",
                    map_location="cpu",
                    check_hash=True
                )
                backbone.load_state_dict(ckpt["model"], strict=False)
                print("Loaded ConvNeXtV2-Tiny pretrained weights")
            except Exception as e:
                print(f"Warning: {e}")

        # Extract components
        self.stem = backbone.downsample_layers[0]
        self.stage1 = backbone.stages[0]
        self.down1 = backbone.downsample_layers[1]
        self.stage2 = backbone.stages[1]
        self.down2 = backbone.downsample_layers[2]
        self.down3 = backbone.downsample_layers[3]

        # Count parameters so far
        base_params = sum(p.numel() for p in self.parameters())
        
        # Attention-only params (no MLP)
        attn_only_params = (
            9 * (384 + 3 * 384 * 384 + 2 * 384) +  # Stage 3
            12 * (768 + 3 * 768 * 768 + 2 * 768) +  # Stage 4
            768 * 2 +  # final norm
            768 * num_classes + num_classes  # head
        )
        
        # Budget remaining
        remaining = target_params - base_params - attn_only_params
        
        print(f"\nParameter budget analysis:")
        print(f"  Base (ConvNeXt 1-2): {base_params:,}")
        print(f"  Attention only:      {attn_only_params:,}")
        print(f"  Remaining for MLPs:  {remaining:,}")
        
        # Calculate optimal MLP ratio
        # Stage 3: 9 blocks * (384 * mlp_dim * 2) = 9 * 768 * mlp_dim
        # Stage 4: 12 blocks * (768 * mlp_dim * 2) = 12 * 1536 * mlp_dim
        # Total: (9 * 768 + 12 * 1536) * mlp_dim = 25,344 * mlp_dim
        
        total_dim = 9 * 384 + 12 * 768
        # mlp params per dim: 2 * (dim * mlp_dim + mlp_dim) ≈ 2 * dim * mlp_dim
        # So: remaining ≈ total_dim * 2 * mlp_dim
        
        mlp_dim_budget = remaining / (2 * total_dim)
        
        # Find closest power of 2 or common fraction
        if mlp_dim_budget < 0.1:
            use_mlp = False
            mlp_ratio = 0.0
        elif mlp_dim_budget < 0.2:
            mlp_ratio = 0.125
        elif mlp_dim_budget < 0.35:
            mlp_ratio = 0.25
        elif mlp_dim_budget < 0.6:
            mlp_ratio = 0.5
        else:
            mlp_ratio = 1.0
        
        use_mlp = mlp_ratio > 0
        
        print(f"  Calculated MLP budget: {mlp_dim_budget:.3f}x")
        print(f"  Using MLP ratio: {mlp_ratio}x")
        
        # Build stages
        self.stage3 = nn.Sequential(
            *[MinimalTransformerBlock(384, use_mlp=use_mlp, mlp_ratio=mlp_ratio) 
              for _ in range(9)]
        )
        self.stage4 = nn.Sequential(
            *[MinimalTransformerBlock(768, use_mlp=use_mlp, mlp_ratio=mlp_ratio) 
              for _ in range(12)]
        )

        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.head = nn.Linear(768, num_classes)
        
        self._init_head()
        del backbone
        
        # Report final count
        final_params = sum(p.numel() for p in self.parameters())
        diff_pct = 100 * (final_params - target_params) / target_params
        print(f"\nFinal parameter count: {final_params:,} ({final_params/1e6:.2f}M)")
        print(f"Target: {int(target_params):,} ({target_params/1e6:.2f}M)")
        print(f"Difference: {diff_pct:+.2f}%")

    def _init_head(self):
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        
        x = self.down2(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.stage3(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        x = self.down3(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.stage4(x)
        
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.head(x)
        
        return x

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    print("Testing Adaptive Hybrid Model")
    print("="*60)
    
    model = HybridConvNeXtV2Adaptive(num_classes=8, pretrained=False)
    
    # Test forward
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        y = model(x)
    
    print(f"\nForward test: {x.shape} -> {y.shape}")
    print(f"Output stats: mean={y.mean():.4f}, std={y.std():.4f}")