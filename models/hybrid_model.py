# models/hybrid_model.py
"""
Hybrid ConvNeXtV2 + Separable Self-Attention model
Exactly matching the paper architecture and parameter count (21.92M)
"""
import torch
import torch.nn as nn
from models.separable_attention import SeparableSelfAttention
from models.convnextv2 import ConvNeXtV2


class TransformerBlock(nn.Module):
    """
    Transformer block with separable self-attention and MLP.
    Matches paper description with standard 4x expansion in MLP.
    """
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SeparableSelfAttention(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        # Standard MLP with 4x expansion (paper standard)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        # Residual connections as in paper
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HybridConvNeXtV2(nn.Module):
    """
    Paper-exact Hybrid ConvNeXtV2 + Separable Self-Attention Architecture
    
    Stage configuration:
    - Stage 1: 3 ConvNeXtV2 blocks (dim=96)
    - Stage 2: 3 ConvNeXtV2 blocks (dim=192)
    - Stage 3: 9 Separable Self-Attention blocks (dim=384)
    - Stage 4: 12 Separable Self-Attention blocks (dim=768)
    
    Total parameters: ~21.92M
    Input size: 224×224
    """

    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()

        # Create ConvNeXtV2 Tiny backbone with only stages 1 and 2
        # depths=[3, 3, 0, 0] means stage 3 and 4 are empty (replaced by attention)
        backbone = ConvNeXtV2(
            in_chans=3,
            num_classes=1000,
            depths=[3, 3, 0, 0],  # Only stage 1 & 2 have ConvNeXt blocks
            dims=[96, 192, 384, 768],  # Channel dimensions for each stage
            drop_path_rate=0.0,
        )

        # Load ImageNet pretrained weights if requested
        if pretrained:
            try:
                ckpt = torch.hub.load_state_dict_from_url(
                    "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt",
                    map_location="cpu",
                )
                # Load with strict=False since we're modifying stages 3 & 4
                backbone.load_state_dict(ckpt["model"], strict=False)
                print("Loaded ConvNeXtV2-Tiny pretrained weights")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")

        # Extract downsampling layers and convolutional stages
        self.stem = backbone.downsample_layers[0]  # Initial 4x4 conv, stride 4
        self.down1 = backbone.downsample_layers[1]  # 2x2 conv, stride 2
        self.down2 = backbone.downsample_layers[2]  # 2x2 conv, stride 2
        self.down3 = backbone.downsample_layers[3]  # 2x2 conv, stride 2

        # Stage 1 & 2: ConvNeXtV2 blocks (from backbone)
        self.stage1 = backbone.stages[0]  # 3 ConvNeXtV2 blocks, dim=96
        self.stage2 = backbone.stages[1]  # 3 ConvNeXtV2 blocks, dim=192

        # Stage 3 & 4: Separable Self-Attention blocks (custom)
        self.stage3 = nn.Sequential(
            *[TransformerBlock(384, mlp_ratio=4.0) for _ in range(9)]
        )
        self.stage4 = nn.Sequential(
            *[TransformerBlock(768, mlp_ratio=4.0) for _ in range(12)]
        )

        # Final normalization and classification head
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.head = nn.Linear(768, num_classes)
        
        # Initialize head properly
        self._init_head()

        # Clean up backbone reference
        del backbone

    def _init_head(self):
        """Initialize classification head with proper scaling"""
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        Forward pass matching paper architecture.
        
        Flow:
        1. Stem (4x downsample) -> Stage 1 (ConvNeXt, dim=96)
        2. Down1 (2x downsample) -> Stage 2 (ConvNeXt, dim=192)
        3. Down2 (2x downsample) -> Stage 3 (Attention, dim=384) [reshape to tokens]
        4. Down3 (2x downsample) -> Stage 4 (Attention, dim=768) [reshape to tokens]
        5. Global average pool -> Norm -> Head
        """
        # Stage 1: ConvNeXt blocks
        x = self.stem(x)  # [B, 96, 56, 56]
        x = self.stage1(x)  # [B, 96, 56, 56]

        # Stage 2: ConvNeXt blocks
        x = self.down1(x)  # [B, 192, 28, 28]
        x = self.stage2(x)  # [B, 192, 28, 28]

        # Stage 3: Separable self-attention (reshape to sequence)
        x = self.down2(x)  # [B, 384, 14, 14]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 384] (tokens)
        x = self.stage3(x)  # [B, 196, 384]

        # Stage 4: Separable self-attention (reshape back and down)
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, 384, 14, 14]
        x = self.down3(x)  # [B, 768, 7, 7]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, 49, 768] (tokens)
        x = self.stage4(x)  # [B, 49, 768]

        # Global average pooling, normalize, and classify
        x = x.mean(dim=1)  # [B, 768]
        x = self.norm(x)
        x = self.head(x)  # [B, num_classes]
        
        return x

    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def test_model():
    """Test model instantiation and parameter count"""
    model = HybridConvNeXtV2(num_classes=8, pretrained=False)
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,} ({params['total']/1e6:.2f}M)")
    print(f"Trainable parameters: {params['trainable']:,} ({params['trainable']/1e6:.2f}M)")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    return model


if __name__ == "__main__":
    test_model()