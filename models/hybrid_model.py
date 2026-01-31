# models/hybrid_model.py
"""
Hybrid ConvNeXtV2 + Separable Self-Attention Model for Skin Lesion Classification.

This model combines:
- ConvNeXtV2 blocks in stages 1-2 for fine-grained local feature extraction
- Transformer blocks with separable self-attention in stages 3-4 for global context

Paper Architecture:
- Stage depths: [3, 3, 9, 12] (ConvNeXtV2 in 1-2, Transformer in 3-4)
- Target parameters: ~21.92M
- Input resolution: 224x224
- Output: 8-class classification (ISIC 2019 dataset)

Paper Results:
- Accuracy: 93.48%
- Precision: 93.24%
- Recall: 90.70%
- F1-Score: 91.82%

Reference:
- "A robust deep learning framework for multiclass skin cancer classification"
- Ozdemir & Pacal, Scientific Reports (2025) 15:4938
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import urllib.request
import os

from models.convnextv2 import ConvNeXtV2
from models.separable_attention import (
    SeparableAttentionPaper,
    SeparableAttentionMinimal,
    SeparableAttentionReduced,
    SeparableAttentionMobileViT,
    create_separable_attention
)


# ==============================================================================
# Configuration
# ==============================================================================

# ConvNeXtV2 backbone configurations
BACKBONE_CONFIGS = {
    'atto':  {'depths': [2, 2, 6, 2], 'dims': [40, 80, 160, 320]},
    'femto': {'depths': [2, 2, 6, 2], 'dims': [48, 96, 192, 384]},
    'pico':  {'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512]},
    'nano':  {'depths': [2, 2, 8, 2], 'dims': [80, 160, 320, 640]},
    'tiny':  {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
    'base':  {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
}

# Available pretrained weight URLs
PRETRAINED_URLS = {
    'atto': {
        'fcmae_1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_atto_1k_224_fcmae.pt',
        'ft_1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt',
    },
    'femto': {
        'fcmae_1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_femto_1k_224_fcmae.pt',
        'ft_1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt',
    },
    'pico': {
        'fcmae_1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_pico_1k_224_fcmae.pt',
        'ft_1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt',
    },
    'nano': {
        'fcmae_1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_nano_1k_224_fcmae.pt',
        'ft_1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt',
        'ft_22k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_224_ema.pt',
    },
    'tiny': {
        'fcmae_1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_tiny_1k_224_fcmae.pt',
        'ft_1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt',
        'ft_22k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt',
    },
    'base': {
        'fcmae_1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_base_1k_224_fcmae.pt',
        'ft_1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt',
        'ft_22k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt',
    },
}


# ==============================================================================
# Transformer Block
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    Transformer block with separable self-attention and MLP.
    
    Architecture (from paper Fig. 4):
        Input → LayerNorm → Separable Attention → (+residual) → 
              → LayerNorm → MLP → (+residual) → Output
    
    Args:
        dim: Input/output dimension (channel dimension)
        attn_type: Type of separable attention ('paper', 'minimal', 'reduced', 'mobilevit')
        mlp_ratio: Expansion ratio for MLP hidden dimension
        attn_proj_ratio: Projection ratio for reduced attention (only used if attn_type='reduced')
    """
    def __init__(
        self, 
        dim: int, 
        attn_type: str = 'minimal',
        mlp_ratio: float = 1.125,
        attn_proj_ratio: float = 0.5
    ):
        super().__init__()
        
        self.dim = dim
        self.attn_type = attn_type
        self.mlp_ratio = mlp_ratio
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Separable self-attention
        self.attn = create_separable_attention(
            dim, 
            attn_type=attn_type,
            proj_ratio=attn_proj_ratio
        )
        
        # MLP (channel-wise processing)
        hidden_dim = max(1, int(dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, C]
        
        Returns:
            Output tensor of shape [B, N, C]
        """
        # Attention block with residual
        x = x + self.attn(self.norm1(x))
        
        # MLP block with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


# ==============================================================================
# Hybrid Model (Configurable)
# ==============================================================================

class HybridConvNeXtV2(nn.Module):
    """
    Hybrid model combining ConvNeXtV2 and Transformer blocks.
    
    Architecture:
        - Stem: 4x4 conv + LayerNorm (patch embedding)
        - Stage 1: ConvNeXtV2 blocks (depth=3 for tiny)
        - Downsample 1: LayerNorm + 2x2 conv
        - Stage 2: ConvNeXtV2 blocks (depth=3 for tiny)
        - Downsample 2: LayerNorm + 2x2 conv
        - Stage 3: Transformer blocks (9 blocks as per paper)
        - Downsample 3: LayerNorm + 2x2 conv
        - Stage 4: Transformer blocks (12 blocks as per paper)
        - Head: Global avg pool + LayerNorm + Linear
    
    Args:
        backbone_variant: ConvNeXtV2 variant ('atto', 'femto', 'pico', 'nano', 'tiny', 'base')
        attn_type: Separable attention type ('paper', 'minimal', 'reduced', 'mobilevit')
        mlp_ratio: MLP expansion ratio in Transformer blocks
        attn_proj_ratio: Projection ratio for reduced attention
        stage3_blocks: Number of Transformer blocks in stage 3 (paper: 9)
        stage4_blocks: Number of Transformer blocks in stage 4 (paper: 12)
        num_classes: Number of output classes (ISIC 2019: 8)
        drop_path_rate: Stochastic depth rate
        pretrained: Whether to load pretrained weights
        pretrained_weights: Type of pretrained weights ('ft_1k', 'ft_22k', 'fcmae_1k')
    """
    def __init__(
        self,
        backbone_variant: str = 'tiny',
        attn_type: str = 'minimal',
        mlp_ratio: float = 1.125,
        attn_proj_ratio: float = 0.5,
        stage3_blocks: int = 9,
        stage4_blocks: int = 12,
        num_classes: int = 8,
        drop_path_rate: float = 0.0,
        pretrained: bool = True,
        pretrained_weights: str = 'ft_1k',
    ):
        super().__init__()
        
        # Store configuration
        self.backbone_variant = backbone_variant
        self.attn_type = attn_type
        self.mlp_ratio = mlp_ratio
        self.stage3_blocks = stage3_blocks
        self.stage4_blocks = stage4_blocks
        self.num_classes = num_classes
        
        # Get backbone configuration
        if backbone_variant not in BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone variant: {backbone_variant}. "
                           f"Choose from: {list(BACKBONE_CONFIGS.keys())}")
        
        config = BACKBONE_CONFIGS[backbone_variant]
        depths = config['depths']
        dims = config['dims']
        self.dims = dims
        
        # Create backbone with only stages 1-2 (set depths 3,4 to 0)
        backbone = ConvNeXtV2(
            in_chans=3,
            num_classes=1000,
            depths=[depths[0], depths[1], 0, 0],  # Only stages 1 and 2
            dims=dims,
            drop_path_rate=drop_path_rate,
        )
        
        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights(backbone, backbone_variant, pretrained_weights)
        
        # Extract components from backbone
        self.stem = backbone.downsample_layers[0]
        self.down1 = backbone.downsample_layers[1]
        self.down2 = backbone.downsample_layers[2]
        self.down3 = backbone.downsample_layers[3]
        
        self.stage1 = backbone.stages[0]  # ConvNeXtV2 blocks
        self.stage2 = backbone.stages[1]  # ConvNeXtV2 blocks
        
        # Transformer stages (3 and 4)
        self.stage3 = nn.Sequential(*[
            TransformerBlock(
                dim=dims[2],
                attn_type=attn_type,
                mlp_ratio=mlp_ratio,
                attn_proj_ratio=attn_proj_ratio
            )
            for _ in range(stage3_blocks)
        ])
        
        self.stage4 = nn.Sequential(*[
            TransformerBlock(
                dim=dims[3],
                attn_type=attn_type,
                mlp_ratio=mlp_ratio,
                attn_proj_ratio=attn_proj_ratio
            )
            for _ in range(stage4_blocks)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dims[3])
        self.head = nn.Linear(dims[3], num_classes)
        
        # Clean up
        del backbone
    
    def _load_pretrained_weights(
        self, 
        backbone: nn.Module, 
        variant: str, 
        weights_type: str
    ) -> Tuple[int, int]:
        """Load pretrained weights into backbone."""
        if variant not in PRETRAINED_URLS:
            print(f"Warning: No pretrained weights available for {variant}")
            return 0, 0
        
        if weights_type not in PRETRAINED_URLS.get(variant, {}):
            print(f"Warning: Weights type '{weights_type}' not available for {variant}. "
                  f"Available: {list(PRETRAINED_URLS.get(variant, {}).keys())}")
            return 0, 0
        
        url = PRETRAINED_URLS[variant][weights_type]
        
        try:
            # Load weights using torch.hub
            ckpt = torch.hub.load_state_dict_from_url(url, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
            
            # Load with strict=False to ignore missing keys
            missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
            
            # Count matched keys
            matched = len(state_dict) - len(unexpected)
            total = len(state_dict)
            
            print(f"Loaded pretrained weights: {matched}/{total} keys from {weights_type}")
            
            return matched, total
            
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
            return 0, 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape [B, 3, H, W] (typically [B, 3, 224, 224])
        
        Returns:
            Logits tensor of shape [B, num_classes]
        """
        # Stage 1: Stem + ConvNeXtV2 blocks
        x = self.stem(x)  # [B, C1, H/4, W/4]
        x = self.stage1(x)  # [B, C1, H/4, W/4]
        
        # Stage 2: Downsample + ConvNeXtV2 blocks
        x = self.down1(x)  # [B, C2, H/8, W/8]
        x = self.stage2(x)  # [B, C2, H/8, W/8]
        
        # Stage 3: Downsample + Transformer blocks
        x = self.down2(x)  # [B, C3, H/16, W/16]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C] where N = H*W
        x = self.stage3(x)  # [B, N, C3]
        
        # Stage 4: Downsample + Transformer blocks
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, C3, H/16, W/16]
        x = self.down3(x)  # [B, C4, H/32, W/32]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C4]
        x = self.stage4(x)  # [B, N, C4]
        
        # Classification head
        x = x.mean(dim=1)  # Global average pooling [B, C4]
        x = self.norm(x)  # [B, C4]
        x = self.head(x)  # [B, num_classes]
        
        return x
    
    def get_config(self) -> Dict:
        """Return model configuration."""
        return {
            'backbone_variant': self.backbone_variant,
            'attn_type': self.attn_type,
            'mlp_ratio': self.mlp_ratio,
            'stage3_blocks': self.stage3_blocks,
            'stage4_blocks': self.stage4_blocks,
            'num_classes': self.num_classes,
            'dims': self.dims,
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        result = {
            'stem': sum(p.numel() for p in self.stem.parameters()),
            'stage1': sum(p.numel() for p in self.stage1.parameters()),
            'down1': sum(p.numel() for p in self.down1.parameters()),
            'stage2': sum(p.numel() for p in self.stage2.parameters()),
            'down2': sum(p.numel() for p in self.down2.parameters()),
            'stage3': sum(p.numel() for p in self.stage3.parameters()),
            'down3': sum(p.numel() for p in self.down3.parameters()),
            'stage4': sum(p.numel() for p in self.stage4.parameters()),
            'norm': sum(p.numel() for p in self.norm.parameters()),
            'head': sum(p.numel() for p in self.head.parameters()),
        }
        result['total'] = sum(result.values())
        result['convnext_stages'] = result['stage1'] + result['stage2']
        result['transformer_stages'] = result['stage3'] + result['stage4']
        return result


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_hybrid_model(
    config: str = 'paper_target',
    num_classes: int = 8,
    pretrained: bool = True,
) -> HybridConvNeXtV2:
    """
    Create a hybrid model with predefined configurations.
    
    Args:
        config: Configuration name
            - 'paper_target': Closest match to paper's 21.92M params
            - 'paper_full_attn': Using paper's full attention implementation
            - 'minimal': Minimal attention for fastest inference
            - 'tiny_4x': Tiny backbone with standard 4x MLP
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
    
    Returns:
        HybridConvNeXtV2 model instance
    """
    configs = {
        'paper_target': {
            'backbone_variant': 'tiny',
            'attn_type': 'minimal',
            'mlp_ratio': 1.125,
            'pretrained_weights': 'ft_1k',
        },
        'paper_full_attn': {
            'backbone_variant': 'pico',
            'attn_type': 'paper',
            'mlp_ratio': 1.0,
            'pretrained_weights': 'ft_1k',
        },
        'minimal': {
            'backbone_variant': 'nano',
            'attn_type': 'minimal',
            'mlp_ratio': 1.0,
            'pretrained_weights': 'ft_1k',
        },
        'tiny_4x': {
            'backbone_variant': 'tiny',
            'attn_type': 'minimal',
            'mlp_ratio': 4.0,
            'pretrained_weights': 'ft_1k',
        },
    }
    
    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Choose from: {list(configs.keys())}")
    
    cfg = configs[config]
    return HybridConvNeXtV2(
        num_classes=num_classes,
        pretrained=pretrained,
        **cfg
    )


# ==============================================================================
# Utility
# ==============================================================================

def print_model_summary(model: HybridConvNeXtV2):
    """Print model summary with parameter counts."""
    config = model.get_config()
    params = model.count_parameters()
    
    print("=" * 60)
    print("HYBRID CONVNEXTV2 MODEL SUMMARY")
    print("=" * 60)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    print(f"\nParameter counts:")
    for name in ['stem', 'stage1', 'down1', 'stage2', 'down2', 'stage3', 'down3', 'stage4', 'norm', 'head']:
        print(f"  {name:<15}: {params[name]/1e6:.3f}M")
    
    print(f"\n  ConvNeXtV2 (1-2): {params['convnext_stages']/1e6:.3f}M")
    print(f"  Transformer (3-4): {params['transformer_stages']/1e6:.3f}M")
    print(f"  Total:            {params['total']/1e6:.3f}M")
    print("=" * 60)


# ==============================================================================
# Main (for testing)
# ==============================================================================

if __name__ == '__main__':
    print("Testing HybridConvNeXtV2 model...")
    
    # Test default configuration
    model = HybridConvNeXtV2(num_classes=8, pretrained=False)
    print_model_summary(model)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print(f"\nForward pass: {x.shape} -> {y.shape}")
    
    # Test different configurations
    print("\n\nTesting different configurations:")
    for backbone in ['nano', 'tiny']:
        for attn_type in ['minimal', 'paper']:
            for mlp_ratio in [1.0, 1.125]:
                model = HybridConvNeXtV2(
                    backbone_variant=backbone,
                    attn_type=attn_type,
                    mlp_ratio=mlp_ratio,
                    pretrained=False,
                )
                total = sum(p.numel() for p in model.parameters()) / 1e6
                print(f"  {backbone}/{attn_type}/mlp={mlp_ratio}: {total:.2f}M params")
                del model
