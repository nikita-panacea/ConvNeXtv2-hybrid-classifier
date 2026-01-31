# models/__init__.py
"""
Models module for Hybrid ConvNeXtV2 + Separable Attention skin lesion classifier.

This module provides:
- HybridConvNeXtV2: Main hybrid model class
- TransformerBlock: Transformer block with separable attention
- Separable attention implementations
- ConvNeXtV2 backbone

Usage:
    from models import HybridConvNeXtV2

    model = HybridConvNeXtV2(
        backbone_variant='tiny',
        attn_type='minimal',
        mlp_ratio=1.125,
        num_classes=8,
        pretrained=True,
    )
"""

from models.hybrid_model import (
    HybridConvNeXtV2,
    TransformerBlock,
    BACKBONE_CONFIGS,
    PRETRAINED_URLS,
    create_hybrid_model,
    print_model_summary,
)

from models.separable_attention import (
    SeparableAttentionPaper,
    SeparableAttentionMinimal,
    SeparableAttentionReduced,
    SeparableAttentionMobileViT,
    create_separable_attention,
    SeparableSelfAttention,  # Backward compatibility
)

from models.convnextv2 import (
    ConvNeXtV2,
    Block as ConvNeXtV2Block,
    convnextv2_atto,
    convnextv2_femto,
    convnext_pico,
    convnextv2_nano,
    convnextv2_tiny,
    convnextv2_base,
    convnextv2_large,
    convnextv2_huge,
)

__all__ = [
    # Main model
    'HybridConvNeXtV2',
    'create_hybrid_model',
    'print_model_summary',
    
    # Building blocks
    'TransformerBlock',
    'ConvNeXtV2Block',
    
    # Attention modules
    'SeparableAttentionPaper',
    'SeparableAttentionMinimal',
    'SeparableAttentionReduced',
    'SeparableAttentionMobileViT',
    'create_separable_attention',
    'SeparableSelfAttention',
    
    # ConvNeXtV2
    'ConvNeXtV2',
    'convnextv2_atto',
    'convnextv2_femto',
    'convnext_pico',
    'convnextv2_nano',
    'convnextv2_tiny',
    'convnextv2_base',
    'convnextv2_large',
    'convnextv2_huge',
    
    # Configuration
    'BACKBONE_CONFIGS',
    'PRETRAINED_URLS',
]
