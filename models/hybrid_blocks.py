# models/hybrid_blocks.py
"""
Hybrid building blocks for ConvNeXtV2 + Transformer model.

This module provides the TransformerBlock class used in stages 3 and 4
of the hybrid model architecture.

Note: This file provides an alternative import path. The main implementation
is in hybrid_model.py.
"""

import torch
import torch.nn as nn
from models.separable_attention import (
    SeparableAttentionPaper,
    SeparableAttentionMinimal,
    SeparableAttentionReduced,
    SeparableAttentionMobileViT,
    create_separable_attention
)


class TransformerBlock(nn.Module):
    """
    Transformer block with separable self-attention and MLP.
    
    Architecture (from paper Fig. 4):
        Input → LayerNorm → Separable Attention → (+residual) → 
              → LayerNorm → MLP → (+residual) → Output
    
    This is the building block for stages 3 and 4 of the hybrid model.
    
    Args:
        dim: Input/output dimension (channel dimension)
        attn_type: Type of separable attention 
            - 'paper': Full implementation as per paper equations (5)-(10)
            - 'minimal': Minimal O(d) parameter attention
            - 'reduced': Reduced projection attention
            - 'mobilevit': MobileViT-v2 style attention
        mlp_ratio: Expansion ratio for MLP hidden dimension (default: 4.0)
        attn_proj_ratio: Projection ratio for 'reduced' attention type
    """
    def __init__(
        self, 
        dim: int, 
        attn_type: str = 'minimal',
        mlp_ratio: float = 4.0,
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
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, N, C] where B=batch, N=sequence, C=channels
        
        Returns:
            Output tensor of shape [B, N, C]
        """
        # Attention block with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def get_attention_params(self) -> int:
        """Get number of parameters in attention module."""
        return sum(p.numel() for p in self.attn.parameters())
    
    def get_mlp_params(self) -> int:
        """Get number of parameters in MLP module."""
        return sum(p.numel() for p in self.mlp.parameters())
    
    def get_norm_params(self) -> int:
        """Get number of parameters in normalization layers."""
        return sum(p.numel() for p in self.norm1.parameters()) + \
               sum(p.numel() for p in self.norm2.parameters())


class TransformerStage(nn.Module):
    """
    A full transformer stage consisting of multiple TransformerBlocks.
    
    This is a convenience wrapper for creating stages 3 or 4 of the hybrid model.
    
    Args:
        dim: Input/output dimension
        num_blocks: Number of transformer blocks (paper: 9 for stage 3, 12 for stage 4)
        attn_type: Type of separable attention
        mlp_ratio: MLP expansion ratio
        attn_proj_ratio: Projection ratio for reduced attention
    """
    def __init__(
        self,
        dim: int,
        num_blocks: int,
        attn_type: str = 'minimal',
        mlp_ratio: float = 4.0,
        attn_proj_ratio: float = 0.5,
    ):
        super().__init__()
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                attn_type=attn_type,
                mlp_ratio=mlp_ratio,
                attn_proj_ratio=attn_proj_ratio,
            )
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, N, C]
        
        Returns:
            Output tensor of shape [B, N, C]
        """
        return self.blocks(x)
    
    def count_parameters(self) -> int:
        """Count total parameters in the stage."""
        return sum(p.numel() for p in self.parameters())


# For quick parameter count estimation
def estimate_transformer_block_params(
    dim: int,
    attn_type: str = 'minimal',
    mlp_ratio: float = 4.0,
    attn_proj_ratio: float = 0.5,
) -> dict:
    """
    Estimate parameter count for a TransformerBlock without creating it.
    
    Args:
        dim: Block dimension
        attn_type: Attention type
        mlp_ratio: MLP expansion ratio
        attn_proj_ratio: Projection ratio for reduced attention
    
    Returns:
        Dictionary with parameter estimates by component
    """
    # Layer norms: 2 * 2 * dim (weight + bias for each)
    norm_params = 4 * dim
    
    # Attention params
    if attn_type == 'minimal':
        attn_params = 3 * dim  # WI (d) + key_scale (d) + value_scale (d)
    elif attn_type == 'paper':
        attn_params = dim + 3 * dim * dim  # WI (d) + WK (d²) + WV (d²) + WO (d²)
    elif attn_type == 'reduced':
        d_proj = max(1, int(dim * attn_proj_ratio))
        attn_params = dim + 3 * dim * d_proj  # WI + reduced projections
    elif attn_type == 'mobilevit':
        attn_params = 2 * dim * dim + dim  # input_proj + WI + output_proj
    else:
        attn_params = 0
    
    # MLP params: dim -> hidden -> dim
    hidden = max(1, int(dim * mlp_ratio))
    mlp_params = dim * hidden + hidden + hidden * dim + dim  # weights + biases
    
    return {
        'norm': norm_params,
        'attention': attn_params,
        'mlp': mlp_params,
        'total': norm_params + attn_params + mlp_params,
    }


def estimate_stage_params(
    dim: int,
    num_blocks: int,
    attn_type: str = 'minimal',
    mlp_ratio: float = 4.0,
    attn_proj_ratio: float = 0.5,
) -> int:
    """
    Estimate total parameters for a transformer stage.
    
    Args:
        dim: Stage dimension
        num_blocks: Number of blocks in the stage
        attn_type: Attention type
        mlp_ratio: MLP expansion ratio
        attn_proj_ratio: Projection ratio for reduced attention
    
    Returns:
        Estimated total parameters
    """
    block_params = estimate_transformer_block_params(
        dim, attn_type, mlp_ratio, attn_proj_ratio
    )['total']
    return num_blocks * block_params
