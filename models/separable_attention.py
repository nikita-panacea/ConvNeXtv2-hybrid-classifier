# models/separable_attention.py
"""
Separable Self-Attention implementations for the hybrid ConvNeXtV2-ViT model.

This module provides multiple implementations of separable self-attention:
1. SeparableAttentionPaper: Full implementation matching paper equations (5)-(10)
2. SeparableAttentionMinimal: Minimal channel-wise attention with O(d) params
3. SeparableAttentionReduced: Reduced projection version with configurable ratio

Reference:
- Paper equations (5)-(10) describe the separable self-attention mechanism
- Original source: MobileViT-v2 (2022) by Mehta & Rastegari
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableAttentionPaper(nn.Module):
    """
    Full Separable Self-Attention as per paper equations (5)-(10).
    
    The separable attention mechanism computes attention with linear complexity O(n)
    instead of quadratic O(n²) by using a single latent token.
    
    Equations from paper:
        cs = softmax(x @ WI)           # Eq. 5: Context scores, WI ∈ R^d
        cv = Σ cs(i) * (x @ WK)(i)     # Eq. 6: Context vector, WK ∈ R^(d×d)
        xV = ReLU(x @ WV)              # Eq. 7: Value projection, WV ∈ R^(d×d)
        z = cv ⊙ xV                    # Eq. 8: Element-wise multiply (broadcast)
        y = z @ WO                     # Eq. 9: Output projection, WO ∈ R^(d×d)
    
    Parameters: d + 3*d² (one linear projection to scalar, three d×d projections)
    
    Args:
        dim: Input/output dimension (channel dimension)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # WI: Projects input to context scores (d -> 1)
        self.WI = nn.Linear(dim, 1, bias=False)
        
        # WK: Key projection (d -> d)
        self.WK = nn.Linear(dim, dim, bias=False)
        
        # WV: Value projection (d -> d)
        self.WV = nn.Linear(dim, dim, bias=False)
        
        # WO: Output projection (d -> d)
        self.WO = nn.Linear(dim, dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize projections
        nn.init.xavier_uniform_(self.WI.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WO.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, C] where B=batch, N=sequence length, C=channels
        
        Returns:
            Output tensor of shape [B, N, C]
        """
        B, N, C = x.shape
        
        # Eq. 5: Context scores
        # cs = softmax(x @ WI), cs ∈ R^(B×N×1)
        cs = F.softmax(self.WI(x), dim=1)  # [B, N, 1]
        
        # Key projection: xK = x @ WK, xK ∈ R^(B×N×C)
        xK = self.WK(x)  # [B, N, C]
        
        # Eq. 6: Context vector (weighted sum of keys)
        # cv = Σ cs(i) * xK(i), cv ∈ R^(B×C)
        cv = torch.sum(cs * xK, dim=1)  # [B, C]
        
        # Eq. 7: Value projection with ReLU
        # xV = ReLU(x @ WV), xV ∈ R^(B×N×C)
        xV = F.relu(self.WV(x))  # [B, N, C]
        
        # Eq. 8: Broadcast multiply
        # z = cv ⊙ xV, z ∈ R^(B×N×C)
        z = cv.unsqueeze(1) * xV  # [B, N, C]
        
        # Eq. 9: Output projection
        # y = z @ WO, y ∈ R^(B×N×C)
        y = self.WO(z)  # [B, N, C]
        
        return y
    
    @staticmethod
    def param_count(dim: int) -> int:
        """Calculate parameter count for given dimension."""
        return dim + 3 * dim * dim


class SeparableAttentionMinimal(nn.Module):
    """
    Minimal Separable Self-Attention with O(d) parameters.
    
    This is a simplified version using only channel-wise operations:
    - Single linear projection for context scores
    - Channel-wise scaling parameters instead of full projections
    
    Parameters: ~3d (much fewer than the full version)
    
    Note: This does NOT match the paper equations exactly but is parameter-efficient.
    
    Args:
        dim: Input/output dimension
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Latent token projection (d -> 1)
        self.WI = nn.Linear(dim, 1, bias=False)
        
        # Channel-wise parameters (instead of d×d matrices)
        self.key_scale = nn.Parameter(torch.ones(dim))
        self.value_scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, C]
        
        Returns:
            Output tensor of shape [B, N, C]
        """
        # Context scores
        scores = self.WI(x)  # [B, N, 1]
        weights = F.softmax(scores, dim=1)  # [B, N, 1]
        
        # Context vector (channel-wise weighted sum)
        context = torch.sum(weights * x, dim=1)  # [B, C]
        
        # Apply key scaling to context
        context = context * self.key_scale  # [B, C]
        
        # Broadcast multiply and apply value scaling
        out = x * context.unsqueeze(1)  # [B, N, C]
        out = out * self.value_scale  # [B, N, C]
        
        return out
    
    @staticmethod
    def param_count(dim: int) -> int:
        """Calculate parameter count for given dimension."""
        return 3 * dim


class SeparableAttentionReduced(nn.Module):
    """
    Separable Self-Attention with reduced projection dimension.
    
    Uses intermediate dimension d_proj < d for key/value projections to reduce parameters.
    This provides a middle ground between minimal and full implementations.
    
    Parameters: d + 3 * d * d_proj
    
    Args:
        dim: Input/output dimension
        proj_ratio: Ratio for projection dimension (d_proj = dim * proj_ratio)
    """
    def __init__(self, dim: int, proj_ratio: float = 0.5):
        super().__init__()
        self.dim = dim
        self.d_proj = max(1, int(dim * proj_ratio))
        
        # WI: Context score projection (d -> 1)
        self.WI = nn.Linear(dim, 1, bias=False)
        
        # Reduced projections
        self.WK = nn.Linear(dim, self.d_proj, bias=False)  # d -> d_proj
        self.WV = nn.Linear(dim, self.d_proj, bias=False)  # d -> d_proj
        self.WO = nn.Linear(self.d_proj, dim, bias=False)  # d_proj -> d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, C]
        
        Returns:
            Output tensor of shape [B, N, C]
        """
        # Context scores
        cs = F.softmax(self.WI(x), dim=1)  # [B, N, 1]
        
        # Key projection and context vector
        xK = self.WK(x)  # [B, N, d_proj]
        cv = torch.sum(cs * xK, dim=1)  # [B, d_proj]
        
        # Value projection with ReLU
        xV = F.relu(self.WV(x))  # [B, N, d_proj]
        
        # Broadcast multiply
        z = cv.unsqueeze(1) * xV  # [B, N, d_proj]
        
        # Output projection
        y = self.WO(z)  # [B, N, C]
        
        return y
    
    @staticmethod
    def param_count(dim: int, proj_ratio: float = 0.5) -> int:
        """Calculate parameter count for given dimension and ratio."""
        d_proj = max(1, int(dim * proj_ratio))
        return dim + 3 * dim * d_proj


class SeparableAttentionMobileViT(nn.Module):
    """
    MobileViT-v2 style Separable Self-Attention.
    
    This implementation closely follows the original MobileViT-v2 paper,
    using input and output projections with channel-wise context aggregation.
    
    Parameters: 2*d² + d
    
    Args:
        dim: Input/output dimension
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Input projection
        self.input_proj = nn.Linear(dim, dim, bias=False)
        
        # Context score projection
        self.WI = nn.Linear(dim, 1, bias=False)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, N, C]
        
        Returns:
            Output tensor of shape [B, N, C]
        """
        # Input projection
        x_proj = self.input_proj(x)  # [B, N, C]
        
        # Context scores
        cs = F.softmax(self.WI(x_proj), dim=1)  # [B, N, 1]
        
        # Context vector
        cv = torch.sum(cs * x_proj, dim=1)  # [B, C]
        
        # Broadcast and multiply with original input
        out = x * cv.unsqueeze(1)  # [B, N, C]
        
        # Output projection
        out = self.output_proj(out)  # [B, N, C]
        
        return out
    
    @staticmethod
    def param_count(dim: int) -> int:
        """Calculate parameter count for given dimension."""
        return 2 * dim * dim + dim


# Factory function
def create_separable_attention(dim: int, attn_type: str = 'paper', **kwargs) -> nn.Module:
    """
    Factory function to create separable attention module.
    
    Args:
        dim: Input/output dimension
        attn_type: Type of attention - 'paper', 'minimal', 'reduced', 'mobilevit'
        **kwargs: Additional arguments for specific attention types
    
    Returns:
        Separable attention module
    """
    if attn_type == 'paper':
        return SeparableAttentionPaper(dim)
    elif attn_type == 'minimal':
        return SeparableAttentionMinimal(dim)
    elif attn_type == 'reduced':
        proj_ratio = kwargs.get('proj_ratio', 0.5)
        return SeparableAttentionReduced(dim, proj_ratio=proj_ratio)
    elif attn_type == 'mobilevit':
        return SeparableAttentionMobileViT(dim)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}. "
                        f"Choose from: 'paper', 'minimal', 'reduced', 'mobilevit'")


# For backward compatibility
SeparableSelfAttention = SeparableAttentionMinimal
