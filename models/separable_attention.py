# models/separable_attention.py
"""
Separable Self-Attention implementation matching the paper equations (5-10)
Based on MobileViT-v2 architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableSelfAttention(nn.Module):
    """
    Separable self-attention as described in the paper (Equations 5-10).
    
    Key operations:
    1. Context scores via latent token (Eq. 5)
    2. Context vector via weighted key projection (Eq. 6)
    3. Value projection with ReLU (Eq. 7)
    4. Element-wise multiplication (Eq. 8)
    5. Output projection (Eq. 9)
    
    Complexity: O(k) instead of O(k²) for standard attention
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Latent token projection (W_I in paper)
        self.context_score = nn.Linear(dim, 1, bias=False)
        
        # Key projection (W_K in paper)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        
        # Value projection (W_V in paper)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        
        # Output projection (W_O in paper)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, N, C] where B=batch, N=number of tokens, C=channels
        
        Returns:
            y: [B, N, C] attention output
        """
        B, N, C = x.shape
        
        # Equation 5: Context scores via softmax over latent token projection
        # cs = softmax(x @ W_I)
        scores = self.context_score(x)  # [B, N, 1]
        context_scores = F.softmax(scores, dim=1)  # [B, N, 1]
        
        # Equation 6: Context vector as weighted sum of keys
        # cv = Σ cs(i) · (x @ W_K)(i)
        keys = self.key_proj(x)  # [B, N, C]
        context_vector = torch.sum(context_scores * keys, dim=1)  # [B, C]
        
        # Equation 7: Value projection with ReLU
        # x_V = ReLU(x @ W_V)
        values = F.relu(self.value_proj(x))  # [B, N, C]
        
        # Equation 8: Element-wise multiplication (broadcasting context)
        # z = cv ⊙ x_V
        context_vector = context_vector.unsqueeze(1)  # [B, 1, C]
        z = context_vector * values  # [B, N, C]
        
        # Equation 9: Output projection
        # y = z @ W_O
        output = self.out_proj(z)  # [B, N, C]
        
        return output