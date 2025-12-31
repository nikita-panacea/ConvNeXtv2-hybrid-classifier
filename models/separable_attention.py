# models/separable_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableSelfAttention(nn.Module):
    """
    Implements the separable self-attention from the paper (MobileViT-v2 style)
    Input: x [B, N, C]   (N = H*W tokens)
    Output: y [B, N, C]
    """

    def __init__(self, dim):
        super().__init__()
        self.WI = nn.Linear(dim, 1)     # produce scalar score per token
        self.WK = nn.Linear(dim, dim)
        self.WV = nn.Linear(dim, dim)
        self.WO = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, N, C]
        scores = self.WI(x)            # [B, N, 1]
        weights = F.softmax(scores, dim=1)  # [B, N, 1]
        keys = self.WK(x)              # [B, N, C]
        # context vector
        context = (weights * keys).sum(dim=1)  # [B, C]
        v = F.relu(self.WV(x))         # [B, N, C]
        z = v * context.unsqueeze(1)   # broadcast [B, N, C]
        y = self.WO(z)
        return y
