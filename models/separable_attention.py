# models/separable_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableSelfAttention(nn.Module):
    """
    Fully separable self-attention
    Parameter complexity: O(C)
    """

    def __init__(self, dim):
        super().__init__()

        # latent token projection
        self.WI = nn.Linear(dim, 1, bias=False)

        # channel-wise parameters (NO CxC)
        self.key_scale = nn.Parameter(torch.ones(dim))
        self.value_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        x: [B, N, C]
        """
        # latent attention weights
        scores = self.WI(x)                    # [B, N, 1]
        weights = F.softmax(scores, dim=1)

        # context vector (channel-wise)
        context = torch.sum(weights * x, dim=1)   # [B, C]

        # channel-wise modulation
        out = x * context.unsqueeze(1)
        out = out * self.value_scale

        return out
