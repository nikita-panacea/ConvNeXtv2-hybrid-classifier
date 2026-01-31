# experiments/comprehensive_param_search.py
"""
Comprehensive parameter search to find configurations matching the paper's ~21.92M parameters.

This script:
1. Implements multiple separable attention variants matching paper equations
2. Tests all ConvNeXtV2 backbone variants (atto/femto/pico/nano/tiny)
3. Tests different MLP expansion ratios
4. Downloads and loads all available pretrained weights
5. Produces a detailed comparison report

Paper target: 21.92M parameters
Paper results: 93.48% accuracy, 93.24% precision, 90.70% recall, 91.82% F1-score

From the paper:
- Stage depths: [3, 3, 9, 12] (ConvNeXtV2 in stages 1-2, Transformer in stages 3-4)
- Separable self-attention equations (5)-(10)
- ImageNet pretrained weights for ConvNeXtV2 stages
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import OrderedDict
import urllib.request
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==============================================================================
# Configuration
# ==============================================================================

TARGET_PARAMS_M = 21.92  # Paper's target parameter count in millions

BACKBONE_CONFIGS = {
    'atto':  {'depths': [2, 2, 6, 2], 'dims': [40, 80, 160, 320], 'full_params_M': 3.7},
    'femto': {'depths': [2, 2, 6, 2], 'dims': [48, 96, 192, 384], 'full_params_M': 5.2},
    'pico':  {'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512], 'full_params_M': 9.1},
    'nano':  {'depths': [2, 2, 8, 2], 'dims': [80, 160, 320, 640], 'full_params_M': 15.6},
    'tiny':  {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768], 'full_params_M': 28.6},
    'base':  {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024], 'full_params_M': 89.0},
}

# All available pretrained weight URLs
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
# Utility Functions
# ==============================================================================

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    return tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_first and channels_last formats."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """Global Response Normalization layer."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ==============================================================================
# ConvNeXtV2 Block
# ==============================================================================

class ConvNeXtV2Block(nn.Module):
    """ConvNeXtV2 Block with depthwise conv, GRN, and residual connection."""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return residual + self.drop_path(x)


# ==============================================================================
# Separable Self-Attention Variants
# ==============================================================================

class SeparableAttentionMinimal(nn.Module):
    """
    Minimal separable attention with channel-wise operations only.
    Parameters: O(d) - approximately 3d parameters
    
    This is NOT matching the paper's equations but is very parameter-efficient.
    """
    def __init__(self, dim):
        super().__init__()
        self.WI = nn.Linear(dim, 1, bias=False)  # d params
        self.key_scale = nn.Parameter(torch.ones(dim))  # d params
        self.value_scale = nn.Parameter(torch.ones(dim))  # d params

    def forward(self, x):
        # x: [B, N, C]
        scores = self.WI(x)  # [B, N, 1]
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)  # [B, C]
        out = x * context.unsqueeze(1)
        out = out * self.value_scale
        return out

    @staticmethod
    def param_count(dim):
        return 3 * dim


class SeparableAttentionPaper(nn.Module):
    """
    Full separable attention as per paper equations (5)-(10).
    
    cs = softmax(x @ WI)           # WI ∈ R^d -> 1 (scores)
    xK = x @ WK                    # WK ∈ R^(d×d) 
    cv = Σ cs(i) * xK(i)           # context vector
    xV = ReLU(x @ WV)              # WV ∈ R^(d×d)
    z = cv ⊙ xV                    # element-wise multiply (broadcast)
    y = z @ WO                     # WO ∈ R^(d×d)
    
    Parameters: d + 3*d² = O(d²)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.WI = nn.Linear(dim, 1, bias=False)   # d params
        self.WK = nn.Linear(dim, dim, bias=False)  # d² params
        self.WV = nn.Linear(dim, dim, bias=False)  # d² params
        self.WO = nn.Linear(dim, dim, bias=False)  # d² params

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        
        # Context scores (Eq. 5)
        cs = torch.softmax(self.WI(x), dim=1)  # [B, N, 1]
        
        # Key projection
        xK = self.WK(x)  # [B, N, C]
        
        # Context vector (Eq. 6)
        cv = torch.sum(cs * xK, dim=1)  # [B, C]
        
        # Value projection with ReLU (Eq. 7)
        xV = torch.relu(self.WV(x))  # [B, N, C]
        
        # Broadcast multiply (Eq. 8)
        z = cv.unsqueeze(1) * xV  # [B, N, C]
        
        # Output projection (Eq. 9)
        y = self.WO(z)  # [B, N, C]
        
        return y

    @staticmethod
    def param_count(dim):
        return dim + 3 * dim * dim


class SeparableAttentionReduced(nn.Module):
    """
    Separable attention with reduced projection dimension.
    Uses d_proj < d for intermediate projections to reduce params.
    
    Parameters: d + d*d_proj + d*d_proj + d_proj*d = d + 3*d*d_proj
    """
    def __init__(self, dim, proj_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.d_proj = max(1, int(dim * proj_ratio))
        
        self.WI = nn.Linear(dim, 1, bias=False)
        self.WK = nn.Linear(dim, self.d_proj, bias=False)
        self.WV = nn.Linear(dim, self.d_proj, bias=False)
        self.WO = nn.Linear(self.d_proj, dim, bias=False)

    def forward(self, x):
        cs = torch.softmax(self.WI(x), dim=1)
        xK = self.WK(x)
        cv = torch.sum(cs * xK, dim=1)
        xV = torch.relu(self.WV(x))
        z = cv.unsqueeze(1) * xV
        y = self.WO(z)
        return y

    @staticmethod
    def param_count(dim, proj_ratio=0.5):
        d_proj = max(1, int(dim * proj_ratio))
        return dim + 3 * dim * d_proj


class SeparableAttentionMobileViT(nn.Module):
    """
    MobileViT-v2 style separable attention.
    Reference: MobileViT-v2 paper (2022) - the source of separable attention.
    
    Uses input projection and output projection with channel-wise context.
    Parameters: 2*d² + d
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.input_proj = nn.Linear(dim, dim, bias=False)  # d² params
        self.WI = nn.Linear(dim, 1, bias=False)  # d params
        self.output_proj = nn.Linear(dim, dim, bias=False)  # d² params

    def forward(self, x):
        # Input projection
        x_proj = self.input_proj(x)  # [B, N, C]
        
        # Context scores
        cs = torch.softmax(self.WI(x_proj), dim=1)  # [B, N, 1]
        
        # Context vector
        cv = torch.sum(cs * x_proj, dim=1)  # [B, C]
        
        # Broadcast and multiply
        out = x * cv.unsqueeze(1)  # [B, N, C]
        
        # Output projection
        out = self.output_proj(out)  # [B, N, C]
        
        return out

    @staticmethod
    def param_count(dim):
        return 2 * dim * dim + dim


# ==============================================================================
# Transformer Block
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    Transformer block with separable self-attention and MLP.
    
    Structure (from paper Fig. 4):
    Input → LayerNorm → Separable Attention → (+residual) → LayerNorm → MLP → (+residual) → Output
    """
    def __init__(self, dim, attn_type='paper', mlp_ratio=4.0, attn_proj_ratio=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Select attention type
        if attn_type == 'minimal':
            self.attn = SeparableAttentionMinimal(dim)
        elif attn_type == 'paper':
            self.attn = SeparableAttentionPaper(dim)
        elif attn_type == 'reduced':
            self.attn = SeparableAttentionReduced(dim, proj_ratio=attn_proj_ratio)
        elif attn_type == 'mobilevit':
            self.attn = SeparableAttentionMobileViT(dim)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")
        
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        hidden_dim = max(1, int(dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ==============================================================================
# Hybrid Model
# ==============================================================================

class HybridConvNeXtV2(nn.Module):
    """
    Hybrid model combining ConvNeXtV2 and Transformer blocks.
    
    Architecture:
    - Stem: 4x4 conv + LayerNorm
    - Stage 1: ConvNeXtV2 blocks + Downsample
    - Stage 2: ConvNeXtV2 blocks + Downsample
    - Stage 3: Transformer blocks + Downsample
    - Stage 4: Transformer blocks
    - Head: LayerNorm + Linear classifier
    """
    def __init__(
        self,
        backbone_variant: str = 'tiny',
        attn_type: str = 'paper',
        mlp_ratio: float = 4.0,
        attn_proj_ratio: float = 0.5,
        stage3_blocks: int = 9,
        stage4_blocks: int = 12,
        num_classes: int = 8,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        config = BACKBONE_CONFIGS[backbone_variant]
        depths = config['depths']
        dims = config['dims']
        
        self.backbone_variant = backbone_variant
        self.dims = dims
        self.attn_type = attn_type
        self.mlp_ratio = mlp_ratio
        self.stage3_blocks = stage3_blocks
        self.stage4_blocks = stage4_blocks
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # Downsample layers
        self.down1 = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
        )
        self.down2 = nn.Sequential(
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2),
        )
        self.down3 = nn.Sequential(
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2),
        )
        
        # Drop path rates
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depths[0] + depths[1])]
        
        # ConvNeXtV2 stages (1 and 2)
        self.stage1 = nn.Sequential(*[
            ConvNeXtV2Block(dims[0], drop_path=dp_rates[j]) 
            for j in range(depths[0])
        ])
        self.stage2 = nn.Sequential(*[
            ConvNeXtV2Block(dims[1], drop_path=dp_rates[depths[0] + j]) 
            for j in range(depths[1])
        ])
        
        # Transformer stages (3 and 4)
        self.stage3 = nn.Sequential(*[
            TransformerBlock(dims[2], attn_type=attn_type, mlp_ratio=mlp_ratio, attn_proj_ratio=attn_proj_ratio)
            for _ in range(stage3_blocks)
        ])
        self.stage4 = nn.Sequential(*[
            TransformerBlock(dims[3], attn_type=attn_type, mlp_ratio=mlp_ratio, attn_proj_ratio=attn_proj_ratio)
            for _ in range(stage4_blocks)
        ])
        
        # Head
        self.norm = nn.LayerNorm(dims[3])
        self.head = nn.Linear(dims[3], num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Stage 1
        x = self.stage1(x)
        
        # Stage 2
        x = self.down1(x)
        x = self.stage2(x)
        
        # Stage 3 (Conv -> Sequence)
        x = self.down2(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, N, C]
        x = self.stage3(x)
        
        # Stage 4
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, N, C] -> [B, C, H, W]
        x = self.down3(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.stage4(x)
        
        # Head
        x = x.mean(dim=1)  # Global average pooling
        x = self.norm(x)
        return self.head(x)


# ==============================================================================
# Weight Loading
# ==============================================================================

def download_weights(url: str, cache_dir: str = './pretrained_weights') -> str:
    """Download weights from URL and return local path."""
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.split('/')[-1]
    local_path = os.path.join(cache_dir, filename)
    
    if not os.path.exists(local_path):
        print(f"  Downloading {filename}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, local_path)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            return ""
    return local_path


def load_pretrained_weights(
    model: HybridConvNeXtV2, 
    backbone_variant: str, 
    weights_type: str,
    cache_dir: str = './pretrained_weights'
) -> Tuple[int, int]:
    """
    Load pretrained weights into model (stages 1-2 only).
    Returns (matched_keys, total_applicable_keys).
    """
    if backbone_variant not in PRETRAINED_URLS:
        return 0, 0
    if weights_type not in PRETRAINED_URLS.get(backbone_variant, {}):
        return 0, 0
    
    url = PRETRAINED_URLS[backbone_variant][weights_type]
    local_path = download_weights(url, cache_dir)
    if not local_path:
        return 0, 0
    
    # Load checkpoint
    ckpt = torch.load(local_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    
    # Key mapping from pretrained to hybrid model
    key_mapping = {
        'downsample_layers.0': 'stem',
        'downsample_layers.1': 'down1',
        'downsample_layers.2': 'down2',
        'downsample_layers.3': 'down3',
        'stages.0': 'stage1',
        'stages.1': 'stage2',
    }
    
    model_state = model.state_dict()
    matched = 0
    total = 0
    
    for old_key, value in state_dict.items():
        # Find mapped key
        new_key = old_key
        for old_prefix, new_prefix in key_mapping.items():
            if old_key.startswith(old_prefix):
                new_key = old_key.replace(old_prefix, new_prefix, 1)
                break
        
        # Check if key exists in model and shapes match
        if new_key in model_state:
            total += 1
            if model_state[new_key].shape == value.shape:
                model_state[new_key] = value
                matched += 1
    
    model.load_state_dict(model_state, strict=False)
    return matched, total


# ==============================================================================
# Parameter Counting
# ==============================================================================

def count_params(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_params_by_component(model: HybridConvNeXtV2) -> Dict[str, float]:
    """Count parameters by model component (in millions)."""
    result = {}
    components = ['stem', 'down1', 'down2', 'down3', 'stage1', 'stage2', 'stage3', 'stage4', 'norm', 'head']
    
    for name in components:
        module = getattr(model, name, None)
        if module:
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            result[name] = params / 1e6
    
    return result


# ==============================================================================
# Experiment Configuration
# ==============================================================================

@dataclass
class ExperimentConfig:
    backbone: str
    attn_type: str
    mlp_ratio: float
    weights_type: str = 'none'
    stage3_blocks: int = 9
    stage4_blocks: int = 12
    attn_proj_ratio: float = 0.5
    num_classes: int = 8


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    total_params_M: float
    diff_from_target_M: float
    params_by_component: Dict[str, float]
    weights_loaded: int
    weights_total: int
    success: bool
    error_msg: str = ""


def run_single_experiment(config: ExperimentConfig, cache_dir: str = './pretrained_weights') -> ExperimentResult:
    """Run a single experiment configuration."""
    try:
        # Create model
        model = HybridConvNeXtV2(
            backbone_variant=config.backbone,
            attn_type=config.attn_type,
            mlp_ratio=config.mlp_ratio,
            attn_proj_ratio=config.attn_proj_ratio,
            stage3_blocks=config.stage3_blocks,
            stage4_blocks=config.stage4_blocks,
            num_classes=config.num_classes,
        )
        
        # Load pretrained weights
        matched, total = 0, 0
        if config.weights_type != 'none':
            matched, total = load_pretrained_weights(model, config.backbone, config.weights_type, cache_dir)
        
        # Count parameters
        total_params = count_params(model) / 1e6
        params_by_component = count_params_by_component(model)
        
        result = ExperimentResult(
            config=config,
            total_params_M=total_params,
            diff_from_target_M=abs(total_params - TARGET_PARAMS_M),
            params_by_component=params_by_component,
            weights_loaded=matched,
            weights_total=total,
            success=True,
        )
        
        del model
        return result
        
    except Exception as e:
        return ExperimentResult(
            config=config,
            total_params_M=0,
            diff_from_target_M=float('inf'),
            params_by_component={},
            weights_loaded=0,
            weights_total=0,
            success=False,
            error_msg=str(e),
        )


# ==============================================================================
# Main Experiment
# ==============================================================================

def main():
    print("=" * 100)
    print("COMPREHENSIVE PARAMETER SEARCH FOR HYBRID CONVNEXTV2 + SEPARABLE ATTENTION MODEL")
    print("=" * 100)
    print(f"\nPaper target: {TARGET_PARAMS_M}M parameters")
    print(f"Paper results: 93.48% accuracy, 93.24% precision, 90.70% recall, 91.82% F1")
    print("\nPaper ablation study (Table 3):")
    print("  - Baseline model: 24.30M")
    print("  - ConvNeXtV2 only (stages 1-2): 26.14M")
    print("  - Separable attention only (stages 3-4): 20.12M")
    print("  - Proposed hybrid: 21.92M")
    print()
    
    # Define search space
    backbones = ['atto', 'femto', 'pico', 'nano', 'tiny']
    attn_types = ['minimal', 'paper', 'reduced', 'mobilevit']
    mlp_ratios = [0.5, 1.0, 1.125, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    # -------------------------------------------------------------------------
    # PHASE 1: Quick parameter scan without loading weights
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PHASE 1: Quick parameter scan (no weight loading)")
    print("=" * 100)
    
    quick_results = []
    total_configs = len(backbones) * len(attn_types) * len(mlp_ratios)
    count = 0
    
    for backbone in backbones:
        for attn_type in attn_types:
            for mlp_ratio in mlp_ratios:
                count += 1
                config = ExperimentConfig(
                    backbone=backbone,
                    attn_type=attn_type,
                    mlp_ratio=mlp_ratio,
                    weights_type='none',
                )
                result = run_single_experiment(config)
                if result.success:
                    quick_results.append(result)
                
                if count % 30 == 0:
                    print(f"  Progress: {count}/{total_configs}")
    
    # Sort by distance to target
    quick_results.sort(key=lambda r: r.diff_from_target_M)
    
    print(f"\n  Completed {len(quick_results)} configurations")
    
    # -------------------------------------------------------------------------
    # Show top 30 configurations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TOP 30 CONFIGURATIONS (sorted by distance to 21.92M)")
    print("=" * 100)
    print(f"\n{'Rank':<6}{'Backbone':<10}{'Attention':<12}{'MLP Ratio':<12}{'Params (M)':<14}{'Diff (M)':<12}")
    print("-" * 66)
    
    for i, r in enumerate(quick_results[:30]):
        print(f"{i+1:<6}{r.config.backbone:<10}{r.config.attn_type:<12}{r.config.mlp_ratio:<12.3f}{r.total_params_M:<14.3f}{r.diff_from_target_M:<12.3f}")
    
    # -------------------------------------------------------------------------
    # PHASE 2: Test top configurations with pretrained weights
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("PHASE 2: Testing top 15 configurations with all available pretrained weights")
    print("=" * 100)
    
    full_results = []
    top_configs = quick_results[:15]
    
    for i, quick_result in enumerate(top_configs):
        backbone = quick_result.config.backbone
        attn_type = quick_result.config.attn_type
        mlp_ratio = quick_result.config.mlp_ratio
        
        # Get available weight types for this backbone
        available_weights = ['none'] + list(PRETRAINED_URLS.get(backbone, {}).keys())
        
        print(f"\n[{i+1}/{len(top_configs)}] {backbone} / {attn_type} / mlp={mlp_ratio}")
        
        for weights_type in available_weights:
            config = ExperimentConfig(
                backbone=backbone,
                attn_type=attn_type,
                mlp_ratio=mlp_ratio,
                weights_type=weights_type,
            )
            
            result = run_single_experiment(config)
            if result.success:
                full_results.append(result)
                loaded_str = f"{result.weights_loaded}/{result.weights_total}" if result.weights_total > 0 else "N/A"
                print(f"    {weights_type:<12}: {result.total_params_M:.3f}M (weights: {loaded_str})")
    
    # Sort by distance to target
    full_results.sort(key=lambda r: r.diff_from_target_M)
    
    # -------------------------------------------------------------------------
    # Final results
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("FINAL RESULTS - ALL CONFIGURATIONS WITH WEIGHTS")
    print("=" * 100)
    
    print(f"\n{'Rank':<6}{'Backbone':<10}{'Attention':<12}{'MLP':<8}{'Weights':<12}{'Params(M)':<12}{'Diff(M)':<10}{'Loaded':<12}")
    print("-" * 92)
    
    for i, r in enumerate(full_results[:40]):
        loaded_str = f"{r.weights_loaded}/{r.weights_total}" if r.weights_total > 0 else "N/A"
        print(f"{i+1:<6}{r.config.backbone:<10}{r.config.attn_type:<12}{r.config.mlp_ratio:<8.2f}{r.config.weights_type:<12}{r.total_params_M:<12.3f}{r.diff_from_target_M:<10.3f}{loaded_str:<12}")
    
    # -------------------------------------------------------------------------
    # Detailed breakdown of top 5
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("DETAILED BREAKDOWN OF TOP 5 CONFIGURATIONS")
    print("=" * 100)
    
    for i, r in enumerate(full_results[:5]):
        print(f"\n[{i+1}] {r.config.backbone} / {r.config.attn_type} / mlp_ratio={r.config.mlp_ratio} / {r.config.weights_type}")
        print(f"    Total Parameters: {r.total_params_M:.3f}M")
        print(f"    Difference from target (21.92M): {r.diff_from_target_M:.3f}M")
        print(f"    Pretrained weights loaded: {r.weights_loaded}/{r.weights_total}")
        print("    Parameters by component:")
        for comp, params in sorted(r.params_by_component.items()):
            print(f"      {comp:<15}: {params:.3f}M")
        
        # Stage analysis
        convnext_params = r.params_by_component.get('stage1', 0) + r.params_by_component.get('stage2', 0)
        transformer_params = r.params_by_component.get('stage3', 0) + r.params_by_component.get('stage4', 0)
        other_params = r.total_params_M - convnext_params - transformer_params
        print(f"    Stage breakdown:")
        print(f"      ConvNeXtV2 (stages 1-2): {convnext_params:.3f}M")
        print(f"      Transformer (stages 3-4): {transformer_params:.3f}M")
        print(f"      Other (stem, downsamples, head): {other_params:.3f}M")
    
    # -------------------------------------------------------------------------
    # Save results to JSON
    # -------------------------------------------------------------------------
    results_data = []
    for r in full_results:
        results_data.append({
            'backbone': r.config.backbone,
            'attn_type': r.config.attn_type,
            'mlp_ratio': r.config.mlp_ratio,
            'weights_type': r.config.weights_type,
            'total_params_M': round(r.total_params_M, 3),
            'diff_from_target_M': round(r.diff_from_target_M, 3),
            'params_by_component': {k: round(v, 3) for k, v in r.params_by_component.items()},
            'weights_loaded': r.weights_loaded,
            'weights_total': r.weights_total,
        })
    
    output_file = 'comprehensive_param_search_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n\nResults saved to {output_file}")
    
    # -------------------------------------------------------------------------
    # Recommendations
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    best = full_results[0] if full_results else None
    if best:
        print(f"""
The configuration closest to the paper's 21.92M parameters is:

  Backbone:          {best.config.backbone}
  Attention Type:    {best.config.attn_type}
  MLP Ratio:         {best.config.mlp_ratio}
  Pretrained Weights: {best.config.weights_type}
  Total Parameters:  {best.total_params_M:.3f}M
  Difference:        {best.diff_from_target_M:.3f}M

To use this configuration, update the model in hybrid_model.py or use the
configuration directly with HybridConvNeXtV2 class.
""")
    
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)
    print("""
Key observations:

1. ATTENTION TYPE IMPACT:
   - 'minimal' attention: O(d) params - very few attention parameters
   - 'paper' attention: O(d²) params - matches equations (5)-(10) exactly
   - 'reduced' attention: O(d*d_proj) params - intermediate option
   - 'mobilevit' attention: O(d²) params - MobileViT-v2 style

2. MLP RATIO IMPACT:
   - Standard ViT uses 4x expansion
   - Lower ratios (1-2x) significantly reduce parameter count
   - Paper likely uses non-standard MLP ratio to hit 21.92M target

3. BACKBONE CHOICE:
   - 'tiny' backbone has dims [96, 192, 384, 768] and depths [3, 3, 9, 3]
   - Paper uses custom depths [3, 3, 9, 12] in stages 3-4
   - Smaller backbones (nano, pico) might achieve similar results

4. PRETRAINED WEIGHTS:
   - Only stages 1-2 use pretrained weights
   - Stages 3-4 (Transformer) are initialized from scratch
   - Available: FCMAE (self-supervised), IM-1K (fine-tuned), IM-22K (fine-tuned)

5. CONFIGURATION MATCHING PAPER (~21.92M):
   - If using 'tiny' backbone with 'minimal' attention and ~1.125x MLP,
     we can achieve approximately the target parameter count
   - If using 'paper' (full) attention, need much lower MLP ratio or smaller backbone
""")


if __name__ == '__main__':
    main()
