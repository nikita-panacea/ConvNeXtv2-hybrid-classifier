# experiments/full_experiment.py
"""
Complete experiment to find configurations matching the paper's ~21.92M parameters.

This script:
1. Explores all backbone/attention/MLP combinations
2. Downloads and loads pretrained weights
3. Verifies parameter counts with loaded weights
4. Generates a comprehensive report

Usage:
    python experiments/full_experiment.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from dataclasses import dataclass, asdict
import urllib.request
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Self-contained implementations (no timm dependency)
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
        output = x.div(keep_prob) * random_tensor
        return output


# ==============================================================================
# Data Classes for Configuration
# ==============================================================================

@dataclass
class ExperimentConfig:
    backbone: str
    attn_type: str
    mlp_ratio: float
    weights_type: str
    stage3_blocks: int = 9
    stage4_blocks: int = 12
    num_classes: int = 8
    attn_proj_ratio: float = 0.5


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    total_params_M: float
    diff_from_target_M: float
    params_by_stage: Dict[str, float]
    weights_loaded: int
    weights_total: int
    success: bool
    error_msg: str = ""


# ==============================================================================
# Layer Utilities
# ==============================================================================

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
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
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


# ==============================================================================
# Separable Self-Attention Variants
# ==============================================================================

class SeparableAttentionMinimal(nn.Module):
    """Minimal: O(d) parameters."""
    def __init__(self, dim):
        super().__init__()
        self.WI = nn.Linear(dim, 1, bias=False)
        self.key_scale = nn.Parameter(torch.ones(dim))
        self.value_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scores = self.WI(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        out = x * context.unsqueeze(1)
        out = out * self.value_scale
        return out


class SeparableAttentionFull(nn.Module):
    """Full as per paper equations: O(d²) parameters."""
    def __init__(self, dim):
        super().__init__()
        self.WI = nn.Linear(dim, 1, bias=False)
        self.WK = nn.Linear(dim, dim, bias=False)
        self.WV = nn.Linear(dim, dim, bias=False)
        self.WO = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        cs = torch.softmax(self.WI(x), dim=1)
        xK = self.WK(x)
        cv = torch.sum(cs * xK, dim=1)
        xV = torch.relu(self.WV(x))
        z = cv.unsqueeze(1) * xV
        y = self.WO(z)
        return y


class SeparableAttentionReduced(nn.Module):
    """Reduced projection: O(d*d_proj) parameters."""
    def __init__(self, dim, proj_ratio=0.5):
        super().__init__()
        d_proj = max(1, int(dim * proj_ratio))
        self.WI = nn.Linear(dim, 1, bias=False)
        self.WK = nn.Linear(dim, d_proj, bias=False)
        self.WV = nn.Linear(dim, d_proj, bias=False)
        self.WO = nn.Linear(d_proj, dim, bias=False)

    def forward(self, x):
        cs = torch.softmax(self.WI(x), dim=1)
        xK = self.WK(x)
        cv = torch.sum(cs * xK, dim=1)
        xV = torch.relu(self.WV(x))
        z = cv.unsqueeze(1) * xV
        y = self.WO(z)
        return y


class SeparableAttentionMobileViT(nn.Module):
    """
    MobileViT-v2 style separable attention.
    Uses learned input projection but channel-wise key/value.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.input_proj = nn.Linear(dim, dim, bias=False)
        self.WI = nn.Linear(dim, 1, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x_proj = self.input_proj(x)
        cs = torch.softmax(self.WI(x_proj), dim=1)
        cv = torch.sum(cs * x_proj, dim=1)
        out = x * cv.unsqueeze(1)
        out = self.output_proj(out)
        return out


# ==============================================================================
# Transformer Block
# ==============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim, attn_type='minimal', mlp_ratio=1.125, attn_proj_ratio=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        attn_classes = {
            'minimal': SeparableAttentionMinimal,
            'full': SeparableAttentionFull,
            'reduced': lambda d: SeparableAttentionReduced(d, proj_ratio=attn_proj_ratio),
            'mobilevit': SeparableAttentionMobileViT,
        }
        self.attn = attn_classes[attn_type](dim)
        
        self.norm2 = nn.LayerNorm(dim)
        hidden = max(1, int(dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ==============================================================================
# Hybrid Model
# ==============================================================================

BACKBONE_CONFIGS = {
    'atto': {'depths': [2, 2, 6, 2], 'dims': [40, 80, 160, 320]},
    'femto': {'depths': [2, 2, 6, 2], 'dims': [48, 96, 192, 384]},
    'pico': {'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512]},
    'nano': {'depths': [2, 2, 8, 2], 'dims': [80, 160, 320, 640]},
    'tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
    'base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
}

PRETRAINED_URLS = {
    'atto': {
        'fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_atto_1k_224_fcmae.pt',
        'im1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt',
    },
    'femto': {
        'fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_femto_1k_224_fcmae.pt',
        'im1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt',
    },
    'pico': {
        'fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_pico_1k_224_fcmae.pt',
        'im1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt',
    },
    'nano': {
        'fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_nano_1k_224_fcmae.pt',
        'im1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt',
        'im22k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_224_ema.pt',
    },
    'tiny': {
        'fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_tiny_1k_224_fcmae.pt',
        'im1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt',
        'im22k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt',
    },
    'base': {
        'fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_base_1k_224_fcmae.pt',
        'im1k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt',
        'im22k': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt',
    },
}


class HybridConvNeXtV2Experiment(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        
        bc = BACKBONE_CONFIGS[config.backbone]
        depths = bc['depths']
        dims = bc['dims']
        
        self.config = config
        self.dims = dims
        
        # Downsample layers
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
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
        
        # ConvNeXtV2 stages
        self.stage1 = nn.Sequential(*[ConvNeXtV2Block(dims[0]) for _ in range(depths[0])])
        self.stage2 = nn.Sequential(*[ConvNeXtV2Block(dims[1]) for _ in range(depths[1])])
        
        # Transformer stages
        self.stage3 = nn.Sequential(*[
            TransformerBlock(dims[2], attn_type=config.attn_type, 
                           mlp_ratio=config.mlp_ratio, attn_proj_ratio=config.attn_proj_ratio)
            for _ in range(config.stage3_blocks)
        ])
        self.stage4 = nn.Sequential(*[
            TransformerBlock(dims[3], attn_type=config.attn_type,
                           mlp_ratio=config.mlp_ratio, attn_proj_ratio=config.attn_proj_ratio)
            for _ in range(config.stage4_blocks)
        ])
        
        self.norm = nn.LayerNorm(dims[3])
        self.head = nn.Linear(dims[3], config.num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
        return self.head(x)


# ==============================================================================
# Utilities
# ==============================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_params_by_stage(model: HybridConvNeXtV2Experiment) -> Dict[str, float]:
    result = {}
    for name in ['stem', 'down1', 'down2', 'down3', 'stage1', 'stage2', 'stage3', 'stage4', 'norm', 'head']:
        module = getattr(model, name, None)
        if module:
            result[name] = sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6
    return result


def download_weights(url: str, cache_dir: str = './pretrained_weights') -> str:
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.split('/')[-1]
    local_path = os.path.join(cache_dir, filename)
    
    if not os.path.exists(local_path):
        print(f"    Downloading {filename}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, local_path)
            print("Done.")
        except Exception as e:
            print(f"Failed: {e}")
            return ""
    return local_path


def load_weights(model: HybridConvNeXtV2Experiment, variant: str, weights_type: str, 
                 cache_dir: str = './pretrained_weights') -> Tuple[int, int]:
    if variant not in PRETRAINED_URLS:
        return 0, 0
    if weights_type not in PRETRAINED_URLS.get(variant, {}):
        return 0, 0
    
    url = PRETRAINED_URLS[variant][weights_type]
    local_path = download_weights(url, cache_dir)
    if not local_path:
        return 0, 0
    
    ckpt = torch.load(local_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    
    # Key mapping
    mapping = {
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
        new_key = old_key
        for old_prefix, new_prefix in mapping.items():
            if old_key.startswith(old_prefix):
                new_key = old_key.replace(old_prefix, new_prefix, 1)
                break
        
        if new_key in model_state:
            total += 1
            if model_state[new_key].shape == value.shape:
                model_state[new_key] = value
                matched += 1
    
    model.load_state_dict(model_state, strict=False)
    return matched, total


# ==============================================================================
# Main Experiment
# ==============================================================================

def run_experiment(config: ExperimentConfig, cache_dir: str = './pretrained_weights') -> ExperimentResult:
    try:
        model = HybridConvNeXtV2Experiment(config)
        
        # Load weights
        matched, total = 0, 0
        if config.weights_type != 'none':
            matched, total = load_weights(model, config.backbone, config.weights_type, cache_dir)
        
        # Count params
        total_params = count_params(model) / 1e6
        params_by_stage = count_params_by_stage(model)
        
        result = ExperimentResult(
            config=config,
            total_params_M=total_params,
            diff_from_target_M=abs(total_params - 21.92),
            params_by_stage=params_by_stage,
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
            params_by_stage={},
            weights_loaded=0,
            weights_total=0,
            success=False,
            error_msg=str(e),
        )


def main():
    print("=" * 100)
    print("COMPREHENSIVE HYBRID MODEL PARAMETER EXPLORATION")
    print("=" * 100)
    print(f"\nTarget from paper: 21.92M parameters")
    print(f"Paper results: 93.48% accuracy, 93.24% precision, 90.70% recall")
    print()
    
    # Define search space
    backbones = ['atto', 'femto', 'pico', 'nano', 'tiny']
    attn_types = ['minimal', 'full', 'reduced', 'mobilevit']
    mlp_ratios = [1.0, 1.125, 1.5, 2.0, 3.0, 4.0]
    weights_types = ['none', 'fcmae', 'im1k', 'im22k']
    
    # First pass: without loading weights (faster)
    print("\n[PHASE 1] Quick parameter scan (no weight loading)...")
    print("-" * 100)
    
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
                result = run_experiment(config)
                quick_results.append(result)
                
                if count % 20 == 0:
                    print(f"  Progress: {count}/{total_configs}")
    
    # Sort by distance to target
    quick_results.sort(key=lambda r: r.diff_from_target_M)
    
    print("\n[TOP 20 CONFIGURATIONS BY PARAMETER COUNT]")
    print("-" * 100)
    print(f"{'Rank':<6}{'Backbone':<10}{'Attn Type':<12}{'MLP Ratio':<12}{'Params (M)':<14}{'Diff (M)':<12}")
    print("-" * 100)
    
    for i, r in enumerate(quick_results[:20]):
        if r.success:
            print(f"{i+1:<6}{r.config.backbone:<10}{r.config.attn_type:<12}{r.config.mlp_ratio:<12.3f}{r.total_params_M:<14.3f}{r.diff_from_target_M:<12.3f}")
    
    # Phase 2: Test top configs with pretrained weights
    print("\n\n[PHASE 2] Testing top configurations with pretrained weights...")
    print("-" * 100)
    
    full_results = []
    top_configs = quick_results[:15]  # Top 15 configs
    
    for i, quick_result in enumerate(top_configs):
        backbone = quick_result.config.backbone
        attn_type = quick_result.config.attn_type
        mlp_ratio = quick_result.config.mlp_ratio
        
        available_weights = ['none'] + list(PRETRAINED_URLS.get(backbone, {}).keys())
        
        for weights_type in available_weights:
            config = ExperimentConfig(
                backbone=backbone,
                attn_type=attn_type,
                mlp_ratio=mlp_ratio,
                weights_type=weights_type,
            )
            
            print(f"  [{i+1}/{len(top_configs)}] {backbone} / {attn_type} / mlp={mlp_ratio} / {weights_type}")
            result = run_experiment(config)
            full_results.append(result)
    
    # Sort by success and distance to target
    full_results.sort(key=lambda r: (not r.success, r.diff_from_target_M))
    
    print("\n\n" + "=" * 100)
    print("FINAL RESULTS - CONFIGURATIONS CLOSEST TO 21.92M PARAMETERS")
    print("=" * 100)
    
    print(f"\n{'Rank':<6}{'Backbone':<10}{'Attn':<12}{'MLP':<8}{'Weights':<10}{'Params(M)':<12}{'Diff(M)':<10}{'Loaded':<12}")
    print("-" * 100)
    
    for i, r in enumerate(full_results[:30]):
        if r.success:
            loaded_str = f"{r.weights_loaded}/{r.weights_total}" if r.weights_total > 0 else "N/A"
            print(f"{i+1:<6}{r.config.backbone:<10}{r.config.attn_type:<12}{r.config.mlp_ratio:<8.2f}{r.config.weights_type:<10}{r.total_params_M:<12.3f}{r.diff_from_target_M:<10.3f}{loaded_str:<12}")
    
    # Detailed breakdown of top 5
    print("\n\n" + "=" * 100)
    print("DETAILED BREAKDOWN OF TOP 5 CONFIGURATIONS")
    print("=" * 100)
    
    for i, r in enumerate(full_results[:5]):
        if r.success:
            print(f"\n[{i+1}] {r.config.backbone} / {r.config.attn_type} / mlp_ratio={r.config.mlp_ratio} / {r.config.weights_type}")
            print(f"    Total: {r.total_params_M:.3f}M (diff from 21.92M: {r.diff_from_target_M:.3f}M)")
            print(f"    Weights loaded: {r.weights_loaded}/{r.weights_total}")
            print("    Parameters by stage:")
            for stage, params in sorted(r.params_by_stage.items()):
                print(f"      {stage:<15}: {params:.3f}M")
    
    # Save results to JSON
    results_data = []
    for r in full_results:
        if r.success:
            results_data.append({
                'backbone': r.config.backbone,
                'attn_type': r.config.attn_type,
                'mlp_ratio': r.config.mlp_ratio,
                'weights_type': r.config.weights_type,
                'total_params_M': r.total_params_M,
                'diff_from_target_M': r.diff_from_target_M,
                'params_by_stage': r.params_by_stage,
                'weights_loaded': r.weights_loaded,
                'weights_total': r.weights_total,
            })
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n\nResults saved to experiment_results.json")
    
    # Recommendations
    print("\n\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    best = full_results[0]
    if best.success:
        print(f"""
Based on the analysis, the configuration closest to the paper's 21.92M parameters is:

  Backbone:     {best.config.backbone}
  Attention:    {best.config.attn_type}
  MLP Ratio:    {best.config.mlp_ratio}
  Weights:      {best.config.weights_type}
  Parameters:   {best.total_params_M:.3f}M

To train with this configuration, update the model configuration in hybrid_model.py
or create a new training configuration file.
""")
    
    print("\n" + "=" * 100)
    print("ANALYSIS NOTES")
    print("=" * 100)
    print("""
Key observations from the paper:

1. The paper uses stage depths of [3, 3, 9, 12] - this matches the 'tiny' backbone 
   for stages 1-2 (depths [3, 3]) but uses custom depths for stages 3-4 (9, 12).

2. The paper's ablation table (Table 3):
   - Separable attention only (stages 3-4): 20.12M params
   - ConvNeXtV2 only (stages 1-2): 26.14M params  
   - Proposed hybrid: 21.92M params

3. This suggests that with proper MLP ratio tuning, the 'tiny' backbone 
   dimensions [96, 192, 384, 768] should achieve the target param count.

4. The 'minimal' attention type has very few parameters (O(d)),
   while 'full' attention has O(d²) parameters.

5. For ~21.92M params with tiny backbone, we need a specific MLP ratio
   that balances the transformer stage parameters with the ConvNeXtV2 stages.
""")


if __name__ == '__main__':
    main()
