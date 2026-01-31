# experiments/parameter_exploration.py
"""
Comprehensive parameter count exploration for hybrid ConvNeXtV2 + Separable Attention model.

This script explores different configurations to find the combination that matches
the paper's target of ~21.92M parameters.

Key variables:
1. ConvNeXtV2 backbone variant (atto, femto, pico, nano, tiny)
2. MLP expansion ratio in transformer blocks
3. Separable attention implementation (minimal vs full)
4. Pretrained weights (FCMAE, IM-1K, IM-22K)
"""

import os
import sys
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import urllib.request
from tqdm import tqdm

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
# Layer Utilities (from models/utils.py)
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
    """
    Minimal separable attention - channel-wise operations only.
    Very few parameters: O(d) complexity.
    """
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
    """
    Full separable attention as per paper equations (5)-(10).
    
    cs = softmax(x @ WI)          # WI ∈ R^d
    cv = Σ cs(i) * (x @ WK)(i)    # WK ∈ R^(d×d)
    xV = ReLU(x @ WV)             # WV ∈ R^(d×d)
    z = cv ⊙ xV
    y = z @ WO                     # WO ∈ R^(d×d)
    
    Parameters: d + 3*d^2 = O(d^2)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.WI = nn.Linear(dim, 1, bias=False)  # d params
        self.WK = nn.Linear(dim, dim, bias=False)  # d^2 params
        self.WV = nn.Linear(dim, dim, bias=False)  # d^2 params
        self.WO = nn.Linear(dim, dim, bias=False)  # d^2 params

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        
        # Context scores
        cs = torch.softmax(self.WI(x), dim=1)  # [B, N, 1]
        
        # Key projection
        xK = self.WK(x)  # [B, N, C]
        
        # Context vector (weighted sum of keys)
        cv = torch.sum(cs * xK, dim=1)  # [B, C]
        
        # Value projection with ReLU
        xV = torch.relu(self.WV(x))  # [B, N, C]
        
        # Broadcast and multiply
        z = cv.unsqueeze(1) * xV  # [B, N, C]
        
        # Output projection
        y = self.WO(z)  # [B, N, C]
        
        return y


class SeparableAttentionReduced(nn.Module):
    """
    Separable attention with reduced projection dimension.
    Uses d_proj < d for intermediate projections.
    """
    def __init__(self, dim, proj_ratio=0.5):
        super().__init__()
        self.dim = dim
        d_proj = int(dim * proj_ratio)
        self.d_proj = d_proj
        
        self.WI = nn.Linear(dim, 1, bias=False)
        self.WK = nn.Linear(dim, d_proj, bias=False)
        self.WV = nn.Linear(dim, d_proj, bias=False)
        self.WO = nn.Linear(d_proj, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        cs = torch.softmax(self.WI(x), dim=1)
        xK = self.WK(x)
        cv = torch.sum(cs * xK, dim=1)
        xV = torch.relu(self.WV(x))
        z = cv.unsqueeze(1) * xV
        y = self.WO(z)
        return y


# ==============================================================================
# Transformer Block
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    Transformer block with configurable attention and MLP.
    """
    def __init__(self, dim, attn_type='minimal', mlp_ratio=1.125, attn_proj_ratio=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        if attn_type == 'minimal':
            self.attn = SeparableAttentionMinimal(dim)
        elif attn_type == 'full':
            self.attn = SeparableAttentionFull(dim)
        elif attn_type == 'reduced':
            self.attn = SeparableAttentionReduced(dim, proj_ratio=attn_proj_ratio)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")
        
        self.norm2 = nn.LayerNorm(dim)
        
        hidden = int(dim * mlp_ratio)
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
# ConvNeXtV2 Backbone (Configurable)
# ==============================================================================

class ConvNeXtV2Backbone(nn.Module):
    """
    ConvNeXtV2 backbone with configurable depths and dims.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.):
        super().__init__()
        self.depths = depths
        self.dims = dims
        
        # Downsample layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        # Stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ==============================================================================
# Hybrid Model (Configurable)
# ==============================================================================

class HybridConvNeXtV2Configurable(nn.Module):
    """
    Hybrid model with configurable architecture.
    
    Args:
        backbone_variant: 'atto', 'femto', 'pico', 'nano', 'tiny'
        attn_type: 'minimal', 'full', 'reduced'
        mlp_ratio: expansion ratio for transformer MLP
        attn_proj_ratio: projection ratio for reduced attention
        num_classes: number of output classes
        stage3_blocks: number of transformer blocks in stage 3
        stage4_blocks: number of transformer blocks in stage 4
    """
    
    BACKBONE_CONFIGS = {
        'atto': {'depths': [2, 2, 6, 2], 'dims': [40, 80, 160, 320]},
        'femto': {'depths': [2, 2, 6, 2], 'dims': [48, 96, 192, 384]},
        'pico': {'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512]},
        'nano': {'depths': [2, 2, 8, 2], 'dims': [80, 160, 320, 640]},
        'tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
        'base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
    }
    
    def __init__(
        self,
        backbone_variant='tiny',
        attn_type='minimal',
        mlp_ratio=1.125,
        attn_proj_ratio=0.5,
        num_classes=8,
        stage3_blocks=9,
        stage4_blocks=12,
    ):
        super().__init__()
        
        config = self.BACKBONE_CONFIGS[backbone_variant]
        depths = config['depths']
        dims = config['dims']
        
        self.backbone_variant = backbone_variant
        self.dims = dims
        self.attn_type = attn_type
        self.mlp_ratio = mlp_ratio
        
        # Build backbone for stages 1-2 only
        backbone = ConvNeXtV2Backbone(
            depths=[depths[0], depths[1], 0, 0],
            dims=dims,
            drop_path_rate=0.0
        )
        
        # Extract components
        self.stem = backbone.downsample_layers[0]
        self.down1 = backbone.downsample_layers[1]
        self.down2 = backbone.downsample_layers[2]
        self.down3 = backbone.downsample_layers[3]
        
        self.stage1 = backbone.stages[0]
        self.stage2 = backbone.stages[1]
        
        # Transformer stages
        self.stage3 = nn.Sequential(*[
            TransformerBlock(dims[2], attn_type=attn_type, mlp_ratio=mlp_ratio, attn_proj_ratio=attn_proj_ratio)
            for _ in range(stage3_blocks)
        ])
        self.stage4 = nn.Sequential(*[
            TransformerBlock(dims[3], attn_type=attn_type, mlp_ratio=mlp_ratio, attn_proj_ratio=attn_proj_ratio)
            for _ in range(stage4_blocks)
        ])
        
        self.norm = nn.LayerNorm(dims[3])
        self.head = nn.Linear(dims[3], num_classes)
        
        del backbone

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
# Pretrained Weight URLs
# ==============================================================================

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


# ==============================================================================
# Utility Functions
# ==============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_module(model: nn.Module) -> Dict[str, int]:
    """Count parameters grouped by top-level module."""
    by_module = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        top_module = name.split('.')[0]
        by_module[top_module] = by_module.get(top_module, 0) + p.numel()
    return by_module


def download_pretrained_weights(url: str, cache_dir: str = './pretrained_weights') -> str:
    """Download pretrained weights and return local path."""
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.split('/')[-1]
    local_path = os.path.join(cache_dir, filename)
    
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Saved to {local_path}")
    
    return local_path


def load_pretrained_weights(model: HybridConvNeXtV2Configurable, variant: str, weights_type: str, cache_dir: str = './pretrained_weights') -> Tuple[int, int]:
    """
    Load pretrained weights into the model (stages 1-2 only).
    Returns (matched_keys, total_keys).
    """
    if variant not in PRETRAINED_URLS or weights_type not in PRETRAINED_URLS[variant]:
        print(f"No pretrained weights for {variant}/{weights_type}")
        return 0, 0
    
    url = PRETRAINED_URLS[variant][weights_type]
    local_path = download_pretrained_weights(url, cache_dir)
    
    ckpt = torch.load(local_path, map_location='cpu')
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    
    # Map pretrained keys to hybrid model keys
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
        new_key = old_key
        for old_prefix, new_prefix in key_mapping.items():
            if old_key.startswith(old_prefix):
                new_key = old_key.replace(old_prefix, new_prefix, 1)
                break
        
        if new_key in model_state:
            if model_state[new_key].shape == value.shape:
                model_state[new_key] = value
                matched += 1
            total += 1
    
    model.load_state_dict(model_state, strict=False)
    return matched, total


# ==============================================================================
# Experiment Runner
# ==============================================================================

def run_parameter_exploration():
    """
    Run comprehensive parameter exploration.
    """
    print("=" * 80)
    print("HYBRID CONVNEXTV2 + SEPARABLE ATTENTION PARAMETER EXPLORATION")
    print("=" * 80)
    print(f"\nTarget: ~21.92M parameters (from paper)")
    print(f"Paper's ablation results:")
    print(f"  - Baseline model: 24.30M")
    print(f"  - ConvNeXtV2 only (stages 1-2): 26.14M")
    print(f"  - Separable attention only (stages 3-4): 20.12M")
    print(f"  - Proposed hybrid: 21.92M")
    print()
    
    # Configuration space
    backbone_variants = ['atto', 'femto', 'pico', 'nano', 'tiny']
    attn_types = ['minimal', 'full', 'reduced']
    mlp_ratios = [1.0, 1.125, 2.0, 4.0]
    
    results = []
    
    print("\n" + "=" * 80)
    print("EXPLORING CONFIGURATIONS...")
    print("=" * 80)
    
    for backbone in backbone_variants:
        for attn_type in attn_types:
            for mlp_ratio in mlp_ratios:
                try:
                    model = HybridConvNeXtV2Configurable(
                        backbone_variant=backbone,
                        attn_type=attn_type,
                        mlp_ratio=mlp_ratio,
                        attn_proj_ratio=0.5,
                        num_classes=8,
                        stage3_blocks=9,
                        stage4_blocks=12,
                    )
                    
                    total_params = count_parameters(model)
                    by_module = count_parameters_by_module(model)
                    
                    config = {
                        'backbone': backbone,
                        'attn_type': attn_type,
                        'mlp_ratio': mlp_ratio,
                        'total_params_M': total_params / 1e6,
                        'diff_from_target_M': abs(total_params / 1e6 - 21.92),
                        'by_module': by_module,
                    }
                    results.append(config)
                    
                    del model
                except Exception as e:
                    print(f"Error with {backbone}/{attn_type}/{mlp_ratio}: {e}")
    
    # Sort by difference from target
    results.sort(key=lambda x: x['diff_from_target_M'])
    
    print("\n" + "=" * 80)
    print("TOP 20 CONFIGURATIONS (CLOSEST TO 21.92M)")
    print("=" * 80)
    print(f"{'Rank':<6}{'Backbone':<10}{'Attn Type':<12}{'MLP Ratio':<12}{'Params (M)':<14}{'Diff (M)':<10}")
    print("-" * 64)
    
    for i, config in enumerate(results[:20]):
        print(f"{i+1:<6}{config['backbone']:<10}{config['attn_type']:<12}{config['mlp_ratio']:<12}{config['total_params_M']:<14.3f}{config['diff_from_target_M']:<10.3f}")
    
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN OF TOP 5 CONFIGURATIONS")
    print("=" * 80)
    
    for i, config in enumerate(results[:5]):
        print(f"\n[{i+1}] {config['backbone']} / {config['attn_type']} / mlp_ratio={config['mlp_ratio']}")
        print(f"    Total: {config['total_params_M']:.3f}M (diff: {config['diff_from_target_M']:.3f}M)")
        print("    By module:")
        for module, params in sorted(config['by_module'].items(), key=lambda x: -x[1]):
            print(f"      {module:<20}: {params/1e6:.3f}M")
    
    return results


def explore_with_pretrained_weights(top_configs: List[Dict], cache_dir: str = './pretrained_weights'):
    """
    Test loading pretrained weights for top configurations.
    """
    print("\n" + "=" * 80)
    print("TESTING PRETRAINED WEIGHT LOADING")
    print("=" * 80)
    
    weights_types = ['fcmae', 'im1k', 'im22k']
    
    for config in top_configs[:5]:
        backbone = config['backbone']
        attn_type = config['attn_type']
        mlp_ratio = config['mlp_ratio']
        
        print(f"\n{backbone} / {attn_type} / mlp_ratio={mlp_ratio}")
        print("-" * 50)
        
        for weights_type in weights_types:
            if backbone not in PRETRAINED_URLS:
                continue
            if weights_type not in PRETRAINED_URLS[backbone]:
                print(f"  {weights_type}: Not available")
                continue
            
            try:
                model = HybridConvNeXtV2Configurable(
                    backbone_variant=backbone,
                    attn_type=attn_type,
                    mlp_ratio=mlp_ratio,
                    num_classes=8,
                    stage3_blocks=9,
                    stage4_blocks=12,
                )
                
                matched, total = load_pretrained_weights(model, backbone, weights_type, cache_dir)
                params = count_parameters(model)
                
                print(f"  {weights_type}: Loaded {matched}/{total} keys, {params/1e6:.3f}M params")
                
                del model
            except Exception as e:
                print(f"  {weights_type}: Error - {e}")


def main():
    """Main entry point."""
    # Run parameter exploration
    results = run_parameter_exploration()
    
    # Save results
    import json
    with open('parameter_exploration_results.json', 'w') as f:
        # Convert to JSON-serializable format
        json_results = []
        for r in results:
            json_r = {k: v for k, v in r.items() if k != 'by_module'}
            json_r['by_module'] = {k: v/1e6 for k, v in r['by_module'].items()}
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to parameter_exploration_results.json")
    
    # Test pretrained weight loading
    print("\n" + "=" * 80)
    print("Would you like to test pretrained weight loading? (This will download weights)")
    print("Run with --download flag to enable.")
    print("=" * 80)
    
    if '--download' in sys.argv:
        explore_with_pretrained_weights(results)


if __name__ == '__main__':
    main()
