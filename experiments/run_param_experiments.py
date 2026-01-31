# experiments/run_param_experiments.py
"""
Run parameter experiments to find configurations matching the paper.

This script:
1. Creates hybrid models with different configurations
2. Downloads and loads all available pretrained weights
3. Counts parameters for each configuration
4. Identifies configurations closest to paper's 21.92M target
5. Generates a detailed report with recommendations

Usage:
    python experiments/run_param_experiments.py
    python experiments/run_param_experiments.py --download  # Also download weights

Output:
    - Console report with all configurations
    - JSON file with detailed results
    - Recommendations for training
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


# ==============================================================================
# Paper Target
# ==============================================================================

TARGET_PARAMS_M = 21.92

PAPER_RESULTS = {
    'accuracy': 0.9348,
    'precision': 0.9324,
    'recall': 0.9070,
    'f1_score': 0.9182,
    'params_M': 21.92,
}

PAPER_ABLATION = {
    'baseline': 24.30,
    'convnextv2_only': 26.14,
    'separable_attn_only': 20.12,
    'proposed_hybrid': 21.92,
}


# ==============================================================================
# Import model components
# ==============================================================================

try:
    from models.hybrid_model import HybridConvNeXtV2, BACKBONE_CONFIGS, PRETRAINED_URLS
except ImportError:
    print("Error: Could not import from models.hybrid_model")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


# ==============================================================================
# Configuration
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


# ==============================================================================
# Experiment Functions
# ==============================================================================

def run_experiment(config: ExperimentConfig, download_weights: bool = False) -> ExperimentResult:
    """Run a single configuration experiment."""
    try:
        # Determine if we should load pretrained weights
        pretrained = config.weights_type != 'none' and download_weights
        
        # Create model
        model = HybridConvNeXtV2(
            backbone_variant=config.backbone,
            attn_type=config.attn_type,
            mlp_ratio=config.mlp_ratio,
            attn_proj_ratio=config.attn_proj_ratio,
            stage3_blocks=config.stage3_blocks,
            stage4_blocks=config.stage4_blocks,
            num_classes=8,
            pretrained=pretrained,
            pretrained_weights=config.weights_type if pretrained else 'ft_1k',
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        params_by_component = {k: v / 1e6 for k, v in model.count_parameters().items()}
        
        # Weights loaded (approximation based on backbone)
        weights_loaded = 0
        weights_total = 0
        if pretrained and config.backbone in PRETRAINED_URLS:
            if config.weights_type in PRETRAINED_URLS[config.backbone]:
                # Estimate based on stages 1-2 parameters
                weights_total = int((params_by_component.get('stage1', 0) + 
                                    params_by_component.get('stage2', 0) +
                                    params_by_component.get('stem', 0) +
                                    params_by_component.get('down1', 0) +
                                    params_by_component.get('down2', 0) +
                                    params_by_component.get('down3', 0)) * 1000)
                weights_loaded = weights_total  # Assume all match
        
        result = ExperimentResult(
            config=config,
            total_params_M=total_params,
            diff_from_target_M=abs(total_params - TARGET_PARAMS_M),
            params_by_component=params_by_component,
            weights_loaded=weights_loaded,
            weights_total=weights_total,
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


def run_all_experiments(download_weights: bool = False) -> List[ExperimentResult]:
    """Run all configuration experiments."""
    # Define search space
    backbones = ['atto', 'femto', 'pico', 'nano', 'tiny']
    attn_types = ['minimal', 'paper', 'reduced', 'mobilevit']
    mlp_ratios = [0.5, 1.0, 1.125, 1.5, 2.0, 2.5, 3.0, 4.0]
    
    results = []
    total = len(backbones) * len(attn_types) * len(mlp_ratios)
    count = 0
    
    print(f"\nRunning {total} configurations...")
    
    for backbone in backbones:
        for attn_type in attn_types:
            for mlp_ratio in mlp_ratios:
                count += 1
                
                config = ExperimentConfig(
                    backbone=backbone,
                    attn_type=attn_type,
                    mlp_ratio=mlp_ratio,
                    weights_type='ft_1k' if download_weights else 'none',
                )
                
                result = run_experiment(config, download_weights)
                if result.success:
                    results.append(result)
                
                if count % 20 == 0:
                    print(f"  Progress: {count}/{total}")
    
    # Sort by distance to target
    results.sort(key=lambda r: r.diff_from_target_M)
    
    return results


# ==============================================================================
# Analysis Functions
# ==============================================================================

def analyze_results(results: List[ExperimentResult]) -> Dict:
    """Analyze experiment results and generate insights."""
    if not results:
        return {}
    
    # Group by backbone
    by_backbone = {}
    for r in results:
        bb = r.config.backbone
        if bb not in by_backbone:
            by_backbone[bb] = []
        by_backbone[bb].append(r)
    
    # Group by attention type
    by_attn = {}
    for r in results:
        at = r.config.attn_type
        if at not in by_attn:
            by_attn[at] = []
        by_attn[at].append(r)
    
    # Find best config for each backbone
    best_by_backbone = {}
    for bb, bb_results in by_backbone.items():
        best = min(bb_results, key=lambda r: r.diff_from_target_M)
        best_by_backbone[bb] = {
            'attn_type': best.config.attn_type,
            'mlp_ratio': best.config.mlp_ratio,
            'params_M': best.total_params_M,
            'diff_M': best.diff_from_target_M,
        }
    
    # Find configs within 0.5M of target
    close_configs = [r for r in results if r.diff_from_target_M < 0.5]
    
    # Exact or near-exact matches
    exact_matches = [r for r in results if r.diff_from_target_M < 0.1]
    
    return {
        'best_by_backbone': best_by_backbone,
        'close_configs_count': len(close_configs),
        'exact_matches_count': len(exact_matches),
        'total_configs_tested': len(results),
    }


def print_report(results: List[ExperimentResult], analysis: Dict):
    """Print detailed experiment report."""
    print("\n" + "=" * 100)
    print("HYBRID CONVNEXTV2 + SEPARABLE ATTENTION PARAMETER EXPERIMENT REPORT")
    print("=" * 100)
    
    print(f"\n📎 Paper Target: {TARGET_PARAMS_M}M parameters")
    print(f"📊 Paper Results: Acc={PAPER_RESULTS['accuracy']}, Prec={PAPER_RESULTS['precision']}, "
          f"Rec={PAPER_RESULTS['recall']}, F1={PAPER_RESULTS['f1_score']}")
    
    print(f"\n📈 Configurations tested: {len(results)}")
    print(f"✅ Configs within 0.5M of target: {analysis.get('close_configs_count', 0)}")
    print(f"🎯 Exact matches (<0.1M diff): {analysis.get('exact_matches_count', 0)}")
    
    # Top 30 configurations
    print("\n" + "-" * 100)
    print("TOP 30 CONFIGURATIONS (sorted by distance to target)")
    print("-" * 100)
    print(f"{'Rank':<6}{'Backbone':<10}{'Attention':<12}{'MLP Ratio':<12}{'Params (M)':<14}{'Diff (M)':<12}{'Match':<10}")
    print("-" * 100)
    
    for i, r in enumerate(results[:30]):
        match = "🎯" if r.diff_from_target_M < 0.1 else ("✅" if r.diff_from_target_M < 0.5 else "")
        print(f"{i+1:<6}{r.config.backbone:<10}{r.config.attn_type:<12}{r.config.mlp_ratio:<12.3f}"
              f"{r.total_params_M:<14.3f}{r.diff_from_target_M:<12.3f}{match:<10}")
    
    # Best configuration for each backbone
    print("\n" + "-" * 100)
    print("BEST CONFIGURATION FOR EACH BACKBONE")
    print("-" * 100)
    
    for bb, best in analysis.get('best_by_backbone', {}).items():
        print(f"  {bb:<10}: attn={best['attn_type']:<10} mlp={best['mlp_ratio']:<6.2f} "
              f"params={best['params_M']:.2f}M (diff={best['diff_M']:.2f}M)")
    
    # Detailed breakdown of top 5
    print("\n" + "-" * 100)
    print("DETAILED BREAKDOWN OF TOP 5 CONFIGURATIONS")
    print("-" * 100)
    
    for i, r in enumerate(results[:5]):
        print(f"\n[{i+1}] {r.config.backbone} / {r.config.attn_type} / mlp_ratio={r.config.mlp_ratio}")
        print(f"    Total Parameters: {r.total_params_M:.3f}M")
        print(f"    Difference from target: {r.diff_from_target_M:.3f}M")
        
        # Component breakdown
        print("    Parameters by component:")
        for comp in ['stem', 'stage1', 'down1', 'stage2', 'down2', 'stage3', 'down3', 'stage4', 'norm', 'head']:
            params = r.params_by_component.get(comp, 0)
            print(f"      {comp:<15}: {params:.3f}M")
        
        # Stage summary
        convnext = r.params_by_component.get('convnext_stages', 0)
        transformer = r.params_by_component.get('transformer_stages', 0)
        print(f"    ConvNeXtV2 stages (1-2): {convnext:.3f}M")
        print(f"    Transformer stages (3-4): {transformer:.3f}M")
    
    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    if results:
        best = results[0]
        print(f"""
Based on the analysis, the configuration closest to the paper's {TARGET_PARAMS_M}M parameters is:

  ╔════════════════════════════════════════════════════════════════╗
  ║  Backbone:          {best.config.backbone:<40} ║
  ║  Attention Type:    {best.config.attn_type:<40} ║
  ║  MLP Ratio:         {best.config.mlp_ratio:<40.3f} ║
  ║  Total Parameters:  {best.total_params_M:<40.3f}M ║
  ║  Difference:        {best.diff_from_target_M:<40.3f}M ║
  ╚════════════════════════════════════════════════════════════════╝

To train with this configuration, run:

    python train.py \\
        --train_csv data/train.csv \\
        --val_csv data/val.csv \\
        --img_dir data/images \\
        --backbone {best.config.backbone} \\
        --attn_type {best.config.attn_type} \\
        --mlp_ratio {best.config.mlp_ratio} \\
        --pretrained_weights ft_1k \\
        --epochs 100
""")
    
    # Analysis notes
    print("\n" + "-" * 100)
    print("ANALYSIS NOTES")
    print("-" * 100)
    print("""
Key observations:

1. ATTENTION TYPE IMPACT ON PARAMETERS:
   - 'minimal': O(d) params - very few attention parameters (~3d per layer)
   - 'paper': O(d²) params - matches paper equations (5)-(10) exactly
   - 'reduced': O(d*d_proj) params - intermediate option
   - 'mobilevit': O(d²) params - MobileViT-v2 style (2*d² + d)

2. MLP RATIO IMPACT:
   - Standard ViT uses 4x expansion (large increase in params)
   - Lower ratios (1-2x) significantly reduce parameter count
   - Paper likely uses non-standard MLP ratio to achieve 21.92M target

3. BACKBONE CHOICE:
   - 'tiny': dims [96, 192, 384, 768], depths [3, 3, 9, 3]
   - 'nano': dims [80, 160, 320, 640], depths [2, 2, 8, 2]
   - Paper uses custom transformer depths [3, 3, 9, 12]

4. PARAMETER DISTRIBUTION (typical for target ~21.92M):
   - ConvNeXtV2 stages (1-2): ~1-3M params
   - Transformer stages (3-4): ~15-20M params
   - Other (stem, downsamples, head): ~1-2M params

5. PRETRAINED WEIGHTS:
   - Only stages 1-2 use pretrained ConvNeXtV2 weights
   - Stages 3-4 (Transformer) are initialized from scratch
   - Available: FCMAE (self-supervised), IM-1K (fine-tuned), IM-22K (fine-tuned)

6. MOST LIKELY PAPER CONFIGURATION:
   Based on parameter count matching, the paper likely uses:
   - 'tiny' backbone dimensions [96, 192, 384, 768]
   - 'minimal' or modified attention (to keep params low)
   - MLP ratio around 1.0-1.5 (NOT standard 4x)
   - Stage depths [3, 3, 9, 12] as stated in paper
""")


def save_results(results: List[ExperimentResult], output_path: str):
    """Save results to JSON file."""
    data = []
    for r in results:
        item = {
            'backbone': r.config.backbone,
            'attn_type': r.config.attn_type,
            'mlp_ratio': r.config.mlp_ratio,
            'stage3_blocks': r.config.stage3_blocks,
            'stage4_blocks': r.config.stage4_blocks,
            'total_params_M': round(r.total_params_M, 3),
            'diff_from_target_M': round(r.diff_from_target_M, 3),
            'params_by_component': {k: round(v, 4) for k, v in r.params_by_component.items()},
        }
        data.append(item)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run parameter experiments")
    parser.add_argument("--download", action="store_true", 
                       help="Download and load pretrained weights")
    parser.add_argument("--output", type=str, default="param_experiment_results.json",
                       help="Output JSON file path")
    args = parser.parse_args()
    
    print("=" * 100)
    print("HYBRID CONVNEXTV2 + SEPARABLE ATTENTION PARAMETER EXPERIMENTS")
    print("=" * 100)
    print(f"\nTarget: {TARGET_PARAMS_M}M parameters (from paper)")
    print(f"Download weights: {args.download}")
    
    # Run experiments
    results = run_all_experiments(download_weights=args.download)
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print report
    print_report(results, analysis)
    
    # Save results
    save_results(results, args.output)
    
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
