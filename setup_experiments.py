#!/usr/bin/env python3
"""
Experiment Runner for Paper Replication
Systematically tests different model configurations to match paper results
"""
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List

# Experiment configurations matching paper analysis
EXPERIMENTS = {
    "exp1_optimal_in1k": {
        "name": "Optimal MLP + ImageNet-1K",
        "mlp_expansion": 1.30,  # Achieves ~21.92M params
        "pretrained_source": "imagenet1k",
        "epochs": 100,
        "description": "Paper-matched parameters with ImageNet-1K pretraining"
    },
    "exp2_optimal_in22k": {
        "name": "Optimal MLP + ImageNet-22K",
        "mlp_expansion": 1.30,
        "pretrained_source": "imagenet22k",
        "epochs": 100,
        "description": "Paper-matched parameters with ImageNet-22K pretraining"
    },
    "exp3_extended_in1k": {
        "name": "Optimal MLP + Extended Training",
        "mlp_expansion": 1.30,
        "pretrained_source": "imagenet1k",
        "epochs": 150,  # Extended training
        "description": "Extended training with paper-matched parameters"
    },
    "exp4_2x_mlp_in1k": {
        "name": "2x MLP + ImageNet-1K",
        "mlp_expansion": 2.0,
        "pretrained_source": "imagenet1k",
        "epochs": 100,
        "description": "Standard 2x MLP expansion baseline"
    },
    "exp5_half_mlp_in1k": {
        "name": "0.5x MLP + ImageNet-1K (current)",
        "mlp_expansion": 0.5,
        "pretrained_source": "imagenet1k",
        "epochs": 100,
        "description": "Current implementation for comparison"
    },
}

def generate_experiment_script(exp_id: str, config: Dict, base_args: Dict) -> str:
    """
    Generate training script for an experiment.
    
    Args:
        exp_id: Experiment identifier
        config: Experiment configuration
        base_args: Base arguments (data paths, etc.)
    
    Returns:
        Shell script content
    """
    out_dir = os.path.join(base_args['base_out_dir'], exp_id)
    
    script = f"""#!/bin/bash
# Experiment: {exp_id}
# Description: {config['description']}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

set -e

echo "=========================================="
echo "Starting Experiment: {exp_id}"
echo "Description: {config['description']}"
echo "=========================================="

python train_experiment.py \\
    --experiment_id {exp_id} \\
    --train_csv {base_args['train_csv']} \\
    --val_csv {base_args['val_csv']} \\
    --test_csv {base_args['test_csv']} \\
    --img_dir {base_args['img_dir']} \\
    --epochs {config['epochs']} \\
    --batch_size {base_args['batch_size']} \\
    --mlp_expansion {config['mlp_expansion']} \\
    --pretrained_source {config['pretrained_source']} \\
    --out_dir {out_dir} \\
    --peak_lr 0.01 \\
    --start_lr 1e-5 \\
    --weight_decay 2e-5 \\
    --momentum 0.9 \\
    --warmup_epochs 5 \\
    --mixup_alpha 0.4 \\
    --ema_decay 0.9999 \\
    --num_workers {base_args['num_workers']} \\
    --device cuda \\
    --seed 42

echo ""
echo "=========================================="
echo "Experiment {exp_id} completed!"
echo "Results saved to: {out_dir}"
echo "=========================================="
"""
    return script

def generate_batch_script(exp_ids: List[str], sequential: bool = True) -> str:
    """
    Generate script to run multiple experiments.
    
    Args:
        exp_ids: List of experiment IDs to run
        sequential: If True, run sequentially; else run in parallel
    
    Returns:
        Batch script content
    """
    script = f"""#!/bin/bash
# Batch Experiment Runner
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

set -e

echo "=========================================="
echo "Batch Experiment Runner"
echo "Mode: {'Sequential' if sequential else 'Parallel'}"
echo "Experiments: {', '.join(exp_ids)}"
echo "=========================================="
echo ""

"""
    
    if sequential:
        for exp_id in exp_ids:
            script += f"""
echo "Starting {exp_id}..."
bash experiments/{exp_id}.sh
echo "{exp_id} completed!"
echo ""
"""
    else:
        # Parallel execution
        for exp_id in exp_ids:
            script += f"bash experiments/{exp_id}.sh &\n"
        script += "\nwait\n"
        script += 'echo "All experiments completed!"\n'
    
    return script

def generate_analysis_script(exp_ids: List[str]) -> str:
    """Generate script to analyze and compare experiment results."""
    
    script = f"""#!/usr/bin/env python3
\"\"\"
Experiment Results Analysis
Compares results from all experiments
\"\"\"
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Experiment IDs to analyze
EXPERIMENT_IDS = {exp_ids}

def load_experiment_results(exp_id):
    \"\"\"Load results from an experiment.\"\"\"
    base_dir = Path('experiments') / exp_id
    
    # Load final results
    results_file = base_dir / 'final_results.json'
    if not results_file.exists():
        print(f"Warning: Results not found for {{exp_id}}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def create_comparison_table():
    \"\"\"Create comparison table of all experiments.\"\"\"
    
    data = []
    for exp_id in EXPERIMENT_IDS:
        results = load_experiment_results(exp_id)
        if results is None:
            continue
        
        config = results.get('config', {{}})
        test_metrics = results.get('test_metrics', {{}})
        
        data.append({{
            'Experiment': exp_id,
            'MLP Expansion': config.get('mlp_expansion', 'N/A'),
            'Pretrained': config.get('pretrained_source', 'N/A'),
            'Params (M)': config.get('parameters_M', 'N/A'),
            'Test Accuracy': test_metrics.get('accuracy', 0) * 100,
            'Test Precision': test_metrics.get('precision_macro', 0) * 100,
            'Test Recall': test_metrics.get('recall_macro', 0) * 100,
            'Test F1': test_metrics.get('macro_f1', 0) * 100,
        }})
    
    df = pd.DataFrame(data)
    
    # Sort by F1 score
    df = df.sort_values('Test F1', ascending=False)
    
    return df

def plot_results_comparison(df):
    \"\"\"Create visualization of experiment results.\"\"\"
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Experiment Results Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        df_sorted = df.sort_values(metric, ascending=True)
        
        bars = ax.barh(df_sorted['Experiment'], df_sorted[metric])
        
        # Color bars by performance
        for bar, value in zip(bars, df_sorted[metric]):
            if value >= 93:  # Paper target
                bar.set_color('green')
            elif value >= 90:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add paper target line
        ax.axvline(x=93.48 if metric == 'Test Accuracy' else 
                      93.24 if metric == 'Test Precision' else
                      90.70 if metric == 'Test Recall' else
                      91.82,  # F1
                   color='blue', linestyle='--', linewidth=2, 
                   label='Paper Target', alpha=0.7)
        
        ax.set_xlabel(f'{{metric}} (%)')
        ax.set_title(metric)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/results_comparison.png', dpi=300, bbox_inches='tight')
    print("\\nSaved comparison plot to: experiments/results_comparison.png")

def print_summary_report(df):
    \"\"\"Print formatted summary report.\"\"\"
    
    print("\\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Paper targets
    print("\\nPaper Targets:")
    print("  Accuracy:  93.48%")
    print("  Precision: 93.24%")
    print("  Recall:    90.70%")
    print("  F1-score:  91.82%")
    print("  Parameters: 21.92M")
    
    # Best experiment
    best_exp = df.iloc[0]
    print("\\nBest Experiment:")
    print(f"  ID: {{best_exp['Experiment']}}")
    print(f"  F1-score: {{best_exp['Test F1']:.2f}}%")
    print(f"  Accuracy: {{best_exp['Test Accuracy']:.2f}}%")
    print(f"  Parameters: {{best_exp['Params (M)']}}")
    
    # Gap analysis
    f1_gap = best_exp['Test F1'] - 91.82
    print(f"\\nGap from Paper:")
    print(f"  F1-score: {{f1_gap:+.2f}}%")
    
    if f1_gap >= -1.0:
        print("  Status: ✓ Within acceptable range!")
    else:
        print("  Status: ✗ Below target, needs investigation")

if __name__ == "__main__":
    print("Analyzing experiment results...")
    
    # Load and compare results
    df = create_comparison_table()
    
    if df.empty:
        print("Error: No results found!")
        exit(1)
    
    # Print summary
    print_summary_report(df)
    
    # Create visualizations
    print("\\nGenerating comparison plots...")
    plot_results_comparison(df)
    
    # Save results table
    df.to_csv('experiments/results_summary.csv', index=False)
    print("\\nSaved results table to: experiments/results_summary.csv")
"""
    
    return script

def main():
    parser = argparse.ArgumentParser(description="Setup experiments for paper replication")
    parser.add_argument("--train_csv", required=True, help="Training CSV path")
    parser.add_argument("--val_csv", required=True, help="Validation CSV path")
    parser.add_argument("--test_csv", required=True, help="Test CSV path")
    parser.add_argument("--img_dir", required=True, help="Image directory path")
    parser.add_argument("--base_out_dir", default="experiments", help="Base output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--experiments", nargs="+", 
                       choices=list(EXPERIMENTS.keys()) + ["all"],
                       default=["all"],
                       help="Which experiments to setup")
    parser.add_argument("--sequential", action="store_true",
                       help="Run experiments sequentially (default: parallel)")
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    if "all" in args.experiments:
        exp_ids = list(EXPERIMENTS.keys())
    else:
        exp_ids = args.experiments
    
    # Create base directory
    os.makedirs(args.base_out_dir, exist_ok=True)
    
    # Base arguments
    base_args = {
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "img_dir": args.img_dir,
        "base_out_dir": args.base_out_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    
    print("="*60)
    print("EXPERIMENT SETUP")
    print("="*60)
    print(f"Base output directory: {args.base_out_dir}")
    print(f"Number of experiments: {len(exp_ids)}")
    print(f"Execution mode: {'Sequential' if args.sequential else 'Parallel'}")
    print("="*60)
    
    # Generate individual experiment scripts
    print("\nGenerating experiment scripts...")
    for exp_id in exp_ids:
        config = EXPERIMENTS[exp_id]
        script_content = generate_experiment_script(exp_id, config, base_args)
        
        script_path = os.path.join(args.base_out_dir, f"{exp_id}.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        
        print(f"  ✓ {exp_id}: {config['name']}")
    
    # Generate batch runner script
    print("\nGenerating batch runner...")
    batch_script = generate_batch_script(exp_ids, args.sequential)
    batch_path = os.path.join(args.base_out_dir, "run_all.sh")
    with open(batch_path, 'w') as f:
        f.write(batch_script)
    os.chmod(batch_path, 0o755)
    print(f"  ✓ Batch runner: {batch_path}")
    
    # Generate analysis script
    print("\nGenerating analysis script...")
    analysis_script = generate_analysis_script(exp_ids)
    analysis_path = os.path.join(args.base_out_dir, "analyze_results.py")
    with open(analysis_path, 'w') as f:
        f.write(analysis_script)
    os.chmod(analysis_path, 0o755)
    print(f"  ✓ Analysis script: {analysis_path}")
    
    # Save experiment config
    config_path = os.path.join(args.base_out_dir, "experiments_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "experiments": EXPERIMENTS,
            "selected": exp_ids,
            "base_args": base_args,
            "generated_at": datetime.now().isoformat()
        }, f, indent=2)
    print(f"  ✓ Config saved: {config_path}")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nTo run experiments:")
    print(f"  cd {args.base_out_dir}")
    print(f"  bash run_all.sh")
    print("\nTo analyze results:")
    print(f"  python analyze_results.py")
    print("\nIndividual experiments can be run with:")
    for exp_id in exp_ids:
        print(f"  bash {exp_id}.sh")
    print("="*60)

if __name__ == "__main__":
    main()