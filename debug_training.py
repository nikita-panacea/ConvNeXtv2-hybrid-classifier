#!/usr/bin/env python3
"""
Debug script to diagnose training issues.
Checks for common problems that cause poor performance or instability.
"""
import torch
import torch.nn as nn
import numpy as np
from models.hybrid_model import HybridConvNeXtV2
from datasets.isic2019_dataset import ISIC2019Dataset
from utils.augment import build_transforms
from torch.utils.data import DataLoader


def check_model_initialization():
    """Check if model initializes properly"""
    print("="*60)
    print("1. MODEL INITIALIZATION CHECK")
    print("="*60)
    
    model = HybridConvNeXtV2(num_classes=8, pretrained=False)
    
    # Check for dead neurons
    print("\nChecking initial weight statistics:")
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            std = param.data.std().item()
            mean = param.data.mean().item()
            zeros = (param.data == 0).sum().item()
            total = param.numel()
            
            if std < 1e-6:
                print(f"  ⚠ {name}: std={std:.6f} (too small!)")
            elif abs(mean) > 1.0:
                print(f"  ⚠ {name}: mean={mean:.4f} (large bias)")
            elif zeros / total > 0.5:
                print(f"  ⚠ {name}: {zeros}/{total} zeros ({100*zeros/total:.1f}%)")
    
    print("  ✓ Weight initialization check complete")
    
    # Test forward pass with random input
    print("\nTesting forward pass:")
    x = torch.randn(4, 3, 224, 224)
    model.eval()
    
    with torch.no_grad():
        y = model(x)
    
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Output range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  Output mean/std: {y.mean():.4f} / {y.std():.4f}")
    
    if torch.isnan(y).any():
        print(f"  ✗ Output contains NaN!")
        return False
    elif torch.isinf(y).any():
        print(f"  ✗ Output contains Inf!")
        return False
    elif y.std() < 0.01:
        print(f"  ⚠ Output has very low variance")
    else:
        print(f"  ✓ Forward pass OK")
    
    return True


def check_gradient_flow():
    """Check if gradients flow properly"""
    print("\n" + "="*60)
    print("2. GRADIENT FLOW CHECK")
    print("="*60)
    
    model = HybridConvNeXtV2(num_classes=8, pretrained=False)
    model.train()
    
    # Create dummy batch
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 8, (4,))
    
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass
    logits = model(x)
    loss = criterion(logits, y)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print("\nChecking gradient statistics:")
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            
            grad_stats[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std
            }
    
    # Find problematic layers
    zero_grads = [name for name, stats in grad_stats.items() if stats['norm'] < 1e-7]
    large_grads = [name for name, stats in grad_stats.items() if stats['norm'] > 100]
    
    if zero_grads:
        print(f"  ⚠ {len(zero_grads)} layers with near-zero gradients:")
        for name in zero_grads[:5]:
            print(f"    - {name}")
    
    if large_grads:
        print(f"  ⚠ {len(large_grads)} layers with large gradients:")
        for name in large_grads[:5]:
            print(f"    - {name}: {grad_stats[name]['norm']:.2f}")
    
    if not zero_grads and not large_grads:
        print(f"  ✓ Gradient flow looks healthy")
    
    # Check gradient norm distribution
    all_norms = [stats['norm'] for stats in grad_stats.values()]
    print(f"\nGradient norm statistics:")
    print(f"  Min: {min(all_norms):.6f}")
    print(f"  Max: {max(all_norms):.6f}")
    print(f"  Mean: {np.mean(all_norms):.6f}")
    print(f"  Median: {np.median(all_norms):.6f}")
    
    return len(zero_grads) == 0


def check_learning_rate_schedule():
    """Check learning rate schedule"""
    print("\n" + "="*60)
    print("3. LEARNING RATE SCHEDULE CHECK")
    print("="*60)
    
    import math
    
    # Simulate schedule
    total_epochs = 100
    iters_per_epoch = 555  # Approximate from your training
    warmup_epochs = 5
    peak_lr = 0.01
    start_lr = 1e-5
    
    total_steps = total_epochs * iters_per_epoch
    warmup_steps = warmup_epochs * iters_per_epoch
    
    peak_factor = float(peak_lr) / float(start_lr)
    min_factor = 1e-6 / float(start_lr)
    
    def lr_lambda(step):
        if step < warmup_steps:
            progress = float(step) / max(1, warmup_steps)
            return 1.0 + (peak_factor - 1.0) * progress
        else:
            progress = (float(step) - warmup_steps) / max(1, (total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (peak_factor - min_factor) * cosine_decay
    
    # Check key points
    print("\nLearning rate at key epochs:")
    epochs_to_check = [0, 1, 2, 4, 5, 10, 20, 50, 99]
    
    for epoch in epochs_to_check:
        step = epoch * iters_per_epoch
        factor = lr_lambda(step)
        lr = start_lr * factor
        print(f"  Epoch {epoch:3d}: LR = {lr:.6f} (factor = {factor:.2f})")
    
    # Check that warmup reaches peak
    end_warmup_step = warmup_epochs * iters_per_epoch
    end_warmup_factor = lr_lambda(end_warmup_step)
    end_warmup_lr = start_lr * end_warmup_factor
    
    print(f"\nWarmup check:")
    print(f"  Start LR: {start_lr:.6f}")
    print(f"  Peak LR (target): {peak_lr:.6f}")
    print(f"  LR at end of warmup: {end_warmup_lr:.6f}")
    
    if abs(end_warmup_lr - peak_lr) / peak_lr < 0.01:
        print(f"  ✓ Warmup reaches peak LR correctly")
    else:
        print(f"  ✗ Warmup does not reach peak LR!")
        return False
    
    return True


def check_data_loading(train_csv, img_dir):
    """Check data loading and class distribution"""
    print("\n" + "="*60)
    print("4. DATA LOADING CHECK")
    print("="*60)
    
    try:
        transform = build_transforms(train=False, input_size=224)
        dataset = ISIC2019Dataset(train_csv, img_dir, transform=transform)
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Check a few samples
        print("\nChecking first 5 samples:")
        for i in range(min(5, len(dataset))):
            img, label, name = dataset[i]
            print(f"  Sample {i}: {name} -> class {label}, shape {img.shape}")
            
            if torch.isnan(img).any():
                print(f"    ✗ Image contains NaN!")
            if img.min() < -3 or img.max() > 3:
                print(f"    ⚠ Unusual range: [{img.min():.2f}, {img.max():.2f}]")
        
        # Check class distribution
        print("\nClass distribution:")
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        all_labels = []
        
        for imgs, labels, _ in loader:
            all_labels.extend(labels.cpu().numpy().tolist())
            if len(all_labels) > 1000:  # Sample first 1000
                break
        
        from collections import Counter
        counts = Counter(all_labels)
        
        for label in sorted(counts.keys()):
            print(f"  Class {label}: {counts[label]} samples")
        
        # Check imbalance ratio
        max_count = max(counts.values())
        min_count = min(counts.values())
        ratio = max_count / min_count
        
        print(f"\nClass imbalance ratio: {ratio:.1f}:1")
        if ratio > 50:
            print(f"  ⚠ Very high class imbalance!")
        else:
            print(f"  ✓ Class imbalance within reasonable range")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return False


def check_loss_function():
    """Check loss function behavior"""
    print("\n" + "="*60)
    print("5. LOSS FUNCTION CHECK")
    print("="*60)
    
    # Test with and without class weights
    class_weights = torch.tensor([5.6, 1.97, 7.62, 10.0, 9.65, 10.0, 10.0, 10.0])
    
    # Create dummy predictions (slightly favoring class 1, the majority class)
    logits = torch.randn(32, 8)
    logits[:, 1] += 0.5  # Bias toward class 1
    
    labels = torch.randint(0, 8, (32,))
    
    # Without weights
    criterion_no_weight = nn.CrossEntropyLoss()
    loss_no_weight = criterion_no_weight(logits, labels)
    
    # With weights
    criterion_with_weight = nn.CrossEntropyLoss(weight=class_weights)
    loss_with_weight = criterion_with_weight(logits, labels)
    
    print(f"\nLoss comparison:")
    print(f"  Without weights: {loss_no_weight.item():.4f}")
    print(f"  With weights: {loss_with_weight.item():.4f}")
    print(f"  Ratio: {(loss_with_weight / loss_no_weight).item():.2f}x")
    
    if loss_with_weight > loss_no_weight * 0.5:
        print(f"  ✓ Class weights are having an effect")
    else:
        print(f"  ⚠ Class weights may be too weak")
    
    return True


def main():
    """Run all diagnostic checks"""
    print("\n" + "="*60)
    print("TRAINING DIAGNOSTICS")
    print("="*60)
    
    results = {}
    
    # Model checks
    results['initialization'] = check_model_initialization()
    results['gradient_flow'] = check_gradient_flow()
    results['lr_schedule'] = check_learning_rate_schedule()
    
    # Data checks (provide paths if available)
    import sys
    if len(sys.argv) > 2:
        train_csv = sys.argv[1]
        img_dir = sys.argv[2]
        results['data_loading'] = check_data_loading(train_csv, img_dir)
    else:
        print("\nSkipping data loading check (provide train_csv and img_dir as args)")
        results['data_loading'] = None
    
    results['loss_function'] = check_loss_function()
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for check, passed in results.items():
        if passed is None:
            status = "⊘ SKIP"
        elif passed:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"{status}: {check.replace('_', ' ').title()}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if not results['gradient_flow']:
        print("\n⚠ Gradient flow issues detected:")
        print("  - Consider using gradient clipping: --grad_clip 1.0")
        print("  - Check if learning rate is too high")
        print("  - Verify model architecture for skip connections")
    
    if results['data_loading'] is False:
        print("\n⚠ Data loading issues detected:")
        print("  - Verify CSV format and image paths")
        print("  - Check image preprocessing")
    
    print("\n✓ Diagnostic check complete")


if __name__ == "__main__":
    main()