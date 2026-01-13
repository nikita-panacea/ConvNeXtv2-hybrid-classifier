#!/usr/bin/env python3
"""
Training script for Hybrid ConvNeXtV2 + Separable Attention model
VERIFIED CONFIGURATION - Uses ConvNeXtV2 Nano with 21.92M parameters

Configuration (from systematic parameter search):
- Backbone: ConvNeXtV2 Nano
- Dims: [80, 160, 320, 640]
- Depths: [3, 3, 9, 12]
- MLP ratio: 0.25
- Parameters: 21.92M (matches paper exactly)

Paper methodology:
- Learning rate: 0.01 (peak after warmup)
- Warmup: 5 epochs from 1e-5 to 0.01
- Optimizer: SGD, momentum=0.9, weight_decay=2e-5
- Augmentation: scaling, smoothing, mixup, color jitter, flipping
- Loss: Categorical cross-entropy with class weights
- Model EMA: decay=0.9999
- Transfer learning: ImageNet-1K pretrained ConvNeXtV2 Nano

Expected results:
- Accuracy: 93.48%
- Precision: 93.24%
- Recall: 90.70%
- F1-score: 91.82%

Usage:
    python train.py \
        --train_csv path/to/train.csv \
        --val_csv path/to/val.csv \
        --test_csv path/to/test.csv \
        --img_dir path/to/images \
        --epochs 100 \
        --batch_size 32 \
        --out_dir checkpoints/
"""

import os
import time
import math
import argparse
import random
from collections import Counter
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Local imports
from datasets.isic2019_dataset import ISIC2019Dataset, ISIC_CLASSES
from utils.augment import build_transforms
from utils.class_weights import compute_class_weights_from_csv
from utils.ema import ModelEMA
from utils.metrics import compute_classwise_metrics
from models.custom_hybrid_model import create_hybrid_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Hybrid ConvNeXtV2 Nano model on ISIC 2019",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument("--train_csv", required=True, help="Path to training CSV")
    parser.add_argument("--val_csv", required=True, help="Path to validation CSV")
    parser.add_argument("--test_csv", default="", help="Path to test CSV (optional)")
    parser.add_argument("--img_dir", required=True, help="Directory containing images")
    
    # Model configuration (VERIFIED through systematic search)
    parser.add_argument("--mlp_ratio", type=float, default=0.25,
                       help="MLP expansion ratio (verified: 0.25 for 21.92M params)")
    parser.add_argument("--no_pretrained", action="store_true",
                       help="Don't use ImageNet pretrained weights")
    
    # Training hyperparameters (paper defaults)
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--peak_lr", type=float, default=0.01,
                       help="Peak learning rate (after warmup)")
    parser.add_argument("--start_lr", type=float, default=1e-5,
                       help="Starting learning rate (warmup)")
    parser.add_argument("--weight_decay", type=float, default=2e-5,
                       help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9,
                       help="SGD momentum")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                       help="Number of warmup epochs")
    
    # Augmentation
    parser.add_argument("--mixup_alpha", type=float, default=0.4,
                       help="Mixup alpha (0=disabled)")
    
    # Regularization
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                       help="Model EMA decay rate")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                       help="Gradient clipping (0=disabled)")
    
    # System
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Output
    parser.add_argument("--out_dir", type=str, default="checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--save_freq", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--log_freq", type=int, default=50,
                       help="Log training stats every N iterations")
    
    # Resume training
    parser.add_argument("--resume", type=str, default="",
                       help="Resume from checkpoint")
    
    return parser.parse_args()


def set_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_scheduler(optimizer, total_epochs, iters_per_epoch, warmup_epochs, peak_lr, start_lr):
    """
    Create learning rate scheduler with warmup and cosine annealing
    
    Paper methodology:
    - Phase 1 (warmup): Linear from start_lr to peak_lr over warmup_epochs
    - Phase 2 (cosine): Cosine annealing from peak_lr to ~0 over remaining epochs
    
    Args:
        optimizer: PyTorch optimizer (initialized with start_lr)
        total_epochs: Total training epochs
        iters_per_epoch: Iterations per epoch
        warmup_epochs: Number of warmup epochs
        peak_lr: Peak learning rate (after warmup)
        start_lr: Starting learning rate (at warmup start)
    
    Returns:
        LambdaLR scheduler
    """
    total_steps = total_epochs * iters_per_epoch
    warmup_steps = warmup_epochs * iters_per_epoch
    
    # Compute scaling factors relative to start_lr
    peak_factor = peak_lr / start_lr
    min_factor = 1e-6 / start_lr  # Minimum LR at end of training
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup: start_lr → peak_lr
            progress = step / max(1, warmup_steps)
            return 1.0 + (peak_factor - 1.0) * progress
        else:
            # Cosine annealing: peak_lr → min_lr
            progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (peak_factor - min_factor) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_data(x, y, alpha=0.4):
    """
    Apply mixup data augmentation
    
    Paper mentions mixup as one of the augmentation techniques.
    
    Args:
        x: Input images [B, C, H, W]
        y: Labels [B]
        alpha: Mixup interpolation strength (beta distribution parameter)
    
    Returns:
        mixed_x: Mixed images
        y_a, y_b: Original labels for the two mixed samples
        lam: Mixing coefficient
    """
    if alpha <= 0:
        return x, y, y, 1.0
    
    # Sample mixing coefficient from beta distribution
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    
    # Random permutation for mixing pairs
    index = torch.randperm(batch_size).to(x.device)
    
    # Mix images
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def build_weighted_sampler(csv_path, class_names=ISIC_CLASSES):
    """
    Build WeightedRandomSampler to handle class imbalance
    
    Uses inverse frequency weighting to oversample minority classes.
    
    Args:
        csv_path: Path to CSV file
        class_names: List of class names
    
    Returns:
        WeightedRandomSampler instance
    """
    df = pd.read_csv(csv_path)
    
    # Extract labels from CSV
    if "label" in df.columns:
        labels = df["label"].astype(int).values
    else:
        # One-hot encoded columns
        available = [c for c in class_names if c in df.columns]
        if not available:
            raise ValueError("CSV must have 'label' column or one-hot class columns")
        
        onehots = df[available].fillna(0).astype(int).values
        argmax_idx = onehots.argmax(axis=1)
        mapping = {i: class_names.index(col) for i, col in enumerate(available)}
        labels = np.array([mapping[int(a)] for a in argmax_idx])
    
    # Compute class weights (inverse frequency)
    class_counts = np.bincount(labels, minlength=len(class_names)).astype(float)
    class_counts = class_counts + 1e-6  # Avoid division by zero
    
    inv_freq = 1.0 / class_counts
    sample_weights = inv_freq[labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, args,
                   scheduler=None, ema=None, global_step=None):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        epoch: Current epoch number
        args: Command line arguments
        scheduler: Learning rate scheduler (optional)
        ema: Model EMA (optional)
        global_step: List with single element [step] for tracking
    
    Returns:
        avg_loss: Average training loss
        avg_acc: Average training accuracy
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")
    
    for batch_idx, (images, labels, _) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply mixup after warmup (paper methodology)
        use_mixup = args.mixup_alpha > 0 and epoch >= args.warmup_epochs
        
        if use_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels, alpha=args.mixup_alpha)
            
            # Forward pass
            logits = model(images)
            
            # Mixup loss
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            
            # Accuracy (approximate with mixup)
            _, preds = torch.max(logits, 1)
            correct += (lam * (preds == y_a).float() + 
                       (1 - lam) * (preds == y_b).float()).sum().item()
        else:
            # Standard forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Accuracy
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (if enabled)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Update EMA model (after warmup, as per paper)
        if ema is not None and epoch >= args.warmup_epochs:
            ema.update(model)
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.0*correct/total:.2f}%',
            'lr': f'{current_lr:.6f}'
        })
        
        if global_step is not None:
            global_step[0] += 1
    
    avg_loss = running_loss / total
    avg_acc = 100.0 * correct / total
    
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, device, desc="Validation"):
    """
    Validate model
    
    Args:
        model: Model to validate
        loader: Validation data loader
        device: Device to use
        desc: Description for progress bar
    
    Returns:
        per_class: Per-class metrics dict
        overall: Overall metrics dict
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for images, labels, _ in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        
        # Forward pass
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        
        # Collect predictions and labels
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    
    # Compute metrics
    per_class, overall = compute_classwise_metrics(all_labels, all_preds)
    
    return per_class, overall


def print_class_distribution(csv_path, class_names=ISIC_CLASSES):
    """Print class distribution from CSV"""
    df = pd.read_csv(csv_path)
    
    print("\nClass distribution:")
    
    if "label" in df.columns:
        counts = Counter(df["label"].values)
        for label in sorted(counts.keys()):
            class_name = class_names[label] if label < len(class_names) else f"Class{label}"
            count = counts[label]
            pct = 100.0 * count / len(df)
            print(f"  {class_name:>6s} ({label}): {count:>6d} ({pct:>5.2f}%)")
    else:
        for class_name in class_names:
            if class_name in df.columns:
                count = int(df[class_name].sum())
                pct = 100.0 * count / len(df)
                print(f"  {class_name:>6s}: {count:>6d} ({pct:>5.2f}%)")
    
    print(f"  {'Total':>6s}: {len(df):>6d}")


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    set_seeds(args.seed)
    
    # Print configuration
    print("\n" + "="*80)
    print("HYBRID ConvNeXtV2 NANO + SEPARABLE ATTENTION TRAINING")
    print("="*80)
    print(f"\nVERIFIED Configuration (from systematic search):")
    print(f"  Backbone: ConvNeXtV2 Nano")
    print(f"  Dims: [80, 160, 320, 640]")
    print(f"  Depths: [3, 3, 9, 12]")
    print(f"  MLP ratio: {args.mlp_ratio}")
    print(f"  Expected params: 21.92M")
    print(f"\nTraining settings:")
    print(f"  Device: {device}")
    print(f"  Pretrained: {not args.no_pretrained}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Peak LR: {args.peak_lr}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Output dir: {args.out_dir}")
    
    # Print dataset info
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)
    print(f"\nTraining set: {args.train_csv}")
    print_class_distribution(args.train_csv)
    
    print(f"\nValidation set: {args.val_csv}")
    print_class_distribution(args.val_csv)
    
    if args.test_csv:
        print(f"\nTest set: {args.test_csv}")
        print_class_distribution(args.test_csv)
    
    # Build transforms
    train_transform = build_transforms(train=True, input_size=224, deterministic=False)
    val_transform = build_transforms(train=False, input_size=224, deterministic=False)
    
    # Build datasets
    train_ds = ISIC2019Dataset(args.train_csv, args.img_dir, transform=train_transform)
    val_ds = ISIC2019Dataset(args.val_csv, args.img_dir, transform=val_transform)
    
    print(f"\nDataset sizes:")
    print(f"  Training:   {len(train_ds):>6d} samples")
    print(f"  Validation: {len(val_ds):>6d} samples")
    
    # Build data loaders with weighted sampling for class imbalance
    train_sampler = build_weighted_sampler(args.train_csv)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Build model
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    
    model = create_hybrid_model(
        num_classes=len(ISIC_CLASSES),
        pretrained=not args.no_pretrained,
        mlp_ratio=args.mlp_ratio
    )
    model = model.to(device)
    
    # Print parameter count
    params = model.count_parameters()
    target_params = 21920000
    
    print(f"\nParameter count:")
    print(f"  Total:      {params['total']:>12,} ({params['total']/1e6:>6.2f}M)")
    print(f"  Trainable:  {params['trainable']:>12,} ({params['trainable']/1e6:>6.2f}M)")
    print(f"  Target:     {target_params:>12,} ({target_params/1e6:>6.2f}M)")
    
    diff_pct = abs(params['total'] - target_params) / target_params * 100
    print(f"  Difference: {abs(params['total']-target_params):>12,} ({diff_pct:>6.2f}%)")
    
    if diff_pct < 0.01:
        print("  ✓✓✓ EXACT MATCH - Perfect replication!")
    elif diff_pct < 1:
        print("  ✓✓✓ Excellent match (<1%)")
    elif diff_pct < 5:
        print("  ✓✓ Good match (<5%)")
    
    # Loss function with class weights
    class_weights = compute_class_weights_from_csv(
        args.train_csv,
        class_names=ISIC_CLASSES,
        max_weight=10.0
    ).to(device)
    
    print(f"\nClass weights:")
    for i, (name, weight) in enumerate(zip(ISIC_CLASSES, class_weights)):
        print(f"  {name:>6s}: {weight:.4f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer (SGD as per paper)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.start_lr,  # Will be scaled by scheduler to peak_lr
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=False
    )
    
    # Learning rate scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        total_epochs=args.epochs,
        iters_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs,
        peak_lr=args.peak_lr,
        start_lr=args.start_lr
    )
    
    # Model EMA (Exponential Moving Average)
    ema = ModelEMA(model, decay=args.ema_decay, device=device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_f1 = 0.0
    
    if args.resume:
        print(f"\n" + "="*80)
        print(f"RESUMING FROM CHECKPOINT")
        print("="*80)
        print(f"Loading: {args.resume}")
        
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        ema.load_state_dict(ckpt['ema_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        
        print(f"Resumed from epoch {start_epoch}")
        print(f"Best val F1 so far: {best_val_f1*100:.2f}%")
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"\nTarget metrics (from paper):")
    print(f"  Accuracy:  93.48%")
    print(f"  Precision: 93.24%")
    print(f"  Recall:    90.70%")
    print(f"  F1-score:  91.82%")
    print("="*80)
    
    global_step = [0]
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args, scheduler, ema, global_step
        )
        
        print(f"\nTraining results:")
        print(f"  Loss:     {train_loss:.4f}")
        print(f"  Accuracy: {train_acc:.2f}%")
        print(f"  LR:       {optimizer.param_groups[0]['lr']:.6f}")
        
        # Validate with EMA model (after warmup) or regular model
        eval_model = ema.ema if epoch >= args.warmup_epochs else model
        per_class, overall = validate(eval_model, val_loader, device)
        
        # Print validation results
        print(f"\nValidation results:")
        print(f"  Accuracy:  {overall['accuracy']*100:>6.2f}% (target: 93.48%)")
        print(f"  Precision: {overall['precision_macro']*100:>6.2f}% (target: 93.24%)")
        print(f"  Recall:    {overall['recall_macro']*100:>6.2f}% (target: 90.70%)")
        print(f"  F1-score:  {overall['macro_f1']*100:>6.2f}% (target: 91.82%)")
        
        # Check if this is the best model
        is_best = overall['macro_f1'] > best_val_f1
        if is_best:
            best_val_f1 = overall['macro_f1']
            print(f"  ★★★ New best F1: {best_val_f1*100:.2f}% ★★★")
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or is_best or epoch == args.epochs - 1:
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema_state': ema.state_dict(),
                'best_val_f1': best_val_f1,
                'overall_metrics': overall,
                'per_class_metrics': per_class,
                'args': vars(args)
            }
            
            # Save periodic checkpoint
            if epoch % args.save_freq == 0:
                save_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch:03d}.pt")
                torch.save(ckpt, save_path)
                print(f"  Saved checkpoint: {save_path}")
            
            # Save best checkpoint
            if is_best:
                best_path = os.path.join(args.out_dir, "best_model.pt")
                torch.save(ckpt, best_path)
                print(f"  Saved best model: {best_path}")
            
            # Save last checkpoint
            if epoch == args.epochs - 1:
                last_path = os.path.join(args.out_dir, "last_model.pt")
                torch.save(ckpt, last_path)
                print(f"  Saved last model: {last_path}")
    
    # Final evaluation on test set (if provided)
    if args.test_csv:
        print("\n" + "="*80)
        print("FINAL TEST EVALUATION")
        print("="*80)
        
        test_ds = ISIC2019Dataset(args.test_csv, args.img_dir, transform=val_transform)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Load best model
        best_ckpt = torch.load(os.path.join(args.out_dir, "best_model.pt"))
        model.load_state_dict(best_ckpt['model_state'])
        
        per_class, overall = validate(model, test_loader, device, desc="Test")
        
        print(f"\nTest results (best model):")
        print(f"  Accuracy:  {overall['accuracy']*100:>6.2f}% (target: 93.48%)")
        print(f"  Precision: {overall['precision_macro']*100:>6.2f}% (target: 93.24%)")
        print(f"  Recall:    {overall['recall_macro']*100:>6.2f}% (target: 90.70%)")
        print(f"  F1-score:  {overall['macro_f1']*100:>6.2f}% (target: 91.82%)")
        
        print(f"\nPer-class metrics:")
        for class_name in ISIC_CLASSES:
            metrics = per_class[class_name]
            print(f"  {class_name:>6s}: "
                  f"P={metrics['precision']*100:>5.2f}% "
                  f"R={metrics['recall']*100:>5.2f}% "
                  f"F1={metrics['f1']*100:>5.2f}% "
                  f"(n={metrics['support']})")
    
    # Training complete
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation F1-score: {best_val_f1*100:.2f}%")
    print(f"Target (paper): 91.82%")
    
    if best_val_f1 >= 0.918:
        print("✓✓✓ TARGET ACHIEVED! ✓✓✓")
    elif best_val_f1 >= 0.90:
        print("✓✓ Close to target (>90%)")
    elif best_val_f1 >= 0.85:
        print("✓ Good performance (>85%)")
    
    print(f"\nCheckpoints saved to: {args.out_dir}")
    print("="*80)


if __name__ == "__main__":
    main()