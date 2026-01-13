#!/usr/bin/env python3
"""
Training Script with Experiment Support
Modified to work with corrected hybrid model and experiment configurations
"""
import os
import sys
import time
import math
import json
import argparse
from collections import Counter
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

# Import corrected hybrid model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.hybrid_model import create_hybrid_model
from datasets.isic2019_dataset import ISIC2019Dataset, ISIC_CLASSES
from utils.augment import build_transforms
from utils.class_weights import compute_class_weights_from_csv
from utils.ema import ModelEMA
from utils.metrics import compute_classwise_metrics


def parse_args():
    """Parse command line arguments with experiment support"""
    p = argparse.ArgumentParser(description="Train Hybrid Model - Experiment Mode")
    
    # Experiment identification
    p.add_argument("--experiment_id", required=True, help="Unique experiment identifier")
    
    # Data paths
    p.add_argument("--train_csv", required=True, help="Path to training CSV")
    p.add_argument("--val_csv", required=True, help="Path to validation CSV")
    p.add_argument("--test_csv", required=True, help="Path to test CSV")
    p.add_argument("--img_dir", required=True, help="Directory containing images")
    
    # Model configuration - KEY PARAMETERS FOR EXPERIMENTS
    p.add_argument("--mlp_expansion", type=float, default=1.30,
                   help="MLP expansion ratio in transformer blocks (default: 1.30 for 21.92M params)")
    p.add_argument("--pretrained_source", type=str, default="imagenet1k",
                   choices=["imagenet1k", "imagenet22k"],
                   help="Pretrained weights source")
    
    # Training hyperparameters (paper defaults)
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--peak_lr", type=float, default=0.01, help="Peak learning rate")
    p.add_argument("--start_lr", type=float, default=1e-5, help="Warmup start LR")
    p.add_argument("--weight_decay", type=float, default=2e-5, help="Weight decay")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    p.add_argument("--mixup_alpha", type=float, default=0.4, help="Mixup alpha")
    p.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay")
    p.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping")
    
    # System
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--device", type=str, default="cuda", help="Device to use")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument("--save_freq", type=int, default=10, help="Save checkpoint every N epochs")
    
    # Flags
    p.add_argument("--no_scheduler", action="store_true", help="Disable LR scheduler")
    p.add_argument("--no_ema", action="store_true", help="Disable model EMA")
    p.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    
    return p.parse_args()


def set_seeds(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_scheduler(optimizer, total_epochs, iters_per_epoch, warmup_epochs, peak_lr, start_lr):
    """Create learning rate scheduler matching paper methodology"""
    total_steps = total_epochs * iters_per_epoch
    warmup_steps = warmup_epochs * iters_per_epoch
    
    peak_factor = float(peak_lr) / float(start_lr)
    min_factor = 1e-6 / float(start_lr)

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            progress = float(step) / max(1, warmup_steps)
            return 1.0 + (peak_factor - 1.0) * progress
        else:
            # Cosine annealing
            progress = (float(step) - warmup_steps) / max(1, (total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (peak_factor - min_factor) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_data(x, y, alpha=0.4):
    """Apply mixup augmentation"""
    if alpha <= 0:
        return x, y, y, 1.0
    
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def build_weighted_sampler(dataset, csv_path):
    """Build WeightedRandomSampler to handle class imbalance"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    if "label" in df.columns:
        labels = df["label"].astype(int).values
    else:
        available = [c for c in ISIC_CLASSES if c in df.columns]
        onehots = df[available].fillna(0).astype(int).values
        argmax_idx = onehots.argmax(axis=1)
        mapping = {i: ISIC_CLASSES.index(col) for i, col in enumerate(available)}
        labels = np.array([mapping[int(a)] for a in argmax_idx])
    
    if hasattr(dataset, "indices"):
        labels = labels[dataset.indices]
    
    class_counts = np.bincount(labels, minlength=len(ISIC_CLASSES)).astype(float) + 1e-6
    inv_freq = 1.0 / class_counts
    sample_weights = inv_freq[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, 
                   scheduler=None, ema=None, args=None, global_step=[0]):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
    
    for batch_idx, (imgs, labels, _) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply mixup after warmup
        use_mixup = args.mixup_alpha > 0 and epoch >= args.warmup_epochs
        
        if use_mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=args.mixup_alpha)
            logits = model(imgs)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            _, preds = torch.max(logits, 1)
            correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        if scheduler is not None and not args.no_scheduler:
            scheduler.step()
        
        if ema is not None and not args.no_ema and epoch >= args.warmup_epochs:
            ema.update(model)
        
        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': running_loss / total,
            'acc': 100.0 * correct / total,
            'lr': f'{current_lr:.6f}'
        })
        
        global_step[0] += 1
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, device, desc="Val"):
    """Validate model"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for imgs, labels, _ in tqdm(loader, desc=desc):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
    
    per_class, overall = compute_classwise_metrics(all_targets, all_preds)
    
    return per_class, overall


def save_experiment_results(args, model, test_metrics, best_val_f1, out_dir):
    """Save comprehensive experiment results"""
    
    config = model.get_config()
    
    results = {
        "experiment_id": args.experiment_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "peak_lr": args.peak_lr,
            "start_lr": args.start_lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "warmup_epochs": args.warmup_epochs,
            "mixup_alpha": args.mixup_alpha,
            "ema_decay": args.ema_decay,
            "pretrained_source": args.pretrained_source,
        },
        "test_metrics": test_metrics,
        "best_val_f1": best_val_f1,
    }
    
    results_path = os.path.join(out_dir, "final_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    return results


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print(f"EXPERIMENT: {args.experiment_id}")
    print("="*60)
    print(f"Device: {device}")
    print(f"MLP Expansion: {args.mlp_expansion}")
    print(f"Pretrained Source: {args.pretrained_source}")
    print(f"Output Directory: {args.out_dir}")
    print("="*60)
    
    set_seeds(args.seed)
    
    # Build transforms
    train_transform = build_transforms(train=True, input_size=224)
    val_transform = build_transforms(train=False, input_size=224)
    
    # Build datasets
    train_ds = ISIC2019Dataset(args.train_csv, args.img_dir, transform=train_transform)
    val_ds = ISIC2019Dataset(args.val_csv, args.img_dir, transform=val_transform)
    test_ds = ISIC2019Dataset(args.test_csv, args.img_dir, transform=val_transform)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val:   {len(val_ds)}")
    print(f"  Test:  {len(test_ds)}")
    
    # Build dataloaders
    train_sampler = build_weighted_sampler(train_ds, args.train_csv)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Build model using corrected hybrid model
    print("\n" + "="*60)
    print("Building Model...")
    print("="*60)
    
    model = create_hybrid_model(
        num_classes=len(ISIC_CLASSES),
        mlp_expansion=args.mlp_expansion,
        pretrained=True,
        pretrained_source=args.pretrained_source
    )
    model = model.to(device)
    
    # Loss function with class weights
    class_weights = compute_class_weights_from_csv(
        args.train_csv,
        class_names=ISIC_CLASSES,
        max_weight=10.0
    )
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.start_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=False
    )
    
    # Scheduler
    if not args.no_scheduler:
        scheduler = get_scheduler(
            optimizer,
            total_epochs=args.epochs,
            iters_per_epoch=len(train_loader),
            warmup_epochs=args.warmup_epochs,
            peak_lr=args.peak_lr,
            start_lr=args.start_lr
        )
    else:
        scheduler = None
    
    # EMA
    if not args.no_ema:
        ema = ModelEMA(model, decay=args.ema_decay, device=device)
    else:
        ema = None
    
    # Resume if specified
    start_epoch = 0
    best_val_f1 = 0.0
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if ema and 'ema_state' in ckpt:
            ema.load_state_dict(ckpt['ema_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    global_step = [0]
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs-1}")
        print("-"*60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, scheduler, ema, args, global_step
        )
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # Validate
        eval_model = ema.ema if (ema and epoch >= args.warmup_epochs) else model
        per_class, overall = validate(eval_model, val_loader, device)
        
        print(f"Val   - Acc: {overall['accuracy']*100:.2f}%, "
              f"Prec: {overall['precision_macro']*100:.2f}%, "
              f"Rec: {overall['recall_macro']*100:.2f}%, "
              f"F1: {overall['macro_f1']*100:.2f}%")
        
        # Save checkpoint
        is_best = overall['macro_f1'] > best_val_f1
        if is_best:
            best_val_f1 = overall['macro_f1']
        
        if epoch % args.save_freq == 0 or is_best:
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'ema_state': ema.state_dict() if ema else None,
                'best_val_f1': best_val_f1,
                'overall_metrics': overall,
                'per_class_metrics': per_class,
                'args': vars(args)
            }
            
            if is_best:
                best_path = os.path.join(args.out_dir, "best_model.pt")
                torch.save(ckpt, best_path)
                print(f"  ✓ New best F1: {best_val_f1*100:.2f}%")
    
    # Final test evaluation
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    
    best_ckpt = torch.load(os.path.join(args.out_dir, "best_model.pt"))
    model.load_state_dict(best_ckpt['model_state'])
    
    per_class, overall = validate(model, test_loader, device, desc="Test")
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {overall['accuracy']*100:.2f}%")
    print(f"  Precision: {overall['precision_macro']*100:.2f}%")
    print(f"  Recall:    {overall['recall_macro']*100:.2f}%")
    print(f"  F1-score:  {overall['macro_f1']*100:.2f}%")
    
    print(f"\nPaper Targets:")
    print(f"  Accuracy:  93.48%")
    print(f"  Precision: 93.24%")
    print(f"  Recall:    90.70%")
    print(f"  F1-score:  91.82%")
    
    # Save comprehensive results
    save_experiment_results(args, model, overall, best_val_f1, args.out_dir)
    
    print("\n" + "="*60)
    print(f"Experiment {args.experiment_id} Complete!")
    print("="*60)


if __name__ == "__main__":
    main()