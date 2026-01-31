# train.py
#!/usr/bin/env python3
"""
Training script matching paper methodology exactly.

Key paper specifications:
- Learning rate: 0.01 (peak)
- Base LR: 0.1 (for reference)
- Start LR (warmup): 1e-5
- Optimizer: SGD with momentum=0.9
- Weight decay: 2e-5
- Warmup: 5 epochs
- Loss: Categorical cross-entropy
- Data augmentation: scaling, smoothing, mix-up, color jitter, flipping
- Transfer learning: ImageNet pretrained
- Model EMA: decay=0.9999

"""
import os
import time
import math
import argparse
from collections import Counter
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from datasets.isic2019_dataset import ISIC2019Dataset, ISIC_CLASSES
from utils.augment import build_transforms
from utils.class_weights import compute_class_weights_from_csv
from utils.ema import ModelEMA
from utils.metrics import compute_classwise_metrics
from models.hybrid_model import HybridConvNeXtV2


def parse_args():
    """Parse command line arguments"""
    p = argparse.ArgumentParser(description="Train Hybrid ConvNeXtV2 model on ISIC 2019")
    
    # Data paths
    p.add_argument("--train_csv", required=True, help="Path to training CSV")
    p.add_argument("--val_csv", required=False, help="Path to validation CSV")
    p.add_argument("--test_csv", required=False, help="Path to test CSV")
    p.add_argument("--img_dir", required=True, help="Directory containing images")
    
    # Training hyperparameters (paper defaults)
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--peak_lr", type=float, default=0.01, help="Peak learning rate (paper: 0.01)")
    p.add_argument("--start_lr", type=float, default=1e-5, help="Warmup start LR (paper: 1e-5)")
    p.add_argument("--weight_decay", type=float, default=2e-5, help="Weight decay (paper: 2e-5)")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (paper: 0.9)")
    p.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs (paper: 5)")
    
    # Augmentation
    p.add_argument("--mixup_alpha", type=float, default=0.4, help="Mixup alpha (0=disabled)")
    
    # Model EMA
    p.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay (paper: 0.9999)")
    
    # System
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--device", type=str, default="cuda", help="Device to use")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output
    p.add_argument("--out_dir", type=str, default="checkpoints", help="Output directory")
    p.add_argument("--save_freq", type=int, default=10, help="Save checkpoint every N epochs")
    
    # Advanced options
    p.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping (0=disabled)")
    p.add_argument("--no_scheduler", action="store_true", help="Disable LR scheduler")
    p.add_argument("--no_ema", action="store_true", help="Disable model EMA")
    p.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    
    # Debug options
    p.add_argument("--overfit_mode", action="store_true", help="Overfit on small subset for debugging")
    p.add_argument("--overfit_n", type=int, default=32, help="Number of samples for overfit mode")
    p.add_argument("--sanity_run", type=int, default=0, help="Run N iterations only (debug)")
    
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
    """
    Create learning rate scheduler matching paper methodology.
    
    Phase 1 (Warmup): Linear warmup from start_lr to peak_lr over warmup_epochs
    Phase 2 (Cosine): Cosine annealing from peak_lr to near-zero over remaining epochs
    
    Optimizer is initialized with start_lr, scheduler scales it up to peak_lr then down.
    """
    total_steps = total_epochs * iters_per_epoch
    warmup_steps = warmup_epochs * iters_per_epoch
    
    # Scale factors relative to start_lr (optimizer's initial lr)
    peak_factor = float(peak_lr) / float(start_lr)
    min_factor = 1e-6 / float(start_lr)  # Small but non-zero

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup: start_lr -> peak_lr
            progress = float(step) / max(1, warmup_steps)
            return 1.0 + (peak_factor - 1.0) * progress
        else:
            # Cosine annealing: peak_lr -> min_lr
            progress = (float(step) - warmup_steps) / max(1, (total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (peak_factor - min_factor) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_data(x, y, alpha=0.4):
    """
    Apply mixup augmentation.
    Paper mentions mix-up as one of the augmentation techniques.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def build_weighted_sampler(dataset, csv_path):
    """
    Build WeightedRandomSampler to handle class imbalance.
    Uses inverse frequency weighting.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    # Extract labels
    if "label" in df.columns:
        labels = df["label"].astype(int).values
    else:
        available = [c for c in ISIC_CLASSES if c in df.columns]
        onehots = df[available].fillna(0).astype(int).values
        argmax_idx = onehots.argmax(axis=1)
        mapping = {i: ISIC_CLASSES.index(col) for i, col in enumerate(available)}
        labels = np.array([mapping[int(a)] for a in argmax_idx])
    
    # Handle subset datasets
    if hasattr(dataset, "indices"):
        labels = labels[dataset.indices]
    
    # Compute inverse frequency weights
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
            
            # Forward pass
            logits = model(imgs)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            
            # Accuracy (approximate with mixup)
            _, preds = torch.max(logits, 1)
            correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
        else:
            # Standard training
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            # Accuracy
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping if enabled
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None and not args.no_scheduler:
            scheduler.step()
        
        # Update EMA
        if ema is not None and not args.no_ema and epoch >= args.warmup_epochs:
            ema.update(model)
        
        # Statistics
        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': running_loss / total,
            'acc': 100.0 * correct / total,
            'lr': f'{current_lr:.6f}'
        })
        
        global_step[0] += 1
        
        # Sanity run early exit
        if args.sanity_run > 0 and global_step[0] >= args.sanity_run:
            break
    
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
    
    # Compute metrics
    per_class, overall = compute_classwise_metrics(all_targets, all_preds)
    
    return per_class, overall


def print_class_distribution(csv_path):
    """Print class distribution from CSV"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    if "label" in df.columns:
        counts = Counter(df["label"].values.tolist())
        print("Label distribution:")
        for label, count in sorted(counts.items()):
            class_name = ISIC_CLASSES[label] if label < len(ISIC_CLASSES) else f"Class{label}"
            print(f"  {class_name} ({label}): {count}")
    else:
        print("One-hot class distribution:")
        for class_name in ISIC_CLASSES:
            if class_name in df.columns:
                count = int(df[class_name].sum())
                print(f"  {class_name}: {count}")


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seeds(args.seed)
    
    # Print dataset info
    print("\n=== Dataset Information ===")
    print(f"Training set: {args.train_csv}")
    print_class_distribution(args.train_csv)
    
    if args.val_csv:
        print(f"\nValidation set: {args.val_csv}")
        print_class_distribution(args.val_csv)
    
    # Build transforms
    train_transform = build_transforms(
        train=True, 
        input_size=224, 
        deterministic=args.overfit_mode
    )
    val_transform = build_transforms(
        train=False, 
        input_size=224, 
        deterministic=args.overfit_mode
    )
    
    # Build datasets
    if args.overfit_mode:
        print(f"\n=== Overfit Mode: Using {args.overfit_n} samples ===")
        full_ds = ISIC2019Dataset(args.train_csv, args.img_dir, transform=train_transform)
        indices = list(range(min(args.overfit_n, len(full_ds))))
        train_ds = Subset(full_ds, indices)
        val_ds = Subset(full_ds, indices)
    else:
        train_ds = ISIC2019Dataset(args.train_csv, args.img_dir, transform=train_transform)
        if args.val_csv:
            val_ds = ISIC2019Dataset(args.val_csv, args.img_dir, transform=val_transform)
        else:
            raise ValueError("val_csv required unless in overfit mode")
    
    print(f"\nTrain samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    # Build dataloaders
    if args.overfit_mode:
        train_loader = DataLoader(
            train_ds, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        # Use weighted sampler to handle class imbalance
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
    
    # Build model
    print("\n=== Building Model ===")
    model = HybridConvNeXtV2(num_classes=len(ISIC_CLASSES), pretrained=True)
    model = model.to(device)
    
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,} ({params['total']/1e6:.2f}M)")
    print(f"Trainable parameters: {params['trainable']:,} ({params['trainable']/1e6:.2f}M)")
    
    # Loss function with class weights (paper uses categorical cross-entropy)
    if not args.overfit_mode:
        class_weights = compute_class_weights_from_csv(
            args.train_csv, 
            class_names=ISIC_CLASSES,
            max_weight=10.0
        )
        class_weights = class_weights.to(device)
        print(f"\nClass weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer: SGD with momentum (paper specification)
    # Initialize with start_lr (scheduler will handle warmup to peak_lr)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.start_lr,  # Start at warmup LR
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=False
    )
    
    # Learning rate scheduler
    if not args.no_scheduler and not args.overfit_mode:
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
    
    # Model EMA (paper uses EMA with decay 0.9999)
    if not args.no_ema and not args.overfit_mode:
        ema = ModelEMA(model, decay=args.ema_decay, device=device)
    else:
        ema = None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_f1 = 0.0
    if args.resume:
        print(f"\n=== Resuming from {args.resume} ===")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if ema and 'ema_state' in ckpt:
            ema.load_state_dict(ckpt['ema_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n=== Starting Training ===")
    global_step = [0]
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs-1}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, scheduler, ema, args, global_step
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        eval_model = ema.ema if (ema and epoch >= args.warmup_epochs) else model
        per_class, overall = validate(eval_model, val_loader, device)
        
        # Print results
        print(f"\nValidation Results:")
        print(f"  Accuracy:  {overall['accuracy']*100:.2f}%")
        print(f"  Precision: {overall['precision_macro']*100:.2f}%")
        print(f"  Recall:    {overall['recall_macro']*100:.2f}%")
        print(f"  F1-score:  {overall['macro_f1']*100:.2f}%")
        
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
            
            # Save regular checkpoint
            if epoch % args.save_freq == 0:
                save_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save(ckpt, save_path)
            
            # Save best checkpoint
            if is_best:
                best_path = os.path.join(args.out_dir, "best_model.pt")
                torch.save(ckpt, best_path)
                print(f"  *** New best F1: {best_val_f1*100:.2f}% - Saved to {best_path}")
        
        # Early exit for sanity run
        if args.sanity_run > 0 and global_step[0] >= args.sanity_run:
            print("\nSanity run completed. Exiting.")
            break
    
    print("\n=== Training Complete ===")
    print(f"Best validation F1-score: {best_val_f1*100:.2f}%")
    
    # Final test evaluation if test set provided
    if args.test_csv and not args.overfit_mode:
        print("\n=== Final Test Evaluation ===")
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
        
        print(f"\nTest Results:")
        print(f"  Accuracy:  {overall['accuracy']*100:.2f}%")
        print(f"  Precision: {overall['precision_macro']*100:.2f}%")
        print(f"  Recall:    {overall['recall_macro']*100:.2f}%")
        print(f"  F1-score:  {overall['macro_f1']*100:.2f}%")
        
        print(f"\nPer-class metrics:")
        for class_name in ISIC_CLASSES:
            metrics = per_class[class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']*100:.2f}%")
            print(f"    Recall:    {metrics['recall']*100:.2f}%")
            print(f"    F1-score:  {metrics['f1']*100:.2f}%")
            print(f"    Support:   {metrics['support']}")


if __name__ == "__main__":
    main()