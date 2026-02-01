# train.py
"""
Training script for Hybrid ConvNeXtV2 + Separable Attention model.

Paper Training Configuration:
- Optimizer: SGD
- Learning rate: 0.01
- Base learning rate: 0.1
- Momentum: 0.9
- Weight decay: 2.0 × 10^-5
- Warmup epochs: 5 (starting LR: 1.0 × 10^-5)
- Loss: Categorical Cross-Entropy
- Input size: 224 × 224
- Data augmentation: rotation, flipping, scaling, smoothing, mix-up, color jitter
- Transfer learning: ImageNet pretrained weights

Usage:
    python train.py --train_csv data/train.csv --val_csv data/val.csv --img_dir data/images

    # With custom model configuration:
    python train.py --train_csv data/train.csv --val_csv data/val.csv --img_dir data/images \
        --backbone tiny --attn_type minimal --mlp_ratio 1.125 --pretrained_weights ft_1k
"""

import os
import math
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.isic2019_dataset import ISIC2019Dataset, ISIC_CLASSES
from utils.augment import build_transforms, MixupAugmentation, mixup_criterion
from utils.class_weights import compute_class_weights_from_csv
from utils.ema import ModelEMA
from utils.metrics import compute_classwise_metrics
from models.hybrid_model import HybridConvNeXtV2, print_model_summary
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Train Hybrid ConvNeXtV2 model for skin lesion classification")
    
    # Data paths
    p.add_argument("--train_csv", required=True, help="Path to training CSV file")
    p.add_argument("--val_csv", required=True, help="Path to validation CSV file")
    p.add_argument("--img_dir", required=True, help="Path to image directory")
    
    # Training hyperparameters (from paper)
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate (paper: 0.01)")
    p.add_argument("--weight_decay", type=float, default=2e-5, help="Weight decay (paper: 2e-5)")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (paper: 0.9)")
    p.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs (paper: 5)")
    p.add_argument("--warmup_lr", type=float, default=1e-5, help="Warmup starting LR (paper: 1e-5)")
    
    # Model configuration
    p.add_argument("--backbone", type=str, default="tiny",
                   choices=["atto", "femto", "pico", "nano", "tiny", "base"],
                   help="ConvNeXtV2 backbone variant")
    p.add_argument("--attn_type", type=str, default="minimal",
                   choices=["minimal", "paper", "reduced", "mobilevit"],
                   help="Separable attention type")
    p.add_argument("--mlp_ratio", type=float, default=1.125,
                   help="MLP expansion ratio in Transformer blocks")
    p.add_argument("--attn_proj_ratio", type=float, default=0.5,
                   help="Projection ratio for reduced attention")
    p.add_argument("--stage3_blocks", type=int, default=9,
                   help="Number of Transformer blocks in stage 3 (paper: 9)")
    p.add_argument("--stage4_blocks", type=int, default=12,
                   help="Number of Transformer blocks in stage 4 (paper: 12)")
    p.add_argument("--pretrained_weights", type=str, default="ft_1k",
                   choices=["ft_1k", "ft_22k", "fcmae_1k", "none"],
                   help="Pretrained weights type")
    p.add_argument("--drop_path", type=float, default=0.0,
                   help="Stochastic depth rate")
    
    # Training options
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--out_dir", type=str, default="checkpoints", help="Output directory")
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    p.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    p.add_argument("--use_class_weights", action="store_true", default=True,
                   help="Use class weights for imbalanced data")
    p.add_argument("--input_size", type=int, default=224, help="Input image size")
    p.add_argument("--mixup_alpha", type=float, default=0.4,
                   help="Mixup alpha (0.0 disables mixup; paper mentions mix-up)")
    p.add_argument("--debug", action="store_true", help="Enable debug output")
    
    return p.parse_args()


def get_scheduler(optimizer, total_epochs, iters_per_epoch, warmup_epochs):
    """
    Create learning rate scheduler with warmup and cosine annealing.
    
    Paper: 5 warmup epochs starting from 1e-5, then cosine decay.
    """
    total_steps = total_epochs * iters_per_epoch
    warmup_steps = warmup_epochs * iters_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step + 1) / float(max(1, warmup_steps))
        # Cosine annealing
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    scheduler=None, ema=None, debug=False, mixup_fn=None):
    """Train for one epoch."""
    model.train()
    
    losses = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train E{epoch}")
    
    for i, (imgs, labels, _) in pbar:
        # Debug first batch
        if debug and epoch == 0 and i == 0:
            _debug_first_batch(model, optimizer, criterion, imgs, labels, device)
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Optional mixup
        if mixup_fn is not None:
            imgs, labels_a, labels_b, lam = mixup_fn(imgs, labels)
            logits = model(imgs)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
        else:
            # Forward pass
            logits = model(imgs)
            loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        losses += loss.item()
        pbar.set_postfix(loss=losses/(i+1), lr=optimizer.param_groups[0]['lr'])
    
    return losses / len(loader)


def _debug_first_batch(model, optimizer, criterion, imgs, labels, device):
    """Debug output for first batch."""
    lr = optimizer.param_groups[0]['lr']
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n" + "=" * 60)
    print("DEBUG: First batch information")
    print("=" * 60)
    print(f"  Learning rate: {lr}")
    print(f"  Trainable params: {total_trainable:,} / {total_params:,}")
    print(f"  Unique labels in batch: {set(labels.cpu().numpy().tolist())}")
    
    # Forward pass stats
    imgs = imgs.to(device)
    with torch.no_grad():
        logits = model(imgs)
        print(f"  Logits - mean: {logits.mean():.4f}, std: {logits.std():.4f}, "
              f"min: {logits.min():.4f}, max: {logits.max():.4f}")
    print("=" * 60 + "\n")


def validate(model, loader, device):
    """Validate model on validation set."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Val"):
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())
    
    per_class, overall = compute_classwise_metrics(all_targets, all_preds)
    return per_class, overall


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build transforms
    train_tf = build_transforms(train=True, input_size=args.input_size)
    val_tf = build_transforms(train=False, input_size=args.input_size)
    
    # Create datasets
    train_ds = ISIC2019Dataset(args.train_csv, args.img_dir, transform=train_tf)
    val_ds = ISIC2019Dataset(args.val_csv, args.img_dir, transform=val_tf)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
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
    
    # Create model
    pretrained = args.pretrained_weights != "none"
    model = HybridConvNeXtV2(
        backbone_variant=args.backbone,
        attn_type=args.attn_type,
        mlp_ratio=args.mlp_ratio,
        attn_proj_ratio=args.attn_proj_ratio,
        stage3_blocks=args.stage3_blocks,
        stage4_blocks=args.stage4_blocks,
        num_classes=len(ISIC_CLASSES),
        drop_path_rate=args.drop_path,
        pretrained=pretrained,
        pretrained_weights=args.pretrained_weights if pretrained else "ft_1k",
    ).to(device)
    
    # Print model summary
    print_model_summary(model)
    
    # Class weights for imbalanced data
    if args.use_class_weights:
        class_weights = compute_class_weights_from_csv(args.train_csv, device=device)
        print(f"Class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer (SGD as per paper)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = optim.SGD(
        trainable_params, 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = get_scheduler(
        optimizer, 
        total_epochs=args.epochs,
        iters_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs
    )
    
    # EMA (Exponential Moving Average)
    ema = ModelEMA(model, decay=args.ema_decay, device=device)
    
    # Save training configuration
    config = {
        'args': vars(args),
        'model_config': model.get_config(),
        'num_train_samples': len(train_ds),
        'num_val_samples': len(val_ds),
    }
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        import json
        json.dump(config, f, indent=2)
    
    # Training loop
    best_val_macro_f1 = 0.0
    best_val_accuracy = 0.0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    mixup_fn = MixupAugmentation(alpha=args.mixup_alpha) if args.mixup_alpha > 0 else None

    for epoch in range(args.epochs):
        t0 = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, 
            device, epoch, scheduler, ema, debug=args.debug, mixup_fn=mixup_fn
        )
        
        # Validate (using EMA model for better results)
        per_class_val, overall_val = validate(ema.ema, val_loader, device)
        
        epoch_time = time.time() - t0
        
        # Log results
        print(f"Epoch {epoch:3d} | "
              f"Time {epoch_time:.1f}s | "
              f"Loss {train_loss:.4f} | "
              f"Acc {overall_val['accuracy']:.4f} | "
              f"Prec {overall_val['macro_precision']:.4f} | "
              f"Rec {overall_val['macro_recall']:.4f} | "
              f"F1 {overall_val['macro_f1']:.4f}")
        
        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": overall_val,
            "config": config,
        }
        
        # Save latest checkpoint
        torch.save(ckpt, os.path.join(args.out_dir, "latest.pt"))
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, os.path.join(args.out_dir, f"epoch_{epoch}.pt"))
        
        # Save best by macro F1
        if overall_val['macro_f1'] > best_val_macro_f1:
            best_val_macro_f1 = overall_val['macro_f1']
            best_val_accuracy = overall_val['accuracy']
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> New best model saved (F1: {best_val_macro_f1:.4f})")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Best Validation Macro F1: {best_val_macro_f1:.4f}")
    print(f"Checkpoints saved to: {args.out_dir}")
    
    # Compare to paper target
    print("\nPaper target:")
    print("  Accuracy: 0.9348")
    print("  Precision: 0.9324")
    print("  Recall: 0.9070")
    print("  F1-Score: 0.9182")


if __name__ == "__main__":
    main()
