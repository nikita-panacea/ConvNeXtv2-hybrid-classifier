#!/usr/bin/env python3
"""
Robust training script with fixes to reproduce paper results.

Key fixes:
 - optimizer initialized at peak_lr (so scheduler factors are relative to peak_lr)
 - scheduler lr lambda defined relative to peak_lr (warmup start->peak then cosine->0)
 - WeightedRandomSampler used to mitigate severe class imbalance
 - class weights = inverse frequency (capped) used in CrossEntropyLoss
 - explicit classifier head init
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
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=False, help="not used when --overfit_mode")
    p.add_argument("--img_dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--base_lr", type=float, default=0.1)      # kept for compatibility (not used as optimizer lr)
    p.add_argument("--peak_lr", type=float, default=0.01)     # effective peak lr (paper)
    p.add_argument("--start_lr", type=float, default=1e-5)    # warmup start (paper)
    p.add_argument("--weight_decay", type=float, default=2e-5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--mixup_alpha", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_scheduler", action="store_true", help="disable scheduler for debug")
    p.add_argument("--optimizer", choices=["sgd","adamw"], default="sgd")
    p.add_argument("--sanity_run", type=int, default=0, help="run N iterations only then exit (debug)")
    p.add_argument("--overfit_n", type=int, default=0, help="overfit on N samples from train set (debug)")
    p.add_argument("--sanity_check_first_batch", action="store_true", help="run detailed first-batch debug and exit")
    p.add_argument("--overfit_mode", action="store_true", help="run strict overfit protocol (deterministic, same samples for val)")
    p.add_argument("--freeze_backbone", action="store_true", help="freeze backbone parameters (keep head trainable)")
    p.add_argument("--unfreeze_after", type=int, default=-1, help="epoch number to unfreeze whole model (>=0 to enable)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_clip", type=float, default=0.0, help="clip grads by norm if >0")
    return p.parse_args()

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_scheduler(optimizer, total_epochs, iters_per_epoch, warmup_epochs, peak_lr, start_lr):
    """
    Scheduler returns multiplicative factor relative to optimizer.lr (which we set to peak_lr).
    Warmup: start_lr -> peak_lr over warmup_epochs.
    Cosine anneal: peak_lr -> 0 across remainder of total steps.
    """
    total_steps = total_epochs * iters_per_epoch
    warmup_steps = warmup_epochs * iters_per_epoch
    start_factor = float(start_lr) / float(peak_lr)
    peak_factor = 1.0  # optimizer is created with lr=peak_lr

    def lr_lambda(step):
        if step < warmup_steps:
            return start_factor + (peak_factor - start_factor) * (float(step) / max(1, warmup_steps))
        progress = (float(step) - warmup_steps) / max(1, (total_steps - warmup_steps))
        # cosine from 1 -> 0
        return 0.5 * (1.0 + math.cos(math.pi * progress)) * peak_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def show_class_distribution(df_path):
    import pandas as pd
    df = pd.read_csv(df_path)
    if "label" in df.columns:
        cnt = Counter(df["label"].values.tolist())
        print("CSV label counts (label col):", cnt)
    else:
        names = ISIC_CLASSES
        counts = {n:int(df[n].sum()) for n in names if n in df.columns}
        print("CSV one-hot counts:", counts)

def first_batch_debug(imgs, labels, model, criterion, optimizer, device):
    print("=== FIRST BATCH DEBUG ===")
    print("labels unique:", set(labels.cpu().numpy().tolist()))
    imgs = imgs.to(device)
    labels = labels.to(device)
    model.train()
    with torch.no_grad():
        logits = model(imgs)
        print("logits stats mean/std/min/max:", float(logits.mean()), float(logits.std()), float(logits.min()), float(logits.max()))
    logits = model(imgs)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    grads = [(n, (p.grad.norm().item() if p.grad is not None else None)) for n,p in model.named_parameters() if p.requires_grad]
    grads = sorted(grads, key=lambda x: (0 if x[1] is None else -x[1]))[:40]
    print("Top grads:", grads)
    return loss.item()

def reconfigure_optimizer_and_scheduler(model, args, train_loader):
    params = [p for p in model.parameters() if p.requires_grad]
    # Create optimizer with lr = peak_lr (paper effective learning rate)
    if args.optimizer == "sgd":
        optimizer = optim.SGD(params, lr=args.peak_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    else:
        optimizer = optim.AdamW(params, lr=max(1e-5, args.peak_lr * 0.1), weight_decay=args.weight_decay)

    if args.no_scheduler or args.overfit_mode:
        scheduler = None
    else:
        scheduler = get_scheduler(optimizer, total_epochs=args.epochs, iters_per_epoch=len(train_loader),
                                  warmup_epochs=args.warmup_epochs, peak_lr=args.peak_lr, start_lr=args.start_lr)
    return optimizer, scheduler

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scheduler=None, ema=None, args=None, global_step_ref=[0]):
    model.train()
    running_loss = 0.0
    tot = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train E{epoch}")
    for i, (imgs, labels, _) in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        if (i % 200) == 0:
            print(f"  [epoch {epoch} iter {i}] lr={optimizer.param_groups[0]['lr']:.6e}")

        if args.mixup_alpha > 0 and epoch >= args.warmup_epochs:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=args.mixup_alpha)
            logits = model(imgs)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip and args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if scheduler is not None and not args.no_scheduler:
            scheduler.step()

        if ema is not None and epoch >= args.warmup_epochs:
            ema.update(model)

        # train accuracy (mixup-aware)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            if args.mixup_alpha > 0 and epoch >= args.warmup_epochs:
                acc = (lam * (preds == y_a).float() + (1-lam) * (preds == y_b).float()).mean().item()
            else:
                acc = (preds == labels).float().mean().item()

        batch_size = imgs.size(0)
        running_loss += float(loss.item()) * batch_size
        tot += batch_size
        pbar.set_postfix(train_loss=running_loss / tot, train_acc=acc)
        global_step_ref[0] += 1

        if args.sanity_run and global_step_ref[0] >= args.sanity_run:
            print("Sanity-run reached; stopping training loop early.")
            break

    return running_loss / tot if tot > 0 else float('nan')

def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="Val"):
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            if isinstance(labels, torch.Tensor):
                all_targets.extend(labels.cpu().numpy().tolist())
            else:
                all_targets.extend(list(labels))
    per_class, overall = compute_classwise_metrics(all_targets, all_preds)
    return per_class, overall

def freeze_backbone_except_head(model, head_keywords=("head", "classifier")):
    for name, p in model.named_parameters():
        if any(k in name for k in head_keywords):
            p.requires_grad = True
        else:
            p.requires_grad = False

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def build_train_sampler(dataset, csv_path):
    """
    Build a WeightedRandomSampler with sample weights inverse to class frequency.
    Expects dataset.__getitem__ returns (img, label, name)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Extract labels same way as dataset does: attempt 'label' else one-hot mapping
    if "label" in df.columns:
        labels = df["label"].astype(int).values
    else:
        # find available columns in ISIC_CLASSES order
        available = [c for c in ISIC_CLASSES if c in df.columns]
        onehots = df[available].fillna(0).astype(int).values
        argmax_idx = onehots.argmax(axis=1)
        mapping = {i: ISIC_CLASSES.index(col) for i, col in enumerate(available)}
        labels = [mapping[int(a)] for a in argmax_idx]
        labels = np.array(labels, dtype=int)

    # For subset datasets the dataset length may differ: map by dataset indices
    # If dataset is a Subset, extract original indices
    if hasattr(dataset, "indices"):
        labels = np.array(labels)[dataset.indices]
    # class counts
    counts = np.bincount(labels, minlength=len(ISIC_CLASSES)).astype(float) + 1e-6
    inv_freq = 1.0 / counts
    sample_weights = inv_freq[labels]
    sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
    return sampler

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    set_seeds(args.seed)

    print("Train CSV distribution:")
    show_class_distribution(args.train_csv)
    if not args.overfit_mode and args.val_csv:
        print("Val CSV distribution:")
        show_class_distribution(args.val_csv)

    # transforms
    train_tf = build_transforms(train=(not args.overfit_mode), input_size=224, deterministic=args.overfit_mode)
    val_tf   = build_transforms(train=False, input_size=224, deterministic=args.overfit_mode)

    # dataset selection
    if args.overfit_mode:
        full_ds = ISIC2019Dataset(args.train_csv, args.img_dir, transform=train_tf)
        if args.overfit_n and args.overfit_n > 0:
            n = min(args.overfit_n, len(full_ds))
            idxs = list(range(n))
            train_ds = Subset(full_ds, idxs)
            val_ds = Subset(full_ds, idxs)
        else:
            train_ds = full_ds
            val_ds = full_ds
    else:
        train_ds = ISIC2019Dataset(args.train_csv, args.img_dir, transform=train_tf)
        if args.overfit_n and args.overfit_n > 0:
            n = min(args.overfit_n, len(train_ds))
            idxs = list(range(n))
            train_ds = Subset(train_ds, idxs)
        if args.val_csv:
            val_ds   = ISIC2019Dataset(args.val_csv, args.img_dir, transform=val_tf)
        else:
            raise ValueError("val_csv must be provided unless --overfit_mode is used")

    # build sampler to mitigate class imbalance (only in normal mode)
    if not args.overfit_mode:
        train_sampler = build_train_sampler(train_ds, args.train_csv)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = HybridConvNeXtV2(num_classes=len(ISIC_CLASSES), pretrained=True).to(device)

    # initialize classification head (stable init)
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        nn.init.kaiming_normal_(model.head.weight, mode="fan_out", nonlinearity="relu")
        if model.head.bias is not None:
            nn.init.zeros_(model.head.bias)

    if args.overfit_mode:
        freeze_backbone_except_head(model, head_keywords=("head", "classifier", "fc"))
        print("Overfit mode: backbone frozen; training head only.")

    if args.freeze_backbone:
        freeze_backbone_except_head(model, head_keywords=("head", "classifier", "norm", "fc"))
        print("Backbone frozen (only head/classifier params trainable).")

    print("Trainable params count (before optimizer):", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # class weights (inverse frequency, capped)
    class_weights = compute_class_weights_from_csv(args.train_csv, class_names=ISIC_CLASSES, device="cpu", max_weight=10.0)
    print("class_weights (cpu):", class_weights.tolist())
    # move to device for loss
    class_weights_device = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=None if args.overfit_mode else class_weights_device)

    optimizer, scheduler = reconfigure_optimizer_and_scheduler(model, args, train_loader)

    ema = None if args.overfit_mode else ModelEMA(model, decay=args.ema_decay, device=device)

    if args.sanity_check_first_batch:
        imgs, labels, _ = next(iter(train_loader))
        first_batch_debug(imgs, labels, model, criterion, optimizer, device)
        return

    best_val_macro_f1 = -1.0
    global_step = [0]

    for epoch in range(args.epochs):
        t0 = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        sched_lr = scheduler.get_last_lr()[0] if scheduler is not None else None
        print(f"Epoch {epoch} starting. optimizer.lr={current_lr:.6e}, scheduler_last_lr={sched_lr}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                     epoch, scheduler=scheduler, ema=ema, args=args, global_step_ref=global_step)

        # Unfreeze if scheduled
        if args.unfreeze_after >= 0 and epoch == args.unfreeze_after:
            print(f"Unfreezing entire model at epoch {epoch}. Recreating optimizer/scheduler.")
            unfreeze_all(model)
            optimizer, scheduler = reconfigure_optimizer_and_scheduler(model, args, train_loader)

        eval_model = ema.ema if (ema is not None and epoch >= args.warmup_epochs) else model
        per_class_val, overall_val = validate(eval_model, val_loader, device)

        epoch_time = time.time() - t0
        acc = overall_val.get("accuracy", 0.0) if overall_val else 0.0
        macro_f1 = overall_val.get("macro_f1", 0.0) if overall_val else 0.0

        if scheduler is not None and not args.no_scheduler:
            try:
                lr_next = scheduler.get_last_lr()[0]
            except Exception:
                lr_next = optimizer.param_groups[0]['lr']
        else:
            lr_next = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch} time {epoch_time:.1f}s - TrainLoss {train_loss:.4f} - ValAcc {acc:.4f} - ValMacroF1 {macro_f1:.4f} - LR {lr_next:.6e}")

        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "ema_state": (ema.state_dict() if ema is not None else None),
                "optimizer": optimizer.state_dict(), "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "args": vars(args)}
        torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pt"))

        if overall_val and overall_val.get("macro_f1", 0.0) > best_val_macro_f1:
            best_val_macro_f1 = overall_val["macro_f1"]
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            if ema is not None and epoch >= args.warmup_epochs:
                torch.save({"epoch": epoch, "ema_state": ema.state_dict()}, os.path.join(args.out_dir, "best_ema.pt"))

        if args.sanity_run and global_step[0] >= args.sanity_run:
            print("Sanity-run finished. Exiting.")
            break

    print("Training finished. Best val macro_f1:", best_val_macro_f1)

if __name__ == "__main__":
    main()
