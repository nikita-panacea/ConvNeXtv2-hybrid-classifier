# train.py
import os
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.isic2019_dataset import ISIC2019Dataset, ISIC_CLASSES
from utils.augment import build_transforms
from utils.class_weights import compute_class_weights_from_csv
from utils.ema import ModelEMA
from utils.metrics import compute_classwise_metrics
from models.hybrid_model import HybridConvNeXtV2
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--img_dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=2e-5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

def get_scheduler(optimizer, total_epochs, iters_per_epoch, warmup_epochs):
    total_steps = total_epochs * iters_per_epoch
    warmup_steps = warmup_epochs * iters_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # cosine anneal
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scheduler=None, ema=None):
    model.train()
    losses = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train E{epoch}")
    for i, (imgs, labels, _) in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        losses += loss.item()
        pbar.set_postfix(loss=losses/(i+1))
    return losses / len(loader)

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
            all_targets.extend(labels.numpy().tolist())
    per_class, overall = compute_classwise_metrics(all_targets, all_preds)
    return per_class, overall

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_tf = build_transforms(train=True, input_size=224)
    val_tf   = build_transforms(train=False, input_size=224)

    train_ds = ISIC2019Dataset(args.train_csv, args.img_dir, transform=train_tf)
    val_ds   = ISIC2019Dataset(args.val_csv, args.img_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = HybridConvNeXtV2(num_classes=len(ISIC_CLASSES), pretrained=True).to(device)

    # class weights
    class_weights = compute_class_weights_from_csv(args.train_csv, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # scheduler: warmup + cosine via LambdaLR
    scheduler = get_scheduler(optimizer, total_epochs=args.epochs,
                              iters_per_epoch=len(train_loader),
                              warmup_epochs=args.warmup_epochs)

    # EMA
    ema = ModelEMA(model, decay=0.9999, device=device)

    best_val_macro_f1 = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scheduler, ema)
        per_class_val, overall_val = validate(ema.ema, val_loader, device)  # evaluate EMA model
        epoch_time = time.time() - t0

        print(f"Epoch {epoch} time {epoch_time:.1f}s - TrainLoss {train_loss:.4f} - ValAcc {overall_val['accuracy']:.4f} - ValMacroF1 {overall_val['macro_f1']:.4f}")

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args)
        }
        torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pt"))

        # save best by macro_f1
        if overall_val['macro_f1'] > best_val_macro_f1:
            best_val_macro_f1 = overall_val['macro_f1']
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

    print("Training finished. Best val macro_f1:", best_val_macro_f1)

if __name__ == "__main__":
    main()
