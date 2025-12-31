# infer.py
import argparse
import torch
from torch.utils.data import DataLoader
from datasets.isic2019_dataset import ISIC2019Dataset, ISIC_CLASSES
from utils.augment import build_transforms
from models.hybrid_model import HybridConvNeXtV2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--img_dir", required=True)
    p.add_argument("--out_csv", default="predictions.csv")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    return p.parse_args()

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = HybridConvNeXtV2(num_classes=len(ISIC_CLASSES), pretrained=False).to(device)
    if "ema_state" in ckpt:
        model.load_state_dict(ckpt["ema_state"], strict=False)
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    tf = build_transforms(train=False, input_size=224)
    ds = ISIC2019Dataset(args.test_csv, args.img_dir, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    rows = []
    with torch.no_grad():
        for imgs, labels, img_ids in tqdm(loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            for i, img_id in enumerate(img_ids):
                row = {
                    "image": str(img_id),
                    "true_label": int(labels[i].item()),
                    "pred_label": int(preds[i]),
                    "pred_prob": float(probs[i, preds[i]])
                }
                # also store full probability vector as comma-separated
                row["probs"] = ",".join([f"{p:.6f}" for p in probs[i].tolist()])
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print("Saved predictions to", args.out_csv)

if __name__ == "__main__":
    main()
