# eval.py
import argparse
import torch
from torch.utils.data import DataLoader
from datasets.isic2019_dataset import ISIC2019Dataset, ISIC_CLASSES
from utils.augment import build_transforms
from utils.metrics import compute_classwise_metrics
from models.hybrid_model import HybridConvNeXtV2
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--img_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

def _build_model_from_ckpt(ckpt, device):
    model_cfg = None
    if isinstance(ckpt, dict):
        model_cfg = ckpt.get("config", {}).get("model_config")
    if model_cfg:
        model = HybridConvNeXtV2(
            backbone_variant=model_cfg.get("backbone_variant", "tiny"),
            attn_type=model_cfg.get("attn_type", "minimal"),
            mlp_ratio=model_cfg.get("mlp_ratio", 1.125),
            stage3_blocks=model_cfg.get("stage3_blocks", 9),
            stage4_blocks=model_cfg.get("stage4_blocks", 12),
            num_classes=model_cfg.get("num_classes", len(ISIC_CLASSES)),
            pretrained=False,
        ).to(device)
    else:
        model = HybridConvNeXtV2(num_classes=len(ISIC_CLASSES), pretrained=False).to(device)
    return model

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = _build_model_from_ckpt(ckpt, device)
    # prefer EMA state if present
    if "ema_state" in ckpt:
        model_state = ckpt["ema_state"]
    elif "model_state" in ckpt:
        model_state = ckpt["model_state"]
    else:
        model_state = ckpt
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    tf = build_transforms(train=False, input_size=224)
    ds = ISIC2019Dataset(args.test_csv, args.img_dir, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())

    per_class, overall = compute_classwise_metrics(all_targets, all_preds)
    print("Overall:", overall)
    print("Per-class:")
    import json
    print(json.dumps(per_class, indent=2))

    # store results CSV summary
    pd.DataFrame.from_dict(per_class, orient='index').to_csv("per_class_metrics.csv")
    print("Saved per_class_metrics.csv")

if __name__ == "__main__":
    main()
