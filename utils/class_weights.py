# utils/class_weights.py
import numpy as np
import pandas as pd
import torch

ISIC_CLASSES = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"]

def compute_class_weights_from_csv(csv_path, eps=1e-6, device="cpu"):
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        labels = df["label"].values.astype(int)
        counts = np.bincount(labels, minlength=len(ISIC_CLASSES))
    else:
        counts = df[ISIC_CLASSES].sum(axis=0).values.astype(float)

    # inverse frequency
    weights = counts.sum() / (counts + eps)
    # normalize mean to 1
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)
