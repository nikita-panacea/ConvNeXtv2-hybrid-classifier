# utils/class_weights.py
import numpy as np
import pandas as pd
import torch

def compute_class_weights_from_csv(csv_path, class_names=None, eps=1e-6, device="cpu", max_weight=10.0):
    """
    Returns torch.tensor of class weights length = number of classes.
    - Avoids infinite/huge weights for zero-count classes by using smoothing and capping.
    - Normalizes weights to mean=1 for stable loss scale.
    """
    df = pd.read_csv(csv_path)

    # decide counts
    if "label" in df.columns and class_names is None:
        labels = df["label"].astype(int).values
        n_classes = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=n_classes).astype(float)
    else:
        if class_names is None:
            exclude = {"image", "image_id", "filename", "label_name"}
            cand = [c for c in df.columns if c not in exclude]
            cols = []
            for c in cand:
                if pd.api.types.is_numeric_dtype(df[c]):
                    vals = df[c].dropna().unique()
                    if set(np.unique(vals)).issubset({0, 1}):
                        cols.append(c)
            if not cols:
                raise ValueError("Could not infer class columns; provide class_names.")
        else:
            cols = list(class_names)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"Provided class_names missing from CSV: {missing}")
        counts = df[cols].sum(axis=0).astype(float).values

    # smoothing to avoid division by zero
    counts_safe = counts + eps

    # inverse-frequency
    freqs = counts_safe / counts_safe.sum()
    weights = 1.0 / (freqs + eps)

    # cap extreme weights to avoid instability
    weights = np.minimum(weights, max_weight)

    # normalize to mean=1 (keeps overall loss scale sane)
    weights = weights.astype(float)
    weights = weights / float(weights.mean())

    w = torch.tensor(weights, dtype=torch.float32, device=device)
    if not torch.isfinite(w).all():
        raise RuntimeError("Non-finite class weights computed.")
    return w
