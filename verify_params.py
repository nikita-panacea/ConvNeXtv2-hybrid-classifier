# verify_params.py
import torch
from models.hybrid_model import HybridConvNeXtV2
model = HybridConvNeXtV2(num_classes=8, pretrained=True)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {n/1e6:.3f}M")
