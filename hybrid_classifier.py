"""
Legacy standalone implementation of the hybrid ConvNeXtV2 + Separable Attention model.

NOTE:
- This file is NOT used by the main training pipeline.
- Prefer using `models/hybrid_model.py` and `train.py`.
- The training code below is wrapped in `main()` to avoid running on import.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Normalization layer + Global Response Normalization (GRN)
class LayerNorm2d(nn.Module):
    """
    Channel-last LayerNorm for ConvNeXt-style blocks
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B,C,H,W) -> (B,H,W,C)
        x = x.permute(0, 2, 3, 1)
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight * x + self.bias
        return x.permute(0, 3, 1, 2)

class GRN(nn.Module):
    """
    Global Response Normalization (Eq. 3 in paper)
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma.view(1,-1,1,1) * (x * nx) + self.beta.view(1,-1,1,1) + x

# ConvNeXtv2 Blocks (Stage 1 and 2)
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return x + residual

# Separable Self-Attention (Stages 3–4)
class SeparableSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.WI = nn.Linear(dim, 1, bias=False)
        self.WK = nn.Linear(dim, dim, bias=False)
        self.WV = nn.Linear(dim, dim, bias=False)
        self.WO = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: (B, N, D)
        cs = torch.softmax(self.WI(x), dim=1)          # (B,N,1)
        k = self.WK(x)
        cv = torch.sum(cs * k, dim=1)                  # (B,D)
        v = F.relu(self.WV(x))
        z = v * cv.unsqueeze(1)
        return self.WO(z)
class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SeparableSelfAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# Hybrid ConvNeXtV2 + Separable Attention backbone classifier
class ConvNeXtV2_SepAttn(nn.Module):
    def __init__(self, num_classes=8, dims=(96,192,384,768),
                 depths=(3,3,9,12)):
        super().__init__()

        self.stem = nn.Conv2d(3, dims[0], kernel_size=4, stride=4)

        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()

        for i in range(4):
            if i > 0:
                self.downsamples.append(
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
                )

            if i < 2:
                self.stages.append(
                    nn.Sequential(*[
                        ConvNeXtV2Block(dims[i]) for _ in range(depths[i])
                    ])
                )
            else:
                self.stages.append(
                    nn.Sequential(*[
                        TransformerBlock(dims[i]) for _ in range(depths[i])
                    ])
                )

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for i in range(4):
            if i > 0:
                x = self.downsamples[i-1](x)

            if i < 2:
                x = self.stages[i](x)
            else:
                B,C,H,W = x.shape
                x = x.flatten(2).transpose(1,2)  # (B,N,C)
                x = self.stages[i](x)
                x = x.transpose(1,2).reshape(B,C,H,W)

        x = x.mean(dim=[2,3])
        x = self.norm(x)
        return self.head(x)

def main():
    # Dataset + training pipeline (legacy example)
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    train_tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.ToTensor()
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    train_ds = ImageFolder("ISIC2019/train", transform=train_tfms)
    val_ds   = ImageFolder("ISIC2019/val", transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Training loop (legacy)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvNeXtV2_SepAttn(num_classes=8).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=2e-5
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        print(f"Epoch {epoch}: Val Acc = {correct/total:.4f}")

    # parameter count
    print("Params (M):", sum(p.numel() for p in model.parameters()) / 1e6)


if __name__ == "__main__":
    main()
