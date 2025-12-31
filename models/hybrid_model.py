# models/hybrid_model.py
import torch
import torch.nn as nn
from models.separable_attention import SeparableSelfAttention

# Import ConvNeXtV2 from repo-provided file.
# Ensure models/convnextv2.py is on PYTHONPATH (same package).
from models.convnextv2 import ConvNeXtV2

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SeparableSelfAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )

    def forward(self, x):
        # x: [B, N, C]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class HybridConvNeXtV2(nn.Module):
    """
    Hybrid model:
    - Use ConvNeXtV2 stem + stage1 + stage2
    - Replace stage3 & stage4 with separable-attention TransformerBlocks
    Depths: [3,3,9,12] target per paper
    dims: [96,192,384,768] typical
    """

    def __init__(self, num_classes=8, pretrained=True, convnext_name="convnextv2_tiny"):
        super().__init__()
        # build base convnextv2 with desired depths/dims via constructor if needed
        # Use the ConvNeXtV2 class implementation from convnextv2.py
        base = ConvNeXtV2(in_chans=3, num_classes=1000,
                          depths=[3,3,9,12], dims=[96,192,384,768], drop_path_rate=0.0)

        # Load pretrained state for matching layers if requested
        if pretrained:
            try:
                state = torch.hub.load_state_dict_from_url(
                    "https://dl.fbaipublicfiles.com/convnext/convnextv2_tiny_1k_224.pth",
                    map_location="cpu"
                )
                base.load_state_dict(state.get("model", state), strict=False)
                print("Loaded pretrained ConvNeXtV2-Tiny weights (partial load for stem/stage1/stage2).")
            except Exception as e:
                print("Pretrained load failed:", e)

        # reuse base downsample layers and stage blocks 0 and 1
        self.downsample_layers = base.downsample_layers  # list of 4 modules (stem + 3 downsamples)
        # reuse stage1, stage2
        self.stage1 = base.stages[0]
        self.stage2 = base.stages[1]
        # For stage3 and stage4, build transformer blocks with separable attention
        self.stage3_down = base.downsample_layers[2]  # conv to downsample dims[1]->dims[2]
        self.stage4_down = base.downsample_layers[3]  # conv to downsample dims[2]->dims[3]

        # stage3: tokens dimension dims[2] (384)
        self.stage3 = nn.Sequential(*[TransformerBlock(384) for _ in range(9)])
        # stage4: tokens dimension dims[3] (768)
        self.stage4 = nn.Sequential(*[TransformerBlock(768) for _ in range(12)])

        self.final_norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        # stage 0 (stem)
        x = self.downsample_layers[0](x)  # stem -> [B, C0, H/4, W/4]
        x = self.stage1(x)                # [B, C0, H/4, W/4]
        x = self.downsample_layers[1](x)  # downsample -> [B, C1, H/8, W/8]
        x = self.stage2(x)                # [B, C1, H/8, W/8]

        # stage3
        x = self.stage3_down(x)           # [B, C2, H/16, W/16]
        B, C, H, W = x.shape
        x_tokens = x.flatten(2).transpose(1,2)   # [B, N, C] where N=H*W
        x_tokens = self.stage3(x_tokens)        # [B, N, C]
        x = x_tokens.transpose(1,2).view(B,C,H,W)

        # stage4
        x = self.stage4_down(x)           # [B, C3, H/32, W/32]
        B, C, H, W = x.shape
        x_tokens = x.flatten(2).transpose(1,2)   # [B, N, C]
        x_tokens = self.stage4(x_tokens)        # [B, N, C]

        # global pooling
        x = x_tokens.mean(dim=1)          # [B, C]
        x = self.final_norm(x)
        logits = self.head(x)
        return logits
