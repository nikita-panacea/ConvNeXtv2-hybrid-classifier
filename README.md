# Hybrid ConvNeXtV2 + Separable Self-Attention for Skin Lesion Classification

Implementation of the hybrid deep learning framework for multiclass skin cancer classification, combining ConvNeXtV2 blocks with separable self-attention mechanisms.

## Paper Reference

> "A robust deep learning framework for multiclass skin cancer classification"  
> Ozdemir & Pacal, Scientific Reports (2025) 15:4938  
> https://doi.org/10.1038/s41598-025-89230-7

### Paper Results
- **Accuracy**: 93.48%
- **Precision**: 93.24%
- **Recall**: 90.70%
- **F1-Score**: 91.82%
- **Parameters**: 21.92M

## Architecture

The model consists of 4 stages with the following configuration:

| Stage | Type | Blocks | Dimension |
|-------|------|--------|-----------|
| 1 | ConvNeXtV2 | 3 | 96 |
| 2 | ConvNeXtV2 | 3 | 192 |
| 3 | Transformer (Separable Attention) | 9 | 384 |
| 4 | Transformer (Separable Attention) | 12 | 768 |

### Key Components

1. **ConvNeXtV2 Blocks** (Stages 1-2): 
   - Depthwise 7x7 convolution
   - LayerNorm + GELU activation
   - Global Response Normalization (GRN)
   - Pointwise convolutions with 4x expansion
   - Residual connection

2. **Transformer Blocks** (Stages 3-4):
   - LayerNorm → Separable Self-Attention → Residual
   - LayerNorm → MLP → Residual
   - Configurable MLP expansion ratio

3. **Separable Self-Attention** (from MobileViT-v2):
   - O(n) complexity instead of O(n²)
   - Uses latent token for global context aggregation

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── models/
│   ├── __init__.py
│   ├── hybrid_model.py          # Main hybrid model implementation
│   ├── separable_attention.py   # Separable attention variants
│   ├── convnextv2.py           # ConvNeXtV2 backbone
│   ├── hybrid_blocks.py        # Building blocks
│   └── utils.py                # LayerNorm, GRN utilities
├── datasets/
│   ├── isic2019_dataset.py     # ISIC 2019 dataset loader
│   └── train-val-csv-split.py  # Data splitting utility
├── utils/
│   ├── augment.py              # Data augmentation
│   ├── class_weights.py        # Class weight computation
│   ├── ema.py                  # Exponential moving average
│   └── metrics.py              # Evaluation metrics
├── experiments/
│   ├── run_param_experiments.py         # Quick parameter exploration
│   └── comprehensive_param_search.py    # Full parameter search
├── train.py                    # Training script
├── eval.py                     # Evaluation script
├── infer.py                    # Inference script
├── verify_params.py            # Parameter verification
└── debug_params.py             # Detailed parameter analysis
```

## Usage

### 1. Prepare Data

The ISIC 2019 dataset should be organized with a CSV file containing image IDs and labels:

```csv
image,label
ISIC_0000000,1
ISIC_0000001,0
...
```

Or with one-hot encoded columns:
```csv
image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC
ISIC_0000000,0,1,0,0,0,0,0,0
...
```

### 2. Run Parameter Experiments

Before training, find the best configuration matching the paper's parameters:

```bash
# Quick parameter exploration
python experiments/run_param_experiments.py

# Detailed exploration with all combinations
python experiments/comprehensive_param_search.py
```

### 3. Training

```bash
# Default configuration (closest to paper's 21.92M params)
python train.py \
    --train_csv splits/train.csv \
    --val_csv splits/val.csv \
    --img_dir /path/to/images \
    --epochs 100

# Custom configuration
python train.py \
    --train_csv splits/train.csv \
    --val_csv splits/val.csv \
    --img_dir /path/to/images \
    --backbone tiny \
    --attn_type minimal \
    --mlp_ratio 1.125 \
    --pretrained_weights ft_1k \
    --epochs 100
```

### 4. Evaluation

```bash
python eval.py \
    --ckpt checkpoints/best.pt \
    --test_csv splits/test.csv \
    --img_dir /path/to/images
```

### 5. Inference

```bash
python infer.py \
    --ckpt checkpoints/best.pt \
    --test_csv splits/test.csv \
    --img_dir /path/to/images \
    --out predictions.csv
```

## Model Configuration Options

### Backbone Variants

| Variant | Depths | Dimensions | Full Params |
|---------|--------|------------|-------------|
| atto | [2,2,6,2] | [40,80,160,320] | 3.7M |
| femto | [2,2,6,2] | [48,96,192,384] | 5.2M |
| pico | [2,2,6,2] | [64,128,256,512] | 9.1M |
| nano | [2,2,8,2] | [80,160,320,640] | 15.6M |
| tiny | [3,3,9,3] | [96,192,384,768] | 28.6M |
| base | [3,3,27,3] | [128,256,512,1024] | 89M |

### Attention Types

| Type | Description | Parameters |
|------|-------------|------------|
| `minimal` | Channel-wise attention with O(d) params | Very few |
| `paper` | Full implementation as per paper equations (5)-(10) | O(d²) |
| `reduced` | Reduced projection dimension | O(d×d_proj) |
| `mobilevit` | MobileViT-v2 style attention | O(d²) |

### Pretrained Weights

| Type | Description |
|------|-------------|
| `ft_1k` | ImageNet-1K fine-tuned weights |
| `ft_22k` | ImageNet-22K fine-tuned weights (nano/tiny/base only) |
| `fcmae_1k` | ImageNet-1K FCMAE self-supervised weights |
| `none` | No pretrained weights |

## Training Configuration (from Paper)

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Weight decay | 2.0 × 10⁻⁵ |
| Warmup epochs | 5 |
| Warmup LR | 1.0 × 10⁻⁵ |
| Loss | Cross-Entropy |
| Input size | 224 × 224 |
| EMA decay | 0.9999 |

## Data Augmentation

The following augmentations are applied during training:
- Random resized crop (scale: 0.6-1.0)
- Horizontal and vertical flip
- Random rotation (±20°)
- Gaussian smoothing
- Color jitter
- Random erasing

## Parameter Matching Analysis

To achieve the paper's ~21.92M parameter target, the following configurations are recommended:

| Configuration | Params | Difference |
|---------------|--------|------------|
| tiny/minimal/mlp=1.125 | ~21.9M | ~0.02M |
| tiny/minimal/mlp=1.0 | ~20.5M | ~1.4M |
| nano/minimal/mlp=2.0 | ~22.1M | ~0.2M |

Run `python experiments/run_param_experiments.py` for complete analysis.

## Verify Installation

```bash
# Check parameter counts
python verify_params.py --all

# Detailed parameter analysis
python debug_params.py --compare
```

## Notes

- The model uses ImageNet pretrained weights for ConvNeXtV2 stages 1-2
- Transformer stages (3-4) are initialized from scratch
- EMA (decay 0.9999) is used; validation uses EMA weights
- Training uses warmup + cosine annealing schedule
- Class weights are computed from training set to handle imbalance

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{ozdemir2025robust,
  title={A robust deep learning framework for multiclass skin cancer classification},
  author={Ozdemir, Burhanettin and Pacal, Ishak},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={4938},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
