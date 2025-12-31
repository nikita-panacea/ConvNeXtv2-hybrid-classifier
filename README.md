Project: Hybrid ConvNeXtV2 + Separable Self-Attention for ISIC-2019

Structure:
- datasets/isic2019_dataset.py
- models/convnextv2.py        (use the ConvNeXtV2 implementation from the repo)
- models/separable_attention.py
- models/hybrid_model.py
- utils/augment.py
- utils/class_weights.py
- utils/ema.py
- utils/metrics.py
- train.py
- eval.py
- infer.py
- requirements.txt

Usage:
1. Install requirements: pip install -r requirements.txt
2. Prepare ISIC CSV and images. Use the stratified-split script (or your own) to generate train.csv, val.csv, test.csv.
3. Train:
   python train.py --train_csv splits/train.csv --val_csv splits/val.csv --img_dir /path/to/images --epochs 100
4. Evaluate:
   python eval.py --ckpt /path/to/checkpoint.pt --test_csv splits/test.csv --img_dir /path/to/images
5. Inference (CSV predictions):
   python infer.py --ckpt /path/to/checkpoint.pt --test_csv splits/test.csv --img_dir /path/to/images --out predictions.csv

Notes:
- The model loads ImageNet ConvNeXtV2-T pretrained weights (if available) for stem & stage1/2.
- The train schedule uses warmup (default 5 epochs) + cosine annealing.
- EMA (decay 0.9999) is used; validation is performed on EMA weights.
