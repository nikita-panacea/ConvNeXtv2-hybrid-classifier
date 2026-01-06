#!/usr/bin/env bash
# run_train.sh — runs train.py with paper-exact hyperparameters
# Edit the TRAIN_CSV / VAL_CSV / IMG_DIR variables below before running.
set -euo pipefail

### === user-editable paths & environment ===
CONDA_ENV="convnextv2"                       # change if needed (optional)
TRAIN_CSV="/home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/train.csv"      # REQUIRED: path to training CSV
VAL_CSV="/home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/val.csv"          # REQUIRED: path to validation CSV
IMG_DIR="/home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/Images"                    # REQUIRED: folder containing images
OUT_DIR="checkpoints-jan6"                        # where checkpoints/logs will be written

### === fixed paper hyperparameters (no ambiguity) ===
BATCH_SIZE=32
EPOCHS=100
PEAK_LR=0.01
START_LR=1e-5
WARMUP_EPOCHS=5
OPTIMIZER="sgd"
MOMENTUM=0.9
WEIGHT_DECAY=2e-5
MIXUP_ALPHA=0.0
NUM_WORKERS=8
DEVICE="cuda"
SEED=42
GRAD_CLIP=0.0

### === environment tuning (recommended) ===
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${OUT_DIR}"

### activate conda env if available (optional)
if command -v conda >/dev/null 2>&1; then
  # make sure conda is initialized for this shell
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  if conda info --envs | grep -q "^${CONDA_ENV}[[:space:]]"; then
    echo "Activating conda env: ${CONDA_ENV}"
    conda activate "${CONDA_ENV}"
  else
    echo "Conda env '${CONDA_ENV}' not found — continuing without activation."
  fi
fi

### logfile
LOGFILE="${OUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${LOGFILE}"
echo "Command-line: train.py --train_csv ${TRAIN_CSV} --val_csv ${VAL_CSV} --img_dir ${IMG_DIR}" | tee "${LOGFILE}"

### run training (stream to console and logfile)
python -u train.py \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --img_dir "${IMG_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --peak_lr "${PEAK_LR}" \
  --start_lr "${START_LR}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --optimizer "${OPTIMIZER}" \
  --momentum "${MOMENTUM}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --mixup_alpha "${MIXUP_ALPHA}" \
  --num_workers "${NUM_WORKERS}" \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --grad_clip "${GRAD_CLIP}" 2>&1 | tee -a "${LOGFILE}"

echo "Training finished. Check ${LOGFILE} and ${OUT_DIR} for checkpoints."


# python train.py \
#     --train_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/train.csv \
#     --val_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/val.csv \
#     --img_dir /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/Images \
#     --mixup_alpha 0.4 \
#     --ema_decay 0.9999 \
#     --num_workers 8 \
#     --batch_size 32 \
#     --epochs 100 \
#     --peak_lr 0.01 \
#     --start_lr 1e-5 \
#     --warmup_epochs 5 \
#     --optimizer sgd \
#     --momentum 0.9 \
#     --weight_decay 2e-5 \
#     --out_dir checkpts-6jan
