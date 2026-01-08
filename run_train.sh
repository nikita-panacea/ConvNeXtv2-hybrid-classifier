python train.py \
    --train_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/train.csv \
    --val_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/val.csv \
    --test_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/test.csv \
    --img_dir /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/Images \
    --epochs 100 \
    --batch_size 32 \
    --out_dir checkpoints-ISIC2019-7Jan/

# python train.py \
#  (convnextv2) ubuntu@panacea-ml-learn:~/Documents/Nikita/ConvNeXtv2-hybrid-classifier$ python count_params.py
# ============================================================
# DETAILED PARAMETER BREAKDOWN
# ============================================================

# Component-wise breakdown:
# ------------------------------------------------------------
# stem           :        4,896 params ( 0.02% of target)
# stage1         :      239,904 params ( 1.09% of target)
# down1          :       74,112 params ( 0.34% of target)
# stage2         :      922,176 params ( 4.21% of target)
# down2          :      295,680 params ( 1.35% of target)
# stage3         :    5,330,880 params (24.32% of target)
# down3          :    1,181,184 params ( 5.39% of target)
# stage4         :   28,371,456 params (129.43% of target)
# norm           :        1,536 params ( 0.01% of target)
# head           :        6,152 params ( 0.03% of target)
# ------------------------------------------------------------
# TOTAL          :   36,427,976 params
# TARGET         :   21,920,000 params
# DIFFERENCE     :   14,507,976 params (+66.19%)

# ============================================================
# ATTENTION STAGE BREAKDOWN
# ============================================================

# Stage 3 (9 blocks, dim=384):
#   Per block: 592,320 params
#   Total: 5,330,880 params

#   Single block breakdown:
#     norm1               : 768 params
#     attn                : 442,752 params
#     norm2               : 768 params
#     mlp                 : 148,032 params

# Stage 4 (12 blocks, dim=768):
#   Per block: 2,364,288 params
#   Total: 28,371,456 params

#   Single block breakdown:
#     norm1               : 1,536 params
#     attn                : 1,770,240 params
#     norm2               : 1,536 params
#     mlp                 : 590,976 params

# ============================================================
# SEPARABLE ATTENTION ANALYSIS
# ============================================================

# Stage 3 attention (dim=384):
#   context_score.weight          : [1, 384] = 384 params
#   key_proj.weight               : [384, 384] = 147,456 params
#   value_proj.weight             : [384, 384] = 147,456 params
#   out_proj.weight               : [384, 384] = 147,456 params
#   TOTAL                         : 442,752 params

# Stage 4 attention (dim=768):
#   context_score.weight          : [1, 768] = 768 params
#   key_proj.weight               : [768, 768] = 589,824 params
#   value_proj.weight             : [768, 768] = 589,824 params
#   out_proj.weight               : [768, 768] = 589,824 params
#   TOTAL                         : 1,770,240 params

# ============================================================
# MLP ANALYSIS
# ============================================================

# Stage 3 MLP (dim=384):
#   0.weight                      : [192, 384] = 73,728 params
#   0.bias                        : [192] = 192 params
#   2.weight                      : [384, 192] = 73,728 params
#   2.bias                        : [384] = 384 params
#   TOTAL                         : 148,032 params

# Stage 4 MLP (dim=768):
#   0.weight                      : [384, 768] = 294,912 params
#   0.bias                        : [384] = 384 params
#   2.weight                      : [768, 384] = 294,912 params
#   2.bias                        : [768] = 768 params
#   TOTAL                         : 590,976 params

# ============================================================
# TARGET PARAMETER ESTIMATION
# ============================================================

# Estimated parameters:
#   ConvNeXt stages 1-2: 4,000,000
#   Downsample layers:   200,000
#   Stage 3 (9 blocks):  5,330,880
#   Stage 4 (12 blocks): 28,371,456
#   Final norm:          1,536
#   Head:                6,152
#   ----------------------------------------
#   TOTAL:               37,910,024
#   TARGET:              21,920,000
#   DIFFERENCE:          15,990,024

#   Per-block breakdown:
#     Stage 3 block (384): 592,320 params
#       - Attention:       442,752
#       - MLP:             148,032
#       - LayerNorms:      1,536
#     Stage 4 block (768): 2,364,288 params
#       - Attention:       1,770,240
#       - MLP:             590,976
#       - LayerNorms:      3,072

# ============================================================
# RECOMMENDATION
# ============================================================

# If parameters are still too high, reduce MLP expansion further.
# Current: 0.5x expansion (dim -> dim/2 -> dim)
# Consider: Remove MLP entirely or use identity mapping
# (convnextv2) ubuntu@panacea-ml-learn:~/Documents/Nikita/ConvNeXtv2-hybrid-classifier$ python count_params.py
# ============================================================
# DETAILED PARAMETER BREAKDOWN
# ============================================================

# Parameter budget analysis:
#   Base (ConvNeXt 1-2): 2,717,952
#   Attention only:      25,260,680
#   Remaining for MLPs:  -6,058,632.0
#   Calculated MLP budget: -239.056x
#   Using MLP ratio: 0.0x

# Final parameter count: 27,978,632 (27.98M)
# Target: 21,920,000 (21.92M)
# Difference: +27.64%

# Component-wise breakdown:
# ------------------------------------------------------------
# stem           :        4,896 params ( 0.02% of target)
# stage1         :      239,904 params ( 1.09% of target)
# down1          :       74,112 params ( 0.34% of target)
# stage2         :      922,176 params ( 4.21% of target)
# down2          :      295,680 params ( 1.35% of target)
# stage3         :    3,991,680 params (18.21% of target)
# down3          :    1,181,184 params ( 5.39% of target)
# stage4         :   21,261,312 params (97.00% of target)
# norm           :        1,536 params ( 0.01% of target)
# head           :        6,152 params ( 0.03% of target)
# ------------------------------------------------------------
# TOTAL          :   27,978,632 params
# TARGET         :   21,920,000 params
# DIFFERENCE     :    6,058,632 params (+27.64%)

# ============================================================
# ATTENTION STAGE BREAKDOWN
# ============================================================

# Stage 3 (9 blocks, dim=384):
#   Per block: 443,520 params
#   Total: 3,991,680 params

#   Single block breakdown:
#     norm1               : 768 params
#     attn                : 442,752 params

# Stage 4 (12 blocks, dim=768):
#   Per block: 1,771,776 params
#   Total: 21,261,312 params

#   Single block breakdown:
#     norm1               : 1,536 params
#     attn                : 1,770,240 params

# ============================================================
# SEPARABLE ATTENTION ANALYSIS
# ============================================================

# Stage 3 attention (dim=384):
#   context_score.weight          : [1, 384] = 384 params
#   key_proj.weight               : [384, 384] = 147,456 params
#   value_proj.weight             : [384, 384] = 147,456 params
#   out_proj.weight               : [384, 384] = 147,456 params
#   TOTAL                         : 442,752 params

# Stage 4 attention (dim=768):
#   context_score.weight          : [1, 768] = 768 params
#   key_proj.weight               : [768, 768] = 589,824 params
#   value_proj.weight             : [768, 768] = 589,824 params
#   out_proj.weight               : [768, 768] = 589,824 params
#   TOTAL                         : 1,770,240 params

# ============================================================
# MLP ANALYSIS
# ============================================================

# Stage 3 MLP (dim=384):
# Traceback (most recent call last):
#   File "count_params.py", line 234, in <module>
#     main()
#   File "count_params.py", line 222, in main
#     detailed_breakdown()
#   File "count_params.py", line 130, in detailed_breakdown
#     for name, param in mlp.named_parameters():
# AttributeError: 'NoneType' object has no attribute 'named_parameters'
# (convnextv2) ubuntu@panacea-ml-learn:~/Documents/Nikita/ConvNeXtv2-hybrid-classifier$ python fix_init.py
# Testing LayerNorm Fix
# ============================================================

# Before fix:
#   stage3.0.norm1: std=0.000000
#   stage3.0.norm2: std=0.000000
#   stage3.1.norm1: std=0.000000
#   stage3.1.norm2: std=0.000000
#   stage3.2.norm1: std=0.000000

# Total LayerNorms: 43
# Zero-initialized: 43

# Fixed 43 LayerNorm modules

# After fix:
# Zero-initialized: 43

# Testing forward pass...
# Output: torch.Size([2, 8])
# Output stats: mean=0.0328, std=0.6079
# ✓ Output is valid
# (convnextv2) ubuntu@panacea-ml-learn:~/Documents/Nikita/ConvNeXtv2-hybrid-classifier$ 

