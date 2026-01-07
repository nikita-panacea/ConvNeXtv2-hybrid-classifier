python train.py \
    --train_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/train.csv \
    --val_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/val.csv \
    --test_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/test.csv \
    --img_dir /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/Images \
    --epochs 100 \
    --batch_size 32 \
    --out_dir checkpoints-ISIC2019/

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
