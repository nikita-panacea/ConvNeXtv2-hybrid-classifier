# run_train.sh
python train.py \
    --train_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/train.csv \
    --val_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/val.csv \
    --test_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/test.csv \
    --img_dir /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/Images \
    --epochs 200 \
    --batch_size 32 \
    --out_dir checkpoints-ISIC2019-13Jan-Tiny22k-4xMLP-model/

    # --peak_lr 0.01 \
    # --warmup_epochs 5 \
    # --mlp_ratio 0.25 \

