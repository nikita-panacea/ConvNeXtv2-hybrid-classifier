python train.py \
 --train_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/train.csv \
 --img_dir   /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/Images \
 --overfit_mode \
 --overfit_n 8 \
 --epochs 50 \
 --batch_size 8 \
 --optimizer adamw \
 --base_lr 1e-3 \
 --no_scheduler

#  --val_csv   /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/val.csv \