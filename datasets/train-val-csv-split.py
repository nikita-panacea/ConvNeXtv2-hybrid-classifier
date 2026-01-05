from sklearn.model_selection import StratifiedShuffleSplit
from isic2019_dataset import ISIC2019Dataset, ISIC_CLASSES
import pandas as pd

def stratified_split(csv_path, out_dir, seed=42):
    df = pd.read_csv(csv_path)

    if "label" in df.columns:
        y = df["label"]
    else:
        y = df[ISIC_CLASSES].values.argmax(axis=1)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(df, y))

    temp_df = df.iloc[temp_idx]
    y_temp = y[temp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_df, y_temp))

    df.iloc[train_idx].to_csv(f"{out_dir}/train.csv", index=False)
    temp_df.iloc[val_idx].to_csv(f"{out_dir}/val.csv", index=False)
    temp_df.iloc[test_idx].to_csv(f"{out_dir}/test.csv", index=False)


if __name__ == '__main__':
    csv_path = '/home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/ISIC_2019_Training_GroundTruth.csv'
    out_dir = '/home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set'
    stratified_split(csv_path=csv_path, out_dir=out_dir)
