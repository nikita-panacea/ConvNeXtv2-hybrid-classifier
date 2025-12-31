# datasets/isic2019_dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

ISIC_CLASSES = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"]

class ISIC2019Dataset(Dataset):
    """
    CSV-based ISIC-2019 loader.

    CSV expectations:
    - column 'image' with image id (without extension), e.g., ISIC_0000000
    - either 'label' column with integer 0..7 OR one-hot columns for ISIC_CLASSES
    """

    def __init__(self, csv_path, image_dir, transform=None, ext=".jpg"):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.ext = ext

        if "label" in self.df.columns:
            self.labels = self.df["label"].values.astype(int)
        else:
            # expect one-hot columns
            self.labels = self.df[ISIC_CLASSES].values.argmax(axis=1)

        self.image_ids = self.df["image"].astype(str).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = int(self.labels[idx])

        path = os.path.join(self.image_dir, img_id + self.ext)
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label, img_id
