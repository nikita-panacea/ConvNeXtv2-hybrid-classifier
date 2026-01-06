import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

ISIC_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

class ISIC2019Dataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # sanity check
        for c in ISIC_CLASSES:
            assert c in self.df.columns, f"Missing column {c} in CSV"

        # convert one-hot → class index ONCE
        self.labels = self.df[ISIC_CLASSES].values.argmax(axis=1)

        # image column
        if "image" in self.df.columns:
            self.images = self.df["image"].values
        elif "image_id" in self.df.columns:
            self.images = self.df["image_id"].values
        else:
            raise ValueError("CSV must contain 'image' or 'image_id' column")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        img_name = self.images[idx]
        if not img_name.endswith(".jpg"):
            img_name += ".jpg"

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        label = int(self.labels[idx])  # <-- THIS IS CRITICAL
        
        if self.transform:
            image = self.transform(image)

        if idx < 5:
            print("DEBUG label:", label)

        return image, label, img_name
