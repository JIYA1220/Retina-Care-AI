"""
dataset.py
PyTorch Dataset for APTOS 2019.
Expected folder layout:
    data/
        train_images/   *.png
        train.csv       columns: id_code, diagnosis
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.preprocess import load_and_preprocess


# -- Albumentations transforms ------------------------------------------------

def get_train_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=30, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
        ], p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# -- Dataset ------------------------------------------------------------------

class APTOSDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str,
                 img_size: int = 224, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["id_code"] + ".png")
        img = load_and_preprocess(img_path, self.img_size)  # np array RGB

        if self.transform:
            img = self.transform(image=img)["image"]

        label = torch.tensor(int(row["diagnosis"]), dtype=torch.long)
        return img, label


# -- Weighted sampler to handle class imbalance --------------------------------

def build_weighted_sampler(csv_path: str) -> WeightedRandomSampler:
    df = pd.read_csv(csv_path)
    class_counts = df["diagnosis"].value_counts().sort_index().values
    weights_per_class = 1.0 / class_counts
    sample_weights = df["diagnosis"].map(
        lambda c: weights_per_class[c]).values
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )
