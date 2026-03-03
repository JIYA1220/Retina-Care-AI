"""
train.py
Train EfficientNet-B0 or ResNet-50 on APTOS 2019 dataset.

Usage:
    python train.py --data_dir data --model efficientnet_b0 --epochs 15 --batch_size 8
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from model.model import build_model
from utils.dataset import (
    APTOSDataset, build_weighted_sampler,
    get_train_transforms, get_val_transforms,
)
from utils.early_stopping import EarlyStopping
from utils.losses import FocalLoss


# -- Reproducibility -----------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -- Helpers -------------------------------------------------------------------

def save_split_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return kappa


# -- Main ----------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- Split data --
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["diagnosis"], random_state=SEED
    )

    train_csv = os.path.join(args.data_dir, "_train_split.csv")
    val_csv   = os.path.join(args.data_dir, "_val_split.csv")
    save_split_csv(train_df, train_csv)
    save_split_csv(val_df,   val_csv)

    img_dir = os.path.join(args.data_dir, "train_images")

    # -- Datasets & loaders --
    train_ds = APTOSDataset(train_csv, img_dir,
                            img_size=160, transform=get_train_transforms(160))
    val_ds   = APTOSDataset(val_csv,   img_dir,
                            img_size=160, transform=get_val_transforms())

    sampler = build_weighted_sampler(train_csv)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    # -- Model, loss, optimiser --
    print(f"Building model: {args.model}")
    model = build_model(model_name=args.model, pretrained=True).to(device)

    # Balanced Class Weights (normalized)
    class_counts = train_df["diagnosis"].value_counts().sort_index().values
    weights = len(train_df) / (len(class_counts) * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    # Focal Loss with adjusted gamma for stronger signal
    criterion = FocalLoss(weight=class_weights, gamma=1.5)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.05
    )
    # Cosine Annealing with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=5, T_mult=1, eta_min=1e-6
    )

    # -- Training loop --
    os.makedirs("model", exist_ok=True)
    early_stopper = EarlyStopping(patience=args.patience, min_delta=0.001, mode='max')
    
    save_path = f"model/{args.model}_dr.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        # Progress bar for the current epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)
            optimiser.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            
            # Step the scheduler per-batch for a smooth cosine curve
            scheduler.step(epoch - 1 + batch_idx / len(train_loader))
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "lr": optimiser.param_groups[0]['lr']})

        avg_loss = running_loss / len(train_loader)
        kappa = evaluate(model, val_loader, device)

        print(f"  Loss: {avg_loss:.4f}  |  Val Kappa: {kappa:.4f}")

        # Early Stopping Logic
        early_stopper(kappa, model)
        
        if early_stopper.counter == 0:
            # Improvement found, save checkpoint
            torch.save(model.state_dict(), save_path)
            print(f"  ✔ New best model saved (kappa={kappa:.4f})")
        
        if early_stopper.early_stop:
            print(f"\nEarly stopping triggered. No improvement for {args.patience} epochs.")
            print(f"Restoring best weights with Kappa: {early_stopper.best_score:.4f}")
            # Restore best weights before finalizing
            model.load_state_dict(early_stopper.get_best_state())
            torch.save(model.state_dict(), save_path)
            break

    print(f"\nTraining complete. Best val kappa: {early_stopper.best_score:.4f}")
    print(f"Weights finalized -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--model",      default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=5)
    main(parser.parse_args())
