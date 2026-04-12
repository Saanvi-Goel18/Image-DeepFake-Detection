"""
Step 2: PyTorch Dataset & DataLoaders
======================================
- DeepfakeDataset reads CSV split files
- Train: augmentation (flip, rotate, color jitter, crop)
- Val/Test: clean resize + center crop
- ImageNet normalization throughout

Usage:
    from dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    SPLITS_DIR, IMAGE_SIZE, PHYSICAL_BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    IMAGENET_MEAN, IMAGENET_STD, SEED
)


class DeepfakeDataset(Dataset):
    """Binary classification dataset: 0=real, 1=fake."""

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Validate
        valid_mask = self.df['image_path'].apply(os.path.exists)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"[Dataset] WARNING: {invalid_count} missing files in {csv_path}")
            self.df = self.df[valid_mask].reset_index(drop=True)

        print(f"[Dataset] Loaded {len(self.df)} samples from {os.path.basename(csv_path)} "
              f"(real={len(self.df[self.df.label==0])}, fake={len(self.df[self.df.label==1])})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        label = torch.tensor(row['label'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """Return train and val/test transforms."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def _seed_worker(worker_id):
    """Worker init function — must be at module level for Windows pickling."""
    import numpy as np
    np.random.seed(SEED + worker_id)
    import random
    random.seed(SEED + worker_id)


def get_dataloaders():
    """Create train, val, test DataLoaders."""
    train_tf, eval_tf = get_transforms()

    g = torch.Generator()
    g.manual_seed(SEED)

    # Use fewer workers on Windows to avoid multiprocessing issues
    num_workers = min(NUM_WORKERS, 2) if os.name == 'nt' else NUM_WORKERS

    train_ds = DeepfakeDataset(os.path.join(SPLITS_DIR, "train.csv"), transform=train_tf)
    val_ds   = DeepfakeDataset(os.path.join(SPLITS_DIR, "val.csv"),   transform=eval_tf)
    test_ds  = DeepfakeDataset(os.path.join(SPLITS_DIR, "test.csv"),  transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        worker_init_fn=_seed_worker, generator=g, drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"[DataLoader] Workers: {num_workers}, Pin memory: {PIN_MEMORY}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Check one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels[:8]}")
    print(f"Pixel range: [{images.min():.3f}, {images.max():.3f}]")
