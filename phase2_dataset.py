"""
Phase 2 Dataset — Imbalanced Split with Dual Output
=====================================================
Creates 1:9 (Fake:Real) imbalanced dataset and provides both
RGB images and DCT coefficient maps for each sample.

The DCT is computed as a pure NumPy operation to avoid GPU issues
in DataLoader workers. The model handles GPU operations.

Usage:
    python phase2_dataset.py                    # Create imbalanced splits
    from phase2_dataset import get_phase2_loaders
    train_ld, val_ld, test_ld = get_phase2_loaders()
"""

import os
import sys
import random
import csv
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─── Import config values without triggering GPU init in workers ─────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_REAL = os.path.join(BASE_DIR, "processed_data", "real")
PROCESSED_FAKE = os.path.join(BASE_DIR, "processed_data", "fake")
PHASE2_SPLITS_DIR = os.path.join(BASE_DIR, "phase2_splits")
BALANCED_SPLITS_DIR = os.path.join(BASE_DIR, "splits")

NUM_REAL_IMBA = 20_000
NUM_FAKE_IMBA = 2_222
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
IMAGE_SIZE = 224
DCT_BLOCK_SIZE = 8
PHYSICAL_BATCH_SIZE = 16
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── Pure NumPy Block-wise DCT ──────────────────────────────────────────
def _build_dct_basis(N=8):
    """Precompute 1D DCT-II basis matrix."""
    basis = np.zeros((N, N), dtype=np.float32)
    for k in range(N):
        for n in range(N):
            alpha = math.sqrt(1.0 / N) if k == 0 else math.sqrt(2.0 / N)
            basis[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / (2 * N))
    return basis

_DCT_BASIS = _build_dct_basis(DCT_BLOCK_SIZE)


def compute_dct_numpy(img_array, block_size=8):
    """
    Compute block-wise 2D DCT on a numpy image array.

    Args:
        img_array: [H, W, 3] uint8 or float32 numpy array
        block_size: DCT block size (default 8)
    Returns:
        [3*64, H//8, W//8] = [192, 28, 28] float32 numpy array
    """
    if img_array.dtype == np.uint8:
        img_array = img_array.astype(np.float32) / 255.0

    H, W, C = img_array.shape
    bs = block_size
    h_blocks = H // bs
    w_blocks = W // bs

    # Crop to exact multiple of block_size
    img_array = img_array[:h_blocks * bs, :w_blocks * bs, :]

    # Transpose to [C, H, W]
    img = img_array.transpose(2, 0, 1)  # [3, H, W]

    # Reshape into blocks: [C, h_blocks, bs, w_blocks, bs]
    img = img.reshape(C, h_blocks, bs, w_blocks, bs)
    # Permute: [C, h_blocks, w_blocks, bs, bs]
    img = img.transpose(0, 1, 3, 2, 4)

    # Apply 2D DCT: basis @ block @ basis^T
    # shapes: basis [bs, bs], img [C, hb, wb, bs, bs], basis^T [bs, bs]
    dct_coeffs = _DCT_BASIS @ img @ _DCT_BASIS.T
    # dct_coeffs: [C, h_blocks, w_blocks, bs, bs]

    # Rearrange: [C, h_blocks, w_blocks, bs*bs] → [C*64, h_blocks, w_blocks]
    dct_coeffs = dct_coeffs.reshape(C, h_blocks, w_blocks, bs * bs)
    dct_coeffs = dct_coeffs.transpose(0, 3, 1, 2)  # [C, 64, hb, wb]
    dct_coeffs = dct_coeffs.reshape(C * bs * bs, h_blocks, w_blocks)  # [192, 28, 28]

    return dct_coeffs


# ─── Module-level worker init (Windows pickling) ────────────────────────
def _seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


class Phase2Dataset(Dataset):
    """
    Returns (image_tensor, dct_coefficients, label) for each sample.
    DCT is computed on-the-fly using pure NumPy (no GPU).
    """

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Validate
        valid = self.df['image_path'].apply(os.path.exists)
        invalid_count = (~valid).sum()
        if invalid_count > 0:
            print(f"[Phase2 Dataset] WARNING: {invalid_count} missing in {csv_path}")
            self.df = self.df[valid].reset_index(drop=True)

        n_real = (self.df.label == 0).sum()
        n_fake = (self.df.label == 1).sum()
        total = len(self.df)
        pct = n_fake / total * 100 if total > 0 else 0
        print(f"[Phase2 Dataset] {os.path.basename(csv_path)}: "
              f"{total} (real={n_real}, fake={n_fake}, {pct:.1f}% fake)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        label = torch.tensor(row['label'], dtype=torch.float32)

        # Resize for DCT (needs exact 224×224)
        img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        img_np = np.array(img_resized)  # [224, 224, 3] uint8

        # Compute DCT from raw pixels (pure NumPy, CPU-safe)
        dct_coeffs = compute_dct_numpy(img_np, DCT_BLOCK_SIZE)  # [192, 28, 28]
        dct_tensor = torch.from_numpy(dct_coeffs)

        # Apply augmentation + normalization for the RGB branch
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        return img_tensor, dct_tensor, label


def create_imbalanced_splits():
    """Create 1:9 imbalanced train/val/test splits."""
    random.seed(SEED)

    real_images = [os.path.join(PROCESSED_REAL, f) for f in os.listdir(PROCESSED_REAL)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    fake_images = [os.path.join(PROCESSED_FAKE, f) for f in os.listdir(PROCESSED_FAKE)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"[Split] Available: {len(real_images):,} real, {len(fake_images):,} fake")

    real_sampled = random.sample(real_images, min(NUM_REAL_IMBA, len(real_images)))
    fake_sampled = random.sample(fake_images, min(NUM_FAKE_IMBA, len(fake_images)))

    total = len(real_sampled) + len(fake_sampled)
    print(f"[Split] Imbalanced: {len(real_sampled)} real + {len(fake_sampled)} fake "
          f"({len(fake_sampled)/total*100:.1f}% fake)")

    dataset = [(p, 0) for p in real_sampled] + [(p, 1) for p in fake_sampled]
    random.shuffle(dataset)

    n = len(dataset)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train_imba.csv": dataset[:n_train],
        "val_imba.csv": dataset[n_train:n_train + n_val],
        "test_imba.csv": dataset[n_train + n_val:],
    }

    os.makedirs(PHASE2_SPLITS_DIR, exist_ok=True)
    for fname, data in splits.items():
        path = os.path.join(PHASE2_SPLITS_DIR, fname)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])
            writer.writerows(data)
        n_r = sum(1 for _, l in data if l == 0)
        n_f = sum(1 for _, l in data if l == 1)
        print(f"[Split] {fname}: {len(data)} (real={n_r}, fake={n_f}, "
              f"{n_f/len(data)*100:.1f}% fake)")


def get_phase2_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_phase2_loaders(use_balanced=False):
    """Create Phase 2 DataLoaders."""
    train_tf, eval_tf = get_phase2_transforms()

    if use_balanced:
        train_csv = os.path.join(BALANCED_SPLITS_DIR, "train.csv")
        val_csv = os.path.join(BALANCED_SPLITS_DIR, "val.csv")
        test_csv = os.path.join(BALANCED_SPLITS_DIR, "test.csv")
    else:
        train_csv = os.path.join(PHASE2_SPLITS_DIR, "train_imba.csv")
        val_csv = os.path.join(PHASE2_SPLITS_DIR, "val_imba.csv")
        test_csv = os.path.join(PHASE2_SPLITS_DIR, "test_imba.csv")

    if not use_balanced and not os.path.exists(train_csv):
        print("[Phase2] Creating imbalanced splits...")
        create_imbalanced_splits()

    g = torch.Generator()
    g.manual_seed(SEED)
    num_workers = 2 if os.name == 'nt' else 4

    train_ds = Phase2Dataset(train_csv, transform=train_tf)
    val_ds = Phase2Dataset(val_csv, transform=eval_tf)
    test_ds = Phase2Dataset(test_csv, transform=eval_tf)

    kw = dict(pin_memory=True, persistent_workers=num_workers > 0)

    train_ld = DataLoader(train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True,
                          num_workers=num_workers, worker_init_fn=_seed_worker,
                          generator=g, drop_last=True, **kw)
    val_ld = DataLoader(val_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, **kw)
    test_ld = DataLoader(test_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False,
                         num_workers=num_workers, **kw)

    print(f"[Phase2 DataLoader] Workers: {num_workers}")
    return train_ld, val_ld, test_ld


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 2 — Creating Imbalanced Dataset Splits")
    print("=" * 60)
    create_imbalanced_splits()

    print("\n[Test] Loading one batch...")
    train_ld, _, _ = get_phase2_loaders()
    imgs, dcts, labels = next(iter(train_ld))
    print(f"  Images: {imgs.shape}  dtype={imgs.dtype}")
    print(f"  DCT:    {dcts.shape}  dtype={dcts.dtype}")
    print(f"  Labels: {labels[:8]}")
    print(f"  Fake ratio: {labels.sum()/len(labels)*100:.1f}%")
