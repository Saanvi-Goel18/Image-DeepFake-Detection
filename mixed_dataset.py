"""
Mixed Dataset — AIGuard + FaceForensics++ for SFCANet-v3
=========================================================
Combines two source domains into a single DataLoader:
  Domain 0: AIGuard image dataset (GAN/diffusion deepfakes — JPEG statics)
  Domain 1: FaceForensics++ video frames (FaceSwap, Deepfakes, etc. — video codec)

Key techniques for cross-dataset generalization:
  1. JPEG Augmentation: randomly re-compress images at quality 40–95 to prevent
     the DCT branch from memorizing a specific compression fingerprint.
  2. Multi-Scale DCT: computes block-DCT at 3 granularities (4, 8, 16) per image.
  3. Domain Labels: each sample carries a domain tag used by the GRL adversarial head.

Returns per sample:
  rgb       : [3, 224, 224]         — normalized RGB frame
  dct_dict  : {4: [48, 56, 56],
               8: [192, 28, 28],
               16: [768, 14, 14]}   — multi-scale DCT coefficients
  label     : scalar 0|1            — real/fake
  domain    : scalar 0|1            — AIGuard | FF++
"""

import io
import os
import csv
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from phase3_config import (
    AIGU_REAL, AIGU_FAKE,
    FFPP_TRAIN_DIR, FFPP_VAL_DIR, FFPP_TEST_DIR,
    FFPP_REAL_DIR, FFPP_FAKE_DIRS,
    PHASE3_SPLITS_DIR,
    IMAGE_SIZE, DCT_SCALES, DCT_CHANNELS, DCT_SPATIAL_DIMS,
    IMAGENET_MEAN, IMAGENET_STD,
    AIGU_TRAIN_RATIO, AIGU_VAL_RATIO,
    FFPP_MIX_RATIO, PHYSICAL_BATCH_SIZE, SEED,
)


# ─── DCT Basis Matrix (pre-computed, shared) ─────────────────────────────────

def _make_dct_basis(N: int) -> np.ndarray:
    basis = np.zeros((N, N), dtype=np.float32)
    for k in range(N):
        for n in range(N):
            alpha = math.sqrt(1.0 / N) if k == 0 else math.sqrt(2.0 / N)
            basis[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / (2 * N))
    return basis


_DCT_BASIS_CACHE: dict = {}

def _get_basis(N: int) -> np.ndarray:
    if N not in _DCT_BASIS_CACHE:
        _DCT_BASIS_CACHE[N] = _make_dct_basis(N)
    return _DCT_BASIS_CACHE[N]


# ─── Multi-Scale DCT ─────────────────────────────────────────────────────────

def compute_multiscale_dct(img_np: np.ndarray,
                           scales: list = DCT_SCALES) -> dict:
    """
    Compute block-wise 2D DCT at multiple block sizes.

    Args:
        img_np: [H, W, 3] uint8 image
        scales: list of block sizes (e.g. [4, 8, 16])

    Returns:
        dict {block_size: np.ndarray [3*s*s, H//s, W//s]}
    """
    if img_np.dtype == np.uint8:
        img_f = img_np.astype(np.float32) / 255.0
    else:
        img_f = img_np.astype(np.float32)

    H, W, C = img_f.shape
    result = {}

    for s in scales:
        basis = _get_basis(s)
        hb = H // s
        wb = W // s
        # Crop to exact multiple
        arr = img_f[:hb * s, :wb * s, :]          # [H', W', C]
        arr = arr.transpose(2, 0, 1)               # [C, H', W']
        arr = arr.reshape(C, hb, s, wb, s)         # [C, hb, s, wb, s]
        arr = arr.transpose(0, 1, 3, 2, 4)         # [C, hb, wb, s, s]

        # 2D DCT via separable 1D: basis @ block @ basis^T
        dct = basis @ arr @ basis.T                # [C, hb, wb, s, s]
        dct = dct.reshape(C, hb, wb, s * s)       # [C, hb, wb, s²]
        dct = dct.transpose(0, 3, 1, 2)           # [C, s², hb, wb]
        dct = dct.reshape(C * s * s, hb, wb)      # [C*s², hb, wb]

        result[s] = dct  # float32 numpy

    return result


# ─── JPEG Compression Augmentation ───────────────────────────────────────────

def jpeg_augment(img_pil: Image.Image, quality_range=(60, 90), p=0.33) -> Image.Image:
    """Randomly re-compress an image as JPEG to simulate social media (WhatsApp/Instagram)."""
    if random.random() < p:
        quality = random.randint(*quality_range)
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        img_pil = Image.open(buf).copy()  # .copy() to avoid lazy load issues
    return img_pil

# ─── Fourier Perturbation (v4.1) ─────────────────────────────────────────────

def fourier_perturbation(img_pil: Image.Image, p=0.3) -> Image.Image:
    """Randomly perturb high-frequency phase/amplitude to prevent over-reliance on static noise."""
    if random.random() > p:
        return img_pil
    
    img_np = np.array(img_pil).astype(np.float32)
    h, w, c = img_np.shape
    out_np = np.zeros_like(img_np)
    
    for ch in range(c):
        fft = np.fft.fft2(img_np[:, :, ch])
        fshift = np.fft.fftshift(fft)
        
        # Create a high-frequency mask (keep center low-freq intact)
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4  # inner radius (low freq)
        y, x = np.ogrid[:h, :w]
        mask = (x - cx)**2 + (y - cy)**2 > r**2
        
        # Perturb phase and magnitude of high frequencies
        mag = np.abs(fshift)
        phase = np.angle(fshift)
        
        phase_noise = np.random.uniform(-0.5, 0.5, size=(h, w)) * mask
        phase = phase + phase_noise
        
        mag_scale = np.random.uniform(0.8, 1.2, size=(h, w))
        mag = np.where(mask, mag * mag_scale, mag)
        
        fshift_new = mag * np.exp(1j * phase)
        ifft = np.fft.ifft2(np.fft.ifftshift(fshift_new))
        out_np[:, :, ch] = np.real(ifft)
        
    out_np = np.clip(out_np, 0, 255).astype(np.uint8)
    return Image.fromarray(out_np)


# ─── Transforms ──────────────────────────────────────────────────────────────

def _make_transforms(augment: bool):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ─── AIGuard Sample Collection ───────────────────────────────────────────────

def _collect_aigu_samples(split: str) -> list:
    """Returns list of (path, label) tuples from AIGuard processed_data."""
    cache_path = os.path.join(PHASE3_SPLITS_DIR, f"aigu_{split}.csv")

    if os.path.exists(cache_path):
        samples = []
        with open(cache_path) as f:
            for row in csv.DictReader(f):
                samples.append((row['path'], int(row['label'])))
        print(f"[AIGuard] Loaded {len(samples)} {split} samples from cache.")
        return samples

    # Collect & split fresh
    rng = random.Random(SEED)
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    real = [os.path.join(AIGU_REAL, f)
            for f in os.listdir(AIGU_REAL)
            if os.path.splitext(f)[1].lower() in exts]
    fake = [os.path.join(AIGU_FAKE, f)
            for f in os.listdir(AIGU_FAKE)
            if os.path.splitext(f)[1].lower() in exts]

    def _split(items):
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * AIGU_TRAIN_RATIO)
        n_va = int(n * AIGU_VAL_RATIO)
        return {'train': items[:n_tr],
                'val':   items[n_tr:n_tr + n_va],
                'test':  items[n_tr + n_va:]}

    real_splits = _split(real)
    fake_splits = _split(fake)

    all_splits = {
        'train': [(p, 0) for p in real_splits['train']] + [(p, 1) for p in fake_splits['train']],
        'val':   [(p, 0) for p in real_splits['val']]   + [(p, 1) for p in fake_splits['val']],
        'test':  [(p, 0) for p in real_splits['test']]  + [(p, 1) for p in fake_splits['test']],
    }

    # Persist
    for sp, items in all_splits.items():
        pth = os.path.join(PHASE3_SPLITS_DIR, f"aigu_{sp}.csv")
        with open(pth, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['path', 'label'])
            w.writerows(items)
    print(f"[AIGuard] Created splits — train: {len(all_splits['train'])}, "
          f"val: {len(all_splits['val'])}, test: {len(all_splits['test'])}")
    return all_splits[split]


# ─── FF++ Sample Collection ──────────────────────────────────────────────────

def _collect_ffpp_samples(crops_dir: str) -> list:
    """Returns list of (npy_path, frame_idx, label) from FF++ crops dir."""
    if not os.path.isdir(crops_dir):
        print(f"[FF++] WARNING: {crops_dir} not found — skipping FF++.")
        return []

    samples = []
    # Real
    real_dir = os.path.join(crops_dir, FFPP_REAL_DIR)
    if os.path.isdir(real_dir):
        for vid in os.listdir(real_dir):
            vp = os.path.join(real_dir, vid)
            if not os.path.isdir(vp):
                continue
            for f in os.listdir(vp):
                if f.endswith('.npy'):
                    for fi in range(8):  # 8 frames per segment
                        samples.append((os.path.join(vp, f), fi, 0))

    # Fake (all manipulation types)
    for fd in FFPP_FAKE_DIRS:
        fd_path = os.path.join(crops_dir, fd)
        if not os.path.isdir(fd_path):
            continue
        for vid in os.listdir(fd_path):
            vp = os.path.join(fd_path, vid)
            if not os.path.isdir(vp):
                continue
            for f in os.listdir(vp):
                if f.endswith('.npy'):
                    for fi in range(8):
                        samples.append((os.path.join(vp, f), fi, 1))

    print(f"[FF++] {crops_dir.split(os.sep)[-1]}: {len(samples)} frame samples "
          f"({sum(1 for _,_,l in samples if l==0)} real, "
          f"{sum(1 for _,_,l in samples if l==1)} fake)")
    return samples


# ─── Dataset Classes ─────────────────────────────────────────────────────────

class AIGuardDataset(Dataset):
    """Dataset wrapper for AIGuard JPEG images."""

    def __init__(self, samples: list, augment: bool = True):
        self.samples = samples
        self.augment = augment
        self.tf = _make_transforms(augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        # JPEG augmentation (breaks compression fingerprint bias)
        if self.augment:
            img = jpeg_augment(img, quality_range=(40, 95), p=0.6)

        img_np = np.array(img)                             # [224, 224, 3] uint8
        dct_np = compute_multiscale_dct(img_np, DCT_SCALES)
        dct_dict = {s: torch.from_numpy(dct_np[s]) for s in DCT_SCALES}

        rgb = self.tf(img)

        return {
            'rgb':    rgb,
            'dct':    dct_dict,
            'label':  torch.tensor(label, dtype=torch.float32),
            'domain': torch.tensor(0, dtype=torch.float32),  # AIGuard = 0
        }


class FFPPDataset(Dataset):
    """Dataset wrapper for FF++ pre-extracted .npy frame segments."""

    def __init__(self, samples: list, augment: bool = True):
        self.samples = samples
        self.augment = augment
        self.tf = _make_transforms(augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, frame_idx, label = self.samples[idx]

        try:
            stacked = np.load(npy_path)        # [8, H, W, 3] uint8
            frame_np = stacked[frame_idx]      # [H, W, 3]
        except Exception:
            frame_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        img = Image.fromarray(frame_np.astype(np.uint8)).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        # JPEG augmentation
        if self.augment:
            img = jpeg_augment(img, quality_range=(40, 95), p=0.6)

        img_np = np.array(img)
        dct_np = compute_multiscale_dct(img_np, DCT_SCALES)
        dct_dict = {s: torch.from_numpy(dct_np[s]) for s in DCT_SCALES}

        rgb = self.tf(img)

        return {
            'rgb':    rgb,
            'dct':    dct_dict,
            'label':  torch.tensor(label, dtype=torch.float32),
            'domain': torch.tensor(1, dtype=torch.float32),  # FF++ = 1
        }


# ─── Mixed Collate ────────────────────────────────────────────────────────────

def mixed_collate(batch: list) -> dict:
    """Custom collate that handles the dct_dict (dict of tensors) inside each sample."""
    rgb    = torch.stack([b['rgb']    for b in batch])
    label  = torch.stack([b['label'] for b in batch])
    domain = torch.stack([b['domain'] for b in batch])

    dct_dict = {}
    for s in DCT_SCALES:
        dct_dict[s] = torch.stack([b['dct'][s] for b in batch])

    return {'rgb': rgb, 'dct': dct_dict, 'label': label, 'domain': domain}


# ─── DataLoader Builder ───────────────────────────────────────────────────────

def get_mixed_loaders():
    """
    Build train / val / test DataLoaders with AIGuard + FF++ mixed data.

    Training: weighted sampler maintains ~FFPP_MIX_RATIO of FF++ per batch.
    Val/Test: AIGuard only (to stay comparable to Phase 2 metrics) +
              FF++ for cross-domain evaluation (separate loader returned).
    """
    # --- Collect samples ---
    aigu_train = _collect_aigu_samples('train')
    aigu_val   = _collect_aigu_samples('val')
    aigu_test  = _collect_aigu_samples('test')

    ffpp_train = _collect_ffpp_samples(FFPP_TRAIN_DIR)
    ffpp_val   = _collect_ffpp_samples(FFPP_VAL_DIR)
    ffpp_test  = _collect_ffpp_samples(FFPP_TEST_DIR)

    # Subsample FF++ to target mix ratio in training
    n_aigu      = len(aigu_train)
    n_ffpp_want = int(n_aigu * FFPP_MIX_RATIO / (1 - FFPP_MIX_RATIO))
    rng = random.Random(SEED + 1)
    if len(ffpp_train) > n_ffpp_want:
        ffpp_train_sub = rng.sample(ffpp_train, n_ffpp_want)
    else:
        ffpp_train_sub = ffpp_train

    print(f"\n[MixedLoader] Training — AIGuard: {n_aigu} | FF++: {len(ffpp_train_sub)}")
    print(f"[MixedLoader] Val      — AIGuard: {len(aigu_val)} | FF++: {len(ffpp_val)}")

    # --- Build datasets ---
    train_ds = torch.utils.data.ConcatDataset([
        AIGuardDataset(aigu_train,    augment=True),
        FFPPDataset(ffpp_train_sub,   augment=True),
    ])
    val_aigu_ds  = AIGuardDataset(aigu_val,  augment=False)
    val_ffpp_ds  = FFPPDataset(ffpp_val,     augment=False)
    test_aigu_ds = AIGuardDataset(aigu_test, augment=False)
    test_ffpp_ds = FFPPDataset(ffpp_test,    augment=False)

    # Weighted sampler to keep class balance in train batches
    # Weight: fake samples get 9× weight to counter imbalance; FF++ gets 1×
    weights = []
    for ds in train_ds.datasets:
        for item_idx in range(len(ds)):
            if isinstance(ds, AIGuardDataset):
                _, lbl = ds.samples[item_idx]
            else:
                _, _, lbl = ds.samples[item_idx]
            weights.append(9.0 if lbl == 1 else 1.0)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    num_workers = 2 if os.name == 'nt' else 4
    pin = True

    train_ld = DataLoader(
        train_ds, batch_size=PHYSICAL_BATCH_SIZE,
        sampler=sampler, num_workers=num_workers,
        collate_fn=mixed_collate, pin_memory=pin,
        drop_last=True, persistent_workers=(num_workers > 0),
    )
    val_aigu_ld = DataLoader(
        val_aigu_ds, batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False, num_workers=num_workers,
        collate_fn=mixed_collate, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    val_ffpp_ld = DataLoader(
        val_ffpp_ds, batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False, num_workers=num_workers,
        collate_fn=mixed_collate, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    test_aigu_ld = DataLoader(
        test_aigu_ds, batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False, num_workers=num_workers,
        collate_fn=mixed_collate, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    test_ffpp_ld = DataLoader(
        test_ffpp_ds, batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False, num_workers=num_workers,
        collate_fn=mixed_collate, pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    return train_ld, val_aigu_ld, val_ffpp_ld, test_aigu_ld, test_ffpp_ld


# ─── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from phase3_config import DCT_CHANNELS, DCT_SPATIAL_DIMS

    print("=" * 60)
    print("  Mixed Dataset Smoke Test")
    print("=" * 60)

    train_ld, val_aigu_ld, val_ffpp_ld, _, _ = get_mixed_loaders()

    batch = next(iter(train_ld))
    print(f"\nBatch shapes:")
    print(f"  rgb    : {batch['rgb'].shape}")
    for s in DCT_SCALES:
        print(f"  dct[{s:2d}] : {batch['dct'][s].shape}  "
              f"(expected [B, {DCT_CHANNELS[s]}, {DCT_SPATIAL_DIMS[s]}, {DCT_SPATIAL_DIMS[s]}])")
    print(f"  label  : {batch['label']}")
    print(f"  domain : {batch['domain']}")


# =============================================================================
# SFCANet-v4 -- SOTA Cross-Domain Dataset Module
# =============================================================================
# Upgrades over v3:
#   1. Strict video-level split verification for FF++
#   2. advanced augmentation: RandomErasing, Gaussian noise simulation
#   3. idx tracking for Hard Negative Mining
#   4. Equal mix FF++ / AIGuard ratio (50/50 exposed via get_mixed_loaders_v4)
# =============================================================================

import time
from torch.utils.data import ConcatDataset

class AIGuardDatasetV4(Dataset):
    """AIGuard v4 dataset with idx tracking and advanced img padding/erasing."""
    def __init__(self, samples: list, augment: bool = True, start_idx: int = 0):
        self.samples = samples
        self.augment = augment
        self.start_idx = start_idx
        # Extended augmentation for v4
        import torchvision.transforms.v2 as transform_v2
        self.tf_basic = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]) if augment else transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        ])
        self.erasing = transform_v2.RandomErasing(p=0.2, scale=(0.02, 0.1)) if augment else None
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert('RGB')
        
        if self.augment:
            img = fourier_perturbation(img, p=0.3)
            img = jpeg_augment(img, quality_range=(60, 90), p=0.33)
            
        img = self.tf_basic(img)
        img_np = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
        
        # Add random gaussian noise simulating sensor noise
        if self.augment and random.random() < 0.33:
            noise = np.random.normal(0, np.random.uniform(5, 15), img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
        dct_np = compute_multiscale_dct(img_np, DCT_SCALES)
        dct_dict = {s: torch.from_numpy(dct_np[s]) for s in DCT_SCALES}

        rgb = self.norm(img)
        if self.erasing:
            rgb = self.erasing(rgb)

        return {
            'rgb':    rgb,
            'dct':    dct_dict,
            'label':  torch.tensor(label, dtype=torch.float32),
            'domain': torch.tensor(0, dtype=torch.float32), 
            'idx':    self.start_idx + i,  # For HNM
            'segment_id': path,            # For Segment-Aware Pooling
        }

class FFPPDatasetV4(Dataset):
    """FF++ v4 dataset with strict video-split, idx tracking, advanced augs."""
    def __init__(self, samples: list, augment: bool = True, start_idx: int = 0):
        self.samples = samples
        self.augment = augment
        self.start_idx = start_idx
        
        import torchvision.transforms.v2 as transform_v2
        self.tf_basic = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]) if augment else transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        ])
        self.erasing = transform_v2.RandomErasing(p=0.2, scale=(0.02, 0.1)) if augment else None
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        npy_path, frame_idx, label = self.samples[i]

        try:
            stacked = np.load(npy_path)
            frame_np = stacked[frame_idx]
        except Exception:
            frame_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        img = Image.fromarray(frame_np.astype(np.uint8)).convert('RGB')
        
        if self.augment:
            img = fourier_perturbation(img, p=0.3)
            img = jpeg_augment(img, quality_range=(60, 90), p=0.33)
            
        img = self.tf_basic(img)
        img_np = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
        
        if self.augment and random.random() < 0.33:
            noise = np.random.normal(0, np.random.uniform(5, 15), img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
        dct_np = compute_multiscale_dct(img_np, DCT_SCALES)
        dct_dict = {s: torch.from_numpy(dct_np[s]) for s in DCT_SCALES}

        rgb = self.norm(img)
        if self.erasing:
            rgb = self.erasing(rgb)

        return {
            'rgb':    rgb,
            'dct':    dct_dict,
            'label':  torch.tensor(label, dtype=torch.float32),
            'domain': torch.tensor(1, dtype=torch.float32), 
            'idx':    self.start_idx + i, # For HNM
            'segment_id': npy_path,       # For Segment-Aware Pooling
        }

# ─── Mixed Collate for v4.1 ───────────────────────────────────────────────────

def mixed_collate_v4_1(batch: list) -> dict:
    rgb    = torch.stack([b['rgb']    for b in batch])
    label  = torch.stack([b['label'] for b in batch])
    domain = torch.stack([b['domain'] for b in batch])
    
    dct_dict = {}
    for s in DCT_SCALES:
        dct_dict[s] = torch.stack([b['dct'][s] for b in batch])

    return {
        'rgb': rgb, 'dct': dct_dict, 'label': label, 'domain': domain,
        'idx': [b['idx'] for b in batch],
        'segment_id': [b['segment_id'] for b in batch]
    }


def get_mixed_loaders_v4():
    """Builds completely detached DataLoaders for SFCANet-v4 with strict splits."""
    from phase4_config import (
        PHYSICAL_BATCH_SIZE, EFFECTIVE_BATCH_SIZE,
        FFPP_MIX_RATIO, AIGU_TRAIN_RATIO, AIGU_VAL_RATIO,
        PHASE4_SPLITS_DIR
    )

    t0 = time.time()
    
    # 1. AIGuard
    agu_files = [f for f in os.listdir(AIGU_REAL) if f.endswith('.jpg')] + \
                [f for f in os.listdir(AIGU_FAKE) if f.endswith('.jpg')]
    # Use deterministic shuffle so cached splits match
    random.Random(42).shuffle(agu_files)
    
    n_agu = len(agu_files)
    n_train = int(n_agu * AIGU_TRAIN_RATIO)
    n_val   = int(n_agu * AIGU_VAL_RATIO)
    
    agu_train_files = agu_files[:n_train]
    agu_val_files   = agu_files[n_train : n_train+n_val]
    agu_test_files  = agu_files[n_train+n_val:]
    
    def _agu_to_samples(files):
        samples = []
        for f in files:
            is_real = f.startswith('real')
            p = os.path.join(AIGU_REAL if is_real else AIGU_FAKE, f)
            samples.append((p, 0 if is_real else 1))
        return samples

    agu_train = _agu_to_samples(agu_train_files)
    agu_val   = _agu_to_samples(agu_val_files)
    agu_test  = _agu_to_samples(agu_test_files)
    
    # 2. FFPP with strict video split verification
    # We collect samples directly from the pre-split directories
    from mixed_dataset import _collect_ffpp_samples
    
    t1 = time.time()
    train_ffpp = _collect_ffpp_samples(FFPP_TRAIN_DIR)
    val_ffpp   = _collect_ffpp_samples(FFPP_VAL_DIR)
    test_ffpp  = _collect_ffpp_samples(FFPP_TEST_DIR)

    # Ensure strict separation: no video that appears in Train can be in Val or Test
    train_video_names = set()
    for f_path, _, _ in train_ffpp:
        # Extract video name e.g. "000_003" from ".../original/000_003/seg_0.npy"
        vid_name = os.path.basename(os.path.dirname(f_path))
        train_video_names.add(vid_name)
        
    val_ffpp_clean = []
    val_leaks = set()
    for item in val_ffpp:
        vid_name = os.path.basename(os.path.dirname(item[0]))
        if vid_name in train_video_names:
            val_leaks.add(vid_name)
        else:
            val_ffpp_clean.append(item)
    val_ffpp = val_ffpp_clean
            
    test_ffpp_clean = []
    test_leaks = set()
    for item in test_ffpp:
        vid_name = os.path.basename(os.path.dirname(item[0]))
        if vid_name in train_video_names:
            test_leaks.add(vid_name)
        else:
            test_ffpp_clean.append(item)
    test_ffpp = test_ffpp_clean

    if val_leaks:
        print(f"[Data v4] Removed {len(val_leaks)} leaking videos from Val set.")
    if test_leaks:
        print(f"[Data v4] Removed {len(test_leaks)} leaking videos from Test set.")

    print(f"[Data v4] Verified FF++ strict split. "
          f"Train={len(train_ffpp)}, Val={len(val_ffpp)}, Test={len(test_ffpp)}")
    
    # 3. Create Datasets with unique start_idx for HNM
    ds_train_agu = AIGuardDatasetV4(agu_train, augment=True, start_idx=0)
    ds_train_ff  = FFPPDatasetV4(train_ffpp, augment=True, start_idx=len(agu_train))
    
    train_ds = ConcatDataset([ds_train_agu, ds_train_ff])
    
    # Weight calculation for 50/50 mix
    w_out = np.zeros(len(train_ds), dtype=np.float32)
    n_a = len(agu_train)
    n_f = len(train_ffpp)
    
    # 50% AIGuard, 50% FF++
    w_a = FFPP_MIX_RATIO / n_a
    w_f = (1.0 - FFPP_MIX_RATIO) / n_f
    
    w_out[:n_a] = w_a
    w_out[n_a:] = w_f
    
    sampler = WeightedRandomSampler(w_out.tolist(), num_samples=len(train_ds), replacement=True)
    
    num_workers = 2
    train_loader = DataLoader(
        train_ds, batch_size=PHYSICAL_BATCH_SIZE,
        sampler=sampler, num_workers=num_workers,
        collate_fn=mixed_collate, pin_memory=True,
        drop_last=True, persistent_workers=(num_workers > 0)
    )
    
    # Val/Test loaders (No tracking needed)
    # Using v4 dataset wrapper for matching ImageNet Normalisation
    ds_val_agu = AIGuardDatasetV4(agu_val, augment=False)
    val_agu_loader = DataLoader(ds_val_agu, batch_size=PHYSICAL_BATCH_SIZE*2, 
                                num_workers=num_workers, collate_fn=mixed_collate, pin_memory=True)
                                
    ds_val_ff = FFPPDatasetV4(val_ffpp, augment=False)
    val_ff_loader = DataLoader(ds_val_ff, batch_size=PHYSICAL_BATCH_SIZE*2,
                               num_workers=num_workers, collate_fn=mixed_collate, pin_memory=True)
                               
    ds_test_agu = AIGuardDatasetV4(agu_test, augment=False)
    test_agu_loader = DataLoader(ds_test_agu, batch_size=PHYSICAL_BATCH_SIZE*2,
                                num_workers=num_workers, collate_fn=mixed_collate, pin_memory=True)
                                
    ds_test_ff = FFPPDatasetV4(test_ffpp, augment=False)
    test_ff_loader = DataLoader(ds_test_ff, batch_size=PHYSICAL_BATCH_SIZE*2,
                               num_workers=num_workers, collate_fn=mixed_collate, pin_memory=True)

    print(f"[Data v4] Built in {time.time() - t0:.1f}s. Mix ratio = {FFPP_MIX_RATIO:.1f} FF++")
    return train_loader, val_agu_loader, val_ff_loader, test_agu_loader, test_ff_loader, w_out, train_ds


def get_mixed_loaders_v4_1(fold_idx=0, num_folds=5):
    """Builds DataLoaders for SFCANet-v4.1 using 5-Fold Group CV."""
    from phase4_config import (
        PHYSICAL_BATCH_SIZE, FFPP_MIX_RATIO
    )
    from sklearn.model_selection import KFold, GroupKFold

    t0 = time.time()
    
    # 1. AIGuard (Standard KFold)
    agu_files = [f for f in os.listdir(AIGU_REAL) if f.endswith('.jpg')] + \
                [f for f in os.listdir(AIGU_FAKE) if f.endswith('.jpg')]
    # Deterministic sort for stable folds
    agu_files.sort()
    agu_samples = []
    for f in agu_files:
        is_real = f.startswith('real')
        p = os.path.join(AIGU_REAL if is_real else AIGU_FAKE, f)
        agu_samples.append((p, 0 if is_real else 1))
        
    kf_agu = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)
    agu_train_idx, agu_val_idx = list(kf_agu.split(agu_samples))[fold_idx]
    agu_train = [agu_samples[i] for i in agu_train_idx]
    agu_val   = [agu_samples[i] for i in agu_val_idx]
    
    # 2. FF++ (GroupKFold by Video ID)
    # Collect all FF++ samples from all dirs (train/val/test) into one pool for proper CV
    all_ffpp = []
    for d in [FFPP_TRAIN_DIR, FFPP_VAL_DIR, FFPP_TEST_DIR]:
        all_ffpp.extend(_collect_ffpp_samples(d))
        
    # PROFESSIONAL SUBSAMPLING LOGIC (The "First Segment")
    # Reduces 227,000 frames to ~35,000 frames (1 segment per video)
    all_ffpp = [item for item in all_ffpp if os.path.basename(item[0]) == 'seg_0.npy']
        
    ffpp_groups = []
    for item in all_ffpp:
        # e.g., ".../original/000_003/seg_0.npy" -> "000_003"
        vid_name = os.path.basename(os.path.dirname(item[0]))
        ffpp_groups.append(vid_name)
        
    gkf_ffpp = GroupKFold(n_splits=num_folds)
    ffpp_train_idx, ffpp_val_idx = list(gkf_ffpp.split(all_ffpp, groups=ffpp_groups))[fold_idx]
    train_ffpp = [all_ffpp[i] for i in ffpp_train_idx]
    val_ffpp   = [all_ffpp[i] for i in ffpp_val_idx]

    # Verify no leak
    train_vids = set([os.path.basename(os.path.dirname(item[0])) for item in train_ffpp])
    val_vids   = set([os.path.basename(os.path.dirname(item[0])) for item in val_ffpp])
    assert len(train_vids.intersection(val_vids)) == 0, "GroupKFold leaked!"
    
    # 3. Create Datasets with unique start_idx for HNM
    ds_train_agu = AIGuardDatasetV4(agu_train, augment=True, start_idx=0)
    ds_train_ff  = FFPPDatasetV4(train_ffpp, augment=True, start_idx=len(agu_train))
    train_ds = ConcatDataset([ds_train_agu, ds_train_ff])
    
    # Weight calculation for 50/50 mix
    w_out = np.zeros(len(train_ds), dtype=np.float32)
    n_a, n_f = len(agu_train), len(train_ffpp)
    w_out[:n_a] = FFPP_MIX_RATIO / n_a
    w_out[n_a:] = (1.0 - FFPP_MIX_RATIO) / n_f
    
    sampler = WeightedRandomSampler(w_out.tolist(), num_samples=len(train_ds), replacement=True)
    
    num_workers = 2
    train_loader = DataLoader(
        train_ds, batch_size=PHYSICAL_BATCH_SIZE,
        sampler=sampler, num_workers=num_workers,
        collate_fn=mixed_collate_v4_1, pin_memory=True,
        drop_last=True, persistent_workers=(num_workers > 0)
    )
    
    # Val loaders
    ds_val_agu = AIGuardDatasetV4(agu_val, augment=False)
    val_agu_loader = DataLoader(ds_val_agu, batch_size=PHYSICAL_BATCH_SIZE*2, 
                                num_workers=num_workers, collate_fn=mixed_collate_v4_1, pin_memory=True)
                                
    ds_val_ff = FFPPDatasetV4(val_ffpp, augment=False)
    val_ff_loader = DataLoader(ds_val_ff, batch_size=PHYSICAL_BATCH_SIZE*2,
                               num_workers=num_workers, collate_fn=mixed_collate_v4_1, pin_memory=True)

    print(f"[CV Fold {fold_idx}] Built in {time.time() - t0:.1f}s. "
          f"Train: AIGuard={len(agu_train)}, FF++={len(train_ffpp)} | "
          f"Val: AIGuard={len(agu_val)}, FF++={len(val_ffpp)}")
          
    return train_loader, val_agu_loader, val_ff_loader, w_out, train_ds

