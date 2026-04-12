"""
Step 1: Dataset Preparation
============================
1. Extract AIGuard_Deepfake_Dataset.rar (already done — skips if extracted)
2. Discover & sub-sample 20k real + 20k fake images
3. MTCNN face crop to 224×224 (with resize fallback for pre-cropped faces)
4. Create stratified train/val/test CSV splits

Usage:
    python prepare_dataset.py
"""

import os
import sys
import random
import subprocess
import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
from config import (
    BASE_DIR, RAW_DATA_DIR, PROCESSED_REAL, PROCESSED_FAKE, SPLITS_DIR,
    NUM_REAL, NUM_FAKE, IMAGE_SIZE, MTCNN_MARGIN, IMAGE_EXTENSIONS,
    TRAIN_RATIO, VAL_RATIO, SEED
)


def extract_rar():
    """Extract the RAR dataset if not already extracted."""
    rar_path = os.path.join(BASE_DIR, "AIGuard_Deepfake_Dataset.rar")

    if os.path.exists(RAW_DATA_DIR) and len(os.listdir(RAW_DATA_DIR)) > 0:
        print(f"[Extract] Already extracted at: {RAW_DATA_DIR}")
        return True

    if not os.path.exists(rar_path):
        print(f"[Extract] ERROR: RAR file not found at {rar_path}")
        return False

    print(f"[Extract] Extracting {rar_path} → {RAW_DATA_DIR}")
    print("[Extract] This may take several minutes for ~10.7GB...")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Try unrar first, then 7z as fallback
    try:
        result = subprocess.run(
            ["unrar", "x", "-o+", rar_path, RAW_DATA_DIR + os.sep],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode == 0:
            print("[Extract] ✓ Extraction complete (unrar)")
            return True
        else:
            print(f"[Extract] unrar failed: {result.stderr[:200]}")
    except FileNotFoundError:
        print("[Extract] unrar not found, trying 7z...")

    # Fallback: 7-Zip
    seven_zip_paths = [
        "7z",
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    for sz_path in seven_zip_paths:
        try:
            result = subprocess.run(
                [sz_path, "x", rar_path, f"-o{RAW_DATA_DIR}", "-y"],
                capture_output=True, text=True, timeout=3600
            )
            if result.returncode == 0:
                print(f"[Extract] ✓ Extraction complete (7z: {sz_path})")
                return True
        except FileNotFoundError:
            continue

    print("[Extract] ERROR: Could not extract.")
    return False


def find_images_recursive(root_dir, label_name):
    """
    Recursively find all image files under root_dir matching label_name.
    Handles nested structures like:
      extracted_data/AIGuard_Deepfake_Dataset/real/real/*.jpg
    """
    # Walk entire tree to find directories named label_name
    candidate_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        basename = os.path.basename(dirpath).lower()
        if basename == label_name.lower():
            candidate_dirs.append(dirpath)

    if not candidate_dirs:
        print(f"[Discover] ERROR: No '{label_name}/' directory found under {root_dir}")
        return []

    # Pick the deepest directory that contains actual image files
    best_dir = None
    best_count = 0
    for d in candidate_dirs:
        img_count = sum(
            1 for f in os.listdir(d)
            if os.path.isfile(os.path.join(d, f))
            and Path(f).suffix.lower() in IMAGE_EXTENSIONS
        )
        if img_count > best_count:
            best_count = img_count
            best_dir = d

    if best_dir is None or best_count == 0:
        # Fallback: use the first candidate and walk recursively
        best_dir = candidate_dirs[0]
        print(f"[Discover] No direct images found, walking recursively: {best_dir}")

    print(f"[Discover] Using directory: {best_dir}")

    images = []
    for dirpath, _, filenames in os.walk(best_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(dirpath, fname))

    print(f"[Discover] Found {len(images):,} {label_name} images")
    return images


def subsample(image_list, n, label_name):
    """Randomly subsample n images from the list."""
    if len(image_list) < n:
        print(f"[Subsample] WARNING: Only {len(image_list)} {label_name} images "
              f"available (requested {n}). Using all.")
        return image_list

    random.seed(SEED)
    sampled = random.sample(image_list, n)
    print(f"[Subsample] Selected {len(sampled):,} {label_name} images")
    return sampled


def crop_faces(image_paths, output_dir, label):
    """
    Use MTCNN to detect and crop faces from images.
    Falls back to simple resize for pre-cropped face datasets.
    """
    try:
        from facenet_pytorch import MTCNN
        import torch
    except ImportError:
        print("[MTCNN] ERROR: facenet-pytorch not installed.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MTCNN] Using device: {device}")

    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=MTCNN_MARGIN,
        keep_all=False,
        post_process=False,   # Return raw pixel values [0, 255]
        device=device
    )

    os.makedirs(output_dir, exist_ok=True)
    success_mtcnn = 0
    success_resize = 0
    failed = 0
    saved_paths = []

    for idx, img_path in enumerate(tqdm(image_paths, desc=f"Cropping {label}")):
        out_name = f"{label}_{idx:06d}.jpg"
        out_path = os.path.join(output_dir, out_name)

        # Skip if already processed
        if os.path.exists(out_path):
            saved_paths.append(out_path)
            success_mtcnn += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")

            # Try MTCNN face detection
            face_saved = False
            try:
                face = mtcnn(img)
                if face is not None:
                    # face is tensor [3, 224, 224] with values ~[0, 255]
                    face_np = face.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
                    face_img = Image.fromarray(face_np)
                    face_img.save(out_path, "JPEG", quality=95)
                    saved_paths.append(out_path)
                    success_mtcnn += 1
                    face_saved = True
            except Exception:
                pass  # Fall through to resize fallback

            # Fallback: simple resize (these are already face images)
            if not face_saved:
                img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                img_resized.save(out_path, "JPEG", quality=95)
                saved_paths.append(out_path)
                success_resize += 1

        except Exception as e:
            failed += 1
            if failed <= 5:
                tqdm.write(f"[ERROR] {img_path}: {e}")

    print(f"[MTCNN] {label}: {success_mtcnn} MTCNN-cropped, "
          f"{success_resize} resized (fallback), {failed} failed")
    return saved_paths


def create_splits(real_paths, fake_paths):
    """Create stratified train/val/test CSV splits."""
    random.seed(SEED)

    # Build dataset: (path, label) — 0=real, 1=fake
    dataset = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    random.shuffle(dataset)

    n = len(dataset)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train + n_val]
    test_data = dataset[n_train + n_val:]

    splits = {
        "train.csv": train_data,
        "val.csv": val_data,
        "test.csv": test_data
    }

    for fname, data in splits.items():
        csv_path = os.path.join(SPLITS_DIR, fname)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])
            writer.writerows(data)
        n_real = sum(1 for _, l in data if l == 0)
        n_fake = sum(1 for _, l in data if l == 1)
        print(f"[Split] {fname}: {len(data)} samples (real={n_real}, fake={n_fake})")


def main():
    print("=" * 70)
    print("  DEEPFAKE DETECTION — DATASET PREPARATION")
    print("=" * 70)

    # Step 1: Extract
    print("\n[Step 1/4] Extracting dataset...")
    if not extract_rar():
        sys.exit(1)

    # Step 2: Discover images
    print("\n[Step 2/4] Discovering images...")
    real_images = find_images_recursive(RAW_DATA_DIR, "real")
    fake_images = find_images_recursive(RAW_DATA_DIR, "fake")

    if not real_images or not fake_images:
        print("[ERROR] Could not find images. Check extraction and folder structure.")
        print(f"[DEBUG] Top-level contents of {RAW_DATA_DIR}:")
        for item in os.listdir(RAW_DATA_DIR)[:20]:
            full = os.path.join(RAW_DATA_DIR, item)
            label = "DIR" if os.path.isdir(full) else "FILE"
            print(f"  {label}: {item}")
        sys.exit(1)

    # Step 3: Subsample
    print("\n[Step 3/4] Subsampling...")
    real_sampled = subsample(real_images, NUM_REAL, "real")
    fake_sampled = subsample(fake_images, NUM_FAKE, "fake")

    # Step 4: MTCNN face cropping
    print("\n[Step 4/4] MTCNN face cropping + resize fallback...")
    print(f"  (Dataset images are already {256}×{256} or {512}×{512} face crops)")
    print(f"  MTCNN will refine crops; resize used if MTCNN finds no face)")
    real_cropped = crop_faces(real_sampled, PROCESSED_REAL, "real")
    fake_cropped = crop_faces(fake_sampled, PROCESSED_FAKE, "fake")

    # Create splits
    print("\n[Splits] Creating train/val/test splits...")
    create_splits(real_cropped, fake_cropped)

    print("\n" + "=" * 70)
    print("  ✓ DATASET PREPARATION COMPLETE")
    print(f"    Real: {len(real_cropped):,} | Fake: {len(fake_cropped):,}")
    print(f"    Total: {len(real_cropped) + len(fake_cropped):,} images")
    print(f"    Saved to: {PROCESSED_REAL}")
    print(f"              {PROCESSED_FAKE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
