"""
Centralized Configuration for Deepfake Detection Baseline
==========================================================
Hardware target: NVIDIA RTX 3050 (4GB VRAM)
"""

import os
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "extracted_data")       # After RAR extraction
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")      # MTCNN-cropped faces
PROCESSED_REAL = os.path.join(PROCESSED_DIR, "real")
PROCESSED_FAKE = os.path.join(PROCESSED_DIR, "fake")
SPLITS_DIR = os.path.join(BASE_DIR, "splits")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─── Dataset Parameters ─────────────────────────────────────────────────────
NUM_REAL = 20_000
NUM_FAKE = 20_000
IMAGE_SIZE = 224
MTCNN_MARGIN = 20                  # Pixel margin around detected face
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# ─── Data Split Ratios ──────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ─── Training Hyperparameters ───────────────────────────────────────────────
PHYSICAL_BATCH_SIZE = 16           # What fits in 4GB VRAM
ACCUM_STEPS = 4                    # Gradient accumulation steps
EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * ACCUM_STEPS  # = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4                    # DataLoader workers
PIN_MEMORY = True

# ─── Model Definitions ──────────────────────────────────────────────────────
MODEL_NAMES = ['mobilenetv3_large', 'efficientnet_b0', 'resnet50']

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── ImageNet Normalization ──────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Create Directories ─────────────────────────────────────────────────────
for d in [PROCESSED_REAL, PROCESSED_FAKE, SPLITS_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[Config] Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[Config] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Config] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"[Config] Effective batch size: {EFFECTIVE_BATCH_SIZE} "
      f"(physical={PHYSICAL_BATCH_SIZE} × accum={ACCUM_STEPS})")
