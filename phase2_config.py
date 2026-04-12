"""
Phase 2 Configuration
======================
SFCANet: Spatial-Frequency Cross-Attention Network
Cost-Sensitive Frequency Learning on RTX 3050 (4GB VRAM)
"""

import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Phase 2 Directories ────────────────────────────────────────────────────
PHASE2_CHECKPOINT_DIR = os.path.join(BASE_DIR, "phase2_checkpoints")
PHASE2_RESULTS_DIR = os.path.join(BASE_DIR, "phase2_results")
PHASE2_SPLITS_DIR = os.path.join(BASE_DIR, "phase2_splits")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
PROCESSED_REAL = os.path.join(PROCESSED_DIR, "real")
PROCESSED_FAKE = os.path.join(PROCESSED_DIR, "fake")

# Phase 1 checkpoint (frozen spatial backbone)
PHASE1_RESNET50_CKPT = os.path.join(BASE_DIR, "phase1_checkpoints", "resnet50_best.pth")

for d in [PHASE2_CHECKPOINT_DIR, PHASE2_RESULTS_DIR, PHASE2_SPLITS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── DCT Parameters ─────────────────────────────────────────────────────────
DCT_BLOCK_SIZE = 8                 # 8×8 block-wise DCT (JPEG-aligned)
IMAGE_SIZE = 224                   # Input image resolution
DCT_SPATIAL_DIM = IMAGE_SIZE // DCT_BLOCK_SIZE  # 28×28 coefficient maps
DCT_CHANNELS = 3 * (DCT_BLOCK_SIZE ** 2)        # 3 × 64 = 192

# ─── Imbalanced Dataset ─────────────────────────────────────────────────────
# 1:9 ratio → 10% Fake, 90% Real
IMBALANCE_RATIO = 9                # real:fake = 9:1
NUM_REAL_IMBA = 20_000             # Use all real images
NUM_FAKE_IMBA = NUM_REAL_IMBA // IMBALANCE_RATIO  # ≈ 2,222
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ─── Training Hyperparameters ───────────────────────────────────────────────
PHYSICAL_BATCH_SIZE = 4   # Reduced for CPU
ACCUM_STEPS = 8           # Increased to maintain effective batch size
EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * ACCUM_STEPS  # 32
NUM_EPOCHS = 5            # Reduced for faster CPU training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# ─── Punishment Parameter (WBCE) ────────────────────────────────────────────
# pos_weight > 1 penalizes False Negatives (missed fakes) more heavily
POS_WEIGHT_NONE = 1.0             # No punishment (baseline)
POS_WEIGHT_PUNISH = 5.0           # Heavy punishment for missing fakes
POS_WEIGHTS = [POS_WEIGHT_NONE, POS_WEIGHT_PUNISH]

# ─── Architecture Dimensions ────────────────────────────────────────────────
SPATIAL_DIM = 2048                 # ResNet-50 output
FREQ_DIM = 512                    # DCT-CNN output
FUSION_DIM = 512                  # After cross-attention projection
NUM_ATTENTION_HEADS = 4           # Multi-head cross-attention

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── ImageNet Normalization ──────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

print(f"[Phase2 Config] Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[Phase2 Config] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Phase2 Config] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"[Phase2 Config] DCT: {DCT_BLOCK_SIZE}x{DCT_BLOCK_SIZE} blocks -> "
      f"{DCT_CHANNELS} channels x {DCT_SPATIAL_DIM}x{DCT_SPATIAL_DIM}")
print(f"[Phase2 Config] Imbalance: {NUM_REAL_IMBA} real : {NUM_FAKE_IMBA} fake "
      f"({NUM_FAKE_IMBA/(NUM_REAL_IMBA+NUM_FAKE_IMBA)*100:.1f}% fake)")
