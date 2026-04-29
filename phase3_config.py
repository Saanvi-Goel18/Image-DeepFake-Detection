"""
Phase 3 Configuration — SFCANet-v3 Cross-Domain Training
=========================================================
SFCANet-v3: Multi-Scale DCT + Bidirectional Cross-Attention
           + Adversarial Domain Alignment (GRL)
Target: RTX 3050 (4GB VRAM)
"""

import os
import torch

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VIDEO_PROJECT = os.path.join(os.path.dirname(BASE_DIR), "Visual Images DeepFake Detection")

# ─── Data Sources ────────────────────────────────────────────────────────────
# AIGuard dataset (existing processed crops, 224x224 JPEGs)
AIGU_REAL = os.path.join(BASE_DIR, "processed_data", "real")
AIGU_FAKE = os.path.join(BASE_DIR, "processed_data", "fake")

# FaceForensics++ pre-extracted crops (from video project)
FFPP_TRAIN_DIR = os.path.join(VIDEO_PROJECT, "data_crops", "train")
FFPP_VAL_DIR   = os.path.join(VIDEO_PROJECT, "data_crops", "val")
FFPP_TEST_DIR  = os.path.join(VIDEO_PROJECT, "data_crops", "test")

# FF++ folder names
FFPP_REAL_DIR  = "original"
FFPP_FAKE_DIRS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

# ─── Output Directories ───────────────────────────────────────────────────────
PHASE3_CHECKPOINT_DIR = os.path.join(BASE_DIR, "phase3_checkpoints")
PHASE3_RESULTS_DIR    = os.path.join(BASE_DIR, "phase3_results")
PHASE3_SPLITS_DIR     = os.path.join(BASE_DIR, "phase3_splits")

for d in [PHASE3_CHECKPOINT_DIR, PHASE3_RESULTS_DIR, PHASE3_SPLITS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Multi-Scale DCT Settings ────────────────────────────────────────────────
IMAGE_SIZE        = 224
DCT_SCALES        = [4, 8, 16]               # Three granularity levels
DCT_CHANNELS      = {s: 3 * s * s for s in DCT_SCALES}
#   block=4  → 3×16  = 48 channels,  56×56 spatial
#   block=8  → 3×64  = 192 channels, 28×28 spatial
#   block=16 → 3×256 = 768 channels, 14×14 spatial
DCT_SPATIAL_DIMS  = {s: IMAGE_SIZE // s for s in DCT_SCALES}

FREQ_DIM_PER_SCALE = 256      # each scale branch output dim
FREQ_DIM_TOTAL     = FREQ_DIM_PER_SCALE * len(DCT_SCALES)  # 768 before projection
FREQ_DIM           = 512      # after multi-scale projection (matches v2 interface)

# ─── Architecture ────────────────────────────────────────────────────────────
SPATIAL_DIM         = 2048    # ResNet-50 global avg pool output
FUSION_DIM          = 512     # bidirectional cross-attention output
NUM_ATTENTION_HEADS = 8       # increased from v2's 4 for richer attention

# ─── Training Hyperparameters ─────────────────────────────────────────────────
PHYSICAL_BATCH_SIZE  = 4
ACCUM_STEPS          = 8
EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * ACCUM_STEPS   # 32 samples/step
NUM_EPOCHS           = 25          # extended for post-GRL recovery
LEARNING_RATE        = 5e-5        # slightly lower LR for better generalisation
LR_MIN               = 1e-7
WEIGHT_DECAY         = 2e-4        # stronger L2 to fight overfitting

# ─── Cost-Sensitive Learning ──────────────────────────────────────────────────
# Penalize false negatives (missed fakes) — same strategy as v2 pw=5
POS_WEIGHT = 5.0

# ─── Adversarial Domain Alignment (GRL) ──────────────────────────────────────
# lambda is annealed from 0 -> GRL_LAMBDA_MAX over training
# so domain alignment is gently introduced after classification is stable
GRL_LAMBDA_MAX    = 0.1       # FIXED: 10x gentler — prevents gradient explosion
GRL_WARMUP_EPOCHS = 5         # FIXED: longer warmup before domain loss starts
DOMAIN_LOSS_WEIGHT = 0.1      # FIXED: classification loss stays dominant

# ─── Dataset Mixing ───────────────────────────────────────────────────────────
# Approximate target: 65% AIGuard images, 35% FF++ frames per effective batch
FFPP_MIX_RATIO = 0.35         # fraction of training samples from FF++

# AIGuard split ratios (for internal validation set)
AIGU_TRAIN_RATIO = 0.80
AIGU_VAL_RATIO   = 0.10

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── ImageNet Normalisation ───────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Diagnostics ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[Phase3 Config] Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[Phase3 Config] GPU    : {torch.cuda.get_device_name(0)}")
        print(f"[Phase3 Config] VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"[Phase3 Config] DCT scales  : {DCT_SCALES}")
    for s in DCT_SCALES:
        print(f"  block={s:2d}: {DCT_CHANNELS[s]:3d} channels × "
              f"{DCT_SPATIAL_DIMS[s]}×{DCT_SPATIAL_DIMS[s]} spatial")
    print(f"[Phase3 Config] AIGuard real : {AIGU_REAL}")
    print(f"[Phase3 Config] AIGuard fake : {AIGU_FAKE}")
    print(f"[Phase3 Config] FF++ train   : {FFPP_TRAIN_DIR}")
    print(f"[Phase3 Config] Mix ratio    : {FFPP_MIX_RATIO*100:.0f}% FF++ / "
          f"{(1-FFPP_MIX_RATIO)*100:.0f}% AIGuard")
