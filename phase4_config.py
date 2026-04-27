"""
Phase 4 Configuration — SFCANet-v4 SOTA Cross-Domain Training
==============================================================
SFCANet-v4 upgrades:
  - ConvNeXt-Tiny backbone (768-dim, stochastic depth 0.1)
  - Frequency-Guided Attention (FGA) replaces BidirCrossAttn
  - 1x1 pointwise bottleneck in DCT branch
  - No GRL — simple BCE, data-mixing does domain alignment
  - Hard Negative Mining from epoch 12
  - 50% FF++ / 50% AIGuard training mix
  - TTA + EER threshold calibration at evaluation
Target: RTX 3050 (4GB VRAM)
"""

import os
import torch

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
VIDEO_PROJECT = os.path.join(os.path.dirname(BASE_DIR), "Visual Images DeepFake Detection")

# ── Data Sources ──────────────────────────────────────────────────────────────
AIGU_REAL = os.path.join(BASE_DIR, "processed_data", "real")
AIGU_FAKE = os.path.join(BASE_DIR, "processed_data", "fake")

FFPP_TRAIN_DIR = os.path.join(VIDEO_PROJECT, "data_crops", "train")
FFPP_VAL_DIR   = os.path.join(VIDEO_PROJECT, "data_crops", "val")
FFPP_TEST_DIR  = os.path.join(VIDEO_PROJECT, "data_crops", "test")
FFPP_REAL_DIR  = "original"
FFPP_FAKE_DIRS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

# ── Output Directories ────────────────────────────────────────────────────────
PHASE4_CHECKPOINT_DIR = os.path.join(BASE_DIR, "phase4_checkpoints")
PHASE4_RESULTS_DIR    = os.path.join(BASE_DIR, "phase4_results")
PHASE4_SPLITS_DIR     = os.path.join(BASE_DIR, "phase3_splits")   # reuse cached splits

for d in [PHASE4_CHECKPOINT_DIR, PHASE4_RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Multi-Scale DCT ───────────────────────────────────────────────────────────
IMAGE_SIZE         = 224
DCT_SCALES         = [4, 8, 16]
DCT_CHANNELS       = {s: 3 * s * s for s in DCT_SCALES}
DCT_SPATIAL_DIMS   = {s: IMAGE_SIZE // s for s in DCT_SCALES}

# DCT bottleneck: 1x1 conv reduces raw DCT channels before feature extraction
DCT_BOTTLENECK_CH  = 64          # uniform bottleneck dim for all scales
FREQ_DIM_PER_SCALE = 256         # each SingleScaleDCTBranch output dim
FREQ_DIM_TOTAL     = FREQ_DIM_PER_SCALE * len(DCT_SCALES)    # 768
FREQ_DIM           = 512         # final Multi-Scale DCT embedding

# ── Architecture ──────────────────────────────────────────────────────────────
SPATIAL_DIM         = 768    # ConvNeXt-Tiny global pool output
FGA_SPATIAL_SIZE    = 7      # ConvNeXt feature map HxW before avgpool
FUSION_DIM          = 768    # FGA output = spatial_dim (gated, then pooled)
NUM_ATTENTION_HEADS = 8      # kept for compatibility, not used in FGA
STOCHASTIC_DEPTH    = 0.1    # ConvNeXt stochastic depth probability

# ── Training Hyperparameters ──────────────────────────────────────────────────
PHYSICAL_BATCH_SIZE  = 4
ACCUM_STEPS          = 8
EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * ACCUM_STEPS    # 32
NUM_EPOCHS           = 30
LEARNING_RATE        = 5e-5
LR_MIN               = 1e-6
WEIGHT_DECAY         = 1e-4
WARMUP_EPOCHS        = 3     # linear LR warmup before cosine decay

# ── Cost-Sensitive Learning ───────────────────────────────────────────────────
POS_WEIGHT = 5.0    # penalise false negatives 5x

# ── Hard Negative Mining ──────────────────────────────────────────────────────
HARD_NEG_MINING_EPOCH  = 16   # HNM activates from this epoch onward
HARD_NEG_TOP_FRACTION  = 0.20  # top 20% highest-loss samples
HARD_NEG_WEIGHT_MULT   = 1.5   # upweight hard negatives by this factor

# ── Dataset Mixing ────────────────────────────────────────────────────────────
FFPP_MIX_RATIO   = 0.50    # 50% FF++ / 50% AIGuard (equal domain exposure)
AIGU_TRAIN_RATIO = 0.80
AIGU_VAL_RATIO   = 0.10

# ── TTA ───────────────────────────────────────────────────────────────────────
TTA_N_AUGMENTS = 5    # number of augmented views averaged at test time

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── ImageNet Normalisation ────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Diagnostics ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[Phase4 Config] Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[Phase4 Config] GPU    : {torch.cuda.get_device_name(0)}")
        print(f"[Phase4 Config] VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"[Phase4 Config] Backbone   : ConvNeXt-Tiny ({SPATIAL_DIM}-dim)")
    print(f"[Phase4 Config] Fusion     : Frequency-Guided Attention (FGA)")
    print(f"[Phase4 Config] FF++ mix   : {FFPP_MIX_RATIO*100:.0f}%")
    print(f"[Phase4 Config] HNM from   : epoch {HARD_NEG_MINING_EPOCH}")
    print(f"[Phase4 Config] TTA views  : {TTA_N_AUGMENTS}")
