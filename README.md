# SFCANet — Spatial-Frequency Cross-Attention Network for Deepfake Detection

> **Research Project** | Image DeepFake Detection using Multi-Scale DCT Frequency Analysis and Frequency-Guided Attention Fusion

---

## Overview

SFCANet is a novel deepfake detection architecture that fuses **spatial RGB features** with **multi-scale DCT frequency analysis** via a custom **Frequency-Guided Attention (FGA)** module. The model is designed for robust **cross-domain generalization** — trained on one dataset, it must detect deepfakes on a completely different distribution.

### Key Results (SFCANet-v4.1, Fold 0)

| Metric | Value |
|--------|-------|
| AIGuard AUC (in-domain) | **0.994** |
| FaceForensics++ AUC (cross-domain) | **0.889** |
| Combined AUC | **0.938** |
| Inference speed | **7.5 ms/image** (RTX 3050) |
| Parameters (trainable) | 29.2M |

> Combined AUC = √(AIGuard_AUC × FF++_AUC) — geometric mean penalises imbalance between domains.

---

## Architecture — SFCANet-v4

```
Input Image [B, 3, 224, 224]
        │
        ├─── RGB Branch
        │    ConvNeXt-Tiny (ImageNet pretrained)
        │    Stages 0–3: Frozen  │  Stages 4–7: Trainable
        │    Output: [B, 768, 7, 7]  (7×7 feature grid)
        │
        └─── Frequency Branch (MultiScaleDCTCNNv4)
             Block DCT at 3 scales:
               Scale 4  → [B, 48,  56, 56]  large blending artifacts
               Scale 8  → [B, 192, 28, 28]  GAN grid patterns
               Scale 16 → [B, 768, 14, 14]  fine texture residuals
             1×1 bottleneck → 3× SingleScaleDCTBranch (Conv+SE) → [B, 512]
                    │
                    ▼
         ┌─────────────────────────┐
         │  Frequency-Guided       │
         │  Attention (FGA)        │
         │                         │
         │  Channel gate [B,768,1,1] ← which feature channels are suspicious
         │  Spatial gate [B,1,7,7]   ← which face regions are suspicious
         │  Alpha scalar [B,1]       ← soft trust weight                     │
         │  Gating: Maps × (1 + α × ch_gate × sp_gate)                      │
         └─────────────────────────┘
                    │
             [B, 768]  (frequency-modulated spatial features)
                    │
         Classification Head:
         Dropout(0.5) → Linear(768→256) → LN → GELU
         Dropout(0.4) → Linear(256→128) → GELU
         Dropout(0.3) → Linear(128→1)
                    │
              Logit [B, 1]
```

### Why Two Branches?

- **RGB branch** sees what the face *looks like* — texture, colour, shape
- **DCT branch** sees mathematical *frequency anomalies* — block discontinuities and coefficient residuals left by face-swap algorithms that are invisible to the human eye
- **FGA fusion** lets frequency tell spatial *where* to look and *what* to amplify

---

## Datasets

### Dataset A — AIGuard (In-Domain)
- ~22,000 real + fake face images (JPEG, 224×224)
- GAN-generated and diffusion-model fake faces
- Split: 80% train / 10% val / 10% test (5-Fold KFold CV)

### Dataset B — FaceForensics++ (Cross-Domain)
- 1000 real videos + 5 manipulation methods × 1000 videos = 6000 total videos
- Methods: Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures
- Pre-extracted as 8-frame `.npy` segments
- **Strict video-level GroupKFold split** — no frames from the same video appear in both train and val

### Mixing Strategy
- **50% AIGuard / 50% FF++** per batch (WeightedRandomSampler)
- Ensures equal domain exposure every epoch regardless of dataset size differences

---

## Training Pipeline

### Augmentation
1. `RandomResizedCrop(224, scale=0.8–1.0)`
2. `RandomHorizontalFlip(p=0.5)`
3. `RandomRotation(±10°)`
4. `ColorJitter(brightness/contrast/saturation)`
5. `GaussianBlur(σ=0.1–2.0)`
6. **JPEG Re-compression (quality=40–95, p=0.6)** — breaks compression fingerprint bias
7. **Random Gaussian Noise (σ=5–15, p=0.3)** — sensor noise simulation
8. **RandomErasing (p=0.2)** — forces frequency features over skin texture
9. ImageNet Normalize

### Learning Rate Schedule
```
Epochs  1– 3:  Linear warmup  0 → 5e-5
Epochs  4–30:  CosineAnnealing 5e-5 → 1e-6
```

### Training Techniques
| Technique | Details |
|-----------|---------|
| **Gradient accumulation** | 4 physical × 8 accum = 32 effective batch |
| **Mixed precision** | `torch.amp.autocast` + `GradScaler` |
| **Gradient clipping** | `max_norm=1.0` |
| **Cost-sensitive loss** | `BCEWithLogitsLoss(pos_weight=5.0)` |
| **Hard Negative Mining** | From epoch 16: top-20% loss samples upweighted 1.5× |
| **SWA** | Stochastic Weight Averaging from epoch 26 |
| **Gradient checkpointing** | Reduces VRAM peak for RTX 3050 (4GB) |

### Evaluation
- **Segment-Aware Median Pooling** — all frames from the same video segment aggregated via median before AUC computation
- **EER threshold calibration** on val set for binary classification
- **Test-Time Augmentation (TTA)** — 5-view average for final test metrics

---

## Project Structure

```
Image DeepFake Detection/
├── sfcanet.py              # Model definitions: SFCANetV4, FGA, MultiScaleDCTCNNv4
├── sfcanet_v4_1_cv.py      # Main training script (fold-aware, resume support)
├── sfcanet_v4_1_cv_all.py  # Master script to run all 5 folds sequentially
├── sfcanet_v4_train.py     # Earlier v4 training script (no CV)
├── mixed_dataset.py        # Dataset loaders: AIGuard + FF++ with strict splits
├── phase4_config.py        # All hyperparameters (single source of truth)
├── multiscale_dct_cnn.py   # DCT transformation utilities
├── dct_transform.py        # Block DCT implementation
│
├── phase3_config.py        # v3 config (GRL-based, deprecated)
├── sfcanet_v3_train.py     # v3 training script (for comparison)
├── phase3_baselines.py     # Baseline model training (Xception, ConvNeXt, Swin-T)
│
├── phase4_results/         # Training logs (CSV, per epoch)
│   └── sfcanet_v4_1_cv0_log.csv
│
├── requirements.txt        # Python dependencies
└── Experimentation_Report.md
```

---

## Installation & Usage

### Requirements
```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `scikit-learn`, `Pillow`, `numpy`, `scipy`

### Running Training (Fold 0)
```bash
py -3.11 sfcanet_v4_1_cv.py
```
Automatically resumes from checkpoint if `phase4_checkpoints/sfcanet_v4_1_cv0_full.pth` exists.

### Running All 5 Folds
```bash
py -3.11 sfcanet_v4_1_cv_all.py
```

### Architecture Smoke Test
```bash
py -3.11 -c "
import torch
from sfcanet import SFCANetV4
model = SFCANetV4().cuda()
rgb = torch.randn(4, 3, 224, 224).cuda()
dct = {4: torch.randn(4,48,56,56).cuda(), 8: torch.randn(4,192,28,28).cuda(), 16: torch.randn(4,768,14,14).cuda()}
print(model(rgb, dct).shape)  # expect torch.Size([4, 1])
"
```

---

## Evolution of SFCANet

| Version | Backbone | Fusion | Domain Alignment | Best Combined AUC |
|---------|----------|--------|-----------------|-------------------|
| v1 | ResNet-50 | Concat | None | ~0.82 |
| v2 | ResNet-50 | BidirCrossAttn | None | ~0.85 |
| v3 | ResNet-50 | BidirCrossAttn | GRL (adversarial) | ~0.79 (unstable) |
| **v4 / v4.1** | **ConvNeXt-Tiny** | **FGA (channel+spatial gate)** | **Data mixing + JPEG aug** | **0.938** |

---

## Hardware

Trained on **NVIDIA RTX 3050 4GB Laptop GPU** with all memory optimisations enabled.  
~26 min/epoch, ~13 hours for 30-epoch full run.

---

## Citation

If you use this codebase or results in your research, please cite:

```
@misc{sfcanet2026,
  title  = {SFCANet: Spatial-Frequency Cross-Attention Network for Cross-Domain Deepfake Detection},
  author = {Saanvi Goel},
  year   = {2026},
}
```
