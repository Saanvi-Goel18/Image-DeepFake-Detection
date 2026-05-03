# SFCANet — Spatial-Frequency Cross-Attention Network for Deepfake Detection

A deepfake detection model that fuses spatial RGB features with multi-scale DCT frequency analysis via Frequency-Guided Attention (FGA) for robust cross-domain generalization.

## Key Files

| File | Description |
|------|-------------|
| `sfcanet.py` | Model architecture (SFCANetV4, FGA, MultiScaleDCTCNNv4) |
| `mixed_dataset.py` | Data pipeline — AIGuard + FF++ with strict video-level splits |
| `phase4_config.py` | All hyperparameters |
| `sfcanet_v4_1_cv.py` | Training script (fold-aware, auto-resume) |
| `sfcanet_v4_1_cv_all.py` | Run all 5 CV folds |
| `phase3_baselines.py` | Benchmark comparisons (Xception, ConvNeXt, Swin-T) |

## Usage

```bash
pip install -r requirements.txt
py -3.11 sfcanet_v4_1_cv.py
```
