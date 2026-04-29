"""
Multi-Scale DCT CNN — Frequency Branch for SFCANet-v3
======================================================
Captures deepfake artifacts at three frequency granularities simultaneously:

  block=4  → 48ch × 56×56  → 256-dim  (large-scale blending artifacts, FaceSwap)
  block=8  → 192ch × 28×28 → 256-dim  (mid-scale GAN grid noise, Deepfakes)
  block=16 → 768ch × 14×14 → 256-dim  (fine texture artifacts, NeuralTextures)

All three → concat 768-dim → LayerNorm → Linear → 512-dim final embedding.

This multi-granularity approach is key to cross-dataset generalization:
different deepfake methods leave artifacts at different frequency scales, and by
capturing all three we avoid over-specializing to one generation pipeline.
"""

import torch
import torch.nn as nn

from phase3_config import (
    DCT_SCALES, DCT_CHANNELS, FREQ_DIM_PER_SCALE, FREQ_DIM_TOTAL, FREQ_DIM
)


class SqueezeExcitation(nn.Module):
    """Channel attention block for frequency band weighting."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        w = self.excite(self.pool(x)).view(B, C, 1, 1)
        return x * w


class SingleScaleDCTBranch(nn.Module):
    """
    Lightweight CNN for one DCT block-size scale.
    Adapts its first-layer width to the (variable) number of input channels.
    """

    def __init__(self, in_channels: int, out_dim: int = FREQ_DIM_PER_SCALE):
        super().__init__()
        # Clamp intermediate width to a sane range regardless of scale
        mid = min(max(in_channels, 64), 256)

        self.conv = nn.Sequential(
            # Stage 1 — adapt channel count
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),

            # Stage 2 — spatial → halve
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Stage 3 — project to out_dim
            nn.Conv2d(mid, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.se = SqueezeExcitation(out_dim, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C_s, H_s, W_s] → returns [B, out_dim]"""
        x = self.conv(x)
        x = self.se(x)
        return self.pool(x).flatten(1)


class MultiScaleDCTCNN(nn.Module):
    """
    Three parallel DCT branches (scales 4, 8, 16) → fused 512-dim embedding.

    Input:  dict {block_size (int): Tensor [B, C_s, H_s, W_s]}
    Output: Tensor [B, 512]
    """

    def __init__(self, out_dim: int = FREQ_DIM):
        super().__init__()
        self.scales = DCT_SCALES  # [4, 8, 16]

        self.branches = nn.ModuleDict({
            str(s): SingleScaleDCTBranch(
                in_channels=DCT_CHANNELS[s],
                out_dim=FREQ_DIM_PER_SCALE,
            )
            for s in self.scales
        })

        # Project 768-dim concat → 512-dim
        self.project = nn.Sequential(
            nn.LayerNorm(FREQ_DIM_TOTAL),
            nn.Linear(FREQ_DIM_TOTAL, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"[MultiScaleDCTCNN] Scales: {self.scales} | "
              f"Branch dim: {FREQ_DIM_PER_SCALE} each | "
              f"Output: {out_dim}-dim | Params: {total:,}")

    def forward(self, dct_dict: dict) -> torch.Tensor:
        """
        Args:
            dct_dict: {block_size: Tensor [B, C_s, H_s, W_s]}
                      must contain keys for each scale in self.scales
        Returns:
            Tensor [B, 512]
        """
        feats = [self.branches[str(s)](dct_dict[s]) for s in self.scales]
        return self.project(torch.cat(feats, dim=-1))


# ─── quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from phase3_config import DCT_CHANNELS, DCT_SPATIAL_DIMS, DEVICE

    model = MultiScaleDCTCNN().to(DEVICE)

    # Build a dummy multi-scale DCT dict
    B = 4
    dct_dummy = {
        s: torch.randn(B, DCT_CHANNELS[s], DCT_SPATIAL_DIMS[s], DCT_SPATIAL_DIMS[s],
                       device=DEVICE)
        for s in [4, 8, 16]
    }

    out = model(dct_dummy)
    print(f"Input scales : {[(s, dct_dummy[s].shape) for s in [4, 8, 16]]}")
    print(f"Output shape : {out.shape}")  # expect [4, 512]
