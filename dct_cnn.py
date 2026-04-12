"""
Standalone DCT-CNN (Frequency Branch)
======================================
4-layer CNN operating on 192-channel DCT coefficient maps (28×28).
Serves as: (a) standalone frequency baseline, (b) frequency branch inside SFCANet.

Architecture:
    Input: [B, 192, 28, 28] — DCT coefficients
    Conv1: 192 → 128, 3×3, BN, ReLU
    Conv2: 128 → 256, 3×3, BN, ReLU, MaxPool(2)  → 14×14
    Conv3: 256 → 512, 3×3, BN, ReLU, MaxPool(2)  → 7×7
    Conv4: 512 → 512, 3×3, BN, ReLU
    GAP:   AdaptiveAvgPool → 512-dim
    Head:  512 → 1 (for standalone mode)

Usage:
    from dct_cnn import DCTCNN
    model = DCTCNN(standalone=True)   # With classification head
    model = DCTCNN(standalone=False)  # Feature extractor only (for SFCANet)
"""

import torch
import torch.nn as nn

from phase2_config import DCT_CHANNELS, FREQ_DIM


class SqueezeExcitation(nn.Module):
    """Channel attention (SE block) for frequency band selection."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.squeeze(x).view(B, C)
        w = self.excite(w).view(B, C, 1, 1)
        return x * w


class DCTCNN(nn.Module):
    """
    Lightweight 4-layer CNN for DCT coefficient maps.

    Args:
        standalone: If True, includes classification head (512 → 1).
                    If False, returns 512-dim features for SFCANet fusion.
    """

    def __init__(self, standalone=True):
        super().__init__()
        self.standalone = standalone

        self.features = nn.Sequential(
            # Block 1: 192 → 128, 28×28 → 28×28
            nn.Conv2d(DCT_CHANNELS, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 2: 128 → 256, 28×28 → 14×14
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 256 → 512, 14×14 → 7×7
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 512 → 512, 7×7 → 7×7 + SE attention
            nn.Conv2d(512, FREQ_DIM, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(FREQ_DIM),
            nn.ReLU(inplace=True),
        )

        self.se = SqueezeExcitation(FREQ_DIM, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)

        if self.standalone:
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(FREQ_DIM, 1)
            )

        # Count params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[DCT-CNN] Parameters: {total:,} total, {trainable:,} trainable "
              f"({'standalone' if standalone else 'feature extractor'})")

    def forward(self, dct_input):
        """
        Args:
            dct_input: [B, 192, 28, 28] DCT coefficient maps
        Returns:
            standalone=True:  [B, 1] logit
            standalone=False: [B, 512] feature vector
        """
        x = self.features(dct_input)   # [B, 512, 7, 7]
        x = self.se(x)                 # [B, 512, 7, 7] (channel-reweighted)
        x = self.pool(x).flatten(1)    # [B, 512]

        if self.standalone:
            return self.classifier(x)  # [B, 1]
        return x                       # [B, 512]


if __name__ == "__main__":
    """Quick test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standalone mode
    model_s = DCTCNN(standalone=True).to(device)
    x = torch.randn(4, DCT_CHANNELS, 28, 28, device=device)
    out = model_s(x)
    print(f"Standalone → Input: {x.shape}, Output: {out.shape}")

    # Feature extractor mode
    model_f = DCTCNN(standalone=False).to(device)
    feat = model_f(x)
    print(f"Feature ext → Input: {x.shape}, Output: {feat.shape}")
