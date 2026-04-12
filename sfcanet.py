"""
SFCANet-v2 — Spatial-Frequency Cross-Attention Network (Improved)
===================================================================
Frozen ResNet-50 (spatial) + DCT-CNN (frequency) + Bidirectional Cross-Attention.

Improvements over v1:
  1. Bidirectional cross-attention: spatial→freq AND freq→spatial
  2. Concatenated outputs + learned gating → decide which branch to trust
  3. Unfrozen ResNet layer4 (last 2 bottleneck blocks) for deepfake-specific features

Architecture:
    Spatial Branch:  ResNet-50 (layer4 unfrozen) → 2048-dim
    Frequency Branch: DCT-CNN → 512-dim (learnable)
    Fusion:          Bidirectional cross-attention + concat + gate → 1024-dim
    Head:            1024 → 512 → 256 → 1

Usage:
    from sfcanet import SFCANet
    model = SFCANet(resnet50_ckpt_path="phase1_checkpoints/resnet50_best.pth")
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torchvision import models

from phase2_config import (
    SPATIAL_DIM, FREQ_DIM, FUSION_DIM, NUM_ATTENTION_HEADS, PHASE1_RESNET50_CKPT
)
from dct_cnn import DCTCNN


class MultiHeadAttention(nn.Module):
    """Single-direction multi-head attention as a reusable module."""

    def __init__(self, q_dim, kv_dim, fusion_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = fusion_dim // num_heads
        assert fusion_dim % num_heads == 0

        self.q_proj = nn.Linear(q_dim, fusion_dim)
        self.k_proj = nn.Linear(kv_dim, fusion_dim)
        self.v_proj = nn.Linear(kv_dim, fusion_dim)
        self.out_proj = nn.Linear(fusion_dim, fusion_dim)
        self.layer_norm_q = nn.LayerNorm(fusion_dim)
        self.layer_norm_kv = nn.LayerNorm(fusion_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, q_input, kv_input):
        B = q_input.size(0)
        Q = self.layer_norm_q(self.q_proj(q_input)).unsqueeze(1)
        K = self.layer_norm_kv(self.k_proj(kv_input)).unsqueeze(1)
        V = self.v_proj(kv_input).unsqueeze(1)

        Q = Q.view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, 1, -1)
        return self.out_proj(out).squeeze(1)


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention with gating:
      - Direction 1: spatial=Q → freq=K,V  (spatial queries inspect frequency)
      - Direction 2: freq=Q → spatial=K,V (frequency queries inspect spatial)
      - Learnable gate: decide per-sample weight between branch outputs
      - Concatenate both directions + original features → richer representation
    """

    def __init__(self, spatial_dim, freq_dim, fusion_dim, num_heads):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.freq_dim = freq_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads

        self.attn_s2f = MultiHeadAttention(
            q_dim=spatial_dim, kv_dim=freq_dim, fusion_dim=fusion_dim, num_heads=num_heads
        )
        self.attn_f2s = MultiHeadAttention(
            q_dim=freq_dim, kv_dim=spatial_dim, fusion_dim=fusion_dim, num_heads=num_heads
        )

        self.spatial_proj = nn.Linear(spatial_dim, fusion_dim)
        self.spatial_gate = nn.Sequential(
            nn.Linear(spatial_dim, 1),
            nn.Sigmoid()
        )
        self.freq_gate = nn.Sequential(
            nn.Linear(freq_dim, 1),
            nn.Sigmoid()
        )

        combined_dim = fusion_dim + freq_dim + fusion_dim * 2
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )

    def forward(self, spatial_feat, freq_feat):
        attn_s2f = self.attn_s2f(spatial_feat, freq_feat)
        attn_f2s = self.attn_f2s(freq_feat, spatial_feat)

        g_s = self.spatial_gate(spatial_feat)
        g_f = self.freq_gate(freq_feat)

        spatial_proj = self.spatial_proj(spatial_feat)
        combined = torch.cat([
            g_s * spatial_proj,
            g_f * freq_feat,
            attn_s2f,
            attn_f2s,
        ], dim=-1)

        return self.fusion_proj(combined)


class SFCANet(nn.Module):
    """
    Spatial-Frequency Cross-Attention Network (v2).

    Combines a ResNet-50 spatial backbone (layer4 unfrozen) with a learnable
    DCT-CNN frequency branch via bidirectional cross-attention fusion.
    """

    def __init__(self, resnet50_ckpt_path=None, use_gradient_checkpoint=True,
                 unfreeze_layer4=True):
        super().__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        # ─── Spatial Branch (ResNet-50) ─────────────────────────────────
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        resnet.fc = nn.Identity()

        if resnet50_ckpt_path and torch.cuda.is_available():
            try:
                ckpt = torch.load(resnet50_ckpt_path, map_location='cpu', weights_only=False)
                state_dict = ckpt['model_state_dict']
                state_dict = {k: v for k, v in state_dict.items()
                              if not k.startswith('fc.')}
                resnet.load_state_dict(state_dict, strict=False)
                print(f"[SFCANet] Loaded Phase 1 ResNet-50 from {resnet50_ckpt_path}")
            except Exception as e:
                print(f"[SFCANet] WARNING: Could not load Phase 1 checkpoint: {e}")
                print("[SFCANet] Using ImageNet-pretrained ResNet-50.")

        # Freeze everything first, then selectively unfreeze
        for param in resnet.parameters():
            param.requires_grad = False

        if unfreeze_layer4:
            for param in resnet.layer4.parameters():
                param.requires_grad = True
            resnet.layer4.eval()
            print("[SFCANet] Unfroze ResNet-50 layer4 for deepfake-specific learning.")

        resnet.eval()
        self.spatial_backbone = resnet

        # ─── Frequency Branch ────────────────────────────────────────────
        self.freq_backbone = DCTCNN(standalone=False)

        # ─── Bidirectional Cross-Attention Fusion ────────────────────────
        self.cross_attention = BidirectionalCrossAttention(
            spatial_dim=SPATIAL_DIM,
            freq_dim=FREQ_DIM,
            fusion_dim=FUSION_DIM,
            num_heads=NUM_ATTENTION_HEADS
        )

        # ─── Classification Head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(FUSION_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"[SFCANet-v2] Total: {total:,} | Trainable: {trainable:,} | Frozen: {frozen:,}")

    def _spatial_forward(self, x):
        return self.spatial_backbone(x)

    def _fusion_forward(self, spatial_feat, freq_feat):
        return self.cross_attention(spatial_feat, freq_feat)

    def forward(self, images, dct_coeffs=None):
        spatial_feat = self._spatial_forward(images)
        freq_feat = self.freq_backbone(dct_coeffs)

        if self.use_gradient_checkpoint and self.training:
            fused = cp.checkpoint(
                self._fusion_forward, spatial_feat, freq_feat,
                use_reentrant=False
            )
        else:
            fused = self._fusion_forward(spatial_feat, freq_feat)

        return self.classifier(fused)

    def train(self, mode=True):
        super().train(mode)
        self.spatial_backbone.eval()
        for param in self.spatial_backbone.layer4.parameters():
            pass
        return self


if __name__ == "__main__":
    """Quick test with VRAM measurement."""
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = PHASE1_RESNET50_CKPT if os.path.exists(PHASE1_RESNET50_CKPT) else None
    model = SFCANet(resnet50_ckpt_path=ckpt_path).to(device)

    x = torch.randn(4, 3, 224, 224, device=device)
    dct = torch.randn(4, 192, 28, 28, device=device)

    model.eval()
    with torch.no_grad():
        out = model(x, dct)
    print(f"\nInput:  {x.shape} + DCT: {dct.shape}")
    print(f"Output: {out.shape}")
    print(f"Prob:   {torch.sigmoid(out).squeeze().tolist()}")

    if device.type == 'cuda':
        print(f"\nVRAM used: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
