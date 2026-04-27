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



# =============================================================================
# SFCANet-v3 — Cross-Domain Robust Architecture
# =============================================================================
# Changes over v2:
#   1. Pure ImageNet ResNet-50 (no Phase-1 bias) — unfreeze layer3 + layer4
#   2. Multi-Scale DCT branch (3 granularities: 4×4, 8×8, 16×16)
#   3. Stronger classifier head (Dropout 0.5/0.4/0.3)
#   4. Gradient Reversal Layer + Domain Classifier for adversarial alignment
# =============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Reverses the gradient sign during backward pass.
    Forward: identity.   Backward: multiply by -alpha.
    This forces the feature extractor to learn domain-invariant representations.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = alpha


class SFCANetV3(nn.Module):
    """
    Spatial-Frequency Cross-Attention Network — v3 (Cross-Domain Robust).

    Architecture:
        Spatial  : ResNet-50 (ImageNet, layer3+4 unfrozen) → 2048-dim
        Frequency: MultiScaleDCTCNN (4×8×16 blocks) → 512-dim
        Fusion   : BidirectionalCrossAttention → 512-dim
        Main head: strong Dropout → Linear cascade → 1 logit
        Domain head: GRL + 2-layer MLP → 1 logit (source domain)

    Training:
        Loss = BCE(fake/real) + λ × BCE(domain)
        λ is annealed from 0 → 1 over warmup epochs.
        GRL ensures features become domain-invariant.
    """

    def __init__(self, grl_alpha: float = 0.0, use_gradient_checkpoint: bool = True):
        super().__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        # ── 1. Spatial Backbone (pure ImageNet — no dataset-specific bias) ──
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        resnet.fc = nn.Identity()

        # Freeze layers 1 & 2, unfreeze layers 3 & 4 (more capacity than v2)
        for name, param in resnet.named_parameters():
            if any(name.startswith(lyr) for lyr in ['layer3', 'layer4']):
                param.requires_grad = True
            else:
                param.requires_grad = False

        resnet.eval()
        self.spatial_backbone = resnet
        print("[SFCANetV3] Spatial: ResNet-50 (ImageNet), layer3+4 trainable.")

        # ── 2. Multi-Scale Frequency Branch ────────────────────────────────
        # Import here to avoid circular imports at module level
        from multiscale_dct_cnn import MultiScaleDCTCNN
        from phase3_config import SPATIAL_DIM as _SDIM, FREQ_DIM as _FDIM
        from phase3_config import FUSION_DIM as _FUDIM, NUM_ATTENTION_HEADS as _HEADS
        self.freq_backbone = MultiScaleDCTCNN(out_dim=_FDIM)

        # ── 3. Bidirectional Cross-Attention Fusion ─────────────────────────
        self.cross_attention = BidirectionalCrossAttention(
            spatial_dim=_SDIM,
            freq_dim=_FDIM,
            fusion_dim=_FUDIM,
            num_heads=_HEADS,
        )

        # ── 4. Main Classification Head (stronger regularization) ──────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(_FUDIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # ── 5. Adversarial Domain Head (GRL) ───────────────────────────────
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.domain_classifier = nn.Sequential(
            nn.Linear(_FUDIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SFCANetV3] Total: {total:,} | Trainable: {trainable:,} | "
              f"Frozen: {total - trainable:,}")

    # ── GRL annealing helper ────────────────────────────────────────────────
    def set_grl_alpha(self, alpha: float):
        """Call each epoch to anneal the reversal strength."""
        self.grl.set_alpha(alpha)

    def _spatial_forward(self, x):
        return self.spatial_backbone(x)

    def _fusion_forward(self, spatial_feat, freq_feat):
        return self.cross_attention(spatial_feat, freq_feat)

    def forward(self, images: torch.Tensor, dct_dict: dict):
        """
        Args:
            images  : [B, 3, 224, 224]
            dct_dict: {4: [B,48,56,56], 8: [B,192,28,28], 16: [B,768,14,14]}
        Returns:
            logits (main) : [B, 1]
            logits (domain): [B, 1]  — only meaningful during training
        """
        spatial_feat = self._spatial_forward(images)
        freq_feat    = self.freq_backbone(dct_dict)

        if self.use_gradient_checkpoint and self.training:
            fused = torch.utils.checkpoint.checkpoint(
                self._fusion_forward, spatial_feat, freq_feat,
                use_reentrant=False,
            )
        else:
            fused = self._fusion_forward(spatial_feat, freq_feat)

        main_logit   = self.classifier(fused)
        domain_logit = self.domain_classifier(self.grl(fused))

        return main_logit, domain_logit

    def train(self, mode=True):
        super().train(mode)
        # Always keep early ResNet layers in eval (saves BN stats)
        self.spatial_backbone.layer1.eval()
        self.spatial_backbone.layer2.eval()
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


# =============================================================================
# SFCANet-v4 -- SOTA Cross-Domain Architecture
# =============================================================================
# Upgrades over v3:
#   1. ConvNeXt-Tiny backbone (better high-freq preservation, stoch depth 0.1)
#   2. Frequency-Guided Attention (FGA): DCT generates a 2D channel+spatial
#      attention map that GATES ConvNeXt feature maps before pooling.
#      "Frequency tells spatial which pixels are mathematically suspicious."
#   3. 1x1 pointwise bottleneck in DCT branch reduces params significantly.
#   4. No GRL -- data mixing + JPEG augmentation handles domain alignment.
# =============================================================================


class FrequencyGuidedAttention(nn.Module):
    """
    Adaptive Frequency-Guided Attention (AFGA).

    Takes:
      spatial_maps : ConvNeXt feature maps  [B, 768, 7, 7]  (before avg pool)
      freq_feat    : Multi-Scale DCT embed  [B, 512]

    Produces two complementary attention signals from freq_feat:
      1. Channel gate  [B, 768, 1, 1] -- which feature channels look suspicious
      2. Spatial gate  [B, 1,   7, 7] -- which spatial positions look suspicious

    Learnable Soft-Gate (Alpha):
      A small MLP implicitly detects the domain (from pooled spatial features)
      and outputs a weight alpha in [0, 1].

    Gating Equation: spatial_maps * (1 + alpha * c_gate * s_gate)

    Output: [B, 768] frequency-modulated spatial embedding.
    """

    def __init__(self, spatial_dim: int = 768, freq_dim: int = 512,
                 spatial_size: int = 7):
        super().__init__()
        self.spatial_size = spatial_size

        # Channel gate: DCT -> per-channel importance [B, spatial_dim]
        self.channel_gate = nn.Sequential(
            nn.Linear(freq_dim, freq_dim),
            nn.GELU(),
            nn.Linear(freq_dim, spatial_dim),
            nn.Sigmoid(),
        )

        # Spatial gate: DCT -> 7x7 spatial importance [B, 1, H, W]
        self.spatial_gate = nn.Sequential(
            nn.Linear(freq_dim, freq_dim // 2),
            nn.GELU(),
            nn.Linear(freq_dim // 2, spatial_size * spatial_size),
            nn.Sigmoid(),
        )

        # Soft-Gate Alpha: Spatial Maps -> Alpha scalar [B, 1]
        self.alpha_mlp = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.post_norm = nn.LayerNorm(spatial_dim)

    def forward(self, spatial_maps: torch.Tensor,
                freq_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = spatial_maps.shape

        # Channel attention: [B, C, 1, 1]
        c_gate = self.channel_gate(freq_feat).view(B, C, 1, 1)

        # Spatial attention: [B, 1, H, W]
        s_gate = self.spatial_gate(freq_feat).view(B, 1, H, W)

        # Learnable Soft-Gate weight
        pooled_spatial = nn.functional.adaptive_avg_pool2d(spatial_maps, 1).flatten(1)
        alpha = self.alpha_mlp(pooled_spatial).view(B, 1, 1, 1)

        # Gate feature maps with residual soft-gate
        gated = spatial_maps * (1.0 + alpha * c_gate * s_gate)   # [B, C, H, W]

        # Pool -> [B, C]
        out = nn.functional.adaptive_avg_pool2d(gated, 1).flatten(1)

        return self.post_norm(out)


class _SEBlock(nn.Module):
    """Squeeze-Excitation block for DCT branch channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        return x * self.excite(self.pool(x)).view(B, C, 1, 1)


class MultiScaleDCTCNNv4(nn.Module):
    """
    Multi-Scale DCT CNN with 1x1 pointwise bottleneck convolutions.

    Each scale branch: 1x1 conv (bottleneck to 64ch) -> 3x3 feature extraction
    -> SE attention -> pool -> 256-dim.  Concat 768 -> project -> 512-dim.

    Input:  {block_size: Tensor [B, C_s, H_s, W_s]}
    Output: Tensor [B, 512]
    """

    def __init__(self, out_dim: int = 512):
        super().__init__()
        from phase4_config import (
            DCT_SCALES, DCT_CHANNELS, FREQ_DIM_PER_SCALE,
            FREQ_DIM_TOTAL, DCT_BOTTLENECK_CH,
        )
        self.scales = DCT_SCALES
        bot = DCT_BOTTLENECK_CH    # 64

        self.branches = nn.ModuleDict()
        for s in self.scales:
            in_ch = DCT_CHANNELS[s]
            self.branches[str(s)] = nn.Sequential(
                # 1x1 pointwise bottleneck
                nn.Conv2d(in_ch, bot, 1, bias=False),
                nn.BatchNorm2d(bot),
                nn.GELU(),
                # 3x3 spatial feature extraction
                nn.Conv2d(bot, bot * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(bot * 2),
                nn.GELU(),
                nn.MaxPool2d(2),
                # Project to per-scale output dim
                nn.Conv2d(bot * 2, FREQ_DIM_PER_SCALE, 3, padding=1, bias=False),
                nn.BatchNorm2d(FREQ_DIM_PER_SCALE),
                nn.GELU(),
                _SEBlock(FREQ_DIM_PER_SCALE),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        self.project = nn.Sequential(
            nn.LayerNorm(FREQ_DIM_TOTAL),
            nn.Linear(FREQ_DIM_TOTAL, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        total = sum(p.numel() for p in self.parameters())
        print(f"[MultiScaleDCTCNNv4] Bottleneck: {bot}ch | "
              f"Output: {out_dim}-dim | Params: {total:,}")

    def forward(self, dct_dict: dict) -> torch.Tensor:
        feats = [self.branches[str(s)](dct_dict[s]) for s in self.scales]
        return self.project(torch.cat(feats, dim=-1))


class SFCANetV4(nn.Module):
    """
    Spatial-Frequency Cross-Attention Network v4 -- SOTA Cross-Domain.

    Architecture:
        Spatial:   ConvNeXt-Tiny (ImageNet, stages 4-7 trainable,
                   stochastic depth 0.1) -> [B, 768, 7, 7] feature maps
        Frequency: MultiScaleDCTCNNv4 (1x1 bottleneck) -> [B, 512]
        Fusion:    FrequencyGuidedAttention:
                     channel gate [B, 768, 1, 1] x spatial gate [B, 1, 7, 7]
                   -> [B, 768] gated + pooled spatial features
        Head:      Dropout(0.5)->256->LN->GELU->Dropout(0.4)->128->GELU
                   ->Dropout(0.3)->1 logit

    No GRL. Single BCE loss. Data mixing + JPEG aug handles domain shift.
    """

    def __init__(self, use_gradient_checkpoint: bool = True):
        super().__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        from torchvision import models as tvm
        from phase4_config import (
            SPATIAL_DIM, FREQ_DIM, FGA_SPATIAL_SIZE, STOCHASTIC_DEPTH,
        )

        # ── 1. ConvNeXt-Tiny Spatial Backbone ────────────────────────────────
        convnext = tvm.convnext_tiny(
            weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
            stochastic_depth_prob=STOCHASTIC_DEPTH,
        )

        # Freeze stages 0-3 (stem, stage1, down1, stage2)
        # Unfreeze stages 4-7 (down2, stage3, down3, stage4)
        for i, child in enumerate(convnext.features):
            for p in child.parameters():
                p.requires_grad = (i >= 4)

        self.spatial_features = convnext.features   # [B, 768, 7, 7]
        self.spatial_avgpool  = convnext.avgpool     # AdaptiveAvgPool2d(1)
        print("[SFCANetV4] ConvNeXt-Tiny: stages 0-3 frozen, stages 4-7 trainable.")

        # ── 2. Multi-Scale Frequency Branch (1x1 bottleneck) ─────────────────
        self.freq_backbone = MultiScaleDCTCNNv4(out_dim=FREQ_DIM)

        # ── 3. Frequency-Guided Attention ─────────────────────────────────────
        self.fga = FrequencyGuidedAttention(
            spatial_dim=SPATIAL_DIM,
            freq_dim=FREQ_DIM,
            spatial_size=FGA_SPATIAL_SIZE,
        )

        # ── 4. Classification Head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(SPATIAL_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SFCANetV4] Total: {total:,} | Trainable: {trainable:,} | "
              f"Frozen: {total - trainable:,}")

    def _spatial_fwd(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_features(x)   # [B, 768, 7, 7]

    def forward(self, images: torch.Tensor, dct_dict: dict) -> torch.Tensor:
        """
        Args:
            images   : [B, 3, 224, 224]
            dct_dict : {4: [B,48,56,56], 8: [B,192,28,28], 16: [B,768,14,14]}
        Returns:
            logits: [B, 1]
        """
        freq_feat = self.freq_backbone(dct_dict)       # [B, 512]

        if self.use_gradient_checkpoint and self.training:
            spatial_maps = torch.utils.checkpoint.checkpoint(
                self._spatial_fwd, images, use_reentrant=False,
            )
        else:
            spatial_maps = self._spatial_fwd(images)   # [B, 768, 7, 7]

        fused = self.fga(spatial_maps, freq_feat)      # [B, 768]
        return self.classifier(fused)                   # [B, 1]

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen ConvNeXt stages in eval (stable BN statistics)
        for i in range(4):
            self.spatial_features[i].eval()
        return self

