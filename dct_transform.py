"""
Block-wise 2D DCT Transform
=============================
Converts a 224×224×3 RGB image into 192-channel DCT coefficient maps (28×28).
Operates on 8×8 blocks aligned with JPEG compression grid.

Usage:
    from dct_transform import BlockDCT
    dct = BlockDCT(block_size=8)
    coeffs = dct(image_tensor)  # [B, 192, 28, 28]
"""

import torch
import torch.nn as nn
import math


class BlockDCT(nn.Module):
    """
    GPU-accelerated block-wise 2D DCT.

    Input:  [B, 3, 224, 224] — normalized RGB images
    Output: [B, 192, 28, 28] — DCT coefficient maps

    For each 8×8 block in each channel, compute 64 DCT-II coefficients.
    Rearrange so that each of the 64 frequency positions becomes a
    separate channel → 3 channels × 64 = 192 output channels.
    """

    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size

        # Precompute the 1D DCT-II basis matrix (block_size × block_size)
        # D[k, n] = alpha(k) * cos(pi * (2n+1) * k / (2*N))
        N = block_size
        basis = torch.zeros(N, N)
        for k in range(N):
            for n in range(N):
                if k == 0:
                    alpha = math.sqrt(1.0 / N)
                else:
                    alpha = math.sqrt(2.0 / N)
                basis[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / (2 * N))

        # Register as buffer (moves to GPU with model, not trained)
        self.register_buffer('dct_basis', basis)  # [8, 8]

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] RGB tensor (H, W must be divisible by block_size)
        Returns:
            [B, 3*block_size^2, H//block_size, W//block_size]
            i.e., [B, 192, 28, 28] for 224×224 input with 8×8 blocks
        """
        B, C, H, W = x.shape
        bs = self.block_size
        assert H % bs == 0 and W % bs == 0, \
            f"Image dimensions ({H}×{W}) must be divisible by block_size ({bs})"

        h_blocks = H // bs  # 28
        w_blocks = W // bs  # 28

        # Reshape into blocks: [B, C, h_blocks, bs, w_blocks, bs]
        x = x.reshape(B, C, h_blocks, bs, w_blocks, bs)
        # Permute to: [B, C, h_blocks, w_blocks, bs, bs]
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()

        # Apply 2D DCT: D @ block @ D^T for each block
        # x shape: [B, C, h_blocks, w_blocks, bs, bs]
        # dct_basis shape: [bs, bs]
        # 2D DCT = basis @ x @ basis^T
        dct_coeffs = torch.matmul(
            torch.matmul(self.dct_basis, x),
            self.dct_basis.t()
        )
        # dct_coeffs: [B, C, h_blocks, w_blocks, bs, bs]

        # Rearrange: each of the bs*bs=64 frequency positions → separate channel
        # Final shape: [B, C * bs * bs, h_blocks, w_blocks] = [B, 192, 28, 28]
        dct_coeffs = dct_coeffs.reshape(B, C, h_blocks, w_blocks, bs * bs)
        dct_coeffs = dct_coeffs.permute(0, 1, 4, 2, 3)  # [B, C, 64, 28, 28]
        dct_coeffs = dct_coeffs.reshape(B, C * bs * bs, h_blocks, w_blocks)

        return dct_coeffs


if __name__ == "__main__":
    """Quick test: verify shapes and GPU performance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dct = BlockDCT(block_size=8).to(device)

    # Dummy input
    x = torch.randn(4, 3, 224, 224, device=device)
    coeffs = dct(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {coeffs.shape}")
    print(f"Expected: [4, 192, 28, 28]")
    print(f"Value range: [{coeffs.min():.2f}, {coeffs.max():.2f}]")

    # Timing
    import time
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = dct(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 100
    print(f"DCT speed: {elapsed*1000:.2f} ms/batch ({4/elapsed:.0f} img/s)")
