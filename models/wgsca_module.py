"""
Wavelet-Guided Spatial-Channel Attention (WGSCA) Module
=======================================================
This is the NOVEL component of our approach.

Key idea: Instead of using wavelet transform as a pre-processing step
(like WTNet does), we integrate it INSIDE the network as a learnable
attention mechanism. The network learns which frequency bands matter
for satellite detection.

How it works (grandma version):
1. Take feature maps from the U-Net encoder
2. Split them into "smooth" and "detailed" parts using wavelets
   (like separating the bass and treble in music)
3. Pay extra attention to the "detailed" parts — that's where
   tiny satellite blobs hide
4. Recombine and let the network decide what's important

Technical details:
- DWT decomposes features into LL (low-freq) and LH/HL/HH (high-freq)
- Channel attention (SE-style) on high-freq bands: learn which channels
  carry satellite-relevant high-freq info
- Spatial attention on recombined features: learn WHERE to look
- Residual connection: don't lose original information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteWaveletTransform2D(nn.Module):
    """
    2D Discrete Wavelet Transform using Haar wavelet.

    Decomposes a feature map into 4 subbands:
    - LL: low-freq in both directions (smooth approximation)
    - LH: low-freq horizontal, high-freq vertical (horizontal edges)
    - HL: high-freq horizontal, low-freq vertical (vertical edges)
    - HH: high-freq in both directions (diagonal details / points)

    GEO satellites appear as small blobs → strong HH and mixed LH/HL response.
    Star streaks are elongated → strong in one of LH/HL but not both.
    This difference is what our attention module exploits.
    """

    def __init__(self):
        super().__init__()
        # Haar wavelet filters (simplest, fastest, sufficient for our purpose)
        # These are fixed — not learnable
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5

        # Register as buffers (moved to GPU with model, but not trained)
        self.register_buffer('ll_filter', ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('lh_filter', lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('hl_filter', hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('hh_filter', hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature maps

        Returns:
            ll, lh, hl, hh: each (B, C, H/2, W/2)
        """
        B, C, H, W = x.shape

        # Expand filters to match input channels (depthwise convolution)
        ll_f = self.ll_filter.expand(C, -1, -1, -1)
        lh_f = self.lh_filter.expand(C, -1, -1, -1)
        hl_f = self.hl_filter.expand(C, -1, -1, -1)
        hh_f = self.hh_filter.expand(C, -1, -1, -1)

        # Apply filters with stride 2 (downsampling) — groups=C for depthwise
        ll = F.conv2d(x, ll_f, stride=2, groups=C)
        lh = F.conv2d(x, lh_f, stride=2, groups=C)
        hl = F.conv2d(x, hl_f, stride=2, groups=C)
        hh = F.conv2d(x, hh_f, stride=2, groups=C)

        return ll, lh, hl, hh


class InverseDiscreteWaveletTransform2D(nn.Module):
    """
    Inverse 2D DWT — reconstructs feature maps from subbands.
    """

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5

        self.register_buffer('ll_filter', ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('lh_filter', lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('hl_filter', hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('hh_filter', hh.unsqueeze(0).unsqueeze(0))

    def forward(self, ll, lh, hl, hh):
        """
        Args:
            ll, lh, hl, hh: each (B, C, H/2, W/2)

        Returns:
            x: (B, C, H, W) reconstructed feature maps
        """
        B, C, Hh, Wh = ll.shape

        # Expand filters for depthwise transposed convolution
        ll_f = self.ll_filter.expand(C, -1, -1, -1)
        lh_f = self.lh_filter.expand(C, -1, -1, -1)
        hl_f = self.hl_filter.expand(C, -1, -1, -1)
        hh_f = self.hh_filter.expand(C, -1, -1, -1)

        # Transposed convolution with stride 2 (upsampling)
        x = (F.conv_transpose2d(ll, ll_f, stride=2, groups=C) +
             F.conv_transpose2d(lh, lh_f, stride=2, groups=C) +
             F.conv_transpose2d(hl, hl_f, stride=2, groups=C) +
             F.conv_transpose2d(hh, hh_f, stride=2, groups=C))

        return x


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.

    Learns which channels carry the most useful high-frequency information
    for satellite detection. Applied to the concatenated HF subbands.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling → (B, C, 1, 1)
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x * attention_weights: (B, C, H, W)
        """
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """
    Spatial attention module.

    After recombining wavelet subbands, this tells the network WHERE
    to look — highlighting spatial locations likely to contain satellites.

    Uses max-pool and avg-pool across channels → 7x7 conv → sigmoid.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x * spatial_attention_map: (B, C, H, W)
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attention = self.conv(combined)  # (B, 1, H, W)
        return x * attention


class WGSCA(nn.Module):
    """
    Wavelet-Guided Spatial-Channel Attention (WGSCA)
    ================================================

    THE NOVEL MODULE.

    Pipeline:
    1. Input features → DWT → LL, LH, HL, HH subbands
    2. Concatenate HF subbands (LH, HL, HH) → Channel Attention
       → learn which channels carry satellite-like HF patterns
    3. Enhanced HF subbands + LL → Inverse DWT → reconstructed features
    4. Reconstructed features → Spatial Attention
       → learn WHERE satellites likely are
    5. Residual connection with original input

    Why this is better than external wavelet pre-processing:
    - The attention weights are LEARNED end-to-end with the detection task
    - Different encoder levels learn different frequency behaviors
    - The model can adapt to varying noise levels across the dataset
    """

    def __init__(self, channels, reduction=4, use_spatial=True, use_channel=True):
        """
        Args:
            channels: number of input/output channels
            reduction: channel attention squeeze ratio
            use_spatial: enable spatial attention
            use_channel: enable channel attention on HF bands
        """
        super().__init__()

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        # Wavelet transforms
        self.dwt = DiscreteWaveletTransform2D()
        self.idwt = InverseDiscreteWaveletTransform2D()

        # Channel attention on high-frequency subbands
        # 3 HF subbands concatenated → 3*channels input
        if use_channel:
            self.hf_channel_attn = ChannelAttention(channels * 3, reduction)
            # Conv to enhance HF features before splitting back
            self.hf_conv = nn.Sequential(
                nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=3),
                nn.BatchNorm2d(channels * 3),
                nn.ReLU(inplace=True)
            )

        # Spatial attention on reconstructed features
        if use_spatial:
            self.spatial_attn = SpatialAttention(kernel_size=7)

        # Final fusion conv (blends wavelet-enhanced with original)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) encoder features

        Returns:
            out: (B, C, H, W) attention-enhanced features (same size as input)
        """
        identity = x
        B, C, H, W = x.shape

        # Pad to even dimensions if needed (DWT requires even H, W)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Step 1: Wavelet decomposition
        ll, lh, hl, hh = self.dwt(x)

        # Step 2: Channel attention on high-frequency subbands
        if self.use_channel:
            hf = torch.cat([lh, hl, hh], dim=1)  # (B, 3C, H/2, W/2)
            hf = self.hf_conv(hf)                  # enhance HF features
            hf = self.hf_channel_attn(hf)          # channel attention
            lh_e, hl_e, hh_e = torch.chunk(hf, 3, dim=1)
        else:
            lh_e, hl_e, hh_e = lh, hl, hh

        # Step 3: Inverse DWT → reconstruct with enhanced HF
        reconstructed = self.idwt(ll, lh_e, hl_e, hh_e)

        # Remove padding if we added it
        if pad_h or pad_w:
            reconstructed = reconstructed[:, :, :H, :W]

        # Step 4: Spatial attention
        if self.use_spatial:
            reconstructed = self.spatial_attn(reconstructed)

        # Step 5: Fusion + residual
        out = self.fusion(reconstructed) + identity

        return out


# ============================================================
# Quick test to verify shapes
# ============================================================
if __name__ == "__main__":
    # Test WGSCA module
    batch_size = 2
    channels = 64
    height, width = 120, 160  # SpotGEO is 480x640, encoder level 2 would be ~120x160

    x = torch.randn(batch_size, channels, height, width)
    module = WGSCA(channels=channels, reduction=4)

    out = module(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert x.shape == out.shape, "WGSCA must preserve spatial dimensions!"

    # Count parameters
    params = sum(p.numel() for p in module.parameters())
    print(f"WGSCA parameters: {params:,}")  # Should be small — ~few thousand

    # Test with odd dimensions (edge case)
    x_odd = torch.randn(2, 64, 121, 161)
    out_odd = module(x_odd)
    assert x_odd.shape == out_odd.shape, "Must handle odd dimensions!"
    print(f"Odd dimension test passed: {x_odd.shape} → {out_odd.shape}")

    print("\nAll tests passed!")
