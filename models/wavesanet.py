"""
WaveSANet: Wavelet-guided Spatial-channel Attention Network
============================================================

Architecture: U-Net encoder-decoder with WGSCA at each encoder level.

Think of it like this (grandma version):
- U-Net is like a funnel that squishes the image down (encoder),
  then expands it back up (decoder), learning what's important at each scale
- At each level of the funnel, our WGSCA module acts like a "frequency magnifying glass"
  that highlights the tiny satellite dots while dimming the star streaks and noise
- The skip connections carry fine details from encoder to decoder
  so we don't lose the exact satellite positions

The output is a heatmap: bright spots = "satellite probably here"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wgsca_module import WGSCA


class ConvBlock(nn.Module):
    """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder level: ConvBlock → WGSCA → MaxPool"""

    def __init__(self, in_ch, out_ch, use_wgsca=True, reduction=4):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.wgsca = WGSCA(out_ch, reduction=reduction) if use_wgsca else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Conv → attention → pool
        features = self.conv(x)
        features = self.wgsca(features)   # WGSCA enhances satellite features
        pooled = self.pool(features)
        return features, pooled           # features for skip connection, pooled for next level


class DecoderBlock(nn.Module):
    """Decoder level: Upsample → Concat skip → ConvBlock"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch (can happen due to odd dimensions)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class WaveSANet(nn.Module):
    """
    Full WaveSANet model.

    Architecture:
        Input (1, 480, 640)
        │
        ├─ Encoder 1: 1 → 32 channels, WGSCA, pool → (32, 240, 320)
        ├─ Encoder 2: 32 → 64 channels, WGSCA, pool → (64, 120, 160)
        ├─ Encoder 3: 64 → 128 channels, WGSCA, pool → (128, 60, 80)
        ├─ Encoder 4: 128 → 256 channels, WGSCA, pool → (256, 30, 40)
        │
        ├─ Bottleneck: 256 → 512 → 512 channels
        │
        ├─ Decoder 4: 512 + 256 skip → 256 channels
        ├─ Decoder 3: 256 + 128 skip → 128 channels
        ├─ Decoder 2: 128 + 64 skip → 64 channels
        ├─ Decoder 1: 64 + 32 skip → 32 channels
        │
        └─ Head: 32 → 1 channel (sigmoid) → detection heatmap
    """

    def __init__(self, in_channels=1, encoder_channels=None, use_wgsca=True, reduction=4):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]

        self.use_wgsca = use_wgsca
        ec = encoder_channels  # shorthand

        # Encoder
        self.enc1 = EncoderBlock(in_channels, ec[0], use_wgsca=use_wgsca, reduction=reduction)
        self.enc2 = EncoderBlock(ec[0], ec[1], use_wgsca=use_wgsca, reduction=reduction)
        self.enc3 = EncoderBlock(ec[1], ec[2], use_wgsca=use_wgsca, reduction=reduction)
        self.enc4 = EncoderBlock(ec[2], ec[3], use_wgsca=use_wgsca, reduction=reduction)

        # Bottleneck
        self.bottleneck = ConvBlock(ec[3], ec[3] * 2)

        # Decoder
        self.dec4 = DecoderBlock(ec[3] * 2, ec[3], ec[3])
        self.dec3 = DecoderBlock(ec[3], ec[2], ec[2])
        self.dec2 = DecoderBlock(ec[2], ec[1], ec[1])
        self.dec1 = DecoderBlock(ec[1], ec[0], ec[0])

        # Detection head: 1x1 conv → sigmoid → probability map
        self.head = nn.Sequential(
            nn.Conv2d(ec[0], 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) grayscale input image

        Returns:
            heatmap: (B, 1, H, W) detection probability map
        """
        # Encoder path (with WGSCA at each level)
        skip1, x = self.enc1(x)   # skip1: (B, 32, H, W),     x: (B, 32, H/2, W/2)
        skip2, x = self.enc2(x)   # skip2: (B, 64, H/2, W/2), x: (B, 64, H/4, W/4)
        skip3, x = self.enc3(x)   # skip3: (B, 128, H/4, W/4), x: (B, 128, H/8, W/8)
        skip4, x = self.enc4(x)   # skip4: (B, 256, H/8, W/8), x: (B, 256, H/16, W/16)

        # Bottleneck
        x = self.bottleneck(x)    # (B, 512, H/16, W/16)

        # Decoder path (with skip connections)
        x = self.dec4(x, skip4)   # (B, 256, H/8, W/8)
        x = self.dec3(x, skip3)   # (B, 128, H/4, W/4)
        x = self.dec2(x, skip2)   # (B, 64, H/2, W/2)
        x = self.dec1(x, skip1)   # (B, 32, H, W)

        # Detection head
        heatmap = self.head(x)    # (B, 1, H, W)

        return heatmap


class PlainUNet(WaveSANet):
    """
    Plain U-Net without WGSCA — used as ablation baseline.
    Same architecture, just WGSCA replaced with identity.
    """

    def __init__(self, in_channels=1, encoder_channels=None):
        super().__init__(
            in_channels=in_channels,
            encoder_channels=encoder_channels,
            use_wgsca=False  # <-- the only difference
        )


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test WaveSANet
    model = WaveSANet(in_channels=1).to(device)
    x = torch.randn(2, 1, 480, 640).to(device)
    out = model(x)
    print(f"WaveSANet: {x.shape} → {out.shape}")
    assert out.shape == (2, 1, 480, 640)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test Plain U-Net (ablation)
    baseline = PlainUNet(in_channels=1).to(device)
    out_b = baseline(x)
    print(f"\nPlainUNet: {x.shape} → {out_b.shape}")
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"Baseline parameters: {baseline_params:,}")
    print(f"WGSCA overhead: {total_params - baseline_params:,} params "
          f"({(total_params - baseline_params) / baseline_params * 100:.1f}% increase)")

    print("\nAll tests passed!")
