"""
Baseline models for comparison against WaveSANet.

We compare:
1. PlainUNet — same U-Net without WGSCA (ablation)
2. UNetCBAM — U-Net with standard CBAM attention (proves wavelet > generic attention)
3. WTNetStyle — U-Net with wavelet as pre-processing only (proves integrated > external)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavesanet import ConvBlock, DecoderBlock, PlainUNet


# ============================================================
# Baseline 2: U-Net + CBAM (standard attention, no wavelets)
# ============================================================

class CBAMBlock(nn.Module):
    """
    Standard CBAM (Convolutional Block Attention Module).
    Channel attention + Spatial attention — but WITHOUT wavelet guidance.
    This is the "generic attention" baseline to show wavelets add value.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)

        # Channel attention
        self.ca_avg = nn.AdaptiveAvgPool2d(1)
        self.ca_max = nn.AdaptiveMaxPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels)
        )

        # Spatial attention
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        avg_out = self.ca_fc(self.ca_avg(x).flatten(1))
        max_out = self.ca_fc(self.ca_max(x).flatten(1))
        ca_weight = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        x = x * ca_weight

        # Spatial attention
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        sa_weight = torch.sigmoid(self.sa_conv(torch.cat([avg_map, max_map], dim=1)))
        x = x * sa_weight

        return x


class EncoderBlockCBAM(nn.Module):
    """Encoder with CBAM instead of WGSCA."""

    def __init__(self, in_ch, out_ch, reduction=4):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.cbam = CBAMBlock(out_ch, reduction)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv(x)
        features = self.cbam(features)
        pooled = self.pool(features)
        return features, pooled


class UNetCBAM(nn.Module):
    """U-Net with standard CBAM attention at each encoder level."""

    def __init__(self, in_channels=1, encoder_channels=None):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]
        ec = encoder_channels

        self.enc1 = EncoderBlockCBAM(in_channels, ec[0])
        self.enc2 = EncoderBlockCBAM(ec[0], ec[1])
        self.enc3 = EncoderBlockCBAM(ec[1], ec[2])
        self.enc4 = EncoderBlockCBAM(ec[2], ec[3])

        self.bottleneck = ConvBlock(ec[3], ec[3] * 2)

        self.dec4 = DecoderBlock(ec[3] * 2, ec[3], ec[3])
        self.dec3 = DecoderBlock(ec[3], ec[2], ec[2])
        self.dec2 = DecoderBlock(ec[2], ec[1], ec[1])
        self.dec1 = DecoderBlock(ec[1], ec[0], ec[0])

        self.head = nn.Sequential(nn.Conv2d(ec[0], 1, 1), nn.Sigmoid())

    def forward(self, x):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.head(x)


# ============================================================
# Baseline 3: WTNet-style (wavelet as PRE-PROCESSING only)
# ============================================================

class WaveletPreprocess(nn.Module):
    """
    Mimics WTNet's approach: apply wavelet enhancement BEFORE the network.
    The wavelet coefficients are fixed, not learned with the task.
    """

    def __init__(self, boost_factor=2.0):
        super().__init__()
        self.boost_factor = boost_factor

        # Haar wavelet filters (same as WGSCA but applied to raw image)
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5

        self.register_buffer('ll_f', ll.reshape(1, 1, 2, 2))
        self.register_buffer('lh_f', lh.reshape(1, 1, 2, 2))
        self.register_buffer('hl_f', hl.reshape(1, 1, 2, 2))
        self.register_buffer('hh_f', hh.reshape(1, 1, 2, 2))

        # Inverse filters
        self.register_buffer('ll_fi', ll.reshape(1, 1, 2, 2))
        self.register_buffer('lh_fi', lh.reshape(1, 1, 2, 2))
        self.register_buffer('hl_fi', hl.reshape(1, 1, 2, 2))
        self.register_buffer('hh_fi', hh.reshape(1, 1, 2, 2))

    def forward(self, x):
        """Boost high-frequency components and reconstruct."""
        B, C, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        ll = F.conv2d(x, self.ll_f, stride=2)
        lh = F.conv2d(x, self.lh_f, stride=2) * self.boost_factor
        hl = F.conv2d(x, self.hl_f, stride=2) * self.boost_factor
        hh = F.conv2d(x, self.hh_f, stride=2) * self.boost_factor

        out = (F.conv_transpose2d(ll, self.ll_fi, stride=2) +
               F.conv_transpose2d(lh, self.lh_fi, stride=2) +
               F.conv_transpose2d(hl, self.hl_fi, stride=2) +
               F.conv_transpose2d(hh, self.hh_fi, stride=2))

        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out


class UNetWTNetStyle(nn.Module):
    """
    U-Net with wavelet pre-processing (WTNet-style baseline).
    Wavelet enhancement is EXTERNAL to the network — not learned.
    """

    def __init__(self, in_channels=1, encoder_channels=None, boost_factor=2.0):
        super().__init__()
        self.wavelet_preprocess = WaveletPreprocess(boost_factor)
        self.unet = PlainUNet(in_channels, encoder_channels)

    def forward(self, x):
        x = self.wavelet_preprocess(x)
        return self.unet(x)


# ============================================================
# Model factory
# ============================================================

def build_model(model_name, in_channels=1, encoder_channels=None):
    """
    Build model by name. Used in train.py.

    Args:
        model_name: one of 'wavesanet', 'unet', 'unet_cbam', 'unet_wtnet'
    """
    from .wavesanet import WaveSANet, PlainUNet

    models = {
        'wavesanet': lambda: WaveSANet(in_channels, encoder_channels),
        'unet': lambda: PlainUNet(in_channels, encoder_channels),
        'unet_cbam': lambda: UNetCBAM(in_channels, encoder_channels),
        'unet_wtnet': lambda: UNetWTNetStyle(in_channels, encoder_channels),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    model = models[model_name]()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built {model_name} with {params:,} trainable parameters")
    return model
