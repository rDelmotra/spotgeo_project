"""
Pre-processing: Star Removal + Wavelet Denoising
=================================================

Before feeding images to the network, we clean them up:
1. Median frame subtraction — removes star streaks
2. Wavelet denoising — reduces sensor noise while preserving satellite blobs

Grandma version: Stars make bright streaks across all 5 photos.
If we stack all 5 photos and take the "middle value" at each pixel,
we get a "background" image that's mostly stars and noise.
Subtracting this background makes the satellite dots stand out.
"""

import numpy as np
import pywt
import cv2


def median_frame_subtraction(frames):
    """
    Remove star streaks by subtracting the median across frames.

    Stars appear in consistent patterns (shifted slightly between frames).
    The median captures this "static" background.
    Subtracting it leaves behind the moving satellites.

    Args:
        frames: list of 5 grayscale images, each (H, W) float32 in [0, 1]

    Returns:
        cleaned: list of 5 cleaned images, float32 in [0, 1]
    """
    # Stack frames: (5, H, W)
    stack = np.stack(frames, axis=0)

    # Compute median across frames
    median_bg = np.median(stack, axis=0)  # (H, W)

    # Subtract background and clip to [0, 1]
    cleaned = []
    for frame in frames:
        diff = frame - median_bg
        # Keep only positive residuals (satellite is brighter than background)
        diff = np.clip(diff, 0, 1)
        cleaned.append(diff)

    return cleaned


def wavelet_denoise(image, wavelet='db4', level=2, sigma=None):
    """
    Wavelet-based denoising using soft thresholding.

    Decomposes image → thresholds small wavelet coefficients (noise)
    → reconstructs. Preserves satellite blobs (significant coefficients)
    while removing random sensor noise.

    Args:
        image: (H, W) float32 in [0, 1]
        wavelet: wavelet type ('db4' = Daubechies-4, good general choice)
        level: decomposition levels
        sigma: noise standard deviation estimate (None = auto-estimate)

    Returns:
        denoised: (H, W) float32 in [0, 1]
    """
    # Wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Estimate noise sigma from finest detail coefficients (HH band)
    if sigma is None:
        detail_coeffs = coeffs[-1]  # (LH, HL, HH) at finest level
        hh = detail_coeffs[2]  # HH subband
        sigma = np.median(np.abs(hh)) / 0.6745  # Robust MAD estimator

    # Universal threshold (VisuShrink)
    n_pixels = image.shape[0] * image.shape[1]
    threshold = sigma * np.sqrt(2 * np.log(n_pixels))

    # Soft threshold all detail coefficients
    denoised_coeffs = [coeffs[0]]  # Keep approximation (LL) unchanged
    for detail_level in coeffs[1:]:
        denoised_detail = tuple(
            pywt.threshold(c, threshold, mode='soft')
            for c in detail_level
        )
        denoised_coeffs.append(denoised_detail)

    # Reconstruct
    denoised = pywt.waverec2(denoised_coeffs, wavelet)

    # Ensure same size (wavelet can add a pixel)
    denoised = denoised[:image.shape[0], :image.shape[1]]

    return np.clip(denoised, 0, 1).astype(np.float32)


def preprocess_sequence(frames, use_median_sub=True, use_denoise=True,
                        wavelet='db4', denoise_level=2):
    """
    Full pre-processing pipeline for a 5-frame sequence.

    Args:
        frames: list of 5 images, each (H, W) float32 in [0, 1]
        use_median_sub: apply median frame subtraction
        use_denoise: apply wavelet denoising

    Returns:
        processed: list of 5 processed images
    """
    if use_median_sub:
        frames = median_frame_subtraction(frames)

    if use_denoise:
        frames = [wavelet_denoise(f, wavelet=wavelet, level=denoise_level)
                  for f in frames]

    return frames


def enhance_contrast(image, clip_limit=2.0, grid_size=8):
    """
    Optional: CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Can help make very faint satellites more visible.

    Args:
        image: (H, W) float32 in [0, 1]
        clip_limit: contrast amplification limit
        grid_size: size of grid for adaptive equalization

    Returns:
        enhanced: (H, W) float32 in [0, 1]
    """
    img_uint8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(grid_size, grid_size))
    enhanced = clahe.apply(img_uint8)
    return enhanced.astype(np.float32) / 255.0
