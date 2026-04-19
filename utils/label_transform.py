"""
Label Transformation: Centroid → Binary Mask
============================================

The SpotGEO dataset only gives (x, y) centroid coordinates for each satellite.
But satellites aren't single pixels — they're small blobs spread over a few pixels
due to atmospheric distortion and long exposure.

This module converts centroid labels into binary masks that better represent
the actual satellite appearance, following the approach from Lan et al. (2025).

Grandma version: Instead of saying "the satellite is at this exact dot,"
we say "the satellite is somewhere in this small bright blob."
"""

import numpy as np
import cv2


def create_binary_mask(image, centroids, window_size=7, threshold=0.5,
                       dilation_radius=1):
    """
    Create a binary segmentation mask from centroid coordinates.

    For each centroid:
    1. Extract a small window around it from the image
    2. Normalize the window to [0, 1]
    3. Threshold at 0.5 to get the bright blob region
    4. Apply mild dilation to connect fragmented bright spots
    5. Place the binary patch back into the full-size mask

    Args:
        image: (H, W) grayscale image, uint8 or uint16
        centroids: list of (x, y) coordinates
        window_size: size of local window (2*m+1 where m=3 → 7)
        threshold: binarization threshold after normalization
        dilation_radius: radius of circular structuring element for dilation

    Returns:
        mask: (H, W) binary mask, float32 in {0, 1}
    """
    H, W = image.shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)
    m = window_size // 2  # half-window

    # Structuring element for dilation
    if dilation_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_radius + 1, 2 * dilation_radius + 1)
        )
    else:
        kernel = None

    image_float = image.astype(np.float32)

    for (cx, cy) in centroids:
        # Round to nearest pixel
        cx_int = int(round(cx))
        cy_int = int(round(cy))

        # Compute window bounds (clipped to image boundaries)
        y_start = max(0, cy_int - m)
        y_end = min(H, cy_int + m + 1)
        x_start = max(0, cx_int - m)
        x_end = min(W, cx_int + m + 1)

        # Extract local patch
        patch = image_float[y_start:y_end, x_start:x_end]

        if patch.size == 0:
            continue

        # Normalize to [0, 1]
        p_min, p_max = patch.min(), patch.max()
        if p_max - p_min > 1e-6:
            patch_norm = (patch - p_min) / (p_max - p_min)
        else:
            # Flat patch — just mark the center pixel
            patch_norm = np.ones_like(patch)

        # Binarize
        binary_patch = (patch_norm >= threshold).astype(np.uint8)

        # Mild dilation to connect fragments
        if kernel is not None and binary_patch.sum() > 0:
            binary_patch = cv2.dilate(binary_patch, kernel, iterations=1)

        # Place into full mask
        mask[y_start:y_end, x_start:x_end] = np.maximum(
            mask[y_start:y_end, x_start:x_end],
            binary_patch.astype(np.float32)
        )

    return mask


def create_gaussian_mask(centroids, height, width, sigma=2.0):
    """
    Alternative: Gaussian heatmap mask (softer targets, sometimes trains better).

    Each centroid gets a small Gaussian blob. The model learns to predict
    the "heat" of each pixel — brighter = more likely satellite.

    Args:
        centroids: list of (x, y) coordinates
        height, width: image dimensions
        sigma: Gaussian standard deviation in pixels

    Returns:
        heatmap: (H, W) float32 in [0, 1]
    """
    heatmap = np.zeros((height, width), dtype=np.float32)

    if len(centroids) == 0:
        return heatmap

    # Pre-compute Gaussian kernel size (cover 3*sigma)
    k = int(3 * sigma)
    size = 2 * k + 1

    for (cx, cy) in centroids:
        # Create small Gaussian patch
        y_grid, x_grid = np.mgrid[-k:k + 1, -k:k + 1]
        gaussian = np.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))

        # Compute placement bounds
        cx_int, cy_int = int(round(cx)), int(round(cy))
        y_start = max(0, cy_int - k)
        y_end = min(height, cy_int + k + 1)
        x_start = max(0, cx_int - k)
        x_end = min(width, cx_int + k + 1)

        # Corresponding region in gaussian kernel
        gy_start = y_start - (cy_int - k)
        gy_end = gy_start + (y_end - y_start)
        gx_start = x_start - (cx_int - k)
        gx_end = gx_start + (x_end - x_start)

        # Place with max (in case of overlapping blobs)
        heatmap[y_start:y_end, x_start:x_end] = np.maximum(
            heatmap[y_start:y_end, x_start:x_end],
            gaussian[gy_start:gy_end, gx_start:gx_end]
        )

    return heatmap
