"""
SpotGEO Dataset Loader
======================

Loads 5-frame sequences with their annotations.
Each sequence is a group of 5 grayscale 640×480 images.
Annotations are (x, y) centroid coordinates per frame.

The loader:
1. Reads all 5 frames of a sequence
2. Generates binary mask labels from centroids
3. Applies optional augmentation
4. Returns individual frames for single-frame training
   OR the full sequence for multi-frame methods
"""

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .label_transform import create_binary_mask, create_gaussian_mask


class SpotGEODataset(Dataset):
    """
    SpotGEO dataset for single-frame detection training.

    Each __getitem__ returns ONE frame and its corresponding mask.
    For a sequence of 5 frames with N objects, this yields 5 training samples.

    Args:
        data_dir: path to train/ or test/ directory
        annotations_file: path to JSON annotations
        label_type: 'binary' or 'gaussian'
        augment: apply data augmentation
        window_size: label transformation window size
    """

    def __init__(self, data_dir, annotations_file=None, label_type='gaussian',
                 augment=False, window_size=7, gaussian_sigma=2.0):
        super().__init__()
        self.data_dir = data_dir
        self.label_type = label_type
        self.augment = augment
        self.window_size = window_size
        self.gaussian_sigma = gaussian_sigma

        # Load annotations — convert flat list to {seq_id: {frame_idx: [(x,y), ...]}}
        self.annotations = {}
        if annotations_file and os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                raw = json.load(f)
            if isinstance(raw, list):
                for entry in raw:
                    sid = str(entry['sequence_id'])
                    fidx = entry['frame'] - 1  # convert 1-indexed to 0-indexed
                    coords = [tuple(c) for c in entry['object_coords']]
                    self.annotations.setdefault(sid, {})[fidx] = coords
            else:
                self.annotations = raw

        # Discover sequences
        self.sequences = self._discover_sequences()

        # Build flat index: (sequence_idx, frame_idx) for each sample
        self.samples = []
        for seq_idx, seq_info in enumerate(self.sequences):
            for frame_idx in range(5):
                self.samples.append((seq_idx, frame_idx))

    def _discover_sequences(self):
        """Find all sequences in the data directory."""
        sequences = []
        # SpotGEO structure: each sequence is a folder with 5 PNG frames
        seq_dirs = sorted(glob.glob(os.path.join(self.data_dir, '*')))

        for seq_dir in seq_dirs:
            if not os.path.isdir(seq_dir):
                continue

            frames = sorted(glob.glob(os.path.join(seq_dir, '*.png')))
            if len(frames) == 5:
                seq_id = os.path.basename(seq_dir)
                sequences.append({
                    'id': seq_id,
                    'frames': frames,
                    'dir': seq_dir
                })

        # Alternative flat structure: frames named like seq_XXXX_frame_Y.png
        if len(sequences) == 0:
            all_pngs = sorted(glob.glob(os.path.join(self.data_dir, '*.png')))
            if len(all_pngs) > 0:
                # Group into sequences of 5
                for i in range(0, len(all_pngs), 5):
                    if i + 5 <= len(all_pngs):
                        seq_id = f"seq_{i // 5:04d}"
                        sequences.append({
                            'id': seq_id,
                            'frames': all_pngs[i:i + 5],
                            'dir': self.data_dir
                        })

        return sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.samples[idx]
        seq_info = self.sequences[seq_idx]

        # Load frame
        img_path = seq_info['frames'][frame_idx]
        img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)

        # Normalize to [0, 1] using min-max (more robust to low-signal images)
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)

        # Get annotations for this frame
        seq_id = seq_info['id']
        centroids = self._get_centroids(seq_id, frame_idx)

        # Create label mask
        if self.label_type == 'binary':
            mask = create_binary_mask(
                (img * 255).astype(np.uint8), centroids,
                window_size=self.window_size
            )
        else:
            mask = create_gaussian_mask(
                centroids, img.shape[0], img.shape[1],
                sigma=self.gaussian_sigma
            )

        # Data augmentation
        if self.augment:
            img, mask = self._augment(img, mask)

        # Convert to tensors: (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'centroids': centroids,
            'seq_id': seq_id,
            'frame_idx': frame_idx,
        }

    def _get_centroids(self, seq_id, frame_idx):
        """Get centroid coordinates for a specific frame."""
        if seq_id in self.annotations:
            seq_ann = self.annotations[seq_id]
            if isinstance(seq_ann, dict) and frame_idx in seq_ann:
                return seq_ann[frame_idx]
        return []

    def _augment(self, img, mask):
        """Simple augmentations that preserve spatial correspondence."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # Random brightness/contrast (only on image, not mask)
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # contrast
            beta = np.random.uniform(-0.05, 0.05)  # brightness
            img = np.clip(alpha * img + beta, 0, 1)

        return img, mask


class SpotGEOSequenceDataset(Dataset):
    """
    SpotGEO dataset returning FULL 5-frame sequences.
    Used for evaluation with multi-frame post-processing.

    Each __getitem__ returns all 5 frames and their annotations.
    """

    def __init__(self, data_dir, annotations_file=None):
        super().__init__()
        self.data_dir = data_dir
        self.annotations = {}
        if annotations_file and os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                raw = json.load(f)
            if isinstance(raw, list):
                for entry in raw:
                    sid = str(entry['sequence_id'])
                    fidx = entry['frame'] - 1
                    coords = [tuple(c) for c in entry['object_coords']]
                    self.annotations.setdefault(sid, {})[fidx] = coords
            else:
                self.annotations = raw

        self.sequences = self._discover_sequences()

    def _discover_sequences(self):
        """Same logic as SpotGEODataset."""
        sequences = []
        seq_dirs = sorted(glob.glob(os.path.join(self.data_dir, '*')))
        for seq_dir in seq_dirs:
            if not os.path.isdir(seq_dir):
                continue
            frames = sorted(glob.glob(os.path.join(seq_dir, '*.png')))
            if len(frames) == 5:
                seq_id = os.path.basename(seq_dir)
                sequences.append({'id': seq_id, 'frames': frames, 'dir': seq_dir})
        if len(sequences) == 0:
            all_pngs = sorted(glob.glob(os.path.join(self.data_dir, '*.png')))
            for i in range(0, len(all_pngs), 5):
                if i + 5 <= len(all_pngs):
                    seq_id = f"seq_{i // 5:04d}"
                    sequences.append({'id': seq_id, 'frames': all_pngs[i:i + 5], 'dir': self.data_dir})
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        seq_id = seq_info['id']

        # Load all 5 frames with min-max normalization
        frames = []
        for path in seq_info['frames']:
            img = np.array(Image.open(path).convert('L'), dtype=np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-7)
            frames.append(torch.from_numpy(img).unsqueeze(0))  # (1, H, W)

        frames_tensor = torch.stack(frames, dim=0)  # (5, 1, H, W)

        # Load all annotations
        gt_centroids = []
        for f_idx in range(5):
            centroids = []
            if seq_id in self.annotations:
                seq_ann = self.annotations[seq_id]
                if isinstance(seq_ann, dict) and f_idx in seq_ann:
                    centroids = seq_ann[f_idx]
            gt_centroids.append(centroids)

        return {
            'frames': frames_tensor,
            'gt_centroids': gt_centroids,
            'seq_id': seq_id,
        }
