# SpotGEO — Satellite Detection in Star-Field Images

WaveSANet is a deep-learning pipeline for detecting Geostationary (GEO) satellites in
5-frame astronomical image sequences. The network uses a **Wavelet-Guided Spatial-Channel
Attention (WGSCA)** module inside a U-Net encoder-decoder to highlight faint satellite
blobs while suppressing star streaks and sensor noise.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset Layout](#dataset-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
  - [All Training Flags](#all-training-flags)
  - [Training Examples](#training-examples)
- [Evaluation / Testing](#evaluation--testing)
  - [All Evaluation Flags](#all-evaluation-flags)
  - [Evaluation Examples](#evaluation-examples)
- [Models & Ablations](#models--ablations)
- [Pipeline Stages](#pipeline-stages)
- [Output Files](#output-files)
- [Metrics](#metrics)

---

## Project Structure

```
spotgeo_project/
├── data/                        # ← place your dataset here (see below)
│   └── spotgeo/
│       ├── train/               # training sequences
│       ├── test/                # test sequences
│       ├── train_anno.json      # training annotations
│       └── test_anno.json       # test annotations (if available)
├── configs/
│   └── config.py                # all hyper-parameters in one place
├── models/
│   ├── wavesanet.py             # WaveSANet + PlainUNet
│   ├── baselines.py             # UNetCBAM, UNetWTNetStyle, build_model()
│   └── wgsca_module.py          # WGSCA attention module
├── utils/
│   ├── dataset.py               # SpotGEODataset, SpotGEOSequenceDataset
│   ├── preprocessing.py         # median subtraction + wavelet denoising
│   ├── postprocessing.py        # Hungarian matching + trajectory completion
│   ├── evaluation.py            # F1 / MSE metrics (official SpotGEO toolkit)
│   └── label_transform.py       # binary & Gaussian mask generation
├── train.py                     # training entry-point
├── evaluate.py                  # evaluation entry-point
└── requirements.txt
```

---

## Dataset Layout

Place the SpotGEO dataset inside a `data/` folder in the project root:

```
data/
└── spotgeo/
    ├── train/
    │   ├── 0001/          # one folder per sequence
    │   │   ├── frame1.png
    │   │   ├── frame2.png
    │   │   ├── frame3.png
    │   │   ├── frame4.png
    │   │   └── frame5.png
    │   ├── 0002/
    │   └── ...
    ├── test/
    │   └── ...            # same structure as train/
    ├── train_anno.json    # annotations for training split
    └── test_anno.json     # annotations for test split (optional)
```

**Annotation format** (`train_anno.json`): a JSON list where each entry is:

```json
[
  {
    "sequence_id": "0001",
    "frame": 1,
    "object_coords": [[x1, y1], [x2, y2]]
  },
  ...
]
```

> **Alternative flat layout**: If all PNGs live directly inside `train/` (no
> sub-folders), the loader automatically groups every 5 consecutive files into a
> sequence.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/rDelmotra/spotgeo_project.git
cd spotgeo_project

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**requirements.txt** includes:
`torch`, `torchvision`, `numpy`, `scipy`, `opencv-python`, `Pillow`,
`PyWavelets`, `tqdm`, `matplotlib`

A CUDA-capable GPU is recommended but the code falls back to CPU (and MPS on Apple Silicon).

---

## Quick Start

```bash
# Train the main model for 200 epochs
python train.py --model wavesanet --epochs 200

# Evaluate the saved checkpoint on the training split
python evaluate.py \
    --model wavesanet \
    --checkpoint checkpoints/wavesanet_best.pth

# Evaluate on the test split
python evaluate.py \
    --model wavesanet \
    --checkpoint checkpoints/wavesanet_best.pth \
    --data_dir ./data/spotgeo/test \
    --annotations ./data/spotgeo/test_anno.json
```

---

## Training

### All Training Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | str | `wavesanet` | Model to train. Choices: `wavesanet`, `unet`, `unet_cbam`, `unet_wtnet` |
| `--epochs` | int | `200` | Number of training epochs (use 400 if early stopping does not trigger) |
| `--lr` | float | `0.0002` | Initial learning rate for AdamW |
| `--batch_size` | int | `16` | Training batch size |
| `--data_dir` | str | `./data/spotgeo/train` | Path to the training image directory |
| `--annotations` | str | `./data/spotgeo/train_anno.json` | Path to the training annotations JSON |
| `--checkpoint_dir` | str | `./checkpoints` | Directory to save model checkpoints |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--device` | str | `cuda` | Device to use: `cuda`, `mps`, or `cpu` |
| `--val_split` | float | `0.15` | Fraction of training data held out for validation (0.0 – 1.0) |

### Training Examples

```bash
# ── Main model (default settings) ────────────────────────────────────────────
python train.py --model wavesanet

# ── Longer run if early stopping hasn't triggered ────────────────────────────
python train.py --model wavesanet --epochs 400

# ── Custom learning rate ──────────────────────────────────────────────────────
python train.py --model wavesanet --lr 0.0015

# ── Larger batch on a powerful GPU ───────────────────────────────────────────
python train.py --model wavesanet --batch_size 32

# ── Train on CPU (slow, for testing) ─────────────────────────────────────────
python train.py --model wavesanet --device cpu --epochs 5

# ── Point to custom data paths ───────────────────────────────────────────────
python train.py \
    --model wavesanet \
    --data_dir ./data/spotgeo/train \
    --annotations ./data/spotgeo/train_anno.json \
    --checkpoint_dir ./my_checkpoints

# ── Train all baseline models ─────────────────────────────────────────────────
python train.py --model unet
python train.py --model unet_cbam
python train.py --model unet_wtnet
```

---

## Evaluation / Testing

### All Evaluation Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | str | `wavesanet` | Architecture to load. Must match the checkpoint. Choices: `wavesanet`, `unet`, `unet_cbam`, `unet_wtnet` |
| `--checkpoint` | str | **required** | Path to a `.pth` checkpoint file |
| `--data_dir` | str | `./data/spotgeo/train` | Path to the evaluation image directory |
| `--annotations` | str | `./data/spotgeo/train_anno.json` | Path to the evaluation annotations JSON |
| `--device` | str | `cuda` | Device to use: `cuda`, `mps`, or `cpu` |
| `--threshold` | float | `0.5` | Heatmap confidence threshold for converting predictions to binary detections |
| `--no-preprocess` | flag | off | **Ablation**: skip median subtraction + wavelet denoising pre-processing |
| `--no-postprocess` | flag | off | **Ablation**: skip Hungarian matching + trajectory completion post-processing |
| `--tau` | float | `5.0` | Matching distance threshold τ (pixels) — from official SpotGEO metrics |
| `--epsilon` | float | `2.0` | Tolerance distance ε (pixels) for labelling inaccuracy |

### Evaluation Examples

```bash
# ── Evaluate on the training split (default paths) ───────────────────────────
python evaluate.py \
    --model wavesanet \
    --checkpoint checkpoints/wavesanet_best.pth

# ── Evaluate on the test split ───────────────────────────────────────────────
python evaluate.py \
    --model wavesanet \
    --checkpoint checkpoints/wavesanet_best.pth \
    --data_dir ./data/spotgeo/test \
    --annotations ./data/spotgeo/test_anno.json

# ── Tune detection threshold ─────────────────────────────────────────────────
python evaluate.py \
    --model wavesanet \
    --checkpoint checkpoints/wavesanet_best.pth \
    --threshold 0.3

# ── Ablation: no pre-processing ───────────────────────────────────────────────
python evaluate.py \
    --model wavesanet \
    --checkpoint checkpoints/wavesanet_best.pth \
    --no-preprocess

# ── Ablation: no post-processing ──────────────────────────────────────────────
python evaluate.py \
    --model wavesanet \
    --checkpoint checkpoints/wavesanet_best.pth \
    --no-postprocess

# ── Ablation: raw single-frame detection only (no pre- or post-processing) ───
python evaluate.py \
    --model wavesanet \
    --checkpoint checkpoints/wavesanet_best.pth \
    --no-preprocess \
    --no-postprocess

# ── Custom metric thresholds ──────────────────────────────────────────────────
python evaluate.py \
    --model wavesanet \
    --checkpoint checkpoints/wavesanet_best.pth \
    --tau 10.0 \
    --epsilon 3.0

# ── Evaluate a baseline model ─────────────────────────────────────────────────
python evaluate.py \
    --model unet \
    --checkpoint checkpoints/unet_best.pth
```

> **Note**: when `--no-postprocess` is **not** passed, the script automatically
> runs a second pass without post-processing and prints the F1 / MSE improvement,
> so you always get both numbers in one call.

---

## Models & Ablations

Four architectures are available. They share the same U-Net skeleton but differ in
the attention/wavelet component — making them natural ablation baselines for each
other.

| `--model` | Description | Purpose |
|---|---|---|
| `wavesanet` | U-Net + **WGSCA** at every encoder level | Main model |
| `unet` | Plain U-Net, no attention | Ablation: proves WGSCA adds value |
| `unet_cbam` | U-Net + **CBAM** standard attention | Ablation: proves wavelet guidance > generic attention |
| `unet_wtnet` | U-Net + wavelet as **external pre-processing** only | Ablation: proves integrated wavelet > external wavelet |

### Recommended Ablation Study

```bash
# 1. Train all four models
python train.py --model wavesanet  --epochs 200
python train.py --model unet       --epochs 200
python train.py --model unet_cbam  --epochs 200
python train.py --model unet_wtnet --epochs 200

# 2. Evaluate each on the test set — full pipeline
python evaluate.py --model wavesanet  --checkpoint checkpoints/wavesanet_best.pth  \
    --data_dir ./data/spotgeo/test --annotations ./data/spotgeo/test_anno.json
python evaluate.py --model unet       --checkpoint checkpoints/unet_best.pth       \
    --data_dir ./data/spotgeo/test --annotations ./data/spotgeo/test_anno.json
python evaluate.py --model unet_cbam  --checkpoint checkpoints/unet_cbam_best.pth  \
    --data_dir ./data/spotgeo/test --annotations ./data/spotgeo/test_anno.json
python evaluate.py --model unet_wtnet --checkpoint checkpoints/unet_wtnet_best.pth \
    --data_dir ./data/spotgeo/test --annotations ./data/spotgeo/test_anno.json

# 3. Evaluate WaveSANet with pipeline stages removed
python evaluate.py --model wavesanet --checkpoint checkpoints/wavesanet_best.pth \
    --data_dir ./data/spotgeo/test --annotations ./data/spotgeo/test_anno.json \
    --no-preprocess
python evaluate.py --model wavesanet --checkpoint checkpoints/wavesanet_best.pth \
    --data_dir ./data/spotgeo/test --annotations ./data/spotgeo/test_anno.json \
    --no-postprocess
python evaluate.py --model wavesanet --checkpoint checkpoints/wavesanet_best.pth \
    --data_dir ./data/spotgeo/test --annotations ./data/spotgeo/test_anno.json \
    --no-preprocess --no-postprocess
```

---

## Pipeline Stages

The full detection pipeline has three stages, each of which can be toggled:

```
Raw 5-frame sequence
        │
        ▼
[1] Pre-processing  (--no-preprocess to skip)
    • Median frame subtraction  →  removes star background
    • Wavelet denoising (db4)   →  suppresses sensor noise
        │
        ▼
[2] Single-frame detection  (always runs)
    • WaveSANet / baseline model
    • Outputs a probability heatmap per frame
        │
        ▼
[3] Post-processing  (--no-postprocess to skip)
    • Heatmap → centroids (connected-component blobs)
    • Hungarian matching across frames
    • Trajectory interpolation / extrapolation
    • Temporal support filtering (noise removal)
        │
        ▼
(x, y) satellite detections per frame
```

---

## Output Files

| File | Location | Description |
|---|---|---|
| `<model>_best.pth` | `checkpoints/` | Best checkpoint (lowest validation loss) |
| `<model>_epoch_N.pth` | `checkpoints/` | Periodic checkpoint saved every 10 epochs |
| `<model>_history.json` | `checkpoints/` | Per-epoch train loss, val loss, and learning rate |

**Checkpoint contents**:
```python
{
    'epoch': int,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'val_loss': float,
    'train_loss': float,
    'history': {'train_loss': [...], 'val_loss': [...], 'lr': [...]},
}
```

---

## Metrics

Evaluation uses the **official SpotGEO challenge metrics** (Chen et al., CVPRW 2021):

| Metric | Description |
|---|---|
| **F1** | Primary ranking metric. Harmonic mean of precision and recall across all matched detections. |
| **1 − F1** | The actual competition score (lower is better). |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **MSE** | Mean squared error on matched detections (tie-breaker). |

**Matching**: predictions are matched to ground truth with the **Hungarian algorithm**.
A prediction is a **True Positive (TP)** if it falls within τ pixels of a ground truth
point (default `--tau 5.0`). The tolerance parameter ε (`--epsilon 2.0`) affects MSE
only: matched pairs within ε pixels contribute zero squared error.
