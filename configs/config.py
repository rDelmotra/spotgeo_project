"""
Configuration for SpotGEO WaveSANet project.
All hyperparameters in one place — easy to tune and document.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    """Dataset paths and properties."""
    data_root: str = "./data/spotgeo"
    train_dir: str = "./data/spotgeo/train"
    test_dir: str = "./data/spotgeo/test"

    # SpotGEO image properties
    img_height: int = 480
    img_width: int = 640
    num_frames: int = 5          # frames per sequence
    num_sequences: int = 6400    # total sequences
    train_ratio: float = 0.2     # 20% train, 80% test (official split)

    # Label transformation
    label_window_size: int = 7   # m=3 → window = 2*3+1 = 7
    label_threshold: float = 0.5 # binarization threshold
    dilation_radius: int = 1     # morphological dilation


@dataclass
class PreprocessConfig:
    """Pre-processing parameters."""
    # Wavelet denoising
    wavelet: str = "db4"         # Daubechies-4 wavelet
    denoise_level: int = 2       # decomposition levels
    denoise_sigma: float = 15.0  # noise sigma estimate

    # Star removal
    use_median_subtraction: bool = True  # median frame subtraction


@dataclass
class ModelConfig:
    """Model architecture parameters."""
    # Encoder
    in_channels: int = 1              # grayscale input
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    decoder_channels: List[int] = field(default_factory=lambda: [128, 64, 32])

    # WGSCA module
    wavelet_type: str = "haar"        # wavelet for WGSCA (haar is fast)
    attention_reduction: int = 4      # channel attention squeeze ratio
    use_spatial_attention: bool = True
    use_channel_attention: bool = True

    # Output
    num_classes: int = 1              # binary: satellite or not


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 200  # can increase to 400 if early stopping doesn't trigger
    batch_size: int = 16
    early_stopping_patience: int = 30
    learning_rate: float = 0.0002
    weight_decay: float = 1e-4
    optimizer: str = "adamw"

    # Learning rate schedule
    scheduler: str = "cosine"         # cosine annealing
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    # Focal loss
    focal_alpha: float = 0.75         # weight for positive class
    focal_gamma: float = 2.0          # focusing parameter

    # Data augmentation
    use_horizontal_flip: bool = True
    use_vertical_flip: bool = True
    use_random_crop: bool = False     # images are already 640x480

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10              # save checkpoint every N epochs
    eval_every: int = 10              # evaluate every N epochs


@dataclass
class PostprocessConfig:
    """Post-processing (trajectory completion) parameters."""
    # Detection threshold
    confidence_threshold: float = 0.5  # heatmap → binary detection
    min_blob_area: int = 2             # minimum connected component area

    # Hungarian matching
    max_match_distance: float = 50.0   # max distance for valid match (pixels)

    # Temporal support filtering
    temporal_window: int = 2           # frames before/after to check
    support_ratio_threshold: float = 0.3
    spatial_distance_threshold: float = 30.0

    # Progressive completion
    max_interp_gap: int = 2            # max frame gap for interpolation
    aggressive_threshold_factor: float = 1.5


@dataclass
class EvalConfig:
    """Evaluation metric parameters (from official SpotGEO toolkit)."""
    matching_threshold: float = 5.0    # τ: matching distance threshold (pixels)
    tolerance_distance: float = 2.0    # ε: tolerance for labeling inaccuracy


@dataclass
class Config:
    """Master config combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 42
    num_workers: int = 4


def get_config() -> Config:
    """Return default configuration."""
    return Config()
