"""Structured OmegaConf configs and YAML loading utilities."""

from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import OmegaConf, DictConfig, MISSING


@dataclass
class DegradationConfig:
    """Controls which synthetic degradations to apply and their severity ranges."""
    damage_types: List[str] = field(default_factory=lambda: [
        "crack", "paint_loss", "stain", "blur", "color_shift"
    ])
    num_channels: int = 5
    severity_range: List[float] = field(default_factory=lambda: [0.2, 0.8])
    max_simultaneous: int = 3


@dataclass
class ModelConfig:
    """Configuration for the FLUX.2-based restoration model."""
    flux_repo: str = "black-forest-labs/FLUX.1-schnell"
    vae_repo: str = "black-forest-labs/FLUX.1-schnell"
    latent_channels: int = 16
    mask_channels: int = 5
    in_channels: int = 37
    guidance_scale: float = 1.0
    null_emb_path: str = "null_text_emb.pt"


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters."""
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    """Learning rate scheduler config."""
    name: str = "cosine"
    warmup_steps: int = 500
    min_lr: float = 1e-6


@dataclass
class WarmupStageConfig:
    """Hyperparameters for warmup stage (only img_in is trained)."""
    epochs: int = 5
    lr: float = 1e-4


@dataclass
class FullStageConfig:
    """Hyperparameters for full stage (entire backbone fine-tuned)."""
    backbone_lr: float = 1e-5
    img_in_lr: float = 1e-4


@dataclass
class CurriculumConfig:
    """Curriculum learning: gradually introduce multi-degradation."""
    enabled: bool = True
    warmup_epochs: int = 5


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    stage: str = "warmup"  # "warmup" (img_in only) or "full" (all layers)
    data_dir: str = "./data/wikiart"
    val_dir: str = "./data/wikiart_val"
    output_dir: str = "./checkpoints"
    resolution: int = 512
    batch_size: int = 4
    num_epochs: int = 50
    seed: int = 42
    save_every: int = 1000
    log_every: int = 100
    resume_from: Optional[str] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    warmup: WarmupStageConfig = field(default_factory=WarmupStageConfig)
    full: FullStageConfig = field(default_factory=FullStageConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


@dataclass
class Config:
    """Top-level config, mirrors the YAML structure."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)


def load_config(yaml_path: str, overrides: Optional[List[str]] = None) -> Config:
    """Load config from YAML file with optional CLI overrides.

    Merges the YAML file with the structured defaults, then applies
    any dot-list overrides (e.g. ["train.stage=full", "train.batch_size=8"]).

    Args:
        yaml_path: Path to the YAML config file.
        overrides: Optional list of dot-notation overrides from CLI.

    Returns:
        Validated Config object.
    """
    ...
