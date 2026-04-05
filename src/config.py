"""Structured OmegaConf configs and YAML loading utilities.

All hyperparameters are controlled via configs/default.yaml.
Dot-notation CLI overrides are supported (e.g. train.stage=full).
"""

from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import OmegaConf


@dataclass
class DegradationConfig:
    """Controls which synthetic degradations are applied and their severity.

    Attributes:
        damage_types:     Names of K degradation channels, in channel order.
        num_channels:     K — must equal len(damage_types).
        severity_range:   [min, max] severity sampled uniformly per degradation.
        max_simultaneous: Max number of degradation types applied to one image.
    """
    damage_types: List[str] = field(default_factory=lambda: [
        "crack", "paint_loss", "stain", "blur", "color_shift"
    ])
    num_channels: int = 5
    severity_range: List[float] = field(default_factory=lambda: [0.2, 0.8])
    max_simultaneous: int = 3


@dataclass
class ModelConfig:
    """FLUX.2 [klein] 4B base model configuration.

    VAE shape reference (AutoEncoderParams defaults):
        z_channels=32, patchify ps=[2,2]  →  latent_channels = 32 * 2 * 2 = 128
        encoder 3× downsampling + patchify 2×  →  spatial_compression = 16
        input  (B, 3, H, W)  →  latent  (B, 128, H/16, W/16)

    DiT shape reference (Klein4BParams):
        original img_in: Linear(128 → 3072)
        re-initialized:  Linear(261 → 3072)
          where 261 = 128 (z_t) + 128 (z_y) + 5 (mask)
        context_in_dim = 3 Qwen3-4B hidden layers × 2560 = 7680
        use_guidance_embed = False  (base model, not distilled)

    Attributes:
        flux_model_name:      Key in flux2.util.FLUX2_MODEL_INFO.
        latent_channels:      VAE latent channels after patchify (128).
        spatial_compression:  Total VAE spatial downsampling factor (16).
        mask_channels:        K damage-type mask channels.
        in_channels:          img_in input dim = 128 + 128 + K = 261.
        hidden_size:          DiT hidden dimension (3072 for Klein 4B).
        context_in_dim:       Text context dim (7680).
        text_encoder_variant: Qwen3 variant string for load_qwen3_embedder.
        num_steps:            Euler ODE integration steps at inference.
        guidance:             CFG guidance scale.
        null_emb_path:        Path to cached null text embedding .pt file.
    """
    flux_model_name: str = "flux.2-klein-base-4b"
    latent_channels: int = 128
    spatial_compression: int = 16
    mask_channels: int = 5
    in_channels: int = 261          # 128 + 128 + 5
    hidden_size: int = 3072
    context_in_dim: int = 7680
    text_encoder_variant: str = "4B"
    num_steps: int = 50
    guidance: float = 4.0
    null_emb_path: str = "null_text_emb.pt"


@dataclass
class OptimizerConfig:
    """AdamW optimizer hyperparameters.

    Attributes:
        name:         Optimizer name (only "adamw" currently supported).
        lr:           Default learning rate (overridden per stage by TrainConfig).
        weight_decay: L2 regularization coefficient.
        betas:        Adam beta1, beta2.
    """
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    """Cosine LR scheduler with linear warmup.

    Attributes:
        name:          Scheduler name ("cosine").
        warmup_steps:  Linear warmup duration in optimizer steps.
        min_lr:        Minimum LR at end of cosine decay.
    """
    name: str = "cosine"
    warmup_steps: int = 500
    min_lr: float = 1e-6


@dataclass
class WarmupStageConfig:
    """Stage 1: only img_in is trained; backbone is fully frozen.

    Attributes:
        epochs: Number of epochs in this stage.
        lr:     Learning rate for img_in parameters.
    """
    epochs: int = 5
    lr: float = 1e-4


@dataclass
class FullStageConfig:
    """Stage 2: all layers are trainable with separate LRs per group.

    Attributes:
        backbone_lr: LR for pretrained transformer parameters.
        img_in_lr:   LR for the re-initialized img_in layer.
    """
    backbone_lr: float = 1e-5
    img_in_lr: float = 1e-4


@dataclass
class CurriculumConfig:
    """Curriculum: train on single degradation before introducing multi-degradation.

    Attributes:
        enabled:        Whether curriculum is active.
        warmup_epochs:  Epochs at max_simultaneous=1 before full multi-degradation.
    """
    enabled: bool = True
    warmup_epochs: int = 5


@dataclass
class TrainConfig:
    """All training hyperparameters.

    Attributes:
        stage:        "warmup" → freeze backbone, train img_in only.
                      "full"   → train all layers with separate per-group LRs.
        data_dir:     Root dir of clean WikiArt training images.
        val_dir:      Root dir of clean WikiArt validation images.
        output_dir:   Directory for checkpoints.
        resolution:   Square crop resolution (H = W = resolution).
        batch_size:   Per-GPU batch size.
        num_epochs:   Total training epochs across both stages.
        seed:         Random seed.
        save_every:   Checkpoint save interval in optimizer steps.
        log_every:    Loss logging interval in optimizer steps.
        resume_from:  Optional path to a checkpoint .pt file to resume from.
        optimizer:    OptimizerConfig.
        scheduler:    SchedulerConfig.
        warmup:       WarmupStageConfig.
        full:         FullStageConfig.
        curriculum:   CurriculumConfig.
    """
    stage: str = "warmup"
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
    """Top-level config — mirrors the structure of configs/default.yaml."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)


def load_config(yaml_path: str, overrides: Optional[List[str]] = None) -> Config:
    """Load config from YAML, merge with structured defaults, apply CLI overrides.

    Merges in order: structured defaults → YAML file → CLI overrides.

    Args:
        yaml_path: Path to YAML config file (e.g. "configs/default.yaml").
        overrides: Optional dot-notation override strings from CLI
                   (e.g. ["train.stage=full", "train.batch_size=8"]).

    Returns:
        Merged and type-validated Config object.
    """
    ...
