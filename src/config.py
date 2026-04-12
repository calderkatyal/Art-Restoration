"""Structured OmegaConf configs and YAML loading utilities.

All hyperparameters are controlled via configs/default.yaml.
Dot-notation CLI overrides are supported (e.g. train.stage=full).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from omegaconf import OmegaConf


@dataclass
class CorruptionConfig:
    """Controls the stochastic corruption pipeline.

    This configures both the preset selection probabilities and per-channel
    severity scaling, giving fine-grained control for curriculum learning.

    Attributes:
        damage_types:        Names of K=7 degradation channels, in channel order.
        num_channels:        K — must equal len(damage_types).
        individual_prob:     Probability of using an individual (single-type) preset
                             vs a multi-degradation preset. 0.0 = always multi,
                             1.0 = always individual.
        individual_presets:  Relative weight for each individual preset type.
                             Higher weight = more likely to be selected.
        multi_presets:       Relative weight for each multi-degradation preset.
        severity_scale:      Per-channel multiplier applied to mask values after
                             preset generation. Use for curriculum learning:
                             e.g. start with 0.5 and increase to 1.0.
    """
    damage_types: List[str] = field(default_factory=lambda: [
        "cracks", "paint_loss", "yellowing", "stains", "fading", "bloom", "deposits"
    ])
    num_channels: int = 7
    individual_prob: float = 0.4
    individual_presets: Dict[str, float] = field(default_factory=lambda: {
        "cracks": 1.0, "paint_loss": 1.0, "yellowing": 1.0,
        "stains": 1.0, "fading": 1.0, "bloom": 1.0, "deposits": 1.0,
    })
    multi_presets: Dict[str, float] = field(default_factory=lambda: {
        "light_aging": 1.0, "heavy_craquelure": 1.0, "water_damage": 1.0,
        "sun_faded": 1.0, "smoke_damage": 1.0, "neglected_storage": 1.0,
        "heat_damage": 1.0, "flood_damage": 1.0, "museum_wear": 1.0,
        "salt_efflorescence": 1.0,
    })
    severity_scale: Dict[str, float] = field(default_factory=lambda: {
        "cracks": 1.0, "paint_loss": 1.0, "yellowing": 1.0,
        "stains": 1.0, "fading": 1.0, "bloom": 1.0, "deposits": 1.0,
    })


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
    mask_channels: int = 7
    in_channels: int = 263          # 128 + 128 + 7
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
class WandbConfig:
    """Weights & Biases logging configuration.

    Attributes:
        enabled:          Whether to log to wandb at all.
        project:          Wandb project name.
        entity:           Wandb username or team. None = wandb default.
        run_name:         Display name for the run. None = auto-generated.
        tags:             List of string tags attached to the run.
        log_images:       Whether to log image grids (clean / corrupted / restored).
        log_images_every: Log an image grid every N optimizer steps.
    """
    enabled: bool = True
    project: str = "art-restoration"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    log_images: bool = True
    log_images_every: int = 500


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
        wandb:        WandbConfig.
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
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class CorruptionRefConfig:
    """Reference to the corruption config file from train.yaml.

    Attributes:
        config_path: Path to the corruption YAML config file.
    """
    config_path: str = "src/corruption/configs/default.yaml"


@dataclass
class Config:
    """Top-level config — mirrors the structure of configs/default.yaml."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    corruption: CorruptionRefConfig = field(default_factory=CorruptionRefConfig)


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
