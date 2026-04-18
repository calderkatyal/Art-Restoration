"""Shared helpers for training entrypoints.

Currently provides :func:`load_config`, which merges YAML with conservative DeepSpeed
defaults so older config files that omit ``ds_config`` still run ZeRO-2 + bf16 training.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

# Defaults merged under user YAML so older configs still run. Values here are the
# low-level knobs expected by ``deepspeed.initialize``; ``train.py`` overwrites
# ``train_micro_batch_size_per_gpu`` from ``cfg.train.batch_size`` at runtime.
_DEEPSPEED_DEFAULTS = OmegaConf.create(
    {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": 1,
        "steps_per_print": 100,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "wall_clock_breakdown": False,
    }
)


def load_config(yaml_path: str, overrides: Optional[List[str]] = None) -> DictConfig:
    """Load a training YAML file and apply optional OmegaConf dotlist overrides.

    Merge rules:
        * If the file has **no** ``ds_config`` key, inject ``_DEEPSPEED_DEFAULTS`` wholesale.
        * If ``ds_config`` exists, **deep-merge** defaults under each missing leaf so
          users can override only (say) ``zero_optimization.stage`` without restating bf16.
        * ``overrides`` (from unknown CLI args) are merged last so the shell always wins.

    Args:
        yaml_path: Path to YAML, typically ``train/configs/train.yaml``.
        overrides: Dotlist strings such as ``["train.batch_size=16"]``; ``None`` means none.

    Returns:
        A resolved :class:`omegaconf.DictConfig` ready for ``src.train.main``.

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist.
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")

    cfg = OmegaConf.load(path)
    if "ds_config" not in cfg or cfg.get("ds_config") is None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"ds_config": _DEEPSPEED_DEFAULTS}))
    else:
        cfg.ds_config = OmegaConf.merge(_DEEPSPEED_DEFAULTS, cfg.ds_config)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    OmegaConf.resolve(cfg)
    return cfg
