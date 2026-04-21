"""Shared helpers for training entrypoints.

Provides:
    * :func:`load_config` for loading the main training YAML.
    * :func:`load_corruption_config` for loading the corruption YAML referenced by
      ``cfg.corruption.config_path``.
    * :func:`print_training_phase` for logging active trainable parameter groups.
    * :func:`print_vram_debug` for optional rank-0 CUDA memory diagnostics.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from .distributed import is_main_process

# Defaults merged under user YAML so older configs still run. Values here are the
# low-level knobs expected by ``deepspeed.initialize``.
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


def log_message(message: str) -> None:
    """Print a timestamped log line."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _trainable_group_names(model: Any) -> list[str]:
    names: list[str] = []
    for group in model.get_trainable_params():
        params = group.get("params", [])
        if any(p.requires_grad for p in params):
            names.append(str(group.get("name", "unnamed")))
    return names


def print_training_phase(model: Any, warmup_only: bool, step: int) -> None:
    """Log the current training phase and active trainable parameter groups on rank 0."""
    if not is_main_process():
        return
    groups = _trainable_group_names(model)
    phase = "warmup" if warmup_only else "full training"
    trained = ", ".join(groups) if groups else "none"
    log_message(f"[train] Step {step}: {phase} active. Training layers: {trained}")


def print_vram_debug(
    cfg: Any,
    label: str,
    device: str | torch.device | None = None,
) -> None:
    """Print CUDA memory stats on rank 0 when ``train.debug_vram`` is enabled."""
    if not is_main_process():
        return
    train_cfg = getattr(cfg, "train", None)
    if train_cfg is None or not bool(train_cfg.get("debug_vram", False)):
        return
    if not torch.cuda.is_available():
        return

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, str):
        device = torch.device(device)
    if device.type != "cuda":
        return

    idx = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(idx)
    gb = 1024**3
    log_message(
        f"[vram] {label}: "
        f"allocated={torch.cuda.memory_allocated(idx) / gb:.2f}GB "
        f"max_allocated={torch.cuda.max_memory_allocated(idx) / gb:.2f}GB "
        f"reserved={torch.cuda.memory_reserved(idx) / gb:.2f}GB "
        f"max_reserved={torch.cuda.max_memory_reserved(idx) / gb:.2f}GB"
    )

def _resolve_config_path(config_path: str, config_dir: Optional[Path] = None) -> Path:
    """Resolve a config path against a few sensible roots.

    Relative paths are checked in this order:
        1. The directory containing the parent config file, if provided.
        2. The repository root.
        3. The current working directory.
    """
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path

    repo_root = Path(__file__).resolve().parent.parent
    candidates = []
    if config_dir is not None:
        candidates.append((config_dir / path).resolve())
    candidates.append((repo_root / path).resolve())
    candidates.append((Path.cwd() / path).resolve())

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_corruption_config(config_path: str, config_dir: Optional[Path] = None) -> DictConfig:
    """Load the corruption YAML referenced by ``cfg.corruption.config_path``.

    Raises:
        FileNotFoundError: If the referenced YAML does not exist.
    """
    resolved_path = _resolve_config_path(config_path, config_dir=config_dir)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Corruption config not found: {resolved_path}")

    cfg = OmegaConf.load(resolved_path)
    cfg.config_path = config_path
    return cfg


def load_config(yaml_path: str, overrides: Optional[List[str]] = None) -> DictConfig:
    """Load a training YAML file and apply optional OmegaConf dotlist overrides.

    Merge rules:
        * If the file has **no** ``ds_config`` key, inject ``_DEEPSPEED_DEFAULTS`` wholesale.
        * If ``ds_config`` exists, **deep-merge** defaults under each missing leaf so
          users can override only (say) ``zero_optimization.stage`` without restating bf16.
        * ``overrides`` (from unknown CLI args) are merged last so the shell always wins.

    Args:
        yaml_path: Path to YAML, typically ``train/configs/train.yaml``.
        overrides: Dotlist strings such as
                   ``["ds_config.train_micro_batch_size_per_gpu=16"]``; ``None`` means none.

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

    corruption_ref = cfg.get("corruption")
    if corruption_ref is None or not corruption_ref.get("config_path"):
        raise ValueError("Config must define corruption.config_path")

    loaded_corruption = load_corruption_config(
        str(corruption_ref.config_path),
        config_dir=path.parent,
    )
    cfg.corruption = OmegaConf.merge(loaded_corruption, corruption_ref)

    OmegaConf.resolve(cfg)
    return cfg
