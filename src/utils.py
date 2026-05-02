"""Shared helpers for training and inference entrypoints.

Provides:
    * :func:`load_config` for loading the main training YAML.
    * :func:`load_corruption_config` for loading the corruption YAML referenced by
      ``cfg.corruption.config_path``.
    * :func:`sample` and :func:`data_consistency_step` for latent-space restoration.
    * :func:`print_training_phase` for logging active trainable parameter groups.
    * :func:`print_vram_debug` for optional rank-0 CUDA memory diagnostics.
"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import ImageDraw
import torch.distributed as dist
from torchvision.transforms import functional as TF

from .corruption import CHANNEL_NAMES, downsample_mask
from .distributed import get_world_size, is_main_process
from .flux2.sampling import get_schedule

if TYPE_CHECKING:
    from .model import RestorationDiT
    from .vae import FluxVAE

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

_MASK_OUTLINE_COLORS: dict[str, tuple[int, int, int]] = {
    "craquelure": (255, 80, 80),
    "rip_tear": (255, 170, 70),
    "paint_loss": (255, 235, 80),
    "yellowing": (110, 210, 110),
    "fading": (80, 210, 235),
    "deposits": (90, 130, 255),
    "scratches": (220, 110, 255),
}


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


def overlay_mask_boundaries(image: torch.Tensor, mask: torch.Tensor) -> Any:
    out = TF.to_pil_image(image.detach().cpu().clamp(0.0, 1.0))
    draw = ImageDraw.Draw(out)
    for channel_idx, name in enumerate(CHANNEL_NAMES):
        boundary = _mask_boundary(mask[channel_idx])
        if not boundary.any():
            continue
        color = _MASK_OUTLINE_COLORS.get(name, (255, 0, 0))
        ys, xs = torch.nonzero(boundary, as_tuple=True)
        for y, x in zip(ys.tolist(), xs.tolist()):
            draw.point((x, y), fill=color)
    return out


def wandb_log_images_num(cfg: DictConfig) -> int:
    wb = getattr(cfg, "wandb", None)
    if wb is None:
        return 0
    value = wb.get("log_images_num", 0)
    if value in (None, "", 0):
        return 0
    return max(0, int(value))


def fixed_inference_indices(cfg: DictConfig, dataset_len: int) -> list[int]:
    num_images = min(wandb_log_images_num(cfg), int(dataset_len))
    if num_images <= 0:
        return []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.train.seed) + 20_260_421)
    order = torch.randperm(int(dataset_len), generator=generator).tolist()
    return order[:num_images]


def gather_inference_panels(
    local_payload: list[tuple[int, torch.Tensor]],
    ordered_indices: list[int],
) -> torch.Tensor | None:
    payload: list[list[tuple[int, torch.Tensor]]] = [local_payload]
    if dist.is_available() and dist.is_initialized():
        payload = [None] * get_world_size()
        dist.all_gather_object(payload, local_payload)
    if not is_main_process():
        return None

    position = {idx: pos for pos, idx in enumerate(ordered_indices)}
    merged = [item for shard in payload for item in (shard or [])]
    merged.sort(key=lambda item: position[item[0]])
    if not merged:
        return None
    return torch.stack(
        [panel.to(dtype=torch.float32) / 255.0 for _, panel in merged],
        dim=0,
    )


@torch.no_grad()
def sample(
    model: "RestorationDiT",
    vae: "FluxVAE",
    corrupted_image: torch.Tensor,
    mask: torch.Tensor,
    null_emb: torch.Tensor,
    num_steps: int = 50,
    device: str = "cuda",
) -> torch.Tensor:
    """Restore a corrupted RGB batch by integrating the learned velocity field."""
    corrupted_image = corrupted_image.to(device)
    mask = mask.to(device)
    null_emb = null_emb.to(device)

    z_y = vae.encode(corrupted_image)
    m_lat = downsample_mask(mask, factor=vae.spatial_compression)
    b, _, h, w = z_y.shape
    z_t = torch.randn_like(z_y)
    seq_len = h * w
    timesteps = get_schedule(num_steps, seq_len)

    dtype = torch.bfloat16 if z_t.device.type == "cuda" else torch.float32
    z_t = z_t.to(dtype=dtype)
    z_y = z_y.to(dtype=dtype)
    m_lat = m_lat.to(dtype=dtype)
    null_b = null_emb.to(dtype=dtype)

    # Give the model the correct intact-context latent before the first
    # denoising step instead of starting from full-image noise.
    z_t = data_consistency_step(z_t, z_y, m_lat)

    for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((b,), float(t_curr), device=device, dtype=dtype)
        vel = model(z_t, t_vec, z_y, m_lat, null_b)
        z_t = z_t + (float(t_next) - float(t_curr)) * vel
        z_t = data_consistency_step(z_t, z_y, m_lat)

    restored = vae.decode(z_t.float())
    return restored.clamp(0.0, 1.0)


def data_consistency_step(
    z_t: torch.Tensor,
    z_y: torch.Tensor,
    mask_latent: torch.Tensor,
) -> torch.Tensor:
    """Copy intact latent regions from ``z_y`` while preserving damaged regions in ``z_t``."""
    m_dam = mask_latent.max(dim=1, keepdim=True).values
    m_dam = (m_dam >= 0.5).to(dtype=z_t.dtype)
    m_intact = 1.0 - m_dam
    return m_intact * z_y + m_dam * z_t


def checkpoint_tags_desc(checkpoint_dir: Path) -> list[str]:
    """Return ``step_*`` checkpoint tags sorted newest-first."""
    if not checkpoint_dir.exists():
        return []
    candidates = [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith("step_")]
    if not candidates:
        return []

    def key_fn(path: Path) -> tuple[int, float]:
        try:
            step = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            step = -1
        return (step, path.stat().st_mtime)

    return [p.name for p in sorted(candidates, key=key_fn, reverse=True)]


def summarize_train_loader_state(state: Any) -> str:
    """Summarize saved loader state so resume logs show where we expect to restart."""
    if not isinstance(state, dict):
        return "none"

    parts = [f"mode={state.get('mode', 'unknown')}"]
    sampler = state.get("sampler")
    if isinstance(sampler, dict):
        for key in ("epoch", "position", "length", "rank", "num_replicas", "num_samples"):
            if key in sampler:
                parts.append(f"sampler_{key}={sampler[key]}")

    dataset = state.get("dataset")
    if isinstance(dataset, dict):
        for key in ("worker_id", "worker_seed", "corruption_calls"):
            if key in dataset:
                parts.append(f"dataset_{key}={dataset[key]}")

    loader = state.get("loader")
    if isinstance(loader, dict):
        preview_keys = sorted(str(k) for k in loader.keys())[:6]
        parts.append(f"loader_keys={preview_keys}")
        for key in ("epoch", "position", "num_yielded", "_sampler_iter_yielded"):
            if key in loader:
                parts.append(f"loader_{key}={loader[key]}")

    return ", ".join(parts)


def summarize_runtime_loader_progress(
    train_dataset: Any,
    train_sampler: Any,
) -> str:
    """Summarize current sampler / dataset counters after a restore attempt."""
    parts: list[str] = []
    for attr in ("epoch", "position", "rank", "num_replicas", "num_samples"):
        if hasattr(train_sampler, attr):
            parts.append(f"sampler_{attr}={getattr(train_sampler, attr)}")
    for attr in ("_worker_id", "_worker_seed", "_corruption_calls"):
        if hasattr(train_dataset, attr):
            parts.append(f"dataset_{attr.lstrip('_')}={getattr(train_dataset, attr)}")
    return ", ".join(parts) if parts else "no sampler/dataset progress attributes available"


def summarize_next_sampler_batch(
    train_dataset: Any,
    train_sampler: Any,
    batch_size: int,
    limit: int = 4,
) -> str:
    """Summarize the next local batch implied by the sampler cursor."""
    parts: list[str] = []
    epoch = getattr(train_sampler, "epoch", None)
    position = getattr(train_sampler, "position", None)
    if epoch is not None:
        parts.append(f"sampler_epoch={epoch}")
    if position is not None:
        parts.append(f"sampler_position={position}")

    if not hasattr(train_sampler, "current_order") or position is None:
        return ", ".join(parts) if parts else "next batch cursor unavailable"

    order = list(train_sampler.current_order())
    start = int(position)
    stop = max(start, start + int(batch_size))
    next_indices = order[start:stop]
    parts.append(f"next_local_batch_indices={next_indices[:limit]}")

    image_paths = getattr(train_dataset, "image_paths", None)
    if image_paths is not None:
        next_paths = [Path(str(image_paths[idx])).name for idx in next_indices[:limit]]
        parts.append(f"next_local_batch_paths={next_paths}")

    return ", ".join(parts)


def summarize_saved_next_sampler_batch(
    train_dataset: Any,
    train_sampler: Any,
    sampler_state: Any,
    batch_size: int,
    limit: int = 4,
) -> str:
    """Summarize the next local batch implied by a saved sampler state."""
    if not isinstance(sampler_state, dict):
        return "saved sampler state unavailable"
    if not hasattr(train_sampler, "load_state_dict"):
        return "sampler restore preview unavailable"
    try:
        sampler_copy = copy.deepcopy(train_sampler)
        sampler_copy.load_state_dict(sampler_state)
    except Exception as exc:
        return f"saved sampler preview unavailable: {type(exc).__name__}: {exc}"
    return summarize_next_sampler_batch(
        train_dataset,
        sampler_copy,
        batch_size=batch_size,
        limit=limit,
    )


def _mask_boundary(mask: torch.Tensor) -> torch.Tensor:
    m = mask.detach().cpu() > 0.05
    if not m.any():
        return m
    up = torch.zeros_like(m)
    up[1:] = m[:-1]
    down = torch.zeros_like(m)
    down[:-1] = m[1:]
    left = torch.zeros_like(m)
    left[:, 1:] = m[:, :-1]
    right = torch.zeros_like(m)
    right[:, :-1] = m[:, 1:]
    return m & ~(m & up & down & left & right)

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
