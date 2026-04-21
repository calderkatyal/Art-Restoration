"""Training loop for conditional latent rectified flow art restoration.

Rectified flow objective per training step:
    1. Sample clean image ``x``; dataset / corruption yields ``(y, M)``.
    2. ``z_1 = E(x)``, ``z_y = E(y)`` with frozen VAE (no gradients through ``E``).
    3. ``M' = downsample_mask(M, factor=spatial_compression)`` (e.g. 16 for FLUX VAE).
    4. ``z_0 ~ N(0, I)``, ``t ~ U(0, 1)``.
    5. ``z_t = (1 - t) * z_0 + t * z_1`` — note ``z_t`` interpolates **noise ↔ clean**;
       the damaged reference ``z_y`` is **separate conditioning**, not part of this mix.
    6. ``vel = v_θ(z_t, t | z_y, M', null_emb)``.
    7. Mask-weighted velocity MSE (``train.loss_weight_mask``, default ``0.7``) over latent
       union-mask vs outside; see :func:`compute_flow_loss`.
    8. ``engine.backward`` / ``engine.step`` (DeepSpeed); LR scheduler stepped by DeepSpeed.

Warm-up:
    For the first ``cfg.train.warmup_iterations`` optimizer steps, backbone weights stay
    frozen and only ``img_in`` is trained at ``cfg.train.warmup.lr``. After that,
    all layers are trainable and the optimizer switches to
    ``cfg.train.full.backbone_lr`` / ``cfg.train.full.img_in_lr``.

Distributed training:
    Launch with ``torchrun`` / DeepSpeed so ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE``
    are set. All ranks participate in ``save_checkpoint`` / ``load_checkpoint`` (ZeRO-2);
    logging, validation DataLoader, and WandB run on global rank 0 only.

Checkpointing:
    DeepSpeed checkpoints live under ``{train.checkpoint_root}/{wandb_run_name}/<tag>/``.
    ``train.save_every`` triggers saves every N **optimizer** steps;
    ``train.save_every_images`` triggers after approximately that many **global** images
    (``train_micro_batch_size_per_gpu × world_size × gradient_accumulation_steps``
    per optimizer step).
    Resume with ``train.resume_from`` pointing at a tag directory, e.g.
    ``./checkpoints/<run_name>/step_1000``.

Validation:
    All ranks run distributed held-out velocity-loss evaluation on the val split using
    the same corruption pipeline as training; the aggregated result is logged to WandB
    on rank 0.

Usage:
    python -m src.train [--config train/configs/train.yaml] [overrides...]

Arguments:
    --config    Path to YAML (default: ``train/configs/train.yaml``).
    overrides   Dot-notation overrides, e.g. ``train.warmup_iterations=1000``,
                ``ds_config.train_micro_batch_size_per_gpu=8``,
    ``train.resume_from=/nfs/roberts/project/cpsc4520/cpsc4520_ckk25/checkpoints/<run_name>/step_1000``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import os
import time
from pathlib import Path
from typing import Any, Optional

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .corruption import OUTPUT_MASK_CHANNELS, downsample_mask
from .dataset import ArtRestorationDataset, build_wikiart_dataloader
from .distributed import get_device, get_global_rank, get_world_size, is_main_process
from .inference import sample
from .model import RestorationDiT
from .null_emb import load_or_compute_null_embedding
from .utils import (
    fixed_inference_indices,
    gather_inference_panels,
    load_config,
    log_message,
    overlay_mask_boundaries,
    print_training_phase,
    print_vram_debug,
    wandb_log_images_num,
)
from .vae import FluxVAE

# True only on rank 0 after ``wandb.init``; other ranks skip WandB entirely.
_WANDB_ACTIVE = False
_WANDB_RUN_NAME: Optional[str] = None


def wandb_init(cfg: Any, train_cfg: Any) -> Optional[str]:
    """Initialize a WandB run if ``cfg.wandb.enabled`` and this is global rank 0.

    Non-main ranks and disabled config leave ``_WANDB_ACTIVE`` False so ``wandb_log``
    stays a cheap no-op.

    Args:
        cfg:       Top-level config; reads ``cfg.wandb`` (``project``, ``entity``,
                   ``run_name``, ``tags``, ``enabled``).
        train_cfg: Full training config dict-like object serialized into ``wandb.config``
                   via ``OmegaConf.to_container`` for reproducibility.
    """
    global _WANDB_ACTIVE, _WANDB_RUN_NAME
    wb = getattr(cfg, "wandb", None)
    configured_run_name = wb.get("run_name") if wb is not None else None
    if not configured_run_name:
        raise ValueError("cfg.wandb.run_name must be set explicitly.")
    if wb is None or not wb.get("enabled", False) or not is_main_process():
        _WANDB_ACTIVE = False
        _WANDB_RUN_NAME = configured_run_name
        return _WANDB_RUN_NAME
    import wandb

    tags = list(wb.get("tags") or [])
    run = wandb.init(
        project=wb.get("project", "art-restoration"),
        entity=wb.get("entity") or None,
        name=configured_run_name,
        tags=tags if tags else None,
        config=OmegaConf.to_container(train_cfg, resolve=True),
    )
    _WANDB_ACTIVE = True
    _WANDB_RUN_NAME = configured_run_name
    return _WANDB_RUN_NAME


def wandb_log(metrics: dict, step: int, images: dict | None = None) -> None:
    """Log scalar metrics and optional image grids to WandB.

    If ``_WANDB_ACTIVE`` is False (WandB disabled or non-zero rank), this is a no-op.

    Args:
        metrics: Dict of scalar names to floats, e.g. ``{"train/loss": 0.04}``.
        step:    Global optimizer step used as the WandB x-axis.
        images:  Optional mapping from run key to ``(B, 3, H, W)`` tensors in ``[0,1]``.
                 Each value is turned into a ``wandb.Image`` grid via ``make_grid``.
    """
    if not _WANDB_ACTIVE:
        return
    import wandb
    from torchvision.utils import make_grid

    payload = dict(metrics)
    if images:
        for name, tensor in images.items():
            if tensor is None:
                continue
            grid = make_grid(tensor.detach().cpu(), nrow=min(4, tensor.shape[0]))
            payload[name] = wandb.Image(grid.permute(1, 2, 0).numpy())
    wandb.log(payload, step=step)


def wandb_finish() -> None:
    """Call ``wandb.finish()`` on rank 0 and clear ``_WANDB_ACTIVE``."""
    global _WANDB_ACTIVE, _WANDB_RUN_NAME
    if not _WANDB_ACTIVE:
        return
    import wandb

    wandb.finish()
    _WANDB_ACTIVE = False
    _WANDB_RUN_NAME = None


def _sanitize_run_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    return cleaned or "run"


def _resolve_inspect_run_name(cfg: DictConfig) -> str:
    wb = getattr(cfg, "wandb", None)
    if wb is None or not wb.get("run_name"):
        raise ValueError("cfg.wandb.run_name must be set explicitly.")
    return _sanitize_run_name(wb.get("run_name"))


def _inspect_data_every(cfg: DictConfig) -> int:
    value = cfg.train.get("inspect_data_every")
    if value in (None, "", 0):
        return 0
    return max(0, int(value))


def run_validation_inference(
    model_engine: Any,
    vae: FluxVAE,
    val_dataset: ArtRestorationDataset,
    null_emb: torch.Tensor,
    cfg: DictConfig,
    device: torch.device,
    global_step: int,
) -> torch.Tensor | None:
    indices = fixed_inference_indices(cfg, len(val_dataset))
    if not indices:
        return None

    rank = get_global_rank()
    world_size = get_world_size()
    local_indices = indices[rank::world_size]
    local_payload: list[tuple[int, torch.Tensor]] = []

    mdl = _unwrap_model(model_engine)
    was_training = mdl.training
    if hasattr(model_engine, "eval"):
        model_engine.eval()
    else:
        mdl.eval()

    with torch.no_grad():
        if local_indices:
            samples = [val_dataset[idx] for idx in local_indices]
            clean = torch.stack([sample["clean"] for sample in samples], dim=0).to(device)
            corrupted = torch.stack([sample["corrupted"] for sample in samples], dim=0).to(device)
            mask = torch.stack([sample["mask"] for sample in samples], dim=0).to(device)
            restored = sample(
                mdl,
                vae,
                corrupted,
                mask,
                null_emb,
                num_steps=int(getattr(cfg.inference, "num_steps", 50)),
                device=str(device),
            )
            for idx, clean_img, corrupt_img, restored_img in zip(
                local_indices, clean, corrupted, restored
            ):
                panel = torch.cat(
                    [clean_img.detach().cpu(), corrupt_img.detach().cpu(), restored_img.detach().cpu()],
                    dim=2,
                ).clamp(0.0, 1.0)
                local_payload.append((idx, (panel * 255.0).round().to(torch.uint8)))

    if was_training:
        if hasattr(model_engine, "train"):
            model_engine.train()
        else:
            mdl.train()

    panels = gather_inference_panels(local_payload, indices)
    if is_main_process() and panels is not None:
        log_message(
            f"[inference] prepared {int(panels.shape[0])} validation panels for step={global_step}"
        )
    return panels


def _maybe_save_inspect_batch(
    batch: dict[str, torch.Tensor],
    cfg: DictConfig,
    run_name: str,
    global_step: int,
    rank: int,
) -> None:
    inspect_every = _inspect_data_every(cfg)
    if inspect_every <= 0 or global_step <= 0 or global_step % inspect_every != 0:
        return

    root = cfg.train.get("inspect_data_root")
    if root in (None, ""):
        return

    out_dir = Path(str(root)) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    corrupted = batch.get("corrupted")
    if corrupted is None or corrupted.shape[0] == 0:
        return

    out_path = out_dir / f"step_{int(global_step):08d}_rank_{int(rank):02d}.png"
    log_message(f"[inspect] rank={rank} saving training image to {out_path}")
    image = corrupted[0]
    mask = batch.get("mask")
    if mask is not None and mask.shape[0] > 0:
        pil = overlay_mask_boundaries(image, mask[0])
    else:
        pil = TF.to_pil_image(image.detach().cpu().clamp(0.0, 1.0))
    pil.save(out_path)
    log_message(f"[inspect] rank={rank} saved training image to {out_path}")


def _unwrap_model(engine: Any) -> RestorationDiT:
    """Return the underlying ``RestorationDiT`` from a DeepSpeed engine or plain module.

    DeepSpeed wraps the user module in ``engine.module``; in unit tests or non-DS code
    the object may already be the raw ``nn.Module``.
    """
    return engine.module if hasattr(engine, "module") else engine


def setup_model(
    cfg: DictConfig,
    device: torch.device,
    rank: int,
    warmup_only: bool,
    load_pretrained: bool = True,
) -> tuple[RestorationDiT, FluxVAE, torch.Tensor]:
    """Construct the restoration DiT, frozen FLUX VAE, and cached null text embedding.

    Trainability is applied after DeepSpeed initialization, not here. That keeps
    optimizer param groups non-empty during ZeRO setup while still allowing the
    warm-up phase to freeze the backbone immediately afterward.

    Args:
        cfg:    Full OmegaConf ``DictConfig`` (uses ``cfg.model`` and ``cfg.train``).
        device: CUDA device for this process (typically ``cuda:LOCAL_RANK``).
        rank:   Global rank, forwarded to weight download logging and VAE init.
        warmup_only: Whether training should start in img_in-only warm-up mode.
        load_pretrained: Whether to load FLUX pretrained weights before widening ``img_in``.

    Returns:
        Tuple ``(restoration_dit, flux_vae, null_emb)`` where ``null_emb`` has shape
        ``(1, 512, 7680)`` and dtype ``bfloat16`` on ``device``.
    """
    flow_model = RestorationDiT(
        cfg=cfg.model,
        gradient_checkpointing=bool(getattr(cfg.train, "gradient_checkpointing", False)),
        device=device,
        img_in_dtype=torch.bfloat16,
        load_pretrained=load_pretrained,
        rank=rank,
    )
    print_vram_debug(cfg, "after_loading_model_weights", device=device)
    if load_pretrained:
        log_message(f"[model] finished loading FLUX.2 pretrained weights for {cfg.model.flux_model_name}")
    else:
        log_message(f"[model] initialized {cfg.model.flux_model_name} without pretrained FLUX.2 weights")

    vae = FluxVAE(
        flux_model_name=cfg.model.flux_model_name,
        rank=rank,
        device=device,
    )
    print_vram_debug(cfg, "after_loading_vae", device=device)

    null_emb = load_or_compute_null_embedding(
        cache_path=cfg.model.null_emb_path,
        flux_model_name=cfg.model.flux_model_name,
        device=device,
    )
    print_vram_debug(cfg, "after_loading_null_emb", device=device)
    return flow_model, vae, null_emb


def build_optimizer(
    model: RestorationDiT,
    cfg: DictConfig,
    warmup_only: bool,
) -> torch.optim.Optimizer:
    """Build AdamW for either warm-up or full training.

    Warm-up keeps the backbone frozen and sets its optimizer LR to zero. After warm-up,
    ``img_in`` and backbone use their full-training learning rates.

    Weight decay and Adam ``betas`` come from ``cfg.train.optimizer``.

    Args:
        model: ``RestorationDiT`` with ``requires_grad`` flags already set for the phase.
        cfg:   Full config.
        warmup_only: Whether to start in img_in-only warm-up mode.

    Returns:
        A ``torch.optim.AdamW`` instance suitable for passing to ``deepspeed.initialize``.
    """
    wd = cfg.train.optimizer.weight_decay
    betas = tuple(cfg.train.optimizer.betas)
    groups = model.get_trainable_params()
    group_stats = []
    for group in groups:
        params = list(group["params"])
        trainable = sum(int(p.requires_grad) for p in params)
        group_stats.append(f"{group['name']} total={len(params)} trainable={trainable}")
    log_message(f"[optimizer] building param groups: {', '.join(group_stats)}")
    img_in_lr = float(cfg.train.warmup.lr) if warmup_only else float(cfg.train.full.img_in_lr)
    backbone_lr = 0.0 if warmup_only else float(cfg.train.full.backbone_lr)
    return torch.optim.AdamW(
        [
            {"params": groups[0]["params"], "lr": img_in_lr, "name": "img_in"},
            {"params": groups[1]["params"], "lr": backbone_lr, "name": "backbone"},
        ],
        betas=betas,
        weight_decay=wd,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    total_optimizer_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup, implemented as a single ``LambdaLR`` multiplier.

    The multiplier applies equally to all param groups; their *absolute* LRs differ
    if the optimizer used different base rates per group.

    Warmup ramps the multiplier linearly from ``1/warmup_steps`` to ``1`` over
    ``warmup_steps``. Afterward a cosine drops the multiplier from ``1`` down to
    ``min_lr / ref_lr`` where ``ref_lr`` is the first group's initial LR and
    ``min_lr`` is ``cfg.train.scheduler.min_lr``.

    Args:
        optimizer:             AdamW instance (already constructed with per-group LRs).
        cfg:                   Full config; reads ``cfg.train.scheduler``.
        total_optimizer_steps: Approximate total steps (``epochs * steps_per_epoch / grad_accum``);
                               ``max_steps`` from config is clamped to be at least this value.

    Returns:
        ``torch.optim.lr_scheduler.LambdaLR`` — DeepSpeed steps it inside ``engine.step``.
    """
    warmup = int(cfg.train.scheduler.warmup_steps)
    min_lr = float(cfg.train.scheduler.min_lr)
    max_steps = int(getattr(cfg.train.scheduler, "max_steps", total_optimizer_steps))
    max_steps = max(max_steps, total_optimizer_steps)
    ref_lr = max(float(optimizer.param_groups[0]["lr"]), 1e-12)
    min_ratio = min_lr / ref_lr

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        denom = max(1, max_steps - warmup)
        progress = min(1.0, float(step - warmup) / float(denom))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def _train_dir(cfg: DictConfig) -> str:
    value = cfg.train.get("train_dir", cfg.train.get("data_dir"))
    if value in (None, ""):
        raise ValueError("Config must define train.train_dir")
    return str(value)


def _val_dir(cfg: DictConfig) -> str:
    value = cfg.train.get("val_dir")
    if value in (None, ""):
        raise ValueError("Config must define train.val_dir")
    return str(value)


def _micro_batch_size(cfg: DictConfig) -> int:
    """Return the per-GPU batch size and enforce config consistency."""
    train_batch = cfg.train.get("batch_size")
    ds_micro = cfg.ds_config.get("train_micro_batch_size_per_gpu")

    if train_batch in (None, "auto") and ds_micro in (None, "auto"):
        raise ValueError(
            "Set either train.batch_size or ds_config.train_micro_batch_size_per_gpu."
        )

    if train_batch in (None, "auto"):
        batch_size = int(ds_micro)
        cfg.train.batch_size = batch_size
        return batch_size

    batch_size = int(train_batch)
    if ds_micro in (None, "auto"):
        cfg.ds_config.train_micro_batch_size_per_gpu = batch_size
        return batch_size

    if int(ds_micro) != batch_size:
        raise ValueError(
            "train.batch_size must match ds_config.train_micro_batch_size_per_gpu "
            f"(got {batch_size} vs {int(ds_micro)})."
        )
    return batch_size


def _checkpoint_root(cfg: DictConfig) -> Path:
    root = cfg.train.get("checkpoint_root")
    if root in (None, ""):
        root = cfg.train.get("checkpoint_dir")
    if root in (None, ""):
        raise ValueError("Config must define train.checkpoint_root")
    return Path(str(root))


def _checkpoint_dir_for_run(cfg: DictConfig, run_name: str) -> Path:
    return _checkpoint_root(cfg) / run_name


def _distributed_barrier() -> None:
    """Synchronize all ranks when distributed training is initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _warmup_iterations(cfg: DictConfig) -> int:
    return max(0, int(getattr(cfg.train, "warmup_iterations", 0)))


def _warmup_only_for_step(cfg: DictConfig, step: int) -> bool:
    return int(step) < _warmup_iterations(cfg)


def _step_from_tag(tag: str | None) -> int:
    if not tag:
        return 0
    try:
        return max(0, int(str(tag).split("_", 1)[1]))
    except (IndexError, ValueError):
        return 0


def _phase_lrs(cfg: DictConfig, warmup_only: bool) -> dict[str, float]:
    return {
        "img_in": float(cfg.train.warmup.lr) if warmup_only else float(cfg.train.full.img_in_lr),
        "backbone": 0.0 if warmup_only else float(cfg.train.full.backbone_lr),
    }


def _apply_training_phase(
    model: RestorationDiT,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cfg: DictConfig,
    step: int,
) -> bool:
    warmup_only = _warmup_only_for_step(cfg, step)
    model.set_trainability(warmup_only)
    lrs = _phase_lrs(cfg, warmup_only)
    base_lrs: list[float] = []
    for group in optimizer.param_groups:
        lr = lrs.get(group.get("name", ""), float(group["lr"]))
        group["lr"] = lr
        group["initial_lr"] = lr
        base_lrs.append(lr)
    if scheduler is not None and hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = list(base_lrs)
    return warmup_only


def setup_dataloader(
    cfg: DictConfig,
    split: str = "train",
) -> tuple[DataLoader, ArtRestorationDataset]:
    """Create a train or val dataloader using the resumable dataset helpers."""
    dataset, loader, _ = build_wikiart_dataloader(
        train_dir=_train_dir(cfg),
        val_dir=_val_dir(cfg),
        resolution=int(cfg.train.resolution),
        corruption_config=cfg.corruption,
        batch_size=_micro_batch_size(cfg),
        split=split,
        num_workers=int(getattr(cfg.train, "num_workers", 4)),
        sampler_seed=int(getattr(cfg.train, "sampler_seed", cfg.train.seed)),
        corruption_seed=int(getattr(cfg.train, "corruption_seed", cfg.train.seed)),
        pin_memory=bool(getattr(cfg.train, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg.train, "persistent_workers", True)),
        prefetch_factor=int(getattr(cfg.train, "prefetch_factor", 4)),
        snapshot_every_n_steps=int(getattr(cfg.train, "snapshot_every_n_steps", 1)),
        distributed=False,
        drop_last=False,
        return_metadata=False,
    )
    return loader, dataset


def compute_flow_loss(
    model_engine: Any,
    vae: FluxVAE,
    batch: dict,
    null_emb: torch.Tensor,
    spatial_compression: int,
    device: torch.device,
    loss_weight_mask: float,
) -> torch.Tensor:
    """Compute rectified-flow MSE loss for one micro-batch.

    Expects ``batch`` keys ``"clean"``, ``"corrupted"``, ``"mask"`` as emitted by
    :class:`~src.dataset.ArtRestorationDataset`. Encodings use the frozen VAE in
    ``float32``; the velocity target ``(z_1 - z_0)`` is ``float32``; model predictions
    are cast to ``float32`` for the MSE so bf16 forward does not underflow the loss.

    The MSE is split by the **latent** union mask (last mask channel after pooling):
    ``loss_weight_mask * mse_inside + (1 - loss_weight_mask) * mse_outside``, where
    inside/outside are averaged over all elements of ``(vel - target)^2`` where the
    union mask is active (>= 0.5) or inactive, respectively.

    ``model_engine`` may be a DeepSpeed ``DeepSpeedEngine`` (``__call__`` forwards to
    ``RestorationDiT.forward``) or a bare ``RestorationDiT`` for tests.

    Args:
        model_engine:        Callable ``(z_t, t, z_y, M', null_emb) -> velocity``.
        vae:                 Frozen :class:`~src.vae.FluxVAE`.
        batch:               Dict of CPU/GPU tensors from the dataloader.
        null_emb:            Cached text embedding ``(1, 512, 7680)`` on ``device``.
        spatial_compression: VAE spatial factor (16) for ``downsample_mask``.
        device:              Device to run the loss on.
        loss_weight_mask:    Scalar in ``[0, 1]`` weighting inside-mask vs outside-mask MSE.

    Returns:
        Scalar loss tensor with ``grad_fn`` through ``model_engine`` parameters.
    """
    clean = batch["clean"].to(device, non_blocking=True)
    corrupted = batch["corrupted"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    with torch.no_grad():
        z_1 = vae.encode(clean.float())
        z_y = vae.encode(corrupted.float())

    m_prime = downsample_mask(mask, factor=spatial_compression)
    if mask.shape[1] != OUTPUT_MASK_CHANNELS or m_prime.shape[1] != OUTPUT_MASK_CHANNELS:
        raise ValueError(
            f"Expected mask with {OUTPUT_MASK_CHANNELS} channels "
            f"(7 per-type + union), got pixel {mask.shape[1]} latent {m_prime.shape[1]}"
        )

    z_0 = torch.randn_like(z_1)
    t = torch.rand((z_1.shape[0],), device=device, dtype=z_1.dtype)
    t_bc = t.view(-1, 1, 1, 1)
    z_t = (1.0 - t_bc) * z_0 + t_bc * z_1

    vel = model_engine(z_t, t, z_y, m_prime, null_emb)
    target = (z_1 - z_0).float()
    sq = (vel.float() - target).pow(2)
    # Last mask channel is union of per-type masks at latent resolution.
    dam = (m_prime[:, -1:] >= 0.5).to(dtype=sq.dtype).expand_as(sq)
    eps = 1e-8
    sse_in = (sq * dam).sum()
    sse_out = (sq * (1.0 - dam)).sum()
    den_in = dam.sum() + eps
    den_out = (1.0 - dam).sum() + eps
    loss_in = sse_in / den_in
    loss_out = sse_out / den_out
    lw = max(0.0, min(1.0, float(loss_weight_mask)))
    return lw * loss_in + (1.0 - lw) * loss_out


def validate(
    model_engine: Any,
    vae: FluxVAE,
    val_loader: DataLoader,
    null_emb: torch.Tensor,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    """Compute distributed held-out velocity loss on the validation set.

    This mirrors training-time loss evaluation: for each validation batch, it calls
    :func:`compute_flow_loss` with the same synthetic corruption inputs but does not
    backpropagate or step the optimizer. Losses are aggregated across all ranks via
    ``all_reduce`` and returned as a single global average.

    Args:
        model_engine: Wrapped or bare ``RestorationDiT``.
        vae:          Frozen VAE.
        val_loader:   Batches of ``clean``, ``corrupted``, ``mask``.
        null_emb:     Null text embedding.
        cfg:          Uses ``cfg.model.spatial_compression``.
        device:       CUDA device.

    Returns:
        Dict with float key ``"velocity_loss"`` aggregated across all validation samples.
    """
    mdl = _unwrap_model(model_engine)
    was_training = mdl.training
    if hasattr(model_engine, "eval"):
        model_engine.eval()
    else:
        mdl.eval()

    loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
    sample_count = torch.zeros(1, device=device, dtype=torch.float64)
    with torch.no_grad():
        for batch in val_loader:
            loss = compute_flow_loss(
                model_engine,
                vae,
                batch,
                null_emb,
                int(cfg.model.spatial_compression),
                device,
                float(getattr(cfg.train, "loss_weight_mask", 0.7)),
            )
            batch_size = int(batch["clean"].shape[0])
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)

    if was_training:
        if hasattr(model_engine, "train"):
            model_engine.train()
        else:
            mdl.train()

    denom = max(sample_count.item(), 1.0)
    return {
        "velocity_loss": float((loss_sum / denom).item()),
    }


def _write_checkpoint_metadata(output_dir: Path, tag: str, client_state: dict) -> None:
    """Persist a small JSON sidecar next to DeepSpeed checkpoints (rank 0 only).

    DeepSpeed already stores ``client_state`` inside its checkpoint payload; this
    file is a human-readable duplicate for HPC inspection and bookkeeping.

    Args:
        output_dir:   Root directory.
        tag:          Checkpoint tag, e.g. ``step_1000``.
        client_state: Pickle-friendly dict (``step``, ``epoch``, etc.).
    """
    if not is_main_process():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / f"run_meta_{tag}.json"

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu()
            if tensor.ndim == 0:
                return tensor.item()
            return {
                "type": "tensor",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return obj

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(client_state), f, indent=2)


def maybe_save_deepspeed_checkpoint(
    engine: Any,
    cfg: DictConfig,
    client_state: dict,
    do_save: bool,
    run_name: str,
) -> None:
    """Synchronously save a ZeRO-aware checkpoint on **all** ranks when ``do_save``.

    Under ZeRO-2, every process must enter ``engine.save_checkpoint``; rank 0 alone
    writes ``run_meta_<tag>.json`` via :func:`_write_checkpoint_metadata`.

    Args:
        engine:        DeepSpeed-wrapped model.
        cfg:           Provides ``train.checkpoint_root``.
        client_state:  Metadata forwarded to DeepSpeed (also embedded in the checkpoint).
        do_save:       When False, returns immediately without touching the filesystem.
        run_name:      WandB run name used to scope checkpoint output directories.
    """
    if not do_save:
        return
    out = _checkpoint_dir_for_run(cfg, run_name)
    tag = f"step_{int(client_state['step'])}"
    log_message(f"[checkpoint] starting save for {tag} in {out}")
    engine.save_checkpoint(str(out), tag=tag, client_state=client_state)
    if is_main_process():
        _write_checkpoint_metadata(out, tag, client_state)
    log_message(f"[checkpoint] finished save for {tag} in {out}")


def train(cfg: DictConfig) -> None:
    """Public entry alias that forwards to :func:`main`.

    Kept for symmetry with documentation / external callers that import ``train``.
    """
    main(cfg)


def _capture_train_loader_state(
    train_loader: DataLoader,
    train_dataset: ArtRestorationDataset,
    train_sampler: Any,
) -> dict[str, Any]:
    if hasattr(train_loader, "state_dict"):
        return {
            "mode": "loader",
            "loader": train_loader.state_dict(),
        }
    state: dict[str, Any] = {"mode": "manual"}
    if hasattr(train_sampler, "state_dict"):
        state["sampler"] = train_sampler.state_dict()
    if hasattr(train_dataset, "state_dict"):
        state["dataset"] = train_dataset.state_dict()
    return state


def _restore_train_loader_state(
    train_loader: DataLoader,
    train_dataset: ArtRestorationDataset,
    train_sampler: Any,
    state: Any,
) -> bool:
    if not state:
        return False
    if state.get("mode") == "loader" and hasattr(train_loader, "load_state_dict"):
        train_loader.load_state_dict(state["loader"])
        return True
    if "sampler" in state and hasattr(train_sampler, "load_state_dict"):
        train_sampler.load_state_dict(state["sampler"])
    if "dataset" in state and hasattr(train_dataset, "load_state_dict"):
        train_dataset.load_state_dict(state["dataset"])
    return True


def _build_train_loader(
    cfg: DictConfig,
    corrupt_cfg: DictConfig,
    world_size: int,
) -> tuple[DataLoader, ArtRestorationDataset, Any]:
    """Build the training dataloader plus its dataset and sampler."""
    dataset, loader, sampler = build_wikiart_dataloader(
        train_dir=_train_dir(cfg),
        val_dir=_val_dir(cfg),
        resolution=int(cfg.train.resolution),
        corruption_config=corrupt_cfg,
        batch_size=_micro_batch_size(cfg),
        split="train",
        num_workers=int(getattr(cfg.train, "num_workers", 4)),
        deterministic_corruption=False,
        sampler_seed=int(getattr(cfg.train, "sampler_seed", cfg.train.seed)),
        corruption_seed=int(getattr(cfg.train, "corruption_seed", cfg.train.seed)),
        pin_memory=bool(getattr(cfg.train, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg.train, "persistent_workers", True)),
        prefetch_factor=int(getattr(cfg.train, "prefetch_factor", 4)),
        snapshot_every_n_steps=int(getattr(cfg.train, "snapshot_every_n_steps", 1)),
        distributed=world_size > 1,
        num_replicas=world_size if world_size > 1 else None,
        rank=get_global_rank() if world_size > 1 else None,
        drop_last=False,
        return_metadata=False,
    )
    return loader, dataset, sampler


def _build_val_loader(
    cfg: DictConfig,
    corrupt_cfg: DictConfig,
    world_size: int,
) -> tuple[DataLoader, ArtRestorationDataset, Any]:
    """Build validation ``DataLoader`` and optional ``DistributedSampler``."""
    dataset, loader, sampler = build_wikiart_dataloader(
        train_dir=_train_dir(cfg),
        val_dir=_val_dir(cfg),
        resolution=int(cfg.train.resolution),
        corruption_config=corrupt_cfg,
        batch_size=_micro_batch_size(cfg),
        split="val",
        num_workers=int(getattr(cfg.train, "num_workers", 4)),
        sampler_seed=int(getattr(cfg.train, "sampler_seed", cfg.train.seed)),
        corruption_seed=int(getattr(cfg.train, "corruption_seed", cfg.train.seed)),
        pin_memory=bool(getattr(cfg.train, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg.train, "persistent_workers", True)),
        prefetch_factor=int(getattr(cfg.train, "prefetch_factor", 4)),
        snapshot_every_n_steps=int(getattr(cfg.train, "snapshot_every_n_steps", 1)),
        distributed=world_size > 1,
        num_replicas=world_size if world_size > 1 else None,
        rank=get_global_rank() if world_size > 1 else None,
        drop_last=False,
        return_metadata=False,
    )
    return loader, dataset, sampler


def _latest_checkpoint_tag(checkpoint_dir: Path) -> str | None:
    """Return most recent ``step_*`` checkpoint tag under ``checkpoint_dir``."""
    if not checkpoint_dir.exists():
        return None
    candidates = [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith("step_")]
    if not candidates:
        return None

    def key_fn(path: Path) -> tuple[int, float]:
        try:
            step = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            step = -1
        return (step, path.stat().st_mtime)

    return max(candidates, key=key_fn).name


def _resolve_checkpoint_strategy(cfg: DictConfig) -> tuple[str | None, str | None]:
    """Choose one loading source: explicit resume, latest checkpoint, or pretrained."""
    resume_from = cfg.train.get("resume_from")
    if resume_from and str(resume_from) != "from_scratch":
        rp = Path(str(resume_from))
        return str(rp.parent), rp.name

    ckpt_dir = _checkpoint_dir_for_run(cfg, _resolve_inspect_run_name(cfg))
    tag = _latest_checkpoint_tag(ckpt_dir)
    if tag is None:
        return None, None
    return str(ckpt_dir), tag


def main(cfg: DictConfig) -> None:
    """Initialize distributed training, run epochs, log, validate, and checkpoint.

    Pipeline overview:
        1. ``deepspeed.init_distributed`` (NCCL).
        2. Build model / VAE / null embedding.
        3. Build train & val dataloaders (DistributedSampler on train if multi-GPU).
        4. AdamW + LambdaLR, then ``deepspeed.initialize`` (ZeRO-2 + bf16 from YAML).
        5. Optional ``load_checkpoint`` resume.
        6. Epoch loop: each step runs
           :func:`compute_flow_loss`, ``backward``, ``step``; periodic WandB logging,
           distributed validation loss evaluation, warm-up unfreeze when
           ``train.warmup_iterations`` is reached, and DeepSpeed saves on all ranks.

    Args:
        cfg: Merged ``DictConfig`` from :func:`src.utils.load_config`.
    """
    # --- Distributed backend (NCCL expects CUDA_VISIBLE_DEVICES / launcher env) ---
    deepspeed.init_distributed(dist_backend="nccl")
    device = get_device()
    if device.type == "cuda":
        torch.cuda.set_device(device)
    world_size = get_world_size()
    rank = get_global_rank()

    # Per-rank RNG offsets so each GPU sees different corruption / noise unless seeded identically.
    torch.manual_seed(int(cfg.train.seed) + rank)
    random.seed(int(cfg.train.seed) + rank)

    wandb_init(cfg, cfg)
    run_name = _resolve_inspect_run_name(cfg)

    if is_main_process():
        _checkpoint_dir_for_run(cfg, run_name).mkdir(parents=True, exist_ok=True)

    load_dir, load_tag = _resolve_checkpoint_strategy(cfg)
    should_try_checkpoint = load_dir is not None and load_tag is not None
    checkpoint_path = str(Path(load_dir) / load_tag) if should_try_checkpoint else None
    if should_try_checkpoint and is_main_process():
        log_message(f"[model] will restore training state from checkpoint {checkpoint_path}")
    initial_step_hint = _step_from_tag(load_tag)
    warmup_only = _warmup_only_for_step(cfg, initial_step_hint)

    # --- Models: either checkpoint-first init (no pretrained) or pretrained fallback path ---
    flow_model, vae, null_emb = setup_model(
        cfg,
        device,
        rank,
        warmup_only=warmup_only,
        load_pretrained=not should_try_checkpoint,
    )
    vae = vae.to(device).requires_grad_(False).eval()

    corrupt_cfg = cfg.corruption

    # --- Data: train + val ---
    start_epoch = 0
    train_loader, train_dataset, train_sampler = _build_train_loader(cfg, corrupt_cfg, world_size)
    val_loader, _, val_sampler = _build_val_loader(cfg, corrupt_cfg, world_size)
    inference_indices = fixed_inference_indices(cfg, len(val_loader.dataset))
    if is_main_process() and inference_indices:
        log_message(f"[inference] fixed validation indices: {inference_indices}")

    steps_per_epoch = max(len(train_loader), 1)
    grad_accum = int(cfg.ds_config.get("gradient_accumulation_steps", 1))
    total_optimizer_steps = max(
        1, int(cfg.train.num_epochs) * steps_per_epoch // grad_accum
    )

    # --- Optimizer / scheduler / DeepSpeed engine (ZeRO partitioning lives here) ---
    optimizer = build_optimizer(flow_model, cfg, warmup_only=warmup_only)
    scheduler = build_scheduler(optimizer, cfg, total_optimizer_steps)
    
    ds_cfg = OmegaConf.to_container(cfg.ds_config, resolve=True)
    micro_batch = _micro_batch_size(cfg)
    ds_cfg["train_micro_batch_size_per_gpu"] = micro_batch
    ds_cfg["gradient_accumulation_steps"] = grad_accum
    ds_cfg["train_batch_size"] = int(micro_batch) * int(world_size) * int(grad_accum)

    engine, optimizer, _, _ = deepspeed.initialize(
        model=flow_model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_cfg,
    )
    print_vram_debug(cfg, "after_deepspeed_initialize", device=device)

    global_step = 0
    images_since_save = 0
    last_save_step = 0
    restored_train_state = False
    last_logged_step = 0
    warmup_only = _apply_training_phase(flow_model, optimizer, scheduler, cfg, step=global_step)

    # --- Single-source load policy: checkpoint OR pretrained; fallback to pretrained on errors ---
    if should_try_checkpoint:
        try:
            _, client = engine.load_checkpoint(load_dir, tag=load_tag)
            if client:
                start_epoch = int(client.get("epoch", 0))
                global_step = int(client.get("step", 0))
                images_since_save = int(client.get("images_since_save", 0))
                last_save_step = global_step
                restored_train_state = _restore_train_loader_state(
                    train_loader,
                    train_dataset,
                    train_sampler,
                    client.get("train_loader_state"),
                )
                if is_main_process():
                    log_message(f"[train] Loaded checkpoint tag '{load_tag}' from '{load_dir}'.")
                    log_message(f"[model] restored model weights from checkpoint {checkpoint_path}")
            else:
                raise RuntimeError("DeepSpeed returned empty client_state while loading checkpoint.")
        except Exception as exc:
            if is_main_process():
                log_message(
                    f"[train] Checkpoint load failed ({exc}); falling back to pretrained weights."
                )
            _unwrap_model(engine).load_pretrained_backbone(
                cfg.model.flux_model_name, rank=rank
            )
            start_epoch = 0
            global_step = 0
            images_since_save = 0
            last_save_step = 0
            restored_train_state = False
    warmup_only = _apply_training_phase(
        _unwrap_model(engine),
        engine.optimizer,
        engine.lr_scheduler if hasattr(engine, "lr_scheduler") else scheduler,
        cfg,
        step=global_step,
    )
    last_logged_step = global_step
    print_training_phase(_unwrap_model(engine), warmup_only, global_step)
    _distributed_barrier()

    engine.train()

    # Interval knobs: step-based save, image-approximate save, val cadence, WandB image logging
    save_every = int(cfg.train.save_every)
    save_every_images = cfg.train.get("save_every_images")
    val_every = int(getattr(cfg.train, "val_every", 80))
    log_every = int(cfg.train.log_every)
    log_images_every = int(cfg.wandb.get("log_images_every", 500)) if cfg.get("wandb") else 10**9

    micro_batch = _micro_batch_size(cfg)
    # Global images (all GPUs) per *optimizer* step ≈ micro_batch × world_size × grad_accumulation.
    images_per_opt_step = micro_batch * world_size * grad_accum

    for epoch in range(start_epoch, int(cfg.train.num_epochs)):
        resumed_epoch = restored_train_state and epoch == start_epoch

        if not resumed_epoch and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, disable=not is_main_process(), desc=f"epoch {epoch}")
        t_last = time.perf_counter()
        for batch in pbar:
            loss = compute_flow_loss(
                engine,
                vae,
                batch,
                null_emb,
                int(cfg.model.spatial_compression),
                device,
                float(getattr(cfg.train, "loss_weight_mask", 0.7)),
            )
            print_vram_debug(cfg, f"before_backward_step_{global_step + 1}", device=device)
            engine.backward(loss)
            engine.step()
            print_vram_debug(cfg, f"after_optimizer_step_{global_step + 1}", device=device)

            prev_step = global_step
            global_step = int(getattr(engine, "global_steps", prev_step + 1))
            images_since_save += images_per_opt_step
            _maybe_save_inspect_batch(batch, cfg, run_name, global_step, rank)
            if warmup_only and not _warmup_only_for_step(cfg, global_step):
                warmup_only = _apply_training_phase(
                    _unwrap_model(engine),
                    engine.optimizer,
                    engine.lr_scheduler if hasattr(engine, "lr_scheduler") else scheduler,
                    cfg,
                    step=global_step,
                )
                print_training_phase(_unwrap_model(engine), warmup_only, global_step)

            if is_main_process():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if is_main_process() and global_step % log_every == 0:
                dt = time.perf_counter() - t_last
                t_last = time.perf_counter()
                steps_since_log = max(global_step - last_logged_step, 1)
                img_per_sec = (steps_since_log * images_per_opt_step) / max(dt, 1e-6)
                lrs = [pg["lr"] for pg in engine.optimizer.param_groups]
                metrics = {
                    "train/loss": loss.item(),
                    "train/images_per_sec": img_per_sec,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }
                for i, lr in enumerate(lrs):
                    metrics[f"train/lr_group_{i}"] = lr
                log_message(
                    f"[train] epoch={epoch} step={global_step} "
                    f"loss={loss.item():.4f} img_per_sec={img_per_sec:.2f}"
                )
                wandb_log(metrics, step=global_step)
                last_logged_step = global_step

            # Checkpoint when either step interval hits or enough global images seen.
            do_save = global_step > 0 and (global_step % save_every == 0)
            if save_every_images is not None and int(save_every_images) > 0:
                if images_since_save >= int(save_every_images):
                    do_save = True
                    images_since_save = 0

            if do_save and global_step != last_save_step:
                client_state = {
                    "step": global_step,
                    "epoch": epoch,
                    "images_since_save": images_since_save,
                    "train_loader_state": _capture_train_loader_state(
                        train_loader,
                        train_dataset,
                        train_sampler,
                    ),
                }
                maybe_save_deepspeed_checkpoint(engine, cfg, client_state, do_save=True, run_name=run_name)
                last_save_step = global_step

            should_validate = (
                val_every > 0
                and global_step % val_every == 0
                and global_step > 0
            )
            if should_validate:
                val_start_time = None
                should_log_inference = (
                    cfg.get("wandb")
                    and cfg.wandb.get("log_images", False)
                    and global_step % log_images_every == 0
                    and wandb_log_images_num(cfg) > 0
                )
                inference_start_time = None
                if is_main_process():
                    val_start_time = time.perf_counter()
                    log_message("========STARTING VALIDATION========")
                    log_message(f"[val] epoch={epoch} step={global_step}")
                metrics = validate(engine, vae, val_loader, null_emb, cfg, device)
                if is_main_process():
                    log_message(
                        f"[val] epoch={epoch} step={global_step} "
                        f"loss={metrics['velocity_loss']:.4f}"
                    )
                    elapsed = 0.0 if val_start_time is None else time.perf_counter() - val_start_time
                    log_message("========ENDING VALIDATION========")
                    log_message(
                        f"[val] epoch={epoch} step={global_step} "
                        f"duration_sec={elapsed:.2f}"
                    )
                    wandb_log(
                        {"val/velocity_loss": metrics["velocity_loss"]},
                        step=global_step,
                    )
                if should_log_inference:
                    if is_main_process():
                        inference_start_time = time.perf_counter()
                        log_message("========STARTING INFERENCE========")
                        log_message(
                            f"[inference] epoch={epoch} step={global_step} "
                            f"num_images={wandb_log_images_num(cfg)}"
                        )
                    panels = run_validation_inference(
                        engine,
                        vae,
                        val_loader.dataset,
                        null_emb,
                        cfg,
                        device,
                        global_step,
                    )
                    if is_main_process():
                        elapsed = (
                            0.0
                            if inference_start_time is None
                            else time.perf_counter() - inference_start_time
                        )
                        log_message("========ENDING INFERENCE========")
                        log_message(
                            f"[inference] epoch={epoch} step={global_step} "
                            f"duration_sec={elapsed:.2f}"
                        )
                        if panels is not None:
                            wandb_log(
                                {},
                                step=global_step,
                                images={"inference/panels": panels},
                            )
                            log_message(
                                f"[inference] uploaded {int(panels.shape[0])} panels to wandb "
                                f"at step={global_step}"
                            )
                _distributed_barrier()

        restored_train_state = False

    client_state = {
        "step": global_step,
        "epoch": int(cfg.train.num_epochs),
        "images_since_save": images_since_save,
        "train_loader_state": _capture_train_loader_state(
            train_loader,
            train_dataset,
            train_sampler,
        ),
    }
    maybe_save_deepspeed_checkpoint(engine, cfg, client_state, do_save=True, run_name=run_name)
    wandb_finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Art restoration DiT training")
    parser.add_argument(
        "--config",
        type=str,
        default="train/configs/train.yaml",
        help="Path to training YAML config",
    )
    args, unknown = parser.parse_known_args()
    conf = load_config(args.config, unknown if unknown else None)
    main(conf)
