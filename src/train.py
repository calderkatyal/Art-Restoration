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

Training stages (``cfg.train.stage``):
    ``"warmup"``: backbone frozen; only ``img_in`` trained at ``cfg.train.warmup.lr``.
    ``"full"``:   all layers trainable; backbone at ``cfg.train.full.backbone_lr``,
                  ``img_in`` at ``cfg.train.full.img_in_lr``.

Curriculum (``cfg.train.curriculum.enabled``):
    For the first ``curriculum.warmup_epochs`` epochs, ``max_simultaneous=1`` is applied
    by forcing ``individual_prob = 1.0`` in a copy of the corruption config so the
    dataloader favours single-degradation corruptions.

Distributed training:
    Launch with ``torchrun`` / DeepSpeed so ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE``
    are set. All ranks participate in ``save_checkpoint`` / ``load_checkpoint`` (ZeRO-2);
    logging, validation DataLoader, and WandB run on global rank 0 only.

Checkpointing:
    DeepSpeed checkpoints live under ``{train.checkpoint_dir}/<tag>/``.
    ``train.save_every`` triggers saves every N **optimizer** steps;
    ``train.save_every_images`` triggers after approximately that many **global** images
    (``train_micro_batch_size_per_gpu × world_size × gradient_accumulation_steps``
    per optimizer step).
    Resume with ``train.resume_from`` pointing at a tag directory, e.g.
    ``./checkpoints/step_1000``.

Validation:
    All ranks run distributed held-out velocity-loss evaluation on the val split using
    the same corruption pipeline as training; the aggregated result is logged to WandB
    on rank 0.

Usage:
    python -m src.train [--config train/configs/train.yaml] [overrides...]

Arguments:
    --config    Path to YAML (default: ``train/configs/train.yaml``).
    overrides   Dot-notation overrides, e.g. ``train.stage=full``,
                ``ds_config.train_micro_batch_size_per_gpu=8``,
                ``train.resume_from=/nfs/roberts/project/cpsc4520/cpsc4520_ckk25/checkpoints/step_1000``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Optional

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .corruption import OUTPUT_MASK_CHANNELS, downsample_mask
from .dataset import ArtRestorationDataset, build_wikiart_dataloader
from .distributed import get_device, get_global_rank, get_world_size, is_main_process
from .inference import sample
from .model import RestorationDiT
from .null_emb import load_or_compute_null_embedding
from .utils import load_config
from .vae import FluxVAE

# True only on rank 0 after ``wandb.init``; other ranks skip WandB entirely.
_WANDB_ACTIVE = False


def wandb_init(cfg: Any, train_cfg: Any) -> None:
    """Initialize a WandB run if ``cfg.wandb.enabled`` and this is global rank 0.

    Non-main ranks and disabled config leave ``_WANDB_ACTIVE`` False so ``wandb_log``
    stays a cheap no-op.

    Args:
        cfg:       Top-level config; reads ``cfg.wandb`` (``project``, ``entity``,
                   ``run_name``, ``tags``, ``enabled``).
        train_cfg: Full training config dict-like object serialized into ``wandb.config``
                   via ``OmegaConf.to_container`` for reproducibility.
    """
    global _WANDB_ACTIVE
    wb = getattr(cfg, "wandb", None)
    if wb is None or not wb.get("enabled", False) or not is_main_process():
        _WANDB_ACTIVE = False
        return
    import wandb

    tags = list(wb.get("tags") or [])
    wandb.init(
        project=wb.get("project", "art-restoration"),
        entity=wb.get("entity") or None,
        name=wb.get("run_name") or None,
        tags=tags if tags else None,
        config=OmegaConf.to_container(train_cfg, resolve=True),
    )
    _WANDB_ACTIVE = True


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
    global _WANDB_ACTIVE
    if not _WANDB_ACTIVE:
        return
    import wandb

    wandb.finish()
    _WANDB_ACTIVE = False


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
    load_pretrained: bool = True,
) -> tuple[RestorationDiT, FluxVAE, torch.Tensor]:
    """Construct the restoration DiT, frozen FLUX VAE, and cached null text embedding.

    Calls ``model.set_stage(cfg.train.stage)`` so only the intended parameters require
    gradients before the optimizer is built.

    Args:
        cfg:    Full OmegaConf ``DictConfig`` (uses ``cfg.model`` and ``cfg.train.stage``).
        device: CUDA device for this process (typically ``cuda:LOCAL_RANK``).
        rank:   Global rank, forwarded to weight download logging and VAE init.
        load_pretrained: Whether to load FLUX pretrained weights before widening ``img_in``.

    Returns:
        Tuple ``(restoration_dit, flux_vae, null_emb)`` where ``null_emb`` has shape
        ``(1, 512, 7680)`` and dtype ``bfloat16`` on ``device``.
    """
    flow_model = RestorationDiT(
        cfg=cfg.model,
        device=device,
        img_in_dtype=torch.bfloat16,
        load_pretrained=load_pretrained,
        rank=rank,
    )
    flow_model.set_stage(cfg.train.stage)

    vae = FluxVAE(
        flux_model_name=cfg.model.flux_model_name,
        rank=rank,
        device=device,
    )
    null_emb = load_or_compute_null_embedding(
        cache_path=cfg.model.null_emb_path,
        flux_model_name=cfg.model.flux_model_name,
        device=device,
    )
    return flow_model, vae, null_emb


def build_optimizer(model: RestorationDiT, cfg: DictConfig) -> torch.optim.Optimizer:
    """Build AdamW with learning rates appropriate for the current training stage.

    Warmup stage → single param group: all trainable ``img_in`` weights at
    ``cfg.train.warmup.lr`` (backbone should already be frozen via ``set_stage``).

    Full stage → two groups from ``model.get_trainable_params()``:
        ``img_in`` at ``cfg.train.full.img_in_lr``, backbone at ``cfg.train.full.backbone_lr``.

    Weight decay and Adam ``betas`` come from ``cfg.train.optimizer``.

    Args:
        model: ``RestorationDiT`` with ``requires_grad`` flags already set for the stage.
        cfg:   Full config.

    Returns:
        A ``torch.optim.AdamW`` instance suitable for passing to ``deepspeed.initialize``.
    """
    wd = cfg.train.optimizer.weight_decay
    betas = tuple(cfg.train.optimizer.betas)
    if cfg.train.stage == "warmup":
        params = [p for p in model.flow_model.img_in.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=cfg.train.warmup.lr, betas=betas, weight_decay=wd)
    groups = model.get_trainable_params()
    return torch.optim.AdamW(
        [
            {"params": groups[0]["params"], "lr": cfg.train.full.img_in_lr, "name": "img_in"},
            {"params": groups[1]["params"], "lr": cfg.train.full.backbone_lr, "name": "backbone"},
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
    if the optimizer used different base rates per group (full stage).

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


def _distributed_barrier() -> None:
    """Synchronize all ranks when distributed training is initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def setup_dataloader(
    cfg: DictConfig,
    max_simultaneous: int | None = None,
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
        max_simultaneous=max_simultaneous,
        num_workers=int(getattr(cfg.train, "num_workers", 4)),
        sampler_seed=int(getattr(cfg.train, "sampler_seed", cfg.train.seed)),
        corruption_seed=int(getattr(cfg.train, "corruption_seed", cfg.train.seed)),
        pin_memory=bool(getattr(cfg.train, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg.train, "persistent_workers", True)),
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
        client_state: Pickle-friendly dict (``step``, ``epoch``, ``stage``, etc.).
    """
    if not is_main_process():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / f"run_meta_{tag}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(client_state, f, indent=2)


def maybe_save_deepspeed_checkpoint(
    engine: Any,
    cfg: DictConfig,
    client_state: dict,
    do_save: bool,
) -> None:
    """Synchronously save a ZeRO-aware checkpoint on **all** ranks when ``do_save``.

    Under ZeRO-2, every process must enter ``engine.save_checkpoint``; rank 0 alone
    writes ``run_meta_<tag>.json`` via :func:`_write_checkpoint_metadata`.

    Args:
        engine:        DeepSpeed-wrapped model.
        cfg:           Provides ``train.checkpoint_dir``.
        client_state:  Metadata forwarded to DeepSpeed (also embedded in the checkpoint).
        do_save:       When False, returns immediately without touching the filesystem.
    """
    if not do_save:
        return
    out = Path(cfg.train.checkpoint_dir)
    tag = f"step_{int(client_state['step'])}"
    engine.save_checkpoint(str(out), tag=tag, client_state=client_state)
    if is_main_process():
        _write_checkpoint_metadata(out, tag, client_state)


def train(cfg: DictConfig) -> None:
    """Public entry alias that forwards to :func:`main`.

    Kept for symmetry with documentation / external callers that import ``train``.
    """
    main(cfg)


def _curriculum_max_sim(cfg: DictConfig, epoch: int) -> int | None:
    """Return ``max_simultaneous`` mask for the dataset at ``epoch``, or ``None`` if disabled.

    When curriculum is enabled and ``epoch < curriculum.warmup_epochs``, returns ``1``
    so :class:`~src.dataset.ArtRestorationDataset` biases toward single-degradation
    corruptions. Otherwise returns ``None`` (full preset statistics from YAML).
    """
    if not cfg.train.curriculum.get("enabled", False):
        return None
    if epoch < int(cfg.train.curriculum.warmup_epochs):
        return 1
    return None


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
    max_sim: int | None,
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
        max_simultaneous=max_sim,
        num_workers=int(getattr(cfg.train, "num_workers", 4)),
        sampler_seed=int(getattr(cfg.train, "sampler_seed", cfg.train.seed)),
        corruption_seed=int(getattr(cfg.train, "corruption_seed", cfg.train.seed)),
        pin_memory=bool(getattr(cfg.train, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg.train, "persistent_workers", True)),
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
        max_simultaneous=None,
        num_workers=int(getattr(cfg.train, "num_workers", 4)),
        sampler_seed=int(getattr(cfg.train, "sampler_seed", cfg.train.seed)),
        corruption_seed=int(getattr(cfg.train, "corruption_seed", cfg.train.seed)),
        pin_memory=bool(getattr(cfg.train, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg.train, "persistent_workers", True)),
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

    ckpt_dir = Path(str(cfg.train.checkpoint_dir))
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
        6. Epoch loop: curriculum may rebuild the train loader; each step runs
           :func:`compute_flow_loss`, ``backward``, ``step``; periodic WandB logging,
           distributed validation loss evaluation, and DeepSpeed saves on all ranks.

    Args:
        cfg: Merged ``DictConfig`` from :func:`src.utils.load_config`.
    """
    # --- Distributed backend (NCCL expects CUDA_VISIBLE_DEVICES / launcher env) ---
    deepspeed.init_distributed(dist_backend="nccl")
    device = get_device()
    world_size = get_world_size()
    rank = get_global_rank()

    if is_main_process():
        Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Per-rank RNG offsets so each GPU sees different corruption / noise unless seeded identically.
    torch.manual_seed(int(cfg.train.seed) + rank)
    random.seed(int(cfg.train.seed) + rank)

    wandb_init(cfg, cfg)

    load_dir, load_tag = _resolve_checkpoint_strategy(cfg)
    should_try_checkpoint = load_dir is not None and load_tag is not None

    # --- Models: either checkpoint-first init (no pretrained) or pretrained fallback path ---
    flow_model, vae, null_emb = setup_model(
        cfg,
        device,
        rank,
        load_pretrained=not should_try_checkpoint,
    )
    vae = vae.to(device).requires_grad_(False).eval()

    corrupt_cfg = cfg.corruption

    # --- Data: train (possibly curriculum); val only on rank 0 to avoid duplicate work ---
    start_epoch = 0
    max_sim = _curriculum_max_sim(cfg, start_epoch)
    train_loader, train_dataset, train_sampler = _build_train_loader(
        cfg, corrupt_cfg, max_sim, world_size
    )
    val_loader, _, val_sampler = _build_val_loader(cfg, corrupt_cfg, world_size)

    steps_per_epoch = max(len(train_loader), 1)
    grad_accum = int(cfg.ds_config.get("gradient_accumulation_steps", 1))
    total_optimizer_steps = max(
        1, int(cfg.train.num_epochs) * steps_per_epoch // grad_accum
    )

    # --- Optimizer / scheduler / DeepSpeed engine (ZeRO partitioning lives here) ---
    optimizer = build_optimizer(flow_model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_optimizer_steps)

    ds_cfg = OmegaConf.to_container(cfg.ds_config, resolve=True)
    ds_cfg["train_micro_batch_size_per_gpu"] = _micro_batch_size(cfg)
    ds_cfg["gradient_accumulation_steps"] = grad_accum

    engine, optimizer, _, _ = deepspeed.initialize(
        model=flow_model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_cfg,
    )

    global_step = 0
    images_since_save = 0
    last_save_step = 0
    restored_train_state = False

    # --- Single-source load policy: checkpoint OR pretrained; fallback to pretrained on errors ---
    if should_try_checkpoint:
        try:
            _, client = engine.load_checkpoint(load_dir, tag=load_tag)
            if client:
                start_epoch = int(client.get("epoch", 0))
                global_step = int(client.get("step", 0))
                images_since_save = int(client.get("images_since_save", 0))
                last_save_step = global_step
                new_max = _curriculum_max_sim(cfg, start_epoch)
                if new_max != max_sim:
                    max_sim = new_max
                    train_loader, train_dataset, train_sampler = _build_train_loader(
                        cfg, corrupt_cfg, max_sim, world_size
                    )
                restored_train_state = _restore_train_loader_state(
                    train_loader,
                    train_dataset,
                    train_sampler,
                    client.get("train_loader_state"),
                )
                if is_main_process():
                    print(f"[train] Loaded checkpoint tag '{load_tag}' from '{load_dir}'.")
            else:
                raise RuntimeError("DeepSpeed returned empty client_state while loading checkpoint.")
        except Exception as exc:
            if is_main_process():
                print(
                    f"[train] Checkpoint load failed ({exc}); falling back to pretrained weights."
                )
            _unwrap_model(engine).load_pretrained_backbone(
                cfg.model.flux_model_name, rank=rank, device=device
            )
            start_epoch = 0
            global_step = 0
            images_since_save = 0
            last_save_step = 0
            restored_train_state = False
    _distributed_barrier()

    engine.train()

    # Interval knobs: step-based save, image-approximate save, val cadence, WandB image logging
    save_every = int(cfg.train.save_every)
    save_every_images = cfg.train.get("save_every_images")
    val_every = int(getattr(cfg.train, "val_every", 10**9))
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

        curriculum_max_sim = _curriculum_max_sim(cfg, epoch)
        if not resumed_epoch and curriculum_max_sim != max_sim:
            max_sim = curriculum_max_sim
            train_loader, train_dataset, train_sampler = _build_train_loader(
                cfg, corrupt_cfg, max_sim, world_size
            )

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
            engine.backward(loss)
            engine.step()

            prev_step = global_step
            global_step = int(getattr(engine, "global_steps", prev_step + 1))
            images_since_save += images_per_opt_step

            if is_main_process():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if is_main_process() and global_step % log_every == 0:
                dt = time.perf_counter() - t_last
                t_last = time.perf_counter()
                img_per_sec = (micro_batch * world_size) / max(dt, 1e-6)
                lrs = [pg["lr"] for pg in engine.optimizer.param_groups]
                metrics = {
                    "train/loss": loss.item(),
                    "train/images_per_sec": img_per_sec,
                    "train/epoch": epoch,
                }
                for i, lr in enumerate(lrs):
                    metrics[f"train/lr_group_{i}"] = lr
                wandb_log(metrics, step=global_step)

                if cfg.get("wandb") and cfg.wandb.get("log_images", False):
                    if global_step % log_images_every == 0:
                        with torch.no_grad():
                            mdl = _unwrap_model(engine)
                            was_train = mdl.training
                            mdl.eval()
                            rb = {k: v[: min(4, v.shape[0])] for k, v in batch.items()}
                            rest = sample(
                                mdl,
                                vae,
                                rb["corrupted"],
                                rb["mask"],
                                null_emb,
                                num_steps=int(getattr(cfg.inference, "num_steps", 50)),
                                device=str(device),
                            )
                            if was_train:
                                mdl.train()
                            wandb_log(
                                {},
                                step=global_step,
                                images={
                                    "train/grid_clean": rb["clean"],
                                    "train/grid_corrupt": rb["corrupted"],
                                    "train/grid_restored": rest,
                                },
                            )

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
                    "stage": str(cfg.train.stage),
                    "images_since_save": images_since_save,
                    "train_loader_state": _capture_train_loader_state(
                        train_loader,
                        train_dataset,
                        train_sampler,
                    ),
                }
                maybe_save_deepspeed_checkpoint(engine, cfg, client_state, do_save=True)
                last_save_step = global_step

            should_validate = (
                val_every > 0
                and global_step % val_every == 0
                and global_step > 0
            )
            if should_validate:
                metrics = validate(engine, vae, val_loader, null_emb, cfg, device)
                if is_main_process():
                    wandb_log(
                        {"val/velocity_loss": metrics["velocity_loss"]},
                        step=global_step,
                    )
                _distributed_barrier()

        restored_train_state = False

    client_state = {
        "step": global_step,
        "epoch": int(cfg.train.num_epochs),
        "stage": str(cfg.train.stage),
        "images_since_save": images_since_save,
        "train_loader_state": _capture_train_loader_state(
            train_loader,
            train_dataset,
            train_sampler,
        ),
    }
    maybe_save_deepspeed_checkpoint(engine, cfg, client_state, do_save=True)
    wandb_finish()


if __name__ == "__main__":
    # Run training from the command line. Unknown CLI tokens are treated as OmegaConf
    # dotlist overrides (same style as Hydra), e.g.
    # ``ds_config.train_micro_batch_size_per_gpu=8``.
    #
    # Examples:
    #   python -m src.train --config train/configs/train.yaml train.stage=warmup
    #   python -m src.train train.stage=full ds_config.train_micro_batch_size_per_gpu=8
    #   python -m src.train train.resume_from=/nfs/roberts/project/cpsc4520/cpsc4520_ckk25/checkpoints/step_1000
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
