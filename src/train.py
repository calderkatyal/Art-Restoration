"""Training loop for conditional latent rectified flow art restoration.

Rectified flow objective per training step:
    1. Sample clean image ``x``; dataset / corruption yields ``(y, M)``.
    2. ``z_1 = E(x)``, ``z_y = E(y)`` with frozen VAE (no gradients through ``E``).
    3. ``M' = downsample_mask(M, factor=spatial_compression)`` (e.g. 16 for FLUX VAE).
    4. ``z_0 ~ N(0, I)``, ``t ~ U(0, 1)``.
    5. ``z_t = (1 - t) * z_0 + t * z_1`` — note ``z_t`` interpolates **noise ↔ clean**;
       the damaged reference ``z_y`` is **separate conditioning**, not part of this mix.
    6. ``vel = v_θ(z_t, t | z_y, M', null_emb)``.
    7. ``loss = || vel - (z_1 - z_0) ||²`` (MSE in float32 for numerical stability).
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
    DeepSpeed checkpoints live under ``{train.output_dir}/deepspeed_ckpt/<tag>/``.
    ``train.save_every`` triggers saves every N **optimizer** steps;
    ``train.save_every_images`` triggers after approximately that many **global** images
    (``batch_size × world_size × gradient_accumulation_steps`` per optimizer step).
    Resume with ``train.resume_from`` pointing at a tag directory, e.g.
    ``./checkpoints/deepspeed_ckpt/step_1000``.

Validation:
    Rank 0 runs full-image and masked PSNR on the val split (same synthetic corruption
    pipeline as training) via Euler sampling in ``src.inference.sample``.

Usage:
    python -m src.train [--config train/configs/train.yaml] [overrides...]

Arguments:
    --config    Path to YAML (default: ``train/configs/train.yaml``).
    overrides   Dot-notation overrides, e.g. ``train.stage=full``,
                ``train.batch_size=8``,
                ``train.resume_from=./checkpoints/deepspeed_ckpt/step_1000``.
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
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .config import CorruptionConfig
from .corruption import downsample_mask
from .dataset import ArtRestorationDataset
from .distributed import get_device, get_global_rank, get_world_size, is_main_process
from .evaluations import compute_psnr
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


def _to_corruption_config(path: str) -> CorruptionConfig:
    """Load ``CorruptionConfig`` from YAML and merge with structured dataclass defaults.

    Args:
        path: Filesystem path to a corruption YAML (e.g. ``src/corruption/configs/default.yaml``).

    Returns:
        A concrete :class:`~src.config.CorruptionConfig` instance.
    """
    raw = OmegaConf.load(path)
    base = OmegaConf.structured(CorruptionConfig())
    merged = OmegaConf.merge(base, raw)
    return OmegaConf.to_object(merged)


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


def _worker_init_fn(worker_id: int, base_seed: int, rank: int) -> None:
    """Deterministic-ish seeding for DataLoader workers (``worker_init_fn``).

    Combines the global training seed, dataloader worker index, and process rank so
    different ranks and workers do not replay identical corruption noise.
    """
    seed = base_seed + worker_id + rank * 1000
    random.seed(seed)
    torch.manual_seed(seed)


def setup_dataloader(
    cfg: DictConfig,
    max_simultaneous: int | None = None,
    split: str = "train",
    sampler: Optional[DistributedSampler] = None,
) -> tuple[DataLoader, ArtRestorationDataset]:
    """Create a ``DataLoader`` over :class:`~src.dataset.ArtRestorationDataset`.

    Uses ``cfg.train.data_dir`` for ``split=="train"`` and ``cfg.train.val_dir`` for
    validation. When ``sampler`` is ``None`` and ``split=="train"``, ``shuffle=True``;
    when a ``DistributedSampler`` is supplied, ``shuffle`` must be ``False`` on the
    loader (the sampler owns shuffling via ``set_epoch``).

    Args:
        cfg:              Full config (paths, resolution, batch size, ``num_workers``).
        max_simultaneous: Optional curriculum override forwarded to the dataset / corruption.
        split:            ``"train"`` or ``"val"`` selecting the image root directory.
        sampler:          Optional ``DistributedSampler`` for DDP / DeepSpeed multi-GPU.

    Returns:
        ``(dataloader, dataset)`` — the dataset is returned so callers can attach a
        sampler that references the **same** object instance if they rebuild the loader.
    """
    corrupt_cfg = _to_corruption_config(cfg.corruption.config_path)
    data_dir = cfg.train.data_dir if split == "train" else cfg.train.val_dir
    ds = ArtRestorationDataset(
        data_dir=data_dir,
        resolution=int(cfg.train.resolution),
        corruption_config=corrupt_cfg,
        max_simultaneous=max_simultaneous,
    )
    nw = int(getattr(cfg.train, "num_workers", 4))
    loader = DataLoader(
        ds,
        batch_size=int(cfg.train.batch_size),
        sampler=sampler,
        shuffle=(sampler is None and split == "train"),
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
        worker_init_fn=lambda wid: _worker_init_fn(wid, int(cfg.train.seed), get_global_rank()),
        drop_last=False,
    )
    return loader, ds


def compute_flow_loss(
    model_engine: Any,
    vae: FluxVAE,
    batch: dict,
    null_emb: torch.Tensor,
    spatial_compression: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute rectified-flow MSE loss for one micro-batch.

    Expects ``batch`` keys ``"clean"``, ``"corrupted"``, ``"mask"`` as emitted by
    :class:`~src.dataset.ArtRestorationDataset`. Encodings use the frozen VAE in
    ``float32``; the velocity target ``(z_1 - z_0)`` is ``float32``; model predictions
    are cast to ``float32`` for the MSE so bf16 forward does not underflow the loss.

    ``model_engine`` may be a DeepSpeed ``DeepSpeedEngine`` (``__call__`` forwards to
    ``RestorationDiT.forward``) or a bare ``RestorationDiT`` for tests.

    Args:
        model_engine:        Callable ``(z_t, t, z_y, M', null_emb) -> velocity``.
        vae:                 Frozen :class:`~src.vae.FluxVAE`.
        batch:               Dict of CPU/GPU tensors from the dataloader.
        null_emb:            Cached text embedding ``(1, 512, 7680)`` on ``device``.
        spatial_compression: VAE spatial factor (16) for ``downsample_mask``.
        device:              Device to run the loss on.

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
    if m_prime.shape[1] != mask.shape[1]:
        raise ValueError(
            f"Mask channel count {mask.shape[1]} vs downsampled {m_prime.shape[1]}"
        )

    z_0 = torch.randn_like(z_1)
    t = torch.rand((z_1.shape[0],), device=device, dtype=z_1.dtype)
    t_bc = t.view(-1, 1, 1, 1)
    z_t = (1.0 - t_bc) * z_0 + t_bc * z_1

    vel = model_engine(z_t, t, z_y, m_prime, null_emb)
    target = (z_1 - z_0).float()
    return F.mse_loss(vel.float(), target)


@torch.no_grad()
def validate(
    model_engine: Any,
    vae: FluxVAE,
    val_loader: DataLoader,
    null_emb: torch.Tensor,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    """Run sampling on the validation set and report average PSNR metrics.

    For each batch, runs :func:`~src.inference.sample` (Euler ODE + latent data
    consistency) then :func:`~src.evaluations.compute_psnr` in full-image and
    masked modes. Restores the model's previous ``train()`` / ``eval()`` state.

    Args:
        model_engine: Wrapped or bare ``RestorationDiT``.
        vae:          Frozen VAE.
        val_loader:   Batches of ``clean``, ``corrupted``, ``mask``.
        null_emb:     Null text embedding.
        cfg:          Uses ``cfg.inference.num_steps`` when present.
        device:       CUDA device.

    Returns:
        Dict with float keys ``"psnr_full"`` and ``"psnr_masked"`` (batch-averaged).
    """
    mdl = _unwrap_model(model_engine)
    was_training = mdl.training
    mdl.eval()
    steps = int(getattr(cfg.inference, "num_steps", cfg.model.get("num_steps", 50)))
    full_scores: list[float] = []
    mask_scores: list[float] = []
    for batch in val_loader:
        clean = batch["clean"].to(device, non_blocking=True)
        corrupted = batch["corrupted"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        restored = sample(
            mdl,
            vae,
            corrupted,
            mask,
            null_emb,
            num_steps=steps,
            device=str(device),
        )
        full_scores.append(compute_psnr(restored, clean, mask=None))
        mask_scores.append(compute_psnr(restored, clean, mask=mask))
    if was_training:
        mdl.train()
    return {
        "psnr_full": float(sum(full_scores) / max(len(full_scores), 1)),
        "psnr_masked": float(sum(mask_scores) / max(len(mask_scores), 1)),
    }


def _write_checkpoint_metadata(output_dir: Path, tag: str, client_state: dict) -> None:
    """Persist a small JSON sidecar next to DeepSpeed checkpoints (rank 0 only).

    DeepSpeed already stores ``client_state`` inside its checkpoint payload; this
    file is a human-readable duplicate for HPC inspection and bookkeeping.

    Args:
        output_dir:   Root ``.../deepspeed_ckpt`` directory.
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
        cfg:           Provides ``train.output_dir``.
        client_state:  Metadata forwarded to DeepSpeed (also embedded in the checkpoint).
        do_save:       When False, returns immediately without touching the filesystem.
    """
    if not do_save:
        return
    out = Path(cfg.train.output_dir) / "deepspeed_ckpt"
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


def _build_train_loader(
    cfg: DictConfig,
    corrupt_cfg: CorruptionConfig,
    max_sim: int | None,
    world_size: int,
) -> tuple[DataLoader, ArtRestorationDataset, Optional[DistributedSampler]]:
    """Build train ``DataLoader``, backing dataset, and optional ``DistributedSampler``.

    Centralizes the pattern: construct ``ArtRestorationDataset`` from ``cfg.train``,
    wrap with ``DistributedSampler`` when ``world_size > 1``, and apply ``DataLoader``
    knobs (``num_workers``, ``pin_memory``, ``persistent_workers``, worker seeding).

    Args:
        cfg:          Full config.
        corrupt_cfg:  Merged :class:`~src.config.CorruptionConfig` (shared, not mutated).
        max_sim:      Curriculum override (``1`` or ``None``).
        world_size:   ``torch.distributed`` world size (1 means single-process).

    Returns:
        Tuple ``(train_loader, train_dataset, sampler_or_none)``.
    """
    ds = ArtRestorationDataset(
        data_dir=cfg.train.data_dir,
        resolution=int(cfg.train.resolution),
        corruption_config=corrupt_cfg,
        max_simultaneous=max_sim,
    )
    sampler: Optional[DistributedSampler] = None
    if world_size > 1:
        sampler = DistributedSampler(ds, shuffle=True, drop_last=False)
    nw = int(getattr(cfg.train, "num_workers", 4))
    loader = DataLoader(
        ds,
        batch_size=int(cfg.train.batch_size),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=nw,
        pin_memory=True,
        persistent_workers=nw > 0,
        worker_init_fn=lambda wid: _worker_init_fn(wid, int(cfg.train.seed), get_global_rank()),
        drop_last=False,
    )
    return loader, ds, sampler


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

    ckpt_dir = Path(str(cfg.train.get("checkpoint_dir", Path(cfg.train.output_dir) / "deepspeed_ckpt")))
    tag = _latest_checkpoint_tag(ckpt_dir)
    if tag is None:
        return None, None
    return str(ckpt_dir), tag


def main(cfg: DictConfig) -> None:
    """Initialize distributed training, run epochs, log, validate, and checkpoint.

    Pipeline overview:
        1. ``deepspeed.init_distributed`` (NCCL).
        2. Build model / VAE / null embedding; assert mask width matches ``img_in``.
        3. Build train & val dataloaders (DistributedSampler on train if multi-GPU).
        4. AdamW + LambdaLR, then ``deepspeed.initialize`` (ZeRO-2 + bf16 from YAML).
        5. Optional ``load_checkpoint`` resume.
        6. Epoch loop: curriculum may rebuild the train loader; each step runs
           :func:`compute_flow_loss`, ``backward``, ``step``; periodic WandB logging,
           validation on rank 0, and DeepSpeed saves on all ranks.

    Args:
        cfg: Merged ``DictConfig`` from :func:`src.utils.load_config`.
    """
    # --- Distributed backend (NCCL expects CUDA_VISIBLE_DEVICES / launcher env) ---
    deepspeed.init_distributed(dist_backend="nccl")
    device = get_device()
    world_size = get_world_size()
    rank = get_global_rank()

    if is_main_process():
        Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)

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

    # --- Consistency guard: corruption YAML K must match img_in width 128+128+K ---
    corrupt_cfg = _to_corruption_config(cfg.corruption.config_path)
    k_mask = corrupt_cfg.num_channels
    if int(cfg.model.mask_channels) != k_mask or int(cfg.model.in_channels) != 128 + 128 + k_mask:
        raise ValueError(
            f"model.mask_channels / in_channels must match corruption (K={k_mask}): "
            f"model has mask_channels={cfg.model.mask_channels}, in_channels={cfg.model.in_channels}"
        )

    # --- Data: train (possibly curriculum); val only on rank 0 to avoid duplicate work ---
    start_epoch = 0
    max_sim = _curriculum_max_sim(cfg, start_epoch)
    train_loader, _, train_sampler = _build_train_loader(
        cfg, corrupt_cfg, max_sim, world_size
    )

    val_loader: Optional[DataLoader] = None
    if is_main_process():
        val_loader, _ = setup_dataloader(cfg, max_simultaneous=None, split="val", sampler=None)

    steps_per_epoch = max(len(train_loader), 1)
    grad_accum = int(cfg.ds_config.get("gradient_accumulation_steps", 1))
    total_optimizer_steps = max(
        1, int(cfg.train.num_epochs) * steps_per_epoch // grad_accum
    )

    # --- Optimizer / scheduler / DeepSpeed engine (ZeRO partitioning lives here) ---
    optimizer = build_optimizer(flow_model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_optimizer_steps)

    ds_cfg = OmegaConf.to_container(cfg.ds_config, resolve=True)
    ds_cfg["train_micro_batch_size_per_gpu"] = int(cfg.train.batch_size)
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
                    train_loader, _, train_sampler = _build_train_loader(
                        cfg, corrupt_cfg, max_sim, world_size
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

    engine.train()

    # Interval knobs: step-based save, image-approximate save, val cadence, WandB image logging
    save_every = int(cfg.train.save_every)
    save_every_images = cfg.train.get("save_every_images")
    val_every = int(getattr(cfg.train, "val_every", 10**9))
    log_every = int(cfg.train.log_every)
    log_images_every = int(cfg.wandb.get("log_images_every", 500)) if cfg.get("wandb") else 10**9

    micro_batch = int(cfg.train.batch_size)
    # Global images (all GPUs) per *optimizer* step ≈ micro_batch × world_size × grad_accumulation.
    images_per_opt_step = micro_batch * world_size * grad_accum

    for epoch in range(start_epoch, int(cfg.train.num_epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        curriculum_max_sim = _curriculum_max_sim(cfg, epoch)
        if curriculum_max_sim != max_sim:
            max_sim = curriculum_max_sim
            train_loader, _, train_sampler = _build_train_loader(
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
                }
                maybe_save_deepspeed_checkpoint(engine, cfg, client_state, do_save=True)
                last_save_step = global_step

            if (
                is_main_process()
                and val_loader is not None
                and val_every > 0
                and global_step % val_every == 0
                and global_step > 0
            ):
                metrics = validate(engine, vae, val_loader, null_emb, cfg, device)
                wandb_log(
                    {"val/psnr_full": metrics["psnr_full"], "val/psnr_masked": metrics["psnr_masked"]},
                    step=global_step,
                )

    client_state = {
        "step": global_step,
        "epoch": int(cfg.train.num_epochs),
        "stage": str(cfg.train.stage),
        "images_since_save": images_since_save,
    }
    maybe_save_deepspeed_checkpoint(engine, cfg, client_state, do_save=True)
    wandb_finish()


if __name__ == "__main__":
    # Run training from the command line. Unknown CLI tokens are treated as OmegaConf
    # dotlist overrides (same style as Hydra), e.g. ``train.batch_size=8``.
    #
    # Examples:
    #   python -m src.train --config train/configs/train.yaml train.stage=warmup
    #   python -m src.train train.stage=full train.batch_size=8
    #   python -m src.train train.resume_from=./checkpoints/deepspeed_ckpt/step_10000
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
