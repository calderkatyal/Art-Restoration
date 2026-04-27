"""Training loop for learned corruption GAN (pix2pix: clean → damaged painting).

Objective per training step:
    1. Sample (clean, damaged) pair from PairedArtDataset.
    2. Generator forward:  fake_damaged = G(clean).
    3. Generator loss:
           L_adv  = MSE(D(clean, fake_damaged), ones)          [LSGAN]
           L_L1   = ||fake_damaged - damaged||_1 * lambda_l1
           L_G    = L_adv + L_L1
    4. Discriminator loss:
           L_real = MSE(D(clean, damaged),      ones)
           L_fake = MSE(D(clean, fake_damaged), zeros)
           L_D    = 0.5 * (L_real + L_fake)
    5. Update G (gen_optimizer), then D (disc_optimizer) — separate steps.

Distributed training:
    Launch with ``torchrun`` (NCCL). Both generator and discriminator are
    wrapped with ``DistributedDataParallel``. All ranks participate in
    validation and checkpoint barriers; WandB logging is rank-0 only.

Checkpointing:
    Saved under ``{train.checkpoint_root}/{wandb.run_name}/step_N/checkpoint.pt``
    using ``torch.save``.  Follows the same ``step_N`` tag scheme and
    ``checkpoint_tags_desc`` / ``_prune_checkpoint_history`` helpers as
    ``src.train`` so the two scripts are operationally consistent.
    Resume by setting ``train.resume_from`` to a tag directory path.

Validation:
    Held-out pairs from PairedArtDataset (val split) → L1 loss aggregated
    across all ranks via all_reduce.

Visualisation:
    Fixed WikiArt val images (same ``fixed_inference_indices`` logic as
    ``src.train``) → generator applied → (clean | generated_damage) panels
    logged to WandB.  Lets you see at a glance how well the GAN generalises
    to held-out paintings beyond the 100-pair training set.

Usage:
    torchrun --nproc_per_node=NUM_GPUS -m src.corruption_gan.train \\
        --config train/configs/train_corruption_gan.yaml [overrides...]

Arguments:
    --config   Path to YAML (default: ``train/configs/train_corruption_gan.yaml``).
    overrides  Dot-notation OmegaConf overrides, e.g. ``train.num_epochs=300``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ..distributed import get_device, get_global_rank, get_world_size, is_main_process
from ..utils import (
    checkpoint_tags_desc,
    fixed_inference_indices,
    gather_inference_panels,
    log_message,
)
from .model import UNetGenerator, PatchDiscriminator
from .dataset import PairedArtDataset, WikiArtValDataset


# ---------------------------------------------------------------------------
# WandB helpers (mirrors src.train exactly)
# ---------------------------------------------------------------------------

_WANDB_ACTIVE = False
_WANDB_RUN_NAME: Optional[str] = None


def wandb_init(cfg: Any) -> Optional[str]:
    global _WANDB_ACTIVE, _WANDB_RUN_NAME
    wb = getattr(cfg, "wandb", None)
    run_name = wb.get("run_name") if wb is not None else None
    if not run_name:
        raise ValueError("cfg.wandb.run_name must be set explicitly.")
    _WANDB_RUN_NAME = run_name
    if wb is None or not wb.get("enabled", False) or not is_main_process():
        _WANDB_ACTIVE = False
        return _WANDB_RUN_NAME
    import wandb
    tags = list(wb.get("tags") or [])
    wandb.init(
        project=wb.get("project", "art-restoration"),
        entity=wb.get("entity") or None,
        name=run_name,
        tags=tags or None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    _WANDB_ACTIVE = True
    return _WANDB_RUN_NAME


def wandb_log(metrics: dict, step: int, images: dict | None = None) -> None:
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
            payload[name] = wandb.Image(
                grid.permute(1, 2, 0).numpy(),
                caption=f"{name} global_step={step}",
            )
    wandb.log(payload, step=step)


def wandb_finish() -> None:
    global _WANDB_ACTIVE
    if not _WANDB_ACTIVE:
        return
    import wandb
    wandb.finish()
    _WANDB_ACTIVE = False


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _sanitize_run_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    return cleaned or "run"


def load_config(yaml_path: str, overrides: list[str] | None = None) -> DictConfig:
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Checkpoint helpers (mirrors src.train — same tag scheme, same pruning)
# ---------------------------------------------------------------------------

def _checkpoint_root(cfg: DictConfig) -> Path:
    root = cfg.train.get("checkpoint_root") or cfg.train.get("checkpoint_dir")
    if not root:
        raise ValueError("Config must define train.checkpoint_root")
    return Path(str(root))


def _checkpoint_dir_for_run(cfg: DictConfig, run_name: str) -> Path:
    return _checkpoint_root(cfg) / run_name


def _write_latest_tag(checkpoint_dir: Path) -> None:
    """Keep a ``latest`` pointer file aligned with the newest surviving tag."""
    if not is_main_process():
        return
    latest_path = checkpoint_dir / "latest"
    tags = checkpoint_tags_desc(checkpoint_dir)
    if not tags:
        try:
            latest_path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    latest_path.write_text(f"{tags[0]}\n", encoding="utf-8")


def _delete_tag_dir(checkpoint_dir: Path, tag: str, reason: str) -> bool:
    if not is_main_process():
        return False
    target = checkpoint_dir / tag
    if not target.exists() or not target.is_dir():
        return False
    try:
        shutil.rmtree(target)
        log_message(f"[checkpoint] deleted {target} ({reason})")
        _write_latest_tag(checkpoint_dir)
        return True
    except Exception as exc:
        log_message(f"[checkpoint] failed to delete {target}: {type(exc).__name__}: {exc}")
        return False


def _prune_checkpoint_history(checkpoint_dir: Path, keep_latest: int = 2) -> list[str]:
    if not is_main_process():
        return []
    deleted: list[str] = []
    for tag in checkpoint_tags_desc(checkpoint_dir)[max(0, keep_latest):]:
        if _delete_tag_dir(checkpoint_dir, tag, "checkpoint retention"):
            deleted.append(tag)
    if deleted:
        survivors = checkpoint_tags_desc(checkpoint_dir)
        log_message(
            f"[checkpoint] retained newest {keep_latest}: "
            f"{survivors if survivors else 'none'}"
        )
    return deleted


def _step_from_tag(tag: str | None) -> int:
    if not tag:
        return 0
    try:
        return max(0, int(str(tag).split("_", 1)[1]))
    except (IndexError, ValueError):
        return 0


def save_checkpoint(
    cfg: DictConfig,
    run_name: str,
    generator: nn.Module,
    discriminator: nn.Module,
    gen_opt: torch.optim.Optimizer,
    disc_opt: torch.optim.Optimizer,
    gen_sched: Any,
    disc_sched: Any,
    client_state: dict,
) -> None:
    """Save all state to ``step_N/checkpoint.pt`` on rank 0; all ranks barrier."""
    _distributed_barrier()
    if not is_main_process():
        _distributed_barrier()
        return

    out_dir = _checkpoint_dir_for_run(cfg, run_name)
    tag = f"step_{int(client_state['step'])}"
    tag_dir = out_dir / tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    gen_state   = generator.module.state_dict()     if isinstance(generator,     DDP) else generator.state_dict()
    disc_state  = discriminator.module.state_dict() if isinstance(discriminator, DDP) else discriminator.state_dict()

    torch.save(
        {
            "generator":     gen_state,
            "discriminator": disc_state,
            "gen_opt":       gen_opt.state_dict(),
            "disc_opt":      disc_opt.state_dict(),
            "gen_sched":     gen_sched.state_dict() if gen_sched else None,
            "disc_sched":    disc_sched.state_dict() if disc_sched else None,
            "client_state":  client_state,
        },
        tag_dir / "checkpoint.pt",
    )
    # Human-readable sidecar
    meta_path = out_dir / f"run_meta_{tag}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({k: (v if not isinstance(v, Path) else str(v)) for k, v in client_state.items()}, f, indent=2)

    _write_latest_tag(out_dir)
    _prune_checkpoint_history(out_dir, keep_latest=2)
    log_message(f"[checkpoint] saved {tag} in {out_dir}")
    _distributed_barrier()


def load_checkpoint(
    cfg: DictConfig,
    run_name: str,
    generator: nn.Module,
    discriminator: nn.Module,
    gen_opt: torch.optim.Optimizer,
    disc_opt: torch.optim.Optimizer,
    gen_sched: Any,
    disc_sched: Any,
    device: torch.device,
) -> dict:
    """Attempt to load the latest checkpoint; return client_state dict (empty if none)."""
    resume_from = cfg.train.get("resume_from")
    ckpt_dir = _checkpoint_dir_for_run(cfg, run_name)

    if resume_from and str(resume_from) not in ("", "from_scratch", "null"):
        rp = Path(str(resume_from))
        attempts = [(str(rp.parent), rp.name)]
    else:
        tags = checkpoint_tags_desc(ckpt_dir)
        attempts = [(str(ckpt_dir), tags[0])] if tags else []
        if len(tags) > 1:
            attempts.append((str(ckpt_dir), tags[1]))

    for attempt_dir, attempt_tag in attempts:
        ckpt_path = Path(attempt_dir) / attempt_tag / "checkpoint.pt"
        if not ckpt_path.exists():
            log_message(f"[checkpoint] not found: {ckpt_path}")
            continue
        try:
            state = torch.load(str(ckpt_path), map_location=device, weights_only=False)

            raw_gen  = generator.module   if isinstance(generator,     DDP) else generator
            raw_disc = discriminator.module if isinstance(discriminator, DDP) else discriminator
            raw_gen.load_state_dict(state["generator"])
            raw_disc.load_state_dict(state["discriminator"])
            gen_opt.load_state_dict(state["gen_opt"])
            disc_opt.load_state_dict(state["disc_opt"])
            if gen_sched and state.get("gen_sched"):
                gen_sched.load_state_dict(state["gen_sched"])
            if disc_sched and state.get("disc_sched"):
                disc_sched.load_state_dict(state["disc_sched"])

            client = state.get("client_state", {})
            log_message(
                f"[checkpoint] loaded {attempt_tag} from {attempt_dir} "
                f"(step={client.get('step', 0)} epoch={client.get('epoch', 0)})"
            )
            return client
        except Exception as exc:
            log_message(
                f"[checkpoint] failed to load {ckpt_path}: {type(exc).__name__}: {exc}\n"
                + traceback.format_exc().rstrip()
            )
            if is_main_process():
                _delete_tag_dir(Path(attempt_dir), attempt_tag, "failed restore")
    return {}


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_models(
    cfg: DictConfig,
    device: torch.device,
) -> tuple[UNetGenerator, PatchDiscriminator]:
    generator     = UNetGenerator(
        base_filters=int(cfg.model.get("generator_base_filters", 64)),
    ).to(device)
    discriminator = PatchDiscriminator(
        base_filters=int(cfg.model.get("discriminator_base_filters", 64)),
    ).to(device)
    n_gen  = sum(p.numel() for p in generator.parameters())
    n_disc = sum(p.numel() for p in discriminator.parameters())
    log_message(f"[model] generator  params: {n_gen:,}")
    log_message(f"[model] discriminator params: {n_disc:,}")
    return generator, discriminator


# ---------------------------------------------------------------------------
# Optimizer / scheduler (same cosine-with-warmup pattern as src.train)
# ---------------------------------------------------------------------------

def build_optimizers(
    cfg: DictConfig,
    generator: nn.Module,
    discriminator: nn.Module,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    lr_g  = float(cfg.train.optimizer.lr_generator)
    lr_d  = float(cfg.train.optimizer.lr_discriminator)
    betas = tuple(cfg.train.optimizer.betas)
    wd    = float(cfg.train.optimizer.weight_decay)
    gen_opt  = torch.optim.AdamW(generator.parameters(),     lr=lr_g, betas=betas, weight_decay=wd)
    disc_opt = torch.optim.AdamW(discriminator.parameters(), lr=lr_d, betas=betas, weight_decay=wd)
    return gen_opt, disc_opt


def build_schedulers(
    cfg: DictConfig,
    gen_opt: torch.optim.Optimizer,
    disc_opt: torch.optim.Optimizer,
    total_steps: int,
) -> tuple[Any, Any]:
    """Cosine decay with linear warm-up (same logic as src.train.build_scheduler)."""
    warmup   = int(cfg.train.scheduler.warmup_steps)
    min_lr   = float(cfg.train.scheduler.min_lr)
    max_steps = max(int(getattr(cfg.train.scheduler, "max_steps", total_steps)), total_steps)

    def _make_lambda(ref_lr: float):
        min_ratio = min_lr / max(ref_lr, 1e-12)
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = min(1.0, float(step - warmup) / float(max(1, max_steps - warmup)))
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine
        return lr_lambda

    gen_sched  = torch.optim.lr_scheduler.LambdaLR(gen_opt,  _make_lambda(float(gen_opt.param_groups[0]["lr"])))
    disc_sched = torch.optim.lr_scheduler.LambdaLR(disc_opt, _make_lambda(float(disc_opt.param_groups[0]["lr"])))
    return gen_sched, disc_sched


# ---------------------------------------------------------------------------
# Dataloader builders
# ---------------------------------------------------------------------------

def _micro_batch(cfg: DictConfig) -> int:
    return int(cfg.train.batch_size)


def build_train_loader(cfg: DictConfig, world_size: int, rank: int) -> tuple[DataLoader, PairedArtDataset, Any]:
    dataset = PairedArtDataset(
        data_root=str(cfg.train.data_root),
        resolution=int(cfg.train.resolution),
        split="train",
        val_fraction=float(getattr(cfg.train, "val_fraction", 0.15)),
        seed=int(getattr(cfg.train, "seed", 42)),
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=_micro_batch(cfg),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=int(getattr(cfg.train, "num_workers", 4)),
        pin_memory=bool(getattr(cfg.train, "pin_memory", True)),
        persistent_workers=bool(getattr(cfg.train, "persistent_workers", True)),
        prefetch_factor=int(getattr(cfg.train, "prefetch_factor", 2)),
        drop_last=False,
    )
    return loader, dataset, sampler


def build_val_loader(cfg: DictConfig, world_size: int, rank: int) -> tuple[DataLoader, PairedArtDataset, Any]:
    dataset = PairedArtDataset(
        data_root=str(cfg.train.data_root),
        resolution=int(cfg.train.resolution),
        split="val",
        val_fraction=float(getattr(cfg.train, "val_fraction", 0.15)),
        seed=int(getattr(cfg.train, "seed", 42)),
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=_micro_batch(cfg),
        sampler=sampler,
        shuffle=False,
        num_workers=int(getattr(cfg.train, "num_workers", 2)),
        pin_memory=bool(getattr(cfg.train, "pin_memory", True)),
        drop_last=False,
    )
    return loader, dataset, sampler


def build_wikiart_val_dataset(cfg: DictConfig) -> WikiArtValDataset:
    return WikiArtValDataset(
        val_dir=str(cfg.train.wikiart_val_dir),
        resolution=int(cfg.train.resolution),
    )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def lsgan_loss(pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    """Least-squares GAN loss: MSE vs 1 (real) or 0 (fake)."""
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    generator: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Pixel-space L1 loss on held-out pairs, aggregated across all ranks."""
    raw_gen = generator.module if isinstance(generator, DDP) else generator
    was_training = raw_gen.training
    raw_gen.eval()

    loss_sum    = torch.zeros(1, device=device, dtype=torch.float64)
    sample_count = torch.zeros(1, device=device, dtype=torch.float64)

    with torch.no_grad():
        for batch in val_loader:
            clean   = batch["clean"].to(device,   non_blocking=True)
            damaged = batch["damaged"].to(device, non_blocking=True)
            fake    = generator(clean)
            loss_sum    += F.l1_loss(fake, damaged, reduction="sum").double()
            sample_count += clean.shape[0]

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum,     op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)

    if was_training:
        raw_gen.train()

    return {"l1_loss": float((loss_sum / sample_count.clamp(min=1)).item())}


# ---------------------------------------------------------------------------
# Inference / visualisation panels
# ---------------------------------------------------------------------------

def run_wikiart_inference(
    generator: nn.Module,
    wikiart_dataset: WikiArtValDataset,
    cfg: DictConfig,
    device: torch.device,
    global_step: int,
) -> torch.Tensor | None:
    """Apply the generator to a fixed set of WikiArt val images; return panel grid.

    Each panel is a horizontal triplet: [clean | generated_damage | diff].
    Results are gathered from all ranks into a (N, 3, H, 3*W) tensor on rank 0.
    """
    indices = fixed_inference_indices(cfg, len(wikiart_dataset))
    if not indices:
        return None

    rank       = get_global_rank()
    world_size = get_world_size()
    local_indices = indices[rank::world_size]
    local_payload: list[tuple[int, torch.Tensor]] = []

    raw_gen = generator.module if isinstance(generator, DDP) else generator
    was_training = raw_gen.training
    raw_gen.eval()

    with torch.no_grad():
        for idx in local_indices:
            sample = wikiart_dataset[idx]
            clean = sample["clean"].unsqueeze(0).to(device)
            fake_damaged = generator(clean)

            diff = (fake_damaged - clean).abs().clamp(0.0, 1.0)
            panel = torch.cat(
                [clean.squeeze(0).cpu(), fake_damaged.squeeze(0).cpu(), diff.squeeze(0).cpu()],
                dim=2,
            ).clamp(0.0, 1.0)
            local_payload.append((idx, (panel * 255.0).round().to(torch.uint8)))

    if was_training:
        raw_gen.train()

    panels = gather_inference_panels(local_payload, indices)
    if is_main_process() and panels is not None:
        log_message(
            f"[inference] prepared {int(panels.shape[0])} WikiArt panels for step={global_step}"
        )
    return panels


def run_val_pair_panels(
    generator: nn.Module,
    val_loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    global_step: int,
) -> torch.Tensor | None:
    """Sample a few val pairs and show [clean | real_damaged | generated_damage]."""
    num_panels = int(getattr(cfg.wandb, "log_images_num", 8))
    if num_panels <= 0 or not is_main_process():
        return None

    raw_gen = generator.module if isinstance(generator, DDP) else generator
    was_training = raw_gen.training
    raw_gen.eval()

    panels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            clean   = batch["clean"].to(device)
            damaged = batch["damaged"].to(device)
            fake    = generator(clean)
            for i in range(clean.shape[0]):
                panel = torch.cat(
                    [clean[i].cpu(), damaged[i].cpu(), fake[i].cpu()],
                    dim=2,
                ).clamp(0.0, 1.0)
                panels.append(panel)
                if len(panels) >= num_panels:
                    break
            if len(panels) >= num_panels:
                break

    if was_training:
        raw_gen.train()

    if not panels:
        return None
    result = torch.stack(panels[:num_panels], dim=0)
    log_message(f"[val-panels] prepared {len(panels)} val pair panels at step={global_step}")
    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(cfg: DictConfig) -> None:
    """Initialize distributed training, run epochs, log, validate, checkpoint.

    Pipeline:
        1. torch.distributed init (NCCL).
        2. Build generator + discriminator; wrap with DDP.
        3. Build train / val dataloaders + WikiArt val dataset.
        4. Build AdamW optimizers + LambdaLR schedulers.
        5. Attempt checkpoint resume.
        6. Epoch loop: G step → D step → logging → val → checkpoint.
        7. Final save + wandb finish.
    """
    dist.init_process_group(backend="nccl")
    device     = get_device()
    world_size = get_world_size()
    rank       = get_global_rank()

    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(int(cfg.train.seed) + rank)
    random.seed(int(cfg.train.seed) + rank)

    wandb_init(cfg)
    run_name = _sanitize_run_name(cfg.wandb.get("run_name", "corruption-gan"))

    if is_main_process():
        _checkpoint_dir_for_run(cfg, run_name).mkdir(parents=True, exist_ok=True)

    # --- Models ---
    generator, discriminator = setup_models(cfg, device)
    if world_size > 1:
        generator     = DDP(generator,     device_ids=[device.index])
        discriminator = DDP(discriminator, device_ids=[device.index])

    # --- Data ---
    train_loader, train_dataset, train_sampler = build_train_loader(cfg, world_size, rank)
    val_loader,   val_dataset,   val_sampler   = build_val_loader(cfg, world_size, rank)
    wikiart_dataset = build_wikiart_val_dataset(cfg)
    inference_indices = fixed_inference_indices(cfg, len(wikiart_dataset))
    if is_main_process() and inference_indices:
        log_message(f"[inference] fixed WikiArt indices: {inference_indices}")

    if is_main_process():
        log_message(f"[data] train pairs: {len(train_dataset)}  val pairs: {len(val_dataset)}")
        log_message(f"[data] WikiArt val images: {len(wikiart_dataset)}")

    # --- Optimizers / schedulers ---
    steps_per_epoch     = max(len(train_loader), 1)
    total_optimizer_steps = max(1, int(cfg.train.num_epochs) * steps_per_epoch)
    gen_opt,  disc_opt  = build_optimizers(cfg, generator, discriminator)
    gen_sched, disc_sched = build_schedulers(cfg, gen_opt, disc_opt, total_optimizer_steps)

    lambda_l1 = float(cfg.train.lambda_l1)

    # --- Checkpoint resume ---
    global_step    = 0
    start_epoch    = 0
    last_save_step = 0
    last_logged_step = 0

    client = load_checkpoint(
        cfg, run_name,
        generator, discriminator,
        gen_opt, disc_opt, gen_sched, disc_sched,
        device,
    )
    if client:
        global_step    = int(client.get("step",  0))
        start_epoch    = int(client.get("epoch", 0))
        last_save_step = global_step
        last_logged_step = global_step
        if is_main_process():
            log_message(f"[resume] step={global_step} epoch={start_epoch + 1}")
    _distributed_barrier()

    # --- Interval knobs ---
    save_every       = int(cfg.train.save_every)
    val_every        = int(getattr(cfg.train, "val_every", 50))
    log_every        = int(cfg.train.log_every)
    log_images_every = int(cfg.wandb.get("log_images_every", 200)) if cfg.get("wandb") else 10**9
    num_images       = int(cfg.wandb.get("log_images_num", 8))     if cfg.get("wandb") else 0

    if is_main_process():
        log_message(f"[train] starting from epoch {start_epoch + 1}, step {global_step}")

    generator.train()
    discriminator.train()

    for epoch in range(start_epoch, int(cfg.train.num_epochs)):
        display_epoch = epoch + 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        pbar = tqdm(
            train_loader,
            disable=not is_main_process(),
            desc=f"epoch {display_epoch}",
        )
        t_last = time.perf_counter()

        for batch in pbar:
            clean   = batch["clean"].to(device,   non_blocking=True)
            damaged = batch["damaged"].to(device, non_blocking=True)

            # ---- Generator step ----
            gen_opt.zero_grad(set_to_none=True)
            fake_damaged = generator(clean)
            disc_fake_for_g = discriminator(clean, fake_damaged)
            g_adv  = lsgan_loss(disc_fake_for_g, target_is_real=True)
            g_l1   = F.l1_loss(fake_damaged, damaged)
            g_loss = g_adv + lambda_l1 * g_l1
            g_loss.backward()
            if cfg.train.get("gradient_clipping", 0.0) > 0:
                nn.utils.clip_grad_norm_(generator.parameters(), float(cfg.train.gradient_clipping))
            gen_opt.step()
            gen_sched.step()

            # ---- Discriminator step ----
            disc_opt.zero_grad(set_to_none=True)
            disc_real = discriminator(clean, damaged)
            disc_fake = discriminator(clean, fake_damaged.detach())
            d_real = lsgan_loss(disc_real, target_is_real=True)
            d_fake = lsgan_loss(disc_fake, target_is_real=False)
            d_loss = 0.5 * (d_real + d_fake)
            d_loss.backward()
            if cfg.train.get("gradient_clipping", 0.0) > 0:
                nn.utils.clip_grad_norm_(discriminator.parameters(), float(cfg.train.gradient_clipping))
            disc_opt.step()
            disc_sched.step()

            global_step += 1

            if is_main_process():
                pbar.set_postfix(
                    g=f"{g_loss.item():.3f}",
                    d=f"{d_loss.item():.3f}",
                    step=global_step,
                )

            # ---- Logging ----
            if is_main_process() and global_step % log_every == 0:
                dt = time.perf_counter() - t_last
                t_last = time.perf_counter()
                steps_since = max(global_step - last_logged_step, 1)
                imgs_per_sec = (steps_since * clean.shape[0] * world_size) / max(dt, 1e-6)
                lrs = [pg["lr"] for pg in gen_opt.param_groups]
                metrics = {
                    "train/g_loss":     g_loss.item(),
                    "train/g_adv_loss": g_adv.item(),
                    "train/g_l1_loss":  g_l1.item(),
                    "train/d_loss":     d_loss.item(),
                    "train/d_real_loss": d_real.item(),
                    "train/d_fake_loss": d_fake.item(),
                    "train/imgs_per_sec": imgs_per_sec,
                    "train/epoch":      display_epoch,
                    "train/lr_gen":     lrs[0] if lrs else 0.0,
                }
                log_message(
                    f"[train] epoch={display_epoch} step={global_step} "
                    f"g_loss={g_loss.item():.4f} d_loss={d_loss.item():.4f} "
                    f"imgs/s={imgs_per_sec:.1f}"
                )
                wandb_log(metrics, step=global_step)
                last_logged_step = global_step

            # ---- Checkpoint ----
            if global_step > 0 and global_step % save_every == 0 and global_step != last_save_step:
                client_state = {"step": global_step, "epoch": epoch}
                save_checkpoint(
                    cfg, run_name,
                    generator, discriminator,
                    gen_opt, disc_opt, gen_sched, disc_sched,
                    client_state,
                )
                last_save_step = global_step

            # ---- Validation + inference panels ----
            if val_every > 0 and global_step % val_every == 0 and global_step > 0:
                if is_main_process():
                    log_message("========STARTING VALIDATION========")
                val_metrics = validate(generator, val_loader, device)
                if is_main_process():
                    log_message(
                        f"[val] epoch={display_epoch} step={global_step} "
                        f"l1={val_metrics['l1_loss']:.4f}"
                    )
                    wandb_log({"val/l1_loss": val_metrics["l1_loss"]}, step=global_step)
                    log_message("========ENDING VALIDATION========")

                should_log_images = (
                    num_images > 0
                    and cfg.wandb.get("log_images", False)
                    and global_step % log_images_every == 0
                )
                if should_log_images:
                    if is_main_process():
                        log_message("========STARTING INFERENCE========")

                    wiki_panels = run_wikiart_inference(
                        generator, wikiart_dataset, cfg, device, global_step
                    )
                    val_panels = run_val_pair_panels(
                        generator, val_loader, cfg, device, global_step
                    )
                    if is_main_process():
                        img_dict: dict[str, torch.Tensor] = {}
                        if wiki_panels is not None:
                            img_dict["inference/wikiart_panels"] = wiki_panels
                        if val_panels is not None:
                            img_dict["inference/val_pair_panels"] = val_panels
                        if img_dict:
                            wandb_log({}, step=global_step, images=img_dict)
                            log_message(
                                f"[inference] logged {sum(v.shape[0] for v in img_dict.values())} "
                                f"panels to wandb at step={global_step}"
                            )
                        log_message("========ENDING INFERENCE========")

                _distributed_barrier()

    # --- Final checkpoint ---
    client_state = {"step": global_step, "epoch": int(cfg.train.num_epochs)}
    save_checkpoint(
        cfg, run_name,
        generator, discriminator,
        gen_opt, disc_opt, gen_sched, disc_sched,
        client_state,
    )
    if is_main_process():
        log_message(f"[train] finished. final step={global_step}")
    wandb_finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learned corruption GAN training")
    parser.add_argument(
        "--config",
        type=str,
        default="train/configs/train_corruption_gan.yaml",
        help="Path to training YAML config",
    )
    args, unknown = parser.parse_known_args()
    conf = load_config(args.config, unknown if unknown else None)
    main(conf)
