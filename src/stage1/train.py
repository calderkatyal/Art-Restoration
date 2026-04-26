"""Stage 1 training: damage generator G against PatchGAN discriminator D.

Per step:
    1. Sample ``clean`` from the clean stream and ``real_damaged`` from the
       damaged stream (independent, unpaired).
    2. ``z ~ N(0, I)``, ``fake = G(clean, z)``.
    3. D step: hinge loss on ``D(real)`` vs ``D(fake.detach())``.
    4. G step: hinge ``L_adv`` + shallow LPIPS ``L_content`` + diversity ``L_div``.
    5. Logging to WandB on rank 0; periodic image grid (clean | fake | real).

Distributed training uses plain torch DDP via ``torchrun``. WandB and image
logging only run on global rank 0.

Usage:
    torchrun --nproc_per_node=N --master_addr=$MASTER_ADDR --master_port=29500 \\
        -m src.stage1.train --config train/configs/stage1.yaml

Arguments:
    --config    Path to YAML (default: ``train/configs/stage1.yaml``).
    overrides   Dot-notation overrides (OmegaConf), e.g. ``train.batch_size=8``.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .dataset import build_clean_loader, build_damaged_loader, cycle
from .discriminator import PatchDiscriminator
from .generator import DamageGenerator
from .losses import content_loss, diversity_loss, hinge_d_loss, hinge_g_loss


# ---------- Distributed / logging helpers ----------


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


_WANDB_ACTIVE = False


def _wandb_init(cfg: DictConfig) -> None:
    global _WANDB_ACTIVE
    wb = cfg.get("wandb")
    if wb is None or not bool(wb.get("enabled", False)) or not _is_main():
        return
    if not wb.get("run_name"):
        raise ValueError("cfg.wandb.run_name must be set explicitly.")
    import wandb

    tags = list(wb.get("tags") or [])
    wandb.init(
        project=wb.get("project", "art-restoration-stage1"),
        entity=wb.get("entity") or None,
        name=wb.get("run_name"),
        tags=tags if tags else None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    _WANDB_ACTIVE = True


def _wandb_log(metrics: dict, step: int, images: dict | None = None) -> None:
    if not _WANDB_ACTIVE:
        return
    import wandb
    from torchvision.utils import make_grid

    payload = dict(metrics)
    if images:
        for name, tensor in images.items():
            if tensor is None:
                continue
            grid = make_grid(tensor.detach().cpu().clamp(0.0, 1.0), nrow=min(4, int(tensor.shape[0])))
            payload[name] = wandb.Image(
                grid.permute(1, 2, 0).numpy(),
                caption=f"{name} step={step}",
            )
    wandb.log(payload, step=step)


def _wandb_finish() -> None:
    global _WANDB_ACTIVE
    if not _WANDB_ACTIVE:
        return
    import wandb

    wandb.finish()
    _WANDB_ACTIVE = False


# ---------- Checkpointing ----------


def _save_checkpoint(
    out_dir: Path,
    step: int,
    g_module: nn.Module,
    d_module: nn.Module,
    g_opt: torch.optim.Optimizer,
    d_opt: torch.optim.Optimizer,
    cfg: DictConfig,
    keep_last: int,
) -> None:
    if not _is_main():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"step_{int(step):08d}"
    path = out_dir / f"{tag}.pt"
    torch.save(
        {
            "step": int(step),
            "generator": g_module.state_dict(),
            "discriminator": d_module.state_dict(),
            "g_optimizer": g_opt.state_dict(),
            "d_optimizer": d_opt.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )
    _log(f"[checkpoint] saved {path}")
    # Retention
    keep_last = max(int(keep_last), 1)
    existing = sorted(out_dir.glob("step_*.pt"))
    for old in existing[:-keep_last]:
        try:
            old.unlink()
            _log(f"[checkpoint] removed old {old}")
        except OSError:
            pass


def _maybe_load_checkpoint(
    cfg: DictConfig,
    g_module: nn.Module,
    d_module: nn.Module,
    g_opt: torch.optim.Optimizer,
    d_opt: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    resume_from = cfg.train.get("resume_from")
    if not resume_from:
        return 0
    path = Path(str(resume_from))
    if not path.is_file():
        if _is_main():
            _log(f"[resume] resume_from not found: {path}; starting from scratch")
        return 0
    payload = torch.load(path, map_location=device)
    g_module.load_state_dict(payload["generator"])
    d_module.load_state_dict(payload["discriminator"])
    g_opt.load_state_dict(payload["g_optimizer"])
    d_opt.load_state_dict(payload["d_optimizer"])
    step = int(payload.get("step", 0))
    if _is_main():
        _log(f"[resume] loaded {path} at step={step}")
    return step


# ---------- Sample logging ----------


@torch.no_grad()
def _build_sample_grid(
    g_module: DamageGenerator,
    clean_batch: torch.Tensor,
    real_batch: torch.Tensor,
    num_images: int,
    device: torch.device,
) -> dict:
    """Build clean | fake | real triplets for WandB logging.

    All tensors are converted from ``[-1, 1]`` to ``[0, 1]`` for display.
    """
    g_module.eval()
    n = min(int(num_images), int(clean_batch.shape[0]), int(real_batch.shape[0]))
    if n <= 0:
        g_module.train()
        return {}
    clean = clean_batch[:n].to(device)
    real = real_batch[:n].to(device)
    z = g_module.sample_noise(n, device=device)
    fake = g_module(clean, z)
    g_module.train()

    def to_disp(x: torch.Tensor) -> torch.Tensor:
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)

    return {
        "samples/clean": to_disp(clean),
        "samples/fake_damaged": to_disp(fake),
        "samples/real_damaged": to_disp(real),
    }


# ---------- Optimizers ----------


def _build_optimizers(
    g: nn.Module, d: nn.Module, cfg: DictConfig
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    g_lr = float(cfg.train.optimizer.g_lr)
    d_lr = float(cfg.train.optimizer.d_lr)
    betas = tuple(cfg.train.optimizer.betas)
    wd = float(cfg.train.optimizer.weight_decay)
    g_opt = torch.optim.AdamW(g.parameters(), lr=g_lr, betas=betas, weight_decay=wd)
    d_opt = torch.optim.AdamW(d.parameters(), lr=d_lr, betas=betas, weight_decay=wd)
    return g_opt, d_opt


# ---------- Main loop ----------


def main(cfg: DictConfig) -> None:
    """Initialize DDP, build G/D and dataloaders, and run the GAN training loop."""
    distributed = _get_world_size() > 1
    if distributed:
        dist.init_process_group(backend="nccl")
    local_rank = _get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    rank = _get_rank()

    seed = int(cfg.train.seed)
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    _wandb_init(cfg)
    if _is_main():
        _log(f"[setup] world_size={_get_world_size()} rank={rank} device={device}")
        _log(f"[setup] resolution={int(cfg.train.resolution)} batch_size={int(cfg.train.batch_size)}")

    # ----- Models -----
    g = DamageGenerator(
        base_channels=int(cfg.model.g_base_channels),
        noise_dim=int(cfg.model.noise_dim),
        delta_scale=float(cfg.model.delta_scale),
    ).to(device)
    d = PatchDiscriminator(
        base_channels=int(cfg.model.d_base_channels),
        n_layers=int(cfg.model.d_n_layers),
        num_scales=int(cfg.model.d_num_scales),
    ).to(device)

    if _is_main():
        n_g = sum(p.numel() for p in g.parameters())
        n_d = sum(p.numel() for p in d.parameters())
        _log(f"[model] G params={n_g/1e6:.2f}M D params={n_d/1e6:.2f}M")

    if distributed:
        g = DDP(g, device_ids=[local_rank], find_unused_parameters=False)
        d = DDP(d, device_ids=[local_rank], find_unused_parameters=False)

    g_module = g.module if isinstance(g, DDP) else g
    d_module = d.module if isinstance(d, DDP) else d

    # ----- Optimizers -----
    g_opt, d_opt = _build_optimizers(g_module, d_module, cfg)

    # ----- LPIPS (used for content loss) -----
    import lpips  # heavy import; only loaded when training

    lpips_model = lpips.LPIPS(net=str(cfg.train.content.lpips_net), verbose=False).to(device)
    for p in lpips_model.parameters():
        p.requires_grad_(False)
    lpips_model.eval()
    lpips_layers_cfg = cfg.train.content.get("layer_indices")
    if lpips_layers_cfg in (None, "", [], "all"):
        lpips_layers = None
    else:
        lpips_layers = list(int(i) for i in lpips_layers_cfg)

    # ----- Data -----
    clean_roots = list(cfg.data.clean_dirs)
    damaged_roots = list(cfg.data.damaged_dirs)
    if _is_main():
        _log(f"[data] clean roots: {clean_roots}")
        _log(f"[data] damaged roots: {damaged_roots}")
    _, clean_loader = build_clean_loader(
        roots=clean_roots,
        resolution=int(cfg.train.resolution),
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        pin_memory=bool(cfg.train.pin_memory),
        persistent_workers=bool(cfg.train.persistent_workers),
        prefetch_factor=int(cfg.train.prefetch_factor),
        seed=seed,
        distributed=distributed,
    )
    _, damaged_loader = build_damaged_loader(
        roots=damaged_roots,
        resolution=int(cfg.train.resolution),
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        pin_memory=bool(cfg.train.pin_memory),
        persistent_workers=bool(cfg.train.persistent_workers),
        prefetch_factor=int(cfg.train.prefetch_factor),
        seed=seed + 1_000_003,
        distributed=distributed,
    )
    clean_iter = cycle(clean_loader)
    damaged_iter = cycle(damaged_loader)

    if _is_main():
        _log(f"[data] clean dataset size={len(clean_loader.dataset)}")
        _log(f"[data] damaged dataset size={len(damaged_loader.dataset)}")

    # ----- Resume -----
    start_step = _maybe_load_checkpoint(cfg, g_module, d_module, g_opt, d_opt, device)

    # ----- Loss weights -----
    w_adv = float(cfg.train.loss.adv_weight)
    w_content = float(cfg.train.loss.content_weight)
    w_div = float(cfg.train.loss.diversity_weight)
    g_steps_per_d = int(cfg.train.loss.g_steps_per_d)

    # ----- Cadence -----
    total_steps = int(cfg.train.total_steps)
    log_every = int(cfg.train.log_every)
    save_every = int(cfg.train.save_every)
    sample_every = int(cfg.train.sample_every)
    keep_last = int(cfg.train.keep_last_checkpoints)
    log_images_num = int(cfg.wandb.get("log_images_num", 8)) if cfg.get("wandb") else 8

    checkpoint_root = Path(str(cfg.train.checkpoint_root)) / str(cfg.wandb.run_name)
    if _is_main():
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    g.train()
    d.train()

    pbar = tqdm(
        range(start_step, total_steps),
        disable=not _is_main(),
        desc="stage1",
        initial=start_step,
        total=total_steps,
    )
    t_last = time.perf_counter()
    last_logged_step = start_step

    for step in pbar:
        global_step = step + 1

        clean = next(clean_iter).to(device, non_blocking=True)
        real = next(damaged_iter).to(device, non_blocking=True)

        # ---- D step ----
        z = g_module.sample_noise(clean.shape[0], device=device)
        with torch.no_grad():
            fake = g_module(clean, z)
        real_logits = d(real)
        fake_logits = d(fake)
        d_loss = hinge_d_loss(real_logits, fake_logits)

        d_opt.zero_grad(set_to_none=True)
        d_loss.backward()
        d_opt.step()

        # ---- G step(s) ----
        # Single DDP forward over a doubled batch with two noise vectors so we get
        # both diversity samples without two separate DDP forward passes.
        g_loss_total = 0.0
        adv_total = 0.0
        content_total = 0.0
        div_total = 0.0
        for _ in range(max(1, g_steps_per_d)):
            bs = int(clean.shape[0])
            z_a = g_module.sample_noise(bs, device=device)
            z_b = g_module.sample_noise(bs, device=device)
            clean_dup = clean.repeat(2, 1, 1, 1)
            z_dup = torch.cat([z_a, z_b], dim=0)
            fake_dup = g(clean_dup, z_dup)
            fake_a, fake_b = fake_dup[:bs], fake_dup[bs:]

            fake_logits_for_g = d(fake_a)
            adv = hinge_g_loss(fake_logits_for_g)
            l_content = content_loss(lpips_model, fake_a, clean, layer_indices=lpips_layers)
            l_div = diversity_loss(fake_a, fake_b, z_a, z_b)
            g_loss = w_adv * adv + w_content * l_content + w_div * l_div

            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_opt.step()

            g_loss_total += float(g_loss.item())
            adv_total += float(adv.item())
            content_total += float(l_content.item())
            div_total += float(l_div.item())

        denom = max(1, g_steps_per_d)
        g_loss_avg = g_loss_total / denom
        adv_avg = adv_total / denom
        content_avg = content_total / denom
        div_avg = div_total / denom

        if _is_main():
            pbar.set_postfix(
                step=global_step,
                d=f"{float(d_loss.item()):.3f}",
                g=f"{g_loss_avg:.3f}",
                adv=f"{adv_avg:.3f}",
            )

        # ---- Logging ----
        if _is_main() and global_step % log_every == 0:
            dt = time.perf_counter() - t_last
            t_last = time.perf_counter()
            steps_since = max(global_step - last_logged_step, 1)
            imgs_per_sec = (steps_since * int(cfg.train.batch_size) * _get_world_size()) / max(dt, 1e-6)
            metrics = {
                "train/d_loss": float(d_loss.item()),
                "train/g_loss": g_loss_avg,
                "train/adv_loss": adv_avg,
                "train/content_loss": content_avg,
                "train/diversity_loss": div_avg,
                "train/global_step": global_step,
                "train/images_per_sec": imgs_per_sec,
            }
            _log(
                f"[train] step={global_step} "
                f"d={metrics['train/d_loss']:.4f} g={metrics['train/g_loss']:.4f} "
                f"adv={metrics['train/adv_loss']:.4f} content={metrics['train/content_loss']:.4f} "
                f"div={metrics['train/diversity_loss']:.4f} img/s={imgs_per_sec:.1f}"
            )
            _wandb_log(metrics, step=global_step)
            last_logged_step = global_step

        # ---- Image samples to WandB ----
        if _is_main() and sample_every > 0 and global_step % sample_every == 0:
            with torch.no_grad():
                clean_log = next(clean_iter).to(device, non_blocking=True)
                real_log = next(damaged_iter).to(device, non_blocking=True)
            images = _build_sample_grid(g_module, clean_log, real_log, log_images_num, device=device)
            _wandb_log({}, step=global_step, images=images)

        # ---- Checkpoint ----
        if save_every > 0 and global_step % save_every == 0:
            _save_checkpoint(
                checkpoint_root,
                step=global_step,
                g_module=g_module,
                d_module=d_module,
                g_opt=g_opt,
                d_opt=d_opt,
                cfg=cfg,
                keep_last=keep_last,
            )
            if distributed:
                dist.barrier()

    # Final checkpoint
    if save_every > 0:
        _save_checkpoint(
            checkpoint_root,
            step=int(total_steps),
            g_module=g_module,
            d_module=d_module,
            g_opt=g_opt,
            d_opt=d_opt,
            cfg=cfg,
            keep_last=keep_last,
        )
    if distributed:
        dist.barrier()
    _wandb_finish()


def _parse_args() -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(description="Stage 1 damage-generator training")
    parser.add_argument("--config", type=str, default="train/configs/stage1.yaml")
    args, overrides = parser.parse_known_args()
    return args.config, overrides


def _load_config(yaml_path: str, overrides: list[str]) -> DictConfig:
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


if __name__ == "__main__":
    config_path, overrides = _parse_args()
    cfg = _load_config(config_path, overrides)
    main(cfg)
