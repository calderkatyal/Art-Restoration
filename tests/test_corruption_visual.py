"""Visual test: for every corruption type x {local, global} mode that is
enabled in the YAML config, produce two corrupted versions of a single
randomly-sampled painting — one at minimum severity and one at maximum
severity.

Usage:
    python -m tests.test_corruption_visual \
        --data_dir /path/to/wiki_art \
        --output_dir tests/corruption_results

    # Or pass a single specific image:
    python -m tests.test_corruption_visual \
        --image /path/to/persistence_of_memory.jpg \
        --output_dir tests/corruption_results
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

# Allow running as `python -m tests.test_corruption_visual` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corruption import CHANNEL_NAMES
from src.corruption.effects import (
    apply_craquelure, apply_rip_tear, apply_paint_loss,
    apply_yellowing, apply_fading, apply_deposits,
    apply_scratches,
)
from src.corruption.presets import (
    generate_global_mask, generate_local_mask, SHAPE_KIND_BY_CHANNEL,
)
from src.corruption.module import _affected_pixels, _per_component_hull_mask


EFFECT_FNS = {
    'craquelure': apply_craquelure,
    'rip_tear':   apply_rip_tear,
    'paint_loss': apply_paint_loss,
    'yellowing':  apply_yellowing,
    'fading':     apply_fading,
    'deposits':   apply_deposits,
    'scratches':  apply_scratches,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(root: str, max_images: int = 500) -> List[Path]:
    root = Path(root)
    images: List[Path] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            images.append(p)
            if len(images) >= max_images * 10:
                break
    random.shuffle(images)
    return images[:max_images]


def load_image(path: Path, resolution: int = 384) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize(
        (resolution, resolution), Image.LANCZOS
    )
    return T.ToTensor()(img)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    return T.ToPILImage()(t.clamp(0, 1))


def mask_to_pil(m: torch.Tensor) -> Image.Image:
    """Render a (H, W) mask in [0, 1] as a grayscale heatmap PIL image."""
    g = m.detach().cpu().clamp(0, 1)
    return T.ToPILImage()(g.unsqueeze(0))


def mask_boundary(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute a 1-pixel boundary of a binary mask using 4-neighborhood logic.
    Returns a boolean tensor of shape (H, W).
    """
    m = (mask.detach().cpu() > 0.5)

    up = torch.zeros_like(m)
    down = torch.zeros_like(m)
    left = torch.zeros_like(m)
    right = torch.zeros_like(m)

    up[1:] = m[:-1]
    down[:-1] = m[1:]
    left[:, 1:] = m[:, :-1]
    right[:, :-1] = m[:, 1:]

    interior = m & up & down & left & right
    boundary = m & (~interior)
    return boundary


def overlay_mask_boundary(
    image: Image.Image,
    mask: torch.Tensor,
    color: tuple[int, int, int] = (255, 0, 0),
    width: int = 1,
) -> Image.Image:
    """
    Overlay a bold boundary of the binary mask onto the image.
    The boundary is drawn by placing filled circles centered on boundary pixels.
    """
    out = image.copy()
    boundary = mask_boundary(mask)

    ys, xs = torch.nonzero(boundary, as_tuple=True)
    draw = ImageDraw.Draw(out)

    r = 0
    for y, x in zip(ys.tolist(), xs.tolist()):
        draw.point((x, y), fill=color)

    return out


def try_get_font(size: int = 18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def apply_single(
    image: torch.Tensor,
    corruption: str,
    mode: str,
    severity: float,
    type_cfg,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a single-channel mask and apply the corresponding effect."""
    H, W = image.shape[-2:]
    device = image.device

    if mode == "local":
        min_n = max(1, int(type_cfg.get("local_min_num", 1)))
        max_n = max(min_n, int(type_cfg.get("local_max_num", 4)))
        af = type_cfg.get("local_area_frac", [0.01, 0.05])
        shape_kind = SHAPE_KIND_BY_CHANNEL.get(corruption, 'generic')
        soft_mask, region_mask = generate_local_mask(
            H, W, severity,
            (min_n, max_n),
            (float(af[0]), float(af[1])),
            shape_kind=shape_kind,
            generator=generator, device=device,
        )
    elif mode == "global":
        soft_mask, region_mask = generate_global_mask(H, W, severity, device=device)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    fn = EFFECT_FNS[corruption]
    extra = {}
    if corruption == "scratches":
        extra["max_count"] = int(type_cfg.get("local_max_num", 8))
    # Compute the actual affected-pixel hull mask, matching what the training
    # pipeline produces (module.py uses the same before/after diff approach).
    # This gives a tight mask around the actual damage rather than the wider
    # input band, which is particularly important for scratches.
    before = image.clone()
    out_img = fn(image, soft_mask, generator=generator, **extra)
    diff_bool = _affected_pixels(before, out_img, threshold=0.008)
    H2, W2 = image.shape[-2:]
    actual_mask = _per_component_hull_mask(diff_bool, H2, W2, merge_radius=3)
    return out_img, actual_mask


def build_cells(corruption_cfg) -> List[Tuple[str, str]]:
    """Enumerate (type, mode) pairs that are enabled in the config."""
    cells = []
    for name in CHANNEL_NAMES:
        t = corruption_cfg.types[name]
        if bool(t.get("local_enabled", False)):
            cells.append((name, "local"))
        if bool(t.get("global_enabled", False)):
            cells.append((name, "global"))
    return cells


def build_grid(
    output_dir: str,
    corruption_cfg_path: str,
    source_path: Path,
    resolution: int = 384,
    seed: int = 42,
    min_severity: float = 0.01,
    max_severity: float = 1.0,
):
    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    corruption_cfg = OmegaConf.load(corruption_cfg_path)

    print(f"Using source image: {source_path}")
    source = load_image(source_path, resolution)

    cells = build_cells(corruption_cfg)
    n_cells = len(cells)
    print(f"Rendering {n_cells} (type, mode) cells at severities {min_severity}, {max_severity}")

    # Layout: per cell show [min-severity image | max-severity image] with label above.
    img_gap = 8
    cell_w = resolution * 2 + img_gap
    label_h = 28
    cell_h = resolution + label_h + 6
    padding = 10

    # Always 3 columns; wrap rows.
    n_cols = 3
    n_rows = math.ceil(n_cells / n_cols)

    grid_w = n_cols * cell_w + (n_cols + 1) * padding
    grid_h = n_rows * cell_h + (n_rows + 1) * padding + 80  # +80 for title + source tag

    grid = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    mask_grid = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    draw = ImageDraw.Draw(grid)
    mask_draw = ImageDraw.Draw(mask_grid)
    font = try_get_font(18)
    title_font = try_get_font(24)
    small_font = try_get_font(12)

    draw.text(
        (padding, 10),
        "Corruption Pipeline Visual Test  -  per-type min vs max severity",
        fill=(240, 240, 240), font=title_font,
    )
    draw.text(
        (padding, 44),
        f"Source: {source_path.name}   |   min_severity={min_severity}   max_severity={max_severity}",
        fill=(180, 180, 180), font=small_font,
    )
    mask_draw.text(
        (padding, 10),
        "Corruption Pipeline Mask Heatmaps  -  per-type min vs max severity",
        fill=(240, 240, 240), font=title_font,
    )
    mask_draw.text(
        (padding, 44),
        f"Source: {source_path.name}   |   min_severity={min_severity}   max_severity={max_severity}",
        fill=(180, 180, 180), font=small_font,
    )

    for idx, (corr, mode) in enumerate(cells):
        col = idx % n_cols
        row = idx // n_cols
        x0 = padding + col * (cell_w + padding)
        y0 = 80 + padding + row * (cell_h + padding)

        type_cfg = corruption_cfg.types[corr]

        local_min = float(type_cfg.get("min_severity", min_severity))
        local_max = float(type_cfg.get("max_severity", max_severity))
        # User-specified defaults clamped by the per-type range, to keep
        # the corruption within its valid operating regime.
        s_lo = max(min_severity, local_min)
        s_hi = min(max_severity, local_max)

        # Use the SAME seed for both min and max severity so the mask is
        # generated at the same location — lets the user directly compare
        # how intensity alone affects the appearance.
        gen_lo = torch.Generator(device='cpu')
        gen_lo.manual_seed(seed + idx)
        gen_hi = torch.Generator(device='cpu')
        gen_hi.manual_seed(seed + idx)

        lo_img, lo_mask = apply_single(source, corr, mode, s_lo, type_cfg, gen_lo)
        hi_img, hi_mask = apply_single(source, corr, mode, s_hi, type_cfg, gen_hi)

        color = (200, 255, 200) if mode == "local" else (200, 220, 255)
        label = f"[{mode}] {corr}"
        draw.text((x0 + 4, y0), label, fill=color, font=font)
        mask_draw.text((x0 + 4, y0), label, fill=color, font=font)

        lo_img_pil = tensor_to_pil(lo_img)
        hi_img_pil = tensor_to_pil(hi_img)

        # NEW: overlay bold red mask boundaries onto the corrupted images
        lo_img_pil = overlay_mask_boundary(lo_img_pil, lo_mask, color=(255, 0, 0), width=5)
        hi_img_pil = overlay_mask_boundary(hi_img_pil, hi_mask, color=(255, 0, 0), width=5)

        grid.paste(lo_img_pil, (x0, y0 + label_h))
        grid.paste(hi_img_pil, (x0 + resolution + img_gap, y0 + label_h))

        mask_grid.paste(mask_to_pil(lo_mask), (x0, y0 + label_h))
        mask_grid.paste(mask_to_pil(hi_mask), (x0 + resolution + img_gap, y0 + label_h))

        draw.text(
            (x0 + 4, y0 + label_h + resolution - 16),
            f"severity={s_lo:.2f}", fill=(200, 200, 200), font=small_font,
        )
        draw.text(
            (x0 + resolution + img_gap + 4, y0 + label_h + resolution - 16),
            f"severity={s_hi:.2f}", fill=(200, 200, 200), font=small_font,
        )
        mask_draw.text(
            (x0 + 4, y0 + label_h + resolution - 16),
            f"severity={s_lo:.2f}", fill=(200, 200, 200), font=small_font,
        )
        mask_draw.text(
            (x0 + resolution + img_gap + 4, y0 + label_h + resolution - 16),
            f"severity={s_hi:.2f}", fill=(200, 200, 200), font=small_font,
        )

        print(f"  {idx+1:2d}/{n_cells}  [{mode}] {corr:<20s} {s_lo:.2f} -> {s_hi:.2f}")

    grid_path = os.path.join(output_dir, "corruption_grid.png")
    grid.save(grid_path, quality=95)
    print(f"\nGrid saved to {grid_path}")
    mask_grid_path = os.path.join(output_dir, "corruption_grid_masks.png")
    mask_grid.save(mask_grid_path, quality=95)
    print(f"Mask heatmap grid saved to {mask_grid_path}")

    # Snapshot the corruption config used for the run.
    config_out = os.path.join(output_dir, "config.yaml")
    conf_dict = {
        "corruption": OmegaConf.to_container(corruption_cfg, resolve=True),
        "test_settings": {
            "resolution": resolution,
            "seed": seed,
            "source_image": str(source_path),
            "min_severity": min_severity,
            "max_severity": max_severity,
            "num_cells": n_cells,
        },
    }
    OmegaConf.save(OmegaConf.create(conf_dict), config_out)
    print(f"Config saved to {config_out}")

    return grid_path, config_out


def _resolve_source(args) -> Path:
    if args.image is not None:
        p = Path(args.image)
        if not p.exists():
            print(f"ERROR: --image path does not exist: {p}")
            sys.exit(1)
        return p
    print(f"Scanning {args.data_dir} for images...")
    image_paths = find_images(args.data_dir)
    if not image_paths:
        print(f"ERROR: No images found in {args.data_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images")
    random.seed(args.seed)
    return random.choice(image_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual test of corruption pipeline")
    parser.add_argument("--data_dir", type=str,
                        default="/nfs/roberts/project/cpsc4520/cpsc4520_ckk25/data/train/wiki_art",
                        help="Root directory of training images (used when --image is absent)")
    parser.add_argument("--image", type=str, default=None,
                        help="Specific image path to use instead of sampling from --data_dir")
    parser.add_argument("--output_dir", type=str,
                        default="tests/corruption_results",
                        help="Output directory for grid and config")
    parser.add_argument("--corruption_config", type=str,
                        default="src/corruption/configs/default.yaml",
                        help="Path to corruption YAML config")
    parser.add_argument("--resolution", type=int, default=384,
                        help="Image resolution for each cell")
    parser.add_argument("--min_severity", type=float, default=0.01)
    parser.add_argument("--max_severity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = _resolve_source(args)

    build_grid(
        args.output_dir,
        args.corruption_config,
        source,
        args.resolution, args.seed,
        args.min_severity, args.max_severity,
    )