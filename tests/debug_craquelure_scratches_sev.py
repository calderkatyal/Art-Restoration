"""Verify that craquelure and scratches are visible on Mona Lisa at
severity 0.01 and become much more prominent at severity 1.0, and that
for rip_tear / scratches the mask (and therefore the tear) gets thinner
at low severity.

Produces:
  - tests/corruption_results/sev_check/craquelure_scratches.png
  - tests/corruption_results/sev_check/rip_tear_scratches_width.png
  - Prints per-seed delta metrics:
        mean_abs_delta (image-level, on the affected ROI)
        affected_pixel_count
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corruption.effects import (
    apply_craquelure, apply_scratches, apply_rip_tear,
)
from src.corruption.presets import generate_local_mask


MONA = Path("/Users/calderkatyal/Downloads/paintings/mona_lisa.jpg")
OUT = Path("tests/corruption_results/sev_check")
SEEDS = [7, 42, 44, 100, 200]
RES = 384

CRAQ_AREA = (0.02, 0.25)
SCRATCH_AREA = (0.003, 0.05)
RIP_AREA = (0.003, 0.06)


def load_image(path: Path, resolution: int = RES) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize(
        (resolution, resolution), Image.LANCZOS
    )
    return T.ToTensor()(img)


def to_pil(t: torch.Tensor) -> Image.Image:
    return T.ToPILImage()(t.clamp(0, 1))


def measure_delta(before: torch.Tensor, after: torch.Tensor,
                  region: torch.Tensor) -> dict:
    diff = (before - after).abs().mean(0)
    aff = region > 0.5
    if aff.sum().item() < 1:
        return {"mean_delta_in_roi": 0.0, "pixels_changed": 0, "roi_pixels": 0}
    roi_diff = diff[aff]
    changed = (diff > 0.01).sum().item()
    return {
        "mean_delta_in_roi": float(roi_diff.mean().item()),
        "pixels_changed": int(changed),
        "roi_pixels": int(aff.sum().item()),
    }


def render_effect(img, fn, mask_kwargs, apply_kwargs, gen_seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(gen_seed)
    soft, region = generate_local_mask(**mask_kwargs, generator=gen)
    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(gen_seed + 1)
    out = fn(img, soft, generator=gen2, **apply_kwargs)
    return out, region


def try_font(size=12):
    for path in ["/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def grid_craquelure_scratches(img):
    """3 rows (craquelure, scratches, rip_tear) × (orig | sev0.01 × N seeds |
    sev1.0 × N seeds)."""
    H, W = img.shape[-2:]
    font = try_font(11)

    cell = RES
    pad = 4
    label_h = 18
    n_seeds = len(SEEDS)
    n_cols = 1 + 2 * n_seeds   # original + lo*N + hi*N
    rows = [
        ("craquelure", apply_craquelure, CRAQ_AREA, (2, 2), {}),
        ("scratches",  apply_scratches,  SCRATCH_AREA, (3, 3), {"max_count": 3}),
        ("rip_tear",   apply_rip_tear,   RIP_AREA, (1, 1), {}),
    ]
    n_rows = len(rows)
    grid_w = n_cols * cell + (n_cols + 1) * pad
    grid_h = n_rows * (cell + label_h) + (n_rows + 1) * pad + 30
    grid = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    draw = ImageDraw.Draw(grid)
    draw.text((pad, 6),
              f"{MONA.name} — Rows: craquelure | scratches | rip_tear. "
              f"Cols: original + {n_seeds} seeds @ sev=0.01 + {n_seeds} seeds @ sev=1.0",
              fill=(220, 220, 220), font=font)

    print(f"\n=== craquelure / scratches / rip_tear on {MONA.name} ===")
    for ri, (name, fn, area, count_range, kw) in enumerate(rows):
        y0 = 30 + pad + ri * (cell + label_h + pad)
        # Original column.
        grid.paste(to_pil(img), (pad, y0 + label_h))
        draw.text((pad + 4, y0), f"{name}: original",
                  fill=(180, 220, 180), font=font)

        shape_kind = 'generic'
        if name == 'scratches': shape_kind = 'scratches'
        if name == 'rip_tear':  shape_kind = 'rip_tear'

        for si, sev in enumerate([0.01, 1.0]):
            for seed_i, seed in enumerate(SEEDS):
                col = 1 + si * n_seeds + seed_i
                x = pad + col * (cell + pad)
                mask_kwargs = dict(
                    H=H, W=W, severity=sev,
                    count_range=count_range,
                    area_frac_range=area,
                    shape_kind=shape_kind,
                )
                out, region = render_effect(img, fn, mask_kwargs, kw, seed)
                grid.paste(to_pil(out), (x, y0 + label_h))
                draw.text((x + 2, y0),
                          f"{name} s={sev} seed={seed}",
                          fill=(200, 220, 255) if sev < 1 else (255, 180, 160),
                          font=font)
                metrics = measure_delta(img, out, region)
                draw.text((x + 2, y0 + label_h + cell - 14),
                          f"Δ={metrics['mean_delta_in_roi']:.3f} "
                          f"roi={metrics['roi_pixels']} chg={metrics['pixels_changed']}",
                          fill=(220, 220, 220), font=font)
                print(f"  {name:<11s} sev={sev:<4} seed={seed:>3d}  "
                      f"roi={metrics['roi_pixels']:>6d}  "
                      f"mean_delta_in_roi={metrics['mean_delta_in_roi']:.4f}  "
                      f"changed={metrics['pixels_changed']}")
    return grid


def grid_rip_tear_widths(img):
    """rip_tear mask and effect at multiple severities to inspect width scaling."""
    H, W = img.shape[-2:]
    font = try_font(11)
    sevs = [0.01, 0.25, 0.5, 0.75, 1.0]
    cell = RES
    pad = 4
    label_h = 18
    n_cols = len(sevs)
    n_rows = 2  # row 0: mask, row 1: output
    grid_w = n_cols * cell + (n_cols + 1) * pad
    grid_h = n_rows * (cell + label_h) + (n_rows + 1) * pad + 30
    grid = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    draw = ImageDraw.Draw(grid)
    draw.text((pad, 6),
              "rip_tear: mask (row 1) and output (row 2) as severity varies — "
              "both should get visibly thicker from left to right.",
              fill=(220, 220, 220), font=font)

    print("\n=== rip_tear width scaling with severity (seed=42) ===")
    for ci, sev in enumerate(sevs):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)
        mask, region = generate_local_mask(
            H=H, W=W, severity=sev,
            count_range=(1, 1),
            area_frac_range=RIP_AREA,
            shape_kind='rip_tear',
            generator=gen,
        )
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(43)
        out = apply_rip_tear(img, mask, generator=gen2)
        roi_px = int(region.sum().item())
        x = pad + ci * (cell + pad)
        grid.paste(
            T.ToPILImage()(region.unsqueeze(0).expand(3, -1, -1).clamp(0, 1)),
            (x, 30 + pad + label_h),
        )
        grid.paste(to_pil(out),
                   (x, 30 + pad + 2 * label_h + cell + pad))
        draw.text((x + 2, 30 + pad),
                  f"sev={sev}", fill=(220, 220, 220), font=font)
        draw.text((x + 2, 30 + pad + label_h + cell - 14),
                  f"roi={roi_px}",
                  fill=(200, 220, 255), font=font)
        draw.text((x + 2, 30 + pad + label_h + cell + pad),
                  f"output sev={sev}", fill=(255, 200, 160), font=font)
        print(f"  sev={sev:<5}  roi={roi_px:>6d}")
    return grid


def grid_scratches_widths(img):
    H, W = img.shape[-2:]
    font = try_font(11)
    sevs = [0.01, 0.25, 0.5, 0.75, 1.0]
    cell = RES
    pad = 4
    label_h = 18
    n_cols = len(sevs)
    n_rows = 2
    grid_w = n_cols * cell + (n_cols + 1) * pad
    grid_h = n_rows * (cell + label_h) + (n_rows + 1) * pad + 30
    grid = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    draw = ImageDraw.Draw(grid)
    draw.text((pad, 6),
              "scratches: mask (row 1) and output (row 2) as severity varies — "
              "mask bands narrow at low sev; scratch marks become faint.",
              fill=(220, 220, 220), font=font)

    print("\n=== scratches width scaling with severity (seed=42) ===")
    for ci, sev in enumerate(sevs):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)
        mask, region = generate_local_mask(
            H=H, W=W, severity=sev,
            count_range=(3, 3),
            area_frac_range=SCRATCH_AREA,
            shape_kind='scratches',
            generator=gen,
        )
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(43)
        out = apply_scratches(img, mask, generator=gen2, max_count=3)
        roi_px = int(region.sum().item())
        x = pad + ci * (cell + pad)
        grid.paste(
            T.ToPILImage()(region.unsqueeze(0).expand(3, -1, -1).clamp(0, 1)),
            (x, 30 + pad + label_h),
        )
        grid.paste(to_pil(out),
                   (x, 30 + pad + 2 * label_h + cell + pad))
        draw.text((x + 2, 30 + pad),
                  f"sev={sev}", fill=(220, 220, 220), font=font)
        draw.text((x + 2, 30 + pad + label_h + cell - 14),
                  f"roi={roi_px}",
                  fill=(200, 220, 255), font=font)
        draw.text((x + 2, 30 + pad + label_h + cell + pad),
                  f"output sev={sev}", fill=(255, 200, 160), font=font)
        print(f"  sev={sev:<5}  roi={roi_px:>6d}")
    return grid


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img = load_image(MONA)

    g1 = grid_craquelure_scratches(img)
    g1_path = OUT / "craquelure_scratches.png"
    g1.save(g1_path, quality=95)
    print(f"\nSaved {g1_path}")

    g2 = grid_rip_tear_widths(img)
    g2_path = OUT / "rip_tear_widths.png"
    g2.save(g2_path, quality=95)
    print(f"Saved {g2_path}")

    g3 = grid_scratches_widths(img)
    g3_path = OUT / "scratches_widths.png"
    g3.save(g3_path, quality=95)
    print(f"Saved {g3_path}")


if __name__ == "__main__":
    main()
