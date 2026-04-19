"""Isolated debug harness for apply_rip_tear.

Renders several seeds at both min and max severity on a single painting
and measures per-pixel darkening to verify the tear is actually painting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corruption.effects import apply_rip_tear
from src.corruption.presets import generate_local_mask


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


def measure(before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor):
    """Return dict of diagnostics."""
    diff = (before - after).abs().mean(0)
    affected = (diff > 0.01).float()
    n_aff = int(affected.sum().item())
    return {
        "mean_diff": diff.mean().item(),
        "max_diff": diff.max().item(),
        "pixels_affected": n_aff,
        "frac_affected": n_aff / (diff.numel()),
        "mask_max": mask.max().item(),
        "mask_sum": mask.sum().item(),
    }


def main(image_path=None, suffix=""):
    resolution = 384
    if image_path is None:
        image_path = Path("/Users/calderkatyal/Downloads/paintings/persistence_of_memory.jpg")
    out_dir = Path("tests/corruption_results/rip_tear_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path, resolution)
    H, W = image.shape[-2:]

    # Mask params matching default.yaml rip_tear config.
    num_blobs_range = (1, 3)
    radius_frac_range = (0.05, 0.45)

    # Seed 44 is what test_corruption_visual uses for rip_tear (seed=42, idx=2).
    seeds = [44, 42, 100, 200, 300]
    severities = [0.01, 1.0]

    rows = []
    print("seed  severity  mean_diff  max_diff  pixels_affected  frac  mask_max  mask_sum")
    for seed in seeds:
        row = []
        for sev in severities:
            # Match the test harness: one generator used for both mask and effect.
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            mask, tear_bin = generate_local_mask(
                H, W, sev, num_blobs_range, radius_frac_range,
                generator=gen, device=image.device,
            )
            out = apply_rip_tear(image, mask, generator=gen)

            m = measure(image, out, mask)
            print(f"{seed:4d}  {sev:7.2f}  {m['mean_diff']:.4f}  "
                  f"{m['max_diff']:.4f}  {m['pixels_affected']:6d}  "
                  f"{m['frac_affected']:.4f}  {m['mask_max']:.3f}  {m['mask_sum']:.1f}")
            row.append((sev, out, mask, tear_bin, m))
        rows.append((seed, row))

    # Build a grid: rows = seeds; columns = [original, mask@sev_min, out@sev_min, mask@sev_max, out@sev_max]
    n_cols = 5
    n_rows = len(seeds)
    cell = resolution
    pad = 8
    label_h = 24
    grid_w = n_cols * cell + (n_cols + 1) * pad
    grid_h = n_rows * (cell + label_h) + (n_rows + 1) * pad + 40

    grid = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    draw.text((pad, 8), "rip_tear debug — cols: orig | mask@0.01 | out@0.01 | mask@1.0 | out@1.0",
              fill=(240, 240, 240), font=font)

    for ri, (seed, row) in enumerate(rows):
        y0 = 40 + pad + ri * (cell + label_h + pad)

        # col 0: original
        x0 = pad
        grid.paste(tensor_to_pil(image), (x0, y0 + label_h))
        draw.text((x0, y0), f"seed={seed}", fill=(220, 220, 220), font=font)

        # cols 1-4
        for ci, (sev, out, mask, tear_bin, m) in enumerate(row):
            mask_vis = tensor_to_pil(mask.unsqueeze(0).expand(3, -1, -1))
            out_vis = tensor_to_pil(out)

            xm = pad + (ci * 2 + 1) * (cell + pad)
            xo = pad + (ci * 2 + 2) * (cell + pad)
            grid.paste(mask_vis, (xm, y0 + label_h))
            grid.paste(out_vis, (xo, y0 + label_h))

            draw.text((xm, y0), f"mask sev={sev} (max={m['mask_max']:.2f})",
                      fill=(180, 220, 180), font=font)
            draw.text((xo, y0),
                      f"out: Δmax={m['max_diff']:.2f} n={m['pixels_affected']}",
                      fill=(220, 200, 180), font=font)

    out_path = out_dir / f"grid{suffix}.png"
    grid.save(out_path, quality=95)
    print(f"\nGrid saved to {out_path}")


if __name__ == "__main__":
    print("=== persistence_of_memory.jpg ===")
    main(Path("/Users/calderkatyal/Downloads/paintings/persistence_of_memory.jpg"),
         suffix="_persistence")
    print("\n=== mona_lisa.jpg ===")
    main(Path("/Users/calderkatyal/Downloads/paintings/mona_lisa.jpg"),
         suffix="_monalisa")
