"""Close-inspection renders: per seed, for craquelure and scratches,
show (original | mask | corrupted) side by side at severity 0.01 and
at severity 1.0. Larger tiles so the effect is actually visible in a
preview.
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
OUT = Path("tests/corruption_results/close")
RES = 512


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


def to_pil(t):
    return T.ToPILImage()(t.clamp(0, 1))


def try_font(size=14):
    for p in ["/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


CASES = [
    ("craquelure", apply_craquelure, (0.02, 0.25), (2, 2), 'generic', {}),
    ("scratches",  apply_scratches,  (0.003, 0.05), (3, 3), 'scratches', {"max_count": 3}),
    ("rip_tear",   apply_rip_tear,   (0.003, 0.06), (1, 1), 'rip_tear', {}),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img = load_image(MONA)
    H, W = img.shape[-2:]
    font = try_font(14)
    pad = 6
    label_h = 22

    for name, fn, area, counts, shape_kind, kw in CASES:
        # 4 rows (2 seeds × 2 sev). 3 cols (orig | mask | result).
        seeds = [42, 100]
        sevs = [0.01, 1.0]
        n_rows = len(seeds) * len(sevs)
        n_cols = 3
        grid_w = n_cols * RES + (n_cols + 1) * pad
        grid_h = n_rows * (RES + label_h) + (n_rows + 1) * pad + 30
        grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
        draw = ImageDraw.Draw(grid)
        draw.text((pad, 6),
                  f"{name} on {MONA.name}  —  cols: original | mask | corrupted",
                  fill=(230, 230, 230), font=font)

        r = 0
        for seed in seeds:
            for sev in sevs:
                gen = torch.Generator(device='cpu'); gen.manual_seed(seed)
                mask, region = generate_local_mask(
                    H=H, W=W, severity=sev,
                    count_range=counts, area_frac_range=area,
                    shape_kind=shape_kind, generator=gen,
                )
                gen2 = torch.Generator(device='cpu'); gen2.manual_seed(seed + 1)
                out = fn(img, mask, generator=gen2, **kw)

                y0 = 30 + pad + r * (RES + label_h + pad)
                grid.paste(to_pil(img), (pad, y0 + label_h))
                grid.paste(
                    T.ToPILImage()(region.unsqueeze(0).expand(3, -1, -1).clamp(0, 1)),
                    (2 * pad + RES, y0 + label_h),
                )
                grid.paste(to_pil(out), (3 * pad + 2 * RES, y0 + label_h))
                draw.text((pad, y0),
                          f"seed={seed}  sev={sev}",
                          fill=(255, 220, 200), font=font)
                r += 1

        path = OUT / f"{name}.png"
        grid.save(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
