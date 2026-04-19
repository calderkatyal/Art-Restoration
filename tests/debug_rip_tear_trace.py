"""Trace what apply_rip_tear does on seed 44."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import torch
import torchvision.transforms as T
from PIL import Image

from src.corruption.presets import generate_local_mask
from src.corruption.effects import _sample_canvas_color, _local_lum_boost


def load_image(path, resolution=384):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize(
        (resolution, resolution), Image.LANCZOS
    )
    return T.ToTensor()(img)


def trace_rip(image, mask, gen, label):
    C, H, W = image.shape
    active_mask = mask >= 0.02
    ys_a, xs_a = active_mask.nonzero(as_tuple=True)
    y_lo, y_hi = int(ys_a.min()), int(ys_a.max())
    x_lo, x_hi = int(xs_a.min()), int(xs_a.max())
    cx_c = (x_lo + x_hi) * 0.5
    cy_c = (y_lo + y_hi) * 0.5
    diag = ((y_hi - y_lo) ** 2 + (x_hi - x_lo) ** 2) ** 0.5
    print(f"[{label}] bbox=({x_lo}..{x_hi}, {y_lo}..{y_hi}) diag={diag:.1f} "
          f"mask_sum={mask.sum().item():.1f}")

    num_tears = 1 + int(torch.randint(0, 2, (1,), generator=gen).item())
    max_steps = max(60, int(diag * 2.5))
    print(f"  num_tears={num_tears} max_steps={max_steps}")

    for t in range(num_tears):
        side = int(torch.randint(0, 4, (1,), generator=gen).item())
        r1 = torch.rand(1, generator=gen).item()
        if side == 0:
            sx = x_lo + r1 * max(1, x_hi - x_lo); sy = float(y_lo)
        elif side == 1:
            sx = x_lo + r1 * max(1, x_hi - x_lo); sy = float(y_hi)
        elif side == 2:
            sx = float(x_lo); sy = y_lo + r1 * max(1, y_hi - y_lo)
        else:
            sx = float(x_hi); sy = y_lo + r1 * max(1, y_hi - y_lo)

        base = math.atan2(cy_c - sy, cx_c - sx)
        r2 = torch.rand(1, generator=gen).item()
        base += (r2 - 0.5) * (math.pi * 0.5)
        angle = base
        width = 4 + int(torch.rand(1, generator=gen).item() * 4)
        core_w = 1 + int(torch.rand(1, generator=gen).item() * 1.5)
        lip_side = 1 if torch.rand(1, generator=gen).item() < 0.5 else -1

        print(f"  tear {t}: start=({sx:.1f}, {sy:.1f}) side={side} "
              f"angle={angle:.2f} width={width} core_w={core_w} lip={lip_side}")

        px, py = sx, sy
        painted = 0
        in_mask = 0
        out_bounds = 0
        for step in range(max_steps):
            angle += (torch.rand(1, generator=gen).item() - 0.5) * 0.12
            px += math.cos(angle) * 1.6
            py += math.sin(angle) * 1.6
            ix, iy = int(round(px)), int(round(py))
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                out_bounds += 1
                break
            m_val = mask[iy, ix].item()
            if m_val >= 0.02:
                in_mask += 1
                painted += 1
        print(f"    painted={painted} in_mask={in_mask} out_bounds={out_bounds} "
              f"(final pos {px:.1f}, {py:.1f})")


def main():
    img = load_image(Path("/Users/calderkatyal/Downloads/paintings/persistence_of_memory.jpg"))
    for sev in [0.01, 1.0]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(44)
        mask, _ = generate_local_mask(
            img.shape[-2], img.shape[-1], sev, (1, 3), (0.05, 0.45),
            generator=gen, device=img.device,
        )
        trace_rip(img, mask, gen, f"sev={sev}")


main()
