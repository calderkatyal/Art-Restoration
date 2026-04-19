"""Render bloom at multiple severities — verify ~25% extra punch at sev=1.0."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corruption.effects import apply_bloom
from src.corruption.presets import generate_local_mask, generate_global_mask


def load_image(path, resolution=512):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize(
        (resolution, resolution), Image.LANCZOS
    )
    return T.ToTensor()(img)


def run(image_path, out_root, seeds, severities, res=512):
    img = load_image(image_path, res)
    H, W = img.shape[-2:]
    stem = image_path.stem
    base_dir = out_root / stem
    base_dir.mkdir(parents=True, exist_ok=True)
    T.ToPILImage()(img.clamp(0, 1)).save(base_dir / "_original.jpg", quality=92)

    for seed in seeds:
        for sev in severities:
            gen = torch.Generator()
            gen.manual_seed(seed)
            local_mask, _ = generate_local_mask(H, W, sev, (1, 5), (0.06, 0.24), generator=gen)
            out_local = apply_bloom(img, local_mask, generator=gen)
            T.ToPILImage()(out_local.clamp(0, 1)).save(
                base_dir / f"seed{seed:03d}_sev{sev:.2f}_local.jpg", quality=92
            )

            global_mask, _ = generate_global_mask(H, W, sev)
            out_global = apply_bloom(img, global_mask, generator=gen)
            T.ToPILImage()(out_global.clamp(0, 1)).save(
                base_dir / f"seed{seed:03d}_sev{sev:.2f}_global.jpg", quality=92
            )
    print(f"{stem}: done → {base_dir}")


def main():
    paintings = [
        Path("/Users/calderkatyal/Downloads/paintings/mona_lisa.jpg"),
    ]
    out_root = Path("tests/corruption_results/bloom_only")
    seeds = [44, 102]
    severities = [0.01, 0.5, 1.0]
    for p in paintings:
        run(p, out_root, seeds, severities)


if __name__ == "__main__":
    main()
