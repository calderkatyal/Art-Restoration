"""Render rip_tear and base at multiple severities / seeds as individual
large images for close inspection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corruption.effects import apply_rip_tear
from src.corruption.presets import generate_local_mask


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


def tensor_to_pil(t):
    return T.ToPILImage()(t.clamp(0, 1))


def run(name, fn, nb, rf, image_path, out_root, seeds, severities, res=512):
    img = load_image(image_path, res)
    H, W = img.shape[-2:]
    stem = image_path.stem
    base_dir = out_root / name / stem
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save original for reference
    tensor_to_pil(img).save(base_dir / "_original.png")

    for seed in seeds:
        for sev in severities:
            gen = torch.Generator()
            gen.manual_seed(seed)
            mask, _ = generate_local_mask(H, W, sev, nb, rf, generator=gen)
            out = fn(img, mask, generator=gen)
            if isinstance(out, tuple):
                out = out[0]
            tensor_to_pil(out).save(base_dir / f"seed{seed:03d}_sev{sev:.2f}.png")
    print(f"[{name}] {stem}: saved {len(seeds)*len(severities)} images to {base_dir}")


def main():
    paintings = [
        Path("/Users/calderkatyal/Downloads/paintings/persistence_of_memory.jpg"),
        Path("/Users/calderkatyal/Downloads/paintings/mona_lisa.jpg"),
    ]
    out_root = Path("tests/corruption_results/individual")
    seeds = [44, 42, 100, 200, 7, 313]
    severities = [0.01, 0.3, 1.0]

    for p in paintings:
        run('rip_tear', apply_rip_tear, (1, 3), (0.05, 0.45), p, out_root, seeds, severities)


if __name__ == "__main__":
    main()
