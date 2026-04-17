"""Render scratches at multiple severities — confirm reduced visibility at low sev."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corruption.effects import apply_scratches
from src.corruption.presets import generate_local_mask


def count_scratch_pixels(orig: torch.Tensor, out: torch.Tensor, eps: float = 0.02) -> int:
    diff = (orig - out).abs().max(dim=0).values
    return int((diff > eps).sum().item())


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


def run(image_path, out_root, seeds, severities, res=512,
        local_min_num=1, local_max_num=8):
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
            mask, _ = generate_local_mask(
                H, W, sev,
                (local_min_num, local_max_num),
                (0.06, 0.45),
                generator=gen,
            )
            out = apply_scratches(img, mask, generator=gen, max_count=local_max_num)
            n_pix = count_scratch_pixels(img, out)
            T.ToPILImage()(out.clamp(0, 1)).save(
                base_dir / f"seed{seed:03d}_sev{sev:.2f}_max{local_max_num}.jpg", quality=92
            )
            print(f"  seed={seed:3d} sev={sev:.2f} max={local_max_num:3d}: {n_pix:6d} scratch px")
    print(f"{stem}: done → {base_dir}")


def main():
    paintings = [
        Path("/Users/calderkatyal/Downloads/paintings/mona_lisa.jpg"),
    ]
    out_root = Path("tests/corruption_results/scratches_only")
    # 15 distinct seeds for visibility/realism stress test.
    seeds = [7, 13, 17, 21, 33, 44, 67, 88, 100, 102, 103, 150, 200, 271, 313]
    severities = [0.01, 1.0]
    for p in paintings:
        run(p, out_root, seeds, severities, local_max_num=8)
        run(p, out_root, seeds, severities, local_max_num=1)
        run(p, out_root, seeds, severities, local_max_num=50)


if __name__ == "__main__":
    main()
