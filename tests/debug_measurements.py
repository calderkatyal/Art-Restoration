"""Quick per-effect delta measurements on both test paintings."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corruption.effects import (
    apply_craquelure, apply_rip_tear, apply_paint_loss,
    apply_yellowing, apply_fading, apply_bloom, apply_deposits,
    apply_scratches,
)
from src.corruption.presets import generate_global_mask, generate_local_mask


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


FNS = {
    'craquelure': apply_craquelure,
    'rip_tear':   apply_rip_tear,
    'paint_loss': apply_paint_loss,
    'yellowing':  apply_yellowing,
    'fading':     apply_fading,
    'bloom':      apply_bloom,
    'deposits':   apply_deposits,
    'scratches':  apply_scratches,
}

LOCAL_PARAMS = {
    'craquelure': ((3, 8), (0.05, 0.25)),
    'rip_tear':   ((1, 3), (0.05, 0.45)),
    'paint_loss': ((3, 8), (0.03, 0.14)),
    'yellowing':  ((2, 5), (0.10, 0.30)),
    'fading':     ((2, 5), (0.10, 0.35)),
    'bloom':      ((2, 5), (0.06, 0.24)),
    'deposits':   ((3, 7), (0.05, 0.20)),
    'scratches':  ((2, 5), (0.06, 0.45)),
}

GLOBAL_OK = {'craquelure', 'yellowing', 'fading', 'bloom', 'deposits'}


def measure(before, after):
    diff = (before - after).abs().mean(0)
    aff = (diff > 0.01).float()
    return diff.mean().item(), diff.max().item(), int(aff.sum().item())


def run(img_path: Path, label: str):
    print(f"\n=== {label}: {img_path.name} ===")
    img = load_image(img_path)
    H, W = img.shape[-2:]
    seed = 42

    for name in FNS:
        fn = FNS[name]

        for mode in ['local', 'global']:
            if mode == 'global' and name not in GLOBAL_OK:
                continue
            for sev in [0.01, 1.0]:
                gen_m = torch.Generator()
                gen_m.manual_seed(seed)
                if mode == 'local':
                    nb, rf = LOCAL_PARAMS[name]
                    mask, _ = generate_local_mask(H, W, sev, nb, rf, generator=gen_m)
                else:
                    mask, _ = generate_global_mask(H, W, sev)
                gen_e = torch.Generator()
                gen_e.manual_seed(seed + 1)
                out = fn(img, mask, generator=gen_e)
                if isinstance(out, tuple):
                    out = out[0]
                mean_d, max_d, aff = measure(img, out)
                print(f"  [{mode:<6s}] {name:<12s} s={sev:.2f}  "
                      f"mean={mean_d:.4f}  max={max_d:.3f}  aff={aff:6d}")


if __name__ == "__main__":
    run(Path("/Users/calderkatyal/Downloads/paintings/persistence_of_memory.jpg"),
        "Light (Persistence)")
    run(Path("/Users/calderkatyal/Downloads/paintings/mona_lisa.jpg"),
        "Dark (Mona Lisa)")
