"""Closeup v4 hull verification on v2 effects.

Layout: (orig | corrupted | hull overlay) per channel, per mode (local
and/or global where enabled), 512px tiles. One PNG per (channel, mode)
for each test painting.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corruption.module import CorruptionModule
from src.corruption.presets import CHANNEL_NAMES


PAINTINGS = [
    Path("/Users/calderkatyal/Downloads/paintings/mona_lisa.jpg"),
    Path("/Users/calderkatyal/Downloads/paintings/persistence_of_memory.jpg"),
]
OUT = Path("tests/corruption_results/v4_closeup")
RES = 512
CFG_PATH = Path("src/corruption/configs/default.yaml")


def load_image(path: Path, resolution: int = RES) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2,
                    (w - s) // 2 + s, (h - s) // 2 + s)).resize(
        (resolution, resolution), Image.LANCZOS
    )
    return T.ToTensor()(img)


def to_pil(t):
    return T.ToPILImage()(t.clamp(0, 1))


def try_font(size=16):
    for p in ["/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def overlay(img, mask, color=(255, 60, 60), alpha=0.45):
    m = mask.clamp(0, 1)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    r = torch.full_like(m[0], color[0] / 255.0)
    g = torch.full_like(m[0], color[1] / 255.0)
    b = torch.full_like(m[0], color[2] / 255.0)
    col = torch.stack([r, g, b], dim=0)
    a = m * alpha
    return img * (1 - a) + col * a


def forced_config(base_cfg, channel: str, mode: str, severity: float):
    """Return a cfg forcing exactly one channel, one mode, fixed severity."""
    cfg = deepcopy(base_cfg)
    cfg.num_simultaneous = {'min': 1, 'max': 1}
    for name in CHANNEL_NAMES:
        t = cfg.types[name]
        t.weight = 1.0 if name == channel else 0.0
        if name == channel:
            if mode == 'local':
                t.local_enabled = True
                t.global_enabled = False
            else:
                t.local_enabled = False
                t.global_enabled = True
            t.min_severity = severity
            t.max_severity = severity
    return cfg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = try_font(16)
    pad = 6
    label_h = 22
    base = OmegaConf.load(CFG_PATH)

    # Fixed severity for a fair look. 0.7 is moderate/heavy.
    severity = 0.7

    for painting in PAINTINGS:
        img = load_image(painting)
        H, W = img.shape[-2:]
        stem = painting.stem
        print(f"\n=== {stem} ===")

        for channel in CHANNEL_NAMES:
            modes = []
            tcfg = base.types[channel]
            if bool(tcfg.get('local_enabled', False)):
                modes.append('local')
            if bool(tcfg.get('global_enabled', False)):
                modes.append('global')

            for mode in modes:
                cfg = forced_config(base, channel, mode, severity)
                mod = CorruptionModule(cfg)
                corr, masks = mod(img, seed=42)

                ci = CHANNEL_NAMES.index(channel)
                hull_px = int(masks[ci].sum().item())
                active = [(i, CHANNEL_NAMES[i])
                          for i in range(len(CHANNEL_NAMES))
                          if masks[i].sum().item() > 0]
                print(f"  [{mode}] {channel}: hull={hull_px} px, "
                      f"other active channels={[c[1] for c in active if c[0] != ci]}")

                # 3-column grid: original | corrupted | hull overlay
                tile = RES
                n_cols = 3
                grid_w = n_cols * tile + (n_cols + 1) * pad
                grid_h = tile + label_h * 2 + 2 * pad
                grid = Image.new('RGB', (grid_w, grid_h), (20, 20, 20))
                draw = ImageDraw.Draw(grid)
                draw.text((pad, 4),
                          f"[{mode}] {channel} sev={severity}  "
                          f"on {stem}  hull={hull_px}px",
                          fill=(230, 230, 230), font=font)
                y0 = label_h + pad
                grid.paste(to_pil(img), (pad, y0))
                draw.text((pad + 4, y0 + tile + 2),
                          "original", fill=(200, 200, 200), font=font)
                grid.paste(to_pil(corr), (2 * pad + tile, y0))
                draw.text((2 * pad + tile + 4, y0 + tile + 2),
                          "corrupted", fill=(200, 200, 200), font=font)
                ov = overlay(corr, masks[ci])
                grid.paste(to_pil(ov), (3 * pad + 2 * tile, y0))
                draw.text((3 * pad + 2 * tile + 4, y0 + tile + 2),
                          f"hull: {channel}",
                          fill=(255, 180, 180), font=font)

                grid.save(OUT / f"{stem}__{channel}_{mode}.png")


if __name__ == "__main__":
    main()
