"""Visual test: generate a grid of all corruption presets on random training paintings.

Usage (on remote machine):
    python -m tests.test_corruption_visual \
        --data_dir /nfs/roberts/project/cpsc4520/cpsc4520_ckk25/data/train/wiki_art \
        --output_dir tests/corruption_results

Produces:
    tests/corruption_results/
        corruption_grid.png   — grid of all 17 presets (7 individual + 10 multi)
        config.yaml           — the CorruptionConfig used
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

# Allow running as `python -m tests.test_corruption_visual` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CorruptionConfig
from src.corruption import CorruptionModule
from src.corruption.presets import INDIVIDUAL_PRESETS, MULTI_PRESETS, CHANNEL_NAMES


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(root: str, max_images: int = 500) -> list:
    """Recursively find image files under root, return shuffled subset."""
    root = Path(root)
    images = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            images.append(p)
            if len(images) >= max_images * 10:
                break
    random.shuffle(images)
    return images[:max_images]


def load_image(path: Path, resolution: int = 384) -> torch.Tensor:
    """Load, center-crop, resize to (3, res, res) float32 [0,1]."""
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
    """Convert (3, H, W) float [0,1] tensor to PIL Image."""
    return T.ToPILImage()(t.clamp(0, 1))


def try_get_font(size: int = 18):
    """Try to load a reasonable font, fall back to default."""
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


def build_grid(
    data_dir: str,
    output_dir: str,
    resolution: int = 384,
    seed: int = 42,
):
    """Build and save the corruption test grid."""
    random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Find training images
    print(f"Scanning {data_dir} for images...")
    image_paths = find_images(data_dir)
    if not image_paths:
        print(f"ERROR: No images found in {data_dir}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images")

    # Build list of all presets to test
    presets = []
    for name in INDIVIDUAL_PRESETS:
        presets.append(("individual", name))
    for name in MULTI_PRESETS:
        presets.append(("multi", name))

    n_presets = len(presets)
    print(f"Testing {n_presets} presets ({len(INDIVIDUAL_PRESETS)} individual + {len(MULTI_PRESETS)} multi)")

    # Grid layout: each cell shows "Original → Corrupted" side by side
    # with label on top
    cell_w = resolution * 2 + 10  # original + gap + corrupted
    label_h = 28
    cell_h = resolution + label_h + 6
    padding = 8

    # Arrange in rows: 4 columns
    n_cols = 4
    n_rows = math.ceil(n_presets / n_cols)

    grid_w = n_cols * cell_w + (n_cols + 1) * padding
    grid_h = n_rows * cell_h + (n_rows + 1) * padding + 40  # +40 for title

    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)
    font = try_get_font(18)
    title_font = try_get_font(24)

    # Title
    draw.text((grid_w // 2 - 200, 8), "Corruption Pipeline Visual Test",
              fill=(255, 255, 255), font=title_font)

    config = CorruptionConfig()

    for idx, (preset_type, preset_name) in enumerate(presets):
        col = idx % n_cols
        row = idx // n_cols

        x0 = padding + col * (cell_w + padding)
        y0 = 40 + padding + row * (cell_h + padding)

        # Pick a random image for this preset
        img_path = image_paths[idx % len(image_paths)]
        try:
            img_tensor = load_image(img_path, resolution)
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")
            continue

        # Force this specific preset
        if preset_type == "individual":
            cfg = CorruptionConfig(
                individual_prob=1.0,
                individual_presets={n: (1.0 if n == preset_name else 0.0)
                                    for n in INDIVIDUAL_PRESETS},
            )
            label = f"[Individual] {preset_name}"
        else:
            cfg = CorruptionConfig(
                individual_prob=0.0,
                multi_presets={n: (1.0 if n == preset_name else 0.0)
                               for n in MULTI_PRESETS},
            )
            label = f"[Multi] {preset_name}"

        module = CorruptionModule(cfg)
        corrupted, mask = module(img_tensor, seed=seed + idx)

        # Active channels
        active = [CHANNEL_NAMES[i] for i in range(7) if mask[i].max() > 0.01]

        print(f"  {idx+1:2d}/{n_presets} {label:40s} active={active}")

        # Draw label
        draw.text((x0 + 4, y0), label, fill=(200, 255, 200) if preset_type == "individual" else (200, 200, 255),
                  font=font)

        # Paste original
        orig_pil = tensor_to_pil(img_tensor)
        grid.paste(orig_pil, (x0, y0 + label_h))

        # Paste corrupted
        corr_pil = tensor_to_pil(corrupted)
        grid.paste(corr_pil, (x0 + resolution + 10, y0 + label_h))

        # Small labels under images
        draw.text((x0 + resolution // 2 - 20, y0 + label_h + resolution - 18),
                  "Original", fill=(180, 180, 180), font=try_get_font(11))
        draw.text((x0 + resolution + 10 + resolution // 2 - 25, y0 + label_h + resolution - 18),
                  "Corrupted", fill=(180, 180, 180), font=try_get_font(11))

    # Save grid
    grid_path = os.path.join(output_dir, "corruption_grid.png")
    grid.save(grid_path, quality=95)
    print(f"\nGrid saved to {grid_path}")

    # Save config as YAML
    from omegaconf import OmegaConf
    config_path = os.path.join(output_dir, "config.yaml")
    conf_dict = {
        "corruption": OmegaConf.to_container(OmegaConf.structured(config)),
        "test_settings": {
            "resolution": resolution,
            "seed": seed,
            "data_dir": data_dir,
            "num_presets_tested": n_presets,
            "individual_presets": list(INDIVIDUAL_PRESETS.keys()),
            "multi_presets": list(MULTI_PRESETS.keys()),
        },
    }
    OmegaConf.save(OmegaConf.create(conf_dict), config_path)
    print(f"Config saved to {config_path}")

    return grid_path, config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual test of corruption pipeline")
    parser.add_argument("--data_dir", type=str,
                        default="/nfs/roberts/project/cpsc4520/cpsc4520_ckk25/data/train/wiki_art",
                        help="Root directory of training images")
    parser.add_argument("--output_dir", type=str,
                        default="tests/corruption_results",
                        help="Output directory for grid and config")
    parser.add_argument("--resolution", type=int, default=384,
                        help="Image resolution for each cell")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_grid(args.data_dir, args.output_dir, args.resolution, args.seed)
