"""Generate ~100 sample corruptions per degradation type for manual inspection.

For each corruption type, samples N paintings from --data_dir, applies the
effect at random severities, and saves individual images + a contact sheet.

Output layout:
    <output_dir>/
        <type>/
            <painting>_<mode>_sev<S>_seed<N>.jpg   (individual images)
        contact_sheets/
            <type>_sheet.jpg                         (10×10 overview grid)

Usage:
    python tests/generate_degradation_samples.py \\
        --data_dir ~/Downloads/paintings \\
        --output_dir tests/degradation_samples \\
        --n_paintings 5 \\
        --samples_per_type 100
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import List

import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corruption.effects import (
    apply_craquelure, apply_rip_tear, apply_paint_loss,
    apply_yellowing, apply_fading, apply_deposits, apply_scratches,
)
from src.corruption.presets import (
    CHANNEL_NAMES, generate_global_mask, generate_local_mask,
    SHAPE_KIND_BY_CHANNEL,
)
from src.corruption.module import _affected_pixels, _per_component_hull_mask

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

EFFECT_FNS = {
    "craquelure": apply_craquelure,
    "rip_tear":   apply_rip_tear,
    "paint_loss": apply_paint_loss,
    "yellowing":  apply_yellowing,
    "fading":     apply_fading,
    "deposits":   apply_deposits,
    "scratches":  apply_scratches,
}


def find_images(root: str, max_scan: int = 5000) -> List[Path]:
    root = Path(root)
    images: List[Path] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            images.append(p)
            if len(images) >= max_scan:
                break
    return images


def load_image(path: Path, resolution: int = 512) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((resolution, resolution), Image.LANCZOS)
    return T.ToTensor()(img)


def image_luminance(path: Path, thumb: int = 64) -> float:
    """Fast mean luminance of a painting (0=black, 1=white)."""
    try:
        img = Image.open(path).convert("RGB").resize((thumb, thumb), Image.BILINEAR)
        t = T.ToTensor()(img)
        return float((t[0] * 0.299 + t[1] * 0.587 + t[2] * 0.114).mean().item())
    except Exception:
        return 0.5


def stratified_sample(paths: List[Path], n: int, seed: int) -> List[Path]:
    """Pick n paintings spread across dark / mid / bright luminance bands.

    Guarantees roughly 1/3 of samples come from each band so dark paintings
    (where crack / tear visibility is hardest) are always represented.
    """
    rng = random.Random(seed)
    scored = sorted(paths, key=image_luminance)
    third = max(1, len(scored) // 3)
    dark   = scored[:third]
    mid    = scored[third: 2 * third]
    bright = scored[2 * third:]

    n_dark   = max(1, n // 3)
    n_bright = max(1, n // 3)
    n_mid    = max(1, n - n_dark - n_bright)

    def pick(bucket, k):
        rng.shuffle(bucket)
        return bucket[:k]

    selected = pick(dark, n_dark) + pick(mid, n_mid) + pick(bright, n_bright)
    rng.shuffle(selected)
    return selected[:n]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    return T.ToPILImage()(t.clamp(0, 1))


def try_get_font(size: int = 14):
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def overlay_red_mask(
    image: Image.Image,
    actual_mask: torch.Tensor,
    soft_mask: torch.Tensor,
    width: int = 2,
) -> Image.Image:
    """Draw a thin red boundary using actual_mask; fall back to soft_mask if empty."""
    candidate = actual_mask.detach().cpu()
    if candidate.max().item() < 0.05:
        # actual_mask empty (e.g. subtle craquelure) — use soft_mask instead
        candidate = soft_mask.detach().cpu()

    m = candidate > 0.05
    if not m.any():
        return image.copy()

    up    = torch.zeros_like(m); up[1:]      = m[:-1]
    down  = torch.zeros_like(m); down[:-1]   = m[1:]
    left  = torch.zeros_like(m); left[:, 1:] = m[:, :-1]
    right = torch.zeros_like(m); right[:, :-1] = m[:, 1:]
    boundary = m & ~(m & up & down & left & right)

    out = image.copy()
    draw = ImageDraw.Draw(out)
    ys, xs = torch.nonzero(boundary, as_tuple=True)
    if width <= 1:
        for y, x in zip(ys.tolist(), xs.tolist()):
            draw.point((x, y), fill=(255, 0, 0))
    else:
        r = width // 2
        for y, x in zip(ys.tolist(), xs.tolist()):
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
    return out


def count_instances(mask: torch.Tensor, threshold: float = 0.02) -> int:
    """Count connected components (degradation instances) in a mask."""
    from scipy.ndimage import label as _cc_label
    binary = (mask >= threshold).cpu().numpy()
    _, n = _cc_label(binary)
    return int(n)


def apply_corruption(
    image: torch.Tensor,
    corruption: str,
    mode: str,
    severity: float,
    type_cfg,
    generator: torch.Generator,
) -> tuple:
    """Returns (corrupted_image, soft_mask, actual_mask, num_instances)."""
    H, W = image.shape[-2:]
    device = image.device

    if mode == "local":
        min_n = max(1, int(type_cfg.get("local_min_num", 1)))
        max_n = max(min_n, int(type_cfg.get("local_max_num", 4)))
        af = type_cfg.get("local_area_frac", [0.01, 0.05])
        shape_kind = SHAPE_KIND_BY_CHANNEL.get(corruption, "generic")
        soft_mask, _ = generate_local_mask(
            H, W, severity,
            (min_n, max_n),
            (float(af[0]), float(af[1])),
            shape_kind=shape_kind,
            generator=generator,
            device=device,
        )
    else:
        soft_mask, _ = generate_global_mask(H, W, severity, device=device)

    fn = EFFECT_FNS[corruption]
    extra = {}
    if corruption == "scratches":
        extra["max_count"] = int(type_cfg.get("local_max_num", 8))
    before = image.clone()
    out_img = fn(image, soft_mask, generator=generator, **extra)
    # Show the actual affected-pixel hull mask (same as what the training
    # pipeline produces), not the input soft mask. This way the displayed
    # mask matches exactly where the damage was drawn.
    diff_bool = _affected_pixels(before, out_img, threshold=0.008)
    H2, W2 = image.shape[-2:]
    actual_mask = _per_component_hull_mask(diff_bool, H2, W2, merge_radius=3)
    n_instances = count_instances(actual_mask)
    return out_img, soft_mask, actual_mask, n_instances


def mask_to_pil(m: torch.Tensor, resolution: int) -> Image.Image:
    """Render a (H, W) mask as a grayscale image, normalized to full range."""
    g = m.detach().cpu().clamp(0, 1)
    mx = g.max()
    if mx > 0.01:
        g = g / mx  # normalize so the mask structure is clearly visible
    pil = T.ToPILImage()(g.unsqueeze(0))
    return pil.resize((resolution, resolution), Image.LANCZOS)


def make_contact_sheet(
    images: List[Image.Image],
    masks: List[Image.Image],
    labels: List[str],
    cols: int = 6,
    thumb: int = 200,
    label_h: int = 22,
    title: str = "",
) -> Image.Image:
    """Each cell is [corrupted | mask] side by side."""
    pair_w = thumb * 2 + 2  # 2px gap between image and mask
    rows = math.ceil(len(images) / cols)
    gap = 4
    title_pad = 36 if title else 0
    W = cols * (pair_w + gap) + gap
    H = rows * (thumb + label_h + gap) + gap + title_pad
    sheet = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    font = try_get_font(11)
    title_font = try_get_font(16)

    if title:
        draw.text((gap, 8), title, fill=(230, 230, 230), font=title_font)

    for i, (img, msk, label) in enumerate(zip(images, masks, labels)):
        col = i % cols
        row = i // cols
        x = gap + col * (pair_w + gap)
        y = title_pad + gap + row * (thumb + label_h + gap)
        thumb_img = img.resize((thumb, thumb), Image.LANCZOS)
        thumb_msk = msk.resize((thumb, thumb), Image.LANCZOS)
        sheet.paste(thumb_img, (x, y))
        sheet.paste(thumb_msk, (x + thumb + 2, y))
        draw.text((x + 2, y + thumb + 2), label, fill=(200, 200, 200), font=font)

    return sheet


def main():
    parser = argparse.ArgumentParser(description="Generate degradation samples for review")
    parser.add_argument("--data_dir", required=True,
                        help="Root directory containing painting images")
    parser.add_argument("--output_dir", default="tests/degradation_samples")
    parser.add_argument("--corruption_config", default="src/corruption/configs/default.yaml")
    parser.add_argument("--n_paintings", type=int, default=5,
                        help="Number of paintings to use (picked at random)")
    parser.add_argument("--samples_per_type", type=int, default=36,
                        help="Total images to generate per corruption type")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config.
    cfg = OmegaConf.load(args.corruption_config)

    # Find and sample paintings.
    print(f"Scanning {args.data_dir} for images...")
    all_images = find_images(args.data_dir)
    if not all_images:
        print(f"ERROR: No images found in {args.data_dir}")
        sys.exit(1)
    print(f"Scoring luminance for {min(len(all_images), 5000)} candidates...")
    painting_paths = stratified_sample(all_images, args.n_paintings, args.seed)
    print(f"Using {len(painting_paths)} painting(s) (dark/mid/bright stratified):")
    for p in painting_paths:
        lum = image_luminance(p)
        print(f"  {p.name}  lum={lum:.2f}")

    print(f"\nLoading images at {args.resolution}px...")
    paintings = []
    for p in painting_paths:
        try:
            paintings.append((p.stem, load_image(p, args.resolution)))
        except Exception as e:
            print(f"  WARNING: could not load {p.name}: {e}")
    if not paintings:
        print("ERROR: No images could be loaded.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    sheet_dir = os.path.join(args.output_dir, "contact_sheets")
    os.makedirs(sheet_dir, exist_ok=True)

    samples_per_painting = max(1, args.samples_per_type // len(paintings))

    for corruption in CHANNEL_NAMES:
        type_cfg = cfg.types[corruption]
        local_ok = bool(type_cfg.get("local_enabled", True))
        global_ok = bool(type_cfg.get("global_enabled", False))
        if not local_ok and not global_ok:
            print(f"Skipping {corruption} (disabled in config)")
            continue

        type_dir = os.path.join(args.output_dir, corruption)
        os.makedirs(type_dir, exist_ok=True)

        print(f"\n[{corruption}] generating {args.samples_per_type} samples...")
        thumbs: List[Image.Image] = []
        mask_thumbs: List[Image.Image] = []
        labels: List[str] = []
        saved = 0

        for stem, image in paintings:
            for i in range(samples_per_painting):
                seed_i = args.seed * 10000 + CHANNEL_NAMES.index(corruption) * 1000 + saved
                gen = torch.Generator(device="cpu")
                gen.manual_seed(seed_i)

                # Random mode.
                modes = []
                if local_ok:
                    modes.append("local")
                if global_ok:
                    modes.append("global")
                mode = random.choice(modes)

                # Random severity across full range.
                min_s = float(type_cfg.get("min_severity", 0.01))
                max_s = float(type_cfg.get("max_severity", 1.0))
                severity = min_s + random.random() * (max_s - min_s)

                try:
                    out, soft_mask, actual_mask, n_inst = apply_corruption(image, corruption, mode, severity, type_cfg, gen)
                except Exception as e:
                    print(f"  ERROR on {stem} sev={severity:.2f}: {e}")
                    continue

                sev_str = f"{severity:.2f}"
                fname = f"{corruption}_{stem}_sev{sev_str}_n{n_inst}.jpg"
                fpath = os.path.join(type_dir, fname)
                out_pil = overlay_red_mask(tensor_to_pil(out), actual_mask, soft_mask)
                out_pil.save(fpath, quality=90)

                thumbs.append(out_pil)
                mask_thumbs.append(mask_to_pil(actual_mask, args.resolution))
                labels.append(f"{stem[:10]} s={sev_str} n={n_inst}")
                saved += 1

        cols = 6
        blank_img = Image.new("RGB", (args.resolution, args.resolution), (40, 40, 40))
        blank_mask = Image.new("L", (args.resolution, args.resolution), 0)
        while len(thumbs) % cols != 0:
            thumbs.append(blank_img)
            mask_thumbs.append(blank_mask)
            labels.append("")

        sheet = make_contact_sheet(
            thumbs, mask_thumbs, labels, cols=cols, thumb=250, title=f"{corruption}  ({saved} samples)"
        )
        sheet_path = os.path.join(sheet_dir, f"{corruption}_sheet.jpg")
        sheet.save(sheet_path, quality=90)
        print(f"  Saved {saved} images → {type_dir}/")
        print(f"  Contact sheet → {sheet_path}")

    print(f"\nDone. Open {args.output_dir}/ to review.")


if __name__ == "__main__":
    main()
