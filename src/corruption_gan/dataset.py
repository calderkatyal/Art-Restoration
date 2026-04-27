"""Datasets for learned corruption GAN training.

PairedArtDataset
    Loads (clean=undamaged, damaged) image pairs from the Kaggle
    ``pes1ug22am047/damaged-and-undamaged-artworks`` dataset.

    Pairing logic:
        Every damaged file contains "before" (case-insensitive) in its stem.
        The corresponding undamaged file has the same stem with "before"
        replaced by "after", optionally with a different file extension.
        Special cases (e.g. ``marybefore.png`` → ``maryafter.png``,
        ``before.png`` → ``after.png``) are handled by the same regex
        substitution.  Unresolvable files are silently skipped.

    Expected directory layout::

        <data_root>/
          paired_dataset_art/
            damaged/          # *-before.jpg / *_before.png / …
            undamaged/        # *-after.jpg  / *_after.png  / …

    Augmentation (training split only):
        RandomResizedCrop → identical spatial crop on both images
        RandomHorizontalFlip → same flip decision on both images
        ColorJitter → independently jittered per image (lightens domain gap)

WikiArtValDataset
    Thin wrapper around the existing WikiArt validation directory.
    Returns plain resized clean images (no corruption); used to visualise
    what the generator does to held-out paintings during training.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Pairing helpers
# ---------------------------------------------------------------------------

def _iter_images(directory: Path) -> List[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )


def _stem_to_after(stem: str) -> str:
    """Replace the first occurrence of 'before' (any case) with the same-case 'after'."""
    def _replace(m: re.Match) -> str:
        s = m.group(0)
        if s.isupper():
            return "AFTER"
        if s[0].isupper():
            return "After"
        return "after"
    return re.sub(r"(?i)before", _replace, stem, count=1)


def _find_undamaged(damaged_path: Path, undamaged_dir: Path) -> Optional[Path]:
    """Return the undamaged counterpart of *damaged_path*, or None if not found."""
    after_stem = _stem_to_after(damaged_path.stem)
    if after_stem == damaged_path.stem:
        # No 'before' token found — unpaired file, skip.
        return None
    # Try same extension first, then common image extensions.
    for ext in [damaged_path.suffix, ".jpg", ".png", ".jpeg", ".webp"]:
        candidate = undamaged_dir / f"{after_stem}{ext}"
        if candidate.exists():
            return candidate
    # Fallback: glob any extension
    hits = list(undamaged_dir.glob(f"{after_stem}.*"))
    hits = [h for h in hits if h.suffix.lower() in _IMG_EXTS]
    return hits[0] if hits else None


def build_pairs(data_root: str) -> List[Tuple[Path, Path]]:
    """Return list of (undamaged_path, damaged_path) pairs from the Kaggle dataset.

    Args:
        data_root: Path to the ``AI_for_Art_Restoration_2`` root directory.

    Returns:
        Sorted, deterministic list of valid (clean, damaged) pairs.
    """
    root = Path(data_root)
    paired = root / "paired_dataset_art"
    damaged_dir = paired / "damaged"
    undamaged_dir = paired / "undamaged"

    pairs: List[Tuple[Path, Path]] = []
    for dam in _iter_images(damaged_dir):
        undam = _find_undamaged(dam, undamaged_dir)
        if undam is not None:
            pairs.append((undam, dam))
    return pairs


# ---------------------------------------------------------------------------
# Augmentation helpers — must apply identical geometry to both images
# ---------------------------------------------------------------------------

_COLOR_JITTER = ColorJitter(
    brightness=0.15,
    contrast=0.15,
    saturation=0.10,
    hue=0.03,
)


def _paired_augment(
    clean_pil: Image.Image,
    damaged_pil: Image.Image,
    resolution: int,
    is_train: bool,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply identical geometry, independent colour jitter, return float tensors [0,1]."""
    if is_train:
        # Random resized crop — same parameters for both
        scale = rng.uniform(0.65, 1.0)
        ratio = rng.uniform(0.8, 1.25)
        i, j, h, w = _get_crop_params(clean_pil, scale, ratio, resolution)
        clean_pil  = TF.resized_crop(clean_pil,  i, j, h, w, (resolution, resolution), antialias=True)
        damaged_pil = TF.resized_crop(damaged_pil, i, j, h, w, (resolution, resolution), antialias=True)

        # Random horizontal flip
        if rng.random() > 0.5:
            clean_pil   = TF.hflip(clean_pil)
            damaged_pil = TF.hflip(damaged_pil)

        # Independent colour jitter
        clean_pil   = _COLOR_JITTER(clean_pil)
        damaged_pil = _COLOR_JITTER(damaged_pil)
    else:
        clean_pil   = TF.resize(clean_pil,   (resolution, resolution), antialias=True)
        damaged_pil = TF.resize(damaged_pil, (resolution, resolution), antialias=True)

    return TF.to_tensor(clean_pil), TF.to_tensor(damaged_pil)


def _get_crop_params(
    img: Image.Image,
    scale: float,
    ratio: float,
    resolution: int,
) -> Tuple[int, int, int, int]:
    """Compute RandomResizedCrop parameters deterministically."""
    w, h = img.size
    area = h * w
    crop_area = area * scale
    crop_w = int(round((crop_area * ratio) ** 0.5))
    crop_h = int(round((crop_area / ratio) ** 0.5))
    crop_w = max(min(crop_w, w), resolution)
    crop_h = max(min(crop_h, h), resolution)
    i = max(0, (h - crop_h) // 2 + random.randint(0, max(0, h - crop_h)))
    j = max(0, (w - crop_w) // 2 + random.randint(0, max(0, w - crop_w)))
    i = min(i, max(0, h - crop_h))
    j = min(j, max(0, w - crop_w))
    return i, j, crop_h, crop_w


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PairedArtDataset(Dataset):
    """(clean, damaged) image pairs for pix2pix corruption GAN training.

    Args:
        data_root:   Path to ``AI_for_Art_Restoration_2/`` directory.
        resolution:  Square output resolution (default 512).
        split:       ``"train"`` or ``"val"``.
        val_fraction: Fraction of pairs reserved for validation (default 0.15).
        seed:        RNG seed for train/val split (default 42).
    """

    def __init__(
        self,
        data_root: str,
        resolution: int = 512,
        split: str = "train",
        val_fraction: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.resolution = int(resolution)
        self.is_train = split.lower() == "train"

        all_pairs = build_pairs(data_root)
        if not all_pairs:
            raise RuntimeError(
                f"No paired images found under {data_root}/paired_dataset_art/. "
                "Check that the Kaggle dataset was extracted there."
            )

        # Deterministic split: sort by clean filename then cut
        all_pairs = sorted(all_pairs, key=lambda p: p[0].name)
        n_val = max(1, int(len(all_pairs) * val_fraction))
        rng = random.Random(seed)
        indices = list(range(len(all_pairs)))
        rng.shuffle(indices)
        val_idx  = set(indices[:n_val])
        train_idx = set(indices[n_val:])

        chosen = train_idx if self.is_train else val_idx
        self.pairs: List[Tuple[Path, Path]] = [all_pairs[i] for i in sorted(chosen)]
        self._rng = random.Random(seed)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clean_path, damaged_path = self.pairs[idx]
        clean_pil   = _load_rgb(clean_path)
        damaged_pil = _load_rgb(damaged_path)

        # Per-item RNG so augmentation varies across epochs
        item_rng = random.Random(self.seed + idx + id(self))
        clean_t, damaged_t = _paired_augment(
            clean_pil, damaged_pil,
            resolution=self.resolution,
            is_train=self.is_train,
            rng=item_rng,
        )
        return {
            "clean":   clean_t,    # (3, H, W) float32 [0, 1]
            "damaged": damaged_t,  # (3, H, W) float32 [0, 1]
            "clean_path":   str(clean_path),
            "damaged_path": str(damaged_path),
        }

    # ---- state dict for resumable loading (mirrors ArtRestorationDataset) ----

    def state_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "split": "train" if self.is_train else "val"}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass  # stateless — augmentation is index-seeded


class WikiArtValDataset(Dataset):
    """Subset of WikiArt validation images for corruption-model visualisation.

    Returns plain resized tensors in [0, 1].  No corruption is applied.
    Used identically to how train.py's fixed_inference_indices work:
    the training loop selects a fixed random subset, applies the generator,
    and logs (clean | generated_damage) panels to WandB.

    Args:
        val_dir:    Path to the WikiArt val split (same as train.val_dir).
        resolution: Target square resolution.
    """

    def __init__(self, val_dir: str, resolution: int = 512) -> None:
        super().__init__()
        self.resolution = int(resolution)
        val_path = Path(val_dir)
        self.image_paths: List[Path] = sorted(
            p for p in val_path.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found under {val_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        img = _load_rgb(path)
        img = TF.resize(img, (self.resolution, self.resolution), antialias=True)
        return {
            "clean": TF.to_tensor(img),  # (3, H, W) float32 [0, 1]
            "path":  str(path),
        }
