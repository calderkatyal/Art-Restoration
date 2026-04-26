"""Datasets for Stage 1.

Two independent image streams are needed:

* ``CleanImageDataset``  — clean paintings, used as G inputs.
* ``DamagedImageDataset`` — real damaged images (ARTeFACT, MuralDH, Real-Old,
  ...), used as positive samples for D.

The two are sampled independently each step (unpaired).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _find_images(roots: Sequence[Path]) -> List[Path]:
    paths: List[Path] = []
    for root in roots:
        if not root.exists():
            raise ValueError(f"Image root does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Image root is not a directory: {root}")
        paths.extend(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
    if not paths:
        raise ValueError(f"No images found under any of: {[str(r) for r in roots]}")
    paths.sort()
    return paths


def _open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def _load_resized(path: Path, resolution: int) -> torch.Tensor:
    """Center-crop to a square then bicubic-resize to ``resolution`` and return ``[-1, 1]``."""
    img = _open_image(path)
    w, h = img.size
    crop = min(w, h)
    top = max((h - crop) // 2, 0)
    left = max((w - crop) // 2, 0)
    img = TF.crop(img, top=top, left=left, height=crop, width=crop)
    img = TF.resize(img, size=[resolution, resolution], interpolation=InterpolationMode.BICUBIC, antialias=True)
    tensor = TF.to_tensor(img).clamp(0.0, 1.0)
    return tensor * 2.0 - 1.0


class CleanImageDataset(Dataset):
    """Clean painting dataset, returns tensors in ``[-1, 1]``."""

    def __init__(self, roots: Sequence[str], resolution: int):
        self.paths = _find_images([Path(r) for r in roots])
        self.resolution = int(resolution)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _load_resized(self.paths[idx], self.resolution)


class DamagedImageDataset(Dataset):
    """Real damaged image dataset (ARTeFACT, MuralDH, Real-Old, ...).

    Multiple roots may be combined into one shuffled stream.
    """

    def __init__(self, roots: Sequence[str], resolution: int):
        self.paths = _find_images([Path(r) for r in roots])
        self.resolution = int(resolution)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _load_resized(self.paths[idx], self.resolution)


def _build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    seed: int,
    distributed: bool,
) -> DataLoader:
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=int(seed),
            drop_last=True,
        )
    else:
        generator = torch.Generator().manual_seed(int(seed))
        sampler = RandomSampler(dataset, replacement=False, generator=generator)

    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "sampler": sampler,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": True,
        "persistent_workers": bool(persistent_workers) and int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(**kwargs)


def build_clean_loader(
    roots: Sequence[str],
    *,
    resolution: int,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    seed: int = 42,
    distributed: bool = False,
) -> tuple[CleanImageDataset, DataLoader]:
    dataset = CleanImageDataset(roots=roots, resolution=resolution)
    loader = _build_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        distributed=distributed,
    )
    return dataset, loader


def build_damaged_loader(
    roots: Sequence[str],
    *,
    resolution: int,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    seed: int = 4242,
    distributed: bool = False,
) -> tuple[DamagedImageDataset, DataLoader]:
    dataset = DamagedImageDataset(roots=roots, resolution=resolution)
    loader = _build_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        distributed=distributed,
    )
    return dataset, loader


def cycle(loader: DataLoader):
    """Infinite iterator over ``loader`` (re-instantiates the iterator at each epoch)."""
    while True:
        for batch in loader:
            yield batch
