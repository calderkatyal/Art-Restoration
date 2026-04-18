"""Dataset classes for art restoration training and evaluation.

ArtRestorationDataset  — clean WikiArt (or similar) images with on-the-fly synthetic
                         corruption from :class:`~src.corruption.CorruptionModule`.
RealDamageDataset      — pre-damaged images plus optional mask files (e.g. MuralDH).

All RGB tensors are ``(3, H, W)`` ``float32`` in ``[0, 1]`` after center-crop to a
square and resize to ``cfg.train.resolution``.

Masks are ``(K, H, W)`` ``float32`` at **pixel** resolution. The training loop
downsamples them to latent resolution with :func:`~src.corruption.downsample_mask`;
``K`` must match ``cfg.model.mask_channels`` / ``CorruptionConfig.num_channels``.
"""

from pathlib import Path
from typing import Dict, List, Optional
import copy
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import CorruptionConfig
from .corruption import CorruptionModule

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ArtRestorationDataset(Dataset):
    """Training dataset: clean artwork + on-the-fly synthetic corruption.

    Each ``__getitem__`` returns:
        ``clean``:     ``(3, H, W)`` original artwork in ``[0, 1]``.
        ``corrupted``: ``(3, H, W)`` synthetically degraded image.
        ``mask``:      ``(K, H, W)`` per-channel damage mask (``K`` corruption types).

    Curriculum / ``max_simultaneous``:
        When ``max_simultaneous == 1``, a **copy** of ``corruption_config`` is made with
        ``individual_prob = 1.0`` so the corruption module preferentially samples
        single-degradation presets during early training epochs.
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        corruption_config: CorruptionConfig,
        max_simultaneous: Optional[int] = None,
    ):
        """Scan ``data_dir`` for images and build a :class:`~src.corruption.CorruptionModule`.

        Args:
            data_dir:          Root directory scanned recursively for image extensions.
            resolution:        Square ``H == W`` after center-crop and resize.
            corruption_config: Dataclass controlling preset weights and severity.
            max_simultaneous:  If ``1``, bias corruption toward single-type damage; else unused.
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        cfg = copy.deepcopy(corruption_config)
        if max_simultaneous == 1:
            cfg.individual_prob = 1.0
        self.corruptor = CorruptionModule(cfg)
        self.image_paths = _find_images(str(self.data_dir))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {self.data_dir.resolve()}")

        self.tensorize = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        """Number of images discovered under ``data_dir``."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load, preprocess, and corrupt one image.

        Args:
            idx: Index into the sorted image path list.

        Returns:
            Dict with keys ``"clean"``, ``"corrupted"``, ``"mask"`` as ``torch.Tensor``s.
        """
        path = self.image_paths[idx]
        clean = self._load_and_preprocess(path)
        seed = random.randint(0, 2**31 - 1)
        corrupted, mask = self.corruptor(clean, seed=seed)
        return {
            "clean": clean,
            "corrupted": corrupted,
            "mask": mask,
        }

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        """Load an image file, center-crop to square, resize to ``self.resolution``.

        Args:
            path: Path to ``jpg`` / ``png`` / etc.

        Returns:
            ``(3, H, W)`` ``float32`` tensor in ``[0, 1]``.
        """
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            im = im.crop((left, top, left + side, top + side))
            im = im.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            x = self.tensorize(im)
        return x.float().clamp(0.0, 1.0)


class RealDamageDataset(Dataset):
    """Evaluation dataset for real damaged artwork (e.g. MuralDH).

    Expects ``data_dir`` with damaged RGB images. Optional ``mask_dir`` holds
    single-channel PNG masks (filename stem matches the image stem). Values ``> 0``
    denote damaged pixels; the mask is broadcast to ``num_mask_channels`` planes.

    If no mask file is found (or ``mask_dir`` is ``None``), an all-ones mask is used
    (treat entire canvas as damaged region for metrics that depend on ``mask``).
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        mask_dir: Optional[str] = None,
        num_mask_channels: int = 8,
    ):
        """Initialize paths, resolution, and mask channel count ``K``.

        Args:
            data_dir:          Directory of damaged artwork images.
            resolution:        Square side length after crop + resize.
            mask_dir:          Optional directory of mask PNGs aligned by stem name.
            num_mask_channels: ``K`` for the expanded ``(K, H, W)`` mask tensor.
        """
        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.resolution = resolution
        self.num_mask_channels = num_mask_channels
        self.image_paths = _find_images(str(self.data_dir))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {self.data_dir.resolve()}")

        self.tensorize = transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        """Number of damaged images under ``data_dir``."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load one damaged image and its (possibly default) multi-channel mask.

        Returns:
            Dict with keys ``"damaged"`` ``(3,H,W)`` and ``"mask"`` ``(K,H,W)``.
        """
        path = self.image_paths[idx]
        damaged = self._load_and_preprocess(path)
        h, w = damaged.shape[-2:]
        if self.mask_dir is not None:
            stem = path.stem
            for ext in (".png", ".jpg", ".jpeg"):
                cand = self.mask_dir / f"{stem}{ext}"
                if cand.is_file():
                    mask = self._load_mask(cand, h, w)
                    break
            else:
                mask = torch.ones(self.num_mask_channels, h, w, dtype=torch.float32)
        else:
            mask = torch.ones(self.num_mask_channels, h, w, dtype=torch.float32)
        return {"damaged": damaged, "mask": mask}

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        """Same geometry as training: center square crop then resize to ``resolution``."""
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            im = im.crop((left, top, left + side, top + side))
            im = im.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            x = self.tensorize(im)
        return x.float().clamp(0.0, 1.0)

    def _load_mask(self, path: Path, h: int, w: int) -> torch.Tensor:
        """Load a grayscale mask PNG and broadcast to ``(K, H, W)`` binary ``float32``.

        Args:
            path: Mask file path.
            h, w: Spatial size to match the paired image tensor.

        Returns:
            ``(num_mask_channels, h, w)`` with ``1`` where original mask ``> 0``.
        """
        with Image.open(path) as m:
            m = m.convert("L")
            m = m.resize((w, h), Image.Resampling.NEAREST)
            mt = transforms.ToTensor()(m).squeeze(0)
        damaged = (mt > 0.0).float()
        return damaged.unsqueeze(0).expand(self.num_mask_channels, -1, -1).contiguous()


def _find_images(root: str) -> List[Path]:
    """Recursively collect image files under ``root`` whose suffix is in ``IMG_EXTS``.

    Args:
        root: Root directory path.

    Returns:
        Sorted list of :class:`pathlib.Path` objects.
    """
    root_path = Path(root)
    out: List[Path] = []
    for p in sorted(root_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out
