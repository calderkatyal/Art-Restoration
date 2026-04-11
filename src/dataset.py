"""Dataset classes for art restoration training and evaluation.

ArtRestorationDataset  — clean WikiArt images, corruption applied on-the-fly.
RealDamageDataset      — pre-damaged images + optional mask files (e.g. MuralDH).

All images are loaded as (3, H, W) float32 tensors in [0, 1] after
center-crop → resize to cfg.train.resolution.

Masks are (K, H, W) float32 binary tensors at pixel resolution.
They are downsampled to latent resolution inside the training loop, not here.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps

from .config import DegradationConfig
from .corruption import CorruptionModule

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LOGGER = logging.getLogger(__name__)


class ArtRestorationDataset(Dataset):
    """Training dataset: clean artwork + on-the-fly synthetic corruption.

    Each __getitem__ returns:
        clean:     (3, H, W) float32 in [0, 1]  — original artwork
        corrupted: (3, H, W) float32 in [0, 1]  — synthetically degraded
        mask:      (K, H, W) float32 binary      — per-channel damage mask
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        degradation_config: DegradationConfig,
        max_simultaneous: Optional[int] = None,
    ):
        """Scan data_dir recursively for images and initialize corruption module.

        Args:
            data_dir:           Root directory of clean artwork images.
            resolution:         Square crop target (H = W = resolution).
            degradation_config: Config for CorruptionModule.
            max_simultaneous:   Curriculum override: max degradations per image.
                                None → use degradation_config.max_simultaneous.
        """
        _validate_resolution(resolution)

        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.image_paths = _find_images(data_dir)
        self.corruption = CorruptionModule(degradation_config)
        self.max_simultaneous = (
            degradation_config.max_simultaneous
            if max_simultaneous is None
            else max_simultaneous
        )

        if self.max_simultaneous < 1:
            raise ValueError("max_simultaneous must be >= 1")
        if self.max_simultaneous > degradation_config.num_channels:
            raise ValueError(
                "max_simultaneous cannot exceed degradation_config.num_channels"
            )

    def __len__(self) -> int:
        """Number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load, preprocess, and corrupt one image.

        Returns dict with keys: 'clean', 'corrupted', 'mask'.
        """
        path = self.image_paths[idx]
        clean = self._load_and_preprocess(path)
        corrupted, mask = self.corruption(
            clean.clone(),
            max_simultaneous=self.max_simultaneous,
        )
        return {"clean": clean, "corrupted": corrupted, "mask": mask}

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        """Load image, center-crop to square, resize to self.resolution.

        Args:
            path: Path to image file.

        Returns:
            (3, H, W) float32 in [0, 1].
        """
        image = _open_image(path, mode="RGB")
        return _crop_resize_to_tensor(
            image,
            resolution=self.resolution,
            interpolation=InterpolationMode.BICUBIC,
        )


class RealDamageDataset(Dataset):
    """Evaluation dataset for real damaged artwork (e.g. MuralDH).

    Expects data_dir/ containing damaged images; optionally mask_dir/
    with single-channel PNG masks whose filenames match image stems.
    If mask_dir is None, a full-image all-ones mask is returned.

    Each __getitem__ returns:
        damaged: (3, H, W) float32 in [0, 1]
        mask:    (K, H, W) float32 binary
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        mask_dir: Optional[str] = None,
        num_mask_channels: int = 5,
    ):
        """Initialize evaluation dataset.

        Args:
            data_dir:          Directory of damaged artwork images.
            resolution:        Square crop target.
            mask_dir:          Optional directory of corresponding mask PNGs.
                               Grayscale, 0 = intact, >0 = damaged.
                               Broadcast to num_mask_channels output channels.
            num_mask_channels: K for output mask.
        """
        _validate_resolution(resolution)
        if num_mask_channels < 1:
            raise ValueError("num_mask_channels must be >= 1")

        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.num_mask_channels = num_mask_channels
        self.image_paths = _find_images(data_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.mask_paths = (
            _build_stem_index(self.mask_dir) if self.mask_dir is not None else {}
        )

    def __len__(self) -> int:
        """Number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load damaged image and its mask.

        Returns dict with keys: 'damaged', 'mask'.
        """
        image_path = self.image_paths[idx]
        damaged = self._load_and_preprocess(image_path)
        h, w = damaged.shape[-2:]

        if self.mask_dir is None:
            mask = torch.ones((self.num_mask_channels, h, w), dtype=torch.float32)
        else:
            mask_path = self.mask_paths.get(image_path.stem)
            if mask_path is None:
                LOGGER.warning(
                    "No mask found for image '%s' in '%s'; using all-ones mask.",
                    image_path.name,
                    self.mask_dir,
                )
                mask = torch.ones((self.num_mask_channels, h, w), dtype=torch.float32)
            else:
                mask = self._load_mask(mask_path, h, w)

        return {"damaged": damaged, "mask": mask}

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        """Load, center-crop, resize image.

        Args:
            path: Path to image file.

        Returns:
            (3, H, W) float32 in [0, 1].
        """
        image = _open_image(path, mode="RGB")
        return _crop_resize_to_tensor(
            image,
            resolution=self.resolution,
            interpolation=InterpolationMode.BICUBIC,
        )

    def _load_mask(self, path: Path, h: int, w: int) -> torch.Tensor:
        """Load a grayscale mask PNG and broadcast to K channels.

        Args:
            path: Path to mask file.
            h:    Target height.
            w:    Target width.

        Returns:
            (K, H, W) float32 binary mask.
        """
        mask_image = _open_image(path, mode="L")
        mask = _crop_resize_to_tensor(
            mask_image,
            resolution=h,
            interpolation=InterpolationMode.NEAREST,
        )
        mask = (mask > 0).to(dtype=torch.float32)
        return mask.repeat(self.num_mask_channels, 1, 1)


def _find_images(root: str) -> List[Path]:
    """Recursively find all image files under root directory.

    Args:
        root: Root directory path.

    Returns:
        Sorted list of image Paths with extensions in IMG_EXTS.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise ValueError(f"Image root does not exist: {root}")
    if not root_path.is_dir():
        raise ValueError(f"Image root is not a directory: {root}")

    image_paths = sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    )
    if not image_paths:
        raise ValueError(f"No images found under: {root}")
    return image_paths


def _validate_resolution(resolution: int) -> None:
    """Validate target square resolution for VAE-compatible preprocessing."""
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    if resolution % 16 != 0:
        raise ValueError("resolution must be divisible by 16")


def _open_image(path: Path, mode: str) -> Image.Image:
    """Open an image file, apply EXIF orientation, and convert color mode."""
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert(mode)
    except Exception as exc:
        raise RuntimeError(f"Failed to load image: {path}") from exc


def _crop_resize_to_tensor(
    image: Image.Image,
    resolution: int,
    interpolation: InterpolationMode,
) -> torch.Tensor:
    """Center-crop a PIL image to square, resize, and convert to float tensor."""
    width, height = image.size
    crop_size = min(width, height)
    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)
    image = TF.crop(image, top=top, left=left, height=crop_size, width=crop_size)
    image = TF.resize(
        image,
        size=[resolution, resolution],
        interpolation=interpolation,
        antialias=interpolation != InterpolationMode.NEAREST,
    )
    return TF.to_tensor(image)


def _build_stem_index(root: Path) -> Dict[str, Path]:
    """Index image files by stem for mask lookup, rejecting duplicates."""
    if not root.exists():
        raise ValueError(f"Mask root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Mask root is not a directory: {root}")

    index: Dict[str, Path] = {}
    for path in _find_images(str(root)):
        stem = path.stem
        if stem in index:
            raise ValueError(
                f"Multiple mask files found for stem '{stem}': "
                f"{index[stem]} and {path}"
            )
        index[stem] = path
    return index
