"""Dataset classes for art restoration training and evaluation.

ArtRestorationDataset  — clean WikiArt images, corruption applied on-the-fly.
RealDamageDataset      — pre-damaged images + optional mask files (e.g. MuralDH).

All images are loaded as (3, H, W) float32 tensors in [0, 1] after
center-crop → resize to cfg.train.resolution.

Masks are (K, H, W) float32 binary tensors at pixel resolution.
They are downsampled to latent resolution inside the training loop, not here.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .config import DegradationConfig
from .corruption import CorruptionModule

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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
        ...

    def __len__(self) -> int:
        """Number of images in the dataset."""
        ...

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load, preprocess, and corrupt one image.

        Returns dict with keys: 'clean', 'corrupted', 'mask'.
        """
        ...

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        """Load image, center-crop to square, resize to self.resolution.

        Args:
            path: Path to image file.

        Returns:
            (3, H, W) float32 in [0, 1].
        """
        ...


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
        ...

    def __len__(self) -> int:
        """Number of images in the dataset."""
        ...

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load damaged image and its mask.

        Returns dict with keys: 'damaged', 'mask'.
        """
        ...

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        """Load, center-crop, resize image.

        Args:
            path: Path to image file.

        Returns:
            (3, H, W) float32 in [0, 1].
        """
        ...

    def _load_mask(self, path: Path, h: int, w: int) -> torch.Tensor:
        """Load a grayscale mask PNG and broadcast to K channels.

        Args:
            path: Path to mask file.
            h:    Target height.
            w:    Target width.

        Returns:
            (K, H, W) float32 binary mask.
        """
        ...


def _find_images(root: str) -> List[Path]:
    """Recursively find all image files under root directory.

    Args:
        root: Root directory path.

    Returns:
        Sorted list of image Paths with extensions in IMG_EXTS.
    """
    ...
