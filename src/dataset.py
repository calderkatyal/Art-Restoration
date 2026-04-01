"""Dataset classes for art restoration training and evaluation."""

import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict

from .config import DegradationConfig
from .corruption import CorruptionModule


class ArtRestorationDataset(Dataset):
    """Training dataset: loads clean artwork images and applies synthetic corruption on-the-fly.

    Each sample returns:
        - clean: Clean image tensor (C, H, W) in [0, 1].
        - corrupted: Synthetically degraded image (C, H, W) in [0, 1].
        - mask: Multi-channel damage mask (K, H, W), binary.
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        degradation_config: DegradationConfig,
        max_simultaneous: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            data_dir: Path to directory of clean artwork images.
            resolution: Target resolution for square center crops.
            degradation_config: Configuration for the corruption module.
            max_simultaneous: Override for curriculum learning (e.g. 1 during warmup).
        """
        ...

    def __len__(self) -> int:
        """Return number of images in the dataset."""
        ...

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and corrupt a single artwork image.

        Returns:
            Dict with keys 'clean', 'corrupted', 'mask'.
        """
        ...

    def _load_and_preprocess(self, path: str) -> torch.Tensor:
        """Load an image from disk and apply preprocessing (resize, crop, normalize).

        Args:
            path: Path to image file.

        Returns:
            Image tensor (C, H, W) in [0, 1].
        """
        ...


class RealDamageDataset(Dataset):
    """Evaluation dataset for real damaged artwork (e.g. MuralDH).

    Each sample returns:
        - damaged: Real damaged image tensor (C, H, W) in [0, 1].
        - mask: User-provided or estimated damage mask (K, H, W), binary.
    """

    def __init__(self, data_dir: str, mask_dir: str, resolution: int):
        """Initialize real damage evaluation dataset.

        Args:
            data_dir: Path to directory of damaged artwork images.
            mask_dir: Path to directory of corresponding damage masks.
            resolution: Target resolution for square center crops.
        """
        ...

    def __len__(self) -> int:
        """Return number of images in the dataset."""
        ...

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a real damaged image and its mask.

        Returns:
            Dict with keys 'damaged', 'mask'.
        """
        ...
