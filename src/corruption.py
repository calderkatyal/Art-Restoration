"""Synthetic corruption module C(x) -> (y, M).

Generates spatially localized, severity-controlled damage within randomly
sampled masked regions. Each degradation type has its own binary mask channel.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from .config import DegradationConfig


class CorruptionModule:
    """Non-learned stochastic corruption pipeline.

    Given a clean image x, produces a corrupted image y and a multi-channel
    binary mask M in {0,1}^{K x H x W} indicating which pixels were damaged
    and by which degradation type.
    """

    def __init__(self, config: DegradationConfig):
        """Initialize corruption module with degradation config."""
        ...

    def __call__(
        self, image: torch.Tensor, max_simultaneous: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random degradations to a clean image.

        Args:
            image: Clean image tensor, shape (C, H, W), values in [0, 1].
            max_simultaneous: Override max number of simultaneous degradations.

        Returns:
            corrupted: Degraded image tensor, same shape as input.
            mask: Multi-channel damage mask, shape (K, H, W), binary.
        """
        ...

    def _sample_region_mask(self, h: int, w: int) -> torch.Tensor:
        """Sample a random spatial region for a single degradation.

        Args:
            h: Image height.
            w: Image width.

        Returns:
            Binary mask of shape (H, W) indicating the damaged region.
        """
        ...

    def _apply_crack(
        self, image: torch.Tensor, region: torch.Tensor, severity: float
    ) -> torch.Tensor:
        """Apply synthetic crack degradation to masked region.

        Args:
            image: Current image tensor (C, H, W).
            region: Binary spatial mask (H, W).
            severity: Degradation strength in [0, 1].

        Returns:
            Modified image with crack artifacts in the masked region.
        """
        ...

    def _apply_paint_loss(
        self, image: torch.Tensor, region: torch.Tensor, severity: float
    ) -> torch.Tensor:
        """Apply synthetic paint loss / flaking to masked region.

        Args:
            image: Current image tensor (C, H, W).
            region: Binary spatial mask (H, W).
            severity: Degradation strength in [0, 1].

        Returns:
            Modified image with paint loss in the masked region.
        """
        ...

    def _apply_stain(
        self, image: torch.Tensor, region: torch.Tensor, severity: float
    ) -> torch.Tensor:
        """Apply synthetic stain (water/smoke damage) to masked region.

        Args:
            image: Current image tensor (C, H, W).
            region: Binary spatial mask (H, W).
            severity: Degradation strength in [0, 1].

        Returns:
            Modified image with stain artifacts in the masked region.
        """
        ...

    def _apply_blur(
        self, image: torch.Tensor, region: torch.Tensor, severity: float
    ) -> torch.Tensor:
        """Apply localized blur degradation to masked region.

        Args:
            image: Current image tensor (C, H, W).
            region: Binary spatial mask (H, W).
            severity: Degradation strength in [0, 1].

        Returns:
            Modified image with blur in the masked region.
        """
        ...

    def _apply_color_shift(
        self, image: torch.Tensor, region: torch.Tensor, severity: float
    ) -> torch.Tensor:
        """Apply localized color shift / discoloration to masked region.

        Args:
            image: Current image tensor (C, H, W).
            region: Binary spatial mask (H, W).
            severity: Degradation strength in [0, 1].

        Returns:
            Modified image with color shift in the masked region.
        """
        ...


def downsample_mask(mask: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample pixel-resolution mask to latent resolution via max pooling.

    Ensures damaged pixels within a spatial block are preserved in the
    downsampled mask M'.

    Args:
        mask: Binary mask of shape (K, H, W) or (B, K, H, W).
        factor: Spatial compression factor of the VAE (e.g. 8).

    Returns:
        Downsampled binary mask of shape (K, H', W') or (B, K, H', W').
    """
    ...
