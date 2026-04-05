"""Synthetic corruption module  C(x) -> (y, M).

Given a clean image x, produces a degraded image y and a multi-channel
binary damage mask M ∈ {0,1}^{K×H×W}.

Each of the K channels corresponds to one damage type; a pixel is 1 if
that damage type was applied there.  Multiple channels can be active at
the same spatial location.

downsample_mask() converts M from pixel resolution to latent resolution
using max-pooling (kernel = stride = spatial_compression = 16), so any
damaged pixel within a 16×16 block survives in M'.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from .config import DegradationConfig


class CorruptionModule:
    """Non-learned stochastic corruption pipeline.

    Dispatches to per-damage-type methods based on config.damage_types order.
    Each method applies localized degradation within a randomly sampled region.
    """

    def __init__(self, config: DegradationConfig):
        """Initialize with degradation config.

        Args:
            config: DegradationConfig specifying damage types, severity range,
                    and max simultaneous degradations.
        """
        ...

    def __call__(
        self,
        image: torch.Tensor,
        max_simultaneous: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random subset of degradations to a clean image.

        Randomly selects n ∈ [1, max_simultaneous] damage types without
        replacement and applies each to a randomly sampled region.

        Args:
            image: (C, H, W) float32 in [0, 1].
            max_simultaneous: Override for curriculum learning. Defaults to
                              config.max_simultaneous.

        Returns:
            corrupted: (C, H, W) float32 in [0, 1] — degraded image.
            mask:      (K, H, W) float32 binary    — per-channel damage mask.
        """
        ...

    def _sample_region_mask(self, h: int, w: int) -> torch.Tensor:
        """Sample a random rectangular region covering 10–60% of each dimension.

        Args:
            h: Image height.
            w: Image width.

        Returns:
            (H, W) float32 binary mask — 1 inside the sampled box, 0 outside.
        """
        ...

    def _apply_crack(
        self, image: torch.Tensor, region: torch.Tensor, severity: float
    ) -> torch.Tensor:
        """Darken thin Gaussian-noise structures to simulate cracks.

        Args:
            image:    (C, H, W) in [0, 1].
            region:   (H, W) binary — spatial region to corrupt.
            severity: Float in severity_range, controls darkness and density.

        Returns:
            (C, H, W) image with crack artifacts inside region.
        """
        ...

    def _apply_paint_loss(
        self, image: torch.Tensor, region: torch.Tensor, severity: float
    ) -> torch.Tensor:
        """Replace random pixels with a noisy neutral underlayer to simulate flaking.

        Args:
            image:    (C, H, W) in [0, 1].
            region:   (H, W) binary.
            severity: Controls fraction of pixels lost.

        Returns:
            (C, H, W) image with paint-loss patches inside region.
        """
        ...

    def _apply_stain(
        self, image: torch.Tensor, region: torch.Tensor, severity: float
    ) -> torch.Tensor:
        """Blend a brownish tint into the region to simulate water/smoke damage.

        Args:
            image:    (C, H, W) in [0, 1].
            region:   (H, W) binary.
            severity: Controls tint strength.

        Returns:
            (C, H, W) image with stain inside region.
        """
        ...

    def _apply_blur(
        self, image: torch.Tensor, region: torch.Tensor, severity: float
    ) -> torch.Tensor:
        """Apply separable Gaussian blur within the region.

        Kernel size and sigma scale with severity.

        Args:
            image:    (C, H, W) in [0, 1].
            region:   (H, W) binary.
            severity: Controls kernel size (3–15) and sigma (1–5).

        Returns:
            (C, H, W) image with blur inside region.
        """
        ...

    def _apply_color_shift(
        self, image: torch.Tensor, region: torch.Tensor, severity: float
    ) -> torch.Tensor:
        """Randomly scale each color channel within the region.

        Args:
            image:    (C, H, W) in [0, 1].
            region:   (H, W) binary.
            severity: Controls per-channel scale deviation from 1.0.

        Returns:
            (C, H, W) image with color-shifted region.
        """
        ...


def downsample_mask(mask: torch.Tensor, factor: int = 16) -> torch.Tensor:
    """Downsample pixel-resolution mask to latent resolution via max pooling.

    A damaged pixel in any position within a (factor × factor) block
    propagates a 1 to the corresponding latent-resolution cell.

    Args:
        mask:   Binary mask (K, H, W) or (B, K, H, W). float32.
        factor: Spatial compression factor (16 for FLUX.2 VAE).

    Returns:
        Downsampled binary mask (K, H//factor, W//factor)
        or (B, K, H//factor, W//factor).
    """
    ...
