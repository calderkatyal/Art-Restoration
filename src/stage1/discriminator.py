"""PatchGAN discriminator for distinguishing real damaged images from G outputs.

Multi-scale: D operates at the input scale and at a 2x-downsampled scale, then
the per-patch logits from each scale are averaged into a single score map per
example. Spectral normalization on every conv stabilizes training.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _disc_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)),
        nn.LeakyReLU(0.2, inplace=True),
    )


class _SinglePatchD(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = [_disc_block(in_channels, base_channels, stride=2)]
        ch = base_channels
        for i in range(n_layers - 1):
            next_ch = min(ch * 2, base_channels * 8)
            layers.append(_disc_block(ch, next_ch, stride=2))
            ch = next_ch
        next_ch = min(ch * 2, base_channels * 8)
        layers.append(_disc_block(ch, next_ch, stride=1))
        ch = next_ch
        layers.append(spectral_norm(nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """Two-scale PatchGAN discriminator with spectral normalization."""

    def __init__(self, base_channels: int = 64, n_layers: int = 3, num_scales: int = 2):
        super().__init__()
        self.num_scales = int(num_scales)
        self.scales = nn.ModuleList(
            [_SinglePatchD(in_channels=3, base_channels=base_channels, n_layers=n_layers)
             for _ in range(self.num_scales)]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return a list of per-scale logit maps shaped ``(B, 1, h_s, w_s)``."""
        outputs: List[torch.Tensor] = []
        current = x
        for i, scale_d in enumerate(self.scales):
            outputs.append(scale_d(current))
            if i < self.num_scales - 1:
                current = F.avg_pool2d(current, kernel_size=2)
        return outputs
