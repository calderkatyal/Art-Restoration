"""Damage generator G: clean painting + noise z -> damaged painting.

Residual U-Net. Predicts a damage delta that is added to the clean input and
clamped to ``[-1, 1]``. Noise is injected at the bottleneck via FiLM (gamma /
beta scaling) so different noise vectors produce different damage patterns
without modifying the spatial structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=out_ch),
        nn.SiLU(inplace=True),
    )


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = _conv_block(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = _conv_block(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class _FiLM(nn.Module):
    """Apply per-channel affine modulation (gamma, beta) derived from a noise vector."""

    def __init__(self, noise_dim: int, num_channels: int):
        super().__init__()
        self.proj = nn.Linear(noise_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        params = self.proj(z)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class DamageGenerator(nn.Module):
    """Residual U-Net damage generator with FiLM noise conditioning at the bottleneck.

    Args:
        base_channels:  Channels in the first encoder block (doubled at each level).
        noise_dim:      Dimensionality of the per-image noise vector ``z``.
        delta_scale:    Maximum magnitude of the delta added to the clean image.
    """

    def __init__(
        self,
        base_channels: int = 64,
        noise_dim: int = 128,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        c = base_channels
        self.noise_dim = noise_dim
        self.delta_scale = float(delta_scale)

        # Encoder
        self.in_conv = _conv_block(3, c)
        self.down1 = _Down(c, c * 2)
        self.down2 = _Down(c * 2, c * 4)
        self.down3 = _Down(c * 4, c * 8)
        self.down4 = _Down(c * 8, c * 8)

        # Bottleneck conditioned on noise
        self.bottleneck = _conv_block(c * 8, c * 8)
        self.film = _FiLM(noise_dim=noise_dim, num_channels=c * 8)

        # Decoder
        self.up1 = _Up(in_ch=c * 8, skip_ch=c * 8, out_ch=c * 8)
        self.up2 = _Up(in_ch=c * 8, skip_ch=c * 4, out_ch=c * 4)
        self.up3 = _Up(in_ch=c * 4, skip_ch=c * 2, out_ch=c * 2)
        self.up4 = _Up(in_ch=c * 2, skip_ch=c, out_ch=c)

        self.out_conv = nn.Conv2d(c, 3, kernel_size=3, padding=1)

    def forward(self, clean: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Generate a damaged version of ``clean`` conditioned on noise ``z``.

        Args:
            clean: ``(B, 3, H, W)`` tensor in ``[-1, 1]``.
            z:     ``(B, noise_dim)`` noise vector.

        Returns:
            ``(B, 3, H, W)`` damaged image in ``[-1, 1]``.
        """
        x0 = self.in_conv(clean)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        b = self.bottleneck(x4)
        b = self.film(b, z)

        u = self.up1(b, x3)
        u = self.up2(u, x2)
        u = self.up3(u, x1)
        u = self.up4(u, x0)

        delta = self.out_conv(u).tanh() * self.delta_scale
        return (clean + delta).clamp(-1.0, 1.0)

    def sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.noise_dim, device=device)
