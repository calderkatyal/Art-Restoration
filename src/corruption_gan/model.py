"""U-Net Generator and 70x70 PatchGAN Discriminator for learned corruption.

Architecture follows the original pix2pix paper (Isola et al., 2017) with
modifications for 512x512 inputs:

Generator (UNetGenerator):
    8-level encoder–decoder with skip connections.
    Input:  (B, 3, 512, 512)  clean painting, normalised to [-1, 1].
    Output: (B, 3, 512, 512)  synthetic damaged painting in [-1, 1] (tanh).
    Encoder: stride-2 convolutions halve spatial dims each level.
    Decoder: transposed convolutions double spatial dims; skip tensors
             from encoder are channel-concatenated before each conv.
    Dropout (p=0.5) on the three innermost decoder blocks for stochasticity.

Discriminator (PatchDiscriminator):
    Conditional 70x70 PatchGAN.  Receives a 6-channel input
    (clean || damaged/generated) and outputs a (B, 1, H', W') real/fake
    score map.  LSGAN training (MSE vs 0/1 targets) is used in train.py.

Normalisation:
    Input images are assumed to be in [0, 1] at the dataset level.
    _normalise / _denormalise convert between [0,1] and [-1,1] internally
    so callers never need to manage this.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Internal normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(x: Tensor) -> Tensor:
    """[0, 1] → [-1, 1]."""
    return x * 2.0 - 1.0


def _denormalise(x: Tensor) -> Tensor:
    """[-1, 1] → [0, 1]."""
    return (x + 1.0) * 0.5


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _enc_block(
    in_ch: int,
    out_ch: int,
    use_bn: bool = True,
    slope: float = 0.2,
) -> nn.Sequential:
    """Encoder block: Conv(stride=2) → optional InstanceNorm → LeakyReLU."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn),
    ]
    if use_bn:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(slope, inplace=True))
    return nn.Sequential(*layers)


def _dec_block(
    in_ch: int,
    out_ch: int,
    use_dropout: bool = False,
) -> nn.Sequential:
    """Decoder block: ConvTranspose(stride=2) → InstanceNorm → optional Dropout → ReLU."""
    layers: list[nn.Module] = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
    ]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class UNetGenerator(nn.Module):
    """8-level U-Net: clean painting → synthetic damaged painting.

    Designed for 512×512 inputs, giving a 2×2 bottleneck (8 stride-2 convs).
    Skip connections concatenate encoder feature maps to decoder inputs.

    Args:
        in_channels:  Input channels (default 3 for RGB clean).
        out_channels: Output channels (default 3 for RGB damaged).
        base_filters: Feature-map width at the first encoder level (default 64).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 64,
    ) -> None:
        super().__init__()
        nf = base_filters
        # Encoder — feature maps: 64, 128, 256, 512, 512, 512, 512, 512
        self.enc1 = _enc_block(in_channels, nf,     use_bn=False)  # 512→256
        self.enc2 = _enc_block(nf,          nf * 2)                # 256→128
        self.enc3 = _enc_block(nf * 2,      nf * 4)                # 128→64
        self.enc4 = _enc_block(nf * 4,      nf * 8)                # 64→32
        self.enc5 = _enc_block(nf * 8,      nf * 8)                # 32→16
        self.enc6 = _enc_block(nf * 8,      nf * 8)                # 16→8
        self.enc7 = _enc_block(nf * 8,      nf * 8)                # 8→4
        # Bottleneck — no BN on the innermost encoder block
        self.enc8 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 8, kernel_size=4, stride=2, padding=1),  # 4→2
            nn.ReLU(inplace=True),
        )

        # Decoder — each level receives (prev_output || skip), hence doubled in_ch
        self.dec1 = _dec_block(nf * 8,      nf * 8, use_dropout=True)   # 2→4,   in=512
        self.dec2 = _dec_block(nf * 8 * 2,  nf * 8, use_dropout=True)   # 4→8,   in=1024
        self.dec3 = _dec_block(nf * 8 * 2,  nf * 8, use_dropout=True)   # 8→16,  in=1024
        self.dec4 = _dec_block(nf * 8 * 2,  nf * 8)                     # 16→32, in=1024
        self.dec5 = _dec_block(nf * 8 * 2,  nf * 4)                     # 32→64, in=1024
        self.dec6 = _dec_block(nf * 4 * 2,  nf * 2)                     # 64→128,in=512
        self.dec7 = _dec_block(nf * 2 * 2,  nf)                         # 128→256,in=256
        # Final output layer — no norm, tanh activation
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, out_channels, kernel_size=4, stride=2, padding=1),  # 256→512
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, H, W) clean painting in [0, 1].
        Returns:
            (B, 3, H, W) synthetic damaged painting in [0, 1].
        """
        x_norm = _normalise(x)
        e1 = self.enc1(x_norm)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        d1 = self.dec1(e8)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))
        out = self.dec8(torch.cat([d7, e1], dim=1))
        return _denormalise(out)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """Conditional 70x70 PatchGAN discriminator.

    Input: 6-channel (clean || damaged) concatenation in [0, 1].
           Both halves are normalised to [-1, 1] internally.
    Output: (B, 1, H', W') — score map; each cell covers a ~70×70 receptive
            field in the input.  LSGAN training (MSE vs 0/1) is recommended.

    Args:
        in_channels: Channels of one image (default 3).  Actual input is
                     ``in_channels * 2`` due to the (clean || damaged) concat.
        base_filters: Feature-map width at first level (default 64).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 64,
    ) -> None:
        super().__init__()
        nf = base_filters
        self.model = nn.Sequential(
            # No BN on the first layer (standard pix2pix)
            nn.Conv2d(in_channels * 2, nf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf,     nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(nf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(nf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # stride=1 keeps spatial size — gives the 70×70 receptive field
            nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Final 1-channel real/fake map
            nn.Conv2d(nf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, clean: Tensor, damaged: Tensor) -> Tensor:
        """
        Args:
            clean:   (B, 3, H, W) in [0, 1].
            damaged: (B, 3, H, W) in [0, 1] — either real damaged or G(clean).
        Returns:
            (B, 1, H', W') score map (no sigmoid; use with LSGAN MSE loss).
        """
        x = torch.cat([_normalise(clean), _normalise(damaged)], dim=1)
        return self.model(x)
