"""FLUX.2 VAE wrapper for encoding and decoding in latent space.

Uses AutoEncoder from flux2/autoencoder.py, loaded via flux2/util.py.
All VAE parameters are frozen throughout training.

Shape reference:
    encode: (B, 3, H, W) in [0,1]  →  (B, 128, H/16, W/16)
    decode: (B, 128, H/16, W/16)   →  (B, 3, H, W) in [0,1]

    Internally the VAE expects [-1, 1]; this wrapper handles the conversion.

    H and W must be divisible by spatial_compression (16).
"""

import torch
import torch.nn as nn
from torch import Tensor

from .flux2.util import load_ae


class FluxVAE(nn.Module):
    """Frozen FLUX.2 VAE (AutoEncoderKL, 128-channel latents, 16× compression).

    Loaded from black-forest-labs/FLUX.2-dev AE weights (shared across klein models).
    """

    SPATIAL_COMPRESSION = 16   # 8× (encoder downsampling) × 2× (2×2 patchify)
    LATENT_CHANNELS = 128       # z_channels=32 × ps[0]*ps[1]=4

    def __init__(self, flux_model_name: str = "flux.2-klein-base-4b", device: str = "cuda"):
        """Load and freeze the pretrained FLUX.2 VAE.

        Args:
            flux_model_name: Key in FLUX2_MODEL_INFO; resolves the AE repo/weights.
            device: Device to load the model onto.
        """
        ...

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Encode images to latent representations.

        Converts [0,1] → [-1,1] before passing to the VAE encoder.

        Args:
            x: (B, 3, H, W) float32 in [0, 1]. H, W divisible by 16.

        Returns:
            z: (B, 128, H/16, W/16) — BN-normalized latent.
        """
        ...

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """Decode latents back to pixel space.

        Converts decoder output [-1,1] → [0,1] and clamps.

        Args:
            z: (B, 128, H/16, W/16).

        Returns:
            (B, 3, H, W) float32 in [0, 1].
        """
        ...

    @property
    def spatial_compression(self) -> int:
        """Total spatial downsampling factor (16)."""
        ...

    @property
    def latent_channels(self) -> int:
        """Number of latent channels (128)."""
        ...
