"""FLUX.2 VAE wrapper for encoding and decoding in latent space."""

import torch
import torch.nn as nn


class FluxVAE(nn.Module):
    """Frozen FLUX.2 VAE (AutoencoderKL, 16-channel latent space).

    Provides encode/decode interface. All parameters are frozen.
    Spatial compression factor is 8 (H,W -> H/8, W/8).
    """

    def __init__(self, repo_id: str, device: str = "cuda"):
        """Load pretrained FLUX.2 VAE and freeze all parameters.

        Args:
            repo_id: HuggingFace repo ID for FLUX.2 model.
            device: Device to load the model on.
        """
        ...

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space images to latent representations.

        Args:
            x: Image tensor of shape (B, 3, H, W) in [0, 1].

        Returns:
            Latent tensor z of shape (B, 16, H/8, W/8).
        """
        ...

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representations back to pixel space.

        Args:
            z: Latent tensor of shape (B, 16, H/8, W/8).

        Returns:
            Image tensor of shape (B, 3, H, W) in [0, 1].
        """
        ...

    @property
    def spatial_compression(self) -> int:
        """Return the spatial downsampling factor (8 for FLUX VAE)."""
        ...
