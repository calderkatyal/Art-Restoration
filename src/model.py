"""FLUX.2 DiT wrapper for conditional art restoration.

Loads pretrained FLUX.2 [klein] 4B and re-initializes img_in to accept
the concatenated input: [z_t, z_y, M'] along the channel dimension.
All other weights remain pretrained.
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import ModelConfig


class RestorationDiT(nn.Module):
    """Conditional velocity prediction model for latent rectified flow.

    Wraps FLUX.2 [klein] 4B DiT. The img_in projection is re-initialized
    from scratch to accept (latent_channels * 2 + mask_channels) input channels.
    Everything else is loaded from pretrained weights.

    The model predicts v_theta(z_t, t | z_y, M') where:
        - z_t: noisy latent at timestep t, shape (B, 16, H', W')
        - z_y: corrupted image latent, shape (B, 16, H', W')
        - M': downsampled multi-channel mask, shape (B, K, H', W')
        - t: scalar timestep in [0, 1]
    """

    def __init__(self, config: ModelConfig):
        """Load pretrained FLUX.2 [klein] and re-init img_in.

        Args:
            config: Model configuration.
        """
        ...

    def _load_pretrained_flux(self, repo_id: str) -> nn.Module:
        """Load the pretrained FLUX.2 [klein] 4B transformer.

        Args:
            repo_id: HuggingFace repo ID.

        Returns:
            The loaded transformer module.
        """
        ...

    def _reinit_img_in(self, in_channels: int) -> None:
        """Re-initialize the img_in projection layer from scratch.

        The new layer maps from in_channels (= 16 + 16 + K) to the model's
        hidden dimension. Weights are initialized randomly; this is the only
        non-pretrained component.

        Args:
            in_channels: Total number of input channels.
        """
        ...

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_y: torch.Tensor,
        mask: torch.Tensor,
        null_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the velocity field v_theta.

        Concatenates [z_t, z_y, mask] along channel dim, passes through the
        DiT with null text conditioning, and returns the predicted velocity.

        Args:
            z_t: Noisy latent, shape (B, 16, H', W').
            t: Timestep, shape (B,) in [0, 1].
            z_y: Corrupted image latent, shape (B, 16, H', W').
            mask: Downsampled multi-channel damage mask, shape (B, K, H', W').
            null_emb: Precomputed null text embedding, shape (1, seq_len, dim).

        Returns:
            Predicted velocity, shape (B, 16, H', W').
        """
        ...

    def get_trainable_params(self) -> list:
        """Return list of parameter groups for the optimizer.

        img_in parameters may use a higher learning rate than the
        pretrained backbone.

        Returns:
            List of param group dicts for torch optimizer.
        """
        ...
