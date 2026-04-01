"""Inference: ODE integration with data-consistency projection.

Integrates the learned velocity field from t=0 to t=1, applying hard
data-consistency constraints at each step to preserve intact regions.
"""

import torch
from typing import Optional

from .model import RestorationDiT
from .vae import FluxVAE


def sample(
    model: RestorationDiT,
    vae: FluxVAE,
    corrupted_image: torch.Tensor,
    mask: torch.Tensor,
    null_emb: torch.Tensor,
    num_steps: int = 50,
    guidance_scale: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Restore a corrupted image via ODE integration.

    Encodes the corrupted image, samples z_0 ~ N(0,I), and integrates
    the velocity field from t=0 to t=1 using Euler method with
    data-consistency projection at each step.

    Args:
        model: Trained RestorationDiT.
        vae: Frozen FluxVAE for encode/decode.
        corrupted_image: Degraded image tensor (B, 3, H, W) in [0, 1].
        mask: Multi-channel damage mask (B, K, H, W), binary, pixel resolution.
        null_emb: Precomputed null text embedding.
        num_steps: Number of Euler integration steps.
        guidance_scale: CFG scale (1.0 = no guidance).
        device: Device.

    Returns:
        Restored image tensor (B, 3, H, W) in [0, 1].
    """
    ...


def data_consistency_step(
    z_t: torch.Tensor,
    z_y: torch.Tensor,
    mask_latent: torch.Tensor,
) -> torch.Tensor:
    """Apply hard data-consistency projection in latent space.

    Replaces intact (non-damaged) latent regions with the corrupted
    image's latent, preserving unmasked content:
        z_t <- m_intact * z_y + (1 - m_intact) * z_t

    where m_intact = 1 - max_k(M'_k).

    Args:
        z_t: Current latent state, shape (B, C, H', W').
        z_y: Corrupted image latent, shape (B, C, H', W').
        mask_latent: Downsampled mask, shape (B, K, H', W').

    Returns:
        Projected latent, shape (B, C, H', W').
    """
    ...


def compute_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute PSNR between prediction and target, optionally masked.

    Args:
        prediction: Predicted image (B, C, H, W) in [0, 1].
        target: Ground truth image (B, C, H, W) in [0, 1].
        mask: If provided, binary mask (B, 1, H, W) to compute PSNR
              only over damaged pixels. None = full image PSNR.

    Returns:
        PSNR value in dB (averaged over batch).
    """
    ...
