"""Inference via Euler ODE integration with hard data-consistency projection.

At each integration step:
    1. Predict velocity:  vel = v_θ(z_t, t | z_y, M')
    2. Euler step:        z_t ← z_t + (t_prev - t_curr) * vel
    3. Data consistency:  z_t ← m_intact ⊙ z_y  +  m_dam ⊙ z_t
       where  m_dam    = max_k(M'_k)   (any damaged channel → 1)
              m_intact = 1 - m_dam

Timestep schedule: FLUX.2 empirical SNR-shifted schedule from
flux2/sampling.py get_schedule(num_steps, image_seq_len).

Usage:
    python -m src.inference --checkpoint checkpoints/final.pt --input damaged.png --output restored.png \
        [--config configs/default.yaml] [--damage crack paint_loss] [--steps 50] [--device cuda]

Arguments:
    --checkpoint  Path to trained model .pt file (required).
    --input       Path to damaged input image (required).
    --output      Path to save restored image (required).
    --config      Path to YAML config (default: configs/default.yaml).
    --damage      One or more damage types present in the image:
                      crack | paint_loss | stain | blur | color_shift
                  A full-image mask of ones is set for each named type.
                  Omit to mark all channels as damaged.
    --steps       Number of ODE Euler steps (default: model.num_steps from config).
    --device      Device to run on (default: cuda).
"""

import torch
from typing import Optional

from .model import RestorationDiT
from .vae import FluxVAE
from .corruption import downsample_mask
from .flux2.sampling import get_schedule


@torch.no_grad()
def sample(
    model: RestorationDiT,
    vae: FluxVAE,
    corrupted_image: torch.Tensor,
    mask: torch.Tensor,
    null_emb: torch.Tensor,
    num_steps: int = 50,
    device: str = "cuda",
) -> torch.Tensor:
    """Restore a corrupted image via Euler ODE integration.

    Steps:
        1. z_y = vae.encode(corrupted_image)       → (B, 128, H/16, W/16)
        2. M'  = downsample_mask(mask, factor=16)  → (B, K, H/16, W/16)
        3. z_t ~ N(0, I), same shape as z_y.
        4. timesteps = get_schedule(num_steps, H'*W')  — SNR-shifted schedule.
        5. For each (t_curr, t_prev) in timesteps:
               vel  = model(z_t, t_curr, z_y, M', null_emb)
               z_t ← z_t + (t_prev - t_curr) * vel
               z_t ← data_consistency_step(z_t, z_y, M')
        6. return vae.decode(z_t)                  → (B, 3, H, W) in [0, 1]

    Args:
        model:           Trained RestorationDiT in eval mode.
        vae:             Frozen FluxVAE.
        corrupted_image: (B, 3, H, W) float32 in [0, 1]. H, W divisible by 16.
        mask:            (B, K, H, W) float32 binary at pixel resolution.
        null_emb:        (1, 512, 7680) precomputed null text embedding.
        num_steps:       Number of Euler steps.
        device:          Device string.

    Returns:
        (B, 3, H, W) float32 in [0, 1] — restored image.
    """
    ...


def data_consistency_step(
    z_t: torch.Tensor,
    z_y: torch.Tensor,
    mask_latent: torch.Tensor,
) -> torch.Tensor:
    """Hard data-consistency projection in latent space.

    Preserves intact (non-damaged) latent regions from z_y:
        m_dam    = max_k(M'_k)          (B, 1, H', W')
        m_intact = 1 - m_dam
        z_t     ← m_intact ⊙ z_y  +  m_dam ⊙ z_t

    Args:
        z_t:          Current latent (B, 128, H', W').
        z_y:          Corrupted image latent (B, 128, H', W').
        mask_latent:  Downsampled mask (B, K, H', W') values in {0, 1}.

    Returns:
        Projected latent (B, 128, H', W').
    """
    ...


def compute_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute PSNR (dB) between prediction and target.

    Full-image mode (mask=None):
        MSE = mean over all B×C×H×W elements.

    Masked mode (mask provided):
        MSE = sum of squared errors at damaged pixels
              / (number of damaged pixels × C)
        where damaged pixels are those where max_k(mask_k) == 1.

    Args:
        prediction: (B, C, H, W) float32 in [0, 1].
        target:     (B, C, H, W) float32 in [0, 1].
        mask:       Optional (B, K, H, W) or (B, 1, H, W) binary float32.
                    If None, full-image PSNR is computed.

    Returns:
        PSNR in dB averaged over batch. float('inf') if MSE == 0.
        float('nan') if mask provided but no damaged pixels found.
    """
    ...


if __name__ == "__main__":
    """Restore a damaged image using a trained checkpoint.

    Usage:
        python -m src.inference --checkpoint checkpoints/final.pt --input damaged.png --output restored.png \
            [--config configs/default.yaml] [--damage crack paint_loss] [--steps 50] [--device cuda]
    """
    ...
