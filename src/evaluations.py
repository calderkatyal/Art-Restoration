"""Quantitative evaluation metrics for restoration quality.

Metrics:
    compute_psnr      — full-image or masked-region PSNR (dB)
    compute_psnr_stratified — per-damage-type PSNR breakdown

All functions operate on (B, C, H, W) float32 tensors in [0, 1].
Masks are (B, K, H, W) binary float32 at pixel resolution,
one channel per damage type in degradation.damage_types order.
"""

import torch
from typing import Dict, List, Optional


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
        PSNR in dB averaged over the batch.
        float('inf') if MSE == 0. float('nan') if mask has no damaged pixels.
    """
    ...


def compute_psnr_stratified(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    damage_types: List[str],
) -> Dict[str, float]:
    """Compute per-damage-type PSNR over each mask channel independently.

    For each channel k, computes PSNR restricted to pixels where mask[:, k] == 1,
    regardless of whether other channels are also active at those pixels.

    Args:
        prediction:   (B, C, H, W) float32 in [0, 1].
        target:       (B, C, H, W) float32 in [0, 1].
        mask:         (B, K, H, W) binary float32, K == len(damage_types).
        damage_types: List of K damage type names, e.g. ["crack", "paint_loss", ...].

    Returns:
        Dict mapping damage type name → PSNR in dB.
        Value is float('nan') if that channel has no damaged pixels in the batch.
    """
    ...
