"""Quantitative evaluation metrics for restoration quality.

Metrics:
    :func:`compute_psnr` — full-image PSNR or PSNR restricted to damaged pixels
        (union over mask channels when ``K > 1``).
    :func:`compute_psnr_stratified` — per-damage-channel PSNR for ablation / logging.

All functions expect ``(B, C, H, W)`` ``float32`` tensors in ``[0, 1]``. Masks are
``(B, K, H, W)`` binary ``float32`` with one channel per damage type in the same order
as ``cfg.corruption.damage_types`` / the corruption module's channel stack.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def compute_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute PSNR (dB) between ``prediction`` and ``target``.

    **Full-image mode** (``mask is None``):
        ``MSE`` is the mean squared error over all ``B × C × H × W`` elements, then
        ``PSNR = 10 * log10(1 / MSE)`` (peak value 1.0 for images in ``[0,1]``).

    **Masked mode** (``mask`` provided):
        A pixel is **damaged** if ``max_k mask[:,k,h,w] >= 0.5`` when ``K > 1``, else
        if ``mask[:,0] >= 0.5``. ``MSE`` averages squared error over damaged pixels
        across all colour channels. Returns ``nan`` if there are zero damaged pixels
        in the batch slice.

    Args:
        prediction: ``(B, C, H, W)`` restored image.
        target:     ``(B, C, H, W)`` ground truth.
        mask:       Optional ``(B, K, H, W)`` or ``(B, 1, H, W)`` mask.

    Returns:
        Scalar PSNR in decibels (Python ``float``). ``inf`` if ``MSE == 0`` exactly.
    """
    pred = prediction.detach().float()
    tgt = target.detach().float()

    if mask is None:
        mse = F.mse_loss(pred, tgt)
        if mse.item() <= 0.0:
            return float("inf")
        return (10.0 * torch.log10(torch.tensor(1.0, device=mse.device) / mse)).item()

    m = mask.float()
    if m.dim() == 4 and m.shape[1] > 1:
        m = m.max(dim=1, keepdim=True).values
    damaged = (m >= 0.5).float()
    _, c, _, _ = pred.shape
    err2 = (pred - tgt).pow(2) * damaged
    denom = damaged.sum() * float(c)
    if denom <= 0:
        return float("nan")
    mse = err2.sum() / denom
    if mse.item() <= 0.0:
        return float("inf")
    return (10.0 * torch.log10(torch.tensor(1.0, device=mse.device) / mse)).item()


def compute_psnr_stratified(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    damage_types: List[str],
) -> Dict[str, float]:
    """Compute per-damage-type PSNR using each mask channel independently.

    For each channel ``k``, evaluates :func:`compute_psnr` with a single-channel mask
    ``mask[:, k:k+1]``, so the metric only sees pixels where **that** damage type is
    active (even if other channels are also 1 at the same pixel).

    Args:
        prediction:   ``(B, C, H, W)`` in ``[0, 1]``.
        target:       ``(B, C, H, W)`` in ``[0, 1]``.
        mask:         ``(B, K, H, W)`` binary; ``K`` must equal ``len(damage_types)``.
        damage_types: Human-readable names for logging keys.

    Returns:
        Mapping ``damage_type_name → PSNR`` in dB. Value may be ``nan`` if a channel
        has no active pixels in the batch.

    Raises:
        ValueError: If ``mask.shape[1] != len(damage_types)``.
    """
    out: Dict[str, float] = {}
    _, k, _, _ = mask.shape
    if k != len(damage_types):
        raise ValueError(f"mask has {k} channels but {len(damage_types)} damage types")

    for i, name in enumerate(damage_types):
        ch = mask[:, i : i + 1]
        out[name] = compute_psnr(prediction, target, mask=ch)
    return out
