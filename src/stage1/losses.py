"""Adversarial, content, and diversity losses for Stage 1 training."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def hinge_d_loss(real_logits: List[torch.Tensor], fake_logits: List[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for the discriminator (averaged across scales)."""
    loss = 0.0
    for r, f in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
    return loss / max(len(real_logits), 1)


def hinge_g_loss(fake_logits: List[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for the generator (averaged across scales)."""
    loss = 0.0
    for f in fake_logits:
        loss = loss - f.mean()
    return loss / max(len(fake_logits), 1)


def diversity_loss(
    fake_a: torch.Tensor,
    fake_b: torch.Tensor,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Mode-seeking diversity loss (DSGAN-style).

    Encourages G to use the noise vector by maximizing ``||G(c,z_a) - G(c,z_b)|| /
    ||z_a - z_b||``. Returned as a *negated* ratio so it can be minimized.
    """
    img_diff = (fake_a - fake_b).abs().flatten(1).mean(dim=1)
    z_diff = (z_a - z_b).abs().flatten(1).mean(dim=1) + eps
    return -(img_diff / z_diff).mean()


def content_loss(
    lpips_model,
    fake: torch.Tensor,
    clean: torch.Tensor,
    layer_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """Shallow LPIPS between ``fake`` and ``clean`` (both expected in ``[-1, 1]``).

    If ``layer_indices`` is provided, only those LPIPS feature layers contribute,
    so the loss penalizes only low-level structural drift while letting G change
    color / texture freely. With ``layer_indices=None`` the full LPIPS distance
    is used.
    """
    if layer_indices is None:
        return lpips_model(fake, clean).mean()
    feats_fake = lpips_model.net.forward(fake)
    feats_clean = lpips_model.net.forward(clean)
    total = 0.0
    for idx in layer_indices:
        diff = (feats_fake[idx] - feats_clean[idx]).pow(2)
        total = total + diff.mean()
    return total / max(len(layer_indices), 1)
