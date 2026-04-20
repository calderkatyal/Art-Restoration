"""Synthetic corruption module C(x) -> (y, M).

Given a clean image x, produces a degraded image y and an 8-channel BINARY damage
mask M in {0, 1}^{8 x H x W}. The first seven channels are per-effect hull ROIs
(see ``CHANNEL_NAMES``); the eighth is their pixel-wise union (max). Optional
training-time dropout can zero the first seven while keeping the union channel.

Per-sample mask generation is driven by ``src/corruption/configs/default.yaml``,
with optional overrides merged from the training YAML (e.g. ``dropout_prob``).
``downsample_mask()`` converts M from pixel to latent resolution via max-pool.
"""

from .module import (
    CorruptionModule,
    OUTPUT_MASK_CHANNELS,
    downsample_mask,
    PIPELINE_ORDER,
    EFFECT_FNS,
)
from .presets import CHANNEL_NAMES, NUM_CHANNELS

__all__ = [
    "CorruptionModule",
    "OUTPUT_MASK_CHANNELS",
    "downsample_mask",
    "CHANNEL_NAMES",
    "NUM_CHANNELS",
    "PIPELINE_ORDER",
    "EFFECT_FNS",
]
