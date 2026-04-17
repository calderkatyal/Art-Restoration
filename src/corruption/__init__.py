"""Synthetic corruption module C(x) -> (y, M).

Given a clean image x, produces a degraded image y and a multi-channel
BINARY damage mask M in {0, 1}^{K x H x W} — each channel marks the
exact pixels modified by that effect (channels may overlap).

Corruption channels (K=8, each a single visual appearance):
  0: craquelure   -- Voronoi tessellation cracks from paint shrinkage
  1: rip_tear     -- physical canvas tear with exposed substrate (local only)
  2: paint_loss   -- blob-shaped paint loss with substrate reveal (local only)
  3: yellowing    -- varnish yellowing (CIELAB a/b shift)
  4: fading       -- photochemical desaturation / bleach
  5: bloom        -- milky multi-radius haze from degraded varnish
  6: deposits     -- grime / soot darkening veil
  7: scratches    -- thin linear abrasion marks (local only)

Per-sample mask generation is driven by src/corruption/configs/default.yaml.
downsample_mask() converts M from pixel to latent resolution via max-pool.
"""

from .module import CorruptionModule, downsample_mask, PIPELINE_ORDER, EFFECT_FNS
from .presets import CHANNEL_NAMES, NUM_CHANNELS

__all__ = [
    "CorruptionModule",
    "downsample_mask",
    "CHANNEL_NAMES",
    "NUM_CHANNELS",
    "PIPELINE_ORDER",
    "EFFECT_FNS",
]
