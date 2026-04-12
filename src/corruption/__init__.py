"""Synthetic corruption module C(x) -> (y, M).

Given a clean image x, produces a degraded image y and a multi-channel
damage mask M ∈ [0,1]^{K×H×W}.

Corruption types (K=7 channels, in order):
  0: cracks       — Voronoi craquelure
  1: paint_loss   — flaking to substrate (crack-associated + user-region)
  2: yellowing    — varnish yellowing (CIELAB a/b shift)
  3: stains       — water stains with tide lines
  4: fading       — photochemical bleaching / desaturation
  5: bloom        — haze / bloom from degraded varnish
  6: deposits     — grime / soot / salt efflorescence

downsample_mask() converts M from pixel resolution to latent resolution
using max-pooling (kernel = stride = spatial_compression), so any
damaged pixel within a block survives in M'.
"""

from .module import CorruptionModule, downsample_mask
from .presets import INDIVIDUAL_PRESETS, MULTI_PRESETS

__all__ = [
    "CorruptionModule",
    "downsample_mask",
    "INDIVIDUAL_PRESETS",
    "MULTI_PRESETS",
]
