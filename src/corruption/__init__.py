"""Synthetic corruption module C(x) -> (y, M).

Given a clean image x, produces a degraded image y and a multi-channel
damage mask M in [0,1]^{K x H x W}.

Corruption types (K=8 channels, in order):
  0: cracks       -- Voronoi craquelure and/or linear structural cracks
  1: paint_loss   -- flaking to substrate (crack-associated + user-region + edge-peeling)
  2: yellowing    -- varnish yellowing (CIELAB a/b shift)
  3: stains       -- water stains with tide lines and gravity-driven drip patterns
  4: fading       -- photochemical bleaching / desaturation
  5: bloom        -- haze / bloom from degraded varnish
  6: deposits     -- grime / soot / salt efflorescence with corner/edge accumulation
  7: scratches    -- surface scratches / abrasion marks

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
