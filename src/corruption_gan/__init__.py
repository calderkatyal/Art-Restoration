"""Learned corruption module: pix2pix GAN that maps clean → realistically damaged paintings.

Trained on the Kaggle ``pes1ug22am047/damaged-and-undamaged-artworks`` paired dataset.
After training, the generator replaces the hand-crafted CorruptionModule, producing
damage patterns with real-world statistics for use in Stage 2 restoration training.
"""

from .model import UNetGenerator, PatchDiscriminator

__all__ = ["UNetGenerator", "PatchDiscriminator"]
