"""Stage 1: damage generator G and PatchGAN discriminator D.

Trains an unpaired image-to-image generator that maps clean paintings to
damaged-looking paintings whose distribution matches a real damaged-image set
(ARTeFACT, MuralDH, Real-Old). Once converged, ``G`` is frozen and used by the
Stage 2 restoration training to produce realistic ``(damaged, clean)`` pairs
for supervised flow-matching training.
"""

from .generator import DamageGenerator
from .discriminator import PatchDiscriminator
from .losses import (
    hinge_d_loss,
    hinge_g_loss,
    diversity_loss,
    content_loss,
)
from .dataset import (
    CleanImageDataset,
    DamagedImageDataset,
    build_clean_loader,
    build_damaged_loader,
)

__all__ = [
    "DamageGenerator",
    "PatchDiscriminator",
    "hinge_d_loss",
    "hinge_g_loss",
    "diversity_loss",
    "content_loss",
    "CleanImageDataset",
    "DamagedImageDataset",
    "build_clean_loader",
    "build_damaged_loader",
]
