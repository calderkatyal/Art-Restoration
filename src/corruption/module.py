"""Main corruption module: C(x) -> (y, M).

Takes a clean image and produces a corrupted version with per-channel
damage masks, using the preset system to generate realistic mask
patterns and the effect functions to apply pixel-level corruption.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List

from .effects import (
    apply_cracks, apply_linear_cracks, apply_paint_loss_crack,
    apply_paint_loss_region, apply_yellowing, apply_stains, apply_fading,
    apply_bloom, apply_deposits, apply_scratches,
)
from .presets import (
    CHANNEL_NAMES, INDIVIDUAL_PRESETS, MULTI_PRESETS, _empty_masks,
)


# Channel index mapping
CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNEL_NAMES)}
NUM_CHANNELS = len(CHANNEL_NAMES)


class CorruptionModule:
    """Non-learned stochastic corruption pipeline.

    Given a clean image x (3, H, W) in [0, 1], produces:
      - corrupted (3, H, W) in [0, 1]
      - mask (K, H, W) in [0, 1] where K=8 damage channels

    Corruption pipeline order:
      yellowing -> fading -> stains -> bloom -> deposits -> scratches -> cracks -> paint_loss

    Usage:
        config = CorruptionConfig()
        module = CorruptionModule(config)
        corrupted, mask = module(clean_image)
    """

    def __init__(self, config: 'CorruptionConfig'):
        self.config = config

    def __call__(
        self,
        image: torch.Tensor,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random corruption to a clean image.

        Args:
            image: (3, H, W) float32 in [0, 1].
            seed: Optional seed for reproducibility.

        Returns:
            corrupted: (3, H, W) float32 in [0, 1].
            mask: (K, H, W) float32 in [0, 1] — per-channel damage intensity mask.
        """
        C, H, W = image.shape
        assert C == 3, f"Expected 3-channel image, got {C}"
        device = image.device

        generator = torch.Generator(device='cpu')
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.manual_seed(torch.randint(0, 2**31, (1,)).item())

        # Decide: individual vs multi-degradation preset
        cfg = self.config
        use_individual = torch.rand(1, generator=generator).item() < cfg.individual_prob

        if use_individual:
            # Pick one individual preset
            weights = [cfg.individual_presets.get(name, 1.0) for name in INDIVIDUAL_PRESETS]
            total = sum(weights)
            r = torch.rand(1, generator=generator).item() * total
            cumsum = 0
            chosen = list(INDIVIDUAL_PRESETS.keys())[0]
            for name, w in zip(INDIVIDUAL_PRESETS.keys(), weights):
                cumsum += w
                if r <= cumsum:
                    chosen = name
                    break
            masks = INDIVIDUAL_PRESETS[chosen](H, W, generator=generator, device=device)
        else:
            # Pick one multi-degradation preset
            weights = [cfg.multi_presets.get(name, 1.0) for name in MULTI_PRESETS]
            total = sum(weights)
            r = torch.rand(1, generator=generator).item() * total
            cumsum = 0
            chosen = list(MULTI_PRESETS.keys())[0]
            for name, w in zip(MULTI_PRESETS.keys(), weights):
                cumsum += w
                if r <= cumsum:
                    chosen = name
                    break
            masks = MULTI_PRESETS[chosen](H, W, generator=generator, device=device)

        # Apply severity scaling from config
        for name in CHANNEL_NAMES:
            if masks[name].max() > 0:
                scale = cfg.severity_scale.get(name, 1.0)
                masks[name] = (masks[name] * scale).clamp(0, 1)

        # Apply effects in correct pipeline order
        out = image.clone()

        # 1. Yellowing
        if masks['yellowing'].max() > 0.01:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(torch.randint(0, 2**31, (1,), generator=generator).item())
            out = apply_yellowing(out, masks['yellowing'], generator=gen)

        # 2. Fading
        if masks['fading'].max() > 0.01:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(torch.randint(0, 2**31, (1,), generator=generator).item())
            out = apply_fading(out, masks['fading'], generator=gen)

        # 3. Stains
        if masks['stains'].max() > 0.01:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(torch.randint(0, 2**31, (1,), generator=generator).item())
            out = apply_stains(out, masks['stains'], generator=gen)

        # 4. Bloom
        if masks['bloom'].max() > 0.01:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(torch.randint(0, 2**31, (1,), generator=generator).item())
            out = apply_bloom(out, masks['bloom'], generator=gen)

        # 5. Deposits
        if masks['deposits'].max() > 0.01:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(torch.randint(0, 2**31, (1,), generator=generator).item())
            out = apply_deposits(out, masks['deposits'], generator=gen)

        # 6. Scratches
        if masks['scratches'].max() > 0.01:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(torch.randint(0, 2**31, (1,), generator=generator).item())
            out = apply_scratches(out, masks['scratches'], generator=gen)

        # 7. Cracks + crack-associated paint loss
        crack_binary = torch.zeros(H, W, device=device)
        if masks['cracks'].max() > 0.01:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(torch.randint(0, 2**31, (1,), generator=generator).item())
            out, crack_binary = apply_cracks(out, masks['cracks'], generator=gen)

        # 8. Paint loss (crack-associated)
        if masks['paint_loss'].max() > 0.01 and crack_binary.max() > 0:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(torch.randint(0, 2**31, (1,), generator=generator).item())
            out = apply_paint_loss_crack(out, masks['paint_loss'], crack_binary, generator=gen)

        # 9. Paint loss (region-based, for areas not covered by cracks)
        if masks['paint_loss'].max() > 0.01:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(torch.randint(0, 2**31, (1,), generator=generator).item())
            out = apply_paint_loss_region(out, masks['paint_loss'], generator=gen)

        # Stack masks into (K, H, W) tensor
        mask_tensor = torch.stack([masks[name] for name in CHANNEL_NAMES], dim=0)

        return out.clamp(0, 1), mask_tensor

    def corrupt_batch(
        self,
        images: torch.Tensor,
        seeds: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply corruption to a batch of images.

        Args:
            images: (B, 3, H, W) float32 in [0, 1].
            seeds: Optional list of B seeds.

        Returns:
            corrupted: (B, 3, H, W).
            masks: (B, K, H, W).
        """
        B = images.shape[0]
        corrupted_list = []
        mask_list = []
        for i in range(B):
            seed = seeds[i] if seeds is not None else None
            c, m = self(images[i], seed=seed)
            corrupted_list.append(c)
            mask_list.append(m)
        return torch.stack(corrupted_list), torch.stack(mask_list)


def downsample_mask(mask: torch.Tensor, factor: int = 16) -> torch.Tensor:
    """Downsample pixel-resolution mask to latent resolution via max pooling.

    A damaged pixel in any position within a (factor × factor) block
    propagates to the corresponding latent-resolution cell.

    Args:
        mask: (K, H, W) or (B, K, H, W) float32.
        factor: Spatial compression factor (16 for FLUX.2 VAE).

    Returns:
        Downsampled mask (K, H//factor, W//factor) or (B, K, H//factor, W//factor).
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)  # (1, K, H, W)
        return F.max_pool2d(mask, kernel_size=factor, stride=factor).squeeze(0)
    else:
        return F.max_pool2d(mask, kernel_size=factor, stride=factor)
