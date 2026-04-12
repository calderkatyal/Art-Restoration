"""Preset mask generators for individual and multi-degradation scenarios.

Each preset function takes (H, W, generator) and returns a dict mapping
corruption channel names to (H, W) float32 masks in [0, 1].
"""

import math
import torch
from typing import Dict, Optional, List


def _paint_blob(mask: torch.Tensor, cx: int, cy: int, radius: float,
                strength: float, generator: torch.Generator = None):
    """Paint a soft circular blob onto a mask (in-place, additive with clamp)."""
    H, W = mask.shape
    ri = int(math.ceil(radius))
    x0, x1 = max(0, cx - ri), min(W - 1, cx + ri)
    y0, y1 = max(0, cy - ri), min(H - 1, cy + ri)

    ys = torch.arange(y0, y1 + 1, device=mask.device).float()
    xs = torch.arange(x0, x1 + 1, device=mask.device).float()
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    dx = xx - cx
    dy = yy - cy
    d2 = dx**2 + dy**2
    r2 = radius ** 2
    in_circle = d2 <= r2

    falloff = (1.0 - (d2.sqrt() / radius)).clamp(min=0)
    noise = 0.7 + 0.6 * torch.rand(yy.shape, generator=generator, device=mask.device)
    value = strength * falloff * noise * in_circle.float()

    mask[y0:y1+1, x0:x1+1] = (mask[y0:y1+1, x0:x1+1] + value).clamp(max=1.0)


def _fill(masks: Dict[str, torch.Tensor], corr: str, strength: float):
    """Fill entire mask with a uniform strength."""
    masks[corr].fill_(strength)


def _blobs(masks: Dict[str, torch.Tensor], corr: str,
           count: int, min_r: float, max_r: float, strength: float,
           H: int, W: int, generator: torch.Generator = None):
    """Paint random blobs onto a mask."""
    for _ in range(count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = min_r + torch.rand(1, generator=generator).item() * (max_r - min_r)
        _paint_blob(masks[corr], cx, cy, r, strength, generator)


def _correlated_damage(masks: Dict[str, torch.Tensor],
                       count: int, min_r: float, max_r: float,
                       crack_str: float, loss_str: float,
                       H: int, W: int, generator: torch.Generator = None):
    """Paint correlated crack + paint_loss blobs at same locations."""
    for _ in range(count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = min_r + torch.rand(1, generator=generator).item() * (max_r - min_r)
        _paint_blob(masks['cracks'], cx, cy, r * 1.4, crack_str, generator)
        _paint_blob(masks['paint_loss'], cx, cy, r * 0.7, loss_str, generator)


CHANNEL_NAMES = ['cracks', 'paint_loss', 'yellowing', 'stains', 'fading', 'bloom', 'deposits']


def _empty_masks(H: int, W: int, device: torch.device = None) -> Dict[str, torch.Tensor]:
    """Create empty mask dict."""
    return {name: torch.zeros(H, W, device=device) for name in CHANNEL_NAMES}


# ---------------------------------------------------------------------------
# Individual degradation presets (one corruption type each)
# ---------------------------------------------------------------------------

def preset_individual_cracks(H: int, W: int, generator: torch.Generator = None,
                             device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    has_global = torch.rand(1, generator=generator).item() > 0.5
    if has_global:
        _fill(masks, 'cracks', 0.2 + torch.rand(1, generator=generator).item() * 0.3)
    count = 3 + int(torch.randint(0, 5, (1,), generator=generator).item())
    for _ in range(count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.05 + torch.rand(1, generator=generator).item() * 0.2)
        _paint_blob(masks['cracks'], cx, cy, r,
                    0.4 + torch.rand(1, generator=generator).item() * 0.5, generator)
    return masks


def preset_individual_paint_loss(H: int, W: int, generator: torch.Generator = None,
                                 device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    count = 4 + int(torch.randint(0, 8, (1,), generator=generator).item())
    for _ in range(count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.01 + torch.rand(1, generator=generator).item() * 0.06)
        _paint_blob(masks['paint_loss'], cx, cy, r,
                    0.5 + torch.rand(1, generator=generator).item() * 0.5, generator)
    # Correlated cracks
    for _ in range(int(count * 0.6)):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.02 + torch.rand(1, generator=generator).item() * 0.08)
        _paint_blob(masks['cracks'], cx, cy, r,
                    0.3 + torch.rand(1, generator=generator).item() * 0.3, generator)
    return masks


def preset_individual_yellowing(H: int, W: int, generator: torch.Generator = None,
                                device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    _fill(masks, 'yellowing', 0.25 + torch.rand(1, generator=generator).item() * 0.45)
    count = 2 + int(torch.randint(0, 3, (1,), generator=generator).item())
    for _ in range(count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.1 + torch.rand(1, generator=generator).item() * 0.2)
        _paint_blob(masks['yellowing'], cx, cy, r,
                    0.2 + torch.rand(1, generator=generator).item() * 0.2, generator)
    return masks


def preset_individual_stains(H: int, W: int, generator: torch.Generator = None,
                             device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    count = 2 + int(torch.randint(0, 5, (1,), generator=generator).item())
    for _ in range(count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.04 + torch.rand(1, generator=generator).item() * 0.14)
        _paint_blob(masks['stains'], cx, cy, r,
                    0.4 + torch.rand(1, generator=generator).item() * 0.5, generator)
    return masks


def preset_individual_fading(H: int, W: int, generator: torch.Generator = None,
                             device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    if torch.rand(1, generator=generator).item() > 0.4:
        _fill(masks, 'fading', 0.3 + torch.rand(1, generator=generator).item() * 0.5)
    count = 2 + int(torch.randint(0, 4, (1,), generator=generator).item())
    for _ in range(count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.1 + torch.rand(1, generator=generator).item() * 0.25)
        _paint_blob(masks['fading'], cx, cy, r,
                    0.4 + torch.rand(1, generator=generator).item() * 0.5, generator)
    return masks


def preset_individual_bloom(H: int, W: int, generator: torch.Generator = None,
                            device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    count = 2 + int(torch.randint(0, 4, (1,), generator=generator).item())
    for _ in range(count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.06 + torch.rand(1, generator=generator).item() * 0.18)
        _paint_blob(masks['bloom'], cx, cy, r,
                    0.4 + torch.rand(1, generator=generator).item() * 0.5, generator)
    return masks


def preset_individual_deposits(H: int, W: int, generator: torch.Generator = None,
                               device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    if torch.rand(1, generator=generator).item() > 0.5:
        _fill(masks, 'deposits', 0.1 + torch.rand(1, generator=generator).item() * 0.2)
    count = 3 + int(torch.randint(0, 5, (1,), generator=generator).item())
    for _ in range(count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.05 + torch.rand(1, generator=generator).item() * 0.15)
        _paint_blob(masks['deposits'], cx, cy, r,
                    0.3 + torch.rand(1, generator=generator).item() * 0.6, generator)
    return masks


INDIVIDUAL_PRESETS = {
    'cracks': preset_individual_cracks,
    'paint_loss': preset_individual_paint_loss,
    'yellowing': preset_individual_yellowing,
    'stains': preset_individual_stains,
    'fading': preset_individual_fading,
    'bloom': preset_individual_bloom,
    'deposits': preset_individual_deposits,
}


# ---------------------------------------------------------------------------
# Multiple degradation presets (10 common real-world patterns)
# ---------------------------------------------------------------------------

def preset_light_aging(H: int, W: int, generator: torch.Generator = None,
                       device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    _fill(masks, 'yellowing', 0.2 + torch.rand(1, generator=generator).item() * 0.15)
    _fill(masks, 'deposits', 0.08 + torch.rand(1, generator=generator).item() * 0.08)
    _blobs(masks, 'cracks', 2 + int(torch.randint(0, 3, (1,), generator=generator).item()),
           W * 0.12, W * 0.3, 0.2, H, W, generator)
    _blobs(masks, 'bloom', 1 + int(torch.randint(0, 2, (1,), generator=generator).item()),
           W * 0.1, W * 0.2, 0.1, H, W, generator)
    return masks


def preset_heavy_craquelure(H: int, W: int, generator: torch.Generator = None,
                            device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    _fill(masks, 'cracks', 0.6 + torch.rand(1, generator=generator).item() * 0.25)
    _fill(masks, 'yellowing', 0.3 + torch.rand(1, generator=generator).item() * 0.15)
    _blobs(masks, 'deposits', 4 + int(torch.randint(0, 4, (1,), generator=generator).item()),
           W * 0.06, W * 0.15, 0.25, H, W, generator)
    _correlated_damage(masks, 8 + int(torch.randint(0, 8, (1,), generator=generator).item()),
                       W * 0.02, W * 0.08, 0.6, 0.5, H, W, generator)
    return masks


def preset_water_damage(H: int, W: int, generator: torch.Generator = None,
                        device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    n_stains = 3 + int(torch.randint(0, 4, (1,), generator=generator).item())
    for _ in range(n_stains):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.06 + torch.rand(1, generator=generator).item() * 0.14)
        _paint_blob(masks['stains'], cx, cy, r, 0.5 + torch.rand(1, generator=generator).item() * 0.4, generator)
        _paint_blob(masks['bloom'], cx, cy, r * 0.9, 0.3 + torch.rand(1, generator=generator).item() * 0.2, generator)
        _paint_blob(masks['deposits'], cx, cy, r * 0.5, 0.3 + torch.rand(1, generator=generator).item() * 0.3, generator)
        if torch.rand(1, generator=generator).item() > 0.4:
            _paint_blob(masks['paint_loss'], cx, cy, r * 0.2, 0.3 + torch.rand(1, generator=generator).item() * 0.3, generator)
    _fill(masks, 'yellowing', 0.1 + torch.rand(1, generator=generator).item() * 0.15)
    return masks


def preset_sun_faded(H: int, W: int, generator: torch.Generator = None,
                     device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    _fill(masks, 'fading', 0.5 + torch.rand(1, generator=generator).item() * 0.35)
    _blobs(masks, 'fading', 2 + int(torch.randint(0, 3, (1,), generator=generator).item()),
           W * 0.15, W * 0.3, 0.3, H, W, generator)
    _fill(masks, 'bloom', 0.2 + torch.rand(1, generator=generator).item() * 0.2)
    _fill(masks, 'yellowing', 0.15 + torch.rand(1, generator=generator).item() * 0.15)
    _blobs(masks, 'cracks', 2, W * 0.1, W * 0.25, 0.15, H, W, generator)
    return masks


def preset_smoke_damage(H: int, W: int, generator: torch.Generator = None,
                        device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    _fill(masks, 'deposits', 0.4 + torch.rand(1, generator=generator).item() * 0.3)
    _blobs(masks, 'deposits', 3 + int(torch.randint(0, 4, (1,), generator=generator).item()),
           W * 0.1, W * 0.25, 0.4, H, W, generator)
    _fill(masks, 'yellowing', 0.35 + torch.rand(1, generator=generator).item() * 0.25)
    _blobs(masks, 'bloom', 2, W * 0.1, W * 0.2, 0.15, H, W, generator)
    _blobs(masks, 'cracks', 2, W * 0.08, W * 0.18, 0.2, H, W, generator)
    return masks


def preset_neglected_storage(H: int, W: int, generator: torch.Generator = None,
                             device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    # Foxing: many small spots
    fox_count = 10 + int(torch.randint(0, 15, (1,), generator=generator).item())
    for _ in range(fox_count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.005 + torch.rand(1, generator=generator).item() * 0.02)
        _paint_blob(masks['stains'], cx, cy, r,
                    0.4 + torch.rand(1, generator=generator).item() * 0.4, generator)
    # Larger mold patches
    _blobs(masks, 'stains', 2 + int(torch.randint(0, 3, (1,), generator=generator).item()),
           W * 0.05, W * 0.12, 0.4, H, W, generator)
    _fill(masks, 'deposits', 0.15 + torch.rand(1, generator=generator).item() * 0.2)
    _blobs(masks, 'deposits', 3, W * 0.08, W * 0.15, 0.3, H, W, generator)
    _blobs(masks, 'cracks', 3, W * 0.1, W * 0.2, 0.2, H, W, generator)
    _fill(masks, 'yellowing', 0.15 + torch.rand(1, generator=generator).item() * 0.15)
    return masks


def preset_heat_damage(H: int, W: int, generator: torch.Generator = None,
                       device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    _fill(masks, 'cracks', 0.5 + torch.rand(1, generator=generator).item() * 0.3)
    _fill(masks, 'yellowing', 0.3 + torch.rand(1, generator=generator).item() * 0.25)
    blisters = 6 + int(torch.randint(0, 8, (1,), generator=generator).item())
    for _ in range(blisters):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.01 + torch.rand(1, generator=generator).item() * 0.04)
        _paint_blob(masks['paint_loss'], cx, cy, r,
                    0.5 + torch.rand(1, generator=generator).item() * 0.4, generator)
        _paint_blob(masks['cracks'], cx, cy, r * 1.5,
                    0.4 + torch.rand(1, generator=generator).item() * 0.3, generator)
    _blobs(masks, 'bloom', 2 + int(torch.randint(0, 3, (1,), generator=generator).item()),
           W * 0.08, W * 0.18, 0.25, H, W, generator)
    return masks


def preset_flood_damage(H: int, W: int, generator: torch.Generator = None,
                        device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    flood_y = H * (0.3 + torch.rand(1, generator=generator).item() * 0.4)
    n_zones = 3 + int(torch.randint(0, 3, (1,), generator=generator).item())
    for _ in range(n_zones):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(flood_y + (torch.rand(1, generator=generator).item() - 0.3) * H * 0.3)
        cy = max(0, min(H - 1, cy))
        r = W * (0.08 + torch.rand(1, generator=generator).item() * 0.15)
        _paint_blob(masks['stains'], cx, cy, r, 0.6 + torch.rand(1, generator=generator).item() * 0.3, generator)
        _paint_blob(masks['bloom'], cx, cy, r * 0.8, 0.3 + torch.rand(1, generator=generator).item() * 0.3, generator)
        _paint_blob(masks['deposits'], cx, cy, r * 0.6, 0.3 + torch.rand(1, generator=generator).item() * 0.3, generator)
    loss_count = 5 + int(torch.randint(0, 6, (1,), generator=generator).item())
    for _ in range(loss_count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(flood_y + (torch.rand(1, generator=generator).item() - 0.2) * H * 0.25)
        cy = max(0, min(H - 1, cy))
        r = W * (0.01 + torch.rand(1, generator=generator).item() * 0.04)
        _paint_blob(masks['paint_loss'], cx, cy, r, 0.4 + torch.rand(1, generator=generator).item() * 0.5, generator)
    _fill(masks, 'yellowing', 0.15 + torch.rand(1, generator=generator).item() * 0.15)
    return masks


def preset_museum_wear(H: int, W: int, generator: torch.Generator = None,
                       device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    _fill(masks, 'yellowing', 0.15 + torch.rand(1, generator=generator).item() * 0.15)
    _fill(masks, 'deposits', 0.06 + torch.rand(1, generator=generator).item() * 0.08)
    _fill(masks, 'fading', 0.1 + torch.rand(1, generator=generator).item() * 0.15)
    _blobs(masks, 'cracks', 3 + int(torch.randint(0, 3, (1,), generator=generator).item()),
           W * 0.1, W * 0.25, 0.15, H, W, generator)
    _blobs(masks, 'bloom', 1 + int(torch.randint(0, 2, (1,), generator=generator).item()),
           W * 0.08, W * 0.15, 0.1, H, W, generator)
    # Edge wear
    for _ in range(4):
        edge = torch.rand(1, generator=generator).item()
        cx = int(torch.randint(0, max(1, int(W * 0.1)), (1,), generator=generator).item()) if edge < 0.5 else \
             W - int(torch.randint(0, max(1, int(W * 0.1)), (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.01 + torch.rand(1, generator=generator).item() * 0.03)
        _paint_blob(masks['paint_loss'], cx, cy, r,
                    0.3 + torch.rand(1, generator=generator).item() * 0.3, generator)
    return masks


def preset_salt_efflorescence(H: int, W: int, generator: torch.Generator = None,
                              device: torch.device = None) -> Dict[str, torch.Tensor]:
    masks = _empty_masks(H, W, device)
    _fill(masks, 'deposits', 0.3 + torch.rand(1, generator=generator).item() * 0.3)
    _blobs(masks, 'deposits', 5 + int(torch.randint(0, 5, (1,), generator=generator).item()),
           W * 0.05, W * 0.15, 0.5, H, W, generator)
    _blobs(masks, 'cracks', 4 + int(torch.randint(0, 4, (1,), generator=generator).item()),
           W * 0.08, W * 0.2, 0.3, H, W, generator)
    loss_count = 5 + int(torch.randint(0, 7, (1,), generator=generator).item())
    for _ in range(loss_count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        r = W * (0.01 + torch.rand(1, generator=generator).item() * 0.03)
        _paint_blob(masks['paint_loss'], cx, cy, r,
                    0.4 + torch.rand(1, generator=generator).item() * 0.4, generator)
        _paint_blob(masks['cracks'], cx, cy, r * 1.5,
                    0.3 + torch.rand(1, generator=generator).item() * 0.3, generator)
    _fill(masks, 'yellowing', 0.1 + torch.rand(1, generator=generator).item() * 0.1)
    return masks


MULTI_PRESETS = {
    'light_aging': preset_light_aging,
    'heavy_craquelure': preset_heavy_craquelure,
    'water_damage': preset_water_damage,
    'sun_faded': preset_sun_faded,
    'smoke_damage': preset_smoke_damage,
    'neglected_storage': preset_neglected_storage,
    'heat_damage': preset_heat_damage,
    'flood_damage': preset_flood_damage,
    'museum_wear': preset_museum_wear,
    'salt_efflorescence': preset_salt_efflorescence,
}
