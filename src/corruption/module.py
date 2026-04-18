"""Main corruption pipeline C(x) -> (y, M).

Per-sample stochastic pipeline:

  1. Pick N ~ Uniform[cfg.num_simultaneous.min, cfg.num_simultaneous.max]
     distinct channel names, weighted by each channel's `weight`.
  2. For each chosen channel c:
        - Pick mode ∈ {local, global} among enabled modes (uniform).
        - Pick severity ~ Uniform[c.min_severity, c.max_severity].
        - Generate mask_c accordingly (local = blobs, global = uniform fill).
  3. Apply effects in fixed pipeline order, each consuming its own mask.
  4. For each channel, derive the OUTPUT binary mask as the convex hull
     of pixels the effect actually modified (before/after diff).
  5. Return corrupted image and stacked (K, H, W) binary mask tensor.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull, QhullError
from scipy.ndimage import label as _cc_label
from typing import Tuple, Dict, Optional, List, Any

from .effects import (
    apply_craquelure,
    apply_rip_tear,
    apply_paint_loss,
    apply_yellowing,
    apply_fading,
    apply_deposits,
    apply_scratches,
)
from .presets import (
    CHANNEL_NAMES, NUM_CHANNELS, SHAPE_KIND_BY_CHANNEL,
    generate_global_mask, generate_local_mask,
)


# ---------------------------------------------------------------------------
# Convex-hull mask helpers
# ---------------------------------------------------------------------------

def _affected_pixels(before: torch.Tensor, after: torch.Tensor,
                     threshold: float = 0.01) -> torch.Tensor:
    """(H, W) bool tensor of pixels changed by more than `threshold`."""
    diff = (before - after).abs().mean(dim=0)
    return diff > threshold


def _per_component_hull_mask(pixels: torch.Tensor, H: int, W: int,
                             merge_radius: int = 4) -> torch.Tensor:
    """(H, W) float32 binary mask = UNION of convex hulls of each connected
    component of `pixels`.

    Affected pixels are first dilated by `merge_radius` so that speckle
    within one blob merges into the same component, while spatially
    separate blobs remain distinct. The convex hull is computed
    independently per component and then filled. This avoids the
    "single hull spans every blob" failure mode where a multi-instance
    mask collapses into one giant polygon covering the whole image.
    """
    mask = torch.zeros(H, W, dtype=torch.float32, device=pixels.device)
    if not pixels.any():
        return mask

    r = int(merge_radius)
    dil = pixels.float().unsqueeze(0).unsqueeze(0)
    dil = F.max_pool2d(dil, kernel_size=2 * r + 1, stride=1, padding=r) > 0
    dil_np = dil.squeeze(0).squeeze(0).cpu().numpy()

    labels, n = _cc_label(dil_np)
    if n == 0:
        return mask

    ys_all, xs_all = pixels.nonzero(as_tuple=True)
    ys_np = ys_all.cpu().numpy()
    xs_np = xs_all.cpu().numpy()
    comp_of_pt = labels[ys_np, xs_np]

    img = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(img)
    for comp_id in range(1, n + 1):
        sel = comp_of_pt == comp_id
        if not sel.any():
            continue
        cxs = xs_np[sel].astype(np.float64)
        cys = ys_np[sel].astype(np.float64)
        pts = np.stack([cxs, cys], axis=1)

        if pts.shape[0] < 3:
            for (xv, yv) in pts:
                draw.point((int(xv), int(yv)), fill=1)
            continue
        try:
            hull = ConvexHull(pts)
        except QhullError:
            for (xv, yv) in pts:
                draw.point((int(xv), int(yv)), fill=1)
            continue
        verts = pts[hull.vertices]
        poly = [(float(v[0]), float(v[1])) for v in verts]
        draw.polygon(poly, outline=1, fill=1)

    arr = np.asarray(img, dtype=np.float32)
    return torch.from_numpy(arr).to(pixels.device)


# Effects are applied in this fixed order. Channels present in
# CHANNEL_NAMES but absent here would be silently ignored, so keep in
# sync.
PIPELINE_ORDER = [
    'yellowing',
    'fading',
    'deposits',
    'scratches',
    'craquelure',
    'rip_tear',
    'paint_loss',
]


EFFECT_FNS = {
    'craquelure':  apply_craquelure,
    'rip_tear':    apply_rip_tear,
    'paint_loss':  apply_paint_loss,
    'yellowing':   apply_yellowing,
    'fading':      apply_fading,
    'deposits':    apply_deposits,
    'scratches':   apply_scratches,
}


class CorruptionModule:
    """Stochastic corruption pipeline driven by a per-type YAML config.

    Args:
        config: DictConfig or dict with keys:
            num_channels (int)
            channels (list[str])
            num_simultaneous: {min, max}
            types: dict mapping channel name -> {
                weight, local_enabled, global_enabled,
                min_severity, max_severity,
                local_num_blobs, local_blob_radius_frac,
            }

    Usage:
        cfg = OmegaConf.load("src/corruption/configs/default.yaml")
        module = CorruptionModule(cfg)
        corrupted, mask = module(clean_image)
    """

    def __init__(self, config: Any):
        self.config = config
        channels = list(config['channels']) if 'channels' in config else list(CHANNEL_NAMES)
        if channels != CHANNEL_NAMES:
            raise ValueError(
                f"Config channel order must match CHANNEL_NAMES: {CHANNEL_NAMES}, got {channels}"
            )
        self.types = config['types']

    # -- internal helpers ---------------------------------------------------

    def _sample_active_channels(self, generator: torch.Generator) -> List[str]:
        """Pick N distinct channel names with probability proportional to weight."""
        ns = self.config['num_simultaneous']
        n_lo, n_hi = int(ns['min']), int(ns['max'])
        n_hi = max(n_lo, min(n_hi, NUM_CHANNELS))
        n = n_lo + int(torch.randint(0, n_hi - n_lo + 1, (1,), generator=generator).item())
        if n <= 0:
            return []

        # Weighted sample without replacement via Gumbel top-k.
        weights = torch.tensor(
            [float(self.types[name].get('weight', 1.0)) for name in CHANNEL_NAMES],
            dtype=torch.float32,
        )
        weights = weights.clamp(min=1e-8)
        log_w = torch.log(weights)
        u = torch.rand(NUM_CHANNELS, generator=generator)
        gumbel = -torch.log(-torch.log(u.clamp(min=1e-12)))
        scores = log_w + gumbel
        topk = torch.topk(scores, k=n).indices.tolist()
        return [CHANNEL_NAMES[i] for i in topk]

    def _sample_mask_for(self, name: str, H: int, W: int,
                         generator: torch.Generator,
                         device: torch.device
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(soft_mask, region_binary)`` both (H, W)."""
        t = self.types[name]
        local_enabled = bool(t.get('local_enabled', True))
        global_enabled = bool(t.get('global_enabled', False))
        if not local_enabled and not global_enabled:
            zero = torch.zeros(H, W, device=device)
            return zero, zero.clone()

        if local_enabled and global_enabled:
            use_local = torch.rand(1, generator=generator).item() < 0.5
        else:
            use_local = local_enabled

        min_s = float(t.get('min_severity', 0.01))
        max_s = float(t.get('max_severity', 1.0))
        severity = min_s + torch.rand(1, generator=generator).item() * max(0.0, max_s - min_s)

        if use_local:
            # Number of local instances: uniform in [local_min_num, local_max_num].
            min_n = max(1, int(t.get('local_min_num', 1)))
            max_n = max(min_n, int(t.get('local_max_num', 4)))
            af = t.get('local_area_frac', [0.01, 0.05])
            shape_kind = SHAPE_KIND_BY_CHANNEL.get(name, 'generic')
            return generate_local_mask(
                H, W, severity,
                (min_n, max_n),
                (float(af[0]), float(af[1])),
                shape_kind=shape_kind,
                generator=generator, device=device,
            )
        return generate_global_mask(H, W, severity, device=device)

    # -- main entry points --------------------------------------------------

    def __call__(self, image: torch.Tensor,
                 seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random corruption to one clean image.

        Args:
            image: (3, H, W) float32 in [0, 1].
            seed:  Optional int seed for reproducibility.

        Returns:
            corrupted: (3, H, W) float32 in [0, 1].
            mask:      (K, H, W) float32 BINARY {0, 1} with K = NUM_CHANNELS.
                       Each channel is the ROI that a human annotator would
                       paint over this damage — filled, no holes, severity-
                       invariant. Channels may overlap.
        """
        C, H, W = image.shape
        assert C == 3, f"Expected 3-channel image, got {C}"
        device = image.device

        generator = torch.Generator(device='cpu')
        if seed is not None:
            generator.manual_seed(int(seed))
        else:
            generator.manual_seed(int(torch.randint(0, 2**31, (1,)).item()))

        active = set(self._sample_active_channels(generator))
        soft_masks: Dict[str, torch.Tensor] = {
            name: torch.zeros(H, W, device=device) for name in CHANNEL_NAMES
        }
        for name in active:
            soft_masks[name], _ = self._sample_mask_for(
                name, H, W, generator, device,
            )

        # Per-channel affected-pixel accumulators (before/after diff).
        affected: Dict[str, torch.Tensor] = {
            name: torch.zeros(H, W, dtype=torch.bool, device=device)
            for name in CHANNEL_NAMES
        }

        out = image.clone()
        for name in PIPELINE_ORDER:
            m = soft_masks[name]
            if m.max().item() <= 0.01:
                continue
            fn = EFFECT_FNS[name]
            gen = torch.Generator(device='cpu')
            gen.manual_seed(int(torch.randint(0, 2**31, (1,), generator=generator).item()))
            extra = {}
            if name == 'scratches':
                # Hard cap so #scratches <= local_max_num exactly.
                extra['max_count'] = int(self.types[name].get('local_max_num', 8))
            before = out
            out = fn(out, m, generator=gen, **extra)
            # ROI = pixels the effect ACTUALLY modified above the diff
            # threshold. Using affected pixels (rather than the input
            # soft-mask ROI) keeps the mask tight around the drawn
            # damage — a thin rip gets a thin-polygon mask instead of
            # the wider band the generator sampled for it.
            affected[name] = _affected_pixels(before, out)

        # Output mask = per-component convex hull of affected pixels.
        # Per-component (not a single global hull) so multi-blob masks
        # stay as N separate polygons instead of one huge polygon
        # spanning all the blobs.
        mask_tensor = torch.stack(
            [_per_component_hull_mask(affected[name], H, W) for name in CHANNEL_NAMES],
            dim=0,
        )
        return out.clamp(0, 1), mask_tensor

    def corrupt_batch(self, images: torch.Tensor,
                      seeds: Optional[List[int]] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply corruption to a batch of images.

        Args:
            images: (B, 3, H, W) float32 in [0, 1].
            seeds:  Optional list of B seeds.

        Returns:
            (B, 3, H, W), (B, K, H, W).
        """
        B = images.shape[0]
        corrupted_list, mask_list = [], []
        for i in range(B):
            seed = seeds[i] if seeds is not None else None
            c, m = self(images[i], seed=seed)
            corrupted_list.append(c)
            mask_list.append(m)
        return torch.stack(corrupted_list), torch.stack(mask_list)


def downsample_mask(mask: torch.Tensor, factor: int = 16) -> torch.Tensor:
    """Max-pool pixel-resolution mask to latent resolution.

    A damaged pixel anywhere in a (factor x factor) block propagates to
    the corresponding latent cell.

    Args:
        mask:   (K, H, W) or (B, K, H, W) float32.
        factor: Spatial compression factor (16 for FLUX.2 VAE).

    Returns:
        Downsampled mask.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
        return F.max_pool2d(mask, kernel_size=factor, stride=factor).squeeze(0)
    return F.max_pool2d(mask, kernel_size=factor, stride=factor)
