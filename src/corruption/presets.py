"""Per-channel mask generators.

Two mask modes:

* ``local`` — scattered irregular shapes. Each local instance covers a
  target fraction of the image area (drawn uniformly from the per-type
  ``local_area_frac`` range). Three shape families, chosen per channel:

      - ``generic``   : random harmonic-polar irregular blob with an
                        anisotropic (elongated) frame. The region shape
                        is severity-invariant; severity only modulates
                        the SOFT mask value inside the region.
      - ``rip_tear``  : elongated curved band (enclosing tear path with
                        a bit of leeway). Band width scales with
                        severity so low-severity gives a thin mask and
                        therefore a thin tear.
      - ``scratches`` : narrow elongated band (enclosing scratch
                        cluster). Band width scales with severity.

* ``global`` — uniform-fill covering the whole image.

Each generator returns ``(soft_mask, region_binary)`` both (H, W)
float32:

    - ``soft_mask``     in [0, 1]; intensity field consumed by effects.
    - ``region_binary`` in {0, 1}; ROI a human annotator would paint
                        over this damage. For ``generic`` this is
                        severity-invariant; for ``rip_tear`` and
                        ``scratches`` the region width scales with
                        severity.

Severity remap:
    User-facing severity s ∈ [0.01, 1.0] → effective strength
    in [~0.5, 1.0]. The soft mask inside the region has a smooth
    falloff profile peaking at effective_strength.
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple


CHANNEL_NAMES = [
    'craquelure',
    'rip_tear',
    'paint_loss',
    'yellowing',
    'fading',
    'deposits',
    'scratches',
]

NUM_CHANNELS = len(CHANNEL_NAMES)


# Which shape family to use for each channel when sampling local masks.
SHAPE_KIND_BY_CHANNEL = {
    'craquelure': 'generic',
    'rip_tear':   'rip_tear',
    'paint_loss': 'generic',
    'yellowing':  'generic',
    'fading':     'generic',
    'deposits':   'generic',
    'scratches':  'scratches',
}


def _effective_severity(severity: float) -> float:
    """Map user-facing severity in [0.01, 1.0] to [~0.3, 1.0].

    Floor at 0.3 (not higher) so sev=0.01 renders as FAINT damage, not
    a half-strength version of sev=1.0. The wide [0.3, 1.0] dynamic
    range lets effects visibly distinguish minor from catastrophic
    damage. Individual effects raise their own internal floors where
    needed for visibility (e.g. rip_tear width, scratch alpha).
    """
    return 0.3 + 0.7 * float(severity)


# ---------------------------------------------------------------------------
# Shape primitives — each paints ONE binary shape into `region` in-place.
# ---------------------------------------------------------------------------

def _paint_generic(region: torch.Tensor, cx: int, cy: int,
                   target_area: float,
                   generator: torch.Generator,
                   device: torch.device) -> None:
    """Irregular star-shaped region with radial harmonics + anisotropy.

    The boundary is r(θ) = r_est * (1 + Σ_k a_k cos(k θ + φ_k)) in a
    rotated, anisotropically-scaled frame. Harmonic amplitudes decay
    with k for sensible curvature; the resulting shape has lobes, bays
    and asymmetries but is always simply-connected (no holes).
    """
    H, W = region.shape
    r_est = max(2.0, (target_area / math.pi) ** 0.5)

    n_harm = 8
    rand_vals = torch.rand(2 * n_harm, generator=generator).tolist()
    amps = [(rand_vals[k] - 0.5) * 1.4 / (k + 1) for k in range(n_harm)]
    phases = [rand_vals[n_harm + k] * 2 * math.pi for k in range(n_harm)]
    r_max_factor = 1.0 + sum(abs(a) for a in amps)

    # Elongation in [0.5, 2.0] along a random axis.
    elong = 0.5 + 1.5 * torch.rand(1, generator=generator).item()
    axis_ang = torch.rand(1, generator=generator).item() * math.pi
    cos_a, sin_a = math.cos(axis_ang), math.sin(axis_ang)

    # A small axis means the other axis stretches by 1/elong.
    pad = int(math.ceil(r_est * r_max_factor / min(elong, 1.0))) + 3
    x0 = max(0, cx - pad)
    x1 = min(W, cx + pad + 1)
    y0 = max(0, cy - pad)
    y1 = min(H, cy + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return

    ys = torch.arange(y0, y1, device=device, dtype=torch.float32) - cy
    xs = torch.arange(x0, x1, device=device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    u = xx * cos_a + yy * sin_a
    v = -xx * sin_a + yy * cos_a
    v_s = v * elong
    dist = (u * u + v_s * v_s).sqrt()
    ang = torch.atan2(v_s, u)

    r_here = torch.ones_like(dist)
    for k in range(n_harm):
        r_here = r_here + amps[k] * torch.cos((k + 1) * ang + phases[k])
    r_here = r_here.clamp(min=0.15) * r_est

    shape = dist <= r_here
    region[y0:y1, x0:x1] = region[y0:y1, x0:x1] | shape


def _paint_band(region: torch.Tensor, cx: int, cy: int,
                half_len: float, half_w: float, bend: float,
                ang: float, device: torch.device) -> None:
    """Curved-band primitive used by rip_tear and scratches.

    The band is parameterised by:
        half_len  — half-length along the principal axis
        half_w    — half-width (centerline half-thickness)
        bend      — peak perpendicular displacement at the band's center
                    (parabolic bow)
        ang       — orientation of the principal axis

    The band tapers elliptically at the ends so the region looks like
    a natural elongated shape (lozenge) rather than a hard rectangle.
    """
    H, W = region.shape
    cos_a, sin_a = math.cos(ang), math.sin(ang)

    pad = int(math.ceil(max(half_len, half_w) + abs(bend))) + 5
    x0 = max(0, cx - pad)
    x1 = min(W, cx + pad + 1)
    y0 = max(0, cy - pad)
    y1 = min(H, cy + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return

    ys = torch.arange(y0, y1, device=device, dtype=torch.float32) - cy
    xs = torch.arange(x0, x1, device=device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    d_par = xx * cos_a + yy * sin_a
    d_perp = -xx * sin_a + yy * cos_a
    t = d_par / max(half_len, 1e-6)
    bend_off = bend * (1.0 - t * t).clamp(min=0.0)
    taper = (1.0 - t * t).clamp(min=0.0).sqrt()
    shape = (d_par.abs() <= half_len) & ((d_perp - bend_off).abs() <= half_w * taper)

    region[y0:y1, x0:x1] = region[y0:y1, x0:x1] | shape


def _area_sev_scale(severity: float) -> float:
    """Same remap as _effective_severity: 0.01→0.3, 1.0→1.0.

    Used to scale target mask AREA with severity (so a low-severity
    rip_tear / scratches mask is both thinner AND shorter-overall).
    """
    return 0.3 + 0.7 * float(severity)


def _paint_rip_tear_band(region: torch.Tensor, cx: int, cy: int,
                         target_area: float, severity: float,
                         generator: torch.Generator,
                         device: torch.device) -> None:
    """Elongated curved band; half-width scales with severity, length
    does NOT — per user spec, low severity = thinner mask (not shorter).

    Width:   sev=0.01 → ~3.2px half-width (hairline + ~3 px brush leeway);
             sev=1.0  → ~20 px half-width (wide gash + leeway).
    Length:  driven by ``target_area`` regardless of severity, with an
             aspect-ratio cap (length ≤ 8× width) so low-sev bands don't
             become absurdly long thin ribbons across the painting.
    """
    H, W = region.shape
    half_w = 3.0 + 17.0 * float(severity)
    # Length independent of severity — target_area only.
    # Band area ≈ π * half_len * half_w for a lozenge shape.
    half_len_area = target_area / (math.pi * half_w)
    # Aspect cap: band length at most 8× its width so thin bands stay
    # short. Minimum aspect 2.5 so the band always looks elongated.
    half_len = min(half_len_area, half_w * 8.0)
    half_len = max(half_len, half_w * 2.5)
    half_len = min(half_len, 0.45 * max(H, W))

    ang = torch.rand(1, generator=generator).item() * math.pi
    bend = (torch.rand(1, generator=generator).item() - 0.5) * half_len * 0.30
    _paint_band(region, cx, cy, half_len, half_w, bend, ang, device)


def _paint_scratches_band(region: torch.Tensor, cx: int, cy: int,
                          target_area: float, severity: float,
                          generator: torch.Generator,
                          device: torch.device) -> None:
    """Narrow elongated band for scratches; half-width scales with severity.

    Width:   sev=0.01 → ~3.1 px (hairline + ~3 px brush leeway);
             sev=1.0  → ~9  px (multi-scratch cluster + leeway).
    Length:  target_area driven; aspect capped at 12× so bands stay
             proportionate even for very thin scratches.
    """
    H, W = region.shape
    # Minimum 8px half-width so the walker (step_size=1.5, jitter ±7°) stays
    # inside the band for long enough to draw a visible scratch at any severity.
    half_w = max(8.0, 3.0 + 6.0 * float(severity))
    half_len_area = target_area / (math.pi * half_w)
    half_len = min(half_len_area, half_w * 12.0)
    half_len = max(half_len, half_w * 3.0)
    half_len = min(half_len, 0.45 * max(H, W))

    ang = torch.rand(1, generator=generator).item() * math.pi
    bend = (torch.rand(1, generator=generator).item() - 0.5) * half_len * 0.15
    _paint_band(region, cx, cy, half_len, half_w, bend, ang, device)


# ---------------------------------------------------------------------------
# Soft-mask helpers
# ---------------------------------------------------------------------------

def _gaussian_blur_field(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur a (H, W) float tensor via separable 1-D convolutions."""
    if sigma < 0.5:
        return field
    ksize = max(3, int(math.ceil(sigma * 4)) | 1)
    x = torch.arange(ksize, dtype=field.dtype, device=field.device) - ksize // 2
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    f = field.unsqueeze(0).unsqueeze(0)
    pad = ksize // 2
    f = F.conv2d(f, g.view(1, 1, 1, -1), padding=(0, pad))
    f = F.conv2d(f, g.view(1, 1, -1, 1), padding=(pad, 0))
    return f.squeeze(0).squeeze(0)


def _soft_mask_from_region(region_f: torch.Tensor, shape_kind: str) -> torch.Tensor:
    """Normalized distance-from-boundary soft mask (cone profile).

    For each interior pixel we compute its discrete distance-to-boundary
    via iterative 3x3 erosion, then normalize by the deepest distance
    in the shape. The resulting mask:

      • Has 0 at the shape boundary and 1.0 only at the innermost point.
      • Slopes smoothly across the WHOLE interior — there is no flat
        plateau for large shapes (the flat-blob problem).
      • Is strictly zero outside ``region_f`` (no bleed).

    A gentle square-root shaping is applied so the middle of the shape
    reaches a strong value quickly (e.g. pixel at half-depth → 0.71)
    while still tapering to 0 at the edge. Without this, very large
    shapes produced very dim interiors.

    For the narrow band shapes (rip_tear / scratches), no shaping is
    applied; the narrow ridge along the centerline is the desired
    profile.
    """
    area = float(region_f.sum().item())
    if area < 2.0:
        return region_f.clone()

    # Iterative erosion: pixels survive `k` erosions iff they are k pixels
    # from the nearest boundary. `depth` accumulates survival count.
    H, W = region_f.shape
    max_iters = min(max(H, W) // 2 + 2, 200)
    depth = torch.zeros_like(region_f)
    current = region_f.clone()
    for _ in range(max_iters):
        s = current.sum().item()
        if s < 0.5:
            break
        depth = depth + current
        current = -F.max_pool2d(
            -current.unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1,
        ).squeeze(0).squeeze(0)

    max_d = float(depth.max().item())
    if max_d < 0.5:
        return region_f.clone()

    soft = depth / max_d

    if shape_kind == 'generic':
        # sqrt lifts the mid-shape values so the interior isn't dim:
        #   depth=0.5 → soft=0.71 (was 0.5), depth=0.25 → soft=0.5.
        # Edge still smoothly tapers to 0.
        soft = soft.sqrt()
    # For 'rip_tear' and 'scratches' leave the linear ridge profile alone.

    return (soft * region_f).clamp(0, 1)


# ---------------------------------------------------------------------------
# Public mask generators
# ---------------------------------------------------------------------------

def generate_global_mask(H: int, W: int, severity: float,
                         device: torch.device = None
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform-fill mask covering the whole image.

    Returns ``(soft_mask, region_binary)`` both (H, W); region is all ones.
    """
    device = device or torch.device('cpu')
    strength = _effective_severity(severity)
    region = torch.ones(H, W, device=device)
    soft = region * strength
    return soft, region


def generate_local_mask(H: int, W: int, severity: float,
                        count_range: Tuple[int, int],
                        area_frac_range: Tuple[float, float],
                        shape_kind: str = 'generic',
                        generator: torch.Generator = None,
                        device: torch.device = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter N instances of the given shape family.

    Args:
        H, W:            Image dims.
        severity:        User-facing severity ∈ [0.01, 1.0].
        count_range:     (lo, hi) for #instances; sampled uniformly.
        area_frac_range: (lo, hi) per-instance area as fraction of image
                         area (H*W). Sampled uniformly per instance.
        shape_kind:      'generic' | 'rip_tear' | 'scratches'.
        generator:       torch.Generator for reproducibility.
        device:          torch.device.

    Returns ``(soft_mask, region_binary)``; soft = region * effective_severity.
    """
    device = device or torch.device('cpu')
    region = torch.zeros(H, W, dtype=torch.bool, device=device)

    lo, hi = int(count_range[0]), int(count_range[1])
    if hi <= lo:
        count = max(0, lo)
    else:
        count = lo + int(torch.randint(0, hi - lo + 1, (1,), generator=generator).item())

    a_lo, a_hi = float(area_frac_range[0]), float(area_frac_range[1])
    img_area = float(H * W)
    sev = float(severity)

    for _ in range(count):
        cx = int(torch.randint(0, W, (1,), generator=generator).item())
        cy = int(torch.randint(0, H, (1,), generator=generator).item())
        af = a_lo + torch.rand(1, generator=generator).item() * max(0.0, a_hi - a_lo)
        target_area = af * img_area
        if shape_kind == 'rip_tear':
            _paint_rip_tear_band(region, cx, cy, target_area, sev, generator, device)
        elif shape_kind == 'scratches':
            _paint_scratches_band(region, cx, cy, target_area, sev, generator, device)
        else:
            _paint_generic(region, cx, cy, target_area, generator, device)

    region_f = region.float()
    if region_f.sum().item() < 1.0:
        soft = region_f * _effective_severity(severity)
    else:
        # Gaussian falloff soft mask: effects fade naturally at region edges
        # so the painted result looks like natural damage rather than a
        # flat pasted-on blob. region_f (binary) is still returned verbatim
        # for the ROI annotation channel.
        soft_shape = _soft_mask_from_region(region_f, shape_kind)
        soft = soft_shape * _effective_severity(severity)
    return soft, region_f
