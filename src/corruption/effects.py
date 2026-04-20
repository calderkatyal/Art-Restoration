"""Individual corruption effect implementations.

Each effect takes (3, H, W) float32 image in [0, 1] and a (H, W) float32
SOFT mask in [0, 1] controlling per-pixel intensity, and returns the
corrupted (3, H, W) image. The pipeline-level ROI (which pixels a human
annotator would mark as damaged) is tracked by the mask generator, not
by these functions. Each effect produces ONE visual appearance: no
randomized mode selection inside the function.

Channels:
    craquelure  — Voronoi tessellation cracks (paint shrinkage craquelure).
    rip_tear    — jagged tear with exposed canvas and shadow. Local only.
    paint_loss  — blob-shaped paint flaking with substrate reveal.
    yellowing   — varnish yellowing (CIELAB a/b offset).
    fading      — photochemical desaturation / bleach.
    deposits    — grime / soot darkening veil.
    scratches   — thin linear abrasion marks. Local only.
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from .color import rgb_to_lab, lab_to_rgb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_blur_2d(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur on (H, W) or (C, H, W) via separable 1D convolutions."""
    if sigma < 0.5:
        return field
    ksize = int(math.ceil(sigma * 6)) | 1
    ksize = max(3, ksize)
    x = torch.arange(ksize, dtype=field.dtype, device=field.device) - ksize // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    needs_batch = field.dim() == 2
    if needs_batch:
        field = field.unsqueeze(0).unsqueeze(0)
    elif field.dim() == 3:
        field = field.unsqueeze(0)

    C = field.shape[1]
    pad = ksize // 2
    kh = kernel_1d.view(1, 1, 1, ksize).expand(C, -1, -1, -1)
    out = F.conv2d(F.pad(field, (pad, pad, 0, 0), mode='reflect'), kh, groups=C)
    kv = kernel_1d.view(1, 1, ksize, 1).expand(C, -1, -1, -1)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode='reflect'), kv, groups=C)

    if needs_batch:
        return out.squeeze(0).squeeze(0)
    return out.squeeze(0)


def make_noise(H: int, W: int, blur_sigma: float,
               generator: torch.Generator = None,
               device: torch.device = None,
               normalize: bool = True) -> torch.Tensor:
    """Smoothed noise field (H, W) in [0, 1]."""
    noise = torch.rand(H, W, generator=generator, device=device or torch.device('cpu'))
    noise = gaussian_blur_2d(noise, blur_sigma)
    if normalize:
        lo, hi = noise.min(), noise.max()
        span = (hi - lo).clamp(min=1e-6)
        noise = (noise - lo) / span
    return noise


def box_blur_float(field: torch.Tensor, radius: int) -> torch.Tensor:
    """Box blur on (H, W) field. Used for mask smoothing."""
    if radius < 1:
        return field
    ksize = 2 * radius + 1
    kernel = torch.ones(1, 1, ksize, ksize, device=field.device, dtype=field.dtype) / (ksize * ksize)
    f = field.unsqueeze(0).unsqueeze(0)
    return F.conv2d(F.pad(f, (radius, radius, radius, radius), mode='reflect'), kernel).squeeze(0).squeeze(0)


def _sample_canvas_color(image: torch.Tensor) -> Tuple[float, float, float]:
    """Substrate color: 15% image mean blended with light canvas tone [215, 200, 175]/255."""
    flat = image.view(3, -1)[:, ::40]
    mr, mg, mb = flat[0].mean().item(), flat[1].mean().item(), flat[2].mean().item()
    cr = min(1.0, mr * 0.15 + (215/255) * 0.85)
    cg = min(1.0, mg * 0.15 + (200/255) * 0.85)
    cb = min(1.0, mb * 0.15 + (175/255) * 0.85)
    return cr, cg, cb


def _local_lum_boost(image: torch.Tensor, mask: torch.Tensor,
                     min_val: float = 0.02,
                     max_boost: float = 2.2) -> float:
    """Return a luminance-aware strength multiplier.

    For effects that rely on darkening a thin structure (cracks, tears,
    scratches), a pure multiplicative operation produces no visible change
    on already-dark pixels. We measure the mean luminance of the masked
    region and return a multiplier that boosts the effect strength for
    dark regions.

    Output range: ~1.0 for bright regions (lum ≥ 0.5), scaling up to
    `max_boost` at lum → 0.
    """
    lum = image[0] * 0.299 + image[1] * 0.587 + image[2] * 0.114
    w = (mask >= min_val).float()
    if w.sum().item() < 1:
        return 1.0
    mean_lum = float((lum * w).sum().item() / w.sum().item())
    # Piecewise: 1.0 at lum>=0.5, linear up to max_boost at lum=0.
    t = max(0.0, min(1.0, (0.5 - mean_lum) / 0.5))
    return 1.0 + (max_boost - 1.0) * t


def _mask_principal_angle(mask: torch.Tensor, min_val: float = 0.05) -> float:
    """Return the angle (radians) of the mask's principal axis.

    Used by rip_tear and scratches so the walker heads ALONG an
    elongated mask rather than across it. For isotropic masks the
    returned angle is arbitrary (the axis is ill-defined); that's fine
    because the walker tolerates any direction inside a round blob.
    """
    active = mask >= min_val
    if active.sum().item() < 10:
        return 0.0
    ys, xs = active.nonzero(as_tuple=True)
    ys_f = ys.float()
    xs_f = xs.float()
    cx = xs_f.mean().item()
    cy = ys_f.mean().item()
    x_c = xs_f - cx
    y_c = ys_f - cy
    Sxx = float((x_c * x_c).mean().item())
    Sxy = float((x_c * y_c).mean().item())
    Syy = float((y_c * y_c).mean().item())
    return 0.5 * math.atan2(2.0 * Sxy, Sxx - Syy)


def _mask_component_centers(mask: torch.Tensor, min_val: float = 0.02):
    """4-connectivity connected components on the active mask.

    Returns a list of (cx, cy, size) tuples (one per component) sorted
    by size descending. `cx`, `cy` are float centroids. Iterative BFS
    on CPU using a bool grid — O(|active pixels|), fine for our small
    masks (< ~10^5 active pixels per call).
    """
    active = (mask >= min_val).detach().cpu()
    H, W = active.shape
    visited = torch.zeros_like(active, dtype=torch.bool)
    ys_all, xs_all = active.nonzero(as_tuple=True)
    ys_list = ys_all.tolist()
    xs_list = xs_all.tolist()
    active_grid = active.tolist()  # nested Python lists for fast access
    visited_grid = visited.tolist()
    components = []
    for idx in range(len(ys_list)):
        y0 = ys_list[idx]; x0 = xs_list[idx]
        if visited_grid[y0][x0]:
            continue
        stack = [(y0, x0)]
        sx_sum = 0; sy_sum = 0; n = 0
        while stack:
            cy, cx = stack.pop()
            if cy < 0 or cy >= H or cx < 0 or cx >= W:
                continue
            if visited_grid[cy][cx] or not active_grid[cy][cx]:
                continue
            visited_grid[cy][cx] = True
            sx_sum += cx; sy_sum += cy; n += 1
            stack.append((cy + 1, cx))
            stack.append((cy - 1, cx))
            stack.append((cy, cx + 1))
            stack.append((cy, cx - 1))
        if n > 0:
            components.append((sx_sum / n, sy_sum / n, n))
    components.sort(key=lambda c: -c[2])
    return components


def _sample_point_in_mask(mask: torch.Tensor,
                          generator: torch.Generator,
                          min_val: float = 0.05,
                          max_attempts: int = 300
                          ) -> Optional[Tuple[int, int]]:
    """Rejection-sample a point (x, y) whose mask value is above min_val.

    Used by rip_tear and scratches so they originate inside the masked
    region. Returns None if no point passes rejection.

    Severity invariance: acceptance is `rand < m / m_max` rather than
    `rand < m`. This way the same generator state produces the same
    sequence of accepted points regardless of the severity used when
    building `mask` (because mask values scale uniformly with severity,
    so `m / m_max` is the same ratio at each pixel).
    """
    H, W = mask.shape
    mcpu = mask.detach().cpu()
    m_max = max(float(mcpu.max().item()), 1e-6)
    for _ in range(max_attempts):
        x = int(torch.randint(0, W, (1,), generator=generator).item())
        y = int(torch.randint(0, H, (1,), generator=generator).item())
        m = mcpu[y, x].item()
        if m < min_val:
            continue
        if torch.rand(1, generator=generator).item() < m / m_max:
            return x, y
    return None


# ---------------------------------------------------------------------------
# Cracks
# ---------------------------------------------------------------------------

def apply_craquelure(image: torch.Tensor, mask: torch.Tensor,
                     generator: torch.Generator = None) -> torch.Tensor:
    """Craquelure: Voronoi tessellation cracks from age / paint-drying shrinkage.

    Nearest-two-site distance difference detects tessellation edges.
    Grid-based spatial lookup enforces minimum site spacing; edge width
    scales with cell size.
    """
    C, H, W = image.shape
    out = image.clone()

    mask_sum = mask.sum().item()
    if mask_sum < 1:
        return out

    # Cell size is a FIXED function of image area — independent of severity.
    # Severity controls line darkness (below), not tessellation density, so
    # "intensity 0.01" and "intensity 1.0" produce the same cell layout.
    target = max(40, int(H * W / 200))
    min_dist = 8
    min_dist2 = min_dist * min_dist

    grid_cell = min_dist
    grid_w = math.ceil(W / grid_cell)
    grid_h = math.ceil(H / grid_cell)
    grid = [[] for _ in range(grid_w * grid_h)]

    sites = []
    mask_np = mask.cpu()
    # Normalize placement probability by mask max so that the SAME RNG
    # stream produces the same site pattern regardless of severity. Without
    # this the acceptance rate scales with severity (low sev → more
    # attempts → different generator state by the time `target` is met).
    m_max = max(float(mask_np.max().item()), 1e-6)
    attempts = 0
    max_attempts = target * 50

    while len(sites) < target and attempts < max_attempts:
        attempts += 1
        x = int(torch.randint(0, W, (1,), generator=generator).item())
        y = int(torch.randint(0, H, (1,), generator=generator).item())
        m = mask_np[y, x].item()
        if torch.rand(1, generator=generator).item() < m / m_max:
            gx = x // grid_cell
            gy = y // grid_cell
            too_close = False
            for dy in range(-1, 2):
                if too_close:
                    break
                for dx in range(-1, 2):
                    if too_close:
                        break
                    nx, ny = gx + dx, gy + dy
                    if nx < 0 or nx >= grid_w or ny < 0 or ny >= grid_h:
                        continue
                    for sx, sy in grid[ny * grid_w + nx]:
                        if (sx - x) ** 2 + (sy - y) ** 2 < min_dist2:
                            too_close = True
                            break
            if not too_close:
                sites.append((x, y))
                grid[gy * grid_w + gx].append((x, y))

    if len(sites) < 2:
        return out

    sites_t = torch.tensor(sites, dtype=torch.float32, device=image.device)

    # Luminance-aware boost so crack lines remain visible on dark
    # regions of the painting (Mona Lisa backgrounds, etc.). Kept mild
    # here because cracks already cover a wide area — strong boosts
    # cause cracks to dominate the composition on darker paintings.
    lum_boost = _local_lum_boost(image, mask, min_val=0.02, max_boost=3.0)

    # Adaptive crack colour: on dark-background regions the standard near-black
    # crack (0.06, 0.05, 0.04) is invisible against the paint. Real cracks expose
    # the underlying paint layer or gesso primer, which is typically lighter than
    # the surface paint. For dark regions we therefore blend toward a warm
    # off-white so the crack reads as a lighter groove rather than a dark shadow.
    lum_field = image[0] * 0.299 + image[1] * 0.587 + image[2] * 0.114
    active_w = (mask >= 0.02).float()
    if active_w.sum().item() > 0:
        mean_lum = float((lum_field * active_w).sum().item() / active_w.sum().item())
    else:
        mean_lum = 0.5
    if mean_lum < 0.50:
        # Interpolate from near-black toward warm cream (aged gesso/canvas primer).
        # Threshold raised to 0.50 so mid-dark backgrounds (dark forests, night
        # skies) also get the lighter crack colour, not just nearly-black regions.
        t = max(0.0, (0.50 - mean_lum) / 0.50)  # 0 at lum=0.50, 1 at lum=0
        cr = 0.06 + t * 0.56   # up to 0.62
        cg = 0.05 + t * 0.51   # up to 0.56
        cb = 0.04 + t * 0.42   # up to 0.46
    else:
        cr, cg, cb = 0.06, 0.05, 0.04

    chunk_size = 4096
    out_flat = out.view(3, -1).clone()
    mask_flat = mask.view(-1)

    for start in range(0, H * W, chunk_size):
        end = min(start + chunk_size, H * W)
        indices = torch.arange(start, end, device=image.device)
        ys = indices // W
        xs = indices % W

        m_vals = mask_flat[start:end]
        # Low threshold so that severity=0.01 (effective ~0.3) still paints.
        active = m_vals >= 0.02
        if not active.any():
            continue

        active_idx = active.nonzero(as_tuple=True)[0]
        ax = xs[active_idx].float()
        ay = ys[active_idx].float()

        dx = ax.unsqueeze(1) - sites_t[:, 0].unsqueeze(0)
        dy = ay.unsqueeze(1) - sites_t[:, 1].unsqueeze(0)
        dists = dx ** 2 + dy ** 2

        top2 = dists.topk(2, dim=1, largest=False)
        sd1 = top2.values[:, 0].sqrt()
        sd2 = top2.values[:, 1].sqrt()
        diff = sd2 - sd1

        # Thicker cracks: higher coefficient (0.18 vs 0.12) and wider max
        # (2.5px vs 1.5px) so crack lines are 3-5px total width and clearly
        # visible even at small preview sizes.
        edge_width = (sd1 * 0.18).clamp(max=2.5)
        is_edge = (diff < edge_width) & (sd1 > 0.8)

        if is_edge.any():
            edge_idx = active_idx[is_edge]
            global_idx = start + edge_idx
            a = m_vals[edge_idx]
            d = diff[is_edge]
            ew = edge_width[is_edge]
            # Strong alpha so cracks are unmissable. Floor 0.35 ensures even
            # low-severity cracks are clearly visible; ceiling 0.55 gives
            # bold lines at high severity. lum_boost adds up to 2.2x on dark.
            alpha = ((0.35 + 0.55 * a) * (1.0 - d / ew) * lum_boost).clamp(max=1.0)
            inv = 1.0 - alpha

            out_flat[0, global_idx] = out_flat[0, global_idx] * inv + cr * alpha
            out_flat[1, global_idx] = out_flat[1, global_idx] * inv + cg * alpha
            out_flat[2, global_idx] = out_flat[2, global_idx] * inv + cb * alpha

    out = out_flat.view(3, H, W)
    return out


def _walk_tear_spine(out: torch.Tensor, mask: torch.Tensor,
                     sx: float, sy: float, start_angle: float,
                     max_w: float, max_steps: int,
                     generator: torch.Generator,
                     int_r: float = 0.0, int_g: float = 0.0, int_b: float = 0.0) -> None:
    """Walk a tear spine from (sx, sy) along `start_angle` painting a
    variable-width pure-black opening (canvas separation).

    - Interior is pure black (0,0,0) at full opacity — the painting is
      physically separated, no underlying paint shows through.
    - Edge offsets follow a slow random-walk (coherent jagged boundary)
      rather than independent per-step noise (which produced the
      pixel-scribble look). One RNG draw per step keeps the walker path
      severity-invariant.
    - Width tapers parabolically at the ends and drifts along the spine.
    - A 1-pixel-wide darkened "lip" just outside the gash sells the
      paint-pulled-apart appearance without bleeding into the interior.
    """
    _, H, W = out.shape
    angle = start_angle
    px, py = sx, sy
    step_size = 1.0
    out_streak = 0

    # Random-walk width, initialized in the upper half of the allowed range.
    w_now = max_w * (0.6 + 0.4 * torch.rand(1, generator=generator).item())
    w_min = max(0.5, max_w * 0.35)
    w_max = max_w * 1.15

    # Slow random-walk edge offsets. Each step they nudge ±a small amount,
    # producing coherent jagged boundaries instead of noisy per-step jitter.
    edge_off_r = 0.0
    edge_off_l = 0.0
    edge_walk_step = max(0.06, max_w * 0.06)
    edge_off_clip = max(0.30, max_w * 0.30)

    for step in range(max_steps):
        # Parabolic taper 0..1..0 across the walk.
        t = step / max(1, max_steps - 1)
        taper = max(0.20, 1.0 - (2.0 * t - 1.0) ** 2)

        # Tear-like angular wander — small per step so the spine reads
        # as a coherent line / gentle curve rather than a scribble.
        angle += (torch.rand(1, generator=generator).item() - 0.5) * 0.10

        # Smooth width drift (random walk with bounds).
        dw = (torch.rand(1, generator=generator).item() - 0.5) * max_w * 0.14
        w_now = max(w_min, min(w_max, w_now + dw))

        # Slow-walk edge offsets (one RNG draw each side, per step).
        edge_off_r += (torch.rand(1, generator=generator).item() - 0.5) * edge_walk_step
        edge_off_l += (torch.rand(1, generator=generator).item() - 0.5) * edge_walk_step
        edge_off_r = max(-edge_off_clip, min(edge_off_clip, edge_off_r))
        edge_off_l = max(-edge_off_clip, min(edge_off_clip, edge_off_l))

        px += math.cos(angle) * step_size
        py += math.sin(angle) * step_size

        ix, iy = int(round(px)), int(round(py))
        if ix < 0 or ix >= W or iy < 0 or iy >= H:
            break
        if mask[iy, ix].item() < 0.02:
            out_streak += 1
            if out_streak > 6:
                break
            continue
        out_streak = 0

        # Spine direction (cos_a, sin_a) and perpendicular (cos_p, sin_p).
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        perp_ang = angle + math.pi / 2
        cos_p = math.cos(perp_ang)
        sin_p = math.sin(perp_ang)

        w_eff = w_now * taper
        edge_r_eff = max(0.0, w_eff + edge_off_r)
        edge_l_eff = max(0.0, w_eff + edge_off_l)
        # 2D bounding box; +5 for shadow lip + curl highlight.
        max_e = int(math.ceil(max(edge_r_eff, edge_l_eff))) + 5

        for dy in range(-max_e, max_e + 1):
            ny = iy + dy
            if ny < 0 or ny >= H:
                continue
            for dx in range(-max_e, max_e + 1):
                nx = ix + dx
                if nx < 0 or nx >= W:
                    continue
                if mask[ny, nx].item() < 0.02:
                    continue
                # Signed distances in spine frame.
                ox = (nx + 0.5) - px
                oy = (ny + 0.5) - py
                d_par = ox * cos_a + oy * sin_a       # along spine
                d_perp = ox * cos_p + oy * sin_p      # across spine
                # Only paint pixels whose along-spine offset is within
                # ±0.6 step of this spine point — adjacent steps cover
                # the rest. Keeps strips non-overlapping & solid.
                if abs(d_par) > 0.6:
                    continue
                ad = abs(d_perp)
                edge_lim = edge_r_eff if d_perp > 0 else edge_l_eff
                if ad <= edge_lim:
                    # Interior: adaptive colour (canvas/substrate on dark backgrounds,
                    # near-black on light backgrounds), full opacity.
                    out[0, ny, nx] = int_r
                    out[1, ny, nx] = int_g
                    out[2, ny, nx] = int_b
                elif ad <= edge_lim + 2.0 and edge_lim > 0.5:
                    # Shadow band — underside of paint that has lifted up
                    # at the tear edge. Multiplicative darken preserves
                    # underlying hue.
                    lip_t = (ad - edge_lim) / 2.0
                    darken = 0.18 + 0.55 * lip_t
                    out[0, ny, nx] = out[0, ny, nx] * darken
                    out[1, ny, nx] = out[1, ny, nx] * darken
                    out[2, ny, nx] = out[2, ny, nx] * darken
                elif ad <= edge_lim + 4.0 and edge_lim > 0.5:
                    # Highlight band — top of the curled-up paint
                    # catching the light. Warm off-white blended with low
                    # alpha, peaking at the inner edge of the band.
                    hl_t = (ad - (edge_lim + 2.0)) / 2.0   # 0..1
                    hl_a = 0.22 * (1.0 - hl_t)             # peak at inner
                    inv = 1.0 - hl_a
                    out[0, ny, nx] = out[0, ny, nx] * inv + 0.96 * hl_a
                    out[1, ny, nx] = out[1, ny, nx] * inv + 0.93 * hl_a
                    out[2, ny, nx] = out[2, ny, nx] * inv + 0.86 * hl_a


def _paint_tear_puncture(out: torch.Tensor, mask: torch.Tensor,
                         cx: float, cy: float, radius: float,
                         generator: torch.Generator,
                         int_r: float = 0.0, int_g: float = 0.0, int_b: float = 0.0) -> None:
    """Paint a small irregular pure-black hole (puncture-style tear).

    Interior is fully zeroed (paint physically missing). A 1.5px darkened
    lip just outside the hole sells the torn-edge appearance.
    """
    _, H, W = out.shape
    ri = int(math.ceil(radius)) + 5
    # Random radial envelope sampled at 16 angles, interpolated in-between.
    n_samples = 16
    r_profile = [
        radius * (0.70 + 0.50 * torch.rand(1, generator=generator).item())
        for _ in range(n_samples)
    ]

    cxi, cyi = int(round(cx)), int(round(cy))
    for dy in range(-ri, ri + 1):
        ny = cyi + dy
        if ny < 0 or ny >= H:
            continue
        for dx in range(-ri, ri + 1):
            nx = cxi + dx
            if nx < 0 or nx >= W:
                continue
            if mask[ny, nx].item() < 0.02:
                continue
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                r_here = r_profile[0]
            else:
                ang = math.atan2(dy, dx)  # -pi..pi
                idx_f = ((ang + math.pi) / (2 * math.pi)) * n_samples
                i0 = int(idx_f) % n_samples
                i1 = (i0 + 1) % n_samples
                frac = idx_f - int(idx_f)
                r_here = r_profile[i0] * (1 - frac) + r_profile[i1] * frac

            if dist <= r_here:
                # Interior: adaptive colour, fully opaque.
                out[0, ny, nx] = int_r
                out[1, ny, nx] = int_g
                out[2, ny, nx] = int_b
            elif dist <= r_here + 2.0 and r_here > 0.5:
                # Shadow band under lifted paint.
                lip_t = (dist - r_here) / 2.0
                darken = 0.18 + 0.55 * lip_t
                out[0, ny, nx] = out[0, ny, nx] * darken
                out[1, ny, nx] = out[1, ny, nx] * darken
                out[2, ny, nx] = out[2, ny, nx] * darken
            elif dist <= r_here + 4.0 and r_here > 0.5:
                # Highlight on top of curled paint.
                hl_t = (dist - (r_here + 2.0)) / 2.0
                hl_a = 0.22 * (1.0 - hl_t)
                inv = 1.0 - hl_a
                out[0, ny, nx] = out[0, ny, nx] * inv + 0.96 * hl_a
                out[1, ny, nx] = out[1, ny, nx] * inv + 0.93 * hl_a
                out[2, ny, nx] = out[2, ny, nx] * inv + 0.86 * hl_a


def apply_rip_tear(image: torch.Tensor, mask: torch.Tensor,
                   generator: torch.Generator = None) -> torch.Tensor:
    """Physical rip / tear — elongated dark opening with jagged edges.

    Approach:
        1. Sample a start point INSIDE the mask blob.
        2. Pick a shape mode:
             - line (default, ~70%): bidirectional walk from the start.
             - branched (~20%): bidirectional walk PLUS a sub-branch
               coming off at a random perpendicular-ish angle — common
               real-world feature when a tear forks around a weak spot.
             - puncture (~10%): small irregular hole only, no elongation
               — models a localized puncture / small detachment.
        3. For each walked segment, paint a variable-width jagged opening
           via `_walk_tear_spine` (see docstring there).

    Severity governs OPENING WIDTH, not darkness: small tears are thin
    dark slits, large tears are wide dark gashes, and the interior color
    is ≈ the same in both regimes.
    """
    C, H, W = image.shape
    out = image.clone()

    if mask.sum().item() < 1:
        return out

    active_mask = mask >= 0.02
    if not active_mask.any():
        return out

    ys_active, xs_active = active_mask.nonzero(as_tuple=True)
    y_lo, y_hi = int(ys_active.min()), int(ys_active.max())
    x_lo, x_hi = int(xs_active.min()), int(xs_active.max())
    bbox_diag = float(((y_hi - y_lo) ** 2 + (x_hi - x_lo) ** 2) ** 0.5)
    if bbox_diag < 4.0:
        return out

    # Severity scale from the user-facing mask magnitude. `mask.max()`
    # ranges from ≈0.3 (severity 0.01) to 1.0 (severity 1.0) because of
    # `_effective_severity`. We remap to [0, 1].
    sev_scale = float(mask.max().item())
    sev_norm = max(0.0, min(1.0, (sev_scale - 0.3) / 0.7))

    # Adaptive interior colour: tear exposes canvas/substrate beneath the paint.
    # On dark backgrounds, black interior (0,0,0) is invisible; use canvas tone.
    lum_field = image[0] * 0.299 + image[1] * 0.587 + image[2] * 0.114
    active_w = (mask >= 0.02).float()
    if active_w.sum().item() > 0:
        mean_lum = float((lum_field * active_w).sum().item() / active_w.sum().item())
    else:
        mean_lum = 0.5
    # Smooth blend: full canvas colour at lum=0, black at lum>=0.5.
    # This catches mid-dark backgrounds (forest green, dark sky) that the
    # old hard threshold at 0.30 missed.
    c_r, c_g, c_b = _sample_canvas_color(image)
    t_interior = max(0.0, min(1.0, (0.50 - mean_lum) / 0.50))
    int_r = t_interior * c_r * 0.80
    int_g = t_interior * c_g * 0.75
    int_b = t_interior * c_b * 0.65

    # Width scales with severity. Floor raised to 3.5px so a low-severity
    # tear renders as a clearly visible thin slit (the previous 2.5 was
    # on the edge of disappearing into anti-aliasing at typical preview
    # sizes). sev_norm=0 -> max_w=3.5, sev_norm=1 -> max_w=18.
    max_w = 3.5 + sev_norm * 14.5

    # Walker length is driven by the mask bbox — we walk enough to traverse
    # the full mask and rely on the mask-clip in `_walk_tear_spine` (which
    # bails after wandering 6+ steps outside mask) to stop naturally.
    # Floor raised from 15 to 25 so a low-sev tear is reliably elongated
    # (thin but visible) rather than appearing as a tiny dot.
    steps_per_dir = max(25, int(bbox_diag * 0.60))

    # Find every connected component of the mask and guarantee at least
    # one tear per component so a 2-region mask never renders with only
    # a single tear visible.
    components = _mask_component_centers(mask, min_val=0.02)
    extra = int(torch.randint(0, 2, (1,), generator=generator).item())
    num_tears = max(len(components), 1) + extra

    # Reserve one index for a forced-line tear so at least one tear is
    # always an elongated slit — otherwise the RNG could pick puncture
    # for every tear at low sev and leave the effect nearly invisible.
    forced_line_idx = int(torch.randint(0, num_tears, (1,), generator=generator).item())

    for t_idx in range(num_tears):
        if t_idx < len(components):
            # First N tears: start at each component's centroid.
            cx_c, cy_c, _size = components[t_idx]
            sx, sy = float(cx_c), float(cy_c)
            if mask[int(round(sy)), int(round(sx))].item() < 0.02:
                # Centroid fell in a concavity — fall back to sampling.
                alt = _sample_point_in_mask(mask, generator, min_val=0.05)
                if alt is None:
                    continue
                sx, sy = float(alt[0]), float(alt[1])
        else:
            start = _sample_point_in_mask(mask, generator, min_val=0.05)
            if start is None:
                continue
            sx, sy = float(start[0]), float(start[1])

        # Pick shape mode for this tear.
        if t_idx == forced_line_idx:
            shape_mode = 'line'
        else:
            shape_rand = torch.rand(1, generator=generator).item()
            # Puncture probability scales from 0 at the lowest severity
            # (a 3-4px hole is too small to read as a puncture) up to
            # 10% at the highest severity.
            punct_p = 0.10 * sev_norm
            if shape_rand < punct_p:
                shape_mode = 'puncture'
            elif shape_rand < punct_p + 0.20:
                shape_mode = 'branched'
            else:
                shape_mode = 'line'

        if shape_mode == 'puncture':
            # Irregular hole only. Slightly larger than half max_w so it
            # reads as a distinct hole rather than a tiny dot.
            radius = max(2.5, max_w * 0.9)
            _paint_tear_puncture(
                out, mask, sx, sy, radius, generator,
                int_r=int_r, int_g=int_g, int_b=int_b,
            )
            continue

        # Align the tear spine with the mask's principal axis so the
        # walker heads down the long side of a narrow band rather than
        # immediately exiting it. Add ±0.25 rad of jitter for variety.
        mask_axis = _mask_principal_angle(mask)
        base_angle = mask_axis + (torch.rand(1, generator=generator).item() - 0.5) * 0.50

        # Bidirectional main axis.
        for direction in (0.0, math.pi):
            _walk_tear_spine(
                out, mask,
                sx, sy, base_angle + direction,
                max_w, steps_per_dir, generator,
                int_r=int_r, int_g=int_g, int_b=int_b,
            )

        if shape_mode == 'branched':
            # Perpendicular-ish branch, up to ~60° off perpendicular,
            # half the length and width for a secondary rip.
            branch_offset = (torch.rand(1, generator=generator).item() - 0.5) * math.pi * 0.6
            branch_angle = base_angle + math.pi / 2 + branch_offset
            _walk_tear_spine(
                out, mask,
                sx, sy, branch_angle,
                max_w * 0.65, max(20, steps_per_dir // 2),
                generator,
                int_r=int_r, int_g=int_g, int_b=int_b,
            )

    return out.clamp(0, 1)


# ---------------------------------------------------------------------------
# Paint loss
# ---------------------------------------------------------------------------

def apply_paint_loss(image: torch.Tensor, mask: torch.Tensor,
                     generator: torch.Generator = None) -> torch.Tensor:
    """Paint loss — fill the entire masked ROI with substrate color.

    The mask defines the whole affected patch; the effect replaces paint
    with a slightly textured substrate (warm canvas tone) across the
    whole region. A narrow rim just INSIDE the boundary is darkened to
    read as a small cavity / lifted edge. Severity controls how opaque
    the substrate reveal is (low sev ≈ partial thinning, high sev ≈
    total loss).
    """
    C, H, W = image.shape
    out = image.clone()

    active = mask >= 0.02
    if not active.any():
        return out

    m = mask.clamp(0, 1)
    af = active.float()

    # Paint loss lightens the region — the pigment layer is partially or
    # fully gone, exposing the lighter ground/canvas beneath. Rather than
    # replacing with a flat substrate colour, we lighten the existing
    # image toward a warm off-white so the underlying painting texture
    # still shows through, looking like thinned or chipped paint.
    fine_noise = (torch.rand(H, W, generator=generator, device=image.device) - 0.5) * 15.0 / 255.0

    # Target: warm off-white canvas tone, with subtle per-pixel jitter.
    target_r = (0.88 + fine_noise).clamp(0, 1)
    target_g = (0.84 + fine_noise).clamp(0, 1)
    target_b = (0.76 + fine_noise).clamp(0, 1)

    # Blend strength: severity controls how much paint is lost (how much
    # we lighten toward the canvas tone). m^1.3 keeps low-sev subtle.
    a_sub = (m ** 1.3).clamp(max=0.85) * af

    out[0] = out[0] * (1.0 - a_sub) + target_r * a_sub
    out[1] = out[1] * (1.0 - a_sub) + target_g * a_sub
    out[2] = out[2] * (1.0 - a_sub) + target_b * a_sub

    return out.clamp(0, 1)


# ---------------------------------------------------------------------------
# Optical / surface effects
# ---------------------------------------------------------------------------

def apply_yellowing(image: torch.Tensor, mask: torch.Tensor,
                    generator: torch.Generator = None) -> torch.Tensor:
    """Varnish yellowing via positive CIELAB a/b offset.

    Applied with a hard mask boundary — the effect fills the full ROI
    and does not bleed past the binary region.
    """
    C, H, W = image.shape
    out = image.clone()

    active = mask > 0.01
    if not active.any():
        return out

    noise = make_noise(H, W, 30.0, generator=generator, device=image.device, normalize=False)

    rgb_hwc = out.permute(1, 2, 0)
    lab = rgb_to_lab(rgb_hwc)

    m = mask.clamp(max=1.0)
    m_resp = m * m.sqrt()
    offset_b = (16.0 + 36.0 * noise) * m_resp
    offset_a = (4.0 + 12.0 * noise) * m_resp

    af = active.float()
    lab[..., 1] = lab[..., 1] + offset_a * af
    lab[..., 2] = lab[..., 2] + offset_b * af

    rgb_out = lab_to_rgb(lab)
    # Hard-boundary: only modify pixels inside the ROI.
    rgb_out = torch.where(af.unsqueeze(-1) > 0, rgb_out, rgb_hwc)
    return rgb_out.permute(2, 0, 1).clamp(0, 1)


def apply_fading(image: torch.Tensor, mask: torch.Tensor,
                 generator: torch.Generator = None) -> torch.Tensor:
    """Photochemical fading: noise-modulated desaturation + bleach toward chalky off-white.

    Hard mask boundary — effect fills the full ROI, does not bleed past.
    """
    C, H, W = image.shape
    out = image.clone()

    active = mask > 0.01
    if not active.any():
        return out

    fade_noise = make_noise(H, W, 20.0, generator=generator, device=image.device, normalize=False)
    m = mask.clamp(max=1.0)
    bleach_r, bleach_g, bleach_b = 180/255, 180/255, 180/255

    n = 0.6 + 0.4 * fade_noise
    strength = m * n

    r, g, b = out[0], out[1], out[2]
    lum = r * 0.299 + g * 0.587 + b * 0.114

    # Desaturation is the dominant component of real photochemical fading
    # (pigments lose chroma while lightness mostly stays). Coefficient
    # raised so the signature of fading reads clearly at high severity.
    desat_r = r + (lum - r) * strength * 0.85
    desat_g = g + (lum - g) * strength * 0.85
    desat_b = b + (lum - b) * strength * 0.85

    bleach = strength * 0.35
    out_r = desat_r * (1 - bleach) + bleach_r * bleach
    out_g = desat_g * (1 - bleach) + bleach_g * bleach
    out_b = desat_b * (1 - bleach) + bleach_b * bleach

    out[0] = torch.where(active, out_r, out[0])
    out[1] = torch.where(active, out_g, out[1])
    out[2] = torch.where(active, out_b, out[2])

    return out.clamp(0, 1)


def apply_deposits(image: torch.Tensor, mask: torch.Tensor,
                   generator: torch.Generator = None) -> torch.Tensor:
    """Surface deposits: dark grime + soot particulate veil.

    Darkens and reduces contrast — the dominant visual effect of real grime
    and soot accumulation. Warm brown (grime) mixed with near-black (soot),
    soot heavier at top (smoke rises). Hard mask boundary.
    """
    C, H, W = image.shape
    out = image.clone()

    field = mask
    if field.max() < 0.01:
        return out

    patch_noise = make_noise(H, W, 18.0, generator=generator, device=image.device)
    fine_noise = make_noise(H, W, 2.0, generator=generator, device=image.device)
    med_noise = make_noise(H, W, 8.0, generator=generator, device=image.device)

    y_coords = torch.arange(H, device=image.device).float().unsqueeze(1).expand(H, W)
    vert_bias = 1.0 - (y_coords / H) * 0.3
    soot_frac = med_noise * vert_bias

    # Lighter warm-brown anchor (was 70/60/45). On dark paintings the old
    # anchor was too close to the existing pixel values, so the blend
    # produced essentially no visible change; the new anchor is a warm
    # mid-brown that shifts both dark and mid-tones toward a grimy hue.
    anchor_r = (110 * (1 - soot_frac) + 50 * soot_frac + fine_noise * 10) / 255
    anchor_g = (88 * (1 - soot_frac) + 42 * soot_frac + fine_noise * 8) / 255
    anchor_b = (60 * (1 - soot_frac) + 35 * soot_frac + fine_noise * 5) / 255

    # Base strength raised so min severity (field≈0.3) blends ~25% toward
    # the grime anchor — enough to be visible on dark content.
    raw_str = field * (0.45 + 0.55 * patch_noise) * (0.85 + 0.15 * fine_noise)
    strength = (raw_str * 1.1).clamp(max=0.85)

    active = field > 0.01
    out[0] = torch.where(active, out[0] * (1 - strength) + anchor_r * strength, out[0])
    out[1] = torch.where(active, out[1] * (1 - strength) + anchor_g * strength, out[1])
    out[2] = torch.where(active, out[2] * (1 - strength) + anchor_b * strength, out[2])

    return out.clamp(0, 1)


def apply_scratches(image: torch.Tensor, mask: torch.Tensor,
                    generator: torch.Generator = None,
                    max_count: Optional[int] = None) -> torch.Tensor:
    """Thin linear abrasion marks.

    Instead of walking pixel-by-pixel through the soft mask (which fails
    at low severity because the cone profile is too dim), this draws
    scratch lines directly inside each connected component of the binary
    mask region. Per-component principal axis keeps each scratch aligned
    with its local band.

    Scratch colour always contrasts with the local painting: light
    scratches on dark areas, dark scratches on light areas.
    """
    C, H, W = image.shape
    out = image.clone()

    mask_max = float(mask.max().item())
    if mask_max < 0.01:
        return out

    region = mask > 0.005
    if not region.any():
        return out

    # Find connected components so each scratch aligns with its local band.
    components = _mask_component_centers(mask, min_val=0.005)
    if not components:
        return out

    # Build per-component pixel lists and local angles.
    region_np = region.detach().cpu()
    from scipy.ndimage import label as _cc_label
    labels_arr, n_comps = _cc_label(region_np.numpy())
    labels_t = torch.from_numpy(labels_arr)

    comp_data = []  # list of (pixels_ys, pixels_xs, local_angle)
    for comp_id in range(1, n_comps + 1):
        comp_mask = labels_t == comp_id
        cys, cxs = comp_mask.nonzero(as_tuple=True)
        if cys.shape[0] < 10:
            continue
        # Local principal axis for this component.
        cys_f = cys.float()
        cxs_f = cxs.float()
        cx_m, cy_m = cxs_f.mean(), cys_f.mean()
        dx_c = cxs_f - cx_m
        dy_c = cys_f - cy_m
        Sxx = float((dx_c * dx_c).mean())
        Sxy = float((dx_c * dy_c).mean())
        Syy = float((dy_c * dy_c).mean())
        local_angle = 0.5 * math.atan2(2.0 * Sxy, Sxx - Syy)
        comp_data.append((cys, cxs, local_angle))

    if not comp_data:
        return out

    # Decide total scratch count, spread across components proportionally.
    total_region = sum(d[0].shape[0] for d in comp_data)
    base_count = max(len(comp_data) * 2, int(total_region / (H * W) * 200))
    base_count += int(torch.randint(0, 3, (1,), generator=generator).item())
    if max_count is not None:
        base_count = min(base_count, max(1, int(max_count)))

    sev_opacity = max(0.65, mask_max)

    for comp_ys, comp_xs, local_angle in comp_data:
        n_pixels = comp_ys.shape[0]
        n_here = max(1, int(round(base_count * n_pixels / total_region)))

        # Scratch colour: a consistent warm off-white that represents the
        # exposed ground/gesso layer beneath the paint. Real scratches
        # always reveal the lighter substrate regardless of the paint colour
        # on top. Using one fixed tone avoids the black-vs-white
        # inconsistency that happened when we adapted per-pixel or
        # per-component to luminance.
        comp_scratch_r, comp_scratch_g, comp_scratch_b = 0.88, 0.84, 0.76

        for _ in range(n_here):
            idx = int(torch.randint(0, n_pixels, (1,), generator=generator).item())
            sx = int(comp_xs[idx].item())
            sy = int(comp_ys[idx].item())

            angle = local_angle + (torch.rand(1, generator=generator).item() - 0.5) * math.pi * 0.12

            comp_span = float(((comp_ys.max() - comp_ys.min()) ** 2 +
                               (comp_xs.max() - comp_xs.min()) ** 2) ** 0.5)
            scratch_len = max(30, int(comp_span * (0.4 + torch.rand(1, generator=generator).item() * 0.6)))
            scratch_len = min(scratch_len, max(H, W))

            width = 2 if torch.rand(1, generator=generator).item() < 0.6 else 3
            half_w = width // 2
            alpha = 0.60 + torch.rand(1, generator=generator).item() * 0.30

            # Small chance of a burnished (bright) scratch for variety,
            # otherwise use the component-level colour.
            use_bright = torch.rand(1, generator=generator).item() < 0.15
            if use_bright:
                scratch_r, scratch_g, scratch_b = 0.95, 0.93, 0.88
            else:
                scratch_r = comp_scratch_r
                scratch_g = comp_scratch_g
                scratch_b = comp_scratch_b

            # Walk and paint, constrained to the binary region so the
            # scratch stays inside the mask and the training hull mask
            # stays tight around the input band.
            px, py2 = float(sx), float(sy)
            step_size = 1.5
            max_step_curve = 0.15 / max(40.0, float(scratch_len))
            curvature = (torch.rand(1, generator=generator).item() - 0.5) * max_step_curve
            outside_streak = 0

            for step_i in range(scratch_len):
                angle += curvature
                px += math.cos(angle) * step_size
                py2 += math.sin(angle) * step_size

                ix, iy = int(round(px)), int(round(py2))
                if ix < 0 or ix >= W or iy < 0 or iy >= H:
                    break

                # Only paint inside the region so mask and damage align.
                if not region_np[iy, ix]:
                    outside_streak += 1
                    if outside_streak > 8:
                        break
                    continue
                outside_streak = 0

                t = step_i / max(1, scratch_len - 1)
                taper = max(0.35, 1.0 - (2.0 * t - 1.0) ** 2)
                a = min(1.0, alpha * sev_opacity * taper)

                for dy in range(-half_w, half_w + 1):
                    for dx in range(-half_w, half_w + 1):
                        nx, ny = ix + dx, iy + dy
                        if 0 <= nx < W and 0 <= ny < H:
                            out[0, ny, nx] = out[0, ny, nx] * (1 - a) + scratch_r * a
                            out[1, ny, nx] = out[1, ny, nx] * (1 - a) + scratch_g * a
                            out[2, ny, nx] = out[2, ny, nx] * (1 - a) + scratch_b * a

    return out.clamp(0, 1)
