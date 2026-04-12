"""Individual corruption effect implementations.

All effects operate on (3, H, W) float32 tensors in [0, 1] with a
(H, W) float32 mask in [0, 1] controlling per-pixel intensity.
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple

from .color import rgb_to_lab, lab_to_rgb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_blur_2d(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to a 2D field (H, W) or (C, H, W).

    Uses separable 1D convolutions for efficiency.
    """
    if sigma < 0.5:
        return field
    ksize = int(math.ceil(sigma * 6)) | 1  # ensure odd
    ksize = max(3, ksize)
    x = torch.arange(ksize, dtype=field.dtype, device=field.device) - ksize // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    needs_batch = field.dim() == 2
    if needs_batch:
        field = field.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif field.dim() == 3:
        field = field.unsqueeze(0)  # (1, C, H, W)

    C = field.shape[1]
    pad = ksize // 2

    # Horizontal
    kh = kernel_1d.view(1, 1, 1, ksize).expand(C, -1, -1, -1)
    out = F.conv2d(F.pad(field, (pad, pad, 0, 0), mode='reflect'), kh, groups=C)
    # Vertical
    kv = kernel_1d.view(1, 1, ksize, 1).expand(C, -1, -1, -1)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode='reflect'), kv, groups=C)

    if needs_batch:
        return out.squeeze(0).squeeze(0)
    else:
        return out.squeeze(0)


def make_noise(H: int, W: int, blur_sigma: float,
               generator: torch.Generator = None,
               device: torch.device = None,
               normalize: bool = True) -> torch.Tensor:
    """Generate smoothed noise field (H, W) in [0, 1].

    Random noise → Gaussian blur → normalize to [0, 1].
    """
    noise = torch.rand(H, W, generator=generator, device=device or torch.device('cpu'))
    noise = gaussian_blur_2d(noise, blur_sigma)
    if normalize:
        lo, hi = noise.min(), noise.max()
        span = (hi - lo).clamp(min=1e-6)
        noise = (noise - lo) / span
    return noise


def box_blur_float(field: torch.Tensor, radius: int) -> torch.Tensor:
    """Simple box blur on (H, W) field. Used for mask smoothing."""
    if radius < 1:
        return field
    ksize = 2 * radius + 1
    kernel = torch.ones(1, 1, ksize, ksize, device=field.device, dtype=field.dtype) / (ksize * ksize)
    f = field.unsqueeze(0).unsqueeze(0)
    return F.conv2d(F.pad(f, (radius, radius, radius, radius), mode='reflect'), kernel).squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Effect implementations
# ---------------------------------------------------------------------------

def apply_linear_cracks(image: torch.Tensor, mask: torch.Tensor,
                        generator: torch.Generator = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply linear/structural cracks common in plaster and fresco.

    Generates 1-3 long jagged lines traversing large portions of the image,
    simulating structural cracks from wall movement. Each crack is a random
    walk across the image with slight angular wandering and varying width.

    Returns (corrupted_image, crack_mask) where crack_mask is binary (H, W).
    """
    C, H, W = image.shape
    out = image.clone()
    crack_mask = torch.zeros(H, W, device=image.device)

    mask_sum = mask.sum().item()
    if mask_sum < 1:
        return out, crack_mask

    num_cracks = 1 + int(torch.randint(0, 3, (1,), generator=generator).item())

    for _ in range(num_cracks):
        # Pick start point near an edge
        edge_side = int(torch.randint(0, 4, (1,), generator=generator).item())
        if edge_side == 0:  # top
            sx = int(torch.randint(0, W, (1,), generator=generator).item())
            sy = int(torch.randint(0, max(1, H // 10), (1,), generator=generator).item())
        elif edge_side == 1:  # bottom
            sx = int(torch.randint(0, W, (1,), generator=generator).item())
            sy = H - 1 - int(torch.randint(0, max(1, H // 10), (1,), generator=generator).item())
        elif edge_side == 2:  # left
            sx = int(torch.randint(0, max(1, W // 10), (1,), generator=generator).item())
            sy = int(torch.randint(0, H, (1,), generator=generator).item())
        else:  # right
            sx = W - 1 - int(torch.randint(0, max(1, W // 10), (1,), generator=generator).item())
            sy = int(torch.randint(0, H, (1,), generator=generator).item())

        # Initial angle pointing roughly toward center
        cx_center, cy_center = W / 2.0, H / 2.0
        angle = math.atan2(cy_center - sy, cx_center - sx)
        # Add some randomness to initial angle
        angle += (torch.rand(1, generator=generator).item() - 0.5) * math.pi * 0.5

        # Walk across image
        px, py = float(sx), float(sy)
        step_size = 2.0
        max_steps = int(max(H, W) * 1.2)

        for step in range(max_steps):
            # Random angular wandering
            angle += (torch.rand(1, generator=generator).item() - 0.5) * 0.3

            px += math.cos(angle) * step_size
            py += math.sin(angle) * step_size

            ix, iy = int(round(px)), int(round(py))
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                break

            # Check mask
            if mask[iy, ix].item() < 0.05:
                continue

            # Varying width (1-3 pixels)
            width = 1 + int(torch.rand(1, generator=generator).item() * 2.5)
            half_w = width // 2

            for dy in range(-half_w, half_w + 1):
                for dx in range(-half_w, half_w + 1):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        m_val = mask[ny, nx].item()
                        if m_val > 0.05:
                            # Darken along crack
                            k = 1.0 - 0.55 * m_val
                            out[0, ny, nx] *= k
                            out[1, ny, nx] *= k
                            out[2, ny, nx] *= k
                            crack_mask[ny, nx] = 1.0

    return out, crack_mask


def apply_cracks(image: torch.Tensor, mask: torch.Tensor,
                 generator: torch.Generator = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply craquelure cracks — Voronoi, linear, or mixed.

    Returns (corrupted_image, crack_mask) where crack_mask is binary (H, W).

    ~30% chance of linear-only cracks, ~15% chance of mixed (both linear
    and Voronoi), ~55% chance of Voronoi-only. Linear cracks simulate
    structural damage from wall movement. Voronoi cracks simulate age
    craquelure from paint drying/shrinkage.
    """
    roll = torch.rand(1, generator=generator).item()

    if roll < 0.30:
        # Linear cracks only
        return apply_linear_cracks(image, mask, generator=generator)
    elif roll < 0.45:
        # Mixed: apply linear first, then Voronoi on top
        out, crack_mask = apply_linear_cracks(image, mask, generator=generator)
        out, voronoi_mask = _apply_voronoi_cracks(out, mask, generator=generator)
        crack_mask = (crack_mask + voronoi_mask).clamp(max=1.0)
        return out, crack_mask
    else:
        # Voronoi only (original behavior)
        return _apply_voronoi_cracks(image, mask, generator=generator)


def _apply_voronoi_cracks(image: torch.Tensor, mask: torch.Tensor,
                          generator: torch.Generator = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Voronoi craquelure cracks.

    Returns (corrupted_image, crack_mask) where crack_mask is binary (H, W).

    Voronoi tessellation with nearest-two-site distance difference for
    edge detection. Grid-based spatial lookup for minimum site spacing.
    Edge width scales with cell size.
    """
    C, H, W = image.shape
    out = image.clone()
    crack_mask = torch.zeros(H, W, device=image.device)

    mask_sum = mask.sum().item()
    if mask_sum < 1:
        return out, crack_mask

    mean_mask = mask_sum / (H * W)
    target = max(40, int(mean_mask * H * W / 200))
    min_dist = 8
    min_dist2 = min_dist * min_dist

    # Grid-based spatial lookup
    grid_cell = min_dist
    grid_w = math.ceil(W / grid_cell)
    grid_h = math.ceil(H / grid_cell)
    grid = [[] for _ in range(grid_w * grid_h)]

    sites = []
    mask_np = mask.cpu()
    attempts = 0
    max_attempts = target * 50

    while len(sites) < target and attempts < max_attempts:
        attempts += 1
        x = int(torch.randint(0, W, (1,), generator=generator).item())
        y = int(torch.randint(0, H, (1,), generator=generator).item())
        m = mask_np[y, x].item()
        if torch.rand(1, generator=generator).item() < m:
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
        return out, crack_mask

    # Convert sites to tensor for vectorized distance computation
    sites_t = torch.tensor(sites, dtype=torch.float32, device=image.device)  # (N, 2)

    # Process in chunks to avoid OOM on large images
    chunk_size = 4096
    out_flat = out.view(3, -1).clone()  # (3, H*W)
    mask_flat = mask.view(-1)
    crack_flat = crack_mask.view(-1)

    for start in range(0, H * W, chunk_size):
        end = min(start + chunk_size, H * W)
        indices = torch.arange(start, end, device=image.device)
        ys = indices // W
        xs = indices % W

        # Skip pixels with low mask
        m_vals = mask_flat[start:end]
        active = m_vals >= 20.0 / 255.0
        if not active.any():
            continue

        active_idx = active.nonzero(as_tuple=True)[0]
        ax = xs[active_idx].float()
        ay = ys[active_idx].float()

        # Distances to all sites: (num_active, num_sites)
        dx = ax.unsqueeze(1) - sites_t[:, 0].unsqueeze(0)
        dy = ay.unsqueeze(1) - sites_t[:, 1].unsqueeze(0)
        dists = dx ** 2 + dy ** 2

        # Find two nearest sites
        top2 = dists.topk(2, dim=1, largest=False)
        sd1 = top2.values[:, 0].sqrt()
        sd2 = top2.values[:, 1].sqrt()
        diff = sd2 - sd1

        # Edge width proportional to cell size
        edge_width = (sd1 * 0.12).clamp(max=0.8)
        is_edge = (diff < edge_width) & (sd1 > 1.5)

        if is_edge.any():
            edge_idx = active_idx[is_edge]
            global_idx = start + edge_idx
            a = m_vals[edge_idx]
            d = diff[is_edge]
            ew = edge_width[is_edge]
            k = 1.0 - 0.55 * a * (1.0 - d / ew)

            out_flat[0, global_idx] *= k
            out_flat[1, global_idx] *= k
            out_flat[2, global_idx] *= k
            crack_flat[global_idx] = 1.0

    out = out_flat.view(3, H, W)
    crack_mask = crack_flat.view(H, W)
    return out, crack_mask


def apply_paint_loss_crack(image: torch.Tensor, mask: torch.Tensor,
                           crack_mask: torch.Tensor,
                           generator: torch.Generator = None) -> torch.Tensor:
    """Crack-associated paint loss: flakes along crack edges.

    Blur crack mask for organic falloff, smooth noise for flake regions,
    replace with substrate color.
    """
    C, H, W = image.shape
    out = image.clone()

    if crack_mask.sum() < 1 or mask.sum() < 1:
        return out

    # Sample canvas/substrate color
    cr, cg, cb = _sample_canvas_color(image)

    # Blur crack mask for organic falloff
    flake_field = gaussian_blur_2d(crack_mask, 3.0)

    # Smooth noise field
    noise_field = make_noise(H, W, 4.0, generator=generator, device=image.device)

    for i_y in range(H):
        for i_x in range(W):
            ff = flake_field[i_y, i_x].item()
            if ff < 0.05:
                continue
            um = mask[i_y, i_x].item()
            if um < 0.05:
                continue
            nv = noise_field[i_y, i_x].item()
            combined = ff * um
            threshold = combined * 0.5
            if nv > threshold + 0.25:
                continue
            edge = 1.0 - max(0.0, min(1.0, (nv - threshold) / 0.25))
            a = min(0.95, combined * edge)
            if a < 0.02:
                continue
            j = (torch.rand(1, generator=generator).item() - 0.5) * 12.0 / 255.0
            out[0, i_y, i_x] = out[0, i_y, i_x] * (1 - a) + (cr + j) * a
            out[1, i_y, i_x] = out[1, i_y, i_x] * (1 - a) + (cg + j) * a
            out[2, i_y, i_x] = out[2, i_y, i_x] * (1 - a) + (cb + j) * a

    return out.clamp(0, 1)


def apply_paint_loss_region(image: torch.Tensor, mask: torch.Tensor,
                            generator: torch.Generator = None) -> torch.Tensor:
    """Paint loss in user-defined regions (not crack-associated).

    Two noise scales for patch shapes and ragged edges, threshold
    from mask intensity. ~30% chance of edge-peeling mode where paint
    loss follows edges/corners, peeling inward from painting borders.
    """
    C, H, W = image.shape
    out = image.clone()

    if mask.sum() < 1:
        return out

    cr, cg, cb = _sample_canvas_color(image)

    # ~30% chance of edge-peeling mode
    edge_peel = torch.rand(1, generator=generator).item() < 0.30
    if edge_peel:
        # Create gradient mask strongest at edges/corners
        y_coords = torch.arange(H, device=image.device).float().unsqueeze(1).expand(H, W)
        x_coords = torch.arange(W, device=image.device).float().unsqueeze(0).expand(H, W)

        # Distance from each border, normalized to [0, 1]
        dist_top = y_coords / H
        dist_bot = (H - 1 - y_coords) / H
        dist_left = x_coords / W
        dist_right = (W - 1 - x_coords) / W

        # Minimum distance to any edge
        edge_dist = torch.min(torch.min(dist_top, dist_bot), torch.min(dist_left, dist_right))

        # Edge gradient: strong near edges, fading inward
        # Max peel depth ~20% of image dimension
        peel_depth = 0.15 + torch.rand(1, generator=generator).item() * 0.10
        edge_gradient = (1.0 - (edge_dist / peel_depth).clamp(max=1.0))

        # Corners get extra boost (product of two edge proximities)
        corner_boost = (1.0 - (dist_top / peel_depth).clamp(max=1.0)) * \
                       (1.0 - (dist_left / peel_depth).clamp(max=1.0)) + \
                       (1.0 - (dist_top / peel_depth).clamp(max=1.0)) * \
                       (1.0 - (dist_right / peel_depth).clamp(max=1.0)) + \
                       (1.0 - (dist_bot / peel_depth).clamp(max=1.0)) * \
                       (1.0 - (dist_left / peel_depth).clamp(max=1.0)) + \
                       (1.0 - (dist_bot / peel_depth).clamp(max=1.0)) * \
                       (1.0 - (dist_right / peel_depth).clamp(max=1.0))
        corner_boost = corner_boost.clamp(max=1.0) * 0.4

        edge_gradient = (edge_gradient + corner_boost).clamp(max=1.0)

        # Use noise to make peeling boundary irregular
        peel_noise = make_noise(H, W, 6.0, generator=generator, device=image.device)
        edge_gradient = edge_gradient * (0.5 + 0.5 * peel_noise)

        # Combine with user mask
        mask = (mask * edge_gradient).clamp(0, 1)

    # Low-frequency noise (patch shapes)
    noise_low = make_noise(H, W, 10.0, generator=generator, device=image.device)
    # High-frequency noise (edge irregularity)
    noise_hi = make_noise(H, W, 2.0, generator=generator, device=image.device, normalize=False)

    # Compute flake field
    m = mask  # (H, W)
    thresh = (m * m * 0.4 + m * 0.58).clamp(max=0.995)
    sample = noise_low - (noise_hi - 0.5) * 0.22 * m
    is_flake = (sample < thresh) & (m >= 0.04)

    depth = ((thresh - sample) / thresh.clamp(min=0.02)).clamp(0, 1)
    flake = (depth * (0.3 + 0.7 * m)).clamp(0, 1) * is_flake.float()

    # Render: replace with substrate
    active = flake >= 0.04
    if not active.any():
        return out

    a = (flake * 1.05).clamp(0, 1)
    # Rim darkening
    rim = torch.where((flake > 0.1) & (flake < 0.5),
                      (1.0 - (flake - 0.3).abs() / 0.2) * 0.35, torch.zeros_like(flake))

    j = (torch.rand(H, W, generator=generator, device=image.device) - 0.5) * 18.0 / 255.0
    tr = ((cr + j) * (1.0 - rim)).clamp(0, 1)
    tg = ((cg + j) * (1.0 - rim)).clamp(0, 1)
    tb = ((cb + j) * (1.0 - rim)).clamp(0, 1)

    out[0] = torch.where(active, out[0] * (1 - a) + tr * a, out[0])
    out[1] = torch.where(active, out[1] * (1 - a) + tg * a, out[1])
    out[2] = torch.where(active, out[2] * (1 - a) + tb * a, out[2])

    return out.clamp(0, 1)


def apply_yellowing(image: torch.Tensor, mask: torch.Tensor,
                    generator: torch.Generator = None) -> torch.Tensor:
    """Varnish yellowing via CIELAB a/b shift.

    Smooth mask, noise-modulated positive offsets to a and b channels.
    """
    C, H, W = image.shape
    out = image.clone()

    # Smooth the mask
    field = box_blur_float(mask, 25)
    # Noise field
    noise = make_noise(H, W, 30.0, generator=generator, device=image.device, normalize=False)

    active = field > 0.01
    if not active.any():
        return out

    # Convert to LAB: need (H, W, 3)
    rgb_hwc = out.permute(1, 2, 0)  # (H, W, 3)
    lab = rgb_to_lab(rgb_hwc)

    m = field
    offset_b = (16.0 + 36.0 * noise) * m  # toward yellow
    offset_a = (4.0 + 12.0 * noise) * m   # slight reddish

    lab[..., 1] = lab[..., 1] + offset_a * active.float()
    lab[..., 2] = lab[..., 2] + offset_b * active.float()

    rgb_out = lab_to_rgb(lab)
    out = rgb_out.permute(2, 0, 1).clamp(0, 1)
    return out


def apply_stains(image: torch.Tensor, mask: torch.Tensor,
                 generator: torch.Generator = None) -> torch.Tensor:
    """Water stains with tide lines.

    Elliptical stain blobs with brown interior and darker tide-line ring
    at evaporation edge.
    """
    C, H, W = image.shape
    out = image.clone()

    mask_sum = mask.sum().item()
    if mask_sum < 1:
        return out

    mask_area = mask_sum
    num_stains = max(2, int(mask_area / 800))

    fill_field = torch.zeros(H, W, device=image.device)
    edge_field = torch.zeros(H, W, device=image.device)

    mask_cpu = mask.cpu()

    for n in range(num_stains):
        cx, cy = None, None
        for _ in range(300):
            x = int(torch.randint(0, W, (1,), generator=generator).item())
            y = int(torch.randint(0, H, (1,), generator=generator).item())
            if torch.rand(1, generator=generator).item() < mask_cpu[y, x].item():
                cx, cy = x, y
                break
        if cx is None:
            continue

        rx = 20 + torch.rand(1, generator=generator).item() * 60
        ry = 20 + torch.rand(1, generator=generator).item() * 50
        angle = torch.rand(1, generator=generator).item() * math.pi
        ca, sa = math.cos(angle), math.sin(angle)
        stain_str = 0.4 + torch.rand(1, generator=generator).item() * 0.5

        max_r = max(rx, ry) * 1.6
        x0 = max(0, int(cx - max_r))
        x1 = min(W - 1, int(cx + max_r))
        y0 = max(0, int(cy - max_r))
        y1 = min(H - 1, int(cy + max_r))

        ys = torch.arange(y0, y1 + 1, device=image.device).float()
        xs = torch.arange(x0, x1 + 1, device=image.device).float()
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        dx = xx - cx
        dy = yy - cy
        u = (dx * ca + dy * sa) / rx
        v = (-dx * sa + dy * ca) / ry
        d = (u ** 2 + v ** 2).sqrt()

        in_range = d <= 1.5
        fill = (1.0 - d).clamp(min=0) * stain_str
        fill_field[y0:y1+1, x0:x1+1] = torch.min(
            (fill_field[y0:y1+1, x0:x1+1] + fill * in_range.float()).clamp(max=1.0),
            torch.ones_like(fill)
        )

        edge_dist = (d - 0.9).abs()
        tide_mask = (edge_dist < 0.2) & in_range
        tide_line = (1.0 - edge_dist / 0.2) * stain_str * 1.2
        edge_field[y0:y1+1, x0:x1+1] = (
            edge_field[y0:y1+1, x0:x1+1] + tide_line * tide_mask.float()
        ).clamp(max=1.0)

    # Drip patterns: gravity-driven vertical elongation below some stains
    drip_field = torch.zeros(H, W, device=image.device)
    use_drips = torch.rand(1, generator=generator).item() < 0.40
    if use_drips:
        num_drips = max(1, num_stains // 2)
        for _ in range(num_drips):
            # Pick a random start point within stain regions
            dx_start, dy_start = None, None
            for __ in range(200):
                x = int(torch.randint(0, W, (1,), generator=generator).item())
                y = int(torch.randint(0, H, (1,), generator=generator).item())
                if fill_field[y, x].item() > 0.1:
                    dx_start, dy_start = x, y
                    break
            if dx_start is None:
                continue

            # Drip length: 10-40% of image height downward
            drip_len = int(H * (0.10 + torch.rand(1, generator=generator).item() * 0.30))
            drip_width = 2 + int(torch.rand(1, generator=generator).item() * 4)
            drip_str = 0.3 + torch.rand(1, generator=generator).item() * 0.4

            px = float(dx_start)
            for dy_off in range(drip_len):
                py = dy_start + dy_off
                if py >= H:
                    break
                # Slight horizontal wandering
                px += (torch.rand(1, generator=generator).item() - 0.5) * 1.5
                ix = int(round(px))
                ix = max(0, min(W - 1, ix))

                # Fade out toward end of drip
                fade = 1.0 - (dy_off / drip_len) ** 0.7
                half_w = drip_width // 2

                for ddx in range(-half_w, half_w + 1):
                    nx = ix + ddx
                    if 0 <= nx < W:
                        dist_from_center = abs(ddx) / max(1, half_w)
                        lateral_fade = 1.0 - dist_from_center ** 2
                        val = drip_str * fade * lateral_fade
                        drip_field[py, nx] = max(drip_field[py, nx].item(), val)

    # Stain colors (float [0,1])
    stain_r, stain_g, stain_b = 140/255, 105/255, 60/255
    tide_r, tide_g, tide_b = 90/255, 65/255, 35/255

    m = mask
    fill = fill_field.clamp(max=0.7) * m
    edge = edge_field.clamp(max=0.85) * m
    active = (fill > 0.01) | (edge > 0.01)

    if active.any():
        # Interior stain
        f5 = fill * 0.5
        f55 = fill * 0.55
        f6 = fill * 0.6
        out[0] = torch.where(active & (fill > 0.01), out[0] * (1 - f5) + stain_r * f5, out[0])
        out[1] = torch.where(active & (fill > 0.01), out[1] * (1 - f55) + stain_g * f55, out[1])
        out[2] = torch.where(active & (fill > 0.01), out[2] * (1 - f6) + stain_b * f6, out[2])

        # Tide line
        ea = edge * 0.6
        out[0] = torch.where(active & (edge > 0.01), out[0] * (1 - ea) + tide_r * ea, out[0])
        out[1] = torch.where(active & (edge > 0.01), out[1] * (1 - ea) + tide_g * ea, out[1])
        out[2] = torch.where(active & (edge > 0.01), out[2] * (1 - ea) + tide_b * ea, out[2])

    # Drip stains
    drip_active = drip_field > 0.01
    if drip_active.any():
        drip_a = (drip_field * m * 0.6).clamp(max=0.7)
        drip_color_r, drip_color_g, drip_color_b = 120/255, 85/255, 45/255
        out[0] = torch.where(drip_active, out[0] * (1 - drip_a) + drip_color_r * drip_a, out[0])
        out[1] = torch.where(drip_active, out[1] * (1 - drip_a) + drip_color_g * drip_a, out[1])
        out[2] = torch.where(drip_active, out[2] * (1 - drip_a) + drip_color_b * drip_a, out[2])

    return out.clamp(0, 1)


def apply_fading(image: torch.Tensor, mask: torch.Tensor,
                 generator: torch.Generator = None) -> torch.Tensor:
    """Photochemical fading: desaturation + bleaching.

    Smooth mask, noise-modulated desaturation toward luminance,
    contrast compression toward chalky off-white.
    """
    C, H, W = image.shape
    out = image.clone()

    # Smooth mask
    field = box_blur_float(mask, 12)
    # Noise
    fade_noise = make_noise(H, W, 20.0, generator=generator, device=image.device, normalize=False)

    active = field > 0.01
    if not active.any():
        return out

    m = field.clamp(max=1.0)  # direct mapping, no amplification (3x reduction)
    bleach_r, bleach_g, bleach_b = 215/255, 205/255, 195/255

    n = 0.6 + 0.4 * fade_noise
    strength = m * n

    r, g, b = out[0], out[1], out[2]
    lum = r * 0.299 + g * 0.587 + b * 0.114

    # Desaturate
    desat_r = r + (lum - r) * strength * 0.85
    desat_g = g + (lum - g) * strength * 0.85
    desat_b = b + (lum - b) * strength * 0.85

    # Bleach
    bleach = strength * 0.6
    out_r = desat_r * (1 - bleach) + bleach_r * bleach
    out_g = desat_g * (1 - bleach) + bleach_g * bleach
    out_b = desat_b * (1 - bleach) + bleach_b * bleach

    out[0] = torch.where(active, out_r, out[0])
    out[1] = torch.where(active, out_g, out[1])
    out[2] = torch.where(active, out_b, out[2])

    return out.clamp(0, 1)


def apply_bloom(image: torch.Tensor, mask: torch.Tensor,
                generator: torch.Generator = None) -> torch.Tensor:
    """Bloom / haze from degraded varnish.

    Multi-radius glow with screen blend, detail softening,
    milky warm haze overlay.
    """
    C, H, W = image.shape
    out = image.clone()

    if mask.max() < 0.02:
        return out

    # 1. Extract luminance for glow source
    lum = (out[0] * 0.299 + out[1] * 0.587 + out[2] * 0.114)  # (H, W)
    contrib = lum  # low knee: all pixels contribute
    hi_img = out * contrib.unsqueeze(0)  # (3, H, W)

    # 2. Multi-radius blur
    glow1 = gaussian_blur_2d(hi_img, 8.0)   # tight glow
    glow2 = gaussian_blur_2d(hi_img, 20.0)  # wide haze
    glow3 = gaussian_blur_2d(hi_img, 40.0)  # atmospheric

    bloom = glow1 * 0.3 + glow2 * 0.4 + glow3 * 0.3  # (3, H, W)

    # 3. Softened version for detail loss
    softened = gaussian_blur_2d(out, 6.0)

    # 4. Apply per pixel
    m = mask.unsqueeze(0)  # (1, H, W)

    # Screen blend
    bloom_str = m * 0.9
    screen = 1.0 - (1.0 - out) * (1.0 - bloom * bloom_str)

    # Detail softening
    soft_mix = m * 0.55
    result = screen * (1.0 - soft_mix) + softened * soft_mix

    # Milky haze overlay
    haze = torch.tensor([210/255, 200/255, 185/255], device=image.device).view(3, 1, 1)
    haze_mix = m * m * 0.3
    result = result * (1.0 - haze_mix) + haze * haze_mix

    active = mask.unsqueeze(0) >= 0.02
    out = torch.where(active, result, out)
    return out.clamp(0, 1)


def apply_deposits(image: torch.Tensor, mask: torch.Tensor,
                   generator: torch.Generator = None) -> torch.Tensor:
    """Surface deposits: grime, soot, and salt efflorescence.

    Contrast-reduction veil model with patchy noise, vertical soot bias,
    and salt crystalline patches.
    """
    C, H, W = image.shape
    out = image.clone()

    # Smooth mask
    field = box_blur_float(mask, 6)

    if field.max() < 0.01:
        return out

    # Corner/edge recess deposits: grime accumulates more near borders
    use_corner_deposits = torch.rand(1, generator=generator).item() < 0.40
    if use_corner_deposits:
        y_coords = torch.arange(H, device=image.device).float().unsqueeze(1).expand(H, W)
        x_coords = torch.arange(W, device=image.device).float().unsqueeze(0).expand(H, W)

        dist_top = y_coords / H
        dist_bot = (H - 1 - y_coords) / H
        dist_left = x_coords / W
        dist_right = (W - 1 - x_coords) / W

        edge_dist = torch.min(torch.min(dist_top, dist_bot), torch.min(dist_left, dist_right))

        # Stronger deposits within ~15% of edges
        edge_boost = (1.0 - (edge_dist / 0.15).clamp(max=1.0)) * 0.4

        # Extra boost in corners
        corner_proximity = torch.min(
            torch.min(dist_top, dist_bot),
            torch.min(dist_left, dist_right)
        )
        # Corners: where two edges are both close
        corner_mask = (1.0 - (dist_top / 0.15).clamp(max=1.0)) * \
                      (1.0 - (dist_left / 0.15).clamp(max=1.0)) + \
                      (1.0 - (dist_top / 0.15).clamp(max=1.0)) * \
                      (1.0 - (dist_right / 0.15).clamp(max=1.0)) + \
                      (1.0 - (dist_bot / 0.15).clamp(max=1.0)) * \
                      (1.0 - (dist_left / 0.15).clamp(max=1.0)) + \
                      (1.0 - (dist_bot / 0.15).clamp(max=1.0)) * \
                      (1.0 - (dist_right / 0.15).clamp(max=1.0))
        corner_mask = corner_mask.clamp(max=1.0) * 0.3

        field = (field + edge_boost + corner_mask).clamp(0, 1)

    # Noise fields
    patch_noise = make_noise(H, W, 18.0, generator=generator, device=image.device)
    fine_noise = make_noise(H, W, 2.0, generator=generator, device=image.device)
    med_noise = make_noise(H, W, 8.0, generator=generator, device=image.device)
    salt_noise = make_noise(H, W, 4.0, generator=generator, device=image.device)

    # Vertical bias for soot (heavier at top)
    y_coords = torch.arange(H, device=image.device).float().unsqueeze(1).expand(H, W)
    vert_bias = 1.0 - (y_coords / H) * 0.3
    soot_frac = med_noise * vert_bias

    # Anchor colors (float [0,1])
    anchor_r = (95 * (1 - soot_frac) + 55 * soot_frac + fine_noise * 15) / 255
    anchor_g = (80 * (1 - soot_frac) + 50 * soot_frac + fine_noise * 10) / 255
    anchor_b = (60 * (1 - soot_frac) + 50 * soot_frac + fine_noise * 5) / 255

    # Deposit strength
    raw_str = field * (0.3 + 0.7 * patch_noise) * (0.85 + 0.15 * fine_noise)
    strength = (raw_str * 0.8).clamp(max=0.75)

    active = field > 0.01

    # Compress toward anchor (veil model)
    out[0] = torch.where(active, out[0] * (1 - strength) + anchor_r * strength, out[0])
    out[1] = torch.where(active, out[1] * (1 - strength) + anchor_g * strength, out[1])
    out[2] = torch.where(active, out[2] * (1 - strength) + anchor_b * strength, out[2])

    # Salt efflorescence
    salt_active = field > 0.15
    if salt_active.any():
        salt_thresh = 0.7 - field * 0.45
        salt_above = (salt_noise > salt_thresh) & salt_active
        if salt_above.any():
            salt_str = ((salt_noise - salt_thresh) / (1.0 - salt_thresh)).clamp(0, 1)
            salt_opacity = salt_str * field * 0.8

            warm = fine_noise > 0.5
            salt_r = torch.where(warm, torch.tensor(230/255, device=image.device),
                                 torch.tensor(220/255, device=image.device))
            salt_g = torch.where(warm, torch.tensor(225/255, device=image.device),
                                 torch.tensor(222/255, device=image.device))
            salt_b = torch.where(warm, torch.tensor(210/255, device=image.device),
                                 torch.tensor(225/255, device=image.device))

            out[0] = torch.where(salt_above, out[0] * (1 - salt_opacity) + salt_r * salt_opacity, out[0])
            out[1] = torch.where(salt_above, out[1] * (1 - salt_opacity) + salt_g * salt_opacity, out[1])
            out[2] = torch.where(salt_above, out[2] * (1 - salt_opacity) + salt_b * salt_opacity, out[2])

            # Sparkles
            sparkle_mask = (torch.rand(H, W, generator=generator, device=image.device)
                           < 0.15 * salt_str * field) & salt_above
            if sparkle_mask.any():
                sparkle = 0.15 + torch.rand(H, W, generator=generator, device=image.device) * 0.25
                out[0] = torch.where(sparkle_mask, (out[0] + 40/255 * sparkle).clamp(max=1), out[0])
                out[1] = torch.where(sparkle_mask, (out[1] + 38/255 * sparkle).clamp(max=1), out[1])
                out[2] = torch.where(sparkle_mask, (out[2] + 35/255 * sparkle).clamp(max=1), out[2])

    return out.clamp(0, 1)


def apply_scratches(image: torch.Tensor, mask: torch.Tensor,
                    generator: torch.Generator = None) -> torch.Tensor:
    """Surface scratches: thin linear marks from abrasion.

    Generates 2-8 scratch lines per application. Each is a thin line (1-2px)
    with slight curve, slightly lightening or revealing substrate.
    Orientations are random but slightly clustered.
    """
    C, H, W = image.shape
    out = image.clone()

    mask_sum = mask.sum().item()
    if mask_sum < 1:
        return out

    num_scratches = 2 + int(torch.randint(0, 7, (1,), generator=generator).item())

    # Base orientation for clustering: all scratches drift toward this angle
    base_angle = torch.rand(1, generator=generator).item() * math.pi

    # Substrate color for reveal
    cr, cg, cb = _sample_canvas_color(image)

    for _ in range(num_scratches):
        # Clustered angle: base_angle +/- up to 45 degrees
        angle = base_angle + (torch.rand(1, generator=generator).item() - 0.5) * math.pi * 0.5

        # Start point: random within image
        sx = int(torch.randint(0, W, (1,), generator=generator).item())
        sy = int(torch.randint(0, H, (1,), generator=generator).item())

        # Scratch length: 10-50% of max dimension
        max_dim = max(H, W)
        scratch_len = int(max_dim * (0.10 + torch.rand(1, generator=generator).item() * 0.40))

        # Width: 1-2px
        width = 1 + int(torch.rand(1, generator=generator).item() * 1.5)
        half_w = width // 2

        # Scratch intensity: how much to lighten
        scratch_str = 0.15 + torch.rand(1, generator=generator).item() * 0.25

        # Walk along scratch with slight curvature
        px, py = float(sx), float(sy)
        step_size = 1.5
        curvature = (torch.rand(1, generator=generator).item() - 0.5) * 0.05

        for step in range(scratch_len):
            angle += curvature
            px += math.cos(angle) * step_size
            py += math.sin(angle) * step_size

            ix, iy = int(round(px)), int(round(py))
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                break

            m_val = mask[iy, ix].item()
            if m_val < 0.05:
                continue

            for dy in range(-half_w, half_w + 1):
                for dx in range(-half_w, half_w + 1):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        local_m = mask[ny, nx].item()
                        if local_m < 0.05:
                            continue
                        a = scratch_str * local_m
                        # Lighten toward substrate (reveal)
                        out[0, ny, nx] = out[0, ny, nx] * (1 - a) + cr * a
                        out[1, ny, nx] = out[1, ny, nx] * (1 - a) + cg * a
                        out[2, ny, nx] = out[2, ny, nx] * (1 - a) + cb * a

    return out.clamp(0, 1)


def _sample_canvas_color(image: torch.Tensor) -> Tuple[float, float, float]:
    """Sample substrate/canvas color from image.

    Average color blended 15%/85% with light canvas tone [215, 200, 175]/255.
    """
    # Sample every 40th pixel
    flat = image.view(3, -1)[:, ::40]  # (3, N)
    mr, mg, mb = flat[0].mean().item(), flat[1].mean().item(), flat[2].mean().item()
    cr = min(1.0, mr * 0.15 + (215/255) * 0.85)
    cg = min(1.0, mg * 0.15 + (200/255) * 0.85)
    cb = min(1.0, mb * 0.15 + (175/255) * 0.85)
    return cr, cg, cb
