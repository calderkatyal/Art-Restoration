"""Color-space conversions: sRGB ↔ CIELAB, sRGB ↔ HSV.

All functions operate on float32 tensors with values in [0, 1] for RGB
and standard ranges for LAB (L: 0-100, a/b: ~-128 to +128).
"""

import torch
import math


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB [0,1] to CIELAB. Input shape: (..., 3)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Linearize sRGB
    def linearize(c):
        return torch.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)

    r, g, b = linearize(r), linearize(g), linearize(b)

    # To XYZ (D65 illuminant)
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    def f(t):
        return torch.where(t > 0.008856, t.clamp(min=1e-10).pow(1.0 / 3.0), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_out = 200.0 * (fy - fz)
    return torch.stack([L, a, b_out], dim=-1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Convert CIELAB to sRGB [0,1]. Input shape: (..., 3)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def finv(t):
        t3 = t ** 3
        return torch.where(t3 > 0.008856, t3, (t - 16.0 / 116.0) / 7.787)

    x = 0.95047 * finv(fx)
    y = finv(fy)
    z = 1.08883 * finv(fz)

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b_out = x * 0.0557 + y * -0.2040 + z * 1.0570

    def gamma(c):
        return torch.where(c > 0.0031308, 1.055 * c.clamp(min=1e-10).pow(1.0 / 2.4) - 0.055, 12.92 * c)

    return torch.stack([gamma(r), gamma(g), gamma(b_out)], dim=-1).clamp(0, 1)
