"""Inference via Euler ODE integration with hard data-consistency projection.

At each integration step:
    1. Predict velocity:  ``vel = v_θ(z_t, t | z_y, M')``
    2. Euler update:      ``z_t ← z_t + (t_next - t_curr) * vel``
    3. Data consistency:  ``z_t ← m_intact ⊙ z_y + m_dam ⊙ z_t``
       where ``m_dam = max_k(M'_k)`` (any damaged channel active) and ``m_intact = 1 - m_dam``.

Timesteps follow the FLUX.2 empirical SNR-shifted schedule from
:func:`~src.flux2.sampling.get_schedule`.

Typical CLI usage (once a plain ``state_dict`` export exists) is documented in the
``python -m src.inference`` block at the bottom of this file; training validation calls
:func:`sample` directly with a :class:`~src.model.RestorationDiT` checkpoint wrapped or
unwrapped by DeepSpeed.
"""

import torch

from .model import RestorationDiT
from .vae import FluxVAE
from .corruption import downsample_mask
from .flux2.sampling import get_schedule


@torch.no_grad()
def sample(
    model: RestorationDiT,
    vae: FluxVAE,
    corrupted_image: torch.Tensor,
    mask: torch.Tensor,
    null_emb: torch.Tensor,
    num_steps: int = 50,
    device: str = "cuda",
) -> torch.Tensor:
    """Restore a corrupted RGB batch by integrating the learned velocity field.

    Steps (latent space):
        1. ``z_y = vae.encode(corrupted_image)``
        2. ``M' = downsample_mask(mask, factor=vae.spatial_compression)``
        3. ``z_t ~ N(0, I)`` matching ``z_y`` shape
        4. ``timesteps = get_schedule(num_steps, H'*W')``
        5. For each adjacent pair ``(t_curr, t_next)`` in the schedule, apply Euler +
           :func:`data_consistency_step`.
        6. ``return vae.decode(z_t)`` clamped to ``[0, 1]``.

    Args:
        model:           Trained ``RestorationDiT`` in ``eval()`` mode.
        vae:             Frozen ``FluxVAE``.
        corrupted_image: ``(B, 3, H, W)`` in ``[0, 1]``; ``H, W`` divisible by 16.
        mask:            ``(B, K, H, W)`` damage tensor at **pixel** resolution.
        null_emb:        ``(1, 512, 7680)`` cached null text embedding.
        num_steps:       Number of Euler steps (more = slower, often sharper).
        device:          ``"cuda"`` or ``"cpu"`` string controlling tensor placement.

    Returns:
        ``(B, 3, H, W)`` ``float32`` restored RGB in ``[0, 1]``.
    """
    corrupted_image = corrupted_image.to(device)
    mask = mask.to(device)
    null_emb = null_emb.to(device)

    z_y = vae.encode(corrupted_image)
    m_lat = downsample_mask(mask, factor=vae.spatial_compression)
    b, _, h, w = z_y.shape
    z_t = torch.randn_like(z_y)
    seq_len = h * w
    timesteps = get_schedule(num_steps, seq_len)

    dtype = torch.bfloat16 if z_t.device.type == "cuda" else torch.float32
    z_t = z_t.to(dtype=dtype)
    z_y = z_y.to(dtype=dtype)
    m_lat = m_lat.to(dtype=dtype)
    null_b = null_emb.to(dtype=dtype)

    for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((b,), float(t_curr), device=device, dtype=dtype)
        vel = model(z_t, t_vec, z_y, m_lat, null_b)
        z_t = z_t + (float(t_next) - float(t_curr)) * vel
        z_t = data_consistency_step(z_t, z_y, m_lat)

    restored = vae.decode(z_t.float())
    return restored.clamp(0.0, 1.0)


def data_consistency_step(
    z_t: torch.Tensor,
    z_y: torch.Tensor,
    mask_latent: torch.Tensor,
) -> torch.Tensor:
    """Hard data-consistency projection in latent space (intact regions copy ``z_y``).

    Damaged vs intact is derived from **max** over mask channels at each spatial cell:
    if any damage type is active (value ``≥ 0.5``), that cell keeps the denoised ``z_t``;
    otherwise it is reset to the encoded damaged image ``z_y`` (preserves uncorrupted content).

    Args:
        z_t:          Current latent iterate ``(B, 128, H', W')``.
        z_y:          Encoded corrupted image, same shape as ``z_t``.
        mask_latent:  Downsampled ``(B, K, H', W')`` mask.

    Returns:
        Projected latent ``z_t`` with the same dtype/device as inputs.
    """
    m_dam = mask_latent.max(dim=1, keepdim=True).values
    m_dam = (m_dam >= 0.5).to(dtype=z_t.dtype)
    m_intact = 1.0 - m_dam
    return m_intact * z_y + m_dam * z_t


if __name__ == "__main__":
    pass
