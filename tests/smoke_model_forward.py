"""Offline forward-pass smoke test with a tiny random-weight FLUX model.

This does not load any pretrained weights. It only checks that the restoration-style
input path

    [z_t, z_y, mask] -> tokenization -> Flux2 forward -> reshape

produces the expected output shape.

Usage:
    ./.venv/bin/python tests/smoke_model_forward.py
    ./.venv/bin/python tests/smoke_model_forward.py --batch-size 2 --latent-size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.flux2.model import Flux2, Flux2Params
from src.flux2.sampling import batched_prc_img, batched_prc_txt


def build_tiny_flux(device: torch.device) -> Flux2:
    params = Flux2Params(
        in_channels=128,
        context_in_dim=64,
        hidden_size=64,
        num_heads=8,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[2, 2, 2, 2],
        use_guidance_embed=False,
    )
    model = Flux2(params).to(device=device)
    model.img_in = nn.Linear(264, params.hidden_size, bias=False, device=device)
    return model.eval()


@torch.no_grad()
def run_smoke_test(batch_size: int, latent_size: int, text_tokens: int, device: torch.device) -> None:
    torch.manual_seed(0)

    model = build_tiny_flux(device)
    z_t = torch.randn(batch_size, 128, latent_size, latent_size, device=device)
    z_y = torch.randn_like(z_t)
    mask = torch.randint(
        low=0,
        high=2,
        size=(batch_size, 8, latent_size, latent_size),
        device=device,
        dtype=torch.int64,
    ).float()
    t = torch.rand(batch_size, device=device)
    null_emb = torch.randn(1, text_tokens, 64, device=device)

    x = torch.cat([z_t, z_y, mask], dim=1)
    img_tokens, img_ids = batched_prc_img(x)
    txt_tokens, txt_ids = batched_prc_txt(null_emb.expand(batch_size, -1, -1))

    pred = model(
        x=img_tokens,
        x_ids=img_ids,
        timesteps=t,
        ctx=txt_tokens,
        ctx_ids=txt_ids,
        guidance=None,
    )
    vel = rearrange(pred, "b (h w) c -> b c h w", h=latent_size, w=latent_size)

    expected = (batch_size, 128, latent_size, latent_size)
    if tuple(vel.shape) != expected:
        raise AssertionError(f"Expected output shape {expected}, got {tuple(vel.shape)}")

    print(
        "forward smoke test passed:",
        {
            "device": str(device),
            "batch_shape": tuple(x.shape),
            "token_shape": tuple(img_tokens.shape),
            "text_shape": tuple(txt_tokens.shape),
            "output_shape": tuple(vel.shape),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny random-weight FLUX forward smoke test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--text-tokens", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_smoke_test(
        batch_size=args.batch_size,
        latent_size=args.latent_size,
        text_tokens=args.text_tokens,
        device=torch.device(args.device),
    )
