"""Precompute and cache the null text embedding for classifier-free guidance.

For FLUX.2 [klein] 4B base, the text encoder is Qwen3-4B.
The null embedding is obtained by running the encoder on an empty string [""].

Output shape: (1, 512, 7680)
    batch=1, max_length=512 tokens, 3 Qwen3-4B hidden layers × 2560 dim

Run once before training:
    python -m src.null_emb --config configs/default.yaml

The result is saved to cfg.model.null_emb_path and reloaded on subsequent runs.
"""

import torch
from pathlib import Path

from .flux2.util import load_text_encoder


def compute_null_embedding(
    flux_model_name: str = "flux.2-klein-base-4b",
    device: str = "cuda",
) -> torch.Tensor:
    """Run the Qwen3-4B text encoder on an empty string.

    Loads the text encoder via load_text_encoder(), calls it with [""],
    and returns the result cast to bfloat16.

    Args:
        flux_model_name: Key in FLUX2_MODEL_INFO to select the right encoder.
        device: Device to run on.

    Returns:
        Null embedding (1, 512, 7680) bfloat16.
    """
    ...


def load_or_compute_null_embedding(
    cache_path: str,
    flux_model_name: str = "flux.2-klein-base-4b",
    device: str = "cuda",
) -> torch.Tensor:
    """Load cached null embedding from disk, or compute and save it.

    If cache_path exists, loads and returns it.
    Otherwise calls compute_null_embedding(), saves to cache_path, and returns.

    Args:
        cache_path: Path to .pt file.
        flux_model_name: Used only when cache doesn't exist.
        device: Device.

    Returns:
        Null embedding (1, 512, 7680).
    """
    ...


if __name__ == "__main__":
    """Precompute and save the null text embedding to disk.

    Usage:
        python -m src.null_emb [--config configs/default.yaml] [--device cuda]

    Arguments:
        --config   Path to YAML config (default: configs/default.yaml).
                   Reads model.flux_model_name and model.null_emb_path.
        --device   Device to run the text encoder on (default: cuda).

    Output:
        Saves (1, 512, 7680) bfloat16 tensor to model.null_emb_path.
        No-op if the file already exists.
    """
    ...
