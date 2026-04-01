"""Precompute and cache the null text embedding for CFG.

Run once before training. The null embedding is the text encoder's output
for an empty string, used as the unconditional conditioning signal.
"""

import torch
from pathlib import Path


def compute_null_embedding(model_repo: str, device: str = "cuda") -> torch.Tensor:
    """Run the FLUX.2 text encoder on an empty string to get the null embedding.

    Args:
        model_repo: HuggingFace repo ID for FLUX.2 (to load the text encoder).
        device: Device to run the text encoder on.

    Returns:
        Null text embedding tensor, shape (1, seq_len, hidden_dim).
    """
    ...


def load_or_compute_null_embedding(
    cache_path: str, model_repo: str, device: str = "cuda"
) -> torch.Tensor:
    """Load cached null embedding from disk, or compute and save it.

    Args:
        cache_path: Path to the cached .pt file.
        model_repo: HuggingFace repo ID (used only if cache doesn't exist).
        device: Device to load/compute on.

    Returns:
        Null text embedding tensor, shape (1, seq_len, hidden_dim).
    """
    ...


if __name__ == "__main__":
    """CLI entry point: precompute and save the null text embedding."""
    ...
