"""Precompute and cache the null text embedding used by FLUX.2 cross-attention.

For FLUX.2 [klein] 4B **base**, the text tower is Qwen3-4B. The null embedding is the
encoder output for an **empty string** ``""``, which stands in for "no user caption"
during image-only restoration.

Tensor shape: ``(1, 512, 7680)`` — batch 1, max length 512, concatenated hidden states
from three Qwen3 layers at width 2560 (``3 × 2560 = 7680``).

Typical workflow:
    1. Run ``python -m src.null_emb --config train/configs/train.yaml`` once on a GPU node.
    2. Training loads ``cfg.model.null_emb_path`` via :func:`load_or_compute_null_embedding`.

In distributed mode, rank 0 computes (if missing) and saves; all ranks synchronize on
``dist.barrier`` before reading the shared file.
"""

import argparse
from pathlib import Path

import torch

from .flux2.util import load_text_encoder
from .utils import load_config


def compute_null_embedding(
    flux_model_name: str = "flux.2-klein-base-4b",
    device: str = "cuda",
) -> torch.Tensor:
    """Run the Qwen3 text encoder on an empty string and return ``bfloat16`` embeddings.

    Loads the encoder via :func:`~src.flux2.util.load_text_encoder`, runs ``forward``
    with ``[""]``, and casts to ``torch.bfloat16`` for consistency with DiT forward.

    Args:
        flux_model_name: Key in ``FLUX2_MODEL_INFO`` selecting encoder weights.
        device:          CUDA / CPU string for HF model placement.

    Returns:
        Tensor shaped ``(1, 512, 7680)``, dtype ``bfloat16``.
    """
    enc = load_text_encoder(flux_model_name, device=device)
    enc.eval()
    with torch.no_grad():
        emb = enc([""])
    return emb.to(dtype=torch.bfloat16)


def load_or_compute_null_embedding(
    cache_path: str,
    flux_model_name: str = "flux.2-klein-base-4b",
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Load cached null embedding from disk, or compute and save it, then load.

    If ``cache_path`` already exists, loads and returns (still moved to ``device`` /
    ``bfloat16``). Otherwise:
        * **Single process:** compute, save CPU tensor, return on ``device``.
        * **Distributed:** rank 0 computes + saves; barrier; all ranks load.

    Args:
        cache_path:      Filesystem ``.pt`` path (parent dirs created as needed).
        flux_model_name: Passed to :func:`compute_null_embedding` when computing.
        device:          Target device for the returned tensor.

    Returns:
        ``(1, 512, 7680)`` ``bfloat16`` tensor on ``device``.
    """
    path = Path(cache_path)
    if isinstance(device, str):
        device = torch.device(device)

    if path.is_file():
        try:
            emb = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            emb = torch.load(path, map_location=device)
        return emb.to(device=device, dtype=torch.bfloat16)

    import torch.distributed as dist

    path.parent.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            emb = compute_null_embedding(flux_model_name, device=str(device))
            torch.save(emb.cpu(), path)
        dist.barrier()
        try:
            emb = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            emb = torch.load(path, map_location=device)
        return emb.to(device=device, dtype=torch.bfloat16)

    emb = compute_null_embedding(flux_model_name, device=str(device))
    torch.save(emb.cpu(), path)
    return emb.to(device=device, dtype=torch.bfloat16)


if __name__ == "__main__":
    # CLI: precompute once per machine / cache directory before launching training jobs.
    parser = argparse.ArgumentParser(description="Precompute null text embedding")
    parser.add_argument(
        "--config",
        type=str,
        default="train/configs/train.yaml",
        help="YAML containing model.flux_model_name and model.null_emb_path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for the text encoder forward pass",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = load_or_compute_null_embedding(
        cache_path=cfg.model.null_emb_path,
        flux_model_name=cfg.model.flux_model_name,
        device=args.device,
    )
    print(f"Null embedding shape={tuple(out.shape)} dtype={out.dtype} path={cfg.model.null_emb_path}")
