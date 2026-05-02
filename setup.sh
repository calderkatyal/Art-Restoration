#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash setup.sh [hf-model-repo] [--revision <git-ref>] [--skip-ae]

Examples:
  bash setup.sh
  bash setup.sh CalderKat/PaintingRestoration
  bash setup.sh CalderKat/PaintingRestoration --revision main

Defaults:
  hf-model-repo: CalderKat/PaintingRestoration

This script:
  1. Installs Python dependencies into the current Python environment.
  2. Downloads the checkpoint and null embedding named in inference/configs/inference.yaml.
  3. Prefetches the FLUX VAE weights into the Hugging Face cache unless --skip-ae is used.

If the model repo or FLUX assets are gated/private, run `hf auth login` first.
EOF
}

MODEL_REPO="CalderKat/PaintingRestoration"
REVISION=""
SKIP_AE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --revision)
            if [[ $# -lt 2 ]]; then
                echo "Error: --revision requires a value." >&2
                exit 1
            fi
            REVISION="$2"
            shift 2
            ;;
        --skip-ae)
            SKIP_AE=1
            shift
            ;;
        -*)
            echo "Error: unknown option $1" >&2
            usage
            exit 1
            ;;
        *)
            MODEL_REPO="$1"
            shift
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

echo "[setup] repo root: $REPO_ROOT"
echo "[setup] using python: $PYTHON_BIN"
echo "[setup] installing dependencies from requirements.txt"
"$PYTHON_BIN" -m pip install -r "$REPO_ROOT/requirements.txt"

echo "[setup] downloading model assets from $MODEL_REPO"
ART_RESTORATION_REPO_ROOT="$REPO_ROOT" \
ART_RESTORATION_MODEL_REPO="$MODEL_REPO" \
ART_RESTORATION_REVISION="$REVISION" \
ART_RESTORATION_SKIP_AE="$SKIP_AE" \
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from src.flux2.util import FLUX2_MODEL_INFO


def _copy_file_from_hub(repo_id: str, filename: str, local_target: Path, revision: str | None) -> None:
    local_target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision or None,
        )
    )
    shutil.copy2(downloaded, local_target)
    print(f"[setup] downloaded {filename} -> {local_target}")


repo_root = Path(os.environ["ART_RESTORATION_REPO_ROOT"]).resolve()
model_repo = os.environ["ART_RESTORATION_MODEL_REPO"]
revision = os.environ.get("ART_RESTORATION_REVISION") or None
skip_ae = os.environ.get("ART_RESTORATION_SKIP_AE", "0") == "1"

cfg = OmegaConf.load(repo_root / "inference/configs/inference.yaml")

checkpoint_path = repo_root / str(cfg.inference.checkpoint)
null_emb_path = repo_root / str(cfg.model.null_emb_path)

try:
    _copy_file_from_hub(
        repo_id=model_repo,
        filename=str(cfg.inference.checkpoint),
        local_target=checkpoint_path,
        revision=revision,
    )
    _copy_file_from_hub(
        repo_id=model_repo,
        filename=str(cfg.model.null_emb_path),
        local_target=null_emb_path,
        revision=revision,
    )
except Exception as exc:
    print(
        "[setup] failed to download release assets from Hugging Face.\n"
        f"[setup] repo={model_repo} revision={revision or 'main'}\n"
        f"[setup] error={type(exc).__name__}: {exc}\n"
        "[setup] if the repo is private or gated, run `hf auth login` and try again.",
        file=sys.stderr,
    )
    raise

if not skip_ae:
    flux_model_name = str(cfg.model.flux_model_name).lower()
    model_info = FLUX2_MODEL_INFO[flux_model_name]
    ae_repo_id = model_info.get("ae_repo_id", model_info["repo_id"])
    ae_filename = model_info["filename_ae"]
    try:
        ae_path = hf_hub_download(
            repo_id=ae_repo_id,
            filename=ae_filename,
            repo_type="model",
        )
        print(f"[setup] prefetched VAE weights in cache: {ae_path}")
    except Exception as exc:
        print(
            "[setup] warning: unable to prefetch FLUX VAE weights.\n"
            f"[setup] repo={ae_repo_id} file={ae_filename}\n"
            f"[setup] error={type(exc).__name__}: {exc}\n"
            "[setup] gradio_server.py will try again on first launch.",
            file=sys.stderr,
        )

print("[setup] done.")
PY

cat <<EOF

Setup complete.

Next:
  $PYTHON_BIN inference/gradio_server.py --config inference/configs/inference.yaml
EOF
