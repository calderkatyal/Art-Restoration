# Art Restoration via Conditional Latent Rectified Flow

CPSC 4520 Final Project — Yale University

Calder Katyal, Aryan Agarwal, Sohan Bendre, Sameer Bhatti

---

## Overview

This project frames artwork restoration as a **conditional latent rectified flow** problem. A pretrained FLUX.2 [klein] 4B DiT is fine-tuned to predict a velocity field that transports a corrupted image latent to a clean one. The FLUX.2 VAE is frozen throughout. Only `img_in` is re-initialized from scratch to accept the expanded input (corrupted latent + damage mask concatenated to the noisy latent).

Supported degradation types: cracks, paint loss, staining, blur, color shift.

---

## Method

Given a clean painting `x`, a stochastic corruption module produces a degraded image `y` and a multi-channel binary mask `M ∈ {0,1}^{K×H×W}` identifying damage type at each pixel.

Both `x` and `y` are encoded to latent space via the frozen FLUX.2 VAE:

```
z_1 = E(x),  z_y = E(y)
```

The mask is downsampled to latent resolution via max pooling: `M'`.

The DiT is conditioned on `[z_t, z_y, M']` (channel-wise concat) and trained to predict the rectified flow velocity:

```
z_t = (1-t)*z_0 + t*z_1,    z_0 ~ N(0,I),  t ~ U(0,1)
loss = || v_θ(z_t, t | z_y, M') - (z_1 - z_0) ||²
```

At inference, a hard data-consistency projection preserves intact regions at each ODE step:

```
z_t ← m_intact ⊙ z_y + (1 - m_intact) ⊙ z_t
```

---

## Project Structure

```
src/
├── config.py         # OmegaConf structured configs
├── corruption.py     # Stochastic damage pipeline C(x) -> (y, M)
├── dataset.py        # ArtRestorationDataset + RealDamageDataset
├── model.py          # FLUX.2 DiT wrapper with re-initialized img_in
├── vae.py            # Frozen FLUX.2 VAE encode/decode
├── null_emb.py       # Precompute and cache null text embedding
├── inference.py      # ODE sampling + data consistency
├── evaluations.py    # PSNR and stratified per-damage-type metrics
├── train.py          # Training loop (warmup + full stage)
└── flux2/            # Verbatim source from black-forest-labs/flux2

train/
├── configs/
│   └── train.yaml    # Training + inference hyperparameters
└── scripts/
    ├── warmup.sh     # SLURM: stage 1 (img_in only)
    └── full.sh       # SLURM: stage 2 (all layers)

inference/
├── configs/
│   └── inference.yaml  # Inference hyperparameters
└── scripts/
    └── run.sh          # SLURM: run inference on a test set
```

---

## Setup

```bash
pip install -r requirements.txt
```

Download datasets and place them at the paths set in `train/configs/train.yaml`:

- **Training**: [WikiArt](https://www.kaggle.com/datasets/steubk/wikiart) → `./data/wikiart/`
- **Evaluation**: [MuralDH](https://github.com/...) → `./data/muraldh/`

---

## Training

Training has two stages controlled by `train.stage` in `train/configs/train.yaml`:

| Stage | What trains | LR key |
|-------|-------------|--------|
| `warmup` | `img_in` only (backbone frozen) | `train.warmup.lr` |
| `full` | All layers | `train.full.backbone_lr` / `train.full.img_in_lr` |

See `train/scripts/warmup.sh` and `train/scripts/full.sh`.

---

## Configuration

Settings are split by task:

- **`train/configs/train.yaml`** — model, training, degradation, and inference-during-training settings
- **`inference/configs/inference.yaml`** — model and inference settings for standalone runs

---

## Evaluation

PSNR is reported for:
- Full image reconstruction
- Masked (damaged) regions only

For real degraded murals (MuralDH), results are evaluated via human pairwise forced-choice win rate on Prolific, comparing against SwinIR and GPT-4o.
