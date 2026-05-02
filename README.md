# Art Restoration via Conditional Latent Rectified Flow

CPSC 4520 Final Project — Yale University

Calder Katyal, Aryan Agarwal, Sohan Bendre, Sameer Bhatti

---

## Overview

This project frames artwork restoration as a **conditional latent rectified flow** problem. A pretrained FLUX.2 [klein] 4B DiT is fine-tuned to predict a velocity field that transports a corrupted image latent to a clean one. The FLUX.2 VAE is frozen throughout. Only `img_in` is re-initialized from scratch to accept the expanded input (corrupted latent + damage mask concatenated to the noisy latent).

Supported degradation types (8 channels): craquelure, rip/tear, paint loss, yellowing, fading, surface deposits, scratches.

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
├── utils.py           # Shared config, logging, and inference helpers
├── corruption/        # Stochastic damage pipeline C(x) -> (y, M)
│   ├── configs/
│   │   └── default.yaml  # Per-channel corruption hyperparameters
│   ├── effects.py     # Individual corruption effect implementations
│   ├── presets.py     # CHANNEL_NAMES + local/global mask generators
│   ├── module.py      # CorruptionModule: main entry point
│   └── color.py       # Color space conversions (sRGB ↔ CIELAB)
├── dataset.py         # ArtRestorationDataset + RealDamageDataset
├── model.py           # FLUX.2 DiT wrapper with re-initialized img_in
├── vae.py             # Frozen FLUX.2 VAE encode/decode
├── null_emb.py        # Precompute and cache null text embedding
├── evaluations.py     # PSNR and stratified per-damage-type metrics
├── train.py           # Training loop with iteration-based warm-up
└── flux2/             # Verbatim source from black-forest-labs/flux2

tests/
└── test_corruption_visual.py  # Visual test grid for all presets

train/
├── configs/
│   └── train.yaml     # Training hyperparameters
└── scripts/
    ├── warmup.sh      # SLURM: start training from scratch
    └── full.sh        # SLURM: resume from a saved checkpoint

inference/
├── configs/
│   └── inference.yaml # Gradio inference settings
└── scripts/
    └── gradio.sh      # SLURM: launch the Gradio server
```

---

## Setup

```bash
pip install -r requirements.txt
```

For Gradio inference, the quickest path is:

```bash
bash setup.sh
python inference/gradio_server.py --config inference/configs/inference.yaml
```

If the setup script tells you Hugging Face authentication is required for your model
repo or the FLUX VAE assets, run `hf auth login` and then rerun the setup script.

Download datasets and place them at the paths set in `train/configs/train.yaml`:

- **Training**: [WikiArt](https://www.kaggle.com/datasets/steubk/wikiart) → `./data/wikiart/`
- **Evaluation**: [MuralDH](https://github.com/...) → `./data/muraldh/`

---

## Training

Training starts with `img_in`-only warm-up for `train.warmup_iterations` optimizer steps,
then automatically unfreezes the backbone and switches to the full-training learning rates.
`train.warmup_iterations` is evaluated when training starts or resumes from checkpoint.
If you change it mid-run, the live process will not notice; stop and resume from a checkpoint
to apply the new value.

| Phase | What trains | Config key |
|-------|-------------|------------|
| Warm-up | `img_in` only (backbone frozen) | `train.warmup_iterations`, `train.warmup.lr` |
| Full training | All layers | `train.full.backbone_lr` / `train.full.img_in_lr` |

See `train/scripts/warmup.sh` and `train/scripts/full.sh`.

---

## Configuration

Settings are split by task:

- **`train/configs/train.yaml`** — model, training, degradation, and inference-during-training settings
- **`inference/configs/inference.yaml`** — model and Gradio inference settings

---

## Inference

### Launch the Gradio restoration UI

The setup script installs the Python dependencies into your active environment, downloads
the checkpoint and null embedding named in `inference/configs/inference.yaml`, and
prefetches the FLUX VAE cache when possible.

```bash
bash setup.sh
```

Then launch the server from the repo root:

```bash
python inference/gradio_server.py --config inference/configs/inference.yaml
```

`setup.sh` defaults to downloading from `CalderKat/PaintingRestoration`. To override that,
pass a different Hugging Face repo:

```bash
bash setup.sh <your-hf-model-repo>
```

---

## Evaluation

PSNR is reported for:
- Full image reconstruction
- Masked (damaged) regions only

For real degraded murals (MuralDH), results are evaluated via human pairwise forced-choice win rate on Prolific, comparing against SwinIR and GPT-4o.
