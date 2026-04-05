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
art_restoration/
├── config.py         # OmegaConf structured configs
├── corruption.py     # Stochastic damage pipeline C(x) -> (y, M)
├── dataset.py        # ArtRestorationDataset + RealDamageDataset
├── model.py          # FLUX.2 DiT wrapper with re-initialized img_in
├── vae.py            # Frozen FLUX.2 VAE encode/decode
├── null_emb.py       # Precompute and cache null text embedding
├── inference.py      # ODE sampling + data consistency + PSNR
└── train.py          # Training loop (warmup + full stage)

configs/
└── default.yaml      # All hyperparameters and stage settings
```

---

## Setup

```bash
pip install -r requirements.txt
```

Download datasets and place them at the paths set in `configs/default.yaml`:

- **Training**: [WikiArt](https://www.kaggle.com/datasets/steubk/wikiart) → `./data/wikiart/`
- **Evaluation**: [MuralDH](https://github.com/...) → `./data/muraldh/`

Precompute the null text embedding once before training:

```bash
python -m art_restoration.null_emb
```

---

## Training

Training has two stages controlled by `train.stage` in the config:

| Stage | What trains | LR key |
|-------|-------------|--------|
| `warmup` | `img_in` only (backbone frozen) | `train.warmup.lr` |
| `full` | All layers | `train.full.backbone_lr` / `train.full.img_in_lr` |

**Stage 1 — warmup (img_in only):**

```bash
python -m art_restoration.train --config configs/default.yaml train.stage=warmup
```

**Stage 2 — full fine-tune:**

```bash
python -m art_restoration.train --config configs/default.yaml \
    train.stage=full \
    train.resume_from=./checkpoints/warmup_final.pt
```

Any config field can be overridden via dot-notation on the CLI:

```bash
python -m art_restoration.train --config configs/default.yaml \
    train.batch_size=8 \
    train.optimizer.lr=5e-5 \
    degradation.max_simultaneous=1
```

---

## Configuration

All settings live in `configs/default.yaml`. Key fields:

```yaml
train:
  stage: "warmup"          # "warmup" or "full"
  batch_size: 4
  num_epochs: 50
  curriculum:
    enabled: true
    warmup_epochs: 5       # single degradation only for N epochs

model:
  flux_repo: "black-forest-labs/FLUX.1-schnell"
  mask_channels: 5
  guidance_scale: 1.0

degradation:
  damage_types: [crack, paint_loss, stain, blur, color_shift]
  severity_range: [0.2, 0.8]
  max_simultaneous: 3
```

---

## Inference

```python
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from src.config import load_config
from src.vae import FluxVAE
from src.model import RestorationDiT
from src.null_emb import load_or_compute_null_embedding
from src.inference import sample

cfg    = load_config("configs/default.yaml")
device = "cuda"

# Load models
vae      = FluxVAE(cfg.model.flux_model_name, device=device)
model    = RestorationDiT(cfg.model).to(device)
model.load_state_dict(torch.load("checkpoints/final.pt", map_location=device)["model_state_dict"])
model.eval()
null_emb = load_or_compute_null_embedding(cfg.model.null_emb_path, cfg.model.flux_model_name, device)

# Prepare inputs
# H and W must be divisible by 16 (VAE spatial compression)
corrupted = TF.to_tensor(Image.open("damaged.png").convert("RGB")).unsqueeze(0).to(device)

# mask: (B, K, H, W) — one channel per damage type, 1 = damaged
# K = cfg.model.mask_channels = 5  →  [crack, paint_loss, stain, blur, color_shift]
K = cfg.model.mask_channels
mask = torch.zeros(1, K, corrupted.shape[2], corrupted.shape[3], device=device)
mask[:, 0] = 1   # mark "crack" channel as damaged over the full image
# (set multiple channels if multiple damage types are present)

# Restore
restored = sample(model, vae, corrupted, mask, null_emb,
                  num_steps=cfg.model.num_steps, device=device)
# restored: (B, 3, H, W) float32 in [0, 1]
TF.to_pil_image(restored[0].cpu()).save("restored.png")
```

Mask channel order matches `degradation.damage_types` in `configs/default.yaml`:

| Channel | Damage type  |
|---------|--------------|
| 0       | crack        |
| 1       | paint_loss   |
| 2       | stain        |
| 3       | blur         |
| 4       | color_shift  |

---

## Evaluation

PSNR is reported for:
- Full image reconstruction
- Masked (damaged) regions only

For real degraded murals (MuralDH), results are evaluated via human pairwise forced-choice win rate on Prolific, comparing against SwinIR and GPT-4o.
