"""FLUX.2 [klein] 4B base DiT wrapper for conditional art restoration.

Architecture reference (Klein4BParams from flux2/model.py):
    in_channels=128, hidden_size=3072, num_heads=24
    depth=5 (double blocks), depth_single_blocks=20
    context_in_dim=7680, use_guidance_embed=False

img_in modification:
    Original:       nn.Linear(128,  3072, bias=False)
    Re-initialized: nn.Linear(261,  3072, bias=False)
    where 261 = 128 (z_t) + 128 (z_y) + 5 (mask channels K)

    Xavier uniform initialization. All other weights loaded from pretrained.

Token layout:
    Image tokens:   rearrange (B, 261, H', W') → (B, H'*W', 261)  via batched_prc_img
    Context tokens: null_emb  (B, 512, 7680)                       via batched_prc_txt
    Position ids:   (B, seq_len, 4)  with axes (t, h, w, l)

Forward input/output:
    z_t:      (B, 128, H', W')   noisy latent at timestep t
    t:        (B,)               timesteps in [0, 1]
    z_y:      (B, 128, H', W')   corrupted image latent
    mask:     (B, K, H', W')     downsampled damage mask (latent resolution)
    null_emb: (1, 512, 7680)     precomputed null text embedding
    output:   (B, 128, H', W')   predicted velocity v_θ

Training stages (controlled via set_stage):
    "warmup": backbone frozen, only img_in.requires_grad = True
    "full":   all parameters trainable
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from typing import List

from .config import ModelConfig
from .flux2.model import Flux2, Klein4BParams
from .flux2.sampling import batched_prc_img, batched_prc_txt
from .flux2.util import load_flow_model


class RestorationDiT(nn.Module):
    """Conditional velocity predictor v_θ(z_t, t | z_y, M').

    Wraps Flux2 (Klein 4B base) with a re-initialized img_in that accepts
    the concatenated conditioning input [z_t, z_y, M'].
    """

    def __init__(self, cfg: ModelConfig,  device: str = "cuda"):
        """Load pretrained FLUX.2 [klein] 4B base and re-initialize img_in.

        Steps:
            1. load_flow_model(cfg.flux_model_name) → Flux2 with pretrained weights.
            2. Replace self.transformer.img_in with Linear(cfg.in_channels, cfg.hidden_size).
            3. Xavier uniform init on new img_in.

        Args:
            cfg: ModelConfig (flux_model_name, in_channels=261, hidden_size=3072).
        """
        super().__init__()
        self.flow_model = load_flow_model(cfg.flux_model_name, device=device)
        self.flow_model.img_in = nn.Linear(cfg.in_channels, cfg.hidden_size, bias=True).to(device)
        nn.init.xavier_normal_(self.flow_model.img_in.weight)

    def _reinit_img_in(self, in_channels: int, hidden_size: int) -> None:
        """Replace img_in with a new randomly-initialized Linear layer.

        Args:
            in_channels: 261 = 128 (z_t) + 128 (z_y) + K (mask).
            hidden_size: 3072 (Klein 4B).
        """
        ...

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        z_y: Tensor,
        mask: Tensor,
        null_emb: Tensor,
    ) -> Tensor:
        """Predict velocity v_θ(z_t, t | z_y, M').

        Steps:
            1. Concatenate [z_t, z_y, mask] → (B, 261, H', W'), cast to bfloat16.
            2. batched_prc_img → img_tokens (B, H'*W', 261), img_ids (B, H'*W', 4).
            3. Expand null_emb to batch B; batched_prc_txt → txt_tokens, txt_ids.
            4. self.transformer(x=img_tokens, x_ids=img_ids, timesteps=t,
                                ctx=txt_tokens, ctx_ids=txt_ids, guidance=None).
               guidance=None because Klein 4B base has use_guidance_embed=False.
            5. Rearrange output (B, H'*W', 128) → (B, 128, H', W').

        Args:
            z_t:      (B, 128, H', W').
            t:        (B,) in [0, 1].
            z_y:      (B, 128, H', W').
            mask:     (B, K, H', W') binary float32.
            null_emb: (1, 512, 7680) or (B, 512, 7680).

        Returns:
            Predicted velocity (B, 128, H', W'), same dtype as z_t.
        """
        ...

    def set_stage(self, stage: str) -> None:
        """Freeze or unfreeze parameters for the given training stage.

        Args:
            stage: "warmup" → freeze all except img_in.
                   "full"   → unfreeze all parameters.
        """
        ...

    def get_trainable_params(self) -> List[dict]:
        """Return optimizer param groups with 'params' and 'name' keys.

        Groups:
            {"params": img_in_params,   "name": "img_in"}
            {"params": backbone_params, "name": "backbone"}

        The caller sets per-group LRs based on the training stage.

        Returns:
            List of two param group dicts.
        """
        ...
