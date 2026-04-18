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
from .flux2.util import init_flow_model


class RestorationDiT(nn.Module):
    """Conditional velocity predictor v_θ(z_t, t | z_y, M').

    Wraps Flux2 (Klein 4B base) with a re-initialized img_in that accepts
    the concatenated conditioning input [z_t, z_y, M'].
    """

    def __init__(self, cfg: ModelConfig, device: str | torch.device = "cuda", img_in_dtype=torch.bfloat16):
        """Load pretrained FLUX.2 [klein] 4B base and re-initialize img_in.

        Steps:
            1. load_flow_model(cfg.flux_model_name) → Flux2 with pretrained weights.
            2. Replace self.flow_model.img_in with Linear(cfg.in_channels, cfg.hidden_size).
            3. Xavier uniform init on new img_in.

        Args:
            cfg: ModelConfig (flux_model_name, in_channels=261, hidden_size=3072).
        """
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
            
        self.cfg = cfg
        self.flow_model = init_flow_model(cfg.flux_model_name)
        self._reinit_img_in(cfg.in_channels, cfg.hidden_size, device=device, dtype=img_in_dtype)

    def _reinit_img_in(self, in_channels: int, hidden_size: int, device: str | torch.device = "cuda", dtype=torch.bfloat16) -> None:
        """Replace img_in with a new randomly-initialized Linear layer.

        Args:
            in_channels: 261 = 128 (z_t) + 128 (z_y) + K (mask).
            hidden_size: 3072 (Klein 4B).
        """
        new_in = nn.Linear(in_channels, hidden_size, bias=False, device=device, dtype=dtype)
        nn.init.xavier_uniform_(new_in.weight)
        self.flow_model.img_in = new_in

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
            1. Concatenate [z_t, z_y, mask] → (B, 261, H', W'), cast to bfloat16
            2. batched_prc_img → img_tokens (B, H'*W', 261), img_ids (B, H'*W', 4).
            3. Expand null_emb to batch B; batched_prc_txt → txt_tokens, txt_ids.
            4. self.flow_model(x=img_tokens, x_ids=img_ids, timesteps=t,
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
        if z_t.shape != z_y.shape:
            raise ValueError("z_t and z_y must match shape")
        b, c_lat, h, w = z_t.shape
        if mask.shape[0] != b or mask.shape[-2:] != (h, w):
            raise ValueError(f"mask spatial shape {tuple(mask.shape)} vs z {h,w}")
        # FLUX transformer is kept in bfloat16 on CUDA for memory bandwidth.
        dtype = torch.bfloat16 if z_t.device.type == "cuda" else torch.float32
        x = torch.cat([z_t, z_y, mask], dim=1).to(dtype=dtype)
        img_tokens, img_ids = batched_prc_img(x)
        
        if null_emb.shape[0] == 1 and b > 1:
            txt = null_emb.expand(b, -1, -1).to(dtype=dtype)
        else:
            txt = null_emb.to(dtype=dtype)
        txt_tokens, txt_ids = batched_prc_txt(txt)
        
        pred = self.flow_model(
            x=img_tokens,
            x_ids=img_ids,
            timesteps=t.to(dtype=dtype),
            ctx=txt_tokens, 
            ctx_ids=txt_ids,
            guidance=None
        )
        
        vel = rearrange(pred, 'b (h w) c -> b c h w', h=h, w=w)
        return vel.to(dtype=z_t.dtype)
        
            
    def set_stage(self, stage: str) -> None:
        """Freeze or unfreeze parameters for the given training stage.

        Args:
            stage: "warmup" → freeze all except img_in.
                   "full"   → unfreeze all parameters.
        """
        if stage == "warmup": 
            for name, p in self.flow_model.named_parameters(): 
                p.requires_grad = name.startswith("img_in")
        elif stage == "full":
            for p in self.flow_model.parameters(): 
                p.requires_grad_(True)
        else: 
            raise ValueError(f"Unknown stage {stage!r}; expected 'warmup' or 'full'")

    def get_trainable_params(self) -> List[dict]:
        """Return optimizer param groups with 'params' and 'name' keys.

        Groups:
            {"params": img_in_params,   "name": "img_in"}
            {"params": backbone_params, "name": "backbone"}

        The caller sets per-group LRs based on the training stage.

        Returns:
            List of two param group dicts.
        """
        img_ids = {id(p) for p in self.flow_model.img_in.parameters()}
        img_in_params = list(self.flow_model.img_in.parameters())
        backbone_params = [p for p in self.flow_model.parameters() if id(p) not in img_ids]
        return [
            {"params": img_in_params, "name": "img_in"},
            {"params": backbone_params, "name": "backbone"},
        ]
