"""FLUX.2 [klein] 4B base DiT wrapper for conditional art restoration.

Architecture reference (Klein4BParams from flux2/model.py):
    in_channels=128, hidden_size=3072, num_heads=24
    depth=5 (double blocks), depth_single_blocks=20
    context_in_dim=7680, use_guidance_embed=False

img_in modification:
    Original:       nn.Linear(128,  3072, bias=False)
    Re-initialized: nn.Linear(264,  3072, bias=False)
    where 264 = 128 (z_t) + 128 (z_y) + 8 (mask channels K)

    Xavier uniform initialization. Pretrained backbone weights are loaded first,
    then ``img_in`` is replaced (pretrained ``img_in`` is incompatible with the new width).

Token layout:
    Image tokens:   rearrange (B, 264, H', W') → (B, H'*W', 264)  via batched_prc_img
    Context tokens: null_emb  (B, 512, 7680)                       via batched_prc_txt
    Position ids:   (B, seq_len, 4)  with axes (t, h, w, l)

Forward input/output:
    z_t:      (B, 128, H', W')   noisy latent at timestep t
    t:        (B,)               timesteps in [0, 1]
    z_y:      (B, 128, H', W')   corrupted image latent
    mask:     (B, K, H', W')     downsampled damage mask (latent resolution)
    null_emb: (1, 512, 7680)     precomputed null text embedding
    output:   (B, 128, H', W')   predicted velocity v_θ

Warm-up mode:
    For the first configured training iterations, backbone weights stay frozen and
    only ``img_in`` is trainable. After that, all parameters are trainable.
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from typing import Any, List

from .flux2.model import Flux2
from .flux2.sampling import batched_prc_img, batched_prc_txt
from .flux2.util import init_flow_model, load_pretrained_flow_weights


class RestorationDiT(nn.Module):
    """Conditional velocity predictor v_θ(z_t, t | z_y, M').

    Wraps Flux2 (Klein 4B base) with a re-initialized img_in that accepts
    the concatenated conditioning input [z_t, z_y, M'].
    """

    def __init__(
        self,
        cfg: Any,
        gradient_checkpointing: bool = False,
        device: str | torch.device = "cuda",
        img_in_dtype=torch.bfloat16,
        load_pretrained: bool = True,
        rank: int = 0,
    ):
        """Load pretrained FLUX.2 [klein] 4B base and re-initialize img_in.

        Steps:
            1. ``init_flow_model`` → Flux2 on meta device.
            2. Load pretrained checkpoint into ``flow_model`` (128-channel ``img_in``).
            3. Replace ``img_in`` with ``Linear(cfg.in_channels, cfg.hidden_size)`` and Xavier init.

        Args:
            cfg: YAML-backed model config (flux_model_name, in_channels=128+128+K, hidden_size=3072).
            device: Target device for weights and new ``img_in``.
            img_in_dtype: Dtype for the re-initialized ``img_in`` layer.
            load_pretrained: If False, skip weight load (random backbone; debug only).
            rank: Process rank for logging during weight download/load.
        """
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
            
        self.cfg = cfg
        self.flow_model = init_flow_model(cfg.flux_model_name)
        if load_pretrained:
            load_pretrained_flow_weights(
                self.flow_model, cfg.flux_model_name, rank=rank, device=device
            )
        self.flow_model.gradient_checkpointing = gradient_checkpointing
        self._reinit_img_in(cfg.in_channels, cfg.hidden_size, device=device, dtype=img_in_dtype)

    def load_pretrained_backbone(
        self,
        model_name: str,
        rank: int = 0,
        device: str | torch.device = "cuda",
    ) -> None:
        """Load pretrained FLUX weights into matching backbone tensors only.

        This is used as a fallback path after a failed checkpoint restore when the
        current ``img_in`` width differs from the pretrained model's 128-channel
        projection. ``img_in`` is intentionally left unchanged.
        """
        if isinstance(device, str):
            device = torch.device(device)

        temp_model = init_flow_model(model_name)
        load_pretrained_flow_weights(temp_model, model_name, rank=rank, device=device)

        target = self.flow_model.state_dict()
        source = temp_model.state_dict()
        for key, value in source.items():
            if key.startswith("img_in"):
                continue
            if key in target and target[key].shape == value.shape:
                target[key] = value
        self.flow_model.load_state_dict(target, strict=False)

    def _reinit_img_in(self, in_channels: int, hidden_size: int, device: str | torch.device = "cuda", dtype=torch.bfloat16) -> None:
        """Replace img_in with a new randomly-initialized Linear layer.

        Args:
            in_channels: 128 + 128 + K (mask channels).
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
            1. Concatenate [z_t, z_y, mask] → (B, C_in, H', W'), cast to bfloat16
            2. batched_prc_img → img_tokens (B, H'*W', C_in), img_ids (B, H'*W', 4).
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
        
            
    def set_trainability(self, warmup_only: bool) -> None:
        """Freeze backbone params during warm-up, or unfreeze everything afterward."""
        if warmup_only:
            for name, p in self.flow_model.named_parameters():
                p.requires_grad = name.startswith("img_in")
            return
        for p in self.flow_model.parameters():
            p.requires_grad_(True)

    def get_trainable_params(self) -> List[dict]:
        """Return optimizer param groups with 'params' and 'name' keys.

        Groups:
            {"params": img_in_params,   "name": "img_in"}
            {"params": backbone_params, "name": "backbone"}

        The caller sets per-group LRs based on the current warm-up/full phase.

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
