#!/usr/bin/env python3
"""Gradio web server: upload → paint damage masks → restore.

One-stop inference UI combining :mod:`tools.mask_painter`-style multi-channel
painting with the Euler ODE sampler from :mod:`src.inference`. The uploaded
image is resized (center-crop + bicubic) to ``cfg.inference.resolution`` before
the user sees it, so masks are drawn in the exact pixel grid the model
consumes. The original file on disk is never modified.

Usage (from repo root):
    python inference/gradio_server.py --config inference/configs/inference.yaml

``--checkpoint`` overrides ``cfg.inference.checkpoint`` if provided. Accepts
either format:

* A single-file ``torch.save``-ed state_dict of :class:`~src.model.RestorationDiT`
  (optionally wrapped under a ``"module"`` / ``"model"`` / ``"state_dict"`` key).
* A DeepSpeed checkpoint directory as written by ``engine.save_checkpoint``
  (contains a ``latest`` file pointing at a tag subdirectory, e.g.
  ``checkpoints/run_foo/step_1000/mp_rank_00_model_states.pt``). Both ZeRO-2
  (weights replicated in ``mp_rank_00_model_states.pt``) and ZeRO-3
  (weights sharded, consolidated via ``zero_to_fp32``) layouts are supported.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

import gradio as gr

from src.utils import log_message
from src.model import RestorationDiT
from src.vae import FluxVAE
from src.null_emb import load_or_compute_null_embedding
from src.inference import data_consistency_step
from src.corruption import CHANNEL_NAMES, NUM_CHANNELS, downsample_mask
from src.flux2.sampling import get_schedule


# Per-channel overlay colors (RGB). Mirrors tools/mask_painter.py CHANNELS.
CHANNEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "craquelure": (255, 80, 80),
    "rip_tear":   (230, 50, 50),
    "paint_loss": (240, 150, 40),
    "yellowing":  (240, 230, 60),
    "fading":     (180, 180, 255),
    "deposits":   (160, 140, 100),
    "scratches":  (220, 100, 220),
}
OVERLAY_ALPHA = 0.45


# ---------------------------------------------------------------------------
# Preprocessing / mask helpers
# ---------------------------------------------------------------------------

def _resize_square(img: Image.Image, resolution: int) -> Image.Image:
    """Center-crop to square, bicubic-resize to ``resolution x resolution``.

    Matches ``src.dataset._crop_resize_to_tensor`` (PIL path) so the image
    shown in the UI is the same image the model sees.
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    crop = min(w, h)
    top = max((h - crop) // 2, 0)
    left = max((w - crop) // 2, 0)
    img = img.crop((left, top, left + crop, top + crop))
    return img.resize((int(resolution), int(resolution)), Image.BICUBIC)


def _mask_from_layer(layer: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Extract a binary (H, W) uint8 mask from a Gradio ImageEditor layer.

    Returns ``None`` if ``layer`` is missing. Gradio encodes user strokes as
    RGBA with the alpha channel set where the brush painted.
    """
    if layer is None:
        return None
    arr = np.asarray(layer)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return (arr[..., 3] > 0).astype(np.uint8)
    if arr.ndim == 3:
        return (arr.sum(axis=-1) > 0).astype(np.uint8)
    return (arr > 0).astype(np.uint8)


def _layer_from_mask(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Render a binary mask as an RGBA layer painted in ``color`` (fully opaque)."""
    h, w = mask.shape
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    where = mask.astype(bool)
    layer[where, 0] = color[0]
    layer[where, 1] = color[1]
    layer[where, 2] = color[2]
    layer[where, 3] = 255
    return layer


def _composite_other_channels(
    image: np.ndarray,
    masks: Dict[int, np.ndarray],
    active_ch: int,
) -> np.ndarray:
    """Bake per-channel colored overlays for all channels EXCEPT ``active_ch``
    into the background, so the editor always shows accumulated damage context
    while the user edits one channel in isolation.
    """
    bg = image.astype(np.float32).copy()
    for ch in range(NUM_CHANNELS):
        if ch == active_ch:
            continue
        mask = masks[ch]
        if mask.max() == 0:
            continue
        color = CHANNEL_COLORS[CHANNEL_NAMES[ch]]
        alpha = mask.astype(np.float32) * OVERLAY_ALPHA
        for c_idx in range(3):
            bg[..., c_idx] = bg[..., c_idx] * (1.0 - alpha) + color[c_idx] * alpha
    return np.clip(bg, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Checkpoint loading (single-file or DeepSpeed directory)
# ---------------------------------------------------------------------------

def _unwrap_state_dict(blob: Any) -> Dict[str, torch.Tensor]:
    """Peel common wrapper keys (``"module"``, ``"model"``, ``"state_dict"``)
    off a loaded object and strip a leading ``module.`` prefix from all keys.

    Leaves tensor values untouched. Returns a plain ``{str: Tensor}`` dict.
    """
    sd = blob
    if isinstance(sd, dict):
        for key in ("module", "model", "state_dict"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    if not isinstance(sd, dict):
        raise TypeError(f"Expected state_dict mapping, got {type(sd).__name__}")
    if any(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
        sd = {
            (k[len("module."):] if isinstance(k, str) and k.startswith("module.") else k): v
            for k, v in sd.items()
        }
    return sd


def _load_checkpoint_state_dict(
    ckpt_path: str,
    map_location: torch.device,
) -> Dict[str, torch.Tensor]:
    """Load a :class:`RestorationDiT` state_dict from either layout.

    * If ``ckpt_path`` is a directory, treat it as a DeepSpeed checkpoint root.
      Resolve the active tag via the ``latest`` file (or assume ``ckpt_path``
      itself is already the tag dir). Prefer the ZeRO-2 fast path
      (``mp_rank_00_model_states.pt["module"]``); fall back to
      ``zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint`` for ZeRO-3.
    * If ``ckpt_path`` is a file, ``torch.load`` it and unwrap common wrappers.
    """
    p = Path(ckpt_path)
    if p.is_dir():
        latest_file = p / "latest"
        if latest_file.exists():
            tag = latest_file.read_text().strip()
            tag_dir = p / tag
            root_dir = p
        else:
            tag_dir = p
            root_dir = p.parent
            tag = p.name
        shard = tag_dir / "mp_rank_00_model_states.pt"
        if shard.exists():
            log_message(f"[gradio] loading DeepSpeed shard {shard}")
            blob = torch.load(str(shard), map_location=map_location, weights_only=False)
            return _unwrap_state_dict(blob)
        # ZeRO-3 fallback: consolidate sharded weights on the fly.
        log_message(
            f"[gradio] no mp_rank_00 shard; consolidating ZeRO shards from {root_dir} (tag={tag})"
        )
        from deepspeed.utils.zero_to_fp32 import (  # local import: heavy dep
            get_fp32_state_dict_from_zero_checkpoint,
        )
        sd = get_fp32_state_dict_from_zero_checkpoint(str(root_dir), tag=tag)
        return _unwrap_state_dict(sd)

    log_message(f"[gradio] loading checkpoint file {p}")
    blob = torch.load(str(p), map_location=map_location, weights_only=False)
    return _unwrap_state_dict(blob)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class GradioApp:
    """Single-session inference app: loads weights once at startup."""

    def __init__(self, cfg, checkpoint: Optional[str] = None, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.resolution = int(cfg.inference.resolution)
        self.num_steps_default = int(cfg.inference.get("num_steps", 50))

        ckpt_path = checkpoint or cfg.inference.checkpoint
        if not ckpt_path or not Path(str(ckpt_path)).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path!r}. "
                "Set cfg.inference.checkpoint (or pass --checkpoint) to either a "
                "single-file state_dict or a DeepSpeed checkpoint directory."
            )

        log_message(f"[gradio] loading RestorationDiT from {ckpt_path}")
        self.model = RestorationDiT(
            cfg=cfg.model,
            gradient_checkpointing=False,
            device=self.device,
            img_in_dtype=torch.bfloat16,
            load_pretrained=False,
            rank=0,
        )
        state = _load_checkpoint_state_dict(str(ckpt_path), map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            log_message(f"[gradio] WARN {len(missing)} missing checkpoint keys (showing first 4): {missing[:4]}")
        if unexpected:
            log_message(f"[gradio] WARN {len(unexpected)} unexpected keys (showing first 4): {unexpected[:4]}")
        self.model.eval()

        log_message("[gradio] loading VAE")
        self.vae = FluxVAE(
            flux_model_name=cfg.model.flux_model_name,
            rank=0,
            device=str(self.device),
        ).to(self.device).eval()

        log_message("[gradio] loading null embedding")
        self.null_emb = load_or_compute_null_embedding(
            cache_path=cfg.model.null_emb_path,
            flux_model_name=cfg.model.flux_model_name,
            device=self.device,
        )
        log_message("[gradio] ready.")

    # -- callbacks -----------------------------------------------------------

    def on_upload(self, pil_image: Optional[Image.Image]):
        """Resize the uploaded image and initialize the mask state."""
        if pil_image is None:
            return gr.update(value=None), None, None
        resized = _resize_square(pil_image, self.resolution)
        bg = np.array(resized)
        masks = {
            ch: np.zeros((self.resolution, self.resolution), dtype=np.uint8)
            for ch in range(NUM_CHANNELS)
        }
        state = {"image": bg, "masks": masks, "active_ch": 0}
        editor_value = {
            "background": _composite_other_channels(bg, masks, 0),
            "layers": [np.zeros((self.resolution, self.resolution, 4), dtype=np.uint8)],
            "composite": bg,
        }
        return gr.update(value=editor_value), state, None

    def on_channel_change(self, ch_name: str, editor: Any, state: Optional[dict]):
        """Swap the painting canvas to the selected channel.

        Persists the current layer into ``state["masks"][old_ch]``, then
        rebuilds the editor with a background containing every OTHER channel's
        colored overlay plus a fresh layer holding the target channel's
        existing strokes.
        """
        if state is None:
            return gr.update(), state
        new_ch = CHANNEL_NAMES.index(ch_name)
        old_ch = int(state["active_ch"])
        if editor is not None and editor.get("layers"):
            painted = _mask_from_layer(editor["layers"][0])
            if painted is not None:
                state["masks"][old_ch] = painted
        state["active_ch"] = new_ch

        bg_other = _composite_other_channels(state["image"], state["masks"], new_ch)
        new_layer = _layer_from_mask(
            state["masks"][new_ch],
            CHANNEL_COLORS[ch_name],
        )
        editor_value = {
            "background": bg_other,
            "layers": [new_layer],
            "composite": bg_other,
        }
        return gr.update(value=editor_value), state

    def on_clear_channel(self, editor: Any, state: Optional[dict]):
        """Clear strokes on the current channel."""
        if state is None:
            return gr.update(), state
        ch = int(state["active_ch"])
        state["masks"][ch] = np.zeros(
            (self.resolution, self.resolution), dtype=np.uint8
        )
        bg_other = _composite_other_channels(state["image"], state["masks"], ch)
        editor_value = {
            "background": bg_other,
            "layers": [np.zeros((self.resolution, self.resolution, 4), dtype=np.uint8)],
            "composite": bg_other,
        }
        return gr.update(value=editor_value), state

    def on_generate(
        self,
        editor: Any,
        state: Optional[dict],
        num_steps: float,
        progress: gr.Progress = gr.Progress(),
    ):
        """Assemble the (1, 8, R, R) mask tensor and run the Euler sampler."""
        if state is None:
            raise gr.Error("Upload an image first.")
        # Save the current channel's strokes before consuming the state.
        if editor is not None and editor.get("layers"):
            painted = _mask_from_layer(editor["layers"][0])
            if painted is not None:
                state["masks"][int(state["active_ch"])] = painted

        # Per-channel masks (7, R, R) + union channel (1, R, R) -> (8, R, R).
        per_ch = np.stack(
            [state["masks"][c] for c in range(NUM_CHANNELS)], axis=0
        ).astype(np.float32)
        union = per_ch.max(axis=0, keepdims=True)
        mask_np = np.concatenate([per_ch, union], axis=0)  # (8, R, R)
        if mask_np[-1].max() < 0.5:
            raise gr.Error(
                "No damage mask painted. Paint at least one stroke on any "
                "channel before generating."
            )
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        img_np = state["image"].astype(np.float32) / 255.0  # (R, R, 3)
        img_t = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        restored = self._run_inference(img_t, mask_t, int(num_steps), progress)
        out = (
            restored[0]
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .cpu()
            .float()
            .numpy()
            * 255.0
        ).round().astype(np.uint8)
        return out, state

    # -- inference -----------------------------------------------------------

    @torch.no_grad()
    def _run_inference(
        self,
        corrupted_image: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int,
        progress: gr.Progress,
    ) -> torch.Tensor:
        """Inlined equivalent of :func:`src.inference.sample` with progress."""
        device = self.device
        progress(0.02, desc="Encoding with VAE...")
        z_y = self.vae.encode(corrupted_image.to(device))
        m_lat = downsample_mask(mask.to(device), factor=self.vae.spatial_compression)
        b, _, h, w = z_y.shape
        z_t = torch.randn_like(z_y)
        seq_len = h * w
        timesteps = get_schedule(num_steps, seq_len)

        dtype = torch.bfloat16 if z_t.device.type == "cuda" else torch.float32
        z_t = z_t.to(dtype=dtype)
        z_y = z_y.to(dtype=dtype)
        m_lat = m_lat.to(dtype=dtype)
        null_b = self.null_emb.to(device=device, dtype=dtype)

        z_t = data_consistency_step(z_t, z_y, m_lat)
        total = max(1, len(timesteps) - 1)
        for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            progress((i + 1) / (total + 1), desc=f"Denoising {i + 1}/{total}")
            t_vec = torch.full((b,), float(t_curr), device=device, dtype=dtype)
            vel = self.model(z_t, t_vec, z_y, m_lat, null_b)
            z_t = z_t + (float(t_next) - float(t_curr)) * vel
            z_t = data_consistency_step(z_t, z_y, m_lat)

        progress(1.0, desc="Decoding with VAE...")
        restored = self.vae.decode(z_t.float())
        return restored.clamp(0.0, 1.0)

    # -- UI ------------------------------------------------------------------

    def build_ui(self) -> gr.Blocks:
        with gr.Blocks(
            title="Art Restoration",
            theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
            css="""
            .gradio-container { max-width: 1600px !important; }
            #mask-editor canvas { background: #1a1a1a; }
            """,
        ) as demo:
            state = gr.State()

            gr.Markdown(
                "# Art Restoration\n"
                "Upload a damaged artwork, paint per-type damage masks, then "
                "generate a restored image. The upload is center-cropped and "
                f"resized to **{self.resolution}×{self.resolution}** so masks "
                "are pixel-aligned with the model's input grid."
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    upload = gr.Image(
                        label="1. Upload image",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=220,
                    )
                    channel = gr.Radio(
                        choices=list(CHANNEL_NAMES),
                        value=CHANNEL_NAMES[0],
                        label="2. Damage type",
                        info="Paint strokes apply to the selected channel. "
                             "Switching channels preserves each channel's strokes.",
                    )
                    clear_btn = gr.Button("Clear current channel", size="sm")
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=self.num_steps_default,
                        step=1,
                        label="3. Inference steps",
                    )
                    generate_btn = gr.Button(
                        "Restore",
                        variant="primary",
                        size="lg",
                    )
                    gr.Markdown(
                        "_Tip: unpainted channels are zero-filled; the union "
                        "channel is computed automatically as the pixel-wise "
                        "max over all per-type channels._"
                    )

                with gr.Column(scale=2, min_width=400):
                    editor = gr.ImageEditor(
                        label="Paint masks",
                        type="numpy",
                        image_mode="RGB",
                        sources=(),
                        interactive=True,
                        brush=gr.Brush(
                            default_size=15,
                            colors=["#ffffff"],
                            default_color="#ffffff",
                            color_mode="fixed",
                        ),
                        eraser=gr.Eraser(default_size=15),
                        canvas_size=(self.resolution, self.resolution),
                        elem_id="mask-editor",
                        height=self.resolution + 80,
                    )

                with gr.Column(scale=2, min_width=400):
                    output_img = gr.Image(
                        label="Restored",
                        type="numpy",
                        interactive=False,
                        height=self.resolution + 80,
                    )

            upload.change(
                fn=self.on_upload,
                inputs=[upload],
                outputs=[editor, state, output_img],
            )
            channel.change(
                fn=self.on_channel_change,
                inputs=[channel, editor, state],
                outputs=[editor, state],
            )
            clear_btn.click(
                fn=self.on_clear_channel,
                inputs=[editor, state],
                outputs=[editor, state],
            )
            generate_btn.click(
                fn=self.on_generate,
                inputs=[editor, state, num_steps],
                outputs=[output_img, state],
            )

        demo.queue(max_size=8)
        return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio server for art restoration inference.")
    parser.add_argument(
        "--config",
        type=str,
        default="inference/configs/inference.yaml",
        help="Path to inference YAML (default: inference/configs/inference.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override cfg.inference.checkpoint with this state_dict path.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve on (default: 7860).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public *.gradio.live tunnel.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (default: cfg.inference.device or 'cuda').",
    )
    args = parser.parse_args()

    # Inference YAML has no ``corruption`` key, so don't route it through
    # ``src.utils.load_config`` (which requires ``corruption.config_path``).
    # The plain OmegaConf load is sufficient here.
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    device = args.device or cfg.inference.get("device", "cuda")
    app = GradioApp(cfg=cfg, checkpoint=args.checkpoint, device=device)
    demo = app.build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=bool(args.share),
        show_error=True,
    )


if __name__ == "__main__":
    main()
