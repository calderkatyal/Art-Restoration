#!/usr/bin/env python3
"""
Mask Painter — standalone tkinter GUI for painting binary damage masks.

Usage:
    python tools/mask_painter.py path/to/image.jpg [--output_dir masks_output/]

Paint binary masks for 8 damage channels used by the corruption / restoration
pipeline.  Each channel is rendered as a colored overlay on the source image.
Saves individual PNGs, a combined ``masks.pt`` tensor, and an overlay
visualization.
"""

from __future__ import annotations

import argparse
import copy
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageTk

try:
    import torch
except ImportError:
    torch = None  # torch is optional for saving .pt; warn at save time

# ---------------------------------------------------------------------------
# Channel definitions — must match the corruption module
# ---------------------------------------------------------------------------

CHANNELS: List[Tuple[str, Tuple[int, int, int]]] = [
    ("cracks",      (255,  80,  80)),   # red
    ("paint_loss",  (255, 180,  60)),   # orange
    ("yellowing",   (240, 230,  60)),   # yellow
    ("stains",      (120,  70,  40)),   # brown
    ("fading",      (180, 180, 255)),   # light blue
    ("bloom",       ( 80, 220, 220)),   # cyan
    ("deposits",    (160, 140, 100)),   # olive
    ("scratches",   (220, 100, 220)),   # magenta
]

NUM_CHANNELS = len(CHANNELS)
OVERLAY_ALPHA = 0.45          # opacity of mask overlays on the canvas
MAX_CANVAS_DIM = 700          # max width or height for the displayed canvas
HISTORY_LIMIT = 50            # max undo snapshots


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class MaskPainterApp:
    """Main application window."""

    def __init__(self, master: tk.Tk, image_path: str, output_dir: Optional[str] = None):
        self.master = master
        self.master.title("Mask Painter")
        self.master.configure(bg="#1a1a1a")

        # ---- Load image ----
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            messagebox.showerror("Error", f"Image not found: {self.image_path}")
            self.master.destroy()
            return

        self.orig_image = Image.open(self.image_path).convert("RGB")
        self.img_w, self.img_h = self.orig_image.size

        # Output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.image_path.parent / self.image_path.stem

        # ---- Compute display scale ----
        scale = min(MAX_CANVAS_DIM / self.img_w, MAX_CANVAS_DIM / self.img_h, 1.0)
        self.disp_w = int(self.img_w * scale)
        self.disp_h = int(self.img_h * scale)
        self.scale = scale  # display / original

        self.display_image = self.orig_image.resize(
            (self.disp_w, self.disp_h), Image.LANCZOS
        )

        # ---- Masks at original resolution (binary: 0 or 1, stored as uint8) ----
        self.masks: List[np.ndarray] = [
            np.zeros((self.img_h, self.img_w), dtype=np.uint8) for _ in range(NUM_CHANNELS)
        ]

        # ---- State ----
        self.active_channel: int = 0
        self.brush_size: int = 40         # in *original* image pixels
        self.show_overlay: bool = True
        self.drawing: bool = False
        self.last_xy: Optional[Tuple[int, int]] = None

        # ---- Undo / redo ----
        self.undo_stack: List[Tuple[int, np.ndarray]] = []   # (channel_idx, mask_copy)
        self.redo_stack: List[Tuple[int, np.ndarray]] = []

        # ---- Build UI ----
        self._build_ui()
        self._render_canvas()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Use a horizontal layout: sidebar on left, canvas on right
        sidebar = tk.Frame(self.master, bg="#222222", width=250, padx=10, pady=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        canvas_frame = tk.Frame(self.master, bg="#1a1a1a", padx=10, pady=10)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- Sidebar: title ----
        tk.Label(
            sidebar, text="Damage Channels", bg="#222222", fg="#cccccc",
            font=("Helvetica", 13, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))

        # ---- Channel buttons ----
        self.channel_buttons: List[tk.Button] = []
        for idx, (name, color) in enumerate(CHANNELS):
            hex_color = "#%02x%02x%02x" % color
            display_name = name.replace("_", " ").title()
            btn = tk.Button(
                sidebar, text=f"  {display_name}", anchor=tk.W,
                bg="#333333", fg="#dddddd", activebackground="#444444",
                activeforeground="#ffffff", relief=tk.FLAT, padx=6, pady=4,
                font=("Helvetica", 11), cursor="hand2",
                command=lambda i=idx: self._select_channel(i),
            )
            btn.pack(fill=tk.X, pady=1)
            # Color swatch via a small canvas embedded to the left is tricky in
            # tkinter; instead we use the button highlight color and a unicode block.
            btn.configure(text=f"\u2588  {display_name}", fg=hex_color)
            self.channel_buttons.append(btn)

        self._highlight_active_channel()

        # ---- Brush size ----
        tk.Label(
            sidebar, text="Brush size", bg="#222222", fg="#aaaaaa",
            font=("Helvetica", 10),
        ).pack(anchor=tk.W, pady=(14, 0))
        self.brush_var = tk.IntVar(value=self.brush_size)
        self.brush_slider = tk.Scale(
            sidebar, from_=3, to=200, orient=tk.HORIZONTAL, variable=self.brush_var,
            bg="#222222", fg="#cccccc", troughcolor="#444444", highlightthickness=0,
            command=self._on_brush_change,
        )
        self.brush_slider.pack(fill=tk.X)

        # ---- Overlay toggle ----
        self.overlay_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            sidebar, text="Show mask overlay", variable=self.overlay_var,
            bg="#222222", fg="#cccccc", selectcolor="#333333",
            activebackground="#222222", activeforeground="#cccccc",
            command=self._render_canvas,
        ).pack(anchor=tk.W, pady=(10, 0))

        # ---- Action buttons ----
        btn_style = dict(
            bg="#444444", fg="#dddddd", activebackground="#555555",
            activeforeground="#ffffff", relief=tk.FLAT, padx=6, pady=5,
            font=("Helvetica", 10), cursor="hand2",
        )

        tk.Label(sidebar, text="", bg="#222222").pack()  # spacer

        # Undo / Redo row
        undo_redo_frame = tk.Frame(sidebar, bg="#222222")
        undo_redo_frame.pack(fill=tk.X, pady=2)
        tk.Button(undo_redo_frame, text="Undo", command=self._undo, **btn_style).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )
        tk.Button(undo_redo_frame, text="Redo", command=self._redo, **btn_style).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0)
        )

        tk.Button(
            sidebar, text="Fill entire painting (active channel)",
            command=self._fill_active, **btn_style,
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            sidebar, text="Clear active mask",
            command=self._clear_active, **btn_style,
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            sidebar, text="Clear all masks",
            command=self._clear_all, **btn_style,
        ).pack(fill=tk.X, pady=2)

        # Save — green accent
        tk.Button(
            sidebar, text="Save masks", command=self._save_masks,
            bg="#2a8855", fg="#ffffff", activebackground="#35aa66",
            activeforeground="#ffffff", relief=tk.FLAT, padx=6, pady=6,
            font=("Helvetica", 11, "bold"), cursor="hand2",
        ).pack(fill=tk.X, pady=(12, 2))

        # ---- Status label ----
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            sidebar, textvariable=self.status_var, bg="#222222", fg="#888888",
            font=("Helvetica", 9), wraplength=230, justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(10, 0))

        # ---- Canvas ----
        self.canvas = tk.Canvas(
            canvas_frame, width=self.disp_w, height=self.disp_h,
            bg="#111111", highlightthickness=0, cursor="crosshair",
        )
        self.canvas.pack()

        # ---- Canvas mouse bindings ----
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # ---- Keyboard shortcuts ----
        self.master.bind("<Command-z>", lambda e: self._undo())
        self.master.bind("<Control-z>", lambda e: self._undo())
        self.master.bind("<Command-Shift-Z>", lambda e: self._redo())
        self.master.bind("<Control-Shift-Z>", lambda e: self._redo())
        self.master.bind("<Command-s>", lambda e: self._save_masks())
        self.master.bind("<Control-s>", lambda e: self._save_masks())

        # Number keys 1-8 to switch channels
        for i in range(NUM_CHANNELS):
            self.master.bind(str(i + 1), lambda e, idx=i: self._select_channel(idx))

    # ------------------------------------------------------------------
    # Channel selection
    # ------------------------------------------------------------------

    def _select_channel(self, idx: int):
        self.active_channel = idx
        self._highlight_active_channel()
        name = CHANNELS[idx][0].replace("_", " ").title()
        self.status_var.set(f"Active: {name}")

    def _highlight_active_channel(self):
        for i, btn in enumerate(self.channel_buttons):
            if i == self.active_channel:
                btn.configure(bg="#2a8855", relief=tk.FLAT)
            else:
                btn.configure(bg="#333333", relief=tk.FLAT)

    # ------------------------------------------------------------------
    # Brush size
    # ------------------------------------------------------------------

    def _on_brush_change(self, _val):
        self.brush_size = self.brush_var.get()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _canvas_to_orig(self, cx: int, cy: int) -> Tuple[int, int]:
        """Convert canvas (display) coordinates to original image coordinates."""
        ox = int(cx / self.scale)
        oy = int(cy / self.scale)
        return ox, oy

    def _on_press(self, event):
        self.drawing = True
        self.last_xy = (event.x, event.y)
        # Push undo snapshot *before* first stroke
        self._push_undo(self.active_channel)
        self._paint_at(event.x, event.y)
        self._render_canvas()

    def _on_drag(self, event):
        if not self.drawing:
            return
        # Interpolate between last position and current to avoid gaps
        if self.last_xy is not None:
            self._interpolate_and_paint(self.last_xy[0], self.last_xy[1], event.x, event.y)
        else:
            self._paint_at(event.x, event.y)
        self.last_xy = (event.x, event.y)
        self._render_canvas()

    def _on_release(self, _event):
        self.drawing = False
        self.last_xy = None

    def _interpolate_and_paint(self, x0: int, y0: int, x1: int, y1: int):
        """Paint circles along the line from (x0,y0) to (x1,y1) in display coords."""
        dist = max(abs(x1 - x0), abs(y1 - y0), 1)
        # Step size roughly 1/3 of brush radius in display coords for smooth strokes
        step_px = max(1, int(self.brush_size * self.scale / 3))
        steps = max(dist // step_px, 1)
        for i in range(steps + 1):
            t = i / steps
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            self._paint_at(x, y)

    def _paint_at(self, cx: int, cy: int):
        """Paint a filled circle on the active mask at display coords (cx, cy)."""
        ox, oy = self._canvas_to_orig(cx, cy)
        r = self.brush_size
        mask = self.masks[self.active_channel]

        # Compute bounding box in original image coords
        y_min = max(oy - r, 0)
        y_max = min(oy + r + 1, self.img_h)
        x_min = max(ox - r, 0)
        x_max = min(ox + r + 1, self.img_w)

        # Create coordinate grids relative to brush center
        ys = np.arange(y_min, y_max)
        xs = np.arange(x_min, x_max)
        if len(ys) == 0 or len(xs) == 0:
            return
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        dist_sq = (xx - ox) ** 2 + (yy - oy) ** 2
        inside = dist_sq <= r * r
        mask[y_min:y_max, x_min:x_max][inside] = 1

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_canvas(self, *_args):
        """Composite the base image and (optionally) mask overlays, then update canvas."""
        base = np.array(self.display_image, dtype=np.float32)  # (H, W, 3)

        if self.overlay_var.get():
            for idx, (name, color) in enumerate(CHANNELS):
                mask = self.masks[idx]
                if mask.max() == 0:
                    continue
                # Resize mask to display size (nearest-neighbor for binary data)
                mask_disp = np.array(
                    Image.fromarray(mask).resize(
                        (self.disp_w, self.disp_h), Image.NEAREST
                    ),
                    dtype=np.float32,
                )
                alpha = mask_disp * OVERLAY_ALPHA  # (H, W) in [0, OVERLAY_ALPHA]
                for c_idx in range(3):
                    base[:, :, c_idx] = (
                        base[:, :, c_idx] * (1 - alpha) + color[c_idx] * alpha
                    )

        composite = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        self._tk_photo = ImageTk.PhotoImage(composite)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_photo)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def _push_undo(self, channel_idx: int):
        snapshot = (channel_idx, self.masks[channel_idx].copy())
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > HISTORY_LIMIT:
            self.undo_stack.pop(0)
        # Any new stroke clears the redo stack
        self.redo_stack.clear()

    def _undo(self):
        if not self.undo_stack:
            return
        ch_idx, prev_mask = self.undo_stack.pop()
        # Push current state to redo
        self.redo_stack.append((ch_idx, self.masks[ch_idx].copy()))
        self.masks[ch_idx] = prev_mask
        self._render_canvas()
        self.status_var.set("Undo")

    def _redo(self):
        if not self.redo_stack:
            return
        ch_idx, next_mask = self.redo_stack.pop()
        self.undo_stack.append((ch_idx, self.masks[ch_idx].copy()))
        self.masks[ch_idx] = next_mask
        self._render_canvas()
        self.status_var.set("Redo")

    # ------------------------------------------------------------------
    # Mask operations
    # ------------------------------------------------------------------

    def _fill_active(self):
        self._push_undo(self.active_channel)
        self.masks[self.active_channel][:] = 1
        self._render_canvas()
        name = CHANNELS[self.active_channel][0].replace("_", " ").title()
        self.status_var.set(f"Filled: {name}")

    def _clear_active(self):
        self._push_undo(self.active_channel)
        self.masks[self.active_channel][:] = 0
        self._render_canvas()
        name = CHANNELS[self.active_channel][0].replace("_", " ").title()
        self.status_var.set(f"Cleared: {name}")

    def _clear_all(self):
        # Push undo for every non-empty channel
        for idx in range(NUM_CHANNELS):
            if self.masks[idx].max() > 0:
                self._push_undo(idx)
        for idx in range(NUM_CHANNELS):
            self.masks[idx][:] = 0
        self._render_canvas()
        self.status_var.set("All masks cleared")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_masks(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Individual channel PNGs (0/255 binary images)
        for idx, (name, _color) in enumerate(CHANNELS):
            png_path = self.output_dir / f"mask_{name}.png"
            img = Image.fromarray(self.masks[idx] * 255)
            img.save(png_path)

        # 2) Combined tensor  (K, H, W) float32
        if torch is not None:
            stacked = np.stack(self.masks, axis=0).astype(np.float32)  # (K, H, W)
            tensor = torch.from_numpy(stacked)
            pt_path = self.output_dir / "masks.pt"
            torch.save(tensor, pt_path)
        else:
            # Fallback: save as .npy
            stacked = np.stack(self.masks, axis=0).astype(np.float32)
            npy_path = self.output_dir / "masks.npy"
            np.save(npy_path, stacked)
            messagebox.showwarning(
                "torch not available",
                f"PyTorch not installed — saved masks as {npy_path} instead of .pt",
            )

        # 3) Overlay visualization at original resolution
        overlay = np.array(self.orig_image, dtype=np.float32)
        for idx, (name, color) in enumerate(CHANNELS):
            mask = self.masks[idx].astype(np.float32)
            if mask.max() == 0:
                continue
            alpha = mask * OVERLAY_ALPHA
            for c_idx in range(3):
                overlay[:, :, c_idx] = (
                    overlay[:, :, c_idx] * (1 - alpha) + color[c_idx] * alpha
                )
        overlay_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        overlay_img.save(self.output_dir / "overlay.png")

        self.status_var.set(f"Saved to {self.output_dir}")
        messagebox.showinfo("Saved", f"Masks saved to:\n{self.output_dir.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mask Painter — paint binary damage masks for inference.",
    )
    parser.add_argument(
        "image", nargs="?", default=None,
        help="Path to the image file. If omitted, a file dialog will open.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to save masks. Defaults to <image_stem>/ next to the image.",
    )
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("1000x750")

    image_path = args.image
    if image_path is None:
        root.withdraw()
        image_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        root.deiconify()
        if not image_path:
            print("No image selected. Exiting.")
            return

    MaskPainterApp(root, image_path, output_dir=args.output_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
