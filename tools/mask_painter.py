#!/usr/bin/env python3
"""
Mask Painter — browser-based mask painting tool for inference masks.

Works over SSH by running a local HTTP server and serving a self-contained
HTML/JS painting interface. No external dependencies beyond stdlib + torch + PIL.

Usage:
    python tools/mask_painter.py --config inference/configs/inference.yaml
    python tools/mask_painter.py --images img1.jpg img2.jpg --output_dir masks/
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import signal
import socket
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None

# ---------------------------------------------------------------------------
# Channel definitions — must match the corruption module
# ---------------------------------------------------------------------------

CHANNELS = [
    ("craquelure", (255,  80,  80)),
    ("rip_tear",   (230,  50,  50)),
    ("paint_loss", (240, 150,  40)),
    ("yellowing",  (240, 230,  60)),
    ("fading",     (180, 180, 255)),
    ("deposits",   (160, 140, 100)),
    ("scratches",  (220, 100, 220)),
]
NUM_CHANNELS = len(CHANNELS)
OVERLAY_ALPHA = 0.45
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

class ServerState:
    def __init__(self, images: List[Path], output_dir: Path, config_path: Optional[Path]):
        self.images = images
        self.output_dir = output_dir
        self.config_path = config_path
        self.finalized: Dict[int, bool] = {i: False for i in range(len(images))}
        # masks[image_id][channel_idx] = base64 encoded PNG data (or None)
        self.masks: Dict[int, Dict[int, Optional[str]]] = {
            i: {c: None for c in range(NUM_CHANNELS)} for i in range(len(images))
        }
        self.server: Optional[HTTPServer] = None

STATE: Optional[ServerState] = None

# ---------------------------------------------------------------------------
# HTML UI (embedded)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mask Painter</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: #1a1a1a; color: #eee;
    overflow: hidden; height: 100vh;
}
.app { display: flex; height: 100vh; }

/* Sidebar */
.sidebar {
    width: 270px; min-width: 270px;
    background: #222; border-right: 1px solid #333;
    display: flex; flex-direction: column;
    padding: 14px; overflow-y: auto;
}
.sidebar h2 {
    font-size: 15px; font-weight: 600; color: #ccc;
    margin-bottom: 10px;
}
.sidebar h3 {
    font-size: 12px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 0.5px;
    margin: 14px 0 6px 0;
}

/* Image selector */
.image-list {
    max-height: 160px; overflow-y: auto;
    background: #1a1a1a; border: 1px solid #333; border-radius: 6px;
    margin-bottom: 8px;
}
.image-item {
    padding: 6px 10px; cursor: pointer; font-size: 13px;
    display: flex; align-items: center; gap: 6px;
    border-bottom: 1px solid #2a2a2a;
    transition: background 0.1s;
}
.image-item:hover { background: #2a2a2a; }
.image-item.active { background: #2a6634; }
.image-item .check { color: #4c4; font-size: 14px; min-width: 16px; }
.image-item .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.progress-text { font-size: 12px; color: #888; margin-bottom: 10px; }

/* Channel buttons */
.channel-btn {
    display: flex; align-items: center; gap: 8px;
    width: 100%; padding: 7px 10px; margin-bottom: 3px;
    background: #333; border: 1px solid #444; border-radius: 5px;
    color: #ddd; font-size: 13px; cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    text-align: left;
}
.channel-btn:hover { background: #3a3a3a; }
.channel-btn.active { background: #2a6634; border-color: #4a8; }
.channel-swatch {
    width: 14px; height: 14px; border-radius: 3px;
    flex-shrink: 0;
}
.info-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 14px; height: 14px; border-radius: 50%;
    background: rgba(255, 255, 255, 0.18);
    color: rgba(255, 255, 255, 0.95);
    font-size: 10px; font-weight: 700; font-family: serif; font-style: italic;
    margin-left: auto; cursor: help; flex-shrink: 0;
    position: relative;
    user-select: none;
}
.info-icon:hover { background: rgba(255, 255, 255, 0.32); }
.info-icon:hover::after {
    content: attr(data-tooltip);
    position: absolute;
    left: 22px; top: 50%; transform: translateY(-50%);
    background: rgba(20, 20, 20, 0.97);
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: #eee;
    padding: 6px 10px; border-radius: 4px;
    white-space: nowrap; font-size: 12px;
    font-style: normal; font-weight: 400; font-family: inherit;
    z-index: 1000; pointer-events: none;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
}

/* Sliders */
.slider-row {
    display: flex; align-items: center; gap: 8px;
    margin: 4px 0;
}
.slider-row label { font-size: 12px; color: #aaa; min-width: 60px; }
.slider-row input[type=range] { flex: 1; accent-color: #4a8; }
.slider-row .val { font-size: 12px; color: #888; min-width: 30px; text-align: right; }

/* Buttons */
.btn {
    display: block; width: 100%; padding: 8px 10px;
    background: #444; border: 1px solid #555; border-radius: 5px;
    color: #ddd; font-size: 13px; cursor: pointer;
    transition: background 0.15s; margin-bottom: 4px;
    text-align: center;
}
.btn:hover { background: #555; }
.btn:active { background: #666; }
.btn-primary { background: #2a6634; border-color: #3a8; color: #fff; font-weight: 600; }
.btn-primary:hover { background: #35aa55; }
.btn-primary:disabled { background: #333; border-color: #444; color: #666; cursor: not-allowed; }
.btn:disabled { background: #2a2a2a; border-color: #3a3a3a; color: #666; cursor: not-allowed; }
.btn:disabled:hover { background: #2a2a2a; }
.btn-danger { background: #833; border-color: #a44; }
.btn-danger:hover { background: #a44; }
.btn-row { display: flex; gap: 4px; margin-bottom: 4px; }
.btn-row .btn { flex: 1; }

/* Active masks list */
.active-masks { font-size: 12px; color: #888; margin: 4px 0 8px 0; }
.active-masks .mask-tag {
    display: inline-block; padding: 2px 6px; margin: 1px;
    border-radius: 3px; font-size: 11px; color: #fff;
}

/* Toggle */
.toggle-row {
    display: flex; align-items: center; gap: 8px;
    margin: 6px 0; font-size: 13px;
}
.toggle-row input { accent-color: #4a8; }

/* Canvas area */
.canvas-area {
    flex: 1; display: flex; align-items: center; justify-content: center;
    background: #111; position: relative; overflow: hidden;
}
#mainCanvas { cursor: none; }
.cursor-ring {
    position: absolute; border: 2px solid rgba(255,255,255,0.6);
    border-radius: 50%; pointer-events: none;
    transform: translate(-50%, -50%);
    display: none;
}

/* Nav buttons on canvas */
.nav-row {
    position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%);
    display: flex; gap: 8px;
}
.nav-row .btn { width: auto; padding: 6px 18px; }

/* Finalized overlay */
.finalized-banner {
    position: absolute; top: 12px; right: 12px;
    background: rgba(42, 102, 52, 0.9); color: #fff;
    padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 600;
    display: none;
}

/* Scrollbar styling */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
</style>
</head>
<body>
<div class="app">
    <div class="sidebar">
        <h2>Mask Painter</h2>

        <h3>Images</h3>
        <div class="image-list" id="imageList"></div>
        <div class="progress-text" id="progressText">0/0 images finalized</div>

        <h3>Damage type</h3>
        <div id="channelButtons"></div>

        <h3>Brush</h3>
        <div class="slider-row">
            <label>Size</label>
            <input type="range" id="brushSize" min="1" max="100" value="15">
            <span class="val" id="brushSizeVal">15</span>
        </div>

        <h3 style="margin-top:10px;">Actions</h3>
        <div class="btn-row">
            <button class="btn" id="btnUndo" title="Ctrl+Z">Undo</button>
            <button class="btn" id="btnRedo" title="Ctrl+Shift+Z">Redo</button>
        </div>

        <div class="active-masks" id="activeMasks"></div>

        <button class="btn" id="btnFill">Fill entire painting</button>
        <button class="btn" id="btnClearActive">Clear active mask</button>
        <button class="btn btn-danger" id="btnClearAll">Clear all masks</button>

        <div class="toggle-row">
            <input type="checkbox" id="showMasks" checked>
            <label for="showMasks">Show mask overlay</label>
        </div>

        <div style="flex:1;"></div>

        <button class="btn btn-primary" id="btnFinalize" style="margin-top:10px;">Finalize this image</button>
        <button class="btn btn-primary" id="btnComplete" disabled style="margin-top:4px;">Complete &amp; Save All</button>
    </div>

    <div class="canvas-area" id="canvasArea">
        <canvas id="mainCanvas"></canvas>
        <div class="cursor-ring" id="cursorRing"></div>
        <div class="finalized-banner" id="finalizedBanner">Finalized</div>
        <div class="nav-row">
            <button class="btn" id="btnPrev">&#9664; Prev</button>
            <button class="btn" id="btnNext">Next &#9654;</button>
        </div>
    </div>
</div>

<script>
// ---------------------------------------------------------------------------
// Channel definitions
// ---------------------------------------------------------------------------
// globalEnabled mirrors src/corruption/configs/default.yaml — channels
// without a global implementation (rip_tear, paint_loss, scratches) must
// only be painted locally, so the "Fill entire painting" button is
// disabled while those channels are active.
const CHANNELS = [
    { id: "craquelure", label: "Craquelure",        color: [255, 80, 80],   globalEnabled: true  },
    { id: "rip_tear",   label: "Rip / tear",        color: [230, 50, 50],   globalEnabled: false },
    { id: "paint_loss", label: "Paint loss",        color: [240, 150, 40],  globalEnabled: false },
    { id: "yellowing",  label: "Yellowing",         color: [240, 230, 60],  globalEnabled: true  },
    { id: "fading",     label: "Fading",            color: [180, 180, 255], globalEnabled: true  },
    { id: "deposits",   label: "Surface deposits",  color: [160, 140, 100], globalEnabled: true  },
    { id: "scratches",  label: "Scratches",         color: [220, 100, 220], globalEnabled: false },
];
const NUM_CHANNELS = CHANNELS.length;
const OVERLAY_ALPHA = 0.45;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let images = [];          // [{id, name, finalized}]
let currentImageId = -1;
let currentImg = null;    // HTMLImageElement of current image
let activeChannel = 0;
let brushSize = 15;
let showOverlay = true;
let drawing = false;
let lastX = -1, lastY = -1;

// Per-image masks: imageMasks[imageId][channelIdx] = Uint8Array(W*H) or null
const imageMasks = {};
// Undo/redo per image: undoStacks[imageId] = [{channel, data: Uint8Array}]
const undoStacks = {};
const redoStacks = {};
const HISTORY_LIMIT = 50;

// Canvas
const canvas = document.getElementById("mainCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const cursorRing = document.getElementById("cursorRing");

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
    const resp = await fetch("/api/images");
    images = await resp.json();
    buildImageList();
    buildChannelButtons();
    setupEvents();
    if (images.length > 0) {
        loadImage(0);
    }
}

// ---------------------------------------------------------------------------
// Image list
// ---------------------------------------------------------------------------
function buildImageList() {
    const list = document.getElementById("imageList");
    list.innerHTML = "";
    images.forEach((img, idx) => {
        const div = document.createElement("div");
        div.className = "image-item" + (idx === currentImageId ? " active" : "");
        div.innerHTML = `<span class="check">${img.finalized ? "&#10003;" : ""}</span><span class="name">${img.name}</span>`;
        div.onclick = () => loadImage(idx);
        list.appendChild(div);
    });
    updateProgress();
}

function updateImageList() {
    const items = document.querySelectorAll(".image-item");
    items.forEach((item, idx) => {
        item.className = "image-item" + (idx === currentImageId ? " active" : "");
        item.querySelector(".check").innerHTML = images[idx].finalized ? "&#10003;" : "";
    });
    updateProgress();
}

function updateProgress() {
    const done = images.filter(i => i.finalized).length;
    document.getElementById("progressText").textContent = `${done}/${images.length} images finalized`;
    document.getElementById("btnComplete").disabled = done < images.length;
}

// ---------------------------------------------------------------------------
// Load image
// ---------------------------------------------------------------------------
async function loadImage(id) {
    currentImageId = id;
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = `/api/image/${id}?t=${Date.now()}`;
    });
    currentImg = img;

    // Init masks if needed
    if (!imageMasks[id]) {
        imageMasks[id] = {};
        for (let c = 0; c < NUM_CHANNELS; c++) {
            imageMasks[id][c] = new Uint8Array(img.width * img.height);
        }
    }
    if (!undoStacks[id]) undoStacks[id] = [];
    if (!redoStacks[id]) redoStacks[id] = [];

    resizeCanvas();
    renderCanvas();
    updateImageList();
    updateActiveMasks();
    updateFinalizedBanner();
    updateFinalizeButton();
}

// ---------------------------------------------------------------------------
// Canvas sizing
// ---------------------------------------------------------------------------
function resizeCanvas() {
    if (!currentImg) return;
    const area = document.getElementById("canvasArea");
    const maxW = area.clientWidth - 40;
    const maxH = area.clientHeight - 60;
    const scale = Math.min(maxW / currentImg.width, maxH / currentImg.height, 1.0);
    canvas.width = Math.floor(currentImg.width * scale);
    canvas.height = Math.floor(currentImg.height * scale);
    canvas._scale = scale;
    canvas._imgW = currentImg.width;
    canvas._imgH = currentImg.height;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderCanvas() {
    if (!currentImg) return;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(currentImg, 0, 0, w, h);

    if (showOverlay && imageMasks[currentImageId]) {
        const imgData = ctx.getImageData(0, 0, w, h);
        const pixels = imgData.data;
        const scale = canvas._scale;
        const imgW = canvas._imgW, imgH = canvas._imgH;

        for (let c = 0; c < NUM_CHANNELS; c++) {
            const mask = imageMasks[currentImageId][c];
            if (!mask) continue;
            // Check if mask has any painted pixels
            let hasPixels = false;
            for (let i = 0; i < mask.length; i++) {
                if (mask[i]) { hasPixels = true; break; }
            }
            if (!hasPixels) continue;

            const [cr, cg, cb] = CHANNELS[c].color;
            const alpha = OVERLAY_ALPHA;

            for (let dy = 0; dy < h; dy++) {
                const srcY = Math.min(Math.floor(dy / scale), imgH - 1);
                for (let dx = 0; dx < w; dx++) {
                    const srcX = Math.min(Math.floor(dx / scale), imgW - 1);
                    if (mask[srcY * imgW + srcX]) {
                        const pi = (dy * w + dx) * 4;
                        pixels[pi]     = pixels[pi]     * (1 - alpha) + cr * alpha;
                        pixels[pi + 1] = pixels[pi + 1] * (1 - alpha) + cg * alpha;
                        pixels[pi + 2] = pixels[pi + 2] * (1 - alpha) + cb * alpha;
                    }
                }
            }
        }
        ctx.putImageData(imgData, 0, 0);
    }
}

// ---------------------------------------------------------------------------
// Painting
// ---------------------------------------------------------------------------
function canvasToOrig(cx, cy) {
    const scale = canvas._scale;
    return [Math.floor(cx / scale), Math.floor(cy / scale)];
}

function paintAt(cx, cy) {
    if (!currentImg || !imageMasks[currentImageId]) return;
    const [ox, oy] = canvasToOrig(cx, cy);
    const mask = imageMasks[currentImageId][activeChannel];
    const imgW = canvas._imgW, imgH = canvas._imgH;
    const r = brushSize;
    const r2 = r * r;

    const yMin = Math.max(oy - r, 0);
    const yMax = Math.min(oy + r + 1, imgH);
    const xMin = Math.max(ox - r, 0);
    const xMax = Math.min(ox + r + 1, imgW);

    for (let y = yMin; y < yMax; y++) {
        const dy = y - oy;
        const dy2 = dy * dy;
        for (let x = xMin; x < xMax; x++) {
            const dx = x - ox;
            if (dx * dx + dy2 <= r2) {
                mask[y * imgW + x] = 1;
            }
        }
    }
}

function interpolateAndPaint(x0, y0, x1, y1) {
    const dist = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
    const stepPx = Math.max(1, Math.floor(brushSize * canvas._scale / 3));
    const steps = Math.max(Math.floor(dist / stepPx), 1);
    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const x = Math.round(x0 + (x1 - x0) * t);
        const y = Math.round(y0 + (y1 - y0) * t);
        paintAt(x, y);
    }
}

// ---------------------------------------------------------------------------
// Undo / Redo
// ---------------------------------------------------------------------------
function pushUndo(channel) {
    const stack = undoStacks[currentImageId];
    const mask = imageMasks[currentImageId][channel];
    stack.push({ channel, data: new Uint8Array(mask) });
    if (stack.length > HISTORY_LIMIT) stack.shift();
    redoStacks[currentImageId] = [];
}

function undo() {
    const stack = undoStacks[currentImageId];
    if (!stack || stack.length === 0) return;
    const entry = stack.pop();
    const currentData = new Uint8Array(imageMasks[currentImageId][entry.channel]);
    redoStacks[currentImageId].push({ channel: entry.channel, data: currentData });
    imageMasks[currentImageId][entry.channel] = entry.data;
    renderCanvas();
    updateActiveMasks();
}

function redo() {
    const stack = redoStacks[currentImageId];
    if (!stack || stack.length === 0) return;
    const entry = stack.pop();
    const currentData = new Uint8Array(imageMasks[currentImageId][entry.channel]);
    undoStacks[currentImageId].push({ channel: entry.channel, data: currentData });
    imageMasks[currentImageId][entry.channel] = entry.data;
    renderCanvas();
    updateActiveMasks();
}

// ---------------------------------------------------------------------------
// Mask operations
// ---------------------------------------------------------------------------
function fillActive() {
    const ch = CHANNELS[activeChannel];
    if (ch && ch.globalEnabled === false) return;
    pushUndo(activeChannel);
    imageMasks[currentImageId][activeChannel].fill(1);
    renderCanvas();
    updateActiveMasks();
}

function clearActive() {
    pushUndo(activeChannel);
    imageMasks[currentImageId][activeChannel].fill(0);
    renderCanvas();
    updateActiveMasks();
}

function clearAll() {
    for (let c = 0; c < NUM_CHANNELS; c++) {
        const mask = imageMasks[currentImageId][c];
        let hasPixels = false;
        for (let i = 0; i < mask.length; i++) { if (mask[i]) { hasPixels = true; break; } }
        if (hasPixels) pushUndo(c);
        mask.fill(0);
    }
    renderCanvas();
    updateActiveMasks();
}

function updateActiveMasks() {
    const container = document.getElementById("activeMasks");
    if (!imageMasks[currentImageId]) { container.innerHTML = ""; return; }
    let html = "";
    for (let c = 0; c < NUM_CHANNELS; c++) {
        const mask = imageMasks[currentImageId][c];
        let hasPixels = false;
        for (let i = 0; i < mask.length; i++) { if (mask[i]) { hasPixels = true; break; } }
        if (hasPixels) {
            const [r, g, b] = CHANNELS[c].color;
            html += `<span class="mask-tag" style="background:rgba(${r},${g},${b},0.6)">${CHANNELS[c].label}</span> `;
        }
    }
    container.innerHTML = html || '<span style="color:#555;">No masks painted</span>';
}

// ---------------------------------------------------------------------------
// Channel buttons
// ---------------------------------------------------------------------------
function buildChannelButtons() {
    const container = document.getElementById("channelButtons");
    container.innerHTML = "";
    CHANNELS.forEach((ch, idx) => {
        const btn = document.createElement("button");
        btn.className = "channel-btn" + (idx === activeChannel ? " active" : "");
        const [r, g, b] = ch.color;
        let html = `<span class="channel-swatch" style="background:rgb(${r},${g},${b})"></span><span>${ch.label}</span>`;
        if (ch.tooltip) {
            html += `<span class="info-icon" data-tooltip="${ch.tooltip}" onclick="event.stopPropagation();">i</span>`;
        }
        btn.innerHTML = html;
        btn.onclick = () => selectChannel(idx);
        container.appendChild(btn);
    });
    updateFillButton();
}

function selectChannel(idx) {
    activeChannel = idx;
    document.querySelectorAll(".channel-btn").forEach((btn, i) => {
        btn.className = "channel-btn" + (i === activeChannel ? " active" : "");
    });
    updateFillButton();
}

function updateFillButton() {
    const btn = document.getElementById("btnFill");
    if (!btn) return;
    const ch = CHANNELS[activeChannel];
    if (ch && ch.globalEnabled === false) {
        btn.disabled = true;
        btn.title = `${ch.label} has no global implementation — paint locally instead.`;
    } else {
        btn.disabled = false;
        btn.title = "";
    }
}

// ---------------------------------------------------------------------------
// Finalization
// ---------------------------------------------------------------------------
function updateFinalizedBanner() {
    const banner = document.getElementById("finalizedBanner");
    banner.style.display = images[currentImageId]?.finalized ? "block" : "none";
}

function updateFinalizeButton() {
    const btn = document.getElementById("btnFinalize");
    if (images[currentImageId]?.finalized) {
        btn.textContent = "Edit (un-finalize)";
        btn.classList.remove("btn-primary");
        btn.classList.add("btn");
    } else {
        btn.textContent = "Finalize this image";
        btn.classList.add("btn-primary");
    }
}

async function finalize() {
    if (images[currentImageId].finalized) {
        // Un-finalize
        images[currentImageId].finalized = false;
        updateImageList();
        updateFinalizedBanner();
        updateFinalizeButton();
        return;
    }

    // Send masks to server
    const maskData = {};
    const imgW = canvas._imgW, imgH = canvas._imgH;
    for (let c = 0; c < NUM_CHANNELS; c++) {
        const mask = imageMasks[currentImageId][c];
        // Encode as base64 raw bytes
        const b64 = uint8ArrayToBase64(mask);
        maskData[c] = b64;
    }

    const resp = await fetch("/api/save_mask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            image_id: currentImageId,
            width: imgW,
            height: imgH,
            masks: maskData
        })
    });

    if (!resp.ok) {
        alert("Failed to save masks: " + (await resp.text()));
        return;
    }

    await fetch(`/api/finalize/${currentImageId}`, { method: "POST" });
    images[currentImageId].finalized = true;
    updateImageList();
    updateFinalizedBanner();
    updateFinalizeButton();

    // Auto-advance to next unfinalized
    const nextIdx = images.findIndex((img, i) => i > currentImageId && !img.finalized);
    if (nextIdx >= 0) {
        loadImage(nextIdx);
    } else {
        const wrapIdx = images.findIndex(img => !img.finalized);
        if (wrapIdx >= 0) loadImage(wrapIdx);
    }
}

async function completeAll() {
    const allDone = images.every(i => i.finalized);
    if (!allDone) { alert("Please finalize all images first."); return; }

    const resp = await fetch("/api/complete", { method: "POST" });
    if (resp.ok) {
        document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-size:20px;color:#4c4;">All masks saved. You can close this tab.</div>';
    } else {
        alert("Error: " + (await resp.text()));
    }
}

// Base64 encode Uint8Array
function uint8ArrayToBase64(arr) {
    let binary = "";
    const len = arr.length;
    const chunkSize = 8192;
    for (let i = 0; i < len; i += chunkSize) {
        const chunk = arr.subarray(i, Math.min(i + chunkSize, len));
        binary += String.fromCharCode.apply(null, chunk);
    }
    return btoa(binary);
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function prevImage() {
    if (currentImageId > 0) loadImage(currentImageId - 1);
}
function nextImage() {
    if (currentImageId < images.length - 1) loadImage(currentImageId + 1);
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
function setupEvents() {
    // Brush size
    const brushSlider = document.getElementById("brushSize");
    const brushVal = document.getElementById("brushSizeVal");
    brushSlider.oninput = () => { brushSize = parseInt(brushSlider.value); brushVal.textContent = brushSize; };

    // Overlay toggle
    document.getElementById("showMasks").onchange = (e) => { showOverlay = e.target.checked; renderCanvas(); };

    // Buttons
    document.getElementById("btnUndo").onclick = undo;
    document.getElementById("btnRedo").onclick = redo;
    document.getElementById("btnFill").onclick = fillActive;
    document.getElementById("btnClearActive").onclick = clearActive;
    document.getElementById("btnClearAll").onclick = clearAll;
    document.getElementById("btnFinalize").onclick = finalize;
    document.getElementById("btnComplete").onclick = completeAll;
    document.getElementById("btnPrev").onclick = prevImage;
    document.getElementById("btnNext").onclick = nextImage;

    // Canvas painting
    // Convert mouse event to canvas pixel coords (handles CSS scaling)
    function mouseToCanvas(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return [
            (e.clientX - rect.left) * scaleX,
            (e.clientY - rect.top) * scaleY,
        ];
    }

    canvas.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;
        drawing = true;
        const [cx, cy] = mouseToCanvas(e);
        pushUndo(activeChannel);
        paintAt(cx, cy);
        lastX = cx; lastY = cy;
        renderCanvas();
    });

    canvas.addEventListener("mousemove", (e) => {
        const [cx, cy] = mouseToCanvas(e);
        updateCursor(e);
        if (drawing) {
            interpolateAndPaint(lastX, lastY, cx, cy);
            lastX = cx; lastY = cy;
            renderCanvas();
        }
    });

    canvas.addEventListener("mouseup", () => { drawing = false; lastX = -1; lastY = -1; updateActiveMasks(); });
    canvas.addEventListener("mouseleave", () => { drawing = false; lastX = -1; lastY = -1; cursorRing.style.display = "none"; });

    canvas.addEventListener("mouseenter", () => { cursorRing.style.display = "block"; });

    // Keyboard
    document.addEventListener("keydown", (e) => {
        if (e.key === "ArrowLeft") { e.preventDefault(); prevImage(); }
        else if (e.key === "ArrowRight") { e.preventDefault(); nextImage(); }
        else if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "z") { e.preventDefault(); redo(); }
        else if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); undo(); }
        else if (e.key >= "1" && e.key <= "9") { selectChannel(parseInt(e.key) - 1); }
    });

    // Window resize
    window.addEventListener("resize", () => { resizeCanvas(); renderCanvas(); });
}

function updateCursor(e) {
    const ring = cursorRing;
    const rect = canvas.getBoundingClientRect();
    const cssScale = rect.width / canvas.width;
    const displaySize = brushSize * 2 * (canvas._scale || 1) * cssScale;
    ring.style.width = displaySize + "px";
    ring.style.height = displaySize + "px";
    // Position relative to canvas-area (the position:relative container)
    const areaRect = canvas.parentElement.getBoundingClientRect();
    ring.style.left = (e.clientX - areaRect.left) + "px";
    ring.style.top = (e.clientY - areaRect.top) + "px";
    ring.style.display = "block";
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
init();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class MaskPainterHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status, msg):
        self._send_json({"error": msg}, status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_html(HTML_PAGE)
            return

        if path == "/api/images":
            result = []
            for i, img_path in enumerate(STATE.images):
                result.append({
                    "id": i,
                    "name": img_path.name,
                    "finalized": STATE.finalized[i],
                })
            self._send_json(result)
            return

        if path.startswith("/api/image/"):
            try:
                img_id = int(path.split("/")[-1])
            except ValueError:
                self._send_error(400, "Invalid image ID")
                return
            if img_id < 0 or img_id >= len(STATE.images):
                self._send_error(404, "Image not found")
                return
            img_path = STATE.images[img_id]
            mime = mimetypes.guess_type(str(img_path))[0] or "image/jpeg"
            try:
                data = img_path.read_bytes()
            except Exception as e:
                self._send_error(500, str(e))
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/save_mask":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_error(400, "Invalid JSON")
                return

            img_id = data.get("image_id")
            width = data.get("width")
            height = data.get("height")
            masks_data = data.get("masks", {})

            if img_id is None or img_id < 0 or img_id >= len(STATE.images):
                self._send_error(400, "Invalid image_id")
                return

            # Decode and store masks
            for ch_str, b64data in masks_data.items():
                ch = int(ch_str)
                raw = base64.b64decode(b64data)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
                STATE.masks[img_id][ch] = (arr, width, height)

            self._send_json({"status": "ok"})
            return

        if path.startswith("/api/finalize/"):
            try:
                img_id = int(path.split("/")[-1])
            except ValueError:
                self._send_error(400, "Invalid image ID")
                return
            STATE.finalized[img_id] = True
            print(f"  Finalized: {STATE.images[img_id].name}")
            self._send_json({"status": "ok"})
            return

        if path == "/api/complete":
            try:
                save_all_masks(STATE)
                self._send_json({"status": "ok"})
                # Schedule shutdown
                threading.Timer(0.5, shutdown_server).start()
            except Exception as e:
                self._send_error(500, str(e))
            return

        self._send_error(404, "Not found")


# ---------------------------------------------------------------------------
# Mask saving
# ---------------------------------------------------------------------------

def save_all_masks(state: ServerState):
    """Save all masks to disk and update config."""
    state.output_dir.mkdir(parents=True, exist_ok=True)

    for img_id, img_path in enumerate(state.images):
        stem = img_path.stem
        img_dir = state.output_dir / stem
        img_dir.mkdir(parents=True, exist_ok=True)

        # Load original image for overlay
        orig = Image.open(img_path).convert("RGB")
        img_w, img_h = orig.size

        mask_arrays = []
        for ch in range(NUM_CHANNELS):
            mask_data = state.masks[img_id].get(ch)
            if mask_data is not None and isinstance(mask_data, tuple):
                arr, w, h = mask_data
                # Resize if needed
                if w != img_w or h != img_h:
                    mask_img = Image.fromarray(arr * 255).resize((img_w, img_h), Image.NEAREST)
                    arr = (np.array(mask_img) > 127).astype(np.uint8)
                mask_arrays.append(arr)
            else:
                mask_arrays.append(np.zeros((img_h, img_w), dtype=np.uint8))

        # Save individual PNGs
        for ch in range(NUM_CHANNELS):
            name = CHANNELS[ch][0]
            png_path = img_dir / f"mask_{name}.png"
            Image.fromarray(mask_arrays[ch] * 255).save(png_path)

        # Save combined tensor
        stacked = np.stack(mask_arrays, axis=0).astype(np.float32)
        if torch is not None:
            tensor = torch.from_numpy(stacked)
            torch.save(tensor, img_dir / "masks.pt")
        else:
            np.save(img_dir / "masks.npy", stacked)
            print(f"  Warning: torch not available, saved {stem}/masks.npy instead of .pt")

        # Save overlay
        overlay = np.array(orig, dtype=np.float32)
        for ch in range(NUM_CHANNELS):
            mask = mask_arrays[ch].astype(np.float32)
            if mask.max() == 0:
                continue
            color = CHANNELS[ch][1]
            alpha = mask * OVERLAY_ALPHA
            for c_idx in range(3):
                overlay[:, :, c_idx] = (
                    overlay[:, :, c_idx] * (1 - alpha) + color[c_idx] * alpha
                )
        overlay_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        overlay_img.save(img_dir / "overlay.png")

        print(f"  Saved masks for: {img_path.name} -> {img_dir}")

    # Update config if provided
    if state.config_path and yaml is not None:
        update_config(state.config_path, str(state.output_dir))

    print(f"\nDone! Masks saved to {state.output_dir.resolve()}")


def update_config(config_path: Path, mask_dir: str):
    """Update the inference config YAML with mask_dir."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if "inference" in config:
            config["inference"]["mask_dir"] = mask_dir

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"  Updated config: {config_path}")
    except Exception as e:
        print(f"  Warning: Could not update config: {e}")


def shutdown_server():
    """Gracefully shut down the HTTP server."""
    if STATE and STATE.server:
        STATE.server.shutdown()


# ---------------------------------------------------------------------------
# Find available port
# ---------------------------------------------------------------------------

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Find images in directory
# ---------------------------------------------------------------------------

def find_images(input_dir: Path) -> List[Path]:
    """Find all image files recursively in the given directory."""
    images = []
    if not input_dir.exists():
        return images
    for f in sorted(input_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    return images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global STATE

    parser = argparse.ArgumentParser(
        description="Mask Painter — browser-based mask painting tool.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to inference config YAML file.",
    )
    parser.add_argument(
        "--images", nargs="+", default=None,
        help="Image paths to paint masks for.",
    )
    parser.add_argument(
        "--output_dir", default="./masks",
        help="Directory to save masks (default: ./masks).",
    )
    parser.add_argument(
        "--port", type=int, default=0,
        help="Port to run server on (default: auto-select).",
    )
    args = parser.parse_args()

    # Determine images and output dir
    image_paths = []
    output_dir = Path(args.output_dir)
    config_path = None

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

        if yaml is None:
            print("Error: PyYAML is required for config mode. Install with: pip install pyyaml")
            sys.exit(1)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        inference_cfg = config.get("inference", {})
        input_dir = Path(inference_cfg.get("input_dir", "./data/test"))

        # Resolve relative paths to CWD (not config file location)
        if not input_dir.is_absolute():
            input_dir = Path.cwd() / input_dir
        input_dir = input_dir.resolve()

        print(f"Looking for images in: {input_dir}")
        image_paths = find_images(input_dir)
        if not image_paths:
            print(f"Error: No images found in {input_dir}")
            print(f"  (searched recursively for {', '.join(IMAGE_EXTENSIONS)})")
            sys.exit(1)

        # Output dir relative to CWD
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

    elif args.images:
        image_paths = [Path(p).resolve() for p in args.images]
        for p in image_paths:
            if not p.exists():
                print(f"Error: Image not found: {p}")
                sys.exit(1)
        output_dir = Path(args.output_dir).resolve()

    else:
        print("Error: Provide --config or --images")
        sys.exit(1)

    print(f"Found {len(image_paths)} images:")
    for p in image_paths:
        print(f"  {p.name}")

    STATE = ServerState(image_paths, output_dir, config_path)

    port = args.port if args.port else find_free_port()
    server = HTTPServer(("0.0.0.0", port), MaskPainterHandler)
    STATE.server = server

    # Detect hostname for SSH instructions
    hostname = socket.gethostname()
    username = os.environ.get("USER", os.environ.get("USERNAME", "user"))

    print(f"\n{'='*60}")
    print(f"  Mask Painter running on port {port}")
    print(f"  Host: {hostname}  User: {username}")
    print(f"")
    print(f"  Open in your browser:")
    print(f"    http://localhost:{port}")
    print(f"")
    print(f"  If running on a remote machine, first set up port")
    print(f"  forwarding in a NEW local terminal:")
    print(f"")
    print(f"  Direct SSH:")
    print(f"    ssh -N -L {port}:localhost:{port} {username}@<REMOTE_HOST>")
    print(f"")
    print(f"  Behind a login/jump node (e.g. HPC):")
    print(f"    ssh -N -L {port}:{hostname}:{port} {username}@<LOGIN_NODE>")
    print(f"")
    print(f"  Then open http://localhost:{port} in your browser.")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
