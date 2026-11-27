"""
Pure-CPU Vectorized / Numba-Accelerated Fallback Renderer.
Used automatically when running in headless environments without an OpenGL GPU context
(e.g., CI containers or remote Linux servers without EGL/WGL).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from config import QualityPreset


class CPUFallbackRenderer:
    """Simplified NumPy/SciPy raymarching & bloom fallback preserving the 4-act narrative arc."""

    def __init__(self, preset: QualityPreset) -> None:
        self.preset = preset
        self.w = preset.width
        self.h = preset.height
        self.gpu_name = "CPU Vectorized Fallback (NumPy + SciPy)"

        # Render internal raymarch on a half-scale grid if >=1080p on CPU to maintain reasonable throughput
        self.scale = 2 if self.w >= 1280 else 1
        self.rw = self.w // self.scale
        self.rh = self.h // self.scale
        xs = (np.arange(self.rw, dtype=np.float32) + 0.5 - 0.5 * self.rw) / self.rh
        ys = (np.arange(self.rh, dtype=np.float32) + 0.5 - 0.5 * self.rh) / self.rh
        self.ux, self.uy = np.meshgrid(xs, ys)

    def render_frame(self, t: float, audio_vec: Tuple[float, float, float, float], energy: float) -> np.ndarray:
        sub, mid, high, impact = audio_vec
        a1 = float(1.0 - np.clip((t - 8.8) / 2.4, 0.0, 1.0))
        a2 = float(np.clip((t - 8.8) / 2.4, 0.0, 1.0) * (1.0 - np.clip((t - 18.8) / 2.4, 0.0, 1.0)))
        a3 = float(np.clip((t - 18.8) / 2.4, 0.0, 1.0) * (1.0 - np.clip((t - 29.3) / 1.5, 0.0, 1.0)))
        a4 = float(np.clip((t - 29.3) / 1.5, 0.0, 1.0))

        ang = -0.5 + t * 0.14
        rad = 3.2 * a1 + 4.6 * a2 + 3.6 * a3 + 5.2 * a4
        ro = np.array([math.sin(ang) * rad, 0.25 + 0.6 * a2 + 0.2 * a4, math.cos(ang) * rad], dtype=np.float32)
        ta = np.array([0.0, 0.15 * a2 + 0.2 * a4, 0.0], dtype=np.float32)
        ww = ta - ro
        ww /= np.linalg.norm(ww)
        uu = np.cross(ww, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        uu /= np.linalg.norm(uu)
        vv = np.cross(uu, ww)

        rdx = self.ux * uu[0] + self.uy * vv[0] + 1.25 * ww[0]
        rdy = self.ux * uu[1] + self.uy * vv[1] + 1.25 * ww[1]
        rdz = self.ux * uu[2] + self.uy * vv[2] + 1.25 * ww[2]
        inv_len = 1.0 / np.sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
        rdx *= inv_len
        rdy *= inv_len
        rdz *= inv_len

        dist_t = np.full((self.rh, self.rw), 0.1, dtype=np.float32)
        glow_acc = np.zeros((self.rh, self.rw), dtype=np.float32)
        trap_min = np.full((self.rh, self.rw), 10.0, dtype=np.float32)

        steps = min(32, self.preset.max_march_steps)
        fold = 1.0 + 0.15 * a2 + 0.04 * sub
        scale_f = 2.05

        for _ in range(steps):
            px = ro[0] + rdx * dist_t
            py = ro[1] + rdy * dist_t
            pz = ro[2] + rdz * dist_t

            zx, zy, zz = px.copy(), py.copy(), pz.copy()
            dz = np.ones_like(zx)
            for _ in range(4):
                zx = np.clip(zx, -fold, fold) * 2.0 - zx
                zy = np.clip(zy, -fold, fold) * 2.0 - zy
                zz = np.clip(zz, -fold, fold) * 2.0 - zz
                r2 = zx * zx + zy * zy + zz * zz
                trap_min = np.minimum(trap_min, np.abs(r2 - 0.65))
                k = np.clip(0.3 / np.maximum(r2, 1e-3), 0.3, 1.0) / 0.3
                zx = zx * k * scale_f + px
                zy = zy * k * scale_f + py
                zz = zz * k * scale_f + pz
                dz = dz * k * scale_f + 1.0

            d_frac = (np.sqrt(zx * zx + zy * zy + zz * zz) - 1.4) / np.abs(dz)
            bound = np.sqrt(px * px + py * py + pz * pz) - (1.25 + 1.8 * a2 + 0.6 * a3 + 0.2 * a4)
            d = np.maximum(d_frac, bound)
            glow_acc += 0.02 / (0.04 + d * d)
            dist_t += d * 0.85

        hit_mask = (dist_t < 15.0).astype(np.float32)
        c1 = np.array([0.08, 0.65, 1.15], dtype=np.float32)
        c2 = np.array([1.25, 0.75, 0.22], dtype=np.float32)
        c3 = np.array([1.45, 0.35, 0.65], dtype=np.float32)
        c4 = np.array([0.75, 0.88, 1.15], dtype=np.float32)
        tint = c1 * a1 + c2 * a2 + c3 * a3 + c4 * a4

        hdr = np.zeros((self.rh, self.rw, 3), dtype=np.float32)
        shade = hit_mask * np.exp(-trap_min * 3.0) * (0.8 + 0.8 * high) + glow_acc * 0.045 * (0.7 + 0.6 * sub)
        for ch in range(3):
            hdr[:, :, ch] = shade * tint[ch] + 0.015 * tint[ch]

        bloom = gaussian_filter(np.maximum(hdr - 0.5, 0.0), sigma=(3.0, 3.0, 0.0))
        hdr += bloom * 0.8
        ldr = np.clip((hdr * (2.51 * hdr + 0.03)) / (hdr * (2.43 * hdr + 0.59) + 0.14), 0.0, 1.0)
        srgb = (np.power(ldr, 1.0 / 2.2) * 255.0).astype(np.uint8)
        if self.scale > 1:
            srgb = np.repeat(np.repeat(srgb, self.scale, axis=0), self.scale, axis=1)[: self.h, : self.w]
        return srgb
