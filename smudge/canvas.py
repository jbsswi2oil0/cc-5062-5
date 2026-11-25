"""Canvas storage and paint quantity bookkeeping.

The canvas keeps *straight* (non-premultiplied) RGBA in float64 with values in
[0, 1]. Paint transport, however, is computed in premultiplied quantities:

    P_rgb = rgb * alpha          (premultiplied color "amount")
    P_a   = alpha                (total paint volume)

Premultiplied amounts are additive when paint is laid down, so conservation of
paint is a plain sum of premultiplied channels. Transparent pixels carry zero
paint in every channel, which makes moving paint across transparent regions
unambiguous: nothing is picked up, and deposited paint keeps its own color.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Canvas:
    def __init__(self, height: int, width: int, fill=(0.0, 0.0, 0.0, 0.0)):
        self.rgba = np.zeros((height, width, 4), dtype=np.float64)
        self.rgba[:] = fill

    @property
    def height(self) -> int:
        return self.rgba.shape[0]

    @property
    def width(self) -> int:
        return self.rgba.shape[1]

    @classmethod
    def from_rgba_array(cls, arr: np.ndarray) -> "Canvas":
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 3 or a.shape[2] != 4:
            raise ValueError("expected an HxWx4 RGBA array")
        c = cls(a.shape[0], a.shape[1])
        c.rgba[:] = np.clip(a, 0.0, 1.0)
        return c

    @classmethod
    def from_png(cls, path: str | Path) -> "Canvas":
        from PIL import Image

        img = Image.open(path).convert("RGBA")
        arr = np.asarray(img, dtype=np.float64) / 255.0
        return cls.from_rgba_array(arr)

    def save_png(self, path: str | Path) -> None:
        from PIL import Image

        u8 = self.to_uint8()
        Image.fromarray(u8, mode="RGBA").save(path)

    def to_uint8(self) -> np.ndarray:
        return np.rint(np.clip(self.rgba, 0.0, 1.0) * 255.0).astype(np.uint8)

    def save_raw(self, path: str | Path) -> None:
        """Export exact float64 pixels for off-line verification."""
        np.save(path, self.rgba)

    # ---- premultiplied paint view -------------------------------------

    def premultiplied(self) -> np.ndarray:
        """HxWx4 array (r*a, g*a, b*a, a)."""
        out = np.empty_like(self.rgba)
        a = self.rgba[..., 3:4]
        out[..., :3] = self.rgba[..., :3] * a
        out[..., 3] = self.rgba[..., 3]
        return out

    def write_premultiplied(
        self, rows: np.ndarray, cols: np.ndarray, pm: np.ndarray
    ) -> np.ndarray:
        """Replace pixels by premultiplied amounts; return per-channel spill.

        Alpha above 1 is clamped (a finite canvas pixel cannot hold more than
        full coverage); the overflow mass is returned so the session ledger can
        report it. Straight RGB of fully transparent pixels is defined as 0.
        """
        a_new = np.clip(pm[:, 3], 0.0, 1.0)
        spill = np.maximum(pm[:, 3] - 1.0, 0.0)
        spill_rgb = np.maximum(pm[:, :3] - a_new[:, None], 0.0)
        rgb_new = np.zeros_like(pm[:, :3])
        nz = a_new > 1e-12
        rgb_new[nz] = pm[nz, :3] / a_new[nz, None]
        rgb_new = np.clip(rgb_new, 0.0, 1.0)
        self.rgba[rows, cols, :3] = rgb_new
        self.rgba[rows, cols, 3] = a_new
        return np.concatenate([spill_rgb, spill[:, None]], axis=1)

    def paint_mass(self) -> np.ndarray:
        """Total premultiplied paint on the canvas, summed per channel."""
        return self.premultiplied().reshape(-1, 4).sum(axis=0)
