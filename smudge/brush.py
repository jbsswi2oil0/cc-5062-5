"""Brush footprint kernels.

A footprint is the set of integer pixels touched at a given sub-pixel brush
position together with non-negative weights. Weights are computed by
supersampling a radial profile over each pixel's square area, so the same kernel
evaluated at any sub-pixel position uses the same definition.

Two profiles:
  "disk"     - uniform weight inside radius r (hard circular brush);
  "gaussian" - exp(-(d / sigma)^2 / 2), clipped to radius r, where
               sigma = r / 2 by default.

Weights are normalized so the full (unclipped) footprint sums to 1. When the
brush overlaps the canvas edge the remaining weights are renormalized inside
the canvas; that is reported via `clipped_fraction`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Footprint:
    rows: np.ndarray          # int pixel rows
    cols: np.ndarray          # int pixel cols
    weights: np.ndarray       # float64, sum == 1 over in-canvas pixels
    clipped_fraction: float   # profile weight lost outside the canvas [0, 1)


class Brush:
    def __init__(
        self,
        radius: float,
        shape: str = "disk",
        supersample: int = 5,
        sigma_scale: float = 0.5,
    ):
        if radius <= 0:
            raise ValueError("brush radius must be positive")
        if shape not in ("disk", "gaussian"):
            raise ValueError("shape must be 'disk' or 'gaussian'")
        if supersample < 1:
            raise ValueError("supersample must be >= 1")
        self.radius = float(radius)
        self.shape = shape
        self.supersample = int(supersample)
        self.sigma = self.radius * sigma_scale
        rint = int(math.ceil(self.radius))
        # offsets relative to the anchor pixel, fixed at construction time
        dy, dx = np.mgrid[-rint:rint + 1, -rint:rint + 1].astype(np.float64)
        self._dy = dy.ravel()
        self._dx = dx.ravel()
        self._profile = self._profile_weights(dy.ravel(), dx.ravel())

    def _profile_weights(self, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Mean profile value over each unit pixel square (supersampled)."""
        s = self.supersample
        # sample positions inside each pixel square, centered on the pixel
        offs = (np.arange(s, dtype=np.float64) + 0.5) / s - 0.5
        oy, ox = np.meshgrid(offs, offs, indexing="ij")
        oy = oy.ravel()
        ox = ox.ravel()
        # distance of every sub-sample of every candidate pixel
        ddy = dy[:, None] + oy[None, :]
        ddx = dx[:, None] + ox[None, :]
        dist = np.sqrt(ddy * ddy + ddx * ddx)
        if self.shape == "disk":
            inside = (dist <= self.radius).astype(np.float64)
            w = inside.mean(axis=1)
        else:
            val = np.exp(-0.5 * (dist / self.sigma) ** 2)
            val[dist > self.radius] = 0.0
            w = val.mean(axis=1)
        return w

    def footprint(self, y: float, x: float, h: int, w: int) -> Footprint:
        """Evaluate the footprint centered at sub-pixel position (y, x)."""
        cy = int(math.floor(y + 0.5))
        cx = int(math.floor(x + 0.5))
        rows = self._dy.astype(np.int64) + cy
        cols = self._dx.astype(np.int64) + cx
        prof = self._profile.copy()
        valid = prof > 0.0
        rows, cols, prof = rows[valid], cols[valid], prof[valid]
        total = prof.sum()
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        clipped = float(1.0 - prof[inside].sum() / total) if total > 0 else 0.0
        rows, cols, prof = rows[inside], cols[inside], prof[inside]
        if prof.size:
            prof = prof / prof.sum()
        return Footprint(rows, cols, prof, clipped)
