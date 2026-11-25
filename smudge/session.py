"""Smudge session: path tessellation, finite reservoir, bookkeeping.

Batch-invariant tessellation
----------------------------
Pointer samples arrive in arbitrary batch sizes. The session keeps only two
pieces of carry state between batches:

* the anchor point (last raw sample);
* ``need``: the distance still to travel from that anchor before the next
  spatial stamp.

Stamps are therefore placed on a fixed-length arc-length lattice that depends
solely on the geometry of the concatenated path, never on where one batch ended
and the next began. Re-feeding the same samples split differently produces the
exact same stamp positions, pressures and speed factors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .brush import Brush
from .canvas import Canvas
from .model import LEDGER_KEYS, step_contact, step_flux


@dataclass
class SmudgeConfig:
    model: str = "contact"            # "contact" (A) or "flux" (B)
    pickup_rate: float = 0.5          # fraction of touched paint picked up
    deposit_rate: float = 0.5         # fraction of reservoir issued per stamp
    capacity: float = 1.0             # max alpha-volume the brush can carry
    flow: float = 1.0                 # color-mix strength multiplier; 1.0 is
                                      # the strictly mass-conserving physical
                                      # default, >1 boosts visible mixing
    deposit_concentration: float = 1.0  # reshape deposit strength toward the
                                         # brush core; 1 = flat weight shape,
                                         # larger = stronger, narrower core
    footprint_norm: str = "full"      # "full" (divide by whole-disk profile)
                                      # or "effective" (divide by in-canvas,
                                      # above-threshold pixel count)
    blend: float = 1.0                # 1 = convex color-replacement smudge,
                                      # 0 = additive premultiplied compositing
    step_length: float = 1.0          # spatial sampling distance in pixels
    speed_mode: str = "geometry"      # "geometry" (factor=1) or "time"
    speed_power: float = 0.0          # exponent applied to speed factor
    reference_dt: float = 1.0         # nominal sample interval for time mode
    min_speed_factor: float = 0.05
    max_speed_factor: float = 20.0

    def __post_init__(self):
        if self.model not in ("contact", "flux"):
            raise ValueError("model must be 'contact' or 'flux'")
        for name in ("pickup_rate", "deposit_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("blend must lie in [0, 1]")
        if self.flow <= 0.0:
            raise ValueError("flow must be positive")
        if self.deposit_concentration <= 0.0:
            raise ValueError("deposit_concentration must be positive")
        if self.footprint_norm not in ("full", "effective"):
            raise ValueError("footprint_norm must be 'full' or 'effective'")
        if self.step_length <= 0:
            raise ValueError("step_length must be positive")


class Reservoir:
    """Finite premultiplied paint store carried by the brush."""

    def __init__(self, capacity: float):
        self.capacity = float(capacity)
        self.pm = np.zeros(4, dtype=np.float64)

    def load_premultiplied(self, rgba_pm) -> None:
        pm = np.asarray(rgba_pm, dtype=np.float64).copy()
        if pm[3] > self.capacity:
            scale = self.capacity / pm[3]
            pm *= scale
        self.pm = pm

    def load_straight(self, rgba) -> None:
        r, g, b, a = rgba
        self.load_premultiplied((r * a, g * a, b * a, a))

    def held_volume(self) -> float:
        return float(self.pm[3])


class SmudgeSession:
    def __init__(self, canvas: Canvas, brush: Brush, config: SmudgeConfig,
                 pixel_order: str = "C"):
        if pixel_order not in ("C", "F"):
            raise ValueError("pixel_order must be 'C' (normal) or 'F' (flip)")
        self.canvas = canvas
        self.brush = brush
        self.cfg = config
        self.pixel_order = pixel_order
        self.reservoir = Reservoir(config.capacity)
        self.coverage = np.zeros((canvas.height, canvas.width),
                                 dtype=np.float64)
        self.ledger = {k: np.zeros(4, dtype=np.float64) for k in LEDGER_KEYS}
        self.stamp_count = 0
        self._active = False
        self._anchor = None          # (y, x, p, t)
        self._need = 0.0

    # ------------------------------------------------------------------

    def begin_stroke(self, load_rgba=None) -> None:
        """Start a contact; optionally preload the brush with paint."""
        self._active = True
        self.reservoir = Reservoir(self.cfg.capacity)
        if load_rgba is not None:
            self.reservoir.load_straight(load_rgba)
        self.coverage.fill(0.0)
        self._anchor = None
        self._need = 0.0

    def end_stroke(self) -> None:
        self._active = False
        self._anchor = None

    def add_points(self, points) -> None:
        """Feed one or more pointer samples.

        Each sample is ``(y, x)``, ``(y, x, pressure)`` or
        ``(y, x, pressure, t)``. May be called repeatedly with arbitrary batch
        sizes; the tessellation carry makes the split irrelevant.
        """
        if not self._active:
            raise RuntimeError("call begin_stroke() before add_points()")
        for raw in points:
            pt = self._normalize(raw)
            if self._anchor is None:
                self._anchor = pt
                self._need = 0.0
                self._emit(pt[0], pt[1], pt[2], 1.0)
                continue
            self._consume_segment(pt)
            self._anchor = pt

    def run_stroke(self, points, load_rgba=None) -> None:
        self.begin_stroke(load_rgba=load_rgba)
        self.add_points(points)
        self.end_stroke()

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(raw) -> tuple:
        raw = tuple(float(v) for v in raw)
        if len(raw) == 2:
            return (raw[0], raw[1], 1.0, 0.0)
        if len(raw) == 3:
            return (raw[0], raw[1], raw[2], 0.0)
        if len(raw) == 4:
            return raw
        raise ValueError("samples must have 2, 3 or 4 components")

    def _consume_segment(self, end) -> None:
        ay, ax, ap, at = self._anchor
        ey, ex, ep, et = end
        dx, dy = ex - ax, ey - ay
        length = math.hypot(dx, dy)
        if length < 1e-12:
            # stationary: no new stamp; waiting must not generate paint
            self._need = 0.0 if self._need < 1e-12 else self._need
            return
        speed_factor = self._speed_factor(length, et - at)
        traveled = 0.0
        while self._need <= length - traveled:
            traveled += self._need
            f = traveled / length
            y = ay + dy * f
            x = ax + dx * f
            p = ap + (ep - ap) * f
            self._emit(y, x, p, speed_factor)
            self._need = self.cfg.step_length
        self._need -= (length - traveled)

    def _speed_factor(self, length: float, dt: float) -> float:
        if self.cfg.speed_mode != "time" or dt <= 1e-12:
            return 1.0
        nominal = length / self.cfg.step_length
        nominal_time = nominal * self.cfg.reference_dt
        factor = nominal_time / dt
        return float(np.clip(factor, self.cfg.min_speed_factor,
                             self.cfg.max_speed_factor))

    def _emit(self, y, x, pressure, speed_factor) -> None:
        cfg = self.cfg
        if cfg.model == "contact":
            delta = step_contact(
                self.canvas, self.brush, y, x, pressure, speed_factor,
                cfg, self.reservoir, self.pixel_order)
        else:
            delta = step_flux(
                self.canvas, self.brush, y, x, pressure, speed_factor,
                cfg, self.reservoir, self.coverage, self.pixel_order)
        for k in LEDGER_KEYS:
            self.ledger[k] += delta[k]
        self.stamp_count += 1

    # ------------------------------------------------------------------

    def paint_balance(self) -> dict:
        """Mass-conservation report over the session's history."""
        canvas_mass = self.canvas.paint_mass()
        held = self.reservoir.pm
        return {
            "canvas": canvas_mass,
            "reservoir": held,
            "picked": self.ledger["picked"],
            "issued": self.ledger["issued"],
            "retained": self.ledger["retained"],
            "occluded": self.ledger["occluded"],
            "clipped_paint": self.ledger["clipped_paint"],
            "spill": self.ledger["spill"],
            "stamps": self.stamp_count,
        }
