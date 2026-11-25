"""Deterministic digital-painting smudge engine.

Paint transport model (premultiplied RGBA, all arithmetic float64):

* the *canvas* stores straight-alpha color; the *brush reservoir* is a separate
  finite-volume store of premultiplied paint (rgb premult + total alpha volume);
* at every fixed spatial step along the pointer path the brush first *picks up*
  paint from the canvas it touches, then *deposits* from the reservoir;
* pickup always precedes deposit and both operate on snapshots taken before any
  write in the step, so the result cannot depend on pixel traversal order;
* two settlement models are provided:
  - "contact": per-contact coverage settlement, the whole footprint is settled
    every stamp (fractions make slow strokes not create paint);
  - "flux":    integrated pickup/deposition flux over the swept path, each canvas
    point is only picked up the first time the brush sweeps over it.
"""

from .brush import Brush
from .canvas import Canvas
from .session import SmudgeConfig, SmudgeSession
from .model import step_contact, step_flux

__all__ = [
    "Brush",
    "Canvas",
    "SmudgeConfig",
    "SmudgeSession",
    "step_contact",
    "step_flux",
]
