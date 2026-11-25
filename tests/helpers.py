"""Test helpers: deterministic scene builders and run/replay utilities."""

import numpy as np

from smudge import Brush, Canvas, SmudgeConfig, SmudgeSession


def solid_canvas(h=40, w=60, rgba=(0.8, 0.1, 0.1, 1.0)):
    c = Canvas(h, w)
    c.rgba[:] = rgba
    return c


def half_canvas(h=40, w=60):
    """Left half red opaque, right half fully transparent."""
    c = Canvas(h, w)
    c.rgba[:, : w // 2] = (0.8, 0.1, 0.1, 1.0)
    return c


def blocks_canvas(h=40, w=60):
    """Two adjacent opaque color blocks sharing a vertical edge."""
    c = Canvas(h, w)
    c.rgba[:, : w // 2] = (0.9, 0.1, 0.1, 1.0)
    c.rgba[:, w // 2:] = (0.1, 0.1, 0.9, 1.0)
    return c


def stripes_canvas(h=50, w=120):
    """Opaque background with vertical stripes of strongly different hues."""
    c = Canvas(h, w)
    c.rgba[:] = (0.93, 0.93, 0.93, 1.0)          # light gray ground
    c.rgba[10:40, 10:30] = (0.85, 0.1, 0.1, 1.0)   # red block
    c.rgba[10:40, 45:65] = (0.1, 0.75, 0.2, 1.0)   # green block
    c.rgba[10:40, 80:100] = (0.1, 0.2, 0.85, 1.0)  # blue block
    return c


def thin_line_canvas(h=40, w=80, row=20, x0=10, x1=30):
    """A 1px fully opaque green line on a transparent canvas."""
    c = Canvas(h, w)
    c.rgba[row, x0:x1 + 1] = (0.1, 0.8, 0.1, 1.0)
    return c


def soft_edge_canvas(h=40, w=60, edge=20, width=4):
    """Opaque red block fading to transparent over a 4px alpha ramp."""
    c = Canvas(h, w)
    c.rgba[:, :edge] = (0.8, 0.1, 0.1, 1.0)
    for k in range(width):
        a = 1.0 - (k + 1) / (width + 1)
        c.rgba[:, edge + k] = (0.8, 0.1, 0.1, a)
    return c


def make_session(canvas, model="contact", radius=4.0, **kw):
    shape = kw.pop("shape", "disk")
    supersample = kw.pop("supersample", 4)
    pixel_order = kw.pop("pixel_order", "C")
    brush = Brush(radius=radius, shape=shape, supersample=supersample)
    cfg = SmudgeConfig(model=model, **kw)
    return SmudgeSession(canvas, brush, cfg, pixel_order=pixel_order)


def run(points, canvas, seed=None, **kw):
    sess = make_session(canvas, **kw)
    sess.run_stroke(points, load_rgba=seed)
    return sess


def run_batched(points, batch_sizes, canvas, seed=None, **kw):
    """Re-feed the same samples split into batches of given sizes."""
    sess = make_session(canvas, **kw)
    sess.begin_stroke(load_rgba=seed)
    start = 0
    for n in batch_sizes:
        sess.add_points(points[start:start + n])
        start += n
    assert start == len(points)
    sess.end_stroke()
    return sess


def straight_path(y=20.0, x0=8.0, x1=52.0, n=None, spacing=1.0, pressure=1.0):
    if n is None:
        n = int(round((x1 - x0) / spacing)) + 1
    xs = np.linspace(x0, x1, n)
    return [(y, float(x), pressure) for x in xs]
