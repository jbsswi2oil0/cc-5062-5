"""The two settlement models.

Both models share the same within-step protocol, which makes every step
independent of pixel traversal order:

1. evaluate the brush footprint;
2. take a snapshot of every canvas pixel it touches (all reads happen here);
3. compute pickup from the snapshot and compute deposit from the reservoir
   snapshot (neither mutates anything yet);
4. perform a single vectorized write of the new pixel values, then update the
   reservoir.

Paint quantities are premultiplied RGBA vectors. The fourth component is the
alpha *volume* of paint; the reservoir capacity limits that volume, so a brush
can never hold more than ``capacity`` paint and slow strokes cannot breed
paint.

Mass accounting per step (per premultiplied channel):

* at the physical default ``flow = 1``, canvas paint + reservoir paint is
  conserved exactly (the only loss is alpha-1 "spill", possible solely under
  additive ``blend = 0``);
* ``flow > 1`` is a color-mixing *strength* boost. Alpha coverage is still
  conserved exactly, while RGB can deviate by a bounded amount because the
  boost re-mixes existing canvas color more strongly than the physical paint
  moved. Repeated re-pickup of self-deposited color at high flow is what makes
  a real smudge progressively muddy; keep flow at 1 for strict conservation.

Deposit hidden under opaque pixels ("occluded") and deposit falling outside
the canvas edge ("clipped") never leaves the brush and stays in the reservoir;
every term is measured in the ledger, never silently created or dropped.
"""

from __future__ import annotations

import numpy as np

from .brush import Footprint

LEDGER_KEYS = ("picked", "issued", "retained", "occluded",
               "clipped_paint", "spill")


def _empty_delta() -> dict:
    return {
        name: np.zeros(4, dtype=np.float64) for name in LEDGER_KEYS
    } | {"clipped_fraction": 0.0}


def _rate(base: float, pressure: float, speed_factor: float,
          speed_power: float) -> float:
    """Pressure/velocity modulated transfer fraction, clipped to [0, 1]."""
    if speed_factor <= 0.0:
        speed_factor = 1e-9
    return float(np.clip(base * pressure * (speed_factor ** speed_power),
                         0.0, 1.0))


def _issue(reservoir_pm: np.ndarray, d_rate: float, flow: float) -> np.ndarray:
    """Amount drawn from the reservoir; never more than it actually holds."""
    return np.minimum(reservoir_pm * d_rate * flow, reservoir_pm)


def _order_indices(n: int, pixel_order: str) -> np.ndarray:
    idx = np.arange(n)
    if pixel_order == "F":          # reverse traversal through the footprint
        return idx[::-1]
    return idx


def _gather(canvas, fp: Footprint, pixel_order: str):
    idx = _order_indices(fp.weights.size, pixel_order)
    rows = fp.rows[idx]
    cols = fp.cols[idx]
    w = fp.weights[idx]
    pm = canvas.premultiplied()[rows, cols].copy()     # snapshot, all reads
    return rows, cols, w, pm


def _accept_pickup(request: np.ndarray, capacity: float,
                   held: float) -> float:
    """Fraction of a pickup request that fits into the finite reservoir."""
    if request[3] <= 1e-15:
        return 0.0
    room = capacity - held
    if room <= 0.0:
        return 0.0
    return float(min(1.0, room / request[3]))


def _deposit(pm_base, w, issued, reservoir_pm, blend, clipped_fraction,
             norm="full", concentration=1.0, flow=1.0):
    """Deposit reservoir paint onto footprint pixels.

    The issued premultiplied batch is split across pixels by the footprint
    ``shape`` (weights normalized to sum 1). Each pixel receives a real
    premultiplied paint amount ``m_k``.

    Two strengths are kept separate:

        t_alpha = clamp(m_k.alpha, 0, 1)       physical coverage deposit
        t_mix   = t_alpha + (clamp(t_alpha*flow,0,1) - t_alpha) * a_base

    ``t_alpha`` grows the pixel alpha and is never amplified by flow, so the
    brush cannot invent coverage. ``t_mix`` drives the straight-color blend
    toward the reservoir color; flow only strengthens it where the pixel
    already carries color (a_base > 0), falling back to t_alpha on transparent
    pixels. The result straight color is a convex combination of base and
    reservoir colors, so neither can be pushed darker/brighter than the
    contributing paints.

    Mass bookkeeping: the brush is debited by the paint it actually supplies
    (``debit``). At the default ``flow = 1`` the debit equals the canvas's
    non-negative premultiplied gain, so canvas + reservoir mass is conserved
    exactly. With ``flow > 1`` the visible mixing is stronger than the
    physical paint moved; alpha coverage still closes exactly while RGB may
    differ by a bounded, re-mixing amount. ``blend`` interpolates toward plain
    additive premultiplied compositing (whose alpha-1 overflow is spill).
    """
    wsum = w.sum()
    if norm == "effective":
        shape = w / wsum if wsum > 0.0 else w
    else:
        shape = w
    if shape.size and shape.max() > 0.0 and concentration != 1.0:
        sharp = (shape / shape.max()) ** concentration
        shape = sharp / sharp.sum()
    inside = 1.0 - clipped_fraction

    held_a = reservoir_pm[3]
    res_rgb = (reservoir_pm[:3] / held_a) if held_a > 1e-12 \
        else np.zeros(3)
    base_a = pm_base[:, 3:4]

    # real premultiplied paint delivered to each pixel
    m = issued[None, :] * shape[:, None] * inside
    # alpha deposition is physical paint and is never amplified (otherwise
    # coverage could be created out of nothing). Color mixing may be boosted
    # by flow, but only where the base already carries color; on a transparent
    # pixel the mix strength falls back to the physical alpha strength.
    t_alpha = np.clip(m[:, 3:4], 0.0, 1.0)
    t_boost = np.clip(t_alpha * flow, 0.0, 1.0)
    t_mix = t_alpha + (t_boost - t_alpha) * base_a

    # smudge branch. Alpha grows only by the physical deposit t_alpha, so
    # coverage is never invented. Color is a straight-alpha blend at strength
    # t_mix: the result color interpolates from the base straight color toward
    # the reservoir color, then is re-premultiplied by the result alpha. On an
    # opaque pixel this is a plain color mix (no darkening); on a transparent
    # pixel t_mix == t_alpha and it lays down exactly res_rgb * new_alpha.
    smudge_a = base_a + (1.0 - base_a) * t_alpha
    base_straight = pm_base[:, :3] / np.maximum(base_a, 1e-12)
    blended_straight = base_straight * (1.0 - t_mix) \
        + res_rgb[None, :] * t_mix
    smudge_rgb = blended_straight * smudge_a
    smudge_rgb = np.where(base_a > 1e-12, smudge_rgb,
                          res_rgb[None, :] * smudge_a)

    # additive branch (premultiplied sum; may exceed alpha 1 -> spill)
    add_pm = pm_base + m

    new_pm = np.empty_like(pm_base)
    new_pm[:, :3] = (1.0 - blend) * add_pm[:, :3] + blend * smudge_rgb
    new_pm[:, 3:4] = (1.0 - blend) * add_pm[:, 3:4] + blend * smudge_a

    # paint physically leaving the brush.
    # RGB leaving is the reservoir color times the deposited alpha volume; the
    # alpha leaving is the canvas's real alpha gain (1 - base_a) * t_alpha.
    # The flow-driven color_pull only reshuffles pre-existing canvas color and
    # is never taken from the brush.
    alpha_gain = ((1.0 - base_a) * t_alpha).sum()
    physical = np.empty(4)
    physical[:3] = (res_rgb[None, :] * t_alpha).sum(axis=0)
    physical[3] = alpha_gain
    physical_deposit = physical * blend \
        + m.reshape(-1, 4).sum(axis=0) * (1.0 - blend)

    retained = (new_pm - pm_base).sum(axis=0)
    if abs(flow - 1.0) < 1e-15 and blend > 0.0:
        # At flow 1 the smudge rule only redistributes premultiplied mass;
        # negative deltas are internally displaced color, never returned to
        # the brush, so the debit is the non-negative canvas gain.
        debit = np.maximum(retained, 0.0)
    else:
        debit = physical_deposit

    clipped_paint = issued * clipped_fraction
    occluded = np.maximum(issued * inside - debit, 0.0)
    return new_pm, retained, occluded, clipped_paint, debit


# --------------------------------------------------------------------------
# Model A: per-contact coverage settlement
# --------------------------------------------------------------------------

def step_contact(canvas, brush, y, x, pressure, speed_factor, cfg,
                 reservoir, pixel_order="C") -> dict:
    fp = brush.footprint(y, x, canvas.height, canvas.width)
    delta = _empty_delta()
    delta["clipped_fraction"] = fp.clipped_fraction
    if fp.weights.size == 0:
        return delta

    rows, cols, w, pm_snapshot = _gather(canvas, fp, pixel_order)

    # --- pickup: each contact pixel gives its own coverage-weighted share ---
    p_rate = _rate(cfg.pickup_rate, pressure, speed_factor, cfg.speed_power)
    per_pixel = w[:, None] * pm_snapshot
    request = per_pixel.sum(axis=0) * p_rate
    accept = _accept_pickup(request, cfg.capacity, reservoir.pm[3])
    picked = request * accept
    pm_after_pickup = pm_snapshot - per_pixel * (p_rate * accept)

    # --- deposit: finite amount drawn from the reservoir snapshot ---
    d_rate = _rate(cfg.deposit_rate, pressure, speed_factor, cfg.speed_power)
    issued = _issue(reservoir.pm, d_rate, cfg.flow)
    new_pm, retained, occluded, clipped_paint, debit = _deposit(
        pm_after_pickup, w, issued, reservoir.pm, cfg.blend,
        fp.clipped_fraction, cfg.footprint_norm,
        cfg.deposit_concentration, cfg.flow)
    spill = canvas.write_premultiplied(rows, cols, new_pm).sum(axis=0)

    # --- reservoir update, after every canvas read is finished ---
    # Debit exactly the paint that mixed into the canvas. Displaced base color
    # is not the brush's paint and is therefore never subtracted here.
    reservoir.pm += picked - debit
    delta.update(picked=picked, issued=issued, retained=retained,
                 occluded=occluded, clipped_paint=clipped_paint, spill=spill)
    return delta


# --------------------------------------------------------------------------
# Model B: integrated pickup/deposition flux over the swept path
# --------------------------------------------------------------------------

def step_flux(canvas, brush, y, x, pressure, speed_factor, cfg,
              reservoir, coverage, pixel_order="C") -> dict:
    fp = brush.footprint(y, x, canvas.height, canvas.width)
    delta = _empty_delta()
    delta["clipped_fraction"] = fp.clipped_fraction
    if fp.weights.size == 0:
        return delta

    rows, cols, w, pm_snapshot = _gather(canvas, fp, pixel_order)

    # pickup only from area not swept by an earlier stamp of this stroke
    already = coverage[rows, cols].copy()
    new_w = np.maximum(w - already, 0.0)
    np.maximum(coverage[rows, cols], w, out=coverage[rows, cols])

    p_rate = _rate(cfg.pickup_rate, pressure, speed_factor, cfg.speed_power)
    per_pixel = new_w[:, None] * pm_snapshot
    request = per_pixel.sum(axis=0) * p_rate
    accept = _accept_pickup(request, cfg.capacity, reservoir.pm[3])
    picked = request * accept
    pm_after_pickup = pm_snapshot - per_pixel * (p_rate * accept)

    d_rate = _rate(cfg.deposit_rate, pressure, speed_factor, cfg.speed_power)
    issued = _issue(reservoir.pm, d_rate, cfg.flow)
    new_pm, retained, occluded, clipped_paint, debit = _deposit(
        pm_after_pickup, w, issued, reservoir.pm, cfg.blend,
        fp.clipped_fraction, cfg.footprint_norm,
        cfg.deposit_concentration, cfg.flow)
    spill = canvas.write_premultiplied(rows, cols, new_pm).sum(axis=0)

    reservoir.pm += picked - debit
    delta.update(picked=picked, issued=issued, retained=retained,
                 occluded=occluded, clipped_paint=clipped_paint, spill=spill)
    return delta
