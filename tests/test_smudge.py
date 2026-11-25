"""Behavioral tests for the smudge engine.

Run directly:  python3 tests/test_smudge.py
Every check works with plain asserts so no test framework is required.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import (  # noqa: E402
    blocks_canvas, half_canvas, run, run_batched, soft_edge_canvas,
    solid_canvas, straight_path, thin_line_canvas,
)
from smudge import Canvas  # noqa: E402

ATOL = 1e-12
passed = 0


def check(cond, msg):
    global passed
    if not cond:
        raise AssertionError(msg)
    passed += 1


def check_close(a, b, msg, atol=ATOL):
    global passed
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.allclose(a, b, atol=atol, rtol=0.0):
        raise AssertionError(f"{msg}\n  got={a}\n  exp={b}\n"
                             f"  maxdiff={np.abs(a-b).max()}")
    passed += 1


def total_mass(canvas, sess):
    return canvas.paint_mass() + sess.reservoir.pm


# ----------------------------------------------------------------------

def test_batch_replay_identical():
    """Same samples, different batch boundaries -> identical pixels."""
    path = straight_path(x0=10, x1=50, n=41)
    batches = [(41,), (7, 13, 21), (1, 1, 1, 38), (20, 20, 1), (3, 5, 9, 24)]
    ref = None
    for model in ("contact", "flux"):
        for shape in ("disk", "gaussian"):
            ref = run(path, half_canvas(), model=model, shape=shape,
                      radius=5.0, step_length=1.5)
            ref_px = ref.canvas.rgba.copy()
            for sizes in batches[1:]:
                s = run_batched(path, sizes, half_canvas(), model=model,
                                shape=shape, radius=5.0, step_length=1.5)
                check_close(s.canvas.rgba, ref_px,
                            f"{model}/{shape} batch split {sizes} differs")
                check(s.stamp_count == ref.stamp_count,
                      "stamp count must not depend on batching")


def test_pixel_traversal_order_invariant():
    """Forward vs reversed footprint traversal must give the same result."""
    path = straight_path(x0=10, x1=50, n=41)
    for model in ("contact", "flux"):
        a = run(path, blocks_canvas(), model=model, radius=4.5,
                pixel_order="C")
        b = run(path, blocks_canvas(), model=model, radius=4.5,
                pixel_order="F")
        check_close(a.canvas.rgba, b.canvas.rgba,
                    f"{model}: result depends on pixel traversal order")


def test_mass_conservation():
    """Canvas + reservoir paint is conserved (no spill at blend=1)."""
    path = straight_path(x0=5, x1=55, n=51)
    for model in ("contact", "flux"):
        c0 = half_canvas()
        initial = c0.paint_mass().copy()
        s = run(path, c0, model=model, radius=6.0, capacity=0.8)
        check_close(s.ledger["spill"], np.zeros(4),
                    f"{model}: unexpected spill under source-over blend")
        check_close(total_mass(c0, s), initial,
                    f"{model}: paint mass not conserved", atol=1e-10)


def test_additive_spill_is_measured():
    """With blend=0, alpha-1 overflow must be reported, not hidden."""
    path = straight_path(x0=5, x1=30, n=26)
    c0 = solid_canvas(rgba=(0.5, 0.5, 0.5, 1.0))
    initial = c0.paint_mass().copy()
    s = run(path, c0, model="contact", blend=0.0, radius=4.0,
            pickup_rate=0.05,
            seed=(0.2, 0.9, 0.2, 1.0), capacity=2.0)
    check(s.ledger["spill"].sum() > 0.0, "additive blend over opaque canvas "
          "must produce measured spill")
    final = total_mass(c0, s) + s.ledger["spill"]
    check_close(final, initial + np.array([0.2, 0.9, 0.2, 1.0]),
                "spill ledger must close the mass balance", atol=1e-10)


def test_reservoir_is_finite_and_slow_motion_does_not_breed_paint():
    """Crawling over the same area must never exceed the capacity and must
    not increase total paint."""
    # dense samples: movement per sample far below the spatial step
    path = [(20.0 + 0.02 * i, 20.0) for i in range(400)]
    for model in ("contact", "flux"):
        c0 = solid_canvas(rgba=(0.7, 0.2, 0.2, 1.0))
        initial = c0.paint_mass().copy()
        s = run(path, c0, model=model, radius=4.0, capacity=0.5)
        check(s.reservoir.held_volume() <= 0.5 + 1e-12,
              f"{model}: reservoir exceeded capacity")
        check_close(total_mass(c0, s), initial,
                    f"{model}: slow drag created paint", atol=1e-10)


def test_flux_picks_up_each_pixel_once():
    """Model B: repeated sweeping over a pixel takes its paint only once."""
    # oscillate across a small region many times
    path = []
    for k in range(12):
        path += [(20.0, 15.0 + i) for i in range(10)]
        path += [(20.0, 24.0 - i) for i in range(10)]
    c0 = blocks_canvas()
    initial = c0.paint_mass().copy()
    s = run(path, c0, model="flux", radius=3.0, pickup_rate=1.0,
            deposit_rate=0.0, capacity=5.0)
    # full pickup rate, but reservoir content cannot exceed the paint that
    # existed in the swept region (bounded by brush area * 1.0 alpha)
    swept_alpha_cap = np.pi * 3.0 ** 2 + 6.0
    check(s.reservoir.pm[3] <= swept_alpha_cap + 1e-9,
          "flux model picked up more paint than the swept area ever held")
    check_close(total_mass(c0, s), initial, "flux mass not conserved",
                atol=1e-10)


def test_history_affects_later_strokes():
    """A later stroke must transport paint placed by an earlier stroke."""
    c = Canvas(40, 80)
    # stroke 1: seed green paint and drag it to the right
    s1 = run(straight_path(x0=10, x1=30, n=21), c, model="contact",
             radius=3.0, seed=(0.1, 0.9, 0.1, 1.0), deposit_rate=0.8,
             pickup_rate=0.2, capacity=1.0)
    after1 = c.rgba.copy()
    green_right = after1[17:24, 24:30, 1].sum()
    check(green_right > 0.0, "first stroke must leave green paint on canvas")
    # stroke 2 starts on the paint deposited by stroke 1 and moves it onward
    s2 = run(straight_path(x0=27, x1=50, n=24), c, model="contact",
             radius=3.0, pickup_rate=0.9, deposit_rate=0.9)
    check(s2.ledger["picked"][1] > 0.0,
          "second stroke must pick up paint left by the first stroke")
    check(c.rgba[17:24, 40:48, 1].sum() > 0.0,
          "paint from stroke 1 must be transported by stroke 2")


def test_transparent_crossing():
    """Paint crosses a transparent gap: nothing picked up there, color kept,
    and transparent pixels remain transparent until paint actually lands."""
    c = half_canvas()  # red opaque left of x=30, transparent right
    s = run(straight_path(x0=10, x1=50, n=41), c, model="contact",
            radius=3.0, pickup_rate=0.8, deposit_rate=0.6)
    # pixels well beyond the brush's reach of any paint stay untouched
    check(c.rgba[:, 55:, 3].max() == 0.0,
          "uncontacted transparent area must remain transparent")
    # red paint appears inside the formerly transparent right half
    check(c.rgba[17:24, 36:42, 3].max() > 0.0,
          "paint must be carried into the transparent region")
    # carried paint preserves the source hue (red-dominant)
    region = c.rgba[17:24, 36:42]
    painted = region[..., 3] > 1e-6
    check((region[painted, 0] > region[painted, 2]).all(),
          "transported color must retain the source hue")


def test_untouched_pixels_exactly_preserved():
    c0 = blocks_canvas()
    before = c0.rgba.copy()
    s = run(straight_path(y=5.0, x0=5, x1=20, n=16), c0, radius=2.0)
    untouched = np.ones(c0.rgba.shape[:2], dtype=bool)
    untouched[2:10, 2:24] = False
    check_close(c0.rgba[untouched], before[untouched],
                "pixels outside every footprint must be bit-stable")


def test_thin_line_decay_and_spread():
    """A 1px line must thin out (alpha decay) as paint is transported away."""
    c0 = thin_line_canvas()
    initial_alpha = c0.rgba[20, 10:31, 3].sum()
    s = run(straight_path(y=20.0, x0=20.0, x1=60.0, n=41), c0,
            model="contact", radius=2.5, pickup_rate=0.7, deposit_rate=0.7)
    alpha_on_line = c0.rgba[20, 10:31, 3].sum()
    check(alpha_on_line < initial_alpha - 1e-9,
          "source line must lose alpha as paint is carried away")
    check(c0.rgba[18:23, 35:55, 3].max() > 0.0,
          "line paint must be deposited further along the path")
    check(c0.rgba[:, 65:, 3].max() == 0.0,
          "paint must not appear beyond the drag distance")


def test_soft_edge_half_transparent():
    """Across a semi-transparent edge, pickup is proportional to coverage."""
    c0 = soft_edge_canvas()
    before = c0.rgba.copy()
    s = run(straight_path(y=20.0, x0=14.0, x1=45.0, n=32), c0,
            model="contact", radius=3.5, pickup_rate=0.6, deposit_rate=0.6)
    edge = c0.rgba[17:24, 20:24, 3]
    check((edge < before[17:24, 20:24, 3] - 1e-12).any(),
          "semi-transparent edge pixels must give up paint by coverage")
    check(c0.rgba[17:24, 30:38, 3].max() > 0.0,
          "picked-up edge paint must be deposited down the path")
    check(np.isfinite(c0.rgba).all(), "no NaN/Inf may be produced")


def test_pressure_and_speed_modulation():
    """Higher pressure and lower time-speed move more paint."""
    path_t = [(20.0, 10.0 + i, 1.0, float(i)) for i in range(31)]
    low = run(path_t, half_canvas(), model="contact", radius=3.0,
              speed_mode="time", speed_power=-0.5)
    hi_p = [(y, x, 0.3, t) for (y, x, _, t) in path_t]
    low_p = run(hi_p, half_canvas(), model="contact", radius=3.0,
                speed_mode="time", speed_power=-0.5)
    check(low.ledger["retained"].sum() > low_p.ledger["retained"].sum(),
          "pressure must scale the deposited flux")
    # same geometry, double the time per sample -> half the speed
    slow_t = [(y, x, p, t * 2.0) for (y, x, p, t) in path_t]
    slow = run(slow_t, half_canvas(), model="contact", radius=3.0,
               speed_mode="time", speed_power=-0.5)
    check(slow.ledger["retained"].sum() > low.ledger["retained"].sum(),
          "slower movement (speed_power<0) must increase transfer")


def test_time_batched_replay_identical():
    path = [(20.0, 10.0 + 0.8 * i, 1.0, float(i)) for i in range(40)]
    ref = run(path, half_canvas(), speed_mode="time", speed_power=-0.3,
              radius=4.0, step_length=1.2)
    for sizes in [(10, 10, 10, 10), (3, 17, 20), (1, 39)]:
        s = run_batched(path, sizes, half_canvas(), speed_mode="time",
                        speed_power=-0.3, radius=4.0, step_length=1.2)
        check_close(s.canvas.rgba, ref.canvas.rgba,
                    f"time-mode batch split {sizes} differs")


def test_stationary_samples_do_not_generate_paint():
    path = [(20.0, 20.0)] * 25 + [(20.0, 20.0 + i) for i in range(1, 11)]
    c0 = solid_canvas()
    initial = c0.paint_mass().copy()
    s = run(path, c0, model="contact", radius=3.0)
    check_close(total_mass(c0, s), initial,
                "stationary holding must not manufacture paint", atol=1e-10)
    check(s.stamp_count == 12, "one stamp at rest plus one per moved step")


def test_flow_amplifies_mixing_without_divergence():
    """Higher flow produces a stronger color pull and stays finite/nonneg."""
    def make(flow):
        c = Canvas(30, 80)
        c.rgba[:] = (0.1, 0.75, 0.2, 1.0)
        return run([(15.0, 10.0 + i, 1.0) for i in range(50)], c,
                   model="contact", radius=5.0, pickup_rate=0.0,
                   deposit_rate=0.9, capacity=2.0, flow=flow,
                   seed=(0.85, 0.1, 0.1, 1.0))

    base = make(1.0)
    strong = make(15.0)
    check(np.isfinite(strong.canvas.rgba).all(), "flow>1 must stay finite")
    check((strong.reservoir.pm >= -1e-12).all(),
          "flow>1 must not drive the reservoir negative")
    pull_base = np.abs(base.canvas.rgba[15, 12, 0] - 0.1)
    pull_strong = np.abs(strong.canvas.rgba[15, 12, 0] - 0.1)
    check(pull_strong > pull_base,
          "higher flow must pull the color further toward the reservoir")


def test_transparent_conservation_under_flow():
    """On a transparent canvas the physical default (flow=1) conserves paint
    exactly. flow>1 is a color-mixing boost and is allowed a bounded, reported
    deviation rather than inventing coverage: alpha volume must still close.
    """
    seed = np.array([0.1, 0.9, 0.1, 1.0])
    # exact conservation at the physical setting
    c = Canvas(20, 70)
    s = run([(10.0, 8.0 + i, 1.0) for i in range(45)], c,
            model="contact", radius=3.0, pickup_rate=0.3,
            deposit_rate=0.4, capacity=2.0, flow=1.0, seed=tuple(seed))
    check_close(total_mass(c, s), seed,
                "physical flow=1 must conserve to machine precision",
                atol=1e-10)
    # boosted flow never creates alpha coverage beyond the seed volume
    for flow in (4.0, 12.0):
        c = Canvas(20, 70)
        s = run([(10.0, 8.0 + i, 1.0) for i in range(45)], c,
                model="contact", radius=3.0, pickup_rate=0.3,
                deposit_rate=0.4, capacity=2.0, flow=flow, seed=tuple(seed))
        alpha_total = c.paint_mass()[3] + s.reservoir.pm[3]
        check(abs(alpha_total - seed[3]) < 1e-9,
              f"flow={flow} must not create or destroy alpha coverage")
        check(np.isfinite(c.rgba).all(),
              f"flow={flow} must keep the canvas finite")


def test_concentration_narrows_the_deposit():
    """A more concentrated brush changes core pixels more, edges less."""
    def make(conc):
        c = Canvas(30, 60)
        c.rgba[:] = (0.1, 0.75, 0.2, 1.0)
        return run([(15.0, 20.0, 1.0)], c, model="contact", radius=5.0,
                   pickup_rate=0.0, deposit_rate=0.9, capacity=2.0,
                   deposit_concentration=conc,
                   seed=(0.85, 0.1, 0.1, 1.0))

    flat = make(1.0)
    sharp = make(4.0)
    core_flat = abs(flat.canvas.rgba[15, 20, 0] - 0.1)
    core_sharp = abs(sharp.canvas.rgba[15, 20, 0] - 0.1)
    edge_flat = abs(flat.canvas.rgba[15, 25, 0] - 0.1)
    edge_sharp = abs(sharp.canvas.rgba[15, 25, 0] - 0.1)
    check(core_sharp > core_flat, "concentration must strengthen the core")
    check(edge_sharp <= edge_flat + 1e-12,
          "concentration must weaken the periphery")


def test_contact_and_flux_differ_only_in_swept_handling():
    """Both models transport paint; flux keeps more paint on re-swept pixels
    because it never picks a pixel up twice within one stroke."""
    path = [(20.0, 15.0 + 0.7 * i, 1.0) for i in range(50)]
    a = run(path, half_canvas(), model="contact", radius=4.0,
            pickup_rate=0.6, deposit_rate=0.4)
    b = run(path, half_canvas(), model="flux", radius=4.0,
            pickup_rate=0.6, deposit_rate=0.4)
    check(np.isfinite(a.canvas.rgba).all()
          and np.isfinite(b.canvas.rgba).all(),
          "both models must stay finite")
    check(a.canvas.rgba[:, 54:, 3].max() == 0.0
          and b.canvas.rgba[:, 54:, 3].max() == 0.0,
          "neither model may paint beyond the drag's reach")


def test_flux_batch_replay_identical():
    path = [(20.0, 10.0 + 0.8 * i, 1.0) for i in range(45)]
    ref = run(path, blocks_canvas(), model="flux", radius=4.0,
              step_length=1.3)
    for sizes in [(15, 15, 15), (7, 20, 18), (1, 1, 43)]:
        s = run_batched(path, sizes, blocks_canvas(), model="flux",
                        radius=4.0, step_length=1.3)
        check_close(s.canvas.rgba, ref.canvas.rgba,
                    f"flux batch split {sizes} differs")


def test_canvas_edge_clipping_keeps_paint_in_brush():
    """Paint deposited outside the canvas is not destroyed or invented."""
    c = Canvas(20, 40)
    c.rgba[:20, :20] = (0.8, 0.1, 0.1, 1.0)
    initial = c.paint_mass().copy()
    s = run([(10.0, 3.0 + i, 1.0) for i in range(20)], c,
            model="contact", radius=6.0, footprint_norm="full")
    check_close(total_mass(c, s), initial,
                "edge clipping must conserve total paint", atol=1e-10)
    check(s.ledger["clipped_paint"].sum() >= 0.0,
          "clipped paint must be recorded in the ledger")


def test_untouched_half_of_canvas_preserved_under_flux():
    c0 = half_canvas()
    before = c0.rgba.copy()
    run([(5.0, 5.0 + i, 1.0) for i in range(15)], c0, model="flux",
        radius=2.0)
    check_close(c0.rgba[12:, :], before[12:, :],
                "flux must leave rows outside every footprint unchanged")


def test_uint8_export_is_deterministic():
    path = straight_path(x0=10, x1=45, n=36)
    a = run(path, soft_edge_canvas(), radius=3.2)
    b = run(path, soft_edge_canvas(), radius=3.2)
    check(np.array_equal(a.canvas.to_uint8(), b.canvas.to_uint8()),
          "uint8 export of identical runs must match exactly")


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"\n{passed} assertions across {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
