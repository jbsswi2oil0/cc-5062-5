"""End-to-end demos producing inspectable artifacts in ``outputs/``.

For every scenario we export:
  * <name>.png        - the painted canvas as 8-bit RGBA;
  * <name>.npy        - the exact float64 pixels for off-line verification;
  * <name>.report.txt - paint ledger and conservation figures.

The final scenario replays the identical stroke split into several arbitrary
batches and writes a numerical diff proving batch-invariance.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smudge import Brush, Canvas, SmudgeConfig, SmudgeSession  # noqa: E402
from tests.helpers import (  # noqa: E402
    blocks_canvas, half_canvas, soft_edge_canvas, stripes_canvas,
    thin_line_canvas,
)

OUT = Path(__file__).resolve().parent.parent / "outputs"


def session(canvas, model, radius=4.0, **kw):
    brush = Brush(radius=radius, shape=kw.pop("shape", "disk"),
                  supersample=kw.pop("supersample", 5))
    cfg = SmudgeConfig(model=model, **kw)
    return SmudgeSession(canvas, brush, cfg)


def save(name, sess):
    sess.canvas.save_png(OUT / f"{name}.png")
    sess.canvas.save_raw(OUT / f"{name}.npy")
    bal = sess.paint_balance()
    with open(OUT / f"{name}.report.txt", "w") as f:
        f.write(f"stamps: {bal['stamps']}\n")
        f.write(f"reservoir held (premult RGBA): {bal['reservoir']}\n")
        f.write(f"picked:    {bal['picked']}\n")
        f.write(f"issued:    {bal['issued']}\n")
        f.write(f"retained:  {bal['retained']}\n")
        f.write(f"occluded:  {bal['occluded']}\n")
        f.write(f"clipped:   {bal['clipped_paint']}\n")
        f.write(f"spill:     {bal['spill']}\n")
        f.write(f"canvas mass: {bal['canvas']}\n")
    print(f"  wrote {name}.png / .npy / .report.txt")


def main():
    OUT.mkdir(exist_ok=True)
    path = [(20.0, 8.0 + 0.8 * i, 1.0) for i in range(56)]

    # 1. contact model across two adjacent color blocks -----------------
    c = blocks_canvas(40, 80)
    s = session(c, "contact", radius=4.0, pickup_rate=0.35,
                deposit_rate=0.25, capacity=1.5)
    s.run_stroke(path)
    save("01_contact_blocks", s)

    # 2. flux model, same geometry --------------------------------------
    c = blocks_canvas(40, 80)
    s = session(c, "flux", radius=4.0, pickup_rate=0.35, deposit_rate=0.25,
                capacity=1.5)
    s.run_stroke(path)
    save("02_flux_blocks", s)

    # 3. thin line transport: drag and decay along the path -------------
    c = thin_line_canvas(40, 90, x0=8, x1=25)
    line_path = [(20.0, 15.0 + 0.7 * i, 1.0) for i in range(60)]
    s = session(c, "contact", radius=2.0, pickup_rate=0.35,
                deposit_rate=0.12, capacity=1.5)
    s.run_stroke(line_path)
    save("03_thin_line", s)

    # 4. semi-transparent edge ------------------------------------------
    c = soft_edge_canvas(40, 80, edge=25, width=5)
    s = session(c, "contact", radius=3.5, shape="gaussian",
                pickup_rate=0.35, deposit_rate=0.2, capacity=1.5)
    s.run_stroke([(20.0, 18.0 + 0.8 * i) for i in range(50)])
    save("04_soft_edge", s)

    # 5a. preloaded paint dragged over a dark ground (visible color pull) ---
    c = Canvas(40, 80)
    c.rgba[:] = (0.08, 0.08, 0.12, 1.0)
    s = session(c, "contact", radius=2.5, pickup_rate=0.0,
                deposit_rate=0.3, capacity=2.0, flow=3.0)
    s.run_stroke(path, load_rgba=(0.95, 0.7, 0.1, 1.0))
    save("05a_preloaded_on_dark", s)

    # 5b. crossing a fully transparent region: pickup is impossible, so the
    # preloaded yellow keeps its exact hue while alpha decays along the path.
    c = Canvas(40, 80)
    s = session(c, "contact", radius=2.0, pickup_rate=0.0,
                deposit_rate=0.08, capacity=2.0)
    s.run_stroke(path, load_rgba=(0.95, 0.7, 0.1, 1.0))
    save("05b_transparent_crossing", s)

    # 6. pressure ramp + time-based speed -------------------------------
    c = blocks_canvas(40, 80)
    ramp = [(20.0, 8.0 + 0.8 * i, 0.2 + 0.8 * i / 55.0, float(i))
            for i in range(56)]
    s = session(c, "contact", radius=4.0, speed_mode="time",
                speed_power=-0.5, pickup_rate=0.35, deposit_rate=0.25,
                capacity=1.5)
    s.run_stroke(ramp)
    save("06_pressure_speed", s)

    # 7. history: two strokes, the second moves the first's paint -------
    c = Canvas(40, 90)
    s = session(c, "contact", radius=3.0, pickup_rate=0.2,
                deposit_rate=0.2, capacity=2.0)
    s.run_stroke([(20.0, 8.0 + i, 1.0) for i in range(28)],
                 load_rgba=(0.1, 0.9, 0.1, 1.0))
    s.run_stroke([(20.0, 22.0 + i, 1.0) for i in range(40)],
                 load_rgba=(0.1, 0.2, 0.9, 0.8))
    save("07_two_strokes_history", s)

    # 8. visible color transport over an opaque striped background --------
    # Physical flow=1 keeps every result a convex mix of real paint colors
    # (no darkening); strong pickup/deposit still gives a clear directional
    # smudge with history-driven decay.
    c = stripes_canvas()
    s = session(c, "contact", radius=4.0, pickup_rate=0.4,
                deposit_rate=0.7, capacity=4.0, flow=1.0)
    # drag red paint horizontally across the green and blue stripes
    s.run_stroke([(25.0, 18.0 + 0.8 * i, 1.0) for i in range(90)])
    save("08_stripes_transport", s)

    # 9. batch replay invariance ----------------------------------------
    samples = [(20.0, 8.0 + 0.8 * i, 1.0) for i in range(56)]
    ref = session(blocks_canvas(40, 80), "contact", radius=4.0,
                  pickup_rate=0.35, deposit_rate=0.25, capacity=1.5)
    ref.run_stroke(samples)
    splits = {
        "one_shot": [56],
        "small_batches": [3, 7, 11, 5, 13, 17],
        "singletons_tail": [20, 20] + [1] * 16,
    }
    lines = ["batch replay diff against a single-run reference\n"]
    all_match = True
    for name, sizes in splits.items():
        c2 = blocks_canvas(40, 80)
        r = session(c2, "contact", radius=4.0, pickup_rate=0.35,
                    deposit_rate=0.25, capacity=1.5)
        r.begin_stroke()
        start = 0
        for n in sizes:
            r.add_points(samples[start:start + n])
            start += n
        assert start == len(samples)
        r.end_stroke()
        diff = np.abs(c2.rgba - ref.canvas.rgba)
        match = np.allclose(c2.rgba, ref.canvas.rgba, atol=1e-12)
        all_match &= match
        lines.append(f"{name:16s} max|diff|={diff.max():.3e} "
                     f"identical={match} stamps={r.stamp_count}")
        c2.save_png(OUT / f"09_replay_{name}.png")
    ref.canvas.save_png(OUT / "09_replay_reference.png")
    (OUT / "09_replay_diff.txt").write_text("\n".join(lines) + "\n")
    print("  wrote 09_replay_* artifacts")
    print("\n".join(lines))
    if not all_match:
        sys.exit("batch replay mismatch")

    print("\nAll demo artifacts written to", OUT)


if __name__ == "__main__":
    main()
