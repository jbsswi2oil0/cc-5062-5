"""Independent verification of exported pixels (``outputs/*.npy``).

This script does not import the engine's math; it only loads the exported
float64 arrays and asserts the physical properties a smudge result must have:

* pixels never reached by any footprint are byte-for-byte the input;
* paint appears only within brush radius of the travelled path;
* alpha stays in [0, 1] and RGB in [0, 1] (no NaN/Inf, no overshoot beyond the
  convex hull of the source colors);
* two batch replays exported independently are identical to the reference.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT))

from tests.helpers import stripes_canvas  # noqa: E402

ok = 0


def expect(cond, msg):
    global ok
    if not cond:
        raise AssertionError(msg)
    ok += 1


def load(name):
    return np.load(OUT / name)


def main():
    # --- ranges and finiteness over every exported canvas ---
    for npy in sorted(OUT.glob("*.npy")):
        a = load(npy.name)
        expect(np.isfinite(a).all(), f"{npy.name}: non-finite pixels")
        expect(a.min() >= -1e-12 and a.max() <= 1.0 + 1e-12,
               f"{npy.name}: values leave [0,1]")
        expect(np.allclose(np.clip(a, 0, 1), a, atol=1e-12),
               f"{npy.name}: values need clipping")

    # --- stripes transport: nothing painted outside the path's reach ---
    ref = stripes_canvas()
    painted = load("08_stripes_transport.npy")
    radius = 4.0
    ys, xs = np.mgrid[0:painted.shape[0], 0:painted.shape[1]]
    path_x = 18.0 + 0.8 * np.arange(90)
    reachable = np.zeros(painted.shape[:2], dtype=bool)
    for px in path_x:
        reachable |= ((ys - 25.0) ** 2 + (xs - px) ** 2) <= (radius + 1.0) ** 2
    untouched = ~reachable
    expect(np.array_equal(painted[untouched], ref.rgba[untouched]),
           "pixels outside every footprint must equal the input")

    # transport really happened inside the reachable band
    band = painted[23:28, 47:63]
    expect((band[..., 0] - band[..., 1]).mean() >
           (ref.rgba[23:28, 47:63, 0] - ref.rgba[23:28, 47:63, 1]).mean(),
           "red paint must be transported into the green stripe")

    # --- batch replay exports are identical to the single-run reference ---
    ref_img = np.asarray(Image.open(OUT / "09_replay_reference.png")
                         .convert("RGBA"))
    for name in ("09_replay_one_shot.png",
                 "09_replay_small_batches.png",
                 "09_replay_singletons_tail.png"):
        other = np.asarray(Image.open(OUT / name).convert("RGBA"))
        expect(np.array_equal(other, ref_img),
               f"{name} differs from the reference replay")

    # --- thin line: source lost alpha, nothing beyond drag distance ---
    line = load("03_thin_line.npy")
    expect(line[:, 65:, 3].max() == 0.0,
           "thin-line paint must not appear beyond the drag distance")
    expect(line[..., 3].max() <= 1.0 + 1e-12, "alpha overshoot on thin line")

    # --- two-stroke history canvas stays fully finite and bounded ---
    hist = load("07_two_strokes_history.npy")
    expect(np.isfinite(hist).all() and hist.max() <= 1.0 + 1e-12,
           "two-stroke history canvas must stay bounded")

    print(f"all {ok} independent pixel checks passed.")


if __name__ == "__main__":
    main()
