"""Acceptance tests for legacy tip-color brush migration.

Run directly:  python3 tests/test_migration.py
The functions are also pytest-compatible.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smudge import (  # noqa: E402
    Brush,
    BrushWorkbench,
    Canvas,
    MigrationStatus,
    SmudgeConfig,
    SmudgeSession,
    compare_replays,
    edge_diff,
    infer_capacity_bounds,
    migrate_legacy_brush,
    restore_nib_by_capacity,
)
from smudge.migration import source_fingerprint  # noqa: E402

passed = 0


def check(cond, msg):
    global passed
    if not cond:
        raise AssertionError(msg)
    passed += 1


def check_close(a, b, msg, atol=1e-10):
    global passed
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.allclose(a, b, atol=atol, rtol=0.0):
        raise AssertionError(f"{msg}\n  got={a}\n  exp={b}\n"
                             f"  maxdiff={np.abs(a-b).max()}")
    passed += 1


# ---------------------------------------------------------------------------

TIP = (0.8, 0.1, 0.1, 1.0)


def rgb_crossing_canvas():
    c = Canvas(40, 90)
    c.rgba[:, :30] = (0.8, 0.1, 0.1, 1.0)
    c.rgba[:, 60:] = (0.1, 0.1, 0.9, 1.0)
    return c


CROSS_PATH = [(20.0, float(x), 1.0) for x in range(10, 81)]
SPLIT = 41
CROSS_CFG = dict(model="contact", pickup_rate=0.9, deposit_rate=0.4,
                 step_length=1.0)
EDGE_SLICES = {
    "red_to_transparent_edge": (slice(16, 25), slice(27, 34)),
    "transparent_gap": (slice(16, 25), slice(38, 57)),
    "transparent_to_blue_edge": (slice(16, 25), slice(57, 65)),
    "blue_interior": (slice(16, 25), slice(66, 82)),
}


def original_crossing_session(capacity=1.0):
    sess = SmudgeSession(
        rgb_crossing_canvas(),
        Brush(radius=3.0, supersample=4),
        SmudgeConfig(capacity=capacity, **CROSS_CFG),
        record_steps=True,
    )
    sess.run_stroke(CROSS_PATH)
    return sess


# ---------------------------------------------------------------------------

def test_explicit_zero_and_never_stored_capacity_remain_distinct():
    zero_raw = {
        "id": "same-brush",
        "format": "legacy_tip_color",
        "tip_color": list(TIP),
        "capacity": 0.0,
    }
    missing_raw = {
        "id": "same-brush",
        "format": "legacy_tip_color",
        "tip_color": list(TIP),
    }

    zero = migrate_legacy_brush(zero_raw)
    missing = migrate_legacy_brush(missing_raw)

    check(zero.decisions["capacity"].status == MigrationStatus.CONFIRMED.value,
          "explicit zero capacity must be confirmed, not pending")
    check(zero.decisions["capacity"].value == 0.0,
          "explicit zero capacity must survive conversion")
    check(zero.config is not None and zero.config.capacity == 0.0,
          "zero capacity must produce a real zero-capacity config")
    check(not zero.decisions["capacity"].is_pending,
          "confirmed zero capacity must not be marked pending")
    check_close(zero.initial_pm, np.zeros(4),
                "zero capacity cannot restore any nib paint")
    check(restore_nib_by_capacity(TIP, 0.0).held_volume() == 0.0,
          "reference restoration at zero capacity must be empty")
    full = restore_nib_by_capacity(TIP, 1.5)
    check_close(full.pm, (1.2, 0.15, 0.15, 1.5),
                "capacity restoration must scale the nib hue to full capacity")
    stored_alpha = restore_nib_by_capacity(
        (0.8, 0.1, 0.1, 0.4), 2.0, alpha_mode="stored_alpha")
    check_close(stored_alpha.held_volume(), 0.4,
                "stored-alpha restoration must preserve the nib alpha volume")

    cap = missing.decisions["capacity"]
    check(cap.status == MigrationStatus.PENDING.value,
          "absent capacity key must remain pending")
    check(cap.value is None, "missing capacity must not become an implicit zero")
    check("capacity" in missing.unresolved_fields,
          "missing capacity must appear in unresolved fields")
    check(missing.config is None,
          "a pending capacity without provisional value cannot become a config")

    provisional = migrate_legacy_brush(
        missing_raw, provisional_capacity=1.25,
        config_overrides={"pickup_rate": 0.7})
    check(provisional.config.capacity == 1.25,
          "provisional replay capacity must be usable")
    check(provisional.decisions["capacity"].is_pending,
          "provisional capacity must still be marked unconfirmed")
    check(provisional.unresolved_fields.count("capacity") == 1,
          "provisional value must not clear the pending decision")


def test_project_files_distinguish_zero_and_missing_for_same_brush(tmp_path=None):
    if tmp_path is None:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
    else:
        tmp = None
        root = Path(tmp_path)
    try:
        zero_path = root / "zero_project.json"
        missing_path = root / "missing_project.json"
        zero_path.write_text(json.dumps({
            "format": "smudge-project",
            "active_brush_id": "same",
            "brushes": {"same": {"tip_color": list(TIP), "capacity": 0}},
        }), encoding="utf-8")
        missing_path.write_text(json.dumps({
            "format": "smudge-project",
            "active_brush_id": "same",
            "brushes": {"same": {"tip_color": list(TIP)}},
        }), encoding="utf-8")

        bench = BrushWorkbench()
        zero_project = bench.open_project(zero_path)
        missing_project = bench.open_project(missing_path)
        z = zero_project.brushes["same"].decisions["capacity"]
        m = missing_project.brushes["same"].decisions["capacity"]
        check(z.status == "confirmed" and z.value == 0.0,
              "project zero capacity must remain confirmed zero")
        check(m.status == "pending" and m.value is None,
              "project without capacity key must remain pending missing")
        check(zero_project.source_unchanged and missing_project.source_unchanged,
              "opening/migrating projects must not alter source bytes")
    finally:
        if tmp is not None:
            tmp.cleanup()


def test_humidity_and_stroke_origin_are_explained():
    with_humidity = migrate_legacy_brush({
        "tip_color": list(TIP), "humidity": 0.65,
    }, provisional_capacity=1.0)
    check(with_humidity.decisions["humidity"].value == 0.65
          and with_humidity.decisions["humidity"].status == "confirmed",
          "stored humidity must be read as confirmed")
    check(with_humidity.config.humidity == 0.65,
          "confirmed humidity must be carried by the new recipe config")

    legacy = {"tip_color": list(TIP), "wetness": "not-a-number"}
    bad_humidity = migrate_legacy_brush(legacy)
    check(bad_humidity.decisions["humidity"].is_pending,
          "invalid humidity must stay pending rather than crash or become zero")

    original = original_crossing_session()
    migrated = migrate_legacy_brush(
        {"tip_color": list(TIP)}, history=original.step_log)
    humidity = migrated.decisions["humidity"]
    check(humidity.value is None and humidity.is_pending,
          "humidity absent from a tip-only archive must remain pending")
    check(humidity.provisional_value is None,
          "no humidity default should be invented unless caller supplies one")

    origin = migrated.record.origin
    check(origin is not None and origin.status == MigrationStatus.MARKED.value,
          "first replayable historical event must be marked as an origin")
    check(origin.historical_start_known is False,
          "fallback origin must not claim to be the true historical start")
    check(origin.x == 10.0 and origin.y == 20.0,
          "origin must carry the first event coordinates")
    start_decision = migrated.decisions["stroke_start"]
    check(start_decision.status == MigrationStatus.MARKED.value
          and start_decision.is_pending,
          "stroke_start field must be a marked but unconfirmed decision")
    check(start_decision.value == {"y": 20.0, "x": 10.0,
                                   "stroke_id": 1, "point_index": 0},
          "stroke_start decision must contain the marked coordinates")
    check("earlier pickup" in origin.reason,
          "origin record must explain why the start is only a marker")

    explicit_start = migrate_legacy_brush({
        "tip_color": list(TIP), "stroke_start": [12.0, 34.0],
    })
    explicit_decision = explicit_start.decisions["stroke_start"]
    check(explicit_decision.status == "confirmed"
          and not explicit_decision.is_pending,
          "an explicitly stored stroke start should remain confirmed")
    check(explicit_decision.value["x"] == 34.0
          and explicit_decision.value["y"] == 12.0,
          "explicit start coordinates should be interpreted as [y, x]")


def test_capacity_backcalculation_scope():
    # Synthetic net flow: -2, +3, -1 gives prefixes 0,-2,1,0.
    events = [
        {"picked": [0, 0, 0, 0], "debit": [0, 0, 0, 2]},
        {"picked": [0, 0, 0, 3], "debit": [0, 0, 0, 0]},
        {"picked": [0, 0, 0, 0], "debit": [0, 0, 0, 1]},
    ]
    bound = infer_capacity_bounds(events)
    check_close(bound.minimum_initial_load, 2.0,
                "minimum initial load must cover the early debit")
    check_close(bound.minimum_capacity, 3.0,
                "capacity bound is the prefix range when initial load is free")
    fixed = infer_capacity_bounds(events, initial_volume=2.0)
    check_close(found := fixed.fixed_initial_capacity, 3.0,
                "fixed initial load must give max(h0 + cumulative net)")

    saturated = original_crossing_session(capacity=1.0)
    exact = infer_capacity_bounds(saturated.step_log)
    check(exact.saturated and exact.exact_capacity is not None,
          "recorded rejected pickup plus held-before must identify saturation")
    check(abs(exact.exact_capacity - 1.0) < 1e-8,
          "saturation events must recover the exact capacity")
    check(exact.applicability == "exact_at_recorded_saturation",
          "saturated history should have exact applicability")

    large = original_crossing_session(capacity=100.0)
    lower = infer_capacity_bounds(large.step_log)
    check(lower.exact_capacity is None,
          "unsaturated history must not claim an exact capacity")
    check(lower.minimum_capacity <= 100.0 + 1e-8,
          "back-calculated minimum must not exceed the known reference capacity")
    check(lower.applicability == "lower_bound_only",
          "unsaturated history must be reported as lower-bound-only")

    migrated = migrate_legacy_brush(
        {"tip_color": list(TIP)}, history=large.step_log)
    cap = migrated.decisions["capacity"]
    check(cap.value is None and cap.lower_bound is not None and cap.is_pending,
          "a lower bound is known but true capacity remains unconfirmed")

    aggregate_session = SmudgeSession(
        rgb_crossing_canvas(), Brush(3, supersample=4),
        SmudgeConfig(capacity=1.0, **CROSS_CFG), record_steps=False)
    aggregate_session.run_stroke(CROSS_PATH)
    aggregate = infer_capacity_bounds(aggregate_session)
    check(aggregate.aggregate_only
          and aggregate.applicability == "aggregate_total_only_weak_lower_bound",
          "cumulative-only history must be marked as a weak aggregate bound")
    check(aggregate.minimum_capacity is not None,
          "cumulative totals should still provide a weak bound")


def test_halfway_save_reopen_completes_red_blue_stroke_and_replays_edges():
    original = original_crossing_session(capacity=1.0)
    initial_pixels = rgb_crossing_canvas().rgba.copy()

    legacy = {
        "id": "red-through-gap-to-blue",
        "format": "legacy_tip_color_only",
        "tip_color": list(TIP),
    }
    migration = migrate_legacy_brush(
        legacy,
        history=original.step_log,
        reference_capacity=1.0,
        config_overrides=CROSS_CFG,
    )
    check(migration.config is not None, "saturated history should yield config")
    check(abs(migration.config.capacity - 1.0) < 1e-8,
          "migration should recover the reference capacity")

    half_canvas = rgb_crossing_canvas()
    half = SmudgeSession(half_canvas, Brush(3.0, supersample=4),
                         migration.config, record_steps=True)
    half.begin_stroke()
    half.add_points(CROSS_PATH[:SPLIT])
    midpoint_pixels = half.canvas.rgba.copy()
    midpoint_reservoir = half.reservoir.pm.copy()
    midpoint_origin = half.stroke_origin

    with tempfile.TemporaryDirectory() as td:
        checkpoint_path = Path(td) / "half_stroke.json"
        record_path = Path(td) / "conversion_record.json"
        half.save_checkpoint(checkpoint_path)
        reopened = SmudgeSession.load_checkpoint(checkpoint_path)

        check(reopened._active is True and reopened._anchor is not None,
              "checkpoint must preserve an active continuous stroke")
        check_close(np.asarray(reopened.stroke_origin, dtype=float),
                    np.asarray(midpoint_origin, dtype=float),
                    "checkpoint must preserve the continuous-stroke origin")
        check_close(reopened.reservoir.pm, midpoint_reservoir,
                    "reopened reservoir must equal midpoint reservoir")
        check(np.array_equal(reopened.canvas.rgba, midpoint_pixels),
              "checkpoint canvas must round-trip exactly")

        reopened.add_points(CROSS_PATH[SPLIT:])
        reopened.end_stroke()

        # Persisted conversion record is independently re-readable JSON.
        comparison = compare_replays(
            original, reopened,
            before_original=initial_pixels,
            before_converted=initial_pixels.copy(),
            name="red_gap_blue_halfway_checkpoint",
        )
        migration.record.append_comparison(comparison)
        edges = edge_diff(original.canvas.rgba, reopened.canvas.rgba,
                          EDGE_SLICES)
        migration.record.event("edge_replay", edges)
        migration.record.save_json(record_path)
        reloaded_record = json.loads(record_path.read_text(encoding="utf-8"))
        check(reloaded_record["brush_id"] == migration.brush_id,
              "conversion record must survive JSON reload")
        check(len(reloaded_record["comparisons"]) == 1,
              "conversion record must contain the replay comparison")

    check(comparison["canvas_diff"]["exact_match"],
          "completed checkpoint replay must match the original recipe exactly")
    check_close(reopened.reservoir.pm, original.reservoir.pm,
                "remaining color amount must match reference replay")
    check(reopened.reservoir.held_volume() <= 1.0 + 1e-10,
          "remaining color must never exceed capacity")
    check(edges["all_edges_exact"],
          "red/transparent/blue edge bands must replay exactly")

    gap = reopened.canvas.rgba[16:25, 38:57]
    painted = gap[..., 3] > 1e-6
    check(gap[..., 3].max() > 0.0 and painted.any(),
          "red paint must be carried through the transparent gap")
    check((gap[painted, 0] > gap[painted, 2]).all(),
          "carried paint must retain red hue before reaching blue")
    check(original.ledger["picked"][2] > 0.0,
          "stroke must actually pick up blue-channel paint in the blue region")
    check(comparison["pickup"]["original_pickup_lost_alpha"] >= 0.0
          and "converted_loss_fraction" in comparison["pickup"],
          "comparison must report pickup loss amounts and fractions")
    check(comparison["canvas_increment"]["original"] is not None
          and comparison["canvas_increment"]["converted"] is not None,
          "comparison must report canvas increments for both recipes")


def test_checkpoint_split_is_batch_independent():
    one_shot = SmudgeSession(rgb_crossing_canvas(), Brush(3, supersample=4),
                             SmudgeConfig(capacity=0.8, **CROSS_CFG))
    one_shot.run_stroke(CROSS_PATH)

    batched = SmudgeSession(rgb_crossing_canvas(), Brush(3, supersample=4),
                            SmudgeConfig(capacity=0.8, **CROSS_CFG))
    batched.begin_stroke()
    batched.add_points(CROSS_PATH[:17])
    batched.add_points(CROSS_PATH[17:46])
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "split.json"
        batched.save_checkpoint(path)
        restored = SmudgeSession.load_checkpoint(path)
        restored.add_points(CROSS_PATH[46:])
        restored.end_stroke()
    check(np.array_equal(one_shot.canvas.rgba, restored.canvas.rgba),
          "save/reload and arbitrary batch boundaries must not change pixels")
    check_close(one_shot.reservoir.pm, restored.reservoir.pm,
                "save/reload must preserve remaining reservoir color")


def test_cross_layer_replay_matches_edges_and_remaining_color():
    from smudge import LayeredDocument

    def document():
        doc = LayeredDocument(40, 90, active_layer="fields")
        doc.layers["fields"] = rgb_crossing_canvas()
        line = doc.add_layer("thin_line")
        line.rgba[20, 10:31] = (0.1, 0.8, 0.1, 1.0)
        return doc

    specs = [
        {"layer": "fields", "points": CROSS_PATH,
         "config": SmudgeConfig(capacity=1.0, **CROSS_CFG), "radius": 3.0},
        {"layer": "thin_line",
         "points": [(20.0, float(x), 1.0) for x in range(18, 70)],
         "config": SmudgeConfig(capacity=0.7, pickup_rate=0.7,
                                deposit_rate=0.6),
         "radius": 2.5},
    ]
    first = document().replay(specs)
    second = document().replay(specs)

    for index, (a, b) in enumerate(zip(first, second)):
        check(np.array_equal(a.canvas.rgba, b.canvas.rgba),
              f"layer stroke {index} must replay bit-identically")
        check_close(a.reservoir.pm, b.reservoir.pm,
                    f"layer stroke {index} remaining color must replay exactly")
        check(a.reservoir.held_volume() <= a.cfg.capacity + 1e-10,
              f"layer stroke {index} must respect capacity")

    edges = edge_diff(first[0].canvas.rgba, second[0].canvas.rgba, EDGE_SLICES)
    check(edges["all_edges_exact"], "cross-layer field edges must be stable")
    line_edge = {
        "source_line": (slice(19, 22), slice(10, 32)),
        "transported_tip": (slice(18, 23), slice(45, 68)),
    }
    line_edges = edge_diff(first[1].canvas.rgba, second[1].canvas.rgba,
                           line_edge)
    check(line_edges["all_edges_exact"],
          "thin-line layer edges must be stable across replay")
    check(first[1].canvas.rgba[18:23, 45:68, 1].max() > 0.0,
          "second layer stroke must transport green paint forward")


def test_workbench_opens_legacy_and_modern_brushes_together():
    raw = {"id": "dual", "tip_color": list(TIP)}
    bench = BrushWorkbench()
    pair = bench.open_pair(raw, provisional_capacity=1.0,
                          config_overrides=CROSS_CFG)
    check("dual" in bench.legacy and "dual" in bench.modern,
          "workbench must retain both old raw and migrated handles")
    check(pair.legacy_raw == raw,
          "legacy handle must be an untouched deep copy")

    legacy_canvas = rgb_crossing_canvas()
    modern_canvas = rgb_crossing_canvas()
    legacy_run = pair.run_legacy_recipe(
        legacy_canvas, CROSS_PATH, radius=3.0, capacity=100.0)
    modern_run = pair.run_converted(modern_canvas, CROSS_PATH, radius=3.0)
    check(legacy_run.stamp_count == modern_run.stamp_count,
          "old and new recipes must tessellate the same geometry")
    check(np.isfinite(legacy_run.canvas.rgba).all()
          and np.isfinite(modern_run.canvas.rgba).all(),
          "dual recipes must produce finite pixels")
    check(legacy_run.ledger["pickup_lost"][3] == 0.0,
          "large reference legacy capacity should have no rejected pickup")
    check(modern_run.ledger["pickup_lost"][3] > 0.0,
          "finite converted capacity should exhibit measurable pickup loss")


def test_reload_and_another_stroke_never_rewrite_legacy_file():
    raw = {
        "format": "smudge-project",
        "active_brush_id": "same",
        "brushes": {"same": {
            "format": "legacy_tip_color_only",
            "tip_color": list(TIP),
        }},
    }
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        legacy_path = root / "old_project.json"
        converted_path = root / "converted_project.json"
        checkpoint_path = root / "reloaded_then_new_stroke.json"
        encoded = json.dumps(raw, ensure_ascii=False, indent=2).encode("utf-8")
        legacy_path.write_bytes(encoded)
        before_hash = source_fingerprint(legacy_path.read_bytes())

        bench = BrushWorkbench()
        project = bench.open_project(
            legacy_path, provisional_capacity=0.9,
            config_overrides=CROSS_CFG)
        migration = project.brushes["same"]
        check(migration.decisions["humidity"].is_pending,
              "test precondition: humidity needs confirmation")

        # The user resolves a pending value. That decision belongs to the
        # conversion sidecar, never to the source legacy archive.
        migration.confirm_field("humidity", 0.5)
        conversion_path = project.save(converted_path)
        check(conversion_path.exists(), "converted project must be written separately")
        try:
            project.save(legacy_path)
            raise AssertionError("conversion must not silently overwrite legacy source")
        except ValueError:
            pass
        check(legacy_path.read_bytes() == encoded,
              "refused overwrite attempt must leave legacy bytes intact")

        sess = SmudgeSession(rgb_crossing_canvas(), Brush(3, supersample=4),
                             migration.config, record_steps=True)
        sess.begin_stroke()
        sess.add_points(CROSS_PATH[:45])
        sess.save_checkpoint(checkpoint_path)
        reloaded = SmudgeSession.load_checkpoint(checkpoint_path)
        reloaded.add_points(CROSS_PATH[45:])
        reloaded.end_stroke()

        # Reload again and make a second complete stroke.
        reloaded.save_checkpoint(checkpoint_path)
        second = SmudgeSession.load_checkpoint(checkpoint_path)
        second.run_stroke([(20.0, 70.0 - 0.7 * i) for i in range(35)])

        after_bytes = legacy_path.read_bytes()
        check(after_bytes == encoded,
              "migration, confirmation, reload and later strokes must not "
              "modify the legacy file bytes")
        check(source_fingerprint(after_bytes) == before_hash,
              "legacy file hash must remain stable")
        check(project.source_unchanged,
              "project migration must report source unchanged")
        reopened_conversion = json.loads(conversion_path.read_text("utf-8"))
        check(reopened_conversion["source_sha256"] == before_hash,
              "conversion record must point back to the original source hash")
        check("humidity" not in reopened_conversion["brushes"]["same"][
            "unresolved_fields"], "confirmed decision must be saved in conversion")


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        if "tmp_path" in test.__code__.co_varnames:
            test(None)
        else:
            test()
        print(f"  PASS {test.__name__}")
    print(f"\n{len(tests)} migration tests / {passed} assertions passed.")


if __name__ == "__main__":
    main()
