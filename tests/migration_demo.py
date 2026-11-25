"""Generate the traceable legacy-brush migration scenario and JSON record."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smudge import (  # noqa: E402
    Brush,
    BrushWorkbench,
    Canvas,
    SmudgeConfig,
    SmudgeSession,
    compare_replays,
    edge_diff,
)
from smudge.migration import migrate_legacy_brush, source_fingerprint  # noqa: E402

OUT = ROOT / "outputs"
BRUSH_ID = "red-gap-blue"
TIP = [0.8, 0.1, 0.1, 1.0]
PATH = [(20.0, float(x), 1.0) for x in range(10, 81)]
SPLIT = 41
CONFIG = dict(model="contact", pickup_rate=0.9, deposit_rate=0.4,
              step_length=1.0)
EDGES = {
    "red_to_transparent_edge": (slice(16, 25), slice(27, 34)),
    "transparent_gap": (slice(16, 25), slice(38, 57)),
    "transparent_to_blue_edge": (slice(16, 25), slice(57, 65)),
    "blue_interior": (slice(16, 25), slice(66, 82)),
}


def scene() -> Canvas:
    canvas = Canvas(40, 90)
    canvas.rgba[:, :30] = (0.8, 0.1, 0.1, 1.0)
    canvas.rgba[:, 60:] = (0.1, 0.1, 0.9, 1.0)
    return canvas


def main() -> None:
    OUT.mkdir(exist_ok=True)
    legacy_project_path = OUT / "10_legacy_project.json"
    converted_project_path = OUT / "10_converted_project.json"
    checkpoint_path = OUT / "10_halfway_checkpoint.json"
    record_path = OUT / "10_migration_record.json"

    original = SmudgeSession(scene(), Brush(3.0, supersample=4),
                             SmudgeConfig(capacity=1.0, **CONFIG),
                             record_steps=True)
    original.run_stroke(PATH)
    initial_pixels = scene().rgba.copy()

    legacy_project = {
        "format": "smudge-project",
        "note": "Deliberately contains only the legacy nib color for the brush.",
        "active_brush_id": BRUSH_ID,
        "brushes": {
            BRUSH_ID: {
                "id": BRUSH_ID,
                "format": "legacy_tip_color_only",
                "tip_color": TIP,
            }
        },
    }
    legacy_bytes = json.dumps(legacy_project, ensure_ascii=False, indent=2,
                              sort_keys=True).encode("utf-8")
    legacy_project_path.write_bytes(legacy_bytes)
    legacy_hash = source_fingerprint(legacy_bytes)

    bench = BrushWorkbench()
    project = bench.open_project(
        legacy_project_path,
        histories={BRUSH_ID: original.step_log},
        reference_capacity=1.0,
        config_overrides=CONFIG,
    )
    migration = project.brushes[BRUSH_ID]

    # Explicit zero versus key-absent projects, kept distinguishable.
    zero = migrate_legacy_brush({"id": "same", "tip_color": TIP,
                                 "capacity": 0.0})
    missing = migrate_legacy_brush({"id": "same", "tip_color": TIP})
    migration.record.event("zero_vs_missing_capacity", {
        "explicit_zero": zero.decisions["capacity"].to_dict(),
        "never_stored": missing.decisions["capacity"].to_dict(),
    })

    halfway = SmudgeSession(scene(), Brush(3.0, supersample=4),
                            migration.config, record_steps=True)
    halfway.begin_stroke()
    halfway.add_points(PATH[:SPLIT])
    halfway.save_checkpoint(checkpoint_path, include_history=False)

    reopened = SmudgeSession.load_checkpoint(checkpoint_path)
    reopened.add_points(PATH[SPLIT:])
    reopened.end_stroke()

    comparison = compare_replays(
        original, reopened,
        before_original=initial_pixels,
        before_converted=initial_pixels.copy(),
        name="red_pickup_transparent_gap_blue_pickup_halfway_reload",
    )
    edges = edge_diff(original.canvas.rgba, reopened.canvas.rgba, EDGES)
    migration.record.append_comparison(comparison)
    migration.record.event("edge_and_remaining_color_replay", {
        "edges": edges,
        "remaining_capacity": migration.config.capacity,
        "remaining_reservoir_pm": reopened.reservoir.pm.tolist(),
        "remaining_alpha": reopened.reservoir.held_volume(),
    })

    reopened.canvas.save_png(OUT / "10_converted_final.png")
    reopened.canvas.save_raw(OUT / "10_converted_final.npy")
    original.canvas.save_png(OUT / "10_original_final.png")
    original.canvas.save_raw(OUT / "10_original_final.npy")

    project.save(converted_project_path)
    migration.record.source_path = str(legacy_project_path)
    migration.record.source_sha256 = legacy_hash
    migration.record.save_json(record_path)

    after_hash = source_fingerprint(legacy_project_path.read_bytes())
    summary = {
        "record": str(record_path),
        "converted_project": str(converted_project_path),
        "checkpoint": str(checkpoint_path),
        "legacy_source_unchanged": after_hash == legacy_hash,
        "legacy_source_sha256": legacy_hash,
        "capacity_status": migration.decisions["capacity"].status,
        "unresolved_fields": migration.unresolved_fields,
        "canvas_exact_match": comparison["canvas_diff"]["exact_match"],
        "all_named_edges_exact": edges["all_edges_exact"],
        "remaining_alpha": reopened.reservoir.held_volume(),
    }
    (OUT / "10_migration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["legacy_source_unchanged"]:
        raise SystemExit("legacy project file was modified")
    if not summary["canvas_exact_match"] or not summary["all_named_edges_exact"]:
        raise SystemExit("migration replay mismatch")


if __name__ == "__main__":
    main()
