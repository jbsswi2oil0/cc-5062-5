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
from .migration import (
    BrushMigration,
    CapacityBound,
    CapacityEstimate,
    CapacityInference,
    CapacityReport,
    ConversionRecord,
    ConversionReport,
    FieldDecision,
    LegacyBrush,
    MigratedBrush,
    MigrationRecord,
    MigrationReport,
    MigrationResult,
    MigrationStatus,
    ModernBrush,
    OldBrush,
    OriginMarker,
    StrokeOrigin,
    backcalculate_capacity_bounds,
    capacity_from_history,
    estimate_capacity_bounds,
    infer_capacity_bounds,
    load_legacy_brush_file,
    migrate_brush,
    migrate_legacy_brush,
    restore_nib_by_capacity,
)
from .project import (
    BrushWorkbench,
    DualBrushWorkbench,
    LayeredDocument,
    OpenBrushPair,
    ProjectArchive,
    ProjectMigration,
    SmudgeWorkbench,
    StrokeSpec,
    canvas_delta,
    canvas_increment,
    compare_edges,
    compare_recipes,
    compare_replays,
    convert_project,
    edge_diff,
    load_project,
    migrate_legacy_project,
    migrate_project,
    replay_cross_layer,
    replay_document,
    replay_layered,
    run_stroke_recipe,
    save_project,
    SmudgeTool,
)

__all__ = [
    "Brush",
    "Canvas",
    "SmudgeConfig",
    "SmudgeSession",
    "step_contact",
    "step_flux",
    "CapacityInference",
    "CapacityBound",
    "CapacityEstimate",
    "CapacityReport",
    "FieldDecision",
    "LegacyBrush",
    "OldBrush",
    "MigratedBrush",
    "ModernBrush",
    "MigrationRecord",
    "MigrationReport",
    "ConversionRecord",
    "ConversionReport",
    "BrushMigration",
    "MigrationResult",
    "MigrationStatus",
    "StrokeOrigin",
    "OriginMarker",
    "infer_capacity_bounds",
    "backcalculate_capacity_bounds",
    "estimate_capacity_bounds",
    "capacity_from_history",
    "load_legacy_brush_file",
    "migrate_brush",
    "migrate_legacy_brush",
    "restore_nib_by_capacity",
    "BrushWorkbench",
    "SmudgeWorkbench",
    "DualBrushWorkbench",
    "SmudgeTool",
    "LayeredDocument",
    "StrokeSpec",
    "OpenBrushPair",
    "ProjectArchive",
    "ProjectMigration",
    "migrate_project",
    "convert_project",
    "migrate_legacy_project",
    "canvas_increment",
    "canvas_delta",
    "compare_recipes",
    "compare_replays",
    "compare_edges",
    "replay_document",
    "replay_layered",
    "replay_cross_layer",
    "edge_diff",
    "load_project",
    "run_stroke_recipe",
    "save_project",
]
