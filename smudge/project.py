"""Project archives, dual old/new brush opening, and replay comparisons."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .brush import Brush
from .canvas import Canvas
from .migration import (
    MigrationResult,
    _jsonable,
    _number,
    _RADIUS_KEYS,
    _find_key,
    migrate_legacy_brush,
    parse_rgba_color,
    source_fingerprint,
)
from .session import SmudgeConfig, SmudgeSession


# --------------------------------------------------------------------------
# Recipe execution and comparison
# --------------------------------------------------------------------------

def make_brush(raw: dict | None = None, *, radius: float | None = None,
               brush: Brush | None = None) -> Brush:
    if brush is not None:
        return brush
    if raw:
        key, value = _find_key(raw, _RADIUS_KEYS)
        inferred = _number(value)
        if key and key.endswith("diameter") and inferred is not None:
            inferred *= 0.5
        if radius is None:
            radius = inferred
    return Brush(radius=float(radius if radius is not None else 4.0))


def run_stroke_recipe(
    canvas: Canvas,
    points,
    *,
    raw_brush: dict | None = None,
    migration: MigrationResult | None = None,
    brush: Brush | None = None,
    config: SmudgeConfig | dict | None = None,
    radius: float | None = None,
    capacity: float | None = None,
    load_tip: bool = False,
    initial_pm=None,
    restore_initial_pm: bool = False,
    record_steps: bool = True,
    pixel_order: str = "C",
):
    """Run either a legacy-compatible or migrated recipe on a canvas copy."""
    if migration is not None:
        raw_brush = migration.legacy
        if config is None:
            config = migration.config
        if capacity is None and config is not None:
            capacity = config.capacity
        if radius is None and migration.geometry.get("radius") is not None:
            radius = migration.geometry["radius"]
    if config is None:
        config = SmudgeConfig(capacity=float(capacity if capacity is not None
                                             else 1.0))
    elif isinstance(config, dict):
        overrides = dict(config)
        if capacity is not None:
            overrides["capacity"] = capacity
        config = SmudgeConfig(**overrides)
    elif capacity is not None:
        config.capacity = float(capacity)

    brush_obj = make_brush(raw_brush, radius=radius, brush=brush)
    session = SmudgeSession(canvas, brush_obj, config,
                            pixel_order=pixel_order,
                            record_steps=record_steps)
    load_rgba = None
    if load_tip and raw_brush is not None:
        from .migration import _COLOR_KEYS
        _, color_value = _find_key(raw_brush, _COLOR_KEYS)
        load_rgba = parse_rgba_color(color_value)
    pm_load = initial_pm
    if pm_load is None and restore_initial_pm and migration is not None:
        pm_load = migration.initial_pm
    session.run_stroke(points, load_rgba=load_rgba, initial_pm=pm_load)
    return session


def canvas_increment(before: Canvas | np.ndarray,
                     after: Canvas | np.ndarray) -> dict:
    before_rgba = before.rgba if isinstance(before, Canvas) else np.asarray(before)
    after_rgba = after.rgba if isinstance(after, Canvas) else np.asarray(after)
    delta = after_rgba - before_rgba
    before_canvas = (before if isinstance(before, Canvas)
                     else Canvas.from_rgba_array(before_rgba))
    after_canvas = (after if isinstance(after, Canvas)
                    else Canvas.from_rgba_array(after_rgba))
    pm_delta = after_canvas.paint_mass() - before_canvas.paint_mass()
    return {
        "rgba_net_sum": delta.sum(axis=(0, 1)).tolist(),
        "rgba_abs_sum": np.abs(delta).sum(axis=(0, 1)).tolist(),
        "rgba_positive_sum": np.maximum(delta, 0.0).sum(axis=(0, 1)).tolist(),
        "paint_mass_delta_pm": pm_delta.tolist(),
        "alpha_net_sum": float(delta[..., 3].sum()),
        "alpha_positive_sum": float(np.maximum(delta[..., 3], 0.0).sum()),
    }


def compare_replays(original, converted, *, before_original=None,
                    before_converted=None, name: str = "recipe_comparison",
                    atol: float = 1e-10) -> dict:
    """Compare pickup loss, canvas increment, edges and remaining reservoir."""
    orig_canvas = original.canvas if hasattr(original, "canvas") else original[0]
    conv_canvas = converted.canvas if hasattr(converted, "canvas") else converted[0]
    if before_original is None:
        before_original = getattr(original, "initial_canvas", None)
    if before_converted is None:
        before_converted = getattr(converted, "initial_canvas", None)

    result = {
        "name": name,
        "stamp_counts": [getattr(original, "stamp_count", None),
                         getattr(converted, "stamp_count", None)],
        "pickup": {
            "original_picked_pm": original.ledger["picked"].tolist(),
            "converted_picked_pm": converted.ledger["picked"].tolist(),
            "original_pickup_lost_pm": original.ledger["pickup_lost"].tolist(),
            "converted_pickup_lost_pm": converted.ledger["pickup_lost"].tolist(),
            "original_pickup_lost_alpha": float(
                original.ledger["pickup_lost"][3]),
            "converted_pickup_lost_alpha": float(
                converted.ledger["pickup_lost"][3]),
        },
        "canvas_increment": {
            "original": (canvas_increment(before_original, orig_canvas)
                         if before_original is not None else None),
            "converted": (canvas_increment(before_converted, conv_canvas)
                          if before_converted is not None else None),
        },
        "remaining_reservoir_pm": {
            "original": original.reservoir.pm.tolist(),
            "converted": converted.reservoir.pm.tolist(),
            "original_alpha": float(original.reservoir.held_volume()),
            "converted_alpha": float(converted.reservoir.held_volume()),
        },
    }
    for label, session in (("original", original), ("converted", converted)):
        picked_alpha = float(session.ledger["picked"][3])
        lost_alpha = float(session.ledger["pickup_lost"][3])
        result["pickup"][f"{label}_loss_fraction"] = (
            lost_alpha / (picked_alpha + lost_alpha)
            if picked_alpha + lost_alpha > 0.0 else 0.0
        )
    diff = np.abs(orig_canvas.rgba - conv_canvas.rgba)
    result["canvas_diff"] = {
        "max_abs_rgba": float(diff.max(initial=0.0)),
        "mean_abs_rgba": float(diff.mean()),
        "exact_match": bool(np.array_equal(orig_canvas.rgba, conv_canvas.rgba)),
        "close_match": bool(np.allclose(orig_canvas.rgba, conv_canvas.rgba,
                                        atol=atol, rtol=0.0)),
    }
    result["matches_within_atol"] = result["canvas_diff"]["close_match"]
    return _jsonable(result)


def edge_diff(reference_rgba: np.ndarray, candidate_rgba: np.ndarray,
              slices: dict[str, tuple[slice, ...]] | None = None) -> dict:
    """Check named edge bands; ``slices`` maps names to NumPy slice tuples."""
    a = np.asarray(reference_rgba)
    b = np.asarray(candidate_rgba)
    output = {}
    for name, sl in (slices or {}).items():
        if isinstance(sl, (slice, int)) or sl is Ellipsis:
            sl = (sl,)
        diff = np.abs(a[sl] - b[sl])
        output[name] = {
            "max_abs": float(diff.max(initial=0.0)),
            "mean_abs": float(diff.mean()),
            "exact": bool(np.array_equal(a[sl], b[sl])),
        }
    output["all_edges_exact"] = all(item["exact"] for item in output.values())
    return output


# --------------------------------------------------------------------------
# Project archive and dual brush workbench
# --------------------------------------------------------------------------

@dataclass
class ProjectArchive:
    raw: dict
    path: Path | None = None
    source_bytes_sha256: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "ProjectArchive":
        path = Path(path)
        data = path.read_bytes()
        return cls(raw=json.loads(data.decode("utf-8")), path=path,
                   source_bytes_sha256=source_fingerprint(data))

    def brush_ids(self) -> list[str]:
        brushes = self.raw.get("brushes")
        if isinstance(brushes, dict):
            return [str(k) for k in brushes]
        if isinstance(self.raw.get("brush"), dict):
            return [str(self.raw.get("active_brush_id")
                        or self.raw["brush"].get("id") or "brush")]
        if any(k in self.raw for k in ("tip_color", "nib_color", "color")):
            return [str(self.raw.get("id") or self.raw.get("brush_id")
                        or "brush")]
        return []

    def brush_raw(self, brush_id: str) -> dict:
        brushes = self.raw.get("brushes")
        if isinstance(brushes, dict) and brush_id in brushes:
            return copy.deepcopy(brushes[brush_id])
        if isinstance(self.raw.get("brush"), dict):
            return copy.deepcopy(self.raw["brush"])
        if brush_id in self.brush_ids():
            return copy.deepcopy(self.raw)
        raise KeyError(f"brush {brush_id!r} not found in project")

    def current_bytes_sha256(self) -> str | None:
        if self.path is None or not self.path.exists():
            return None
        return source_fingerprint(self.path.read_bytes())

    def source_unchanged(self) -> bool:
        return (self.source_bytes_sha256 is not None
                and self.current_bytes_sha256() == self.source_bytes_sha256)

    def save_converted(self, path: str | Path, converted: dict) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(converted), ensure_ascii=False,
                                   indent=2, sort_keys=True),
                        encoding="utf-8")
        return path


@dataclass
class OpenBrushPair:
    brush_id: str
    legacy_raw: dict
    migration: MigrationResult
    archive: ProjectArchive | None = None

    def run_legacy_recipe(self, canvas: Canvas, points, **kwargs):
        kwargs.setdefault("load_tip", False)
        return run_stroke_recipe(canvas, points, raw_brush=self.legacy_raw,
                                 migration=None, **kwargs)

    def run_converted(self, canvas: Canvas, points, **kwargs):
        return run_stroke_recipe(canvas, points, migration=self.migration,
                                 **kwargs)


class BrushWorkbench:
    """Keeps the untouched legacy handle and migrated handle open together."""

    def __init__(self):
        self.archives: dict[str, ProjectArchive] = {}
        self.legacy: dict[str, dict] = {}
        self.modern: dict[str, MigrationResult] = {}
        self.records: dict[str, MigrationResult] = {}

    def open_pair(self, source: dict | str | Path, *, brush_id: str | None = None,
                  history: Any = None, **migration_kwargs) -> OpenBrushPair:
        archive = None
        if isinstance(source, ProjectArchive):
            archive = source
            bid = brush_id or archive.brush_ids()[0]
            raw = archive.brush_raw(bid)
        elif isinstance(source, (str, Path)) and Path(source).exists():
            archive = ProjectArchive.load(source)
            key = str(Path(source))
            self.archives[key] = archive
            bid = brush_id or archive.raw.get("active_brush_id")
            if bid is None:
                ids = archive.brush_ids()
                if not ids:
                    raise KeyError("project contains no brush")
                bid = ids[0]
            raw = archive.brush_raw(str(bid))
        else:
            raw = dict(source)  # type: ignore[arg-type]
            bid = brush_id or str(raw.get("id") or "brush")
        result = migrate_legacy_brush(raw, brush_id=str(bid), history=history,
                                      **migration_kwargs)
        self.legacy[result.brush_id] = copy.deepcopy(raw)
        self.modern[result.brush_id] = result
        self.records[result.brush_id] = result
        return OpenBrushPair(result.brush_id, copy.deepcopy(raw), result, archive)

    def open_project(self, path: str | Path, *,
                     histories: dict[str, Any] | None = None,
                     **migration_kwargs) -> "ProjectMigration":
        archive = ProjectArchive.load(path)
        histories = histories or {}
        results = {}
        for bid in archive.brush_ids():
            raw = archive.brush_raw(bid)
            results[bid] = migrate_legacy_brush(
                raw, brush_id=bid, history=histories.get(bid),
                source_path=archive.path,
                source_sha256=archive.source_bytes_sha256,
                **migration_kwargs)
            self.legacy[bid] = copy.deepcopy(raw)
            self.modern[bid] = results[bid]
            self.records[bid] = results[bid]
        return ProjectMigration(archive=archive, brushes=results)


class SmudgeTool(BrushWorkbench):
    """Smudge-tool facade that can keep legacy and migrated brushes open."""

    def open_old_and_new(self, source, **kwargs) -> OpenBrushPair:
        return self.open_pair(source, **kwargs)


@dataclass
class ProjectMigration:
    archive: ProjectArchive
    brushes: dict[str, MigrationResult] = field(default_factory=dict)

    @property
    def source_unchanged(self) -> bool:
        return self.archive.source_unchanged()

    def to_dict(self) -> dict:
        return {
            "format": "smudge-converted-project",
            "source_path": str(self.archive.path) if self.archive.path else None,
            "source_sha256": self.archive.source_bytes_sha256,
            "source_unchanged_after_migration": self.source_unchanged,
            "brushes": {bid: result.to_dict()
                        for bid, result in self.brushes.items()},
        }

    def save(self, path: str | Path, *, overwrite_source: bool = False) -> Path:
        target = Path(path)
        if (not overwrite_source and self.archive.path is not None
                and target.exists()
                and target.resolve() == self.archive.path.resolve()):
            raise ValueError(
                "refusing to overwrite the legacy source archive; save the "
                "conversion to a new path or pass overwrite_source=True")
        return self.archive.save_converted(target, self.to_dict())


def load_project(path: str | Path) -> ProjectArchive:
    return ProjectArchive.load(path)


def migrate_project(
    source: str | Path | dict | ProjectArchive,
    *,
    output_path: str | Path | None = None,
    histories: dict[str, Any] | None = None,
    **migration_kwargs,
) -> ProjectMigration:
    """Migrate every brush in a project without modifying its source archive."""
    if isinstance(source, ProjectArchive):
        archive = source
    elif isinstance(source, dict):
        archive = ProjectArchive(raw=source)
    else:
        archive = ProjectArchive.load(source)

    histories = histories or {}
    results = {}
    for brush_id in archive.brush_ids():
        results[brush_id] = migrate_legacy_brush(
            archive.brush_raw(brush_id),
            brush_id=brush_id,
            history=histories.get(brush_id),
            source_path=archive.path,
            source_sha256=archive.source_bytes_sha256,
            **migration_kwargs,
        )
    migration = ProjectMigration(archive=archive, brushes=results)
    if output_path is not None:
        migration.save(output_path)
    return migration


convert_project = migrate_project
migrate_legacy_project = migrate_project
compare_recipes = compare_replays
compare_edges = edge_diff
canvas_delta = canvas_increment
SmudgeWorkbench = BrushWorkbench
DualBrushWorkbench = BrushWorkbench


def replay_document(document: "LayeredDocument", specs):
    return document.replay(specs)


replay_layered = replay_document
replay_cross_layer = replay_document


@dataclass
class StrokeSpec:
    points: list
    layer: str = "layer"
    name: str | None = None
    raw_brush: dict | None = None
    migration: MigrationResult | None = None
    config: Any = None
    radius: float | None = None
    capacity: float | None = None
    load_tip: bool = False
    kwargs: dict = field(default_factory=dict)


class LayeredDocument:
    """A small layer container for deterministic cross-layer replay checks."""

    def __init__(self, height: int, width: int,
                 layers: dict[str, Canvas] | None = None,
                 active_layer: str = "layer"):
        self.height = int(height)
        self.width = int(width)
        self.layers = layers or {active_layer: Canvas(height, width)}
        if active_layer not in self.layers:
            self.layers[active_layer] = Canvas(height, width)
        self.active_layer = active_layer

    def add_layer(self, name: str, canvas: Canvas | None = None) -> Canvas:
        self.layers[name] = canvas or Canvas(self.height, self.width)
        return self.layers[name]

    def flattened(self) -> Canvas:
        """Source-over flattening of layers in insertion order."""
        out = Canvas(self.height, self.width)
        for layer in self.layers.values():
            src = layer.rgba
            sa = src[..., 3:4]
            da = out.rgba[..., 3:4]
            out_a = sa + da * (1.0 - sa)
            premult_src_rgb = src[..., :3] * sa
            premult_dst_rgb = out.rgba[..., :3] * da
            out_pm_rgb = premult_src_rgb + premult_dst_rgb * (1.0 - sa)
            safe_a = np.maximum(out_a, 1e-12)
            out.rgba[..., :3] = out_pm_rgb / safe_a
            out.rgba[..., 3:4] = out_a
        return out

    def replay(self, specs: list[StrokeSpec | dict]) -> list[SmudgeSession]:
        sessions = []
        for index, spec in enumerate(specs):
            if not isinstance(spec, StrokeSpec):
                spec = StrokeSpec(
                    points=spec["points"],
                    layer=spec.get("layer", self.active_layer),
                    name=spec.get("name", f"stroke-{index}"),
                    raw_brush=spec.get("raw_brush"),
                    migration=spec.get("migration"),
                    config=spec.get("config"),
                    radius=spec.get("radius"),
                    capacity=spec.get("capacity"),
                    load_tip=spec.get("load_tip", False),
                    kwargs=spec.get("kwargs", {}),
                )
            canvas = self.layers[spec.layer]
            session = run_stroke_recipe(
                canvas, spec.points,
                raw_brush=spec.raw_brush,
                migration=spec.migration,
                config=spec.config,
                radius=spec.radius,
                capacity=spec.capacity,
                load_tip=spec.load_tip,
                **spec.kwargs,
            )
            sessions.append(session)
        return sessions


def save_project(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2,
                               sort_keys=True), encoding="utf-8")
    return path
