"""Migration of legacy tip-color brushes to finite paint reservoirs.

The important distinction in this module is between a value that is merely
usable for a replay and a value that the archive can establish:

* an explicit ``capacity: 0`` is a confirmed zero-capacity brush;
* an archive in which no capacity key was ever stored stays pending even when a
  provisional replay capacity is supplied;
* stroke history can prove a lower bound on capacity, but usually cannot prove
  that the real capacity equals that bound;
* when the true beginning of a continuous stroke is unavailable, the first
  replayable event is explicitly marked as the calculation origin.  It is not
  silently claimed to be the historical stroke start.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .session import Reservoir, SmudgeConfig


class MigrationStatus(str, Enum):
    CONFIRMED = "confirmed"
    DERIVED = "derived"
    LOWER_BOUND = "lower_bound"
    PENDING = "pending"
    PENDING_CONFIRMATION = "pending"
    UNRESOLVED = "pending"
    ASSUMED = "assumed"
    MARKED = "marked"


PENDING = MigrationStatus.PENDING.value
PENDING_CONFIRMATION = MigrationStatus.PENDING_CONFIRMATION.value
UNRESOLVED = MigrationStatus.UNRESOLVED.value
CONFIRMED = MigrationStatus.CONFIRMED.value
DERIVED = MigrationStatus.DERIVED.value
LOWER_BOUND = MigrationStatus.LOWER_BOUND.value
ASSUMED = MigrationStatus.ASSUMED.value
MARKED = MigrationStatus.MARKED.value

STATUS_LABELS_ZH = {
    CONFIRMED: "已确认",
    DERIVED: "历史推导",
    LOWER_BOUND: "仅有下界",
    PENDING: "待确认",
    ASSUMED: "临时假设，待确认",
    MARKED: "已标起算点，真起点待确认",
}

_COLOR_KEYS = ("tip_color", "tipColor", "nib_color", "nibColor",
               "brush_color", "brushColor", "color", "nib", "tip")
_HUMIDITY_KEYS = ("humidity", "wetness", "moisture", "moisture_level",
                  "moistureLevel", "wetnessLevel")
_CAPACITY_KEYS = ("paint_capacity", "paintCapacity", "color_capacity",
                  "colorCapacity", "reservoir_capacity", "reservoirCapacity",
                  "capacity", "max_paint", "maxPaint", "max_load", "maxLoad",
                  "max_capacity", "maxCapacity")
_ORIGIN_KEYS = ("continuous_stroke_start", "continuousStrokeStart",
                "stroke_start", "strokeStart", "stroke_origin", "strokeOrigin",
                "origin", "start_point", "startPoint", "start")
_RADIUS_KEYS = ("radius", "brush_radius", "brushRadius", "size", "diameter")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class FieldDecision:
    name: str
    value: Any
    status: str = PENDING
    source_key: str | None = None
    reason: str = ""
    lower_bound: float | None = None
    upper_bound: float | None = None
    provisional_value: Any = None
    needs_confirmation: bool = False

    @property
    def is_pending(self) -> bool:
        return self.status in (PENDING, ASSUMED, MARKED) or self.needs_confirmation

    @property
    def status_zh(self) -> str:
        return STATUS_LABELS_ZH.get(self.status, str(self.status))

    @property
    def runtime_value(self) -> Any:
        return self.provisional_value if self.value is None else self.value

    def to_dict(self) -> dict:
        return _jsonable({
            "name": self.name,
            "value": self.value,
            "status": self.status,
            "status_zh": self.status_zh,
            "source_key": self.source_key,
            "reason": self.reason,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "provisional_value": self.provisional_value,
            "needs_confirmation": self.needs_confirmation,
            "is_pending": self.is_pending,
        })


@dataclass
class StrokeOrigin:
    """Marks where capacity accounting starts.

    ``historical_start_known`` distinguishes the real continuous-stroke start
    from a fallback calculation marker.
    """

    event_index: int = 0
    point_index: int | None = None
    stroke_id: str | int | None = None
    layer: str | None = None
    y: float | None = None
    x: float | None = None
    status: str = MARKED
    historical_start_known: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return _jsonable({
            "event_index": self.event_index,
            "point_index": self.point_index,
            "stroke_id": self.stroke_id,
            "layer": self.layer,
            "y": self.y,
            "x": self.x,
            "status": self.status,
            "historical_start_known": self.historical_start_known,
            "reason": self.reason,
        })


@dataclass
class CapacityInference:
    event_count: int
    origin: StrokeOrigin
    minimum_capacity: float | None = None
    minimum_initial_load: float | None = None
    fixed_initial_capacity: float | None = None
    exact_capacity: float | None = None
    saturated: bool = False
    feasible: bool = True
    aggregate_only: bool = False
    applicability: str = "no_history"
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "event_count": self.event_count,
            "origin": self.origin.to_dict(),
            "minimum_capacity": self.minimum_capacity,
            "minimum_initial_load": self.minimum_initial_load,
            "fixed_initial_capacity": self.fixed_initial_capacity,
            "exact_capacity": self.exact_capacity,
            "saturated": self.saturated,
            "feasible": self.feasible,
            "aggregate_only": self.aggregate_only,
            "applicability": self.applicability,
            "assumptions": list(self.assumptions),
        }


@dataclass
class MigrationRecord:
    brush_id: str
    source_format: str = "legacy_tip_color"
    source_path: str | None = None
    source_sha256: str | None = None
    decisions: dict[str, FieldDecision] = field(default_factory=dict)
    capacity_inference: CapacityInference | None = None
    origin: StrokeOrigin | None = None
    events: list[dict] = field(default_factory=list)
    comparisons: list[dict] = field(default_factory=list)

    def decide(self, decision: FieldDecision) -> FieldDecision:
        self.decisions[decision.name] = decision
        self.event(f"field:{decision.name}", decision.to_dict())
        return decision

    def event(self, kind: str, detail: dict | str | None = None) -> None:
        self.events.append({"kind": kind, "detail": _jsonable(detail)})

    @property
    def unresolved_fields(self) -> list[str]:
        return [name for name, d in self.decisions.items() if d.is_pending]

    @property
    def pending_fields(self) -> list[str]:
        return self.unresolved_fields

    @property
    def status(self) -> str:
        return PENDING if self.unresolved_fields else CONFIRMED

    def to_dict(self) -> dict:
        return {
            "record_type": "brush_migration_record",
            "brush_id": self.brush_id,
            "source_format": self.source_format,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "status": self.status,
            "unresolved_fields": self.unresolved_fields,
            "decisions": {k: v.to_dict() for k, v in self.decisions.items()},
            "capacity_inference": (
                self.capacity_inference.to_dict()
                if self.capacity_inference is not None else None
            ),
            "origin": self.origin.to_dict() if self.origin is not None else None,
            "events": _jsonable(self.events),
            "comparisons": _jsonable(self.comparisons),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent,
                          sort_keys=True)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    save = save_json

    def append_comparison(self, comparison: dict) -> None:
        self.comparisons.append(_jsonable(comparison))
        self.event("replay_comparison", comparison)


@dataclass
class MigrationResult:
    brush_id: str
    legacy: dict
    tip_color: tuple[float, float, float, float] | None
    decisions: dict[str, FieldDecision]
    record: MigrationRecord
    config: SmudgeConfig | None = None
    initial_pm: tuple[float, float, float, float] | None = None
    geometry: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.config is not None

    @property
    def unresolved_fields(self) -> list[str]:
        return self.record.unresolved_fields

    @property
    def pending_fields(self) -> list[str]:
        return self.unresolved_fields

    @property
    def needs_confirmation(self) -> bool:
        return bool(self.unresolved_fields)

    @property
    def status(self) -> str:
        return PENDING if self.needs_confirmation else CONFIRMED

    @property
    def capacity_status(self) -> str | None:
        decision = self.decisions.get("capacity")
        return None if decision is None else decision.status

    def restored_reservoir(self) -> Reservoir:
        capacity = self.runtime_capacity()
        reservoir = Reservoir(0.0 if capacity is None else capacity)
        if self.initial_pm is not None:
            reservoir.pm = np.asarray(self.initial_pm, dtype=np.float64).copy()
        return reservoir

    @property
    def capacity(self) -> Any:
        return self.decisions["capacity"].value

    @property
    def humidity(self) -> Any:
        return self.decisions["humidity"].value

    @property
    def stroke_start(self) -> StrokeOrigin | None:
        return self.record.origin

    def runtime_capacity(self) -> float | None:
        decision = self.decisions.get("capacity")
        if decision is None:
            return None
        value = decision.runtime_value
        return None if value is None else float(value)

    def confirm_field(self, name: str, value: Any = None, *,
                      reason: str = "confirmed during migration review") -> FieldDecision:
        previous = self.decisions.get(name)
        if value is None and previous is not None:
            value = previous.runtime_value
        if value is None:
            raise ValueError(f"no value supplied to confirm for {name}")
        decision = FieldDecision(
            name=name,
            value=_jsonable(value),
            status=CONFIRMED,
            source_key="user_confirmation",
            reason=reason,
            lower_bound=previous.lower_bound if previous else None,
            upper_bound=previous.upper_bound if previous else None,
            needs_confirmation=False,
        )
        self.decisions[name] = decision
        self.record.decisions[name] = decision
        self.record.event("field_confirmed", decision.to_dict())
        if name == "capacity":
            from dataclasses import replace
            capacity = float(value)
            self.config = (replace(self.config, capacity=capacity)
                           if self.config is not None
                           else SmudgeConfig(capacity=capacity))
        elif name == "humidity" and self.config is not None:
            from dataclasses import replace
            self.config = replace(self.config, humidity=float(value))
        return decision

    confirm = confirm_field

    def to_dict(self) -> dict:
        return {
            "brush_id": self.brush_id,
            "status": self.status,
            "needs_confirmation": self.needs_confirmation,
            "legacy": _jsonable(self.legacy),
            "tip_color": list(self.tip_color) if self.tip_color else None,
            "initial_pm": list(self.initial_pm) if self.initial_pm else None,
            "geometry": _jsonable(self.geometry),
            "config": _jsonable(self.config.__dict__) if self.config else None,
            "ready": self.ready,
            "unresolved_fields": self.unresolved_fields,
            "record": self.record.to_dict(),
        }


# --------------------------------------------------------------------------
# Reading legacy scalar fields
# --------------------------------------------------------------------------

def parse_rgba_color(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#"):
            hexed = text[1:]
            if len(hexed) == 3:
                hexed = "".join(ch * 2 for ch in hexed)
            if len(hexed) not in (6, 8):
                raise ValueError(f"unsupported hex color: {value!r}")
            nums = [int(hexed[i:i + 2], 16) / 255.0
                    for i in range(0, len(hexed), 2)]
            if len(nums) == 3:
                nums.append(1.0)
            return tuple(float(x) for x in nums)  # type: ignore[return-value]
        value = json.loads(text)
    if isinstance(value, dict):
        if "r" in value:
            nums = [value.get("r", 0.0), value.get("g", 0.0),
                    value.get("b", 0.0), value.get("a", value.get("alpha", 1.0))]
        else:
            return None
    else:
        nums = list(value)
    if len(nums) == 3:
        nums.append(1.0)
    if len(nums) != 4:
        raise ValueError("RGBA colors must have 3 or 4 components")
    r, g, b, alpha = (float(x) for x in nums)
    if max(r, g, b) > 1.5:
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        if alpha > 1.5:
            alpha /= 255.0
    elif alpha > 1.5:
        alpha /= 255.0
    return tuple(float(np.clip(x, 0.0, 1.0))
                 for x in (r, g, b, alpha))  # type: ignore[return-value]


def _find_key(raw: dict, names: Iterable[str]) -> tuple[str | None, Any]:
    for name in names:
        if name in raw:
            return name, raw[name]
    for nested_name in ("reservoir", "paint", "pigment", "settings", "brush"):
        nested = raw.get(nested_name)
        if isinstance(nested, dict):
            key, value = _find_key(nested, names)
            if key is not None:
                return f"{nested_name}.{key}", value
    return None, None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _read_humidity(raw: dict) -> tuple[str | None, float | None, str | None]:
    key, raw_value = _find_key(raw, _HUMIDITY_KEYS)
    value = _number(raw_value) if key is not None else None
    if value is None:
        return key, None, None
    if 0.0 <= value <= 1.0:
        return key, value, None
    if 1.0 < value <= 100.0:
        return key, value / 100.0, f"normalized {value:g}% humidity to [0, 1]"
    return key, value, "humidity is outside [0, 1] and is not a 0-100 percentage"


def _read_origin(raw: dict) -> StrokeOrigin | dict | None:
    key, value = _find_key(raw, _ORIGIN_KEYS)
    if key is None or value is None:
        return None
    if isinstance(value, StrokeOrigin):
        return value
    if isinstance(value, dict):
        data = dict(value)
        data.setdefault("status", CONFIRMED)
        data.setdefault("historical_start_known", True)
        data.setdefault("reason", f"continuous-stroke start read from {key}")
        return data
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return StrokeOrigin(
            y=float(value[0]), x=float(value[1]), status=CONFIRMED,
            historical_start_known=True,
            reason=f"continuous-stroke start read from {key} as [y, x]")
    return None


def source_fingerprint(raw_or_bytes: dict | str | bytes | Path) -> str:
    if isinstance(raw_or_bytes, (str, Path)):
        path = Path(raw_or_bytes)
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if is_file:
            data = path.read_bytes()
        elif isinstance(raw_or_bytes, str):
            data = raw_or_bytes.encode("utf-8")
        else:
            data = b""
    elif isinstance(raw_or_bytes, bytes):
        data = raw_or_bytes
    else:
        data = json.dumps(_jsonable(raw_or_bytes), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Capacity back-calculation
# --------------------------------------------------------------------------

def _alpha(value: Any) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "pm"):
        return _alpha(value.pm)
    if isinstance(value, dict):
        if "pm" in value:
            return _alpha(value["pm"])
        for key in ("alpha", "a", "held_alpha", "volume"):
            if key in value:
                scalar = _number(value[key])
                if scalar is not None:
                    return scalar
        value = value.get("rgba", value.get("premultiplied", 0.0))
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 1:
        return float(arr[0])
    if arr.size >= 4:
        return float(arr[3])
    return float(np.sum(arr))


def _first_value(event: dict, keys: Iterable[str]) -> Any:
    for key in keys:
        if key in event and event[key] is not None:
            return event[key]
    return None


def history_events(history: Any, stroke_id: str | int | None = None) -> list[dict]:
    if history is None:
        return []
    if hasattr(history, "step_log"):
        events = list(history.step_log)
    elif isinstance(history, dict):
        events = list(history.get("step_log") or history.get("events")
                      or history.get("strokes") or [])
    else:
        events = list(history)
    normalized = []
    for event in events:
        if hasattr(event, "to_dict"):
            event = event.to_dict()
        if not isinstance(event, dict):
            continue
        if stroke_id is not None and str(
                event.get("stroke_id", stroke_id)) != str(stroke_id):
            continue
        normalized.append(event)
    return normalized


def _origin_from_events(events: list[dict],
                        origin: StrokeOrigin | dict | None) -> StrokeOrigin:
    if isinstance(origin, StrokeOrigin):
        return origin
    if isinstance(origin, dict):
        return StrokeOrigin(**origin)
    if not events:
        return StrokeOrigin(
            status=PENDING,
            historical_start_known=False,
            reason="no historical stroke event is available to mark an origin",
        )
    first = events[0]
    point_index = first.get("point_index", first.get("stamp"))
    return StrokeOrigin(
        event_index=0,
        point_index=None if point_index is None else int(point_index),
        stroke_id=first.get("stroke_id"),
        layer=first.get("layer"),
        y=_number(first.get("y")),
        x=_number(first.get("x")),
        status=MARKED,
        historical_start_known=False,
        reason=("fallback origin at first replayable event; earlier pickup "
                "and the real continuous-stroke start cannot be reconstructed"),
    )


def infer_capacity_bounds(
    history: Any,
    *,
    stroke_id: str | int | None = None,
    initial_volume: float | None = None,
    origin: StrokeOrigin | dict | None = None,
    historical_start_known: bool | None = None,
    tolerance: float = 1e-10,
) -> CapacityInference:
    """Back-calculate capacity bounds from recorded transport events.

    Each event should contain premultiplied ``picked`` and actual reservoir
    ``debit`` amounts.  ``pickup_lost`` plus ``reservoir_before`` can identify
    an exact capacity at a saturation event.
    """
    events = history_events(history, stroke_id)
    aggregate_only = False
    if not events and stroke_id is None:
        ledger = getattr(history, "ledger", None)
        if ledger is None and isinstance(history, dict):
            ledger = history.get("ledger")
        if isinstance(ledger, dict) and ledger.get("picked") is not None:
            debit = ledger.get("debit", ledger.get("debited",
                                                   ledger.get("issued")))
            if _alpha(ledger.get("picked")) or _alpha(debit):
                events = [{
                    "picked": ledger.get("picked"),
                    "debit": debit,
                    "pickup_lost": ledger.get("pickup_lost"),
                }]
                aggregate_only = True
    marker = _origin_from_events(events, origin)
    if historical_start_known is not None:
        marker.historical_start_known = bool(historical_start_known)
        if marker.historical_start_known and events:
            marker.status = CONFIRMED
            marker.reason = "history starts at the recorded continuous-stroke start"
    if not events:
        if not marker.historical_start_known:
            marker.status = PENDING
        return CapacityInference(
            event_count=0,
            origin=marker,
            applicability="no_history",
            assumptions=["no transport events; capacity cannot be bounded"],
        )

    cumulative = [0.0]
    exact_values: list[float] = []
    saturated = False
    for event in events:
        picked = _alpha(_first_value(event, (
            "picked", "picked_pm", "accepted_pickup", "pickup",
            "picked_alpha", "pickup_alpha")))
        debit_value = _first_value(event, (
            "debit", "debited", "reservoir_debit", "actual_debit",
            "debit_alpha", "debited_alpha", "paint_used", "used", "issued"))
        debit = _alpha(debit_value)
        cumulative.append(cumulative[-1] + picked - debit)

        lost_value = _first_value(event, (
            "pickup_lost", "rejected_pickup", "pickup_rejected",
            "pickup_lost_alpha", "rejected_pickup_alpha", "lost_pickup"))
        lost = _alpha(lost_value)
        before = _first_value(event, (
            "reservoir_before", "held_before", "before_reservoir", "before"))
        held_before = _alpha(before) if before is not None else None
        if lost > tolerance and held_before is not None:
            saturated = True
            exact_values.append(held_before + picked)

    min_cum = min(cumulative)
    max_cum = max(cumulative)
    minimum_initial_load = max(0.0, -min_cum)
    minimum_capacity = max_cum - min_cum

    fixed_capacity = None
    feasible = True
    assumptions = [
        "amounts are premultiplied alpha volumes",
        "the marked origin is the zero point for cumulative net transport",
    ]
    if initial_volume is not None:
        h0 = max(0.0, float(initial_volume))
        held_path = [h0 + value for value in cumulative]
        feasible = min(held_path) >= -tolerance
        fixed_capacity = max(held_path)
        assumptions.append("caller supplied the reservoir volume at the origin")

    exact_capacity = None
    if exact_values:
        exact_capacity = float(np.median(exact_values))
        if max(exact_values) - min(exact_values) > max(tolerance, 1e-8):
            feasible = False
            assumptions.append("saturation events imply inconsistent capacities")

    if aggregate_only:
        applicability = "aggregate_total_only_weak_lower_bound"
        assumptions.append(
            "only cumulative ledger totals were available; the intermediate "
            "peak is unknown, so the bound is intentionally weak"
        )
    elif exact_capacity is not None and feasible:
        applicability = "exact_at_recorded_saturation"
    elif initial_volume is not None and feasible:
        applicability = "minimum_for_fixed_origin_load"
    else:
        applicability = "lower_bound_only"
        assumptions.append(
            "history without a rejected-pickup saturation event cannot "
            "determine the true larger capacity"
        )

    marker.status = DERIVED if marker.historical_start_known else MARKED
    return CapacityInference(
        event_count=len(events),
        origin=marker,
        minimum_capacity=float(max(0.0, minimum_capacity)),
        minimum_initial_load=float(minimum_initial_load),
        fixed_initial_capacity=(None if fixed_capacity is None
                                else float(fixed_capacity)),
        exact_capacity=exact_capacity,
        saturated=saturated,
        feasible=feasible,
        aggregate_only=aggregate_only,
        applicability=applicability,
        assumptions=assumptions,
    )


# --------------------------------------------------------------------------
# Nib restoration and brush migration
# --------------------------------------------------------------------------

def restore_nib_by_capacity(
    nib_color: Any,
    capacity: float,
    *,
    alpha_mode: str = "capacity",
) -> Reservoir:
    """Create the reference full reservoir corresponding to an old nib color.

    ``alpha_mode='capacity'`` fills the whole capacity using the nib's straight
    RGB.  ``alpha_mode='stored_alpha'`` interprets RGBA alpha as the stored
    paint volume.  ``alpha_mode='fraction'`` treats alpha as a fill fraction.
    """
    cap = max(0.0, float(capacity))
    reservoir = Reservoir(cap)
    color = parse_rgba_color(nib_color)
    if color is None or cap == 0.0:
        return reservoir
    r, g, b, a = color
    if alpha_mode == "capacity":
        volume = cap
    elif alpha_mode == "stored_alpha":
        volume = min(cap, a)
    elif alpha_mode == "fraction":
        volume = cap * a
    else:
        raise ValueError("alpha_mode must be capacity, stored_alpha or fraction")
    reservoir.pm = np.array([r * volume, g * volume, b * volume, volume],
                            dtype=np.float64)
    return reservoir


def _geometry_from_raw(raw: dict) -> dict[str, Any]:
    key, value = _find_key(raw, _RADIUS_KEYS)
    radius = _number(value)
    if key and key.endswith("diameter") and radius is not None:
        radius *= 0.5
    geometry: dict[str, Any] = {}
    if radius is not None and radius > 0:
        geometry["radius"] = radius
    for key in ("shape", "supersample", "sigma_scale"):
        if key in raw:
            geometry[key] = raw[key]
    return geometry


def _unwrap_project_archive(raw: dict,
                            brush_id: str | None) -> tuple[dict, str | None, dict | None]:
    brushes = raw.get("brushes")
    if isinstance(brushes, list) and brushes:
        selected_index = 0
        if brush_id is not None:
            for index, item in enumerate(brushes):
                if isinstance(item, dict) and str(
                        item.get("id") or item.get("brush_id")) == str(brush_id):
                    selected_index = index
                    break
        else:
            active = raw.get("active_brush_id")
            for index, item in enumerate(brushes):
                if isinstance(item, dict) and str(
                        item.get("id") or item.get("brush_id")) == str(active):
                    selected_index = index
                    break
        chosen = brushes[selected_index]
        if not isinstance(chosen, dict):
            raise TypeError("project brush entries must be objects")
        resolved = str(chosen.get("id") or chosen.get("brush_id")
                       or brush_id or selected_index)
        return chosen, resolved, raw

    if not isinstance(brushes, dict) or not brushes:
        brush = raw.get("brush")
        if isinstance(brush, dict) and any(
                key in brush for key in _COLOR_KEYS + _CAPACITY_KEYS):
            return brush, brush_id, raw
        return raw, brush_id, None

    selected = str(brush_id or raw.get("active_brush_id")
                    or next(iter(brushes)))
    if selected not in brushes:
        raise KeyError(f"brush {selected!r} not found in project archive")
    chosen = brushes[selected]
    if not isinstance(chosen, dict):
        raise TypeError(f"project brush {selected!r} must be an object")
    return chosen, selected, raw


def migrate_legacy_brush(
    source: dict | str | Path,
    *,
    brush_id: str | None = None,
    history: Any = None,
    stroke_id: str | int | None = None,
    reference_capacity: float | None = None,
    provisional_capacity: float | None = None,
    default_humidity: float | None = None,
    config_overrides: dict | None = None,
    origin: StrokeOrigin | dict | None = None,
    historical_start_known: bool | None = None,
    source_path: str | Path | None = None,
    source_sha256: str | None = None,
) -> MigrationResult:
    """Read a legacy tip-color brush without treating missing fields as zero."""
    path = None
    if isinstance(source, (str, Path)) and not isinstance(source, dict):
        candidate = Path(source)
        try:
            source_is_file = candidate.is_file()
        except OSError:
            source_is_file = False
        if source_is_file:
            path = candidate
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        else:
            raw = json.loads(str(source))
    else:
        raw = copy.deepcopy(dict(source))

    if source_path is not None:
        path = Path(source_path)
    digest = source_sha256 or source_fingerprint(raw)
    project_raw = raw
    raw, project_brush_id, project_raw = _unwrap_project_archive(raw, brush_id)
    brush_id = str(project_brush_id or brush_id or raw.get("id")
                   or raw.get("brush_id") or raw.get("uuid")
                   or "legacy-brush")
    record = MigrationRecord(
        brush_id=brush_id,
        source_path=str(path) if path is not None else (
            str(source_path) if source_path is not None else None),
        source_sha256=digest,
    )
    if project_raw is not None:
        available = project_raw.get("brushes")
        if isinstance(available, dict):
            brush_ids: Any = list(available)
        elif isinstance(available, list):
            brush_ids = [item.get("id", index) if isinstance(item, dict)
                         else index for index, item in enumerate(available)]
        else:
            brush_ids = [brush_id]
        record.event("unwrapped_project_archive", {
            "selected_brush_id": brush_id,
            "active_brush_id": project_raw.get("active_brush_id"),
            "brush_ids": brush_ids,
        })

    tip_key, tip_raw = _find_key(raw, _COLOR_KEYS)
    try:
        tip_color = parse_rgba_color(tip_raw) if tip_key else None
    except Exception as exc:  # malformed archive remains inspectable
        tip_color = None
        record.decide(FieldDecision(
            "tip_color", None, PENDING, tip_key,
            f"could not parse tip color: {exc}", needs_confirmation=True))
    else:
        record.decide(FieldDecision(
            "tip_color", tip_color,
            CONFIRMED if tip_color is not None else PENDING,
            tip_key,
            "stored legacy nib color" if tip_color is not None
            else "legacy archive has no readable tip color",
            needs_confirmation=tip_color is None))

    humidity_key, humidity, humidity_note = _read_humidity(raw)
    if humidity is not None and 0.0 <= humidity <= 1.0 and humidity_note is None:
        humidity_decision = FieldDecision(
            "humidity", humidity, CONFIRMED, humidity_key,
            "humidity was explicitly stored")
    elif (humidity is not None and 0.0 <= humidity <= 1.0
          and humidity_note is not None):
        humidity_decision = FieldDecision(
            "humidity", humidity, CONFIRMED, humidity_key, humidity_note)
    else:
        if humidity_key and humidity is None:
            reason = "stored humidity is not a valid [0, 1] number"
        elif humidity_note:
            reason = humidity_note
        else:
            reason = ("legacy tip-only archive does not store humidity; "
                      "it remains unconfirmed rather than being inferred")
        humidity_decision = FieldDecision(
            "humidity", None, PENDING, humidity_key, reason,
            provisional_value=default_humidity,
            needs_confirmation=True)
    record.decide(humidity_decision)

    capacity_key, capacity_raw = _find_key(raw, _CAPACITY_KEYS)
    explicit_capacity = _number(capacity_raw) if capacity_key else None
    explicit_origin = origin if origin is not None else _read_origin(raw)
    inference = infer_capacity_bounds(
        history, stroke_id=stroke_id, origin=explicit_origin,
        historical_start_known=historical_start_known)
    record.capacity_inference = inference
    record.origin = inference.origin
    origin_value = None
    if inference.origin.y is not None and inference.origin.x is not None:
        origin_value = {"y": inference.origin.y, "x": inference.origin.x,
                        "stroke_id": inference.origin.stroke_id,
                        "point_index": inference.origin.point_index}
    record.decide(FieldDecision(
        "stroke_start",
        origin_value,
        inference.origin.status,
        "stroke_history" if inference.event_count else None,
        inference.origin.reason,
        needs_confirmation=not inference.origin.historical_start_known,
    ))

    capacity_decision: FieldDecision
    config_capacity: float | None = None
    initial_pm = None
    if capacity_key is not None:
        if capacity_raw is None or explicit_capacity is None or explicit_capacity < 0:
            capacity_decision = FieldDecision(
                "capacity", None, PENDING, capacity_key,
                "stored capacity is null or not a valid non-negative number",
                needs_confirmation=True)
        else:
            capacity_decision = FieldDecision(
                "capacity", explicit_capacity, CONFIRMED, capacity_key,
                ("archive explicitly stores zero capacity"
                 if explicit_capacity == 0.0
                 else "capacity was explicitly stored"))
            config_capacity = explicit_capacity
    else:
        minimum = inference.minimum_capacity
        exact = inference.exact_capacity
        if exact is not None:
            capacity_decision = FieldDecision(
                "capacity", exact, DERIVED, None,
                "capacity recovered from a recorded saturation event",
                lower_bound=minimum, upper_bound=exact)
            config_capacity = exact
        else:
            provisional = (provisional_capacity if provisional_capacity is not None
                           else reference_capacity)
            capacity_decision = FieldDecision(
                "capacity", None, PENDING, None,
                "capacity key was never stored; history gives only a bound"
                if minimum is not None
                else "capacity key was never stored and no history is available",
                lower_bound=minimum,
                provisional_value=provisional,
                needs_confirmation=True)
            if provisional is not None:
                config_capacity = float(provisional)
    record.decide(capacity_decision)

    geometry = _geometry_from_raw(raw)
    config = None
    if config_capacity is not None:
        overrides = dict(config_overrides or {})
        nib_alpha_mode = str(overrides.pop("nib_alpha_mode", "capacity"))
        overrides.setdefault("capacity", config_capacity)
        overrides.setdefault("humidity", humidity_decision.runtime_value)
        config = SmudgeConfig(**overrides)
        if tip_color is not None:
            restored = restore_nib_by_capacity(
                tip_color, config_capacity, alpha_mode=nib_alpha_mode)
            initial_pm = tuple(float(x) for x in restored.pm)

    if reference_capacity is not None and tip_color is not None:
        ref = restore_nib_by_capacity(tip_color, float(reference_capacity))
        record.event("reference_nib_restoration", {
            "capacity": float(reference_capacity),
            "initial_pm": ref.pm.tolist(),
            "meaning": "full reservoir restored by the supplied capacity",
        })

    return MigrationResult(
        brush_id=brush_id,
        legacy=raw,
        tip_color=tip_color,
        decisions=record.decisions,
        record=record,
        config=config,
        initial_pm=initial_pm,
        geometry=geometry,
    )


def migration_record_from_file(path: str | Path, **kwargs) -> MigrationResult:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return migrate_legacy_brush(raw, source_path=path,
                                source_sha256=source_fingerprint(path.read_bytes()),
                                **kwargs)


migrate_brush = migrate_legacy_brush
convert_legacy_brush = migrate_legacy_brush
backcalculate_capacity_bounds = infer_capacity_bounds
estimate_capacity_bounds = infer_capacity_bounds
capacity_from_history = infer_capacity_bounds
load_legacy_brush_file = migration_record_from_file
ConversionRecord = MigrationRecord
MigrationReport = MigrationRecord
ConversionReport = MigrationRecord
BrushMigration = MigrationResult
MigratedBrush = MigrationResult
ModernBrush = MigrationResult
CapacityBound = CapacityInference
CapacityEstimate = CapacityInference
CapacityReport = CapacityInference
OriginMarker = StrokeOrigin


class LegacyBrush:
    """Small object handle for callers that prefer an old-brush type."""

    def __init__(self, source: dict | str | Path | None = None, **fields: Any):
        if source is None:
            self.raw = {}
        elif isinstance(source, dict):
            self.raw = copy.deepcopy(source)
        else:
            path = Path(source)
            try:
                source_is_file = path.is_file()
            except OSError:
                source_is_file = False
            self.raw = (json.loads(path.read_text(encoding="utf-8"))
                        if source_is_file else json.loads(str(source)))
        self.raw.update(_jsonable(fields))

    @classmethod
    def open(cls, path: str | Path) -> "LegacyBrush":
        return cls(Path(path))

    @property
    def tip_color(self):
        _, value = _find_key(self.raw, _COLOR_KEYS)
        return parse_rgba_color(value) if value is not None else None

    @property
    def humidity(self) -> float | None:
        _, value = _find_key(self.raw, _HUMIDITY_KEYS)
        return _number(value)

    @property
    def capacity(self) -> float | None:
        key, value = _find_key(self.raw, _CAPACITY_KEYS)
        return _number(value) if key is not None else None

    @property
    def has_capacity_key(self) -> bool:
        key, _ = _find_key(self.raw, _CAPACITY_KEYS)
        return key is not None

    def reservoir(self, *, alpha_mode: str = "capacity") -> Reservoir:
        capacity = self.capacity if self.capacity is not None else 0.0
        return restore_nib_by_capacity(self.tip_color, capacity,
                                       alpha_mode=alpha_mode)

    def migrate(self, **kwargs) -> MigrationResult:
        return migrate_legacy_brush(self.raw, **kwargs)


OldBrush = LegacyBrush
