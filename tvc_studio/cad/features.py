"""Parametric feature definitions.

A part is a feature history. Each feature consumes the shape produced by the previous one
and returns a new exact shape, so edits re-run the history in order (Part 2.2). Feature
parameters are plain data, which is what gets serialised into the project file.

Operations that need the kernel raise :class:`CadError`; the rebuild collects those into
diagnostics and keeps the part marked as failed instead of silently producing a wrong shape.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import CadError
from ..core.math3d import as_vec3, vec3


def _fid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


@dataclass
class BuildContext:
    backend: Any
    document: Any
    shape: Any = None
    diagnostics: list[str] = field(default_factory=list)


class Feature:
    """Base class for all features."""

    type: str = "feature"
    label: str = "Feature"

    def __init__(self, id: str | None = None, name: str = "", suppressed: bool = False) -> None:
        self.id = id or _fid(self.type)
        self.name = name or self.label
        self.suppressed = suppressed

    # -- lifecycle ---------------------------------------------------------
    def apply(self, ctx: BuildContext):  # pragma: no cover - abstract
        raise NotImplementedError

    def parameters(self) -> dict:
        return {}

    def set_parameters(self, values: dict) -> None:
        for key, value in values.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "suppressed": self.suppressed,
            "parameters": self.parameters(),
        }

    def summary(self) -> str:
        return self.label

    @staticmethod
    def from_dict(d: dict) -> "Feature":
        feature_type = d.get("type")
        cls = FEATURE_REGISTRY.get(feature_type)
        if cls is None:
            raise CadError(f"unknown feature type in project file: {feature_type}")
        feature = cls.from_parameters(d.get("parameters", {}))
        feature.id = d.get("id", feature.id)
        feature.name = d.get("name", feature.name)
        feature.suppressed = bool(d.get("suppressed", False))
        return feature

    @classmethod
    def from_parameters(cls, params: dict) -> "Feature":
        return cls(**params)


# --------------------------------------------------------------------------- #
# Primitive solids
# --------------------------------------------------------------------------- #
class BoxFeature(Feature):
    type = "box"
    label = "Box"

    def __init__(self, sx: float = 0.02, sy: float = 0.02, sz: float = 0.02,
                 origin=(0.0, 0.0, 0.0), centered: bool = False, operation: str = "new",
                 id: str | None = None, name: str = "", suppressed: bool = False) -> None:
        super().__init__(id, name or f"Box {sx*1000:.1f}x{sy*1000:.1f}x{sz*1000:.1f} mm", suppressed)
        self.sx, self.sy, self.sz = float(sx), float(sy), float(sz)
        self.origin = as_vec3(origin)
        self.centered = bool(centered)
        self.operation = operation

    def parameters(self) -> dict:
        return {"sx": self.sx, "sy": self.sy, "sz": self.sz,
                "origin": self.origin.tolist(), "centered": self.centered,
                "operation": self.operation}

    def apply(self, ctx: BuildContext):
        shape = ctx.backend.box(self.sx, self.sy, self.sz, self.origin, self.centered)
        return _combine(ctx, shape, self.operation)


class CylinderFeature(Feature):
    type = "cylinder"
    label = "Cylinder"

    def __init__(self, radius: float = 0.01, height: float = 0.02, origin=(0.0, 0.0, 0.0),
                 axis=(0.0, 0.0, 1.0), operation: str = "new", id: str | None = None,
                 name: str = "", suppressed: bool = False) -> None:
        super().__init__(id, name or f"Cylinder ⌀{radius*2000:.1f} x {height*1000:.1f} mm", suppressed)
        self.radius = float(radius)
        self.height = float(height)
        self.origin = as_vec3(origin)
        self.axis = as_vec3(axis)
        self.operation = operation

    def parameters(self) -> dict:
        return {"radius": self.radius, "height": self.height, "origin": self.origin.tolist(),
                "axis": self.axis.tolist(), "operation": self.operation}

    def apply(self, ctx: BuildContext):
        shape = ctx.backend.cylinder(self.radius, self.height, self.origin, self.axis)
        return _combine(ctx, shape, self.operation)


class ConeFeature(Feature):
    type = "cone"
    label = "Cone"

    def __init__(self, radius1: float = 0.02, radius2: float = 0.01, height: float = 0.03,
                 origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), operation: str = "new",
                 id: str | None = None, name: str = "", suppressed: bool = False) -> None:
        super().__init__(id, name or "Cone", suppressed)
        self.radius1, self.radius2, self.height = float(radius1), float(radius2), float(height)
        self.origin = as_vec3(origin)
        self.axis = as_vec3(axis)
        self.operation = operation

    def parameters(self) -> dict:
        return {"radius1": self.radius1, "radius2": self.radius2, "height": self.height,
                "origin": self.origin.tolist(), "axis": self.axis.tolist(),
                "operation": self.operation}

    def apply(self, ctx: BuildContext):
        shape = ctx.backend.cone(self.radius1, self.radius2, self.height, self.origin, self.axis)
        return _combine(ctx, shape, self.operation)


class SphereFeature(Feature):
    type = "sphere"
    label = "Sphere"

    def __init__(self, radius: float = 0.01, origin=(0.0, 0.0, 0.0), operation: str = "new",
                 id: str | None = None, name: str = "", suppressed: bool = False) -> None:
        super().__init__(id, name or "Sphere", suppressed)
        self.radius = float(radius)
        self.origin = as_vec3(origin)
        self.operation = operation

    def parameters(self) -> dict:
        return {"radius": self.radius, "origin": self.origin.tolist(), "operation": self.operation}

    def apply(self, ctx: BuildContext):
        shape = ctx.backend.sphere(self.radius, self.origin)
        return _combine(ctx, shape, self.operation)


class TorusFeature(Feature):
    type = "torus"
    label = "Torus"

    def __init__(self, major_radius: float = 0.02, minor_radius: float = 0.004,
                 origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), operation: str = "new",
                 id: str | None = None, name: str = "", suppressed: bool = False) -> None:
        super().__init__(id, name or "Torus", suppressed)
        self.major_radius = float(major_radius)
        self.minor_radius = float(minor_radius)
        self.origin = as_vec3(origin)
        self.axis = as_vec3(axis)
        self.operation = operation

    def parameters(self) -> dict:
        return {"major_radius": self.major_radius, "minor_radius": self.minor_radius,
                "origin": self.origin.tolist(), "axis": self.axis.tolist(),
                "operation": self.operation}

    def apply(self, ctx: BuildContext):
        shape = ctx.backend.torus(self.major_radius, self.minor_radius, self.origin, self.axis)
        return _combine(ctx, shape, self.operation)


class TubeFeature(Feature):
    type = "tube"
    label = "Tube"

    def __init__(self, inner_radius: float = 0.019, outer_radius: float = 0.020,
                 height: float = 0.20, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                 operation: str = "new", id: str | None = None, name: str = "",
                 suppressed: bool = False) -> None:
        super().__init__(id, name or "Tube", suppressed)
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.height = float(height)
        self.origin = as_vec3(origin)
        self.axis = as_vec3(axis)
        self.operation = operation

    def parameters(self) -> dict:
        return {"inner_radius": self.inner_radius, "outer_radius": self.outer_radius,
                "height": self.height, "origin": self.origin.tolist(),
                "axis": self.axis.tolist(), "operation": self.operation}

    def apply(self, ctx: BuildContext):
        shape = ctx.backend.tube(self.inner_radius, self.outer_radius, self.height,
                                 self.origin, self.axis)
        return _combine(ctx, shape, self.operation)


# --------------------------------------------------------------------------- #
# Sketch-driven features
# --------------------------------------------------------------------------- #
class ExtrudeFeature(Feature):
    type = "extrude"
    label = "Extrude"

    def __init__(self, sketch_id: str = "", length: float = 0.02, symmetric: bool = False,
                 reverse: bool = False, operation: str = "new", id: str | None = None,
                 name: str = "", suppressed: bool = False) -> None:
        super().__init__(id, name or "Extrude", suppressed)
        self.sketch_id = sketch_id
        self.length = float(length)
        self.symmetric = bool(symmetric)
        self.reverse = bool(reverse)
        self.operation = operation

    def parameters(self) -> dict:
        return {"sketch_id": self.sketch_id, "length": self.length,
                "symmetric": self.symmetric, "reverse": self.reverse,
                "operation": self.operation}

    def apply(self, ctx: BuildContext):
        sketch = ctx.document.sketch(self.sketch_id)
        if sketch is None:
            raise CadError(f"extrude references missing sketch {self.sketch_id}")
        loops = sketch.profile_loops()
        if not loops:
            raise CadError("sketch has no closed profile to extrude")
        shape = ctx.backend.extrude_profile(loops, self.length, sketch.plane_origin,
                                            sketch.plane_normal, self.symmetric, self.reverse)
        return _combine(ctx, shape, self.operation)


class RevolveFeature(Feature):
    type = "revolve"
    label = "Revolve"

    def __init__(self, sketch_id: str = "", angle: float = 2 * math.pi,
                 axis_origin=(0.0, 0.0, 0.0), axis_dir=(0.0, 0.0, 1.0),
                 operation: str = "new", id: str | None = None, name: str = "",
                 suppressed: bool = False) -> None:
        super().__init__(id, name or "Revolve", suppressed)
        self.sketch_id = sketch_id
        self.angle = float(angle)
        self.axis_origin = as_vec3(axis_origin)
        self.axis_dir = as_vec3(axis_dir)
        self.operation = operation

    def parameters(self) -> dict:
        return {"sketch_id": self.sketch_id, "angle": self.angle,
                "axis_origin": self.axis_origin.tolist(), "axis_dir": self.axis_dir.tolist(),
                "operation": self.operation}

    def apply(self, ctx: BuildContext):
        sketch = ctx.document.sketch(self.sketch_id)
        if sketch is None:
            raise CadError(f"revolve references missing sketch {self.sketch_id}")
        loops = sketch.profile_loops()
        if not loops:
            raise CadError("sketch has no closed profile to revolve")
        # take the outer loop, flattened to points (exact lines; arcs discretised for the
        # revolve profile, which OCCT requires as a planar polygon wire)
        points = _loop_to_points(loops[0], segments=48)
        shape = ctx.backend.revolve_profile(points, self.angle, self.axis_origin, self.axis_dir,
                                           sketch.plane_origin, sketch.plane_normal)
        return _combine(ctx, shape, self.operation)


class HoleFeature(Feature):
    type = "hole"
    label = "Hole"

    def __init__(self, position=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), diameter: float = 0.003,
                 depth: float = 0.006, through: bool = True, pre_threaded: bool = True,
                 thread_designation: str = "", id: str | None = None, name: str = "",
                 suppressed: bool = False) -> None:
        super().__init__(id, name or "Hole", suppressed)
        self.position = as_vec3(position)
        self.axis = as_vec3(axis)
        self.diameter = float(diameter)
        self.depth = float(depth)
        self.through = bool(through)
        self.pre_threaded = bool(pre_threaded)
        self.thread_designation = thread_designation

    def parameters(self) -> dict:
        return {"position": self.position.tolist(), "axis": self.axis.tolist(),
                "diameter": self.diameter, "depth": self.depth, "through": self.through,
                "pre_threaded": self.pre_threaded,
                "thread_designation": self.thread_designation}

    def apply(self, ctx: BuildContext):
        if ctx.shape is None:
            raise CadError("a hole needs an existing solid to cut into")
        return ctx.backend.hole(ctx.shape, self.position, self.axis, self.diameter,
                                self.depth, self.through)

    def summary(self) -> str:
        return f"Hole ⌀{self.diameter*1000:.2f} mm{' through' if self.through else f' x {self.depth*1000:.1f} mm'}"


class FilletFeature(Feature):
    type = "fillet"
    label = "Fillet"

    def __init__(self, radius: float = 0.001, edges: list[int] | None = None,
                 id: str | None = None, name: str = "", suppressed: bool = False) -> None:
        super().__init__(id, name or "Fillet", suppressed)
        self.radius = float(radius)
        self.edges = list(edges or [])

    def parameters(self) -> dict:
        return {"radius": self.radius, "edges": list(self.edges)}

    def apply(self, ctx: BuildContext):
        if ctx.shape is None:
            raise CadError("fillet needs an existing solid")
        if not self.edges:
            raise CadError("fillet has no edges selected")
        return ctx.backend.fillet(ctx.shape, self.edges, self.radius)


class ChamferFeature(Feature):
    type = "chamfer"
    label = "Chamfer"

    def __init__(self, distance: float = 0.001, edges: list[int] | None = None,
                 id: str | None = None, name: str = "", suppressed: bool = False) -> None:
        super().__init__(id, name or "Chamfer", suppressed)
        self.distance = float(distance)
        self.edges = list(edges or [])

    def parameters(self) -> dict:
        return {"distance": self.distance, "edges": list(self.edges)}

    def apply(self, ctx: BuildContext):
        if ctx.shape is None:
            raise CadError("chamfer needs an existing solid")
        if not self.edges:
            raise CadError("chamfer has no edges selected")
        return ctx.backend.chamfer(ctx.shape, self.edges, self.distance)


class PatternLinearFeature(Feature):
    type = "pattern_linear"
    label = "Linear pattern"

    def __init__(self, direction=(1.0, 0.0, 0.0), count: int = 2, spacing: float = 0.01,
                 id: str | None = None, name: str = "", suppressed: bool = False) -> None:
        super().__init__(id, name or "Linear pattern", suppressed)
        self.direction = as_vec3(direction)
        self.count = int(count)
        self.spacing = float(spacing)

    def parameters(self) -> dict:
        return {"direction": self.direction.tolist(), "count": self.count, "spacing": self.spacing}

    def apply(self, ctx: BuildContext):
        if ctx.shape is None:
            raise CadError("pattern needs an existing solid")
        return ctx.backend.pattern_linear(ctx.shape, self.direction, self.count, self.spacing)


class PatternPolarFeature(Feature):
    type = "pattern_polar"
    label = "Polar pattern"

    def __init__(self, count: int = 4, angle: float = 2 * math.pi,
                 axis_origin=(0.0, 0.0, 0.0), axis_dir=(0.0, 0.0, 1.0),
                 id: str | None = None, name: str = "", suppressed: bool = False) -> None:
        super().__init__(id, name or "Polar pattern", suppressed)
        self.count = int(count)
        self.angle = float(angle)
        self.axis_origin = as_vec3(axis_origin)
        self.axis_dir = as_vec3(axis_dir)

    def parameters(self) -> dict:
        return {"count": self.count, "angle": self.angle,
                "axis_origin": self.axis_origin.tolist(), "axis_dir": self.axis_dir.tolist()}

    def apply(self, ctx: BuildContext):
        if ctx.shape is None:
            raise CadError("pattern needs an existing solid")
        return ctx.backend.pattern_polar(ctx.shape, self.axis_origin, self.axis_dir,
                                         self.count, self.angle)


class ImportFeature(Feature):
    """Imported exact geometry. The BREP payload makes the project self-contained."""

    type = "import"
    label = "Imported geometry"

    def __init__(self, source_name: str = "", fmt: str = "step", brep: str = "",
                 operation: str = "new", id: str | None = None, name: str = "",
                 suppressed: bool = False) -> None:
        super().__init__(id, name or f"Import ({source_name or fmt})", suppressed)
        self.source_name = source_name
        self.fmt = fmt
        self.brep = brep
        self.operation = operation

    def parameters(self) -> dict:
        return {"source_name": self.source_name, "fmt": self.fmt, "brep": self.brep,
                "operation": self.operation}

    def apply(self, ctx: BuildContext):
        if not self.brep:
            raise CadError("imported feature has no embedded geometry")
        data = ctx.backend.decode_bytes(self.brep)
        shape = ctx.backend.shape_from_bytes(data)
        return _combine(ctx, shape, self.operation)


class ThreadFeature(Feature):
    type = "thread"
    label = "Detailed thread"

    def __init__(self, thread_designation: str = "M3", length: float = 0.01,
                 id: str | None = None, name: str = "", suppressed: bool = False) -> None:
        super().__init__(id, name or f"Thread {thread_designation}", suppressed)
        self.thread_designation = thread_designation
        self.length = float(length)

    def parameters(self) -> dict:
        return {"thread_designation": self.thread_designation, "length": self.length}

    def apply(self, ctx: BuildContext):
        from ..fasteners.catalog import metric_thread, unified_thread
        try:
            thread = metric_thread(self.thread_designation)
        except KeyError:
            thread = unified_thread(self.thread_designation)
        shape = ctx.backend.thread_helix(thread, self.length)
        return _combine(ctx, shape, "new")


FEATURE_REGISTRY: dict[str, type[Feature]] = {
    cls.type: cls for cls in [
        BoxFeature, CylinderFeature, ConeFeature, SphereFeature, TorusFeature, TubeFeature,
        ExtrudeFeature, RevolveFeature, HoleFeature, FilletFeature, ChamferFeature,
        PatternLinearFeature, PatternPolarFeature, ImportFeature, ThreadFeature,
    ]
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _combine(ctx: BuildContext, shape, operation: str):
    if ctx.shape is None:
        if operation != "new":
            raise CadError(f"'{operation}' feature requires an existing body in this part; "
                           "the first feature must use the 'new' operation")
        return shape
    if operation == "new":
        # a second body in the same part is a compound of solids
        return ctx.backend.compound([ctx.shape, shape])
    if operation == "cut":
        return ctx.backend.boolean_cut(ctx.shape, shape)
    if operation == "join":
        return ctx.backend.boolean_fuse(ctx.shape, shape)
    if operation == "intersect":
        return ctx.backend.boolean_common(ctx.shape, shape)
    return ctx.backend.boolean_fuse(ctx.shape, shape)


def _loop_to_points(loop, segments: int = 48) -> list[tuple[float, float]]:
    """Flatten a profile loop into a polyline (used for revolve profiles only)."""
    points: list[tuple[float, float]] = []
    for seg in loop:
        if seg.kind == "line":
            points.append(tuple(seg.start))
        elif seg.kind == "circle":
            cx, cy = seg.centre
            for i in range(segments):
                a = 2 * math.pi * i / segments
                points.append((cx + seg.radius * math.cos(a), cy + seg.radius * math.sin(a)))
        elif seg.kind == "arc":
            cx, cy = seg.centre
            a0 = math.atan2(seg.start[1] - cy, seg.start[0] - cx)
            a1 = math.atan2(seg.end[1] - cy, seg.end[0] - cx)
            if seg.ccw and a1 <= a0:
                a1 += 2 * math.pi
            if not seg.ccw and a1 >= a0:
                a1 -= 2 * math.pi
            steps = max(int(abs(a1 - a0) / (2 * math.pi) * segments), 2)
            for i in range(steps):
                a = a0 + (a1 - a0) * i / steps
                points.append((cx + seg.radius * math.cos(a), cy + seg.radius * math.sin(a)))
    # remove duplicate consecutive points
    cleaned: list[tuple[float, float]] = []
    for p in points:
        if not cleaned or (abs(p[0] - cleaned[-1][0]) > 1e-12 or abs(p[1] - cleaned[-1][1]) > 1e-12):
            cleaned.append(p)
    if len(cleaned) > 1 and abs(cleaned[0][0] - cleaned[-1][0]) < 1e-12 and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-12:
        cleaned.pop()
    return cleaned
