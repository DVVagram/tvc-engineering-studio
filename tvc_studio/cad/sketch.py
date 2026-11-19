"""2-D sketches with a dimensional/geometric constraint solver.

Sketches are the parametric source of profile geometry for extrude/revolve features. The
solver treats point coordinates and radii as variables and minimises constraint residuals
with SciPy's least-squares, then reports remaining degrees of freedom so the sketch panel
can say "fully constrained" instead of guessing.

Geometry handed to the kernel keeps exact lines and circular arcs; the helper
:meth:`Sketch.profile_loops` returns segments the OCCT backend turns into a real wire.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from ..core.errors import SketchError
from ..core.interfaces import Diagnostic
from ..core.math3d import vec3


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


@dataclass
class SketchPoint:
    id: str = field(default_factory=lambda: _uid("p"))
    x: float = 0.0
    y: float = 0.0
    fixed: bool = False


@dataclass
class Line:
    id: str = field(default_factory=lambda: _uid("l"))
    p1: str = ""
    p2: str = ""
    construction: bool = False


@dataclass
class Circle:
    id: str = field(default_factory=lambda: _uid("c"))
    centre: str = ""
    radius: float = 0.005
    construction: bool = False


@dataclass
class Arc:
    id: str = field(default_factory=lambda: _uid("a"))
    centre: str = ""
    radius: float = 0.005
    start_angle: float = 0.0
    end_angle: float = math.pi
    construction: bool = False


class SketchConstraintType(str, Enum):
    COINCIDENT = "coincident"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DISTANCE = "distance"
    RADIUS = "radius"
    DIAMETER = "diameter"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    EQUAL = "equal"
    POINT_ON_LINE = "point_on_line"
    ANGLE = "angle"
    CONCENTRIC = "concentric"
    TANGENT = "tangent"
    FIXED = "fixed"


@dataclass
class SketchConstraint:
    id: str = field(default_factory=lambda: _uid("sc"))
    type: SketchConstraintType = SketchConstraintType.COINCIDENT
    entities: tuple[str, ...] = ()
    value: float = 0.0
    #: 0 = hard constraint, >0 = weighted soft constraint (for dragging)
    weight: float = 1.0
    reference: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "entities": list(self.entities),
            "value": self.value,
            "weight": self.weight,
            "reference": self.reference,
        }

    @staticmethod
    def from_dict(d: dict) -> "SketchConstraint":
        return SketchConstraint(
            id=str(d.get("id", _uid("sc"))),
            type=SketchConstraintType(d.get("type", "coincident")),
            entities=tuple(d.get("entities", ())),
            value=float(d.get("value", 0.0)),
            weight=float(d.get("weight", 1.0)),
            reference=bool(d.get("reference", False)),
        )


@dataclass
class ProfileSegment:
    kind: str                 # "line" | "arc"
    start: tuple[float, float]
    end: tuple[float, float]
    radius: float = 0.0
    centre: tuple[float, float] = (0.0, 0.0)
    ccw: bool = True

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "start": list(self.start),
            "end": list(self.end),
            "radius": self.radius,
            "centre": list(self.centre),
            "ccw": self.ccw,
        }


@dataclass
class SketchSolveResult:
    converged: bool
    residual: float
    dof: int
    iterations: int
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def summary(self) -> str:
        state = "fully constrained" if self.dof == 0 else f"{self.dof} DOF free"
        return f"{'solved' if self.converged else 'unsolved'}, {state}, residual {self.residual:.2e}"


class Sketch:
    """A 2-D sketch on the XY plane of its owning part."""

    def __init__(self, name: str = "Sketch") -> None:
        self.id = _uid("sketch")
        self.name = name
        self.points: dict[str, SketchPoint] = {}
        self.lines: dict[str, Line] = {}
        self.circles: dict[str, Circle] = {}
        self.arcs: dict[str, Arc] = {}
        self.constraints: list[SketchConstraint] = []
        self.plane_origin = vec3()
        self.plane_normal = vec3(0, 0, 1)
        self.last_result: SketchSolveResult | None = None

    # ------------------------------------------------------------- creation
    def add_point(self, x: float, y: float, fixed: bool = False) -> SketchPoint:
        p = SketchPoint(x=x, y=y, fixed=fixed)
        self.points[p.id] = p
        return p

    def add_line(self, p1: "SketchPoint | tuple[float, float]", p2: "SketchPoint | tuple[float, float]",
                 construction: bool = False) -> Line:
        a = p1 if isinstance(p1, SketchPoint) else self.add_point(*p1)
        b = p2 if isinstance(p2, SketchPoint) else self.add_point(*p2)
        line = Line(p1=a.id, p2=b.id, construction=construction)
        self.lines[line.id] = line
        return line

    def add_rectangle(self, width: float, height: float, centre: tuple[float, float] = (0.0, 0.0),
                      constrained: bool = True) -> list[Line]:
        cx, cy = centre
        pts = [
            self.add_point(cx - width / 2, cy - height / 2),
            self.add_point(cx + width / 2, cy - height / 2),
            self.add_point(cx + width / 2, cy + height / 2),
            self.add_point(cx - width / 2, cy + height / 2),
        ]
        lines = [
            self.add_line(pts[0], pts[1]),
            self.add_line(pts[1], pts[2]),
            self.add_line(pts[2], pts[3]),
            self.add_line(pts[3], pts[0]),
        ]
        if constrained:
            self.constrain(SketchConstraintType.HORIZONTAL, (lines[0].id,))
            self.constrain(SketchConstraintType.HORIZONTAL, (lines[2].id,))
            self.constrain(SketchConstraintType.VERTICAL, (lines[1].id,))
            self.constrain(SketchConstraintType.VERTICAL, (lines[3].id,))
            self.constrain(SketchConstraintType.DISTANCE, (pts[0].id, pts[1].id), width)
            self.constrain(SketchConstraintType.DISTANCE, (pts[1].id, pts[2].id), height)
            pts[0].fixed = True
        return lines

    def add_circle(self, centre: "SketchPoint | tuple[float, float]", radius: float,
                   construction: bool = False) -> Circle:
        c = centre if isinstance(centre, SketchPoint) else self.add_point(*centre)
        circle = Circle(centre=c.id, radius=radius, construction=construction)
        self.circles[circle.id] = circle
        return circle

    def add_arc(self, centre: "SketchPoint | tuple[float, float]", radius: float,
                start_angle: float, end_angle: float, construction: bool = False) -> Arc:
        c = centre if isinstance(centre, SketchPoint) else self.add_point(*centre)
        arc = Arc(centre=c.id, radius=radius, start_angle=start_angle, end_angle=end_angle,
                  construction=construction)
        self.arcs[arc.id] = arc
        return arc

    def add_circle_circle(self, radius: float) -> Circle:
        return self.add_circle((0.0, 0.0), radius)

    def constrain(self, ctype: SketchConstraintType, entities: Iterable[str],
                  value: float = 0.0, weight: float = 1.0) -> SketchConstraint:
        constraint = SketchConstraint(type=ctype, entities=tuple(entities), value=value, weight=weight)
        self.constraints.append(constraint)
        return constraint

    # -------------------------------------------------------------- solving
    def _variables(self) -> tuple[list[str], np.ndarray]:
        names: list[str] = []
        values: list[float] = []
        for pid, p in self.points.items():
            if p.fixed:
                continue
            names.extend([f"{pid}.x", f"{pid}.y"])
            values.extend([p.x, p.y])
        for cid, c in self.circles.items():
            if self._circle_radius_fixed(cid):
                continue
            names.append(f"{cid}.r")
            values.append(c.radius)
        for aid, a in self.arcs.items():
            if self._arc_radius_fixed(aid):
                continue
            names.append(f"{aid}.r")
            values.append(a.radius)
        return names, np.array(values, dtype=float)

    def _circle_radius_fixed(self, circle_id: str) -> bool:
        return any(c.type in (SketchConstraintType.RADIUS, SketchConstraintType.DIAMETER)
                   and c.entities and c.entities[0] == circle_id and c.reference is False
                   and False for c in self.constraints) or False

    def _arc_radius_fixed(self, arc_id: str) -> bool:
        return any(c.type is SketchConstraintType.RADIUS and c.entities and c.entities[0] == arc_id
                   and c.reference for c in self.constraints)

    def _apply(self, names: list[str], x: np.ndarray) -> dict[str, float]:
        state: dict[str, float] = {}
        for pid, p in self.points.items():
            state[f"{pid}.x"] = p.x
            state[f"{pid}.y"] = p.y
        for cid, c in self.circles.items():
            state[f"{cid}.r"] = c.radius
        for aid, a in self.arcs.items():
            state[f"{aid}.r"] = a.radius
        for name, value in zip(names, x):
            state[name] = float(value)
        return state

    def _residuals(self, state: dict[str, float]) -> np.ndarray:
        out: list[float] = []
        for c in self.constraints:
            if c.reference:
                continue
            w = math.sqrt(max(c.weight, 1e-12))
            r = self._constraint_residual(c, state)
            if r is not None:
                out.extend(w * np.asarray(r, dtype=float).ravel())
        return np.array(out, dtype=float)

    def _constraint_residual(self, c: SketchConstraint, state: dict[str, float]):
        t = c.type
        ent = c.entities

        def pt(pid: str) -> np.ndarray:
            return np.array([state[f"{pid}.x"], state[f"{pid}.y"]])

        def line_points(lid: str) -> tuple[np.ndarray, np.ndarray]:
            line = self.lines[lid]
            return pt(line.p1), pt(line.p2)

        if t is SketchConstraintType.COINCIDENT:
            return pt(ent[0]) - pt(ent[1])
        if t is SketchConstraintType.HORIZONTAL:
            a, b = line_points(ent[0])
            return np.array([a[1] - b[1]])
        if t is SketchConstraintType.VERTICAL:
            a, b = line_points(ent[0])
            return np.array([a[0] - b[0]])
        if t is SketchConstraintType.DISTANCE:
            first, second = ent[0], ent[1]
            if first in self.points and second in self.points:
                d = pt(first) - pt(second)
                return np.array([np.linalg.norm(d) - c.value])
            if first in self.points and second in self.lines:
                a, b = line_points(second)
                p = pt(first)
                edge = b - a
                length = np.linalg.norm(edge) or 1.0
                return np.array([abs(np.cross(edge / length, p - a)) - c.value])
            return np.array([0.0])
        if t is SketchConstraintType.RADIUS:
            key = ent[0]
            return np.array([state[f"{key}.r"] - c.value])
        if t is SketchConstraintType.DIAMETER:
            key = ent[0]
            return np.array([2.0 * state[f"{key}.r"] - c.value])
        if t is SketchConstraintType.PARALLEL:
            a1, b1 = line_points(ent[0])
            a2, b2 = line_points(ent[1])
            d1, d2 = b1 - a1, b2 - a2
            return np.array([np.cross(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) or 1.0)])
        if t is SketchConstraintType.PERPENDICULAR:
            a1, b1 = line_points(ent[0])
            a2, b2 = line_points(ent[1])
            d1, d2 = b1 - a1, b2 - a2
            return np.array([np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) or 1.0)])
        if t is SketchConstraintType.EQUAL:
            a1, b1 = line_points(ent[0])
            a2, b2 = line_points(ent[1])
            return np.array([np.linalg.norm(b1 - a1) - np.linalg.norm(b2 - a2)])
        if t is SketchConstraintType.POINT_ON_LINE:
            a, b = line_points(ent[1])
            p = pt(ent[0])
            edge = b - a
            length = np.linalg.norm(edge) or 1.0
            return np.array([np.cross(edge / length, p - a)])
        if t is SketchConstraintType.ANGLE:
            a1, b1 = line_points(ent[0])
            a2, b2 = line_points(ent[1])
            d1, d2 = b1 - a1, b2 - a2
            cosang = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) or 1.0)
            return np.array([math.acos(float(np.clip(cosang, -1.0, 1.0))) - c.value])
        if t is SketchConstraintType.CONCENTRIC:
            c1 = self.circles.get(ent[0]) or self.arcs.get(ent[0])
            c2 = self.circles.get(ent[1]) or self.arcs.get(ent[1])
            return pt(c1.centre) - pt(c2.centre)
        if t is SketchConstraintType.TANGENT:
            # line-circle tangency: distance from centre to line equals radius
            line_ent = self.lines.get(ent[0])
            circle_ent = self.circles.get(ent[1]) or self.arcs.get(ent[1])
            if line_ent is None:
                line_ent = self.lines.get(ent[1])
                circle_ent = self.circles.get(ent[0]) or self.arcs.get(ent[0])
            a, b = pt(line_ent.p1), pt(line_ent.p2)
            centre = pt(circle_ent.centre)
            edge = b - a
            length = np.linalg.norm(edge) or 1.0
            dist = abs(np.cross(edge / length, centre - a))
            return np.array([dist - state[f"{circle_ent.id}.r"]])
        if t is SketchConstraintType.FIXED:
            # handled by marking points fixed; nothing to solve
            return np.array([0.0])
        return np.array([0.0])

    def solve(self, apply: bool = True, max_iter: int = 200) -> SketchSolveResult:
        names, x0 = self._variables()
        diagnostics: list[Diagnostic] = []
        if not names:
            result = SketchSolveResult(True, 0.0, 0, 0, diagnostics)
            self.last_result = result
            return result
        base_state = self._apply(names, x0)

        def fun(x: np.ndarray) -> np.ndarray:
            state = dict(base_state)
            for name, value in zip(names, x):
                state[name] = float(value)
            return self._residuals(state)

        solution = least_squares(fun, x0, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                 max_nfev=max_iter * len(names))
        residual = float(np.linalg.norm(fun(solution.x)))
        J = solution.jac if solution.jac is not None else np.zeros((0, len(names)))
        rank = int(np.linalg.matrix_rank(J, tol=1e-7)) if J.size else 0
        dof = max(len(names) - rank, 0)
        converged = residual < 1e-9

        if not converged:
            diagnostics.append(Diagnostic("error", "sketch-unsolved",
                                          "Sketch constraints cannot be satisfied; check conflicting dimensions."))
        elif dof == 0:
            diagnostics.append(Diagnostic("info", "sketch-constrained", "Sketch is fully constrained."))
        else:
            diagnostics.append(Diagnostic("warning", "sketch-underconstrained",
                                          f"Sketch has {dof} free degree(s) of freedom."))

        if apply:
            state = dict(base_state)
            for name, value in zip(names, solution.x):
                state[name] = float(value)
            for pid, p in self.points.items():
                p.x = state[f"{pid}.x"]
                p.y = state[f"{pid}.y"]
            for cid, c in self.circles.items():
                c.radius = state[f"{cid}.r"]
            for aid, a in self.arcs.items():
                a.radius = state[f"{aid}.r"]

        result = SketchSolveResult(converged, residual, dof, int(getattr(solution, "nfev", 0)), diagnostics)
        self.last_result = result
        return result

    # ------------------------------------------------------------- profiles
    def profile_loops(self) -> list[list[ProfileSegment]]:
        """Ordered closed loops of non-construction geometry, for the CAD backend."""
        segments: list[tuple[str, str, ProfileSegment]] = []
        for line in self.lines.values():
            if line.construction:
                continue
            p1 = self.points[line.p1]
            p2 = self.points[line.p2]
            a = (p1.x, p1.y)
            b = (p2.x, p2.y)
            segments.append((line.p1, line.p2, ProfileSegment("line", a, b)))
        for arc in self.arcs.values():
            if arc.construction:
                continue
            centre = self.points[arc.centre]
            start = (centre.x + arc.radius * math.cos(arc.start_angle),
                     centre.y + arc.radius * math.sin(arc.start_angle))
            end = (centre.x + arc.radius * math.cos(arc.end_angle),
                   centre.y + arc.radius * math.sin(arc.end_angle))
            # arc endpoints are new nodes unless they coincide with existing point ids
            segments.append((
                f"arc_start:{arc.id}",
                f"arc_end:{arc.id}",
                ProfileSegment("arc", start, end, arc.radius, (centre.x, centre.y),
                               ccw=arc.end_angle > arc.start_angle),
            ))

        if not segments:
            # whole circles are closed loops on their own
            loops: list[list[ProfileSegment]] = []
            for circle in self.circles.values():
                if circle.construction:
                    continue
                centre = self.points[circle.centre]
                loops.append([ProfileSegment(
                    "circle", (centre.x + circle.radius, centre.y), (centre.x + circle.radius, centre.y),
                    circle.radius, (centre.x, centre.y), ccw=True,
                )])
            return loops

        return _chain_segments(segments)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "plane_origin": np.asarray(self.plane_origin).tolist(),
            "plane_normal": np.asarray(self.plane_normal).tolist(),
            "points": [{"id": p.id, "x": p.x, "y": p.y, "fixed": p.fixed} for p in self.points.values()],
            "lines": [{"id": l.id, "p1": l.p1, "p2": l.p2, "construction": l.construction}
                      for l in self.lines.values()],
            "circles": [{"id": c.id, "centre": c.centre, "radius": c.radius,
                         "construction": c.construction} for c in self.circles.values()],
            "arcs": [{"id": a.id, "centre": a.centre, "radius": a.radius,
                      "start_angle": a.start_angle, "end_angle": a.end_angle,
                      "construction": a.construction} for a in self.arcs.values()],
            "constraints": [c.to_dict() for c in self.constraints],
        }

    @staticmethod
    def from_dict(d: dict) -> "Sketch":
        sketch = Sketch(str(d.get("name", "Sketch")))
        sketch.id = str(d.get("id", sketch.id))
        sketch.plane_origin = np.asarray(d.get("plane_origin", [0, 0, 0]), dtype=float)
        sketch.plane_normal = np.asarray(d.get("plane_normal", [0, 0, 1]), dtype=float)
        for p in d.get("points", []):
            sketch.points[p["id"]] = SketchPoint(id=p["id"], x=float(p["x"]), y=float(p["y"]),
                                                 fixed=bool(p.get("fixed", False)))
        for l in d.get("lines", []):
            sketch.lines[l["id"]] = Line(id=l["id"], p1=l["p1"], p2=l["p2"],
                                         construction=bool(l.get("construction", False)))
        for c in d.get("circles", []):
            sketch.circles[c["id"]] = Circle(id=c["id"], centre=c["centre"], radius=float(c["radius"]),
                                             construction=bool(c.get("construction", False)))
        for a in d.get("arcs", []):
            sketch.arcs[a["id"]] = Arc(id=a["id"], centre=a["centre"], radius=float(a["radius"]),
                                       start_angle=float(a["start_angle"]),
                                       end_angle=float(a["end_angle"]),
                                       construction=bool(a.get("construction", False)))
        sketch.constraints = [SketchConstraint.from_dict(c) for c in d.get("constraints", [])]
        return sketch


def _chain_segments(segments: list[tuple[str, str, ProfileSegment]]) -> list[list[ProfileSegment]]:
    """Chain segments into closed loops by shared node ids."""
    adjacency: dict[str, list[int]] = {}
    for index, (a, b, _) in enumerate(segments):
        adjacency.setdefault(a, []).append(index)
        adjacency.setdefault(b, []).append(index)
    used: set[int] = set()
    loops: list[list[ProfileSegment]] = []
    for start_index, (start_node, end_node, seg) in enumerate(segments):
        if start_index in used:
            continue
        loop: list[ProfileSegment] = [seg]
        used.add(start_index)
        node = end_node
        guard = 0
        while node != start_node and guard < len(segments) + 2:
            guard += 1
            candidates = [i for i in adjacency.get(node, []) if i not in used]
            if not candidates:
                break
            choice = candidates[0]
            a, b, next_seg = segments[choice]
            if a == node:
                loop.append(next_seg)
                node = b
            else:
                loop.append(_reverse_segment(next_seg))
                node = a
            used.add(choice)
        if node == start_node and len(loop) >= 1:
            loops.append(loop)
    return loops


def _reverse_segment(seg: ProfileSegment) -> ProfileSegment:
    return ProfileSegment(seg.kind, seg.end, seg.start, seg.radius, seg.centre, not seg.ccw)


def rectangle_sketch(width: float, height: float) -> Sketch:
    sketch = Sketch("Rectangle")
    sketch.add_rectangle(width, height)
    sketch.solve()
    return sketch


def circle_sketch(radius: float) -> Sketch:
    sketch = Sketch("Circle")
    centre = sketch.add_point(0.0, 0.0, fixed=True)
    sketch.add_circle(centre, radius)
    return sketch
