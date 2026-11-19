"""Collision detection and interference analysis.

Two levels, deliberately separated:

* **Static interference / clearance** uses the exact B-rep kernel (boolean common volume,
  minimum distance). This is what the engineering report uses, because it is exact.
* **Dynamic contact** uses convex proxies built from the collision mesh with a
  separating-axis test (SAT) and sequential impulses in the dynamics solver. It is a
  rigid-body approximation: no compliance, no deformation. The UI labels it as such and
  pairs that are designed to touch can be filtered out.

Never claimed: contact stress, wear or plastic deformation from these collisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..core.interfaces import Mesh
from ..core.math3d import Transform, as_vec3, unit


@dataclass
class ContactInfo:
    """One contact point between two bodies."""

    body_a: str
    body_b: str
    normal: np.ndarray        # world normal, pointing from B towards A
    point: np.ndarray         # world contact point
    penetration: float        # metres, positive = overlapping


@dataclass
class InterferenceResult:
    volume: float             # m^3, exact boolean common volume
    is_interfering: bool
    minimum_distance: float   # m, 0 when interfering
    detail: str = ""
    exact: bool = True

    def to_dict(self) -> dict:
        return {
            "volume": self.volume,
            "is_interfering": self.is_interfering,
            "minimum_distance": self.minimum_distance,
            "detail": self.detail,
            "exact": self.exact,
        }


class ConvexProxy:
    """Convex hull of a mesh, in body-local coordinates, for SAT contact."""

    __slots__ = ("vertices", "faces", "normals", "edges")

    def __init__(self, vertices: np.ndarray) -> None:
        verts = np.asarray(vertices, dtype=float).reshape(-1, 3)
        if len(verts) < 4:
            self.vertices = verts
            self.faces = np.zeros((0, 3), dtype=int)
            self.normals = np.zeros((0, 3))
            self.edges = np.zeros((0, 2), dtype=int)
            return
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(verts)
            self.vertices = verts[hull.vertices]
            remap = {old: new for new, old in enumerate(hull.vertices)}
            faces = []
            for simplex in hull.simplices:
                if all(int(i) in remap for i in simplex):
                    faces.append([remap[int(i)] for i in simplex])
            self.faces = np.array(faces, dtype=int) if faces else np.zeros((0, 3), dtype=int)
            # unique edge pairs
            edges = set()
            for simplex in hull.simplices:
                for i, j in combinations(simplex, 2):
                    a, b = int(i), int(j)
                    if a in remap and b in remap:
                        edges.add(tuple(sorted((remap[a], remap[b]))))
            self.edges = np.array(sorted(edges), dtype=int) if edges else np.zeros((0, 2), dtype=int)
        except Exception:
            # degenerate point cloud: fall back to its bounding box corners
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            self.vertices = np.array([[x, y, z] for x in (lo[0], hi[0])
                                      for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            self.faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                                   [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                                   [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
            self.edges = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4],
                                   [0, 4], [1, 5], [2, 6], [3, 7]])
        # outward face normals (hull faces are outward-oriented by scipy)
        normals = []
        for face in self.faces:
            if len(face) < 3:
                normals.append((0.0, 0.0, 1.0))
                continue
            p0, p1, p2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            normals.append(n / nn if nn > 1e-15 else np.array([0.0, 0.0, 1.0]))
        self.normals = np.array(normals) if normals else np.zeros((0, 3))

    @staticmethod
    def from_mesh(mesh: Mesh) -> "ConvexProxy":
        return ConvexProxy(mesh.vertices)

    def world_vertices(self, transform: Transform) -> np.ndarray:
        R = transform.rotation.matrix()
        return self.vertices @ R.T + transform.translation

    @property
    def is_valid(self) -> bool:
        return len(self.vertices) >= 4 and len(self.faces) >= 4


def _project(points: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    dots = points @ axis
    return float(dots.min()), float(dots.max())


def sat_contact(a: ConvexProxy, pose_a: Transform, b: ConvexProxy, pose_b: Transform,
                margin: float = 0.0) -> ContactInfo | None:
    """Separating-axis test between two convex proxies with a penetration estimate."""
    if not a.is_valid or not b.is_valid:
        return None
    va = a.world_vertices(pose_a)
    vb = b.world_vertices(pose_b)
    R_a = pose_a.rotation.matrix()
    R_b = pose_b.rotation.matrix()
    axes: list[np.ndarray] = []
    for n in a.normals:
        axes.append(unit(R_a @ n))
    for n in b.normals:
        axes.append(unit(R_b @ n))
    for ea in a.edges[:8]:
        wa = R_a @ (a.vertices[ea[1]] - a.vertices[ea[0]])
        for eb in b.edges[:8]:
            wb = R_b @ (b.vertices[eb[1]] - b.vertices[eb[0]])
            c = np.cross(wa, wb)
            norm = np.linalg.norm(c)
            if norm > 1e-9:
                axes.append(c / norm)
    if not axes:
        return None
    best_depth = float("inf")
    best_axis = None
    for axis in axes:
        amin, amax = _project(va, axis)
        bmin, bmax = _project(vb, axis)
        overlap = min(amax, bmax) - max(amin, bmin)
        if overlap < -margin:
            return None
        if overlap < best_depth:
            best_depth = overlap
            best_axis = axis
    if best_axis is None:
        return None
    # orient the normal from B towards A
    centre_a = va.mean(axis=0)
    centre_b = vb.mean(axis=0)
    if float(np.dot(centre_a - centre_b, best_axis)) < 0:
        best_axis = -best_axis
    # contact point: deepest vertex of A along -normal (approximation)
    depths = (va - centre_b) @ best_axis
    point = va[int(np.argmin(depths))]
    return ContactInfo("", "", best_axis, point, max(best_depth, 0.0))


class CollisionFilter:
    """Pairs that are allowed to touch without generating contact reports."""

    def __init__(self) -> None:
        self._allowed: set[tuple[str, str]] = set()

    def allow(self, body_a: str, body_b: str) -> None:
        self._allowed.add(tuple(sorted((body_a, body_b))))

    def disallow(self, body_a: str, body_b: str) -> None:
        self._allowed.discard(tuple(sorted((body_a, body_b))))

    def is_allowed(self, body_a: str, body_b: str) -> bool:
        return tuple(sorted((body_a, body_b))) in self._allowed

    def to_dict(self) -> dict:
        return {"allowed": [list(p) for p in sorted(self._allowed)]}

    @staticmethod
    def from_dict(d: dict) -> "CollisionFilter":
        filt = CollisionFilter()
        for pair in d.get("allowed", []):
            filt.allow(str(pair[0]), str(pair[1]))
        return filt


def static_interference(backend, shape_a, shape_b, transform_a: Transform | None = None,
                        transform_b: Transform | None = None) -> InterferenceResult:
    """Exact interference analysis using the CAD kernel."""
    if backend is None:
        raise ValueError("static interference requires the CAD backend")
    sa = backend.transform(shape_a, transform_a) if transform_a is not None else shape_a
    sb = backend.transform(shape_b, transform_b) if transform_b is not None else shape_b
    volume = backend.common_volume(sa, sb)
    distance = 0.0
    if volume <= 1e-15:
        try:
            distance = backend.distance(sa, sb)
        except Exception:
            distance = float("nan")
    return InterferenceResult(
        volume=volume,
        is_interfering=volume > 1e-12,
        minimum_distance=distance,
        detail=("solids overlap" if volume > 1e-12 else f"clearance {distance*1000:.3f} mm"),
        exact=True,
    )


def clearance_report(backend, shapes: dict[str, object], transforms: dict[str, Transform],
                     threshold: float = 0.001) -> list[dict]:
    """Pairwise exact clearances between components, sorted by tightest first."""
    rows: list[dict] = []
    ids = list(shapes.keys())
    for i, j in combinations(ids, 2):
        try:
            result = static_interference(backend, shapes[i], shapes[j], transforms.get(i), transforms.get(j))
        except Exception as exc:
            rows.append({"a": i, "b": j, "error": str(exc)})
            continue
        status = "interference" if result.is_interfering else (
            "warning" if result.minimum_distance < threshold else "clear")
        rows.append({
            "a": i,
            "b": j,
            "volume": result.volume,
            "distance": result.minimum_distance,
            "status": status,
        })
    rows.sort(key=lambda r: (r.get("status") == "clear", r.get("distance", 0.0)))
    return rows
