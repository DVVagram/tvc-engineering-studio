"""Scene assembly for the 3D viewport.

The scene is built only from CAD-derived tessellations and assembly placements — no
decorative geometry that is not in the project. Fasteners are drawn from their catalogue
dimensions (a fastener has no modelled solid), and are marked as such in the legend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..assembly.model import Assembly
from ..core.interfaces import TessellationQuality
from ..core.math3d import Transform, as_vec3, vec3
from ..fasteners.catalog import FastenerSpec, HeadType
from ..fasteners.install import HoleReference, InstalledFastener


@dataclass
class SceneItem:
    component_id: str
    name: str
    kind: str                      # "part" | "fastener"
    vertices: np.ndarray
    triangles: np.ndarray
    color: tuple[float, float, float, float]
    edges: np.ndarray | None = None
    center: np.ndarray = field(default_factory=lambda: vec3())
    provenance: str = "cad_derived"
    exploded_vertices: np.ndarray | None = None
    material_name: str = ""

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.vertices.size == 0:
            return vec3(), vec3()
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def parse_color(value: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    text = (value or "#9fb3c8").lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    try:
        r, g, b = (int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except Exception:
        r, g, b = 0.62, 0.70, 0.78
    return (r, g, b, alpha)


def _cylinder(origin, axis, radius: float, height: float, segments: int = 16
              ) -> tuple[np.ndarray, np.ndarray]:
    """A capped cylinder from origin along axis (used for fastener rendering only)."""
    axis = as_vec3(axis)
    norm = float(np.linalg.norm(axis))
    axis = axis / norm if norm > 1e-12 else vec3(0, 0, 1)
    helper = vec3(1, 0, 0) if abs(axis[0]) < 0.9 else vec3(0, 1, 0)
    u = np.cross(axis, helper)
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(axis, u)
    base = as_vec3(origin)
    verts: list[np.ndarray] = []
    tris: list[tuple[int, int, int]] = []
    for i in range(segments):
        a0 = 2 * math.pi * i / segments
        a1 = 2 * math.pi * (i + 1) / segments
        p0 = base + radius * (math.cos(a0) * u + math.sin(a0) * v)
        p1 = base + radius * (math.cos(a1) * u + math.sin(a1) * v)
        q0 = p0 + axis * height
        q1 = p1 + axis * height
        start = len(verts)
        verts.extend([p0, p1, q1, q0])
        tris.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])
        # caps
        cb = len(verts)
        verts.extend([base, p1, p0])
        tris.append((cb, cb + 1, cb + 2))
        ct = len(verts)
        verts.extend([base + axis * height, q0, q1])
        tris.append((ct, ct + 1, ct + 2))
    return np.array(verts), np.array(tris, dtype=np.int32)


def fastener_geometry(fastener: InstalledFastener, spec: FastenerSpec,
                      hole: HoleReference) -> tuple[np.ndarray, np.ndarray]:
    """Cylinder representation of the installed screw plus its head."""
    axis = as_vec3(hole.axis)
    norm = float(np.linalg.norm(axis))
    axis = axis / norm if norm > 1e-12 else vec3(0, 0, 1)
    length = max(fastener.depth or spec.length, 1e-3)
    # start slightly proud of the hole so the head sits on the surface
    head = getattr(spec, "head_diameter", 0.0) or spec.thread.major_diameter * 1.8
    shaft_radius = spec.thread.major_diameter / 2.0
    origin = as_vec3(hole.position) - axis * 0.0005
    v_shaft, t_shaft = _cylinder(origin, axis, shaft_radius, length)
    head_height = getattr(spec, "head_height", 0.0) or 0.0015
    v_head, t_head = _cylinder(origin, axis, head / 2.0, head_height, segments=20)
    offset = len(v_shaft)
    return np.vstack([v_shaft, v_head]), np.vstack([t_shaft, t_head + offset])


class SceneBuilder:
    """Builds renderable items from a project assembly."""

    def __init__(self, quality: TessellationQuality = TessellationQuality.COARSE) -> None:
        self.quality = quality
        self.diagnostics: list[str] = []

    def build(self, assembly: Assembly, parts_repo, materials, fasteners,
              explode: float = 0.0) -> list[SceneItem]:
        self.diagnostics = []
        items: list[SceneItem] = []
        try:
            lo, hi = assembly.bounds(parts_repo)
            center = (as_vec3(lo) + as_vec3(hi)) / 2.0
            span = float(np.linalg.norm(as_vec3(hi) - as_vec3(lo))) or 1.0
        except Exception:
            center, span = vec3(), 0.3

        for comp in assembly.active_components():
            if not comp.part_id:
                continue
            part = parts_repo.get(comp.part_id) if parts_repo else None
            if part is None:
                continue
            try:
                mesh = part.mesh(parts_repo.backend, self.quality)
            except Exception as exc:
                self.diagnostics.append(f"{comp.name}: {exc}")
                continue
            if mesh.is_empty:
                continue
            world = assembly.world_transform(comp.id)
            verts = mesh.vertices @ world.rotation.matrix().T + world.translation
            color = parse_color(comp.color or part.color)
            if not comp.visible:
                continue
            direction = world.translation - center
            norm = float(np.linalg.norm(direction))
            direction = direction / norm if norm > 1e-9 else vec3(0, 0, 1)
            exploded = verts + direction * (explode * span * 0.25)
            material = materials.get(comp.physical.material_id)
            items.append(SceneItem(
                component_id=comp.id, name=comp.name, kind="part",
                vertices=verts, triangles=mesh.triangles.copy(), color=color,
                center=world.translation.copy(), provenance="cad_derived",
                exploded_vertices=exploded, material_name=material.name if material else "",
            ))

        # fasteners: catalogue dimension cylinders, clearly not modelled solids
        holes = {h.id: h for h in assembly.holes.values()}
        for fastener in assembly.fasteners.values():
            if not fastener.installed:
                continue
            spec = fasteners.get(fastener.fastener_id)
            hole = holes.get(fastener.hole_id)
            if spec is None or hole is None:
                continue
            verts, tris = fastener_geometry(fastener, spec, hole)
            items.append(SceneItem(
                component_id=f"fastener:{fastener.id}", name=f"{spec.name} in {hole.label}",
                kind="fastener", vertices=verts, triangles=tris,
                color=(0.75, 0.76, 0.78, 1.0), center=as_vec3(hole.position),
                provenance="catalogue", exploded_vertices=verts.copy(),
                material_name="fastener steel",
            ))
        return items


def section_filter(items: list[SceneItem], axis: int, offset: float) -> dict[str, list[int]]:
    """Triangle indices to keep for a section view.

    The cut is per-triangle (no capping surface), which is stated in the viewport status.
    """
    keep: dict[str, list[int]] = {}
    for item in items:
        if item.vertices.size == 0 or item.triangles.size == 0:
            continue
        tris = item.triangles
        centroids = item.vertices[tris].mean(axis=1)
        mask = centroids[:, axis] <= offset
        keep[item.component_id] = list(np.nonzero(mask)[0])
    return keep
