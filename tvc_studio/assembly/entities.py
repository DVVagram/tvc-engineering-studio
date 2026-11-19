"""Entity references extracted from CAD geometry for constraint creation.

The UI selects faces/edges/vertices in the viewport; this module turns those selections
into :class:`EntityRef` values with correct local points and directions, and enumerates
sensible default entities (origin, principal axes, detected holes) for a component.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.math3d import as_vec3, unit, vec3
from .constraints import EntityKind, EntityRef


@dataclass
class FaceReference:
    component_id: str
    index: int
    centre: np.ndarray
    normal: np.ndarray
    area: float = 0.0
    is_cylindrical: bool = False
    radius: float = 0.0
    axis_point: np.ndarray | None = None

    def to_entity(self) -> EntityRef:
        if self.is_cylindrical and self.axis_point is not None:
            return EntityRef(self.component_id, EntityKind.AXIS, self.axis_point,
                             self.normal, f"face {self.index} axis")
        return EntityRef(self.component_id, EntityKind.FACE, self.centre, self.normal,
                         f"face {self.index}")


@dataclass
class EdgeReference:
    component_id: str
    index: int
    start: np.ndarray
    end: np.ndarray
    radius: float = 0.0
    is_circular: bool = False

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.start + self.end)

    @property
    def direction(self) -> np.ndarray:
        d = self.end - self.start
        n = np.linalg.norm(d)
        return d / n if n > 1e-12 else vec3(0, 0, 1)

    def to_entity(self) -> EntityRef:
        return EntityRef(self.component_id, EntityKind.EDGE, self.midpoint, self.direction,
                         f"edge {self.index}")


@dataclass
class VertexReference:
    component_id: str
    index: int
    point: np.ndarray

    def to_entity(self) -> EntityRef:
        return EntityRef(self.component_id, EntityKind.VERTEX, self.point, (0, 0, 1),
                         f"vertex {self.index}")


class EntityResolver:
    """Resolves geometry selections into constraint entities.

    ``geometry`` is an object exposing ``faces(part_id)``, ``edges(part_id)`` and
    ``vertices(part_id)`` returning lists of the reference dataclasses above, which the
    OCCT backend provides from the exact shape.
    """

    def __init__(self, geometry, parts) -> None:
        self.geometry = geometry
        self.parts = parts

    def default_entities(self, component_id: str) -> list[EntityRef]:
        entities = [
            EntityRef(component_id, EntityKind.ORIGIN, (0, 0, 0), (0, 0, 1), "origin"),
            EntityRef(component_id, EntityKind.AXIS, (0, 0, 0), (1, 0, 0), "X axis"),
            EntityRef(component_id, EntityKind.AXIS, (0, 0, 0), (0, 1, 0), "Y axis"),
            EntityRef(component_id, EntityKind.AXIS, (0, 0, 0), (0, 0, 1), "Z axis"),
        ]
        comp = self.parts.component(component_id) if hasattr(self.parts, "component") else None
        part_id = comp.part_id if comp is not None else component_id
        try:
            for face in self.geometry.faces(part_id):
                if face.is_cylindrical:
                    entities.append(face.to_entity())
        except Exception:
            pass
        return entities

    def face_entity(self, component_id: str, part_id: str, face_index: int) -> EntityRef:
        for face in self.geometry.faces(part_id):
            if face.index == face_index:
                return face.to_entity()
        raise KeyError(f"face {face_index} not found on {part_id}")

    def edge_entity(self, component_id: str, part_id: str, edge_index: int) -> EntityRef:
        for edge in self.geometry.edges(part_id):
            if edge.index == edge_index:
                return edge.to_entity()
        raise KeyError(f"edge {edge_index} not found on {part_id}")

    def vertex_entity(self, component_id: str, part_id: str, vertex_index: int) -> EntityRef:
        for vertex in self.geometry.vertices(part_id):
            if vertex.index == vertex_index:
                return vertex.to_entity()
        raise KeyError(f"vertex {vertex_index} not found on {part_id}")


def mate_suggestions(component_a: str, component_b: str,
                     face_a: FaceReference | None = None,
                     face_b: FaceReference | None = None) -> list[EntityRef]:
    """Suggest likely mate entities from two selected faces (UI helper)."""
    out: list[EntityRef] = []
    if face_a is not None:
        out.append(face_a.to_entity())
    if face_b is not None:
        out.append(face_b.to_entity())
    return out
