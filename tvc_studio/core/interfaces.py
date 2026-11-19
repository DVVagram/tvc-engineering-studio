"""Stable interfaces between the fifteen subsystems.

Subsystem boundaries are expressed as :class:`typing.Protocol` classes so the
rest of the application codes against capability, not against a concrete
vendor library. Concrete implementations live in their own packages:

===============  =============================================  ========================
Interface        Implemented by                                 Package
===============  =============================================  ========================
GeometryBackend  OCCTGeometry (Open CASCADE via OCP)             ``tvc_studio.cad.occt_backend``
ConstraintSolver NumericConstraintSolver (SciPy)                  ``tvc_studio.assembly.constraints``
DynamicsBackend  SequentialImpulseWorld                          ``tvc_studio.mech.dynamics``
AeroModel        CoefficientAero / GeometryAero                 ``tvc_studio.physics.aero``
ProjectStore     JsonProjectStore                                ``tvc_studio.project.io``
===============  =============================================  ========================
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .math3d import MassProperties, Transform


class Provenance(str, enum.Enum):
    """Where a physical number came from. Never silently upgraded."""

    CAD_DERIVED = "cad_derived"            # computed from exact B-rep + material
    MEASURED = "measured"                  # user measured real hardware
    MANUFACTURER = "manufacturer"          # datasheet / vendor data
    ESTIMATED = "estimated"                # engineering estimate, uncertain
    USER_OVERRIDE = "user_override"        # deliberately typed over a derived value
    UNKNOWN = "unknown"                    # not characterised

    def label(self) -> str:
        return {
            "cad_derived": "CAD-derived",
            "measured": "Measured",
            "manufacturer": "Manufacturer data",
            "estimated": "Estimated (unverified)",
            "user_override": "User override",
            "unknown": "Not characterised",
        }[self.value]


class GeometryRepr(str, enum.Enum):
    """Separate representations, per Part 2.1 of the specification."""

    EXACT = "exact"          # B-rep solid used for mass properties and booleans
    VISUAL = "visual"        # tessellation used for drawing
    COLLISION = "collision"  # convex/decomposed mesh used by contact detection
    SIMULATION = "simulation"  # geometry handed to the mechanical solver


class TessellationQuality(str, enum.Enum):
    COARSE = "coarse"
    NORMAL = "normal"
    FINE = "fine"

    def deflection(self, size: float) -> float:
        base = max(size, 1e-3)
        return {
            TessellationQuality.COARSE: base * 0.02,
            TessellationQuality.NORMAL: base * 0.005,
            TessellationQuality.FINE: base * 0.001,
        }[self]


@dataclass
class Mesh:
    """Triangle mesh + optional edge overlay, used for all rendering."""

    vertices: np.ndarray  # (n, 3) float
    triangles: np.ndarray  # (m, 3) int
    normals: np.ndarray | None = None
    edges: np.ndarray | None = None  # (k, 2) index pairs for wireframe overlay

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
        self.triangles = np.asarray(self.triangles, dtype=np.int32).reshape(-1, 3)

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.vertices.size == 0 or self.triangles.size == 0

    def transformed(self, transform: Transform) -> "Mesh":
        R = transform.rotation.matrix()
        verts = self.vertices @ R.T + transform.translation
        normals = None
        if self.normals is not None:
            normals = self.normals @ R.T
        return Mesh(verts, self.triangles.copy(), normals, None if self.edges is None else self.edges.copy())

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.is_empty:
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


@runtime_checkable
class GeometryBackend(Protocol):
    """Exact-geometry capability required by the CAD and assembly systems."""

    name: str

    def box(self, sx: float, sy: float, sz: float) -> object: ...
    def cylinder(self, radius: float, height: float) -> object: ...
    def boolean_cut(self, base: object, tool: object) -> object: ...
    def mass_properties(self, shape: object, density: float) -> MassProperties: ...
    def tessellate(self, shape: object, quality: TessellationQuality) -> Mesh: ...
    def export_step(self, shape: object, path: str) -> None: ...
    def import_step(self, path: str) -> list[object]: ...


@dataclass
class Measurement:
    """One measured quantity with its unit and the entities it came from."""

    kind: str
    value: float
    unit: str
    detail: str = ""
    entities: tuple[str, ...] = ()

    def formatted(self, digits: int = 3) -> str:
        return f"{self.kind}: {self.value:.{digits}f} {self.unit}"


@runtime_checkable
class ProjectStore(Protocol):
    def save(self, project: object, path: str) -> None: ...
    def load(self, path: str) -> object: ...


@dataclass
class Diagnostic:
    """A solver/validator message with actionable detail."""

    severity: str  # "info" | "warning" | "error"
    code: str
    message: str
    subject: str = ""

    def __str__(self) -> str:
        tag = {"info": "INFO", "warning": "WARN", "error": "ERROR"}.get(self.severity, "INFO")
        return f"[{tag}] {self.code}: {self.message}"


class SupportLevel(str, enum.Enum):
    """Honest capability status attached to every feature and panel."""

    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    PLANNED = "planned"
    UNSUPPORTED = "unsupported"

    def glyph(self) -> str:
        return {
            "implemented": "●",
            "partial": "◐",
            "planned": "○",
            "unsupported": "✕",
        }[self.value]


def rotation_to_matrix(q) -> np.ndarray:
    return q.matrix() if hasattr(q, "matrix") else np.asarray(q, dtype=float)


def identity_inertia() -> np.ndarray:
    return np.zeros((3, 3), dtype=float)


def require_finite(values: Sequence[float], what: str) -> None:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} contains non-finite values")
