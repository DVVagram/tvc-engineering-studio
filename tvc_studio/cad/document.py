"""Part documents and the part repository.

A :class:`PartDocument` owns a feature history and one or more sketches, plus the material
and property blocks that follow the geometry. :class:`PartRepository` caches exact shapes
and tessellations per representation so the viewport, mass properties and export all read
from one truth.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from ..components.properties import MechanicalProperties, PhysicalProperties
from ..core.errors import CadError
from ..core.interfaces import GeometryRepr, Mesh, TessellationQuality
from ..core.math3d import MassProperties
from ..fasteners.install import HoleReference
from .features import BuildContext, Feature, HoleFeature, ImportFeature
from .sketch import Sketch


def _pid(prefix: str = "part") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class RebuildResult:
    success: bool
    shape: object | None
    diagnostics: list[str] = field(default_factory=list)
    failed_feature: str = ""


@dataclass
class PartDocument:
    id: str = field(default_factory=lambda: _pid("part"))
    name: str = "Part"
    features: list[Feature] = field(default_factory=list)
    sketches: list[Sketch] = field(default_factory=list)
    material_id: str = "pla"
    physical: PhysicalProperties = field(default_factory=PhysicalProperties)
    mechanical: MechanicalProperties = field(default_factory=MechanicalProperties)
    notes: str = ""
    color: str = "#9fb3c8"
    #: representation caches (not serialised)
    _shape: object | None = field(default=None, repr=False, compare=False)
    _meshes: dict = field(default_factory=dict, repr=False, compare=False)
    _last_rebuild: RebuildResult | None = field(default=None, repr=False, compare=False)
    _dirty: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.physical.material_id == "unknown":
            self.physical.material_id = self.material_id

    # -------------------------------------------------------------- sketches
    def add_sketch(self, sketch: Sketch) -> Sketch:
        self.sketches.append(sketch)
        self.invalidate()
        return sketch

    def sketch(self, sketch_id: str) -> Sketch | None:
        for s in self.sketches:
            if s.id == sketch_id:
                return s
        return None

    # -------------------------------------------------------------- features
    def add_feature(self, feature: Feature) -> Feature:
        self.features.append(feature)
        self.invalidate()
        return feature

    def remove_feature(self, feature_id: str) -> bool:
        for index, feature in enumerate(self.features):
            if feature.id == feature_id:
                del self.features[index]
                self.invalidate()
                return True
        return False

    def move_feature(self, feature_id: str, new_index: int) -> bool:
        for index, feature in enumerate(self.features):
            if feature.id == feature_id:
                item = self.features.pop(index)
                self.features.insert(max(0, min(new_index, len(self.features))), item)
                self.invalidate()
                return True
        return False

    def invalidate(self) -> None:
        self._dirty = True
        self._shape = None
        self._meshes.clear()

    # --------------------------------------------------------------- rebuild
    def rebuild(self, backend) -> RebuildResult:
        ctx = BuildContext(backend=backend, document=self, shape=None)
        diagnostics: list[str] = []
        failed = ""
        for feature in self.features:
            if feature.suppressed:
                continue
            try:
                ctx.shape = feature.apply(ctx)
            except CadError as exc:
                diagnostics.append(f"{feature.name}: {exc}")
                failed = feature.id
                ctx.shape = None
                break
            except Exception as exc:  # kernel-level surprise
                diagnostics.append(f"{feature.name}: unexpected kernel error: {exc}")
                failed = feature.id
                ctx.shape = None
                break
        result = RebuildResult(
            success=ctx.shape is not None,
            shape=ctx.shape,
            diagnostics=diagnostics,
            failed_feature=failed,
        )
        if result.success:
            self._shape = ctx.shape
        self._last_rebuild = result
        self._dirty = False
        return result

    def shape(self, backend):
        if self._dirty or self._shape is None:
            result = self.rebuild(backend)
            if not result.success:
                raise CadError(
                    f"part '{self.name}' failed to rebuild: " + "; ".join(result.diagnostics or ["no geometry"])
                )
        return self._shape

    # ---------------------------------------------------------------- meshes
    def mesh(self, backend, quality: TessellationQuality = TessellationQuality.NORMAL,
             representation: GeometryRepr = GeometryRepr.VISUAL) -> Mesh:
        key = (quality.value, representation.value)
        if key in self._meshes:
            return self._meshes[key]
        shape = self.shape(backend)
        mesh = backend.tessellate(shape, quality)
        self._meshes[key] = mesh
        return mesh

    def collision_mesh(self, backend) -> Mesh:
        return self.mesh(backend, TessellationQuality.COARSE, GeometryRepr.COLLISION)

    # ------------------------------------------------------- mass properties
    def cad_mass_properties(self, backend, density: float) -> MassProperties:
        return backend.mass_properties(self.shape(backend), density)

    def bounding_box(self, backend):
        return backend.bounding_box(self.shape(backend))

    # ---------------------------------------------------------------- holes
    def holes(self) -> list[HoleReference]:
        holes: list[HoleReference] = []
        for feature in self.features:
            if isinstance(feature, HoleFeature) and not feature.suppressed:
                pre = feature.pre_threaded
                holes.append(HoleReference(
                    id=f"hole_{feature.id}",
                    component_id="",  # filled in by the assembly when the part is placed
                    position=feature.position,
                    axis=feature.axis,
                    diameter=feature.diameter,
                    depth=feature.depth,
                    pre_threaded=pre,
                    material_id=self.material_id,
                    label=feature.name,
                    created_by_feature=feature.id,
                ))
        return holes

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "features": [f.to_dict() for f in self.features],
            "sketches": [s.to_dict() for s in self.sketches],
            "material_id": self.material_id,
            "physical": self.physical.to_dict(),
            "mechanical": self.mechanical.to_dict(),
            "notes": self.notes,
            "color": self.color,
        }

    @staticmethod
    def from_dict(d: dict) -> "PartDocument":
        doc = PartDocument(
            id=str(d.get("id", _pid("part"))),
            name=str(d.get("name", "Part")),
            material_id=str(d.get("material_id", "pla")),
            physical=PhysicalProperties.from_dict(d.get("physical", {})),
            mechanical=MechanicalProperties.from_dict(d.get("mechanical", {})),
            notes=str(d.get("notes", "")),
            color=str(d.get("color", "#9fb3c8")),
        )
        doc.sketches = [Sketch.from_dict(s) for s in d.get("sketches", [])]
        doc.features = [Feature.from_dict(f) for f in d.get("features", [])]
        if doc.physical.material_id in ("unknown", ""):
            doc.physical.material_id = doc.material_id
        return doc

    def summary(self) -> str:
        parts = [self.name, f"{len(self.features)} feature(s)"]
        if self._last_rebuild and not self._last_rebuild.success:
            parts.append("BUILD FAILED")
        return " · ".join(parts)


class PartRepository:
    """Registry of part documents with geometry caches, used by assembly/UI/simulation."""

    def __init__(self, backend=None, materials=None) -> None:
        self.backend = backend
        self.materials = materials
        self.parts: dict[str, PartDocument] = {}

    # -- access ------------------------------------------------------------
    def add(self, part: PartDocument) -> PartDocument:
        self.parts[part.id] = part
        return part

    def create(self, name: str, features: list[Feature] | None = None,
               material_id: str = "pla", sketches: list[Sketch] | None = None) -> PartDocument:
        part = PartDocument(name=name, material_id=material_id)
        part.physical.material_id = material_id
        part.features = list(features or [])
        part.sketches = list(sketches or [])
        return self.add(part)

    def get(self, part_id: str) -> PartDocument | None:
        return self.parts.get(part_id)

    def require(self, part_id: str) -> PartDocument:
        part = self.parts.get(part_id)
        if part is None:
            raise CadError(f"unknown part: {part_id}")
        return part

    def remove(self, part_id: str) -> bool:
        return self.parts.pop(part_id, None) is not None

    # -- geometry ----------------------------------------------------------
    def shape(self, part_id: str):
        part = self.require(part_id)
        return part.shape(self.backend)

    def cad_mass_properties(self, part_id: str, density: float) -> MassProperties:
        part = self.require(part_id)
        return part.cad_mass_properties(self.backend, density)

    def mesh(self, part_id: str, quality: TessellationQuality = TessellationQuality.NORMAL) -> Mesh:
        part = self.require(part_id)
        return part.mesh(self.backend, quality)

    def invalidate(self, part_id: str | None = None) -> None:
        if part_id is None:
            for part in self.parts.values():
                part.invalidate()
        elif part_id in self.parts:
            self.parts[part_id].invalidate()

    def rebuild_all(self) -> dict[str, RebuildResult]:
        return {part_id: part.rebuild(self.backend) for part_id, part in self.parts.items()}

    # -- import ------------------------------------------------------------
    def import_file(self, path: str, name: str | None = None) -> list[PartDocument]:
        """Import STEP/IGES/STL/BREP as new part documents with embedded exact geometry."""
        if self.backend is None:
            raise CadError("CAD backend unavailable")
        ext = path.lower().rsplit(".", 1)[-1]
        loaders = {
            "step": self.backend.import_step,
            "stp": self.backend.import_step,
            "iges": self.backend.import_iges,
            "igs": self.backend.import_iges,
            "stl": self.backend.import_stl,
            "brep": self.backend.import_brep,
        }
        if ext not in loaders:
            raise CadError(f"unsupported import format: .{ext}")
        shapes = loaders[ext](path)
        created: list[PartDocument] = []
        base_name = name or path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        for index, shape in enumerate(shapes):
            payload = self.backend.encode_bytes(self.backend.export_shape_bytes(shape))
            doc = PartDocument(name=base_name if len(shapes) == 1 else f"{base_name} [{index+1}]")
            doc.add_feature(ImportFeature(source_name=base_name, fmt=ext, brep=payload,
                                          name=f"Import {base_name}"))
            self.add(doc)
            created.append(doc)
        return created

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"parts": [p.to_dict() for p in self.parts.values()]}

    def load_dict(self, d: dict) -> None:
        for entry in d.get("parts", []):
            part = PartDocument.from_dict(entry)
            self.parts[part.id] = part

    def holes_for(self, part_id: str, component_id: str) -> list[HoleReference]:
        part = self.require(part_id)
        holes = part.holes()
        for hole in holes:
            hole.component_id = component_id
            hole.id = f"{component_id}:{hole.id}"
        return holes
