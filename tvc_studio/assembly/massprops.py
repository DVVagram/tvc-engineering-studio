"""Assembly mass properties.

Computes component and whole-assembly mass, centre of gravity and inertia tensor using
the parallel-axis theorem, including installed fasteners, then exposes a breakdown per
component so the engineer can see where every gram and every kg*m^2 comes from.

No manual whole-rocket mass entry exists: the flight model consumes these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from ..components.materials import Material, MaterialLibrary
from ..core.errors import MassPropertyError
from ..core.interfaces import Provenance
from ..core.math3d import (
    MassProperties,
    combine_mass_properties,
    parallel_axis,
    transform_mass_properties,
)
from ..fasteners.catalog import FastenerCatalog, FastenerKind
from ..fasteners.install import HoleReference, InstalledFastener
from .model import Assembly, ComponentInstance


@dataclass
class MassRow:
    """One line of the mass/inertia breakdown."""

    name: str
    kind: str                 # component | fastener | subassembly
    mass: float
    percent: float
    com: np.ndarray
    provenance: str
    is_override: bool
    component_id: str = ""
    detail: str = ""

    @property
    def label(self) -> str:
        return f"{self.name}{' (override)' if self.is_override else ''}"


@dataclass
class AssemblyMassResult:
    properties: MassProperties
    rows: list[MassRow] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    @property
    def mass(self) -> float:
        return self.properties.mass

    @property
    def com(self) -> np.ndarray:
        return self.properties.com

    @property
    def inertia(self) -> np.ndarray:
        return self.properties.inertia

    def principal_moments(self) -> tuple[np.ndarray, np.ndarray]:
        return self.properties.principal()

    def material_breakdown(self, materials: MaterialLibrary, assembly: Assembly) -> dict[str, float]:
        totals: dict[str, float] = {}
        for comp in assembly.active_components():
            mat = materials.get(comp.physical.material_id)
            totals[mat.name] = totals.get(mat.name, 0.0) + self._component_mass(comp.id)
        return totals

    def _component_mass(self, component_id: str) -> float:
        for row in self.rows:
            if row.component_id == component_id and row.kind == "component":
                return row.mass
        return 0.0

    def to_dict(self) -> dict:
        return {
            "properties": self.properties.to_dict(),
            "rows": [
                {
                    "name": r.name,
                    "kind": r.kind,
                    "mass": r.mass,
                    "percent": r.percent,
                    "com": np.asarray(r.com).tolist(),
                    "provenance": r.provenance,
                    "is_override": r.is_override,
                    "component_id": r.component_id,
                    "detail": r.detail,
                }
                for r in self.rows
            ],
            "diagnostics": list(self.diagnostics),
        }


class AssemblyMassCalculator:
    """Resolves component properties and folds them into assembly-level properties."""

    def __init__(
        self,
        parts,
        materials: MaterialLibrary,
        fasteners: FastenerCatalog | None = None,
        geometry_resolver: Callable[[ComponentInstance], MassProperties | None] | None = None,
    ) -> None:
        """``parts`` is any object exposing ``cad_mass_properties(part_id)`` and
        ``material_of(part_id)``; decoupled from the CAD implementation so this module
        can be unit tested with analytic solids."""
        self.parts = parts
        self.materials = materials
        self.fasteners = fasteners
        self.geometry_resolver = geometry_resolver

    # -- component ---------------------------------------------------------
    def component_properties(self, comp: ComponentInstance) -> MassProperties:
        material: Material = self.materials.get(comp.physical.material_id)
        cad_mp: MassProperties | None = None
        if comp.part_id and comp.physical.mass_mode.value != "manual":
            if self.geometry_resolver is not None:
                cad_mp = self.geometry_resolver(comp)
            else:
                cad_mp = self.parts.cad_mass_properties(comp.part_id, material.density)
        elif comp.physical.mass_mode.value != "manual":
            cad_mp = None
        mp = comp.physical.resolve(cad_mp, material)
        return transform_mass_properties(mp, self._world_transform(comp))

    def _world_transform(self, comp: ComponentInstance):
        # populated by the calculator during a pass; falls back to local transform
        return getattr(self, "_assembly", None).world_transform(comp.id) if getattr(self, "_assembly", None) else comp.transform

    def fastener_properties(self, fastener: InstalledFastener, assembly: Assembly) -> MassProperties | None:
        if self.fasteners is None:
            return None
        hole = assembly.holes.get(fastener.hole_id)
        if hole is None:
            return None
        try:
            spec = self.fasteners.get(fastener.fastener_id)
        except KeyError:
            return None
        material = self.materials.get(spec.material_id)
        mass = spec.mass
        if spec.kind in (FastenerKind.NUT, FastenerKind.WASHER) and fastener.depth <= 1e-9:
            return None
        total_mass = mass
        if fastener.with_nut:
            nut = next((s for s in self.fasteners.matching_thread(spec.thread)
                        if s.kind is FastenerKind.NUT), None)
            if nut:
                total_mass += nut.mass
        if fastener.with_washer:
            washer = next((s for s in self.fasteners.matching_thread(spec.thread)
                           if s.kind is FastenerKind.WASHER), None)
            if washer:
                total_mass += washer.mass
        if total_mass <= 0:
            return None
        # position along the hole axis, computed from the installed depth
        entry_world = assembly.local_to_world_point(hole.component_id, hole.position)
        axis_world = assembly.world_transform(hole.component_id).apply_vector(hole.axis)
        direction = -axis_world if fastener.reversed_direction else axis_world
        centre = entry_world + direction * (fastener.depth * 0.5)
        # small rod inertia along the screw axis
        length = max(spec.length, 1e-3)
        radius = spec.thread.major_diameter / 2.0
        axial = 0.5 * total_mass * radius * radius
        transverse = total_mass * (3 * radius * radius + length * length) / 12.0
        d = np.asarray(axis_world, dtype=float)
        d = d / (np.linalg.norm(d) or 1.0)
        R = _frame_from_axis(d)
        inertia_local = np.diag([transverse, transverse, axial])
        inertia = R @ inertia_local @ R.T
        return MassProperties(
            mass=total_mass,
            com=centre,
            inertia=inertia,
            provenance=spec.mass_provenance.value,
            frame="world",
        )

    # -- assembly ----------------------------------------------------------
    def calculate(self, assembly: Assembly, include_fasteners: bool = True) -> AssemblyMassResult:
        self._assembly = assembly
        rows: list[MassRow] = []
        diagnostics: list[str] = []
        parts: list[MassProperties] = []

        for comp in assembly.active_components():
            if comp.is_subassembly and not comp.part_id:
                continue
            if comp.part_id is None:
                diagnostics.append(f"'{comp.name}' has no geometry; skipped in mass properties")
                continue
            try:
                mp = self.component_properties(comp)
            except MassPropertyError as exc:
                diagnostics.append(f"'{comp.name}': {exc}")
                continue
            if mp.mass <= 0:
                diagnostics.append(f"'{comp.name}' has zero mass")
            parts.append(mp)
            rows.append(
                MassRow(
                    name=assembly.path_to(comp.id)[-1].name,
                    kind="component",
                    mass=mp.mass,
                    percent=0.0,
                    com=mp.com,
                    provenance=mp.provenance,
                    is_override=comp.physical.is_override(),
                    component_id=comp.id,
                    detail=comp.part_id or "",
                )
            )

        if include_fasteners:
            for fastener in assembly.fasteners.values():
                mp = self.fastener_properties(fastener, assembly)
                if mp is None:
                    continue
                parts.append(mp)
                hole = assembly.holes.get(fastener.hole_id)
                rows.append(
                    MassRow(
                        name=f"Fastener {fastener.fastener_id.split(':')[-1]} in {hole.label if hole else '?'}",
                        kind="fastener",
                        mass=mp.mass,
                        percent=0.0,
                        com=mp.com,
                        provenance=mp.provenance,
                        is_override=False,
                        component_id=hole.component_id if hole else "",
                        detail=fastener.fastener_id,
                    )
                )

        combined = combine_mass_properties(parts)
        total = combined.mass
        for row in rows:
            row.percent = (row.mass / total * 100.0) if total > 0 else 0.0
        if total <= 0:
            diagnostics.append("assembly has no mass: mass properties cannot drive a simulation")
        if combined.provenance != "cad_derived":
            diagnostics.append(
                "assembly mass properties contain manual overrides or estimated values "
                f"(provenance: {combined.provenance})"
            )
        self._assembly = None
        return AssemblyMassResult(properties=combined, rows=rows, diagnostics=diagnostics)

    # -- convenience -------------------------------------------------------
    def swept_mass(self, assembly: Assembly) -> float:
        return self.calculate(assembly).mass

    def inertia_at(self, assembly: Assembly, point) -> np.ndarray:
        result = self.calculate(assembly)
        return parallel_axis(result.inertia, result.mass, result.com - np.asarray(point, dtype=float))

    def compare(self, assemblies: dict[str, Assembly]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, asm in assemblies.items():
            res = self.calculate(asm)
            out[name] = {
                "mass": res.mass,
                "com_x": float(res.com[0]),
                "com_y": float(res.com[1]),
                "com_z": float(res.com[2]),
                "i_xx": float(res.inertia[0, 0]),
                "i_yy": float(res.inertia[1, 1]),
                "i_zz": float(res.inertia[2, 2]),
                "components": float(len(asm.components)),
                "fasteners": float(len(asm.fasteners)),
            }
        return out


def _frame_from_axis(axis: np.ndarray) -> np.ndarray:
    """Orthonormal frame whose third column is ``axis``."""
    z = axis / (np.linalg.norm(axis) or 1.0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x) or 1.0
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def mass_report(assembly: Assembly, calculator: AssemblyMassCalculator) -> str:
    """Human-readable text report (used by the analysis workspace and CSV export)."""
    result = calculator.calculate(assembly)
    lines = [
        f"Assembly: {assembly.name}",
        f"Total mass: {result.mass:.5f} kg",
        f"Centre of gravity (world): X={result.com[0]:+.5f}  Y={result.com[1]:+.5f}  Z={result.com[2]:+.5f} m",
        f"Inertia tensor (kg*m^2, about CoG):",
    ]
    for row in result.inertia:
        lines.append("  [" + "  ".join(f"{v: .6e}" for v in row) + "]")
    vals, axes = result.principal_moments()
    lines.append("Principal moments: " + ", ".join(f"{v:.6e}" for v in vals))
    lines.append("")
    lines.append(f"{'Component':32s} {'Mass (g)':>10s} {'%':>7s}  Provenance")
    for row in sorted(result.rows, key=lambda r: -r.mass):
        lines.append(f"{row.name[:32]:32s} {row.mass*1000:10.2f} {row.percent:6.2f}%  {row.provenance}")
    if result.diagnostics:
        lines.append("")
        lines.append("Diagnostics:")
        lines.extend(f"  - {d}" for d in result.diagnostics)
    return "\n".join(lines)
