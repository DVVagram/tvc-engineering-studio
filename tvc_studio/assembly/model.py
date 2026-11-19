"""Assembly document: component hierarchy, placement, fasteners, configurations.

The assembly is the central source of truth. A component instance references a part
document (exact geometry + properties) and holds a placement transform *relative to its
parent*, so subassembly moves propagate the way an engineer expects.

Grounded components are the datum of the assembly: constraint solving leaves them fixed,
and if nothing is grounded the solver reports it as a diagnostic rather than silently
choosing a reference.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ..components.properties import (
    ComponentConfiguration,
    MechanicalProperties,
    PhysicalProperties,
)
from ..core.errors import AssemblyError
from ..core.math3d import Transform, Quat, as_vec3, vec3
from ..fasteners.install import HoleReference, InstalledFastener


def new_id(prefix: str = "c") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class ComponentInstance:
    """One occurrence of a part (or subassembly) in the assembly tree."""

    id: str
    name: str
    part_id: str | None = None
    parent_id: str | None = None
    transform: Transform = field(default_factory=Transform.identity)
    visible: bool = True
    suppressed: bool = False
    grounded: bool = False
    is_subassembly: bool = False
    color: str = "#8fa3b8"
    opacity: float = 1.0
    physical: PhysicalProperties = field(default_factory=PhysicalProperties)
    mechanical: MechanicalProperties = field(default_factory=MechanicalProperties)
    hardware_type: str = ""      # "", "servo", "motor", "sensor", "board"
    hardware_id: str = ""        # library key of the hardware definition
    metadata: dict = field(default_factory=dict)

    # -- transform helpers -------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.transform.translation

    def set_position(self, p) -> None:
        self.transform.translation = as_vec3(p)

    def rotate_local(self, axis, angle: float) -> None:
        self.transform.rotation = self.transform.rotation * Quat.from_axis_angle(axis, angle)

    def rotate_world(self, axis, angle: float) -> None:
        self.transform.rotation = Quat.from_axis_angle(axis, angle) * self.transform.rotation

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "part_id": self.part_id,
            "parent_id": self.parent_id,
            "transform": self.transform.to_dict(),
            "visible": self.visible,
            "suppressed": self.suppressed,
            "grounded": self.grounded,
            "is_subassembly": self.is_subassembly,
            "color": self.color,
            "opacity": self.opacity,
            "physical": self.physical.to_dict(),
            "mechanical": self.mechanical.to_dict(),
            "hardware_type": self.hardware_type,
            "hardware_id": self.hardware_id,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(d: dict) -> "ComponentInstance":
        return ComponentInstance(
            id=str(d["id"]),
            name=str(d.get("name", "component")),
            part_id=d.get("part_id"),
            parent_id=d.get("parent_id"),
            transform=Transform.from_dict(d.get("transform", {})),
            visible=bool(d.get("visible", True)),
            suppressed=bool(d.get("suppressed", False)),
            grounded=bool(d.get("grounded", False)),
            is_subassembly=bool(d.get("is_subassembly", False)),
            color=str(d.get("color", "#8fa3b8")),
            opacity=float(d.get("opacity", 1.0)),
            physical=PhysicalProperties.from_dict(d.get("physical", {})),
            mechanical=MechanicalProperties.from_dict(d.get("mechanical", {})),
            hardware_type=str(d.get("hardware_type", "")),
            hardware_id=str(d.get("hardware_id", "")),
            metadata=dict(d.get("metadata", {})),
        )


class Assembly:
    """Component tree + constraints + fasteners + configurations."""

    def __init__(self, name: str = "Assembly") -> None:
        self.id = new_id("asm")
        self.name = name
        self.components: dict[str, ComponentInstance] = {}
        self.constraints: list = []          # AssemblyConstraint, typed loosely to avoid cycles
        self.fasteners: dict[str, InstalledFastener] = {}
        self.holes: dict[str, HoleReference] = {}
        self.configurations: dict[str, ComponentConfiguration] = {}
        self.active_configuration: str | None = None
        self.notes: str = ""

    # ------------------------------------------------------------------ tree
    def add_component(self, component: ComponentInstance) -> ComponentInstance:
        if component.id in self.components:
            raise AssemblyError(f"duplicate component id: {component.id}")
        if component.parent_id and component.parent_id not in self.components:
            raise AssemblyError(f"parent component does not exist: {component.parent_id}")
        if component.parent_id:
            self.components[component.parent_id].is_subassembly = True
        # unique display name
        existing = {c.name for c in self.components.values()}
        if component.name in existing:
            base = component.name
            for n in itertools.count(2):
                candidate = f"{base} ({n})"
                if candidate not in existing:
                    component.name = candidate
                    break
        self.components[component.id] = component
        return component

    def create_component(
        self,
        part_id: str | None,
        name: str | None = None,
        parent_id: str | None = None,
        position=(0.0, 0.0, 0.0),
        **kwargs,
    ) -> ComponentInstance:
        comp = ComponentInstance(
            id=kwargs.pop("id", new_id("comp")),
            name=name or part_id or "Component",
            part_id=part_id,
            parent_id=parent_id,
            transform=Transform.identity(),
            **kwargs,
        )
        comp.transform.translation = as_vec3(position)
        return self.add_component(comp)

    def remove_component(self, component_id: str, recursive: bool = True) -> list[str]:
        if component_id not in self.components:
            return []
        if not recursive and any(c.parent_id == component_id for c in self.components.values()):
            raise AssemblyError("component has children; remove them first or use recursive removal")
        removed: list[str] = []
        stack = [component_id]
        while stack:
            cid = stack.pop()
            if cid not in self.components:
                continue
            stack.extend(c.id for c in self.components.values() if c.parent_id == cid)
            removed.append(cid)
            self.components.pop(cid, None)
        # drop constraints and fasteners that referenced removed components
        self.constraints = [
            c for c in self.constraints
            if getattr(getattr(c, "a", None), "component_id", None) not in removed
            and getattr(getattr(c, "b", None), "component_id", None) not in removed
        ]
        for fid, fastener in list(self.fasteners.items()):
            hole = self.holes.get(fastener.hole_id)
            if hole is None or hole.component_id in removed:
                self.fasteners.pop(fid, None)
                if hole is not None:
                    self.holes.pop(hole.id, None)
        return removed

    def children(self, parent_id: str | None) -> list[ComponentInstance]:
        return [c for c in self.components.values() if c.parent_id == parent_id]

    def roots(self) -> list[ComponentInstance]:
        return self.children(None)

    def walk(self, parent_id: str | None = None, depth: int = 0) -> Iterator[tuple[ComponentInstance, int]]:
        for child in self.children(parent_id):
            yield child, depth
            yield from self.walk(child.id, depth + 1)

    def path_to(self, component_id: str) -> list[ComponentInstance]:
        chain: list[ComponentInstance] = []
        current = self.components.get(component_id)
        guard = 0
        while current is not None and guard < 1000:
            chain.append(current)
            current = self.components.get(current.parent_id) if current.parent_id else None
            guard += 1
        return list(reversed(chain))

    def world_transform(self, component_id: str) -> Transform:
        comp = self.components.get(component_id)
        if comp is None:
            raise AssemblyError(f"unknown component: {component_id}")
        t = Transform.identity()
        for node in self.path_to(component_id):
            t = t @ node.transform
        return t

    def set_world_transform(self, component_id: str, world: Transform) -> None:
        """Set a component's world pose, converting to its parent's frame."""
        comp = self.components[component_id]
        if comp.parent_id:
            parent_world = self.world_transform(comp.parent_id)
            comp.transform = parent_world.inverse() @ world
        else:
            comp.transform = world

    def local_to_world_point(self, component_id: str, point) -> np.ndarray:
        return self.world_transform(component_id).apply_point(point)

    # ------------------------------------------------------------- hierarchy
    def reparent(self, component_id: str, new_parent_id: str | None) -> None:
        if component_id == new_parent_id:
            raise AssemblyError("a component cannot be its own parent")
        if new_parent_id and new_parent_id in {c.id for c in self.path_to(component_id)}:
            raise AssemblyError("cannot move a component into its own descendant")
        comp = self.components[component_id]
        world = self.world_transform(component_id)
        comp.parent_id = new_parent_id
        if new_parent_id:
            self.components[new_parent_id].is_subassembly = True
        self.set_world_transform(component_id, world)

    def duplicate(self, component_id: str, with_children: bool = True, name: str | None = None) -> ComponentInstance:
        source = self.components[component_id]
        copy = ComponentInstance.from_dict(source.to_dict())
        copy.id = new_id("comp")
        copy.name = name or f"{source.name} copy"
        copy.parent_id = source.parent_id
        copy.grounded = False
        if copy.parent_id:
            self.components[copy.parent_id].is_subassembly = True
        self.add_component(copy)
        if with_children:
            for child in self.children(component_id):
                self._duplicate_subtree(child, copy.id)
        return copy

    def _duplicate_subtree(self, source: ComponentInstance, new_parent_id: str) -> None:
        copy = ComponentInstance.from_dict(source.to_dict())
        copy.id = new_id("comp")
        copy.parent_id = new_parent_id
        self.components[copy.id] = copy
        for child in self.children(source.id):
            self._duplicate_subtree(child, copy.id)

    def make_subassembly(self, name: str, component_ids: list[str]) -> ComponentInstance:
        """Group existing components under a new subassembly node, preserving world poses."""
        if not component_ids:
            raise AssemblyError("a subassembly needs at least one component")
        parents = {self.components[cid].parent_id for cid in component_ids}
        if len(parents) > 1:
            raise AssemblyError("components must share a parent to be grouped")
        parent_id = parents.pop()
        sub = ComponentInstance(
            id=new_id("sub"),
            name=name,
            parent_id=parent_id,
            is_subassembly=True,
            part_id=None,
        )
        self.add_component(sub)
        for cid in component_ids:
            world = self.world_transform(cid)
            comp = self.components[cid]
            comp.parent_id = sub.id
            self.set_world_transform(cid, world)
        return sub

    def replace_part(self, component_id: str, new_part_id: str) -> None:
        comp = self.components[component_id]
        comp.part_id = new_part_id
        if comp.physical.material_id == "unknown":
            comp.physical.material_id = "unknown"

    # ------------------------------------------------------------- visibility
    def set_visible(self, component_id: str, visible: bool, recursive: bool = False) -> None:
        self.components[component_id].visible = visible
        if recursive:
            for child in self.children(component_id):
                self.set_visible(child.id, visible, recursive=True)

    def set_suppressed(self, component_id: str, suppressed: bool, recursive: bool = False) -> None:
        self.components[component_id].suppressed = suppressed
        if recursive:
            for child in self.children(component_id):
                self.set_suppressed(child.id, suppressed, recursive=True)

    def visible_components(self) -> list[ComponentInstance]:
        out: list[ComponentInstance] = []
        hidden_ancestors: set[str] = set()
        for comp, depth in self.walk():
            parent_hidden = bool(comp.parent_id and (comp.parent_id in hidden_ancestors
                                                     or not self.components[comp.parent_id].visible))
            if not comp.visible:
                hidden_ancestors.add(comp.id)
            if comp.visible and not parent_hidden and not comp.suppressed:
                out.append(comp)
        return out

    def active_components(self) -> list[ComponentInstance]:
        """Components that participate in mass properties and simulation (suppression only)."""
        return [c for c in self.components.values() if not c.suppressed]

    def bounds(self, parts_repo) -> tuple[np.ndarray, np.ndarray]:
        """World-space bounding box of the placed geometry (used by the viewport)."""
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for comp in self.active_components():
            if not comp.part_id or parts_repo is None:
                continue
            part = parts_repo.get(comp.part_id)
            if part is None:
                continue
            try:
                lo, hi = part.bounding_box(parts_repo.backend)
            except Exception:
                continue
            world = self.world_transform(comp.id)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                for z in (lo[2], hi[2])])
            world_corners = np.array([world.apply_point(c) for c in corners])
            lows.append(world_corners.min(axis=0))
            highs.append(world_corners.max(axis=0))
        if not lows:
            return vec3(), vec3(0.1, 0.1, 0.1)
        return np.min(np.array(lows), axis=0), np.max(np.array(highs), axis=0)

    # ------------------------------------------------------------- fasteners
    def add_hole(self, hole: HoleReference) -> HoleReference:
        if hole.component_id not in self.components:
            raise AssemblyError(f"hole references unknown component: {hole.component_id}")
        self.holes[hole.id] = hole
        return hole

    def add_holes(self, holes: list[HoleReference]) -> None:
        for hole in holes:
            self.add_hole(hole)

    def install(self, fastener: InstalledFastener) -> InstalledFastener:
        if fastener.hole_id not in self.holes:
            raise AssemblyError(f"unknown hole: {fastener.hole_id}")
        self.fasteners[fastener.id] = fastener
        return fastener

    def remove_fastener(self, fastener_id: str) -> bool:
        return self.fasteners.pop(fastener_id, None) is not None

    def fasteners_for(self, component_id: str) -> list[InstalledFastener]:
        hole_ids = {h.id for h in self.holes.values() if h.component_id == component_id}
        return [f for f in self.fasteners.values() if f.hole_id in hole_ids]

    def fastener_breakdown(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for fastener in self.fasteners.values():
            counts[fastener.fastener_id] = counts.get(fastener.fastener_id, 0) + 1
        return counts

    # ---------------------------------------------------------- configurations
    def save_configuration(self, name: str) -> ComponentConfiguration:
        config = ComponentConfiguration(
            name=name,
            physical=PhysicalProperties(),
            mechanical=MechanicalProperties(),
            hidden_component_ids=tuple(c.id for c in self.components.values() if not c.visible),
        )
        self.configurations[name] = config
        return config

    def apply_configuration(self, name: str) -> None:
        config = self.configurations.get(name)
        if config is None:
            raise AssemblyError(f"unknown configuration: {name}")
        hidden = set(config.hidden_component_ids)
        for comp in self.components.values():
            comp.visible = comp.id not in hidden
        self.active_configuration = name

    # ------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "components": [c.to_dict() for c in self.components.values()],
            "constraints": [c.to_dict() for c in self.constraints],
            "fasteners": [f.to_dict() for f in self.fasteners.values()],
            "holes": [h.to_dict() for h in self.holes.values()],
            "configurations": [c.to_dict() for c in self.configurations.values()],
            "active_configuration": self.active_configuration,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict, constraint_factory=None) -> "Assembly":
        asm = Assembly(str(d.get("name", "Assembly")))
        asm.id = str(d.get("id", asm.id))
        for entry in d.get("components", []):
            comp = ComponentInstance.from_dict(entry)
            asm.components[comp.id] = comp
        if constraint_factory is not None:
            for entry in d.get("constraints", []):
                asm.constraints.append(constraint_factory(entry))
        for entry in d.get("fasteners", []):
            fastener = InstalledFastener.from_dict(entry)
            asm.fasteners[fastener.id] = fastener
        for entry in d.get("holes", []):
            hole = HoleReference.from_dict(entry)
            asm.holes[hole.id] = hole
        for entry in d.get("configurations", []):
            config = ComponentConfiguration.from_dict(entry)
            asm.configurations[config.name] = config
        asm.active_configuration = d.get("active_configuration")
        asm.notes = str(d.get("notes", ""))
        return asm

    # ------------------------------------------------------------------ stats
    def statistics(self) -> dict:
        return {
            "components": len(self.components),
            "subassemblies": sum(1 for c in self.components.values() if c.is_subassembly),
            "constraints": len(self.constraints),
            "fasteners": len(self.fasteners),
            "holes": len(self.holes),
            "grounded": [c.name for c in self.components.values() if c.grounded],
        }
