"""Screw installation: hole references, kinematic advance, engagement inspection.

The installed state of a fastener is defined by *one* number: its insertion depth along
the hole axis. Rotation is derived from depth through the thread lead, so the visual
rotation can never disagree with the thread pitch (spec 5.2 / rule 11). Depth is what the
user edits; rotation is what the animation shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.interfaces import Provenance
from ..core.math3d import Transform, Quat, as_vec3, unit, vec3
from .catalog import FastenerSpec, FastenerKind, ThreadSpec
from .thread import EngagementMode, EngagementResult, ThreadEngagementModel


@dataclass
class HoleReference:
    """A place a fastener can be installed.

    ``position`` is the entry point on the surface and ``axis`` points *into* the
    material, both in the owning component's local frame.
    """

    id: str
    component_id: str
    position: np.ndarray = field(default_factory=lambda: vec3())
    axis: np.ndarray = field(default_factory=lambda: vec3(0, 0, 1))
    diameter: float = 3.0e-3
    depth: float = 6.0e-3
    pre_threaded: bool = True
    material_id: str | None = None
    label: str = "hole"
    created_by_feature: str = ""

    def __post_init__(self) -> None:
        self.axis = unit(self.axis)
        self.position = as_vec3(self.position)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "component_id": self.component_id,
            "position": np.asarray(self.position).tolist(),
            "axis": np.asarray(self.axis).tolist(),
            "diameter": self.diameter,
            "depth": self.depth,
            "pre_threaded": self.pre_threaded,
            "material_id": self.material_id,
            "label": self.label,
            "created_by_feature": self.created_by_feature,
        }

    @staticmethod
    def from_dict(d: dict) -> "HoleReference":
        return HoleReference(
            id=str(d["id"]),
            component_id=str(d["component_id"]),
            position=as_vec3(d.get("position", [0, 0, 0])),
            axis=as_vec3(d.get("axis", [0, 0, 1])),
            diameter=float(d.get("diameter", 3e-3)),
            depth=float(d.get("depth", 6e-3)),
            pre_threaded=bool(d.get("pre_threaded", True)),
            material_id=d.get("material_id"),
            label=str(d.get("label", "hole")),
            created_by_feature=str(d.get("created_by_feature", "")),
        )


@dataclass
class InstalledFastener:
    """A fastener installed into a hole, with derived rotation and engagement."""

    id: str
    fastener_id: str
    hole_id: str
    depth: float = 0.0
    inset: float = 0.0              # extra head protrusion (e.g. washer stack), m
    applied_torque: float = 0.0     # N*m used for strip-risk reporting
    installed: bool = False
    reversed_direction: bool = False
    representation: str = "simplified"
    with_nut: bool = False
    with_washer: bool = False
    locked: bool = False
    notes: str = ""

    def rotation(self, thread: ThreadSpec) -> float:
        """Derived rotation in radians: depth / (lead / 2pi)."""
        return self.depth / max(thread.advance_per_radian(), 1e-12)

    def set_depth(self, depth: float, thread: ThreadSpec, hole_depth: float | None = None) -> None:
        limit = hole_depth if hole_depth is not None else self.depth
        self.depth = max(0.0, float(depth))
        if limit is not None:
            self.depth = min(self.depth, max(limit, 0.0) + 0.0)
        self.installed = self.depth > 1e-6

    def advance(self, delta_rotation: float, thread: ThreadSpec, hole_depth: float) -> float:
        """Advance by rotating; returns the actual depth change (clamped by the hole)."""
        delta_depth = delta_rotation * thread.advance_per_radian()
        if self.reversed_direction:
            delta_depth = -delta_depth
        before = self.depth
        self.depth = float(np.clip(self.depth + delta_depth, 0.0, max(hole_depth, 0.0)))
        self.installed = self.depth > 1e-6
        return self.depth - before

    def seating_error(self, spec: FastenerSpec, hole: HoleReference) -> float:
        """How far the head is from the surface when fully inserted (m, + = interference)."""
        available = hole.depth + self.inset
        return max(spec.length - available - self.depth, 0.0)

    def engagement(self, spec: FastenerSpec, hole: HoleReference) -> float:
        return float(np.clip(min(self.depth, hole.depth), 0.0, hole.depth))

    def is_interfering(self, spec: FastenerSpec, hole: HoleReference, tol: float = 1e-5) -> bool:
        return self.depth > hole.depth + tol

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "fastener_id": self.fastener_id,
            "hole_id": self.hole_id,
            "depth": self.depth,
            "inset": self.inset,
            "applied_torque": self.applied_torque,
            "installed": self.installed,
            "reversed_direction": self.reversed_direction,
            "representation": self.representation,
            "with_nut": self.with_nut,
            "with_washer": self.with_washer,
            "locked": self.locked,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "InstalledFastener":
        return InstalledFastener(
            id=str(d["id"]),
            fastener_id=str(d["fastener_id"]),
            hole_id=str(d["hole_id"]),
            depth=float(d.get("depth", 0.0)),
            inset=float(d.get("inset", 0.0)),
            applied_torque=float(d.get("applied_torque", 0.0)),
            installed=bool(d.get("installed", False)),
            reversed_direction=bool(d.get("reversed_direction", False)),
            representation=str(d.get("representation", "simplified")),
            with_nut=bool(d.get("with_nut", False)),
            with_washer=bool(d.get("with_washer", False)),
            locked=bool(d.get("locked", False)),
            notes=str(d.get("notes", "")),
        )


class ScrewInstaller:
    """Drives the install/uninstall state machine for one fastener in one hole."""

    def __init__(self, engagement_model: ThreadEngagementModel | None = None) -> None:
        self.engagement_model = engagement_model or ThreadEngagementModel()

    # -- workflow ----------------------------------------------------------
    def start(self, spec: FastenerSpec, hole: HoleReference) -> InstalledFastener:
        if hole.diameter > spec.thread.major_diameter + 1e-6 and hole.pre_threaded:
            # a clearance hole cannot be pre-threaded; the screw needs a nut or an insert
            pass
        mode = self.mode_for(spec, hole)
        return InstalledFastener(
            id=f"fastener-{hole.id}",
            fastener_id=spec.id,
            hole_id=hole.id,
            depth=0.0,
            applied_torque=0.0,
            representation="simplified",
            with_nut=hole.diameter > spec.thread.major_diameter + 1e-6,
        )

    def mode_for(self, spec: FastenerSpec, hole: HoleReference) -> EngagementMode:
        if spec.kind in (FastenerKind.SELF_DRILLING,):
            return EngagementMode.CUTTING
        if spec.is_thread_forming():
            return EngagementMode.THREAD_FORMING
        return EngagementMode.PRE_THREADED if hole.pre_threaded else EngagementMode.THREAD_FORMING

    def install(self, state: InstalledFastener, spec: FastenerSpec, hole: HoleReference) -> InstalledFastener:
        state.set_depth(min(spec.length, hole.depth), spec.thread)
        state.installed = True
        return state

    def uninstall(self, state: InstalledFastener) -> InstalledFastener:
        state.depth = 0.0
        state.installed = False
        return state

    # -- kinematics --------------------------------------------------------
    def turn(self, state: InstalledFastener, spec: FastenerSpec, hole: HoleReference,
             delta_angle: float) -> float:
        """Rotate the screw by ``delta_angle`` (rad); returns axial advance in metres."""
        return state.advance(delta_angle, spec.thread, hole.depth)

    def set_depth(self, state: InstalledFastener, spec: FastenerSpec, hole: HoleReference,
                  depth: float) -> None:
        state.set_depth(depth, spec.thread, hole.depth)

    # -- inspection --------------------------------------------------------
    def inspect(
        self,
        state: InstalledFastener,
        spec: FastenerSpec,
        hole: HoleReference,
        target_ultimate_strength: float | None,
        target_youngs_modulus: float | None = None,
        layer_adhesion_factor: float = 1.0,
    ) -> EngagementResult:
        mode = self.mode_for(spec, hole)
        pilot = hole.diameter if mode is not EngagementMode.PRE_THREADED else 0.0
        return self.engagement_model.analyse(
            spec.thread,
            insertion_depth=state.engagement(spec, hole),
            mode=mode,
            target_ultimate_strength=target_ultimate_strength,
            pilot_diameter=pilot,
            applied_torque=state.applied_torque,
            layer_adhesion_factor=layer_adhesion_factor,
            youngs_modulus=target_youngs_modulus,
        )

    def fastener_pose(self, state: InstalledFastener, spec: FastenerSpec, hole: HoleReference,
                      component_world: Transform) -> Transform:
        """World pose of the fastener axis frame (origin at the head, +Z into the hole)."""
        local = Transform(
            rotation=Quat.from_two_vectors((0.0, 0.0, 1.0), hole.axis),
            translation=hole.position - hole.axis * (state.depth + state.inset),
        )
        # apply thread-driven spin about the hole axis so the animation shows rotation
        spin = Quat.from_axis_angle(hole.axis, state.rotation(spec.thread))
        local.rotation = spin * local.rotation
        return component_world @ local

