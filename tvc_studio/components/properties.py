"""Physical and mechanical property system for components.

Part 4 of the specification: physical properties must not be assumed derivable from
geometry, so this module supports three mass modes, extra non-geometric masses, manual
overrides, and per-property mechanical parameters that are consumed by the solvers.

Every field in :class:`MechanicalProperties` has a documented consumer
(:data:`PROPERTY_CONSUMERS`); nothing here is decorative. The UI shows the consumer as a
tooltip so the engineer knows what a value will change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np

from ..core.errors import MassPropertyError
from ..core.interfaces import Provenance
from ..core.math3d import MassProperties, as_vec3, combine_mass_properties, vec3
from .materials import Material, MaterialLibrary


class MassMode(str, Enum):
    CAD = "cad"                     # density x exact B-rep volume
    MANUAL = "manual"               # user-entered mass/COM/inertia, geometry ignored
    CAD_PLUS_EXTRAS = "cad_plus_extras"  # derived + explicitly modelled extra masses

    def label(self) -> str:
        return {
            "cad": "Calculated from CAD",
            "manual": "Manual override",
            "cad_plus_extras": "CAD + extra masses",
        }[self.value]


@dataclass
class ExtraMass:
    """Lumped mass that is not modelled as solid geometry (glue, wiring, ballast)."""

    name: str
    mass: float
    position: np.ndarray = field(default_factory=lambda: vec3())
    provenance: Provenance = Provenance.ESTIMATED
    note: str = ""

    def to_mass_properties(self) -> MassProperties:
        return MassProperties(
            mass=max(self.mass, 0.0),
            com=as_vec3(self.position),
            inertia=np.zeros((3, 3)),
            provenance=self.provenance.value,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mass": self.mass,
            "position": np.asarray(self.position).tolist(),
            "provenance": self.provenance.value,
            "note": self.note,
        }

    @staticmethod
    def from_dict(d: dict) -> "ExtraMass":
        return ExtraMass(
            name=str(d.get("name", "extra")),
            mass=float(d.get("mass", 0.0)),
            position=as_vec3(d.get("position", [0, 0, 0])),
            provenance=Provenance(d.get("provenance", "estimated")),
            note=str(d.get("note", "")),
        )


@dataclass
class PhysicalProperties:
    """Mass configuration of one component (or one comparable configuration)."""

    material_id: str = "unknown"
    mass_mode: MassMode = MassMode.CAD
    manual_mass: float = 0.0
    manual_com: np.ndarray = field(default_factory=lambda: vec3())
    manual_inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    density_override: float | None = None
    extras: list[ExtraMass] = field(default_factory=list)
    notes: str = ""

    # -- resolution --------------------------------------------------------
    def resolve(
        self,
        cad_mass_properties: MassProperties | None,
        material: Material,
        library: MaterialLibrary | None = None,
    ) -> MassProperties:
        """Produce the mass properties actually used by every simulation.

        Raises :class:`MassPropertyError` rather than fabricating numbers when CAD
        geometry is required but missing.
        """
        unresolved: list[str] = []
        density = self.density_override if self.density_override is not None else material.density

        if self.mass_mode == MassMode.MANUAL:
            if self.manual_mass <= 0.0:
                raise MassPropertyError(
                    "manual mass mode requires a positive mass; enter the measured mass or switch to CAD mode"
                )
            mp = MassProperties(
                mass=float(self.manual_mass),
                com=as_vec3(self.manual_com),
                inertia=np.asarray(self.manual_inertia, dtype=float).reshape(3, 3),
                volume=cad_mass_properties.volume if cad_mass_properties else 0.0,
                surface_area=cad_mass_properties.surface_area if cad_mass_properties else 0.0,
                provenance=Provenance.USER_OVERRIDE.value,
                unresolved=tuple(unresolved),
            )
            return mp

        if cad_mass_properties is None:
            raise MassPropertyError(
                "component has no CAD geometry; supply a manual mass or model the solid"
            )

        if material.provenance is Provenance.UNKNOWN:
            unresolved.append(f"material '{material.name}' is not characterised")

        base = MassProperties(
            mass=cad_mass_properties.volume * density,
            com=as_vec3(cad_mass_properties.com),
            inertia=np.asarray(cad_mass_properties.inertia, dtype=float).reshape(3, 3)
            * (density / max(material.density, 1e-9)),
            volume=cad_mass_properties.volume,
            surface_area=cad_mass_properties.surface_area,
            provenance=Provenance.CAD_DERIVED.value,
            unresolved=tuple(unresolved),
        )
        # cad_mass_properties already used the material density, so avoid double scaling
        if abs(density - material.density) < 1e-12 and cad_mass_properties.mass > 0:
            base.mass = cad_mass_properties.mass
            base.inertia = np.asarray(cad_mass_properties.inertia, dtype=float).reshape(3, 3)

        if self.mass_mode == MassMode.CAD_PLUS_EXTRAS and self.extras:
            parts = [base] + [e.to_mass_properties() for e in self.extras]
            combined = combine_mass_properties(parts)
            combined.volume = base.volume
            combined.surface_area = base.surface_area
            combined.provenance = "mixed"
            combined.unresolved = tuple(unresolved)
            return combined

        return base

    def is_override(self) -> bool:
        return self.mass_mode == MassMode.MANUAL

    def to_dict(self) -> dict:
        return {
            "material_id": self.material_id,
            "mass_mode": self.mass_mode.value,
            "manual_mass": self.manual_mass,
            "manual_com": np.asarray(self.manual_com).tolist(),
            "manual_inertia": np.asarray(self.manual_inertia).tolist(),
            "density_override": self.density_override,
            "extras": [e.to_dict() for e in self.extras],
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "PhysicalProperties":
        return PhysicalProperties(
            material_id=str(d.get("material_id", "unknown")),
            mass_mode=MassMode(d.get("mass_mode", "cad")),
            manual_mass=float(d.get("manual_mass", 0.0)),
            manual_com=as_vec3(d.get("manual_com", [0, 0, 0])),
            manual_inertia=np.asarray(d.get("manual_inertia", np.zeros((3, 3))), dtype=float).reshape(3, 3),
            density_override=(
                None if d.get("density_override") is None else float(d["density_override"])
            ),
            extras=[ExtraMass.from_dict(e) for e in d.get("extras", [])],
            notes=str(d.get("notes", "")),
        )


@dataclass
class MechanicalProperties:
    """Joint/actuator behaviour parameters.

    Consumers are listed in :data:`PROPERTY_CONSUMERS` and surfaced in the property
    editor. Values only appear in a simulation where they are actually used.
    """

    stiffness: float = 0.0          # N*m/rad at joints (0 = rigid)
    damping: float = 0.02           # N*m*s/rad
    friction_static: float = 0.0    # N*m
    friction_kinetic: float = 0.0   # N*m
    backlash: float = 0.0           # rad of free play
    deadband: float = 0.0           # actuator command deadband, rad
    play: float = 0.0               # linear free play, m
    joint_min: float = 0.0          # rad or m depending on joint kind
    joint_max: float = 0.0
    max_torque: float = 0.0         # N*m, 0 = unlimited
    max_force: float = 0.0          # N, 0 = unlimited
    max_speed: float = 0.0          # rad/s or m/s, 0 = unlimited
    efficiency: float = 1.0         # 0..1, scales transmitted torque

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not 0.0 < self.efficiency <= 1.0:
            problems.append("mechanical efficiency must be in (0, 1]")
        if self.stiffness < 0:
            problems.append("stiffness cannot be negative")
        if self.damping < 0:
            problems.append("damping cannot be negative")
        if self.joint_max < self.joint_min:
            problems.append("joint maximum is below joint minimum")
        if self.friction_kinetic > self.friction_static + 1e-12:
            problems.append("kinetic friction exceeds static friction")
        if self.backlash < 0:
            problems.append("backlash cannot be negative")
        return problems

    def effective_torque_limit(self, applied: float) -> float:
        """Apply efficiency and torque limit to a demanded torque."""
        limited = applied * self.efficiency
        if self.max_torque > 0.0:
            limited = float(np.clip(limited, -self.max_torque, self.max_torque))
        return limited

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @staticmethod
    def from_dict(d: dict) -> "MechanicalProperties":
        allowed = MechanicalProperties.__dataclass_fields__
        return MechanicalProperties(**{k: float(v) for k, v in d.items() if k in allowed})


PROPERTY_CONSUMERS: dict[str, str] = {
    "stiffness": "Joint spring stiffness in mech.dynamics (compliant joint torque = k·Δθ).",
    "damping": "Joint viscous damping torque in mech.dynamics.",
    "friction_static": "Static friction threshold at the joint before sliding begins.",
    "friction_kinetic": "Coulomb friction applied while the joint is moving.",
    "backlash": "Free-play gap in the joint/gear train: no torque transmitted inside the gap.",
    "deadband": "Servo command deadband inside mech.servo before the motor responds.",
    "play": "Linear free play for prismatic joints and screw threads.",
    "joint_min": "Lower travel stop enforced as an inequality constraint.",
    "joint_max": "Upper travel stop enforced as an inequality constraint.",
    "max_torque": "Torque saturation of the actuator driving this joint.",
    "max_force": "Force saturation for prismatic actuators.",
    "max_speed": "Rate limit applied to joint velocity targets.",
    "efficiency": "Scales torque actually delivered versus torque demanded.",
}


@dataclass
class ComponentConfiguration:
    """A named set of property overrides; lets one part carry several build states."""

    name: str
    physical: PhysicalProperties = field(default_factory=PhysicalProperties)
    mechanical: MechanicalProperties = field(default_factory=MechanicalProperties)
    hidden_component_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "physical": self.physical.to_dict(),
            "mechanical": self.mechanical.to_dict(),
            "hidden_component_ids": list(self.hidden_component_ids),
        }

    @staticmethod
    def from_dict(d: dict) -> "ComponentConfiguration":
        return ComponentConfiguration(
            name=str(d.get("name", "Configuration")),
            physical=PhysicalProperties.from_dict(d.get("physical", {})),
            mechanical=MechanicalProperties.from_dict(d.get("mechanical", {})),
            hidden_component_ids=tuple(d.get("hidden_component_ids", ())),
        )


def summarise_breakdown(entries: Iterable[tuple[str, float]]) -> list[tuple[str, float, float]]:
    """Return (name, mass, percent) rows for the analysis breakdown table."""
    rows = [(name, mass) for name, mass in entries]
    total = sum(max(m, 0.0) for _, m in rows)
    out: list[tuple[str, float, float]] = []
    for name, mass in rows:
        pct = (mass / total * 100.0) if total > 0 else 0.0
        out.append((name, mass, pct))
    return out
