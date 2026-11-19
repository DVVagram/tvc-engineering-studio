"""Aerodynamics for the assembled rocket.

Two models, both explicit about their provenance:

* **CoefficientAero** — the workhorse. Drag/lift/moment from user or measured coefficients.
  This is what flight simulation uses by default; the coefficients are the user's data.
* **geometry_estimate** — Barrowman-style centre-of-pressure and normal-force estimate from
  a simplified stack (nose, tubes, transitions, fins). It is an *estimate*, labelled as
  such, and never presented as CFD or wind-tunnel accuracy.

The aero model is deliberately decoupled from the CAD kernel: it consumes a stack and a
reference geometry, so a CFD or panel-method model can replace it without touching the CAD
architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..core.interfaces import Provenance
from ..core.math3d import as_vec3, unit, vec3


class AeroModelKind(str, Enum):
    COEFFICIENTS = "coefficients"
    GEOMETRY_ESTIMATE = "geometry_estimate"


@dataclass
class AeroConfig:
    """Coefficient set with provenance. Defaults are generic, not for a specific rocket."""

    model: AeroModelKind = AeroModelKind.COEFFICIENTS
    cd0: float = 0.45                    # zero-lift axial drag coefficient
    cd_alpha2: float = 0.0               # drag rise with alpha^2 (1/rad^2)
    cn_alpha: float = 2.0                # normal force slope (1/rad)
    cm_alpha: float = 0.0                # pitching moment slope (1/rad), negative = stable
    damping_coefficient: float = 0.0     # pitch/yaw damping, N·m·s/rad
    roll_damping: float = 0.0
    reference_area: float = 0.0          # m^2; derived from assembly when zero
    reference_length: float = 0.0        # m; derived from assembly when zero
    cp_position: float = 0.0             # m from nose tip (body frame +Z), 0 = use estimate
    cp_provenance: Provenance = Provenance.ESTIMATED
    coefficient_provenance: Provenance = Provenance.ESTIMATED
    uncertainty_percent: float = 20.0
    notes: str = ""

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.coefficient_provenance is Provenance.ESTIMATED:
            out.append("Aerodynamic coefficients are generic estimates, not measured data for this rocket.")
        if self.cp_provenance is Provenance.ESTIMATED:
            out.append("Centre of pressure is estimated; static margin is indicative only.")
        if self.uncertainty_percent >= 15:
            out.append(f"Coefficient uncertainty is ±{self.uncertainty_percent:.0f}%; expect matching scatter in apogee.")
        return out

    def to_dict(self) -> dict:
        return {
            "model": self.model.value, "cd0": self.cd0, "cd_alpha2": self.cd_alpha2,
            "cn_alpha": self.cn_alpha, "cm_alpha": self.cm_alpha,
            "damping_coefficient": self.damping_coefficient, "roll_damping": self.roll_damping,
            "reference_area": self.reference_area, "reference_length": self.reference_length,
            "cp_position": self.cp_position, "cp_provenance": self.cp_provenance.value,
            "coefficient_provenance": self.coefficient_provenance.value,
            "uncertainty_percent": self.uncertainty_percent, "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "AeroConfig":
        return AeroConfig(
            model=AeroModelKind(d.get("model", "coefficients")),
            cd0=float(d.get("cd0", 0.45)),
            cd_alpha2=float(d.get("cd_alpha2", 0.0)),
            cn_alpha=float(d.get("cn_alpha", 2.0)),
            cm_alpha=float(d.get("cm_alpha", 0.0)),
            damping_coefficient=float(d.get("damping_coefficient", 0.0)),
            roll_damping=float(d.get("roll_damping", 0.0)),
            reference_area=float(d.get("reference_area", 0.0)),
            reference_length=float(d.get("reference_length", 0.0)),
            cp_position=float(d.get("cp_position", 0.0)),
            cp_provenance=Provenance(d.get("cp_provenance", "estimated")),
            coefficient_provenance=Provenance(d.get("coefficient_provenance", "estimated")),
            uncertainty_percent=float(d.get("uncertainty_percent", 20.0)),
            notes=str(d.get("notes", "")),
        )


@dataclass
class StackElement:
    """Simplified outer-moulding element used by the geometry CP estimate."""

    kind: str                 # nose | body | transition | fin | boattail
    start: float              # m from nose tip along the rocket axis
    length: float
    diameter: float           # m, maximum diameter
    rear_diameter: float = 0.0
    fin_count: int = 0
    fin_root_chord: float = 0.0
    fin_tip_chord: float = 0.0
    fin_span: float = 0.0
    fin_sweep: float = 0.0
    name: str = ""

    @property
    def end(self) -> float:
        return self.start + self.length


@dataclass
class BarrowmanResult:
    cp_position: float            # m from nose tip
    cn_alpha: float               # per radian
    contributions: list[dict] = field(default_factory=list)
    assumptions: tuple[str, ...] = (
        "Barrowman slender-body method: subsonic, attached flow, small angles of attack.",
        "Nose cone CP at 2/3 length (ogive approximation); body lift ignored.",
        "Transonic and supersonic corrections are not modelled.",
    )


def barrowman_cp(stack: list[StackElement]) -> BarrowmanResult:
    """Estimate centre of pressure and normal-force slope from a simplified stack."""
    contributions: list[dict] = []
    total_cn = 0.0
    moment = 0.0
    for element in stack:
        d = max(element.diameter, 1e-6)
        if element.kind == "nose":
            cn = 2.0
            cp = element.start + 0.666 * element.length
        elif element.kind in ("transition", "boattail"):
            d1 = max(element.diameter, 1e-6)
            d2 = max(element.rear_diameter or element.diameter, 1e-6)
            area_ratio = (d2 / d1) ** 2
            cn = 2.0 * (area_ratio - 1.0)
            cp = element.start + element.length * (1.0 - (1.0 - area_ratio) / 3.0) if abs(area_ratio - 1) > 1e-9 else element.start + element.length / 2
        elif element.kind == "fin" and element.fin_count > 0:
            s = max(element.fin_span, 1e-6)
            cr = max(element.fin_root_chord, 1e-6)
            ct = max(element.fin_tip_chord, 0.0)
            body_d = max(element.diameter, 1e-6)
            # Barrowman fin normal force referenced to body cross-section
            span_ratio = s / body_d
            # Barrowman fin normal force: s is the semi-span, chords the root/tip values.
            denom = 1.0 + math.sqrt(1.0 + (2.0 * s / max(cr + ct, 1e-6)) ** 2)
            cn = (4.0 * element.fin_count * span_ratio ** 2) / denom
            # Chordwise centroid of the trapezoid plus the sweep offset.
            cp = element.start + (cr * (cr + 2.0 * ct)) / (3.0 * (cr + ct))
            cp += element.fin_sweep * (cr + 2.0 * ct) / (3.0 * (cr + ct))
        else:
            continue
        if cn <= 0:
            continue
        total_cn += cn
        moment += cn * cp
        contributions.append({
            "element": element.name or element.kind,
            "kind": element.kind,
            "cn_alpha": cn,
            "cp_position": cp,
        })
    cp = moment / total_cn if total_cn > 0 else float("nan")
    return BarrowmanResult(cp_position=cp, cn_alpha=total_cn, contributions=contributions)


@dataclass
class AeroForces:
    force_body: np.ndarray
    moment_body: np.ndarray
    alpha: float
    mach: float
    dynamic_pressure: float
    cp_position: float
    static_margin: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "force_body": self.force_body.tolist(),
            "moment_body": self.moment_body.tolist(),
            "alpha": self.alpha,
            "mach": self.mach,
            "dynamic_pressure": self.dynamic_pressure,
            "cp_position": self.cp_position,
            "static_margin": self.static_margin,
        }


class AeroModel:
    """Base aerodynamic model interface (kept replaceable per the specification)."""

    def forces(self, velocity_world, attitude, altitude, density, speed_of_sound,
               com_position: float, angular_velocity_body=None) -> AeroForces:  # pragma: no cover
        raise NotImplementedError


class CoefficientAero(AeroModel):
    """Drag + normal force + moments from coefficients, evaluated in the body frame."""

    def __init__(self, config: AeroConfig) -> None:
        self.config = config

    def forces(self, velocity_world, attitude, altitude, density, speed_of_sound,
               com_position: float, angular_velocity_body=None) -> AeroForces:
        cfg = self.config
        R = attitude.matrix()
        v_body = R.T @ as_vec3(velocity_world)
        speed = float(np.linalg.norm(v_body))
        warnings: list[str] = []
        area = cfg.reference_area
        length = cfg.reference_length or 1.0
        if area <= 0:
            warnings.append("reference area is zero; set it in the Aerodynamics panel or from the assembly")
        q = 0.5 * density * speed * speed
        mach = speed / max(speed_of_sound, 1e-6)
        if speed < 1e-6 or area <= 0:
            return AeroForces(vec3(), vec3(), 0.0, mach, q, cfg.cp_position, 0.0, tuple(warnings))

        v_hat = v_body / speed
        # +Z body axis is the rocket's longitudinal axis
        axial = float(v_hat[2])
        alpha = math.acos(max(min(axial, 1.0), -1.0))
        # perpendicular direction in the body x/y plane
        perp = v_body - axial * speed * np.array([0.0, 0.0, 1.0])
        perp_norm = float(np.linalg.norm(perp))
        perp_hat = perp / perp_norm if perp_norm > 1e-9 else np.array([1.0, 0.0, 0.0])

        cd = cfg.cd0 + cfg.cd_alpha2 * alpha * alpha
        # drag along -v_body direction
        drag = -v_hat * (q * cd * area)
        # normal force in the plane of the body axis and the relative wind
        cn = cfg.cn_alpha * math.sin(alpha) * math.cos(alpha)
        normal = -perp_hat * (q * cn * area)
        force_body = drag + normal

        cp = cfg.cp_position if cfg.cp_position > 0 else com_position
        # positive when the CP is aft of the CG (body z is measured from the tail)
        static_margin = com_position - cp
        # moment about the centre of mass from the CP offset (positive = nose-down)
        arm = cp - com_position
        moment = np.zeros(3)
        if arm != 0.0 and perp_norm > 1e-9:
            # force applied at CP: torque = r_cp x F where r_cp is along +Z
            r_cp = np.array([0.0, 0.0, arm])
            moment += np.cross(r_cp, normal)
        if angular_velocity_body is not None and cfg.damping_coefficient:
            w = as_vec3(angular_velocity_body)
            moment -= cfg.damping_coefficient * w
        if cfg.roll_damping:
            moment[2] -= cfg.roll_damping * float(as_vec3(angular_velocity_body or vec3())[2])
        if mach > 0.8:
            warnings.append("Mach > 0.8: these coefficients are subsonic; drag is underpredicted at "
                            "transonic and supersonic speeds.")
        if static_margin < 0:
            warnings.append("Centre of pressure is ahead of the centre of mass at this condition "
                            "(static margin negative): aerodynamically unstable without TVC.")
        return AeroForces(force_body, moment, alpha, mach, q, cp, static_margin, tuple(warnings))


class GeometryAero(CoefficientAero):
    """Coefficient model whose CP comes from a Barrowman estimate over the stack."""

    def __init__(self, config: AeroConfig, stack: list[StackElement]) -> None:
        super().__init__(config)
        self.stack = stack
        self.estimate = barrowman_cp(stack)
        if config.cp_position <= 0 and np.isfinite(self.estimate.cp_position):
            self.config.cp_position = self.estimate.cp_position
        if config.cn_alpha <= 0:
            self.config.cn_alpha = self.estimate.cn_alpha

    def forces(self, *args, **kwargs) -> AeroForces:
        result = super().forces(*args, **kwargs)
        extra = list(result.warnings) + [
            "Centre of pressure from Barrowman estimate (approximate, subsonic)."
        ]
        return AeroForces(result.force_body, result.moment_body, result.alpha, result.mach,
                          result.dynamic_pressure, result.cp_position, result.static_margin,
                          tuple(extra))


def make_aero(config: AeroConfig, stack: list[StackElement] | None = None) -> AeroModel:
    if config.model is AeroModelKind.GEOMETRY_ESTIMATE and stack:
        return GeometryAero(config, stack)
    return CoefficientAero(config)


def drag_equation(density: float, speed: float, cd: float, area: float) -> float:
    """Convenience for tests and the analysis panel: D = 0.5 rho v^2 Cd A."""
    return 0.5 * density * speed * speed * cd * area
