"""Build a flight-ready rocket description from the CAD assembly.

Every number here comes from the assembly (mass properties, component placements), the
motor definition and the aerodynamic configuration — never from a hand-entered whole-rocket
summary. Components are identified by their hardware metadata:

* ``metadata["role"] = "motor"``    — the solid motor (COM used for thrust application)
* ``metadata["role"] = "nozzle"``   — the nozzle exit (TVC gimbal point)
* ``metadata["role"] = "tvc"``      — the TVC mechanism centroid
* ``hardware_type = "sensor"``      — the IMU mount
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assembly.massprops import AssemblyMassCalculator, AssemblyMassResult
from ..assembly.model import Assembly
from ..components.hardware import HardwareLibrary
from ..core.interfaces import Diagnostic
from ..core.math3d import Quat, Transform, as_vec3, vec3
from .aero import AeroConfig, StackElement, barrowman_cp
from .flight import RocketProperties
from .propulsion import MotorDefinition


@dataclass
class RocketBuild:
    properties: RocketProperties
    stack: list[StackElement]
    diagnostics: list[Diagnostic]
    mass_result: AssemblyMassResult
    components_used: dict[str, str]

    def summary(self) -> dict:
        mp = self.properties
        return {
            "dry_mass_kg": mp.mass_dry,
            "motor_mass_kg": self.mass_result.properties.mass,
            "reference_area_m2": mp.reference_area,
            "reference_length_m": mp.reference_length,
            "max_diameter_m": mp.max_diameter,
            "cp_position_m": mp.cp_position,
            "static_margin_m": mp.static_margin(0.0),
        }


def _component_role(comp) -> str:
    return str(comp.metadata.get("role", ""))


def build_rocket(
    assembly: Assembly,
    mass_calculator: AssemblyMassCalculator,
    parts_repo,
    motor: MotorDefinition,
    hardware: HardwareLibrary,
    aero_config: AeroConfig,
) -> RocketBuild:
    diagnostics: list[Diagnostic] = []
    mass_result = mass_calculator.calculate(assembly)
    total = mass_result.properties
    if total.mass <= 0:
        raise ValueError("assembly has no mass; cannot build a flight model")

    # ---- propellant vs dry mass -----------------------------------------
    propellant_mass = motor.propellant_mass
    if propellant_mass >= total.mass:
        diagnostics.append(Diagnostic(
            "warning", "propellant-mass",
            f"Motor propellant mass ({propellant_mass*1000:.1f} g) is not smaller than the whole "
            f"assembly mass ({total.mass*1000:.1f} g). Check that the motor is modelled and the "
            "assembly includes its hardware."))
    mass_dry = max(total.mass - propellant_mass, 1e-6)
    com_total = as_vec3(total.com)
    inertia_total = np.asarray(total.inertia, dtype=float)

    # ---- find key components --------------------------------------------
    motor_comp = None
    nozzle_comp = None
    tvc_comp = None
    imu_comp = None
    for comp in assembly.active_components():
        role = _component_role(comp)
        if role == "motor" and motor_comp is None:
            motor_comp = comp
        elif role == "nozzle" and nozzle_comp is None:
            nozzle_comp = comp
        elif role == "tvc" and tvc_comp is None:
            tvc_comp = comp
        if comp.hardware_type == "sensor" and imu_comp is None:
            imu_comp = comp
        if comp.hardware_type == "motor" and motor_comp is None:
            motor_comp = comp

    motor_com = as_vec3(mass_calculator.component_properties(motor_comp).com) if motor_comp else com_total.copy()
    nozzle_position = (as_vec3(mass_calculator.component_properties(nozzle_comp).com)
                       if nozzle_comp else motor_com - vec3(0, 0, motor.length / 2 if motor.length else 0.03))
    tvc_position = (as_vec3(mass_calculator.component_properties(tvc_comp).com)
                    if tvc_comp else nozzle_position + vec3(0, 0, 0.03))
    imu_position = (as_vec3(mass_calculator.component_properties(imu_comp).com)
                    if imu_comp else com_total + vec3(0, 0, 0.05))
    imu_orientation = assembly.world_transform(imu_comp.id).rotation if imu_comp else Quat.identity()

    # ---- dry inertia (remove propellant point mass) ---------------------
    prop_pos = motor_com
    dry_com = (mass_dry * com_total - propellant_mass * prop_pos) / max(mass_dry, 1e-9)
    # inertia about dry COM: subtract the propellant's point-mass contribution
    d = prop_pos - com_total
    prop_inertia = propellant_mass * ((d @ d) * np.eye(3) - np.outer(d, d))
    inertia_about_total = inertia_total - prop_inertia
    dry_com_offset = com_total - dry_com
    inertia_dry = inertia_about_total - mass_dry * ((dry_com_offset @ dry_com_offset) * np.eye(3)
                                                    - np.outer(dry_com_offset, dry_com_offset))

    # ---- reference geometry --------------------------------------------
    bbox_min = bbox_max = None
    positions = []
    for comp in assembly.active_components():
        if not comp.part_id or parts_repo is None:
            continue
        try:
            lo, hi = parts_repo.get(comp.part_id).bounding_box(parts_repo.backend)
        except Exception:
            continue
        world = assembly.world_transform(comp.id)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        world_corners = np.array([world.apply_point(c) for c in corners])
        positions.append(world_corners)
    if positions:
        all_corners = np.vstack(positions)
        bbox_min = all_corners.min(axis=0)
        bbox_max = all_corners.max(axis=0)
    if bbox_min is None:
        bbox_min = total.com - vec3(0.03, 0.03, 0.25)
        bbox_max = total.com + vec3(0.03, 0.03, 0.25)
        diagnostics.append(Diagnostic("warning", "geometry", "Could not measure the assembly bounds; "
                                                            "reference area uses a default."))
    length = float(bbox_max[2] - bbox_min[2])
    # body diameter excludes fins and internal avionics: the reference area is the airframe
    # section. Prefer components explicitly named as airframe/body/tube.
    body_diameter = 0.0
    named_diameter = 0.0
    for comp in assembly.active_components():
        name = comp.name.lower()
        if not comp.part_id or parts_repo is None or "fin" in name:
            continue
        if comp.hardware_type in ("board", "sensor", "servo"):
            continue
        try:
            lo, hi = parts_repo.get(comp.part_id).bounding_box(parts_repo.backend)
        except Exception:
            continue
        world = assembly.world_transform(comp.id)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        wc = np.array([world.apply_point(c) for c in corners])
        d = max(float(np.ptp(wc[:, 0])), float(np.ptp(wc[:, 1])))
        body_diameter = max(body_diameter, d)
        if any(key in name for key in ("airframe", "body", "tube")):
            named_diameter = max(named_diameter, d)
    body_diameter = named_diameter or body_diameter
    diameter = body_diameter or float(max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]))
    reference_area = float(np.pi * (diameter / 2.0) ** 2) if diameter > 0 else 0.0

    stack = _stack_from_components(assembly, mass_calculator, parts_repo, bbox_min, bbox_max,
                                   body_diameter)
    estimate = barrowman_cp(stack) if stack else None
    cp_position = aero_config.cp_position
    if cp_position <= 0 and estimate is not None:
        cp_position = estimate.cp_position
        diagnostics.append(Diagnostic(
            "info", "cp-estimate",
            f"Centre of pressure estimated at {cp_position*1000:.1f} mm from the nose "
            "(Barrowman, approximate). Enter a measured value in the Aerodynamics panel to override."))
    nose_tip_z = bbox_min[2]
    cp_from_com = cp_position + nose_tip_z  # cp_position is measured from the nose tip

    props = RocketProperties(
        name=assembly.name,
        mass_dry=mass_dry,
        com_dry=dry_com,
        inertia_dry=inertia_dry,
        motor_position=motor_com,
        nozzle_position=nozzle_position,
        propellant_position=prop_pos,
        tvc_position=tvc_position,
        imu_position=imu_position,
        imu_orientation=imu_orientation,
        reference_area=reference_area or aero_config.reference_area,
        reference_length=length or 1.0,
        max_diameter=diameter,
        cp_position=cp_from_com,
        provenance=total.provenance,
        notes="Derived from CAD assembly mass properties and component placements.",
    )
    static_margin = float(dry_com[2]) - cp_from_com
    if static_margin <= 0:
        diagnostics.append(Diagnostic(
            "warning", "static-margin",
            f"Static margin is {static_margin*1000:.1f} mm (CP ahead of CG): the rocket is "
            "aerodynamically unstable and depends entirely on TVC authority."))
    if total.provenance != "cad_derived":
        diagnostics.append(Diagnostic("warning", "mass-provenance",
                                      f"Assembly mass properties are '{total.provenance}': flight results inherit that uncertainty."))
    return RocketBuild(props, stack, diagnostics, mass_result,
                       {"motor": motor_comp.name if motor_comp else "default",
                        "nozzle": nozzle_comp.name if nozzle_comp else "default",
                        "imu": imu_comp.name if imu_comp else "default"})


def _stack_from_components(assembly: Assembly, mass_calc, parts_repo,
                           bbox_min, bbox_max, body_diameter: float = 0.0) -> list[StackElement]:
    """Approximate the outer moulding stack from component bounding boxes.

    Used only for the geometry-based CP estimate. It is explicitly approximate: real fin
    geometry and nose profiles come from the CAD model, but this stack uses envelopes.
    """
    elements: list[StackElement] = []
    nose_z = float(bbox_min[2])
    boxes: list[tuple] = []
    for comp in assembly.active_components():
        if not comp.part_id or parts_repo is None:
            continue
        try:
            lo, hi = parts_repo.get(comp.part_id).bounding_box(parts_repo.backend)
        except Exception:
            continue
        world = assembly.world_transform(comp.id)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        wc = np.array([world.apply_point(c) for c in corners])
        cmin, cmax = wc.min(axis=0), wc.max(axis=0)
        length = float(cmax[2] - cmin[2])
        diameter = float(max(cmax[0] - cmin[0], cmax[1] - cmin[1]))
        if length <= 0 or diameter <= 0:
            continue
        boxes.append((comp, cmin, cmax, length, diameter))

    def _kind(comp) -> str:
        name = comp.name.lower()
        role = str(comp.metadata.get("role", "")).lower()
        if "nose" in name or role == "nose":
            return "nose"
        if "fin" in name or role == "fin":
            return "fin"
        return "body"

    body_radius = body_diameter / 2.0
    if body_radius <= 0:
        for comp, cmin, cmax, length, diameter in boxes:
            if _kind(comp) == "body":
                body_radius = max(body_radius, diameter / 2.0)

    for comp, cmin, cmax, length, diameter in boxes:
        kind = _kind(comp)
        if kind == "fin":
            # the fin component may contain every fin in a radial pattern; the count is
            # explicit metadata, never guessed from geometry
            count = int(comp.metadata.get("fin_count", 1))
            span = max(diameter / 2.0 - body_radius, 1e-4)
            root = float(comp.metadata.get("fin_root_chord", length))
            tip = float(comp.metadata.get("fin_tip_chord", root * 0.6))
            elements.append(StackElement(
                kind="fin", start=float(cmin[2] - nose_z), length=length, diameter=2.0 * body_radius,
                fin_count=count, fin_root_chord=root, fin_tip_chord=tip, fin_span=span,
                fin_sweep=float(comp.metadata.get("fin_sweep", 0.0)), name=comp.name,
            ))
            continue
        elements.append(StackElement(
            kind=kind, start=float(cmin[2] - nose_z), length=length, diameter=diameter,
            name=comp.name,
        ))
    elements.sort(key=lambda e: e.start)
    return elements
