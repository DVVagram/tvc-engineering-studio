"""Turn an assembly into a mechanical dynamics world.

This is the bridge that makes the CAD assembly the source of truth for mechanical
simulation: bodies take mass, inertia and geometry from the assembly; joints are created
from the assembly's constraints; servos bind to joints through component hardware metadata.

Joint mapping (assembly constraint -> mechanical joint):

    REVOLUTE / CONCENTRIC / SCREW / CYLINDRICAL / SLIDER / BALL / PLANAR / FIXED

The moving component's :class:`MechanicalProperties` supply friction, damping, stiffness,
backlash, limits and actuator ratings, so those UI values reach the physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..assembly.constraints import AssemblyConstraint, ConstraintType
from ..assembly.model import Assembly
from ..assembly.massprops import AssemblyMassCalculator
from ..components.hardware import HardwareLibrary
from ..core.interfaces import Diagnostic, Mesh
from ..core.math3d import Quat, Transform, as_vec3, unit, vec3
from .dynamics import Body, DynamicsWorld, Joint, JointKind
from .servo import ServoPlant

_CONSTRAINT_TO_JOINT = {
    ConstraintType.REVOLUTE: JointKind.REVOLUTE,
    ConstraintType.CONCENTRIC: JointKind.CYLINDRICAL,
    ConstraintType.CYLINDRICAL: JointKind.CYLINDRICAL,
    ConstraintType.SLIDER: JointKind.PRISMATIC,
    ConstraintType.SCREW: JointKind.SCREW,
    ConstraintType.BALL: JointKind.BALL,
    ConstraintType.PLANAR: JointKind.PLANAR,
    ConstraintType.COINCIDENT: JointKind.BALL,
    ConstraintType.FIXED: JointKind.FIXED,
}


@dataclass
class WorldBuildResult:
    world: DynamicsWorld
    servos: dict[str, ServoPlant] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    mass_properties: object | None = None
    component_bodies: dict[str, str] = field(default_factory=dict)


def build_world(
    assembly: Assembly,
    mass_calculator: AssemblyMassCalculator,
    parts_repo,
    hardware: HardwareLibrary,
    backend=None,
    collision_meshes: dict[str, Mesh] | None = None,
    enable_servos: bool = True,
) -> WorldBuildResult:
    """Create a dynamics world from the assembled CAD model."""
    mass_result = mass_calculator.calculate(assembly)
    world = DynamicsWorld(name=f"{assembly.name} mechanics")
    diagnostics: list[Diagnostic] = []
    servos: dict[str, ServoPlant] = {}
    component_bodies: dict[str, str] = {}

    # bodies that take part in a mate are the moving parts; everything else is rigidly
    # mounted structure, held fixed in the bench simulation
    participants = set()
    for constraint in assembly.constraints:
        if not constraint.enabled or constraint.suppressed:
            continue
        if constraint.a:
            participants.add(constraint.a.component_id)
        if constraint.b:
            participants.add(constraint.b.component_id)

    for comp in assembly.active_components():
        if comp.is_subassembly and not comp.part_id:
            continue
        if comp.part_id is None:
            continue
        try:
            mp = mass_calculator.component_properties(comp)
        except Exception as exc:
            diagnostics.append(Diagnostic("error", "mass", f"'{comp.name}': {exc}"))
            continue
        if mp.mass <= 0:
            diagnostics.append(Diagnostic("warning", "zero-mass", f"'{comp.name}' has no mass; excluded from mechanics"))
            continue
        world_tf = assembly.world_transform(comp.id)
        R = world_tf.rotation.matrix()
        com_local = R.T @ (mp.com - world_tf.translation)
        inertia_local = R.T @ np.asarray(mp.inertia) @ R
        mesh = None
        if collision_meshes and comp.id in collision_meshes:
            mesh = collision_meshes[comp.id]
        else:
            try:
                mesh = parts_repo.collision_mesh(comp.part_id) if parts_repo is not None else None
            except Exception:
                mesh = None
        body = Body(
            id=comp.id,
            name=comp.name,
            mass=mp.mass,
            inertia_local=inertia_local,
            com_local=com_local,
            position=mp.com,
            rotation=world_tf.rotation,
            fixed=comp.grounded or comp.id not in participants,
            collision_mesh=mesh,
            friction=max(comp.mechanical.friction_kinetic, 0.05) if comp.mechanical.friction_kinetic else 0.4,
            metadata={"component": comp.name, "part_id": comp.part_id},
        )
        world.add_body(body)
        component_bodies[comp.id] = body.id

    for constraint in assembly.constraints:
        if not constraint.enabled or constraint.suppressed:
            continue
        kind = _CONSTRAINT_TO_JOINT.get(constraint.type)
        if kind is None:
            continue
        a_id = constraint.a.component_id
        b_id = constraint.b.component_id if constraint.b else None
        if a_id not in world.bodies:
            continue
        if b_id is None or b_id not in world.bodies:
            # mate to the world: treat A as a fixed body and the joint to the world
            body_a, body_b = None, world.bodies[a_id]
        else:
            body_a, body_b = world.bodies[a_id], world.bodies[b_id]
        moving = b_id if b_id in assembly.components else a_id
        mech = assembly.components[moving].mechanical if moving in assembly.components else None
        axis_local = constraint.a.direction
        # constraint points are world points; joint anchors are relative to each body's COM
        anchor_a = (as_vec3(constraint.a.point) - (body_a.position if body_a is not None
                                                   else world.bodies[a_id].position))
        anchor_b = (as_vec3(constraint.b.point if constraint.b else constraint.a.point)
                    - body_b.position)
        joint = Joint(
            id=constraint.id,
            name=constraint.name or constraint.type.label(),
            kind=kind,
            body_a=body_a.id if body_a is not None else None,
            body_b=body_b.id,
            anchor_a=anchor_a,
            anchor_b=anchor_b,
            axis_a=axis_local,
            axis_b=(constraint.b.direction if constraint.b else axis_local),
            stiffness=mech.stiffness if mech else 0.0,
            damping=mech.damping if mech else 0.0,
            friction_static=mech.friction_static if mech else 0.0,
            friction_kinetic=mech.friction_kinetic if mech else 0.0,
            backlash=mech.backlash if mech else 0.0,
            limits=((mech.joint_min, mech.joint_max)
                    if mech and (mech.joint_min or mech.joint_max) else None),
            motor_max_torque=mech.max_torque if mech else 0.0,
            motor_max_speed=mech.max_speed if mech else 0.0,
            lead=float(constraint.value) if constraint.type is ConstraintType.SCREW and constraint.value > 0 else 0.0,
            metadata={"constraint_type": constraint.type.value},
        )
        world.add_joint(joint)

        # bind servos declared on either component (prefer the driven body B)
        if enable_servos:
            for cid in (b_id, a_id):
                source = _servo_source(cid, assembly)
                if source is None:
                    continue
                spec = hardware.servo(source.hardware_id or "mg90s")
                plant = ServoPlant(spec)
                joint.servo = plant
                joint.metadata["servo_component"] = source.id
                joint.metadata.setdefault("tvc_channel", source.metadata.get("servo_for", ""))
                joint.servo_command = 0.0
                servos[source.id] = plant
                break

    if not servos and enable_servos:
        diagnostics.append(Diagnostic("info", "no-servos",
                                      "No servo components are assembled; mechanical motion is unactuated."))
    # bench fixture: rigidly mounted structure is fixed and gravity is carried by the mounts
    world.gravity_enabled = False
    diagnostics.append(Diagnostic(
        "info", "bench-fixture",
        "Mechanical simulation treats components without mates as rigidly mounted and disables "
        "gravity; use it as a servo/mechanism bench test, not a free-flight simulation."))
    diagnostics.extend(world.validate())
    return WorldBuildResult(world=world, servos=servos, diagnostics=diagnostics,
                            mass_properties=mass_result, component_bodies=component_bodies)


def _servo_source(component_id: str, assembly: Assembly):
    """A component that provides the servo for a joint (itself or a referenced part)."""
    if component_id is None:
        return None
    comp = assembly.components.get(component_id)
    if comp is None:
        return None
    if comp.hardware_type == "servo":
        return comp
    referenced = comp.metadata.get("servo_component_id")
    if referenced:
        candidate = assembly.components.get(referenced)
        if candidate is not None and candidate.hardware_type == "servo":
            return candidate
    return None


def apply_tvc_commands(world: DynamicsWorld, channels: dict[str, float]) -> None:
    """Assign servo commands by TVC channel ('pitch', 'yaw', or joint name)."""
    for joint in world.joints.values():
        key = joint.metadata.get("tvc_channel") or joint.name.lower()
        for channel, value in channels.items():
            if channel.lower() in key or joint.id == channel:
                world.set_joint_target(joint.id, value)
                break


def describe(world: DynamicsWorld) -> list[dict]:
    rows = []
    for bid, body in world.bodies.items():
        rows.append({
            "id": bid, "name": body.name, "mass_g": body.mass * 1000.0,
            "fixed": body.fixed, "inertia": np.diag(body.inertia_local).tolist(),
        })
    return rows
