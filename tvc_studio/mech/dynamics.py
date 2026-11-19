"""Rigid-body mechanical dynamics with joint-level constraints.

A sequential-impulse solver in world coordinates. Bodies are rigid with mass, centre of
mass and an inertia tensor; joints are velocity constraints with Baumgarte position bias,
optional limits, friction, backlash and torque-limited motors. Contact uses convex proxies
from :mod:`mech.contact`.

Why not an off-the-shelf engine: the TVC mechanism needs screw joints coupling rotation to
axial advance, lash (backlash) in the drive train, per-joint friction with static/kinetic
thresholds, and servo plants whose reported torque must equal the torque actually applied.
Those are first-class here and are validated by the tests in ``tests/test_dynamics.py``.

Approximations (documented, never hidden): single-point contact per body pair, no
deformation, no joint compliance beyond an explicit spring/damper, and no thermal effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np

from ..core.interfaces import Diagnostic, Mesh
from ..core.math3d import GRAVITY, Quat, Transform, as_vec3, unit, vec3
from .contact import CollisionFilter, ConvexProxy, ContactInfo, sat_contact
from .servo import ServoPlant


class JointKind(str, Enum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    CYLINDRICAL = "cylindrical"
    BALL = "ball"
    SCREW = "screw"
    PLANAR = "planar"

    def label(self) -> str:
        return {
            "fixed": "Fixed", "revolute": "Revolute", "prismatic": "Prismatic",
            "cylindrical": "Cylindrical", "ball": "Ball", "screw": "Screw", "planar": "Planar",
        }[self.value]


def _rotvec(q: Quat) -> np.ndarray:
    qq = q.normalized()
    if qq.w < 0:
        qq = Quat(-qq.w, -qq.x, -qq.y, -qq.z)
    s = math.sqrt(qq.x * qq.x + qq.y * qq.y + qq.z * qq.z)
    if s < 1e-12:
        return vec3()
    return qq.vec / s * (2.0 * math.atan2(s, qq.w))


@dataclass
class Body:
    """A rigid body; ``position`` is the world position of its centre of mass."""

    id: str
    name: str
    mass: float
    inertia_local: np.ndarray = field(default_factory=lambda: np.diag([1e-4, 1e-4, 1e-4]))
    com_local: np.ndarray = field(default_factory=lambda: vec3())
    position: np.ndarray = field(default_factory=lambda: vec3())
    rotation: Quat = field(default_factory=Quat.identity)
    velocity: np.ndarray = field(default_factory=lambda: vec3())
    angular_velocity: np.ndarray = field(default_factory=lambda: vec3())
    fixed: bool = False
    collision_mesh: Mesh | None = None
    collision_proxy: ConvexProxy | None = field(default=None, repr=False)
    restitution: float = 0.2
    friction: float = 0.4
    linear_damping: float = 0.0
    angular_damping: float = 0.02
    contact_enabled: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mass = max(float(self.mass), 1e-9)
        self.inertia_local = np.asarray(self.inertia_local, dtype=float).reshape(3, 3)
        self.position = as_vec3(self.position)
        if self.collision_mesh is not None and self.collision_proxy is None:
            self.collision_proxy = ConvexProxy.from_mesh(self.collision_mesh)

    @property
    def inv_mass(self) -> float:
        return 0.0 if self.fixed else 1.0 / self.mass

    @property
    def transform(self) -> Transform:
        return Transform(self.rotation, self.position - self.rotation.rotate(self.com_local))

    def set_transform(self, transform: Transform) -> None:
        self.rotation = transform.rotation
        self.position = transform.apply_point(self.com_local)

    def inertia_world(self) -> np.ndarray:
        R = self.rotation.matrix()
        return R @ self.inertia_local @ R.T

    def inv_inertia_world(self) -> np.ndarray:
        if self.fixed:
            return np.zeros((3, 3))
        R = self.rotation.matrix()
        try:
            inv_local = np.linalg.inv(self.inertia_local + np.eye(3) * 1e-12)
        except np.linalg.LinAlgError:
            inv_local = np.diag([1.0 / max(v, 1e-9) for v in np.diag(self.inertia_local)])
        return R @ inv_local @ R.T

    def point_velocity(self, world_point: np.ndarray) -> np.ndarray:
        r = as_vec3(world_point) - self.position
        return self.velocity + np.cross(self.angular_velocity, r)

    def apply_impulse(self, impulse: np.ndarray, world_point: np.ndarray) -> None:
        if self.fixed:
            return
        self.velocity = self.velocity + as_vec3(impulse) / self.mass
        r = as_vec3(world_point) - self.position
        L = np.cross(r, as_vec3(impulse))
        self.angular_velocity = self.angular_velocity + self.inv_inertia_world() @ L

    def apply_force(self, force: np.ndarray, world_point: np.ndarray | None = None) -> None:
        if self.fixed:
            return
        self.velocity = self.velocity + as_vec3(force) / self.mass
        if world_point is not None:
            r = as_vec3(world_point) - self.position
            self.angular_velocity = self.angular_velocity + self.inv_inertia_world() @ np.cross(r, as_vec3(force))

    def apply_torque(self, torque: np.ndarray) -> None:
        if not self.fixed:
            self.angular_velocity = self.angular_velocity + self.inv_inertia_world() @ as_vec3(torque)

    def kinetic_energy(self) -> float:
        I = self.inertia_world()
        w = self.angular_velocity
        return 0.5 * self.mass * float(self.velocity @ self.velocity) + 0.5 * float(w @ (I @ w))


@dataclass
class Joint:
    """A mechanical joint. ``body_a`` may be None for a joint to the world."""

    id: str
    name: str
    kind: JointKind
    body_a: str | None
    body_b: str
    anchor_a: np.ndarray = field(default_factory=lambda: vec3())
    anchor_b: np.ndarray = field(default_factory=lambda: vec3())
    axis_a: np.ndarray = field(default_factory=lambda: vec3(0, 0, 1))
    axis_b: np.ndarray = field(default_factory=lambda: vec3(0, 0, 1))
    limits: tuple[float, float] | None = None
    stiffness: float = 0.0
    damping: float = 0.0
    friction_static: float = 0.0
    friction_kinetic: float = 0.0
    backlash: float = 0.0
    lead: float = 0.0                 # screw lead, m per revolution
    motor_enabled: bool = False
    motor_target: float = 0.0         # rad or m (joint coordinate)
    motor_max_torque: float = 0.0     # 0 = unlimited
    motor_max_speed: float = 0.0
    motor_kp: float = 40.0
    motor_kd: float = 0.4
    servo: ServoPlant | None = None
    servo_command: float = 0.0
    transmission_ratio: float = 1.0    # servo angle -> joint angle
    externally_applied_torque: float = 0.0
    metadata: dict = field(default_factory=dict)

    __hash__ = object.__hash__

    def axis_world(self, bodies: dict[str, Body]) -> np.ndarray:
        if self.body_a is None:
            return unit(self.axis_a)
        body = bodies[self.body_a]
        return unit(body.rotation.rotate(self.axis_a))

    def anchor_world(self, bodies: dict[str, Body], which: str = "a") -> np.ndarray:
        """World position of a joint anchor. Anchors are stored relative to body COM."""
        if which == "a":
            if self.body_a is None:
                return as_vec3(self.anchor_a)
            body = bodies[self.body_a]
            return body.position + body.rotation.rotate(self.anchor_a)
        body = bodies[self.body_b]
        return body.position + body.rotation.rotate(self.anchor_b)

    def coordinate(self, bodies: dict[str, Body]) -> float:
        """Free coordinate: angle (rad), distance (m) or screw rotation (rad)."""
        a = bodies.get(self.body_a) if self.body_a else None
        b = bodies[self.body_b]
        pa = self.anchor_world(bodies, "a")
        pb = self.anchor_world(bodies, "b")
        axis = self.axis_world(bodies)
        if self.kind in (JointKind.REVOLUTE, JointKind.SCREW):
            qa = a.rotation if a is not None else Quat.identity()
            qrel = b.rotation * qa.inverse()
            return 2.0 * math.atan2(float(np.dot(qrel.vec, axis)), qrel.w)
        if self.kind is JointKind.PRISMATIC:
            return float(np.dot(pb - pa, axis))
        if self.kind is JointKind.CYLINDRICAL:
            qa = a.rotation if a is not None else Quat.identity()
            qrel = b.rotation * qa.inverse()
            return 2.0 * math.atan2(float(np.dot(qrel.vec, axis)), qrel.w)
        return 0.0

    def coordinate_velocity(self, bodies: dict[str, Body]) -> float:
        a = bodies.get(self.body_a) if self.body_a else None
        b = bodies[self.body_b]
        axis = self.axis_world(bodies)
        wa = a.angular_velocity if a is not None else vec3()
        wb = b.angular_velocity
        if self.kind in (JointKind.REVOLUTE, JointKind.SCREW, JointKind.CYLINDRICAL):
            return float(np.dot(wb - wa, axis))
        if self.kind is JointKind.PRISMATIC:
            pa = self.anchor_world(bodies, "a")
            pb = self.anchor_world(bodies, "b")
            va = a.point_velocity(pa) if a is not None else vec3()
            vb = b.point_velocity(pb)
            return float(np.dot(vb - va, axis))
        return 0.0


@dataclass
class ConstraintRow:
    """One scalar velocity constraint."""

    body_a: Body | None
    body_b: Body
    normal: np.ndarray                 # linear direction (zero for pure angular rows)
    angular: np.ndarray                # angular direction
    ra: np.ndarray = field(default_factory=lambda: vec3())
    rb: np.ndarray = field(default_factory=lambda: vec3())
    bias: float = 0.0
    target_velocity: float = 0.0
    lower: float = -np.inf
    upper: float = np.inf
    accumulated: float = 0.0
    label: str = ""

    def jacobian_velocity(self, use_clockwise: bool = False) -> float:
        vb = self.body_b
        wb = vb.angular_velocity
        Jv = float(np.dot(self.normal, vb.velocity) + np.dot(self.normal, np.cross(wb, self.rb))
                   + np.dot(self.angular, wb))
        if self.body_a is not None:
            va = self.body_a
            Jv -= float(np.dot(self.normal, va.velocity) + np.dot(self.normal, np.cross(va.angular_velocity, self.ra))
                        + np.dot(self.angular, va.angular_velocity))
        return Jv

    def effective_mass(self) -> float:
        k = 0.0
        b = self.body_b
        k += b.inv_mass
        if np.any(self.normal):
            rbxn = np.cross(self.rb, self.normal)
            k += float(rbxn @ (b.inv_inertia_world() @ rbxn))
        if np.any(self.angular):
            k += float(self.angular @ (b.inv_inertia_world() @ self.angular))
        if self.body_a is not None:
            a = self.body_a
            k += a.inv_mass
            if np.any(self.normal):
                raxn = np.cross(self.ra, self.normal)
                k += float(raxn @ (a.inv_inertia_world() @ raxn))
            if np.any(self.angular):
                k += float(self.angular @ (a.inv_inertia_world() @ self.angular))
        return k

    def apply(self, impulse: float) -> None:
        if abs(impulse) < 1e-15:
            return
        P = self.normal * impulse
        b = self.body_b
        if not b.fixed:
            b.velocity = b.velocity + P * b.inv_mass
            L = np.cross(self.rb, P) + self.angular * impulse
            b.angular_velocity = b.angular_velocity + b.inv_inertia_world() @ L
        if self.body_a is not None and not self.body_a.fixed:
            a = self.body_a
            a.velocity = a.velocity - P * a.inv_mass
            L = np.cross(self.ra, P) + self.angular * impulse
            a.angular_velocity = a.angular_velocity - a.inv_inertia_world() @ L


@dataclass
class StepRecord:
    time: float
    coordinates: dict[str, float]
    torques: dict[str, float]
    energies: dict[str, float]
    contacts: list[ContactInfo] = field(default_factory=list)
    servo_states: dict[str, dict] = field(default_factory=dict)


class DynamicsWorld:
    """Joint-constrained rigid-body world."""

    def __init__(self, name: str = "Mechanical world", gravity_enabled: bool = True,
                 gravity: float = GRAVITY, solvers: int = 12) -> None:
        self.name = name
        self.bodies: dict[str, Body] = {}
        self.joints: dict[str, Joint] = {}
        self.gravity_enabled = gravity_enabled
        self.gravity_vector = vec3(0, 0, -gravity) if gravity_enabled else vec3()
        self.dt = 0.001
        self.iterations = solvers
        self.baumgarte = 0.2
        self.time = 0.0
        self.step_count = 0
        self.contacts_enabled = True
        self.collision_filter = CollisionFilter()
        self.on_contact: list[ContactInfo] = []
        self.records: list[StepRecord] = []
        self.record_interval = 1
        self.keep_records = True
        self.diagnostics: list[Diagnostic] = []
        self._locked = False

    # ------------------------------------------------------------------ setup
    def add_body(self, body: Body) -> Body:
        if body.id in self.bodies:
            raise ValueError(f"duplicate body id: {body.id}")
        self.bodies[body.id] = body
        return body

    def add_joint(self, joint: Joint) -> Joint:
        if joint.body_b not in self.bodies:
            raise ValueError(f"joint body missing: {joint.body_b}")
        if joint.body_a is not None and joint.body_a not in self.bodies:
            raise ValueError(f"joint body missing: {joint.body_a}")
        self.joints[joint.id] = joint
        return joint

    def body(self, body_id: str) -> Body:
        return self.bodies[body_id]

    def joint(self, joint_id: str) -> Joint:
        return self.joints[joint_id]

    def joints_for(self, body_id: str) -> list[Joint]:
        return [j for j in self.joints.values() if j.body_a == body_id or j.body_b == body_id]

    # ------------------------------------------------------------------- rows
    def _world_anchor(self, body: Body | None, local: np.ndarray, com: np.ndarray) -> np.ndarray:
        if body is None:
            return as_vec3(local)
        world = body.transform.apply_point(local)
        return world

    def _rows_for_joint(self, joint: Joint) -> list[ConstraintRow]:
        b = self.bodies[joint.body_b]
        a = self.bodies.get(joint.body_a) if joint.body_a else None
        pa = joint.anchor_world(self.bodies, "a")
        pb = joint.anchor_world(self.bodies, "b")
        axis = joint.axis_world(self.bodies)
        ra = pa - a.position if a is not None else vec3()
        rb = pb - b.position
        rows: list[ConstraintRow] = []
        dt = max(self.dt, 1e-9)
        beta = self.baumgarte / dt

        def point_row(n: np.ndarray, limit: float | None = None) -> ConstraintRow:
            n = unit(n)
            error = float(np.dot(n, pa - pb))
            return ConstraintRow(a, b, n, vec3(), ra, rb, bias=beta * (0.0 if limit is not None else error),
                                 label=f"{joint.name}:point")

        # perpendicular directions to the joint axis
        helper = vec3(1, 0, 0) if abs(axis[0]) < 0.9 else vec3(0, 1, 0)
        e1 = unit(np.cross(axis, helper))
        e2 = unit(np.cross(axis, e1))

        def angular_row(direction: np.ndarray, bias_angle: float = 0.0) -> ConstraintRow:
            return ConstraintRow(a, b, vec3(), unit(direction), vec3(), vec3(),
                                 bias=beta * bias_angle, label=f"{joint.name}:angle")

        if joint.kind in (JointKind.FIXED, JointKind.BALL, JointKind.REVOLUTE, JointKind.CYLINDRICAL):
            rows.extend([point_row(vec3(1, 0, 0)), point_row(vec3(0, 1, 0)), point_row(vec3(0, 0, 1))])
        elif joint.kind is JointKind.PRISMATIC:
            rows.extend([point_row(e1), point_row(e2)])
        elif joint.kind is JointKind.PLANAR:
            rows.append(point_row(axis))
        elif joint.kind is JointKind.SCREW:
            rows.extend([point_row(e1), point_row(e2)])

        # orientation
        qa = a.rotation if a is not None else Quat.identity()
        qrel = b.rotation * qa.inverse()
        rot_err = _rotvec(qrel)
        if joint.kind is JointKind.FIXED:
            rows.extend([angular_row(e1, float(np.dot(rot_err, e1))),
                         angular_row(e2, float(np.dot(rot_err, e2))),
                         angular_row(axis, float(np.dot(rot_err, axis)))])
        elif joint.kind in (JointKind.REVOLUTE, JointKind.CYLINDRICAL, JointKind.PRISMATIC,
                            JointKind.SCREW, JointKind.PLANAR):
            err1 = float(np.dot(rot_err, e1))
            err2 = float(np.dot(rot_err, e2))
            rows.extend([angular_row(e1, err1), angular_row(e2, err2)])
        elif joint.kind is JointKind.BALL:
            pass

        # screw coupling: axial velocity and twist are locked by the lead
        if joint.kind is JointKind.SCREW and joint.lead:
            row = ConstraintRow(a, b, axis, axis * (joint.lead / (2 * math.pi)), ra, rb,
                                label=f"{joint.name}:screw")
            rows.append(row)

        # limits on the free coordinate
        if joint.limits is not None and joint.kind in (JointKind.REVOLUTE, JointKind.PRISMATIC,
                                                       JointKind.CYLINDRICAL):
            coord = joint.coordinate(self.bodies)
            lo, hi = joint.limits
            free_axis = axis if joint.kind is JointKind.PRISMATIC else axis
            if coord < lo:
                r = ConstraintRow(a, b, free_axis if joint.kind is JointKind.PRISMATIC else vec3(),
                                  vec3() if joint.kind is JointKind.PRISMATIC else free_axis,
                                  ra, rb, bias=beta * (coord - lo),
                                  lower=0.0, upper=np.inf, label=f"{joint.name}:limit-low")
                rows.append(r)
            elif coord > hi:
                r = ConstraintRow(a, b, free_axis if joint.kind is JointKind.PRISMATIC else vec3(),
                                  vec3() if joint.kind is JointKind.PRISMATIC else free_axis,
                                  ra, rb, bias=beta * (coord - hi),
                                  lower=-np.inf, upper=0.0, label=f"{joint.name}:limit-high")
                rows.append(r)
        return rows

    def _contact_rows(self) -> list[ConstraintRow]:
        if not self.contacts_enabled:
            return []
        rows: list[ConstraintRow] = []
        ids = [bid for bid, b in self.bodies.items() if b.contact_enabled and b.collision_proxy]
        contacts: list[ContactInfo] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                if self.collision_filter.is_allowed(id_a, id_b):
                    continue
                ba, bb = self.bodies[id_a], self.bodies[id_b]
                if ba.fixed and bb.fixed:
                    continue
                info = sat_contact(ba.collision_proxy, ba.transform, bb.collision_proxy, bb.transform)
                if info is None:
                    continue
                info.body_a = id_a
                info.body_b = id_b
                contacts.append(info)
                normal = info.normal  # from B towards A
                point = info.point
                ra = point - ba.position
                rb = point - bb.position
                approach = float(np.dot(ba.point_velocity(point) - bb.point_velocity(point), normal))
                restitution = min(ba.restitution, bb.restitution)
                bias = self.baumgarte / max(self.dt, 1e-9) * min(info.penetration, 0.01)
                target = 0.0 if approach > -1e-3 else -restitution * approach
                rows.append(ConstraintRow(
                    ba, bb, -normal, vec3(), ra, rb, bias=bias, target_velocity=-target,
                    lower=0.0, upper=np.inf, label="contact",
                ))
                # friction rows across the contact plane
                helper = vec3(1, 0, 0) if abs(normal[0]) < 0.9 else vec3(0, 1, 0)
                t1 = unit(np.cross(normal, helper))
                t2 = unit(np.cross(normal, t1))
                mu = math.sqrt(max(ba.friction * bb.friction, 0.0))
                for t in (t1, t2):
                    rows.append(ConstraintRow(
                        ba, bb, -t, vec3(), ra, rb,
                        lower=-mu, upper=mu, label="contact-friction",
                    ))
        self.on_contact = contacts
        return rows

    # -------------------------------------------------------------------- step
    def step(self, dt: float | None = None) -> None:
        if self._locked:
            raise RuntimeError("step() is not re-entrant")
        self._locked = True
        try:
            dt = float(dt if dt is not None else self.dt)
            self.dt = dt
            self._apply_external(dt)
            # actuators (servos/motors) act before the constraint solve so that the joint
            # constraints can preserve the allowed motion instead of fighting it afterwards
            self._update_joint_motors(dt)
            rows = []
            for joint in self.joints.values():
                if joint.servo is not None:
                    # prescribed motion: the servo drive advances this joint kinematically
                    continue
                rows.extend(self._rows_for_joint(joint))
            rows.extend(self._contact_rows())
            # tentatively integrate velocity then solve constraints
            for body in self.bodies.values():
                if body.fixed:
                    continue
                body.velocity = body.velocity * (1.0 - min(body.linear_damping * dt, 1.0))
                body.angular_velocity = body.angular_velocity * (1.0 - min(body.angular_damping * dt, 1.0))
            for _ in range(self.iterations):
                for row in rows:
                    Jv = row.jacobian_velocity()
                    k = row.effective_mass()
                    if k <= 1e-15:
                        continue
                    lam = -(Jv - row.target_velocity + row.bias) / k
                    new_acc = row.accumulated + lam
                    clamped = min(max(new_acc, row.lower), row.upper)
                    delta = clamped - row.accumulated
                    row.accumulated = clamped
                    row.apply(delta)
            for body in self.bodies.values():
                if body.fixed:
                    continue
                body.position = body.position + body.velocity * dt
                dq = body.rotation.derivative(body.rotation.rotate_inverse(body.angular_velocity))
                body.rotation = Quat(body.rotation.w + dq.w * dt, body.rotation.x + dq.x * dt,
                                     body.rotation.y + dq.y * dt, body.rotation.z + dq.z * dt).normalized()
            self._drive_servo_joints(dt)
            self.time += dt
            self.step_count += 1
            if self.keep_records and self.step_count % max(self.record_interval, 1) == 0:
                self.records.append(self._record())
        finally:
            self._locked = False

    def _apply_external(self, dt: float) -> None:
        for body in self.bodies.values():
            if body.fixed:
                continue
            body.velocity = body.velocity + self.gravity_vector * dt
        # servo torques are applied by _update_joint_motors after the solve; external
        # torques and joint springs are applied here so they influence this step
        for joint in self.joints.values():
            if joint.externally_applied_torque:
                self._apply_joint_torque(joint, joint.externally_applied_torque)
            if joint.stiffness or joint.damping:
                coord = joint.coordinate(self.bodies)
                coord_vel = joint.coordinate_velocity(self.bodies)
                torque = -joint.stiffness * coord - joint.damping * coord_vel
                self._apply_joint_torque(joint, torque)

    def _apply_joint_torque(self, joint: Joint, torque: float) -> None:
        """Apply a torque/force along the joint's free coordinate."""
        b = self.bodies[joint.body_b]
        a = self.bodies.get(joint.body_a) if joint.body_a else None
        axis = joint.axis_world(self.bodies)
        impulse = torque * self.dt
        if joint.kind is JointKind.PRISMATIC:
            force = axis * impulse
            b.apply_force(force)
            if a is not None and not a.fixed:
                a.apply_force(-force)
        else:
            if not b.fixed:
                b.angular_velocity = b.angular_velocity + b.inv_inertia_world() @ (axis * impulse)
            if a is not None and not a.fixed:
                a.angular_velocity = a.angular_velocity - a.inv_inertia_world() @ (axis * impulse)

    def _update_joint_motors(self, dt: float) -> None:
        """Drive joints that have a motor or servo; apply friction and backlash."""
        for joint in self.joints.values():
            coord = joint.coordinate(self.bodies)
            vel = joint.coordinate_velocity(self.bodies)
            if joint.servo is not None:
                # servo-driven joints are advanced kinematically in _drive_servo_joints();
                # here only friction below applies
                pass
            elif joint.motor_enabled:
                error = joint.motor_target - coord
                if joint.backlash > 0:
                    if abs(error) <= joint.backlash / 2.0:
                        error = 0.0
                    else:
                        error -= math.copysign(joint.backlash / 2.0, error)
                speed_ref = joint.motor_kp * error
                if joint.motor_max_speed > 0:
                    speed_ref = float(np.clip(speed_ref, -joint.motor_max_speed, joint.motor_max_speed))
                torque = joint.motor_kd * (speed_ref - vel)
                if joint.motor_max_torque > 0:
                    torque = float(np.clip(torque, -joint.motor_max_torque, joint.motor_max_torque))
                self._apply_joint_torque(joint, torque)
                joint.metadata["motor_torque"] = torque
            # joint friction as a velocity-damping impulse with a Coulomb limit
            if joint.friction_kinetic > 0 or joint.friction_static > 0:
                impulse_limit = max(joint.friction_kinetic, 0.0) * dt
                if abs(vel) < 1e-6:
                    impulse_limit = max(joint.friction_static, joint.friction_kinetic) * dt
                if impulse_limit > 0:
                    b = self.bodies[joint.body_b]
                    a = self.bodies.get(joint.body_a) if joint.body_a else None
                    axis = joint.axis_world(self.bodies)
                    iv = (b.inv_inertia_world() @ axis) if not b.fixed else vec3()
                    if a is not None and not a.fixed:
                        iv = iv + (a.inv_inertia_world() @ axis)
                    k = float(axis @ iv)
                    if k > 1e-15:
                        impulse = float(np.clip(-vel / k, -impulse_limit, impulse_limit))
                        if not b.fixed:
                            b.angular_velocity = b.angular_velocity + (b.inv_inertia_world() @ (axis * impulse))
                        if a is not None and not a.fixed:
                            a.angular_velocity = a.angular_velocity - (a.inv_inertia_world() @ (axis * impulse))

    def _carry_rigid_children(self, body_id: str, rot: Quat, pivot: np.ndarray,
                              visited: set[str]) -> None:
        if body_id in visited:
            return
        visited.add(body_id)
        for joint in self.joints.values():
            if joint.kind is not JointKind.FIXED or joint.body_a != body_id:
                continue
            child = self.bodies.get(joint.body_b)
            if child is None or child.fixed:
                continue
            child.position = pivot + rot.rotate(child.position - pivot)
            child.rotation = (rot * child.rotation).normalized()
            child.velocity = vec3()
            child.angular_velocity = vec3()
            self._carry_rigid_children(child.id, rot, pivot, visited)

    def _drive_servo_joints(self, dt: float) -> None:
        """Advance servo-driven revolute joints from the servo's achieved output.

        A hobby servo has a very stiff geared output: treating it as a torque source makes the
        joint dynamics far stiffer than any practical timestep. It is therefore driven
        kinematically (the horn angle *is* the joint angle within the lash gap), while the servo
        model supplies latency, deadband, backlash, slew rate and a torque estimate with
        saturation. Torque estimates are reported for telemetry, not integrated as a spring.
        """
        for joint in self.joints.values():
            plant = joint.servo
            if plant is None or joint.kind is not JointKind.REVOLUTE:
                continue
            b = self.bodies[joint.body_b]
            a = self.bodies.get(joint.body_a) if joint.body_a else None
            if b.fixed:
                continue
            ratio = max(joint.transmission_ratio, 1e-9)
            coord = joint.coordinate(self.bodies)
            vel = joint.coordinate_velocity(self.bodies)
            plant.command(joint.servo_command / ratio)
            plant.state.output_angle = coord * ratio
            plant.state.output_speed = vel * ratio
            state = plant.step(dt, load_torque=0.0)
            target = state.applied_command * ratio

            delta = target - coord
            if joint.backlash > 0:
                if abs(delta) <= joint.backlash / 2.0:
                    delta = 0.0
                else:
                    delta -= math.copysign(joint.backlash / 2.0, delta)
            max_speed = abs(plant.spec.no_load_speed * plant.spec.voltage_scale(
                plant.supply_voltage)) / ratio
            bandwidth = max_speed / max(0.05, plant.spec.travel() * 0.25)
            speed = float(np.clip(bandwidth * delta, -max_speed, max_speed))
            if abs(delta) < 1e-6:
                speed = 0.0
            step_angle = speed * dt
            if abs(step_angle) < 1e-12:
                state.torque = 0.0
                state.output_angle = coord * ratio
                state.output_speed = 0.0
                continue

            axis = joint.axis_world(self.bodies)
            rot = Quat.from_axis_angle(axis, step_angle)
            b.rotation = (rot * b.rotation).normalized()
            # keep the hinge anchor coincident
            pa = joint.anchor_world(self.bodies, "a")
            pb = joint.anchor_world(self.bodies, "b")
            b.position = b.position + (pa - pb)
            # the whole step's rotation was applied above: do not let the integrator add it again
            b.angular_velocity = vec3()
            if a is not None and not a.fixed:
                a.angular_velocity = vec3()
            # carry rigidly attached descendants with the driven body, so their attachment
            # constraints see a consistent state instead of chasing it
            self._carry_rigid_children(b.id, rot, pa, set())

            # torque estimate: the inertia that had to be accelerated, clamped to stall
            inertia = 0.0
            if not b.fixed:
                inertia += float(axis @ (b.inertia_world() @ axis))
            if a is not None and not a.fixed:
                inertia += float(axis @ (a.inertia_world() @ axis))
            torque = inertia * (speed - vel) / max(dt, 1e-9)
            stall = plant.spec.stall_torque * plant.spec.voltage_scale(plant.supply_voltage) / ratio
            torque = float(np.clip(torque, -stall, stall))
            state.torque = torque
            state.saturated = abs(torque) >= stall - 1e-12
            state.current = plant.spec.idle_current + abs(torque) / max(plant.spec.kt(), 1e-9)
            state.output_angle = joint.coordinate(self.bodies) * ratio
            state.output_speed = speed
            joint.metadata["servo_torque"] = torque
            joint.metadata["servo_current"] = state.current

    # ------------------------------------------------------------------ output
    def _record(self) -> StepRecord:
        coords = {jid: j.coordinate(self.bodies) for jid, j in self.joints.items()}
        torques = {jid: float(j.metadata.get("servo_torque", j.metadata.get("motor_torque", 0.0)))
                   for jid, j in self.joints.items()}
        energy = self.energy()
        servos = {jid: j.servo.summary() for jid, j in self.joints.items() if j.servo is not None}
        return StepRecord(self.time, coords, torques, energy, list(self.on_contact), servos)

    def kinetic_energy(self) -> float:
        return float(sum(b.kinetic_energy() for b in self.bodies.values()))

    def potential_energy(self) -> float:
        e = 0.0
        for b in self.bodies.values():
            I = b.inertia_world()
            # gravitational potential uses the COM height
            e += b.mass * 9.80665 * float(b.position[2])
        return e

    def energy(self) -> dict[str, float]:
        return {"kinetic": self.kinetic_energy(), "potential": self.potential_energy(),
                "total": self.kinetic_energy() + self.potential_energy()}

    def run(self, duration: float, dt: float | None = None) -> list[StepRecord]:
        dt = dt or self.dt
        steps = max(int(round(duration / dt)), 1)
        for _ in range(steps):
            self.step(dt)
        return self.records

    def reset_time(self) -> None:
        self.time = 0.0
        self.step_count = 0
        self.records.clear()

    def set_joint_target(self, joint_id: str, value: float) -> None:
        joint = self.joints[joint_id]
        if joint.servo is not None:
            joint.servo_command = float(value)
        else:
            joint.motor_target = float(value)
            joint.motor_enabled = True

    # -------------------------------------------------------------- diagnostics
    def validate(self) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        for bid, body in self.bodies.items():
            if body.mass <= 0:
                out.append(Diagnostic("error", "zero-mass", f"'{body.name}' has zero mass"))
            evals = np.linalg.eigvalsh(body.inertia_local)
            if np.any(evals <= 0):
                out.append(Diagnostic("warning", "inertia", f"'{body.name}' has a non-positive inertia principal value"))
        for jid, joint in self.joints.items():
            if joint.kind is JointKind.SCREW and joint.lead <= 0:
                out.append(Diagnostic("error", "screw-lead", f"Screw joint '{joint.name}' needs a positive lead"))
            if joint.limits and joint.limits[1] <= joint.limits[0]:
                out.append(Diagnostic("error", "joint-limits", f"Joint '{joint.name}' limits are inverted"))
        if not any(b.fixed for b in self.bodies.values()):
            out.append(Diagnostic("warning", "no-ground-body", "No fixed body: the mechanism can float"))
        return out

    def summary(self) -> dict:
        return {
            "bodies": len(self.bodies),
            "joints": len(self.joints),
            "time": self.time,
            "steps": self.step_count,
            "kinetic_energy": self.kinetic_energy(),
            "potential_energy": self.potential_energy(),
            "contacts": len(self.on_contact),
        }
