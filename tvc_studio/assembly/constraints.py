"""Assembly constraints and their solver.

Mate geometry is expressed with :class:`EntityRef`, which carries a point and a direction
in the owning component's local frame. Constraints return residual vectors; the solver
minimises them with SciPy's least-squares and then reports degrees of freedom, redundancy
and conflicts from the Jacobian's rank, so "over-constrained" is a measured fact rather
than a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import least_squares

from ..core.interfaces import Diagnostic
from ..core.math3d import Quat, Transform, as_vec3, unit, vec3
from .model import Assembly, ComponentInstance


class ConstraintType(str, Enum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    SLIDER = "slider"
    CYLINDRICAL = "cylindrical"
    BALL = "ball"
    PLANAR = "planar"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    CONCENTRIC = "concentric"
    COINCIDENT = "coincident"
    DISTANCE = "distance"
    ANGLE = "angle"
    SCREW = "screw"

    def label(self) -> str:
        return {
            "fixed": "Fixed",
            "revolute": "Revolute (hinge)",
            "slider": "Slider",
            "cylindrical": "Cylindrical",
            "ball": "Ball joint",
            "planar": "Planar",
            "parallel": "Parallel",
            "perpendicular": "Perpendicular",
            "concentric": "Concentric",
            "coincident": "Coincident",
            "distance": "Distance",
            "angle": "Angle",
            "screw": "Screw",
        }[self.value]

    def dof_removed(self) -> int:
        return {
            "fixed": 6, "revolute": 5, "slider": 5, "cylindrical": 4, "ball": 3,
            "planar": 3, "parallel": 2, "perpendicular": 1, "concentric": 4,
            "coincident": 3, "distance": 1, "angle": 1, "screw": 5,
        }[self.value]


class EntityKind(str, Enum):
    ORIGIN = "origin"
    POINT = "point"
    AXIS = "axis"
    PLANE = "plane"
    FACE = "face"
    EDGE = "edge"
    VERTEX = "vertex"


@dataclass
class EntityRef:
    """A geometric reference on a component, in that component's local frame."""

    component_id: str
    kind: EntityKind = EntityKind.ORIGIN
    point: np.ndarray = field(default_factory=lambda: vec3())
    direction: np.ndarray = field(default_factory=lambda: vec3(0, 0, 1))
    label: str = ""

    def __post_init__(self) -> None:
        self.point = as_vec3(self.point)
        d = as_vec3(self.direction)
        n = np.linalg.norm(d)
        self.direction = d / n if n > 1e-12 else vec3(0, 0, 1)

    def world(self, pose: Transform) -> tuple[np.ndarray, np.ndarray]:
        return pose.apply_point(self.point), pose.apply_vector(self.direction)

    def to_dict(self) -> dict:
        return {
            "component_id": self.component_id,
            "kind": self.kind.value,
            "point": np.asarray(self.point).tolist(),
            "direction": np.asarray(self.direction).tolist(),
            "label": self.label,
        }

    @staticmethod
    def from_dict(d: dict) -> "EntityRef":
        return EntityRef(
            component_id=str(d["component_id"]),
            kind=EntityKind(d.get("kind", "origin")),
            point=as_vec3(d.get("point", [0, 0, 0])),
            direction=as_vec3(d.get("direction", [0, 0, 1])),
            label=str(d.get("label", "")),
        )


def rotation_vector(q: Quat) -> np.ndarray:
    """Log map of a quaternion to a rotation vector (axis * angle)."""
    qq = q.normalized()
    if qq.w < 0:
        qq = Quat(-qq.w, -qq.x, -qq.y, -qq.z)
    s = math.sqrt(qq.x * qq.x + qq.y * qq.y + qq.z * qq.z)
    if s < 1e-12:
        return vec3()
    angle = 2.0 * math.atan2(s, qq.w)
    return np.array([qq.x, qq.y, qq.z]) / s * angle


def twist_angle(q: Quat, axis) -> float:
    """Rotation angle about ``axis`` contained in quaternion ``q``."""
    qq = q.normalized()
    a = unit(axis)
    return 2.0 * math.atan2(float(np.dot(qq.vec, a)), qq.w)


def _perp(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Component of ``v`` perpendicular to unit ``axis``."""
    return v - float(np.dot(v, axis)) * axis


@dataclass
class AssemblyConstraint:
    """One mate between two entities (or one entity and the world)."""

    id: str
    type: ConstraintType
    a: EntityRef
    b: EntityRef | None = None
    value: float = 0.0
    name: str = ""
    enabled: bool = True
    suppressed: bool = False
    locked_value: bool = False
    target: Transform | None = None       # used by FIXED when b is None
    notes: str = ""

    # -- residuals ---------------------------------------------------------
    def residuals(self, pose_a: Transform, pose_b: Transform | None) -> np.ndarray:
        if self.suppressed or not self.enabled:
            return np.zeros(0)
        pa, da = self.a.world(pose_a)
        if self.type is ConstraintType.FIXED and (self.b is None or self.target is not None) and pose_b is None:
            target = self.target or Transform.identity()
            pos_err = pa - target.apply_point(self.a.point)
            rot_err = rotation_vector(target.rotation.inverse() * pose_a.rotation)
            return np.concatenate([pos_err, rot_err])
        if self.b is not None and pose_b is not None:
            pb, db = self.b.world(pose_b)
        elif self.b is not None and pose_b is None:
            # entity b is world-anchored: interpret its stored point as world coordinates
            pb, db = self.b.point, self.b.direction
        else:
            pb, db = pa, da

        t = self.type
        if t is ConstraintType.COINCIDENT:
            return pa - pb
        if t is ConstraintType.BALL:
            return pa - pb
        if t in (ConstraintType.CONCENTRIC, ConstraintType.REVOLUTE, ConstraintType.SCREW):
            align = np.cross(da, db)
            offset = _perp(pb - pa, da)
            return np.concatenate([offset, align])
        if t is ConstraintType.CYLINDRICAL:
            align = np.cross(da, db)
            offset = _perp(pb - pa, da)
            return np.concatenate([offset, align])
        if t is ConstraintType.SLIDER:
            align = np.cross(da, db)
            offset = _perp(pb - pa, da)
            q_rel = pose_b.rotation * pose_a.rotation.inverse() if pose_b is not None else Quat.identity()
            lock = np.array([twist_angle(q_rel, da)])
            return np.concatenate([offset, align, lock])
        if t is ConstraintType.PLANAR:
            dist = np.array([float(np.dot(pb - pa, da)) - self.value])
            align = _perp(db, da)
            return np.concatenate([dist, align])
        if t is ConstraintType.PARALLEL:
            return np.cross(da, db)
        if t is ConstraintType.PERPENDICULAR:
            return np.array([float(np.dot(da, db))])
        if t is ConstraintType.DISTANCE:
            if self.a.kind is EntityKind.PLANE and self.b is not None and self.b.kind is EntityKind.PLANE:
                return np.array([float(np.dot(pb - pa, da)) - self.value])
            d = pb - pa
            dist = float(np.linalg.norm(d))
            return np.array([dist - self.value])
        if t is ConstraintType.ANGLE:
            cosang = float(np.clip(np.dot(da, db), -1.0, 1.0))
            return np.array([math.acos(cosang) - self.value])
        if t is ConstraintType.FIXED:
            target = self.target or Transform.identity()
            pos_err = pa - target.apply_point(self.a.point)
            rot_err = rotation_vector(target.rotation.inverse() * (pose_b.rotation if pose_b is not None else pose_a.rotation))
            return np.concatenate([pos_err, rot_err])
        return np.zeros(0)

    # -- helpers -----------------------------------------------------------
    def component_ids(self) -> tuple[str, ...]:
        ids = [self.a.component_id]
        if self.b is not None:
            ids.append(self.b.component_id)
        return tuple(ids)

    def free_axis(self) -> np.ndarray | None:
        """World axis left free by a revolute/slider/cylindrical/screw mate."""
        if self.type in (ConstraintType.REVOLUTE, ConstraintType.SLIDER, ConstraintType.CYLINDRICAL,
                         ConstraintType.SCREW):
            return self.a.direction.copy()
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "a": self.a.to_dict(),
            "b": self.b.to_dict() if self.b else None,
            "value": self.value,
            "name": self.name,
            "enabled": self.enabled,
            "suppressed": self.suppressed,
            "locked_value": self.locked_value,
            "target": self.target.to_dict() if self.target else None,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "AssemblyConstraint":
        return AssemblyConstraint(
            id=str(d["id"]),
            type=ConstraintType(d.get("type", "fixed")),
            a=EntityRef.from_dict(d["a"]),
            b=EntityRef.from_dict(d["b"]) if d.get("b") else None,
            value=float(d.get("value", 0.0)),
            name=str(d.get("name", "")),
            enabled=bool(d.get("enabled", True)),
            suppressed=bool(d.get("suppressed", False)),
            locked_value=bool(d.get("locked_value", False)),
            target=Transform.from_dict(d["target"]) if d.get("target") else None,
            notes=str(d.get("notes", "")),
        )


@dataclass
class SolveResult:
    converged: bool
    residual_norm: float
    iterations: int
    dof: int
    equation_count: int
    constraint_rank: int
    per_constraint_residual: dict[str, float] = field(default_factory=dict)
    component_dof: dict[str, int] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    free_components: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.converged:
            return "conflict"
        if self.dof == 0:
            return "fully constrained"
        return f"under-constrained ({self.dof} DOF)"

    def summary(self) -> str:
        return (
            f"{self.status}; residual={self.residual_norm:.2e}, "
            f"equations={self.equation_count}, rank={self.constraint_rank}, iterations={self.iterations}"
        )


class AssemblySolver:
    """Numerically solves the assembly's mate set.

    Components participating: every non-grounded, non-suppressed component referenced by at
    least one enabled constraint (plus an optional driven component). Grounded components
    are held fixed; if none is grounded the solver says so instead of picking one silently.
    """

    def __init__(self, assembly: Assembly, tolerance: float = 1e-7, linearisation: float = 1e-6) -> None:
        self.assembly = assembly
        self.tolerance = tolerance
        self.linearisation = linearisation

    # -- variable bookkeeping ---------------------------------------------
    def _active_constraints(self) -> list[AssemblyConstraint]:
        return [c for c in self.assembly.constraints if c.enabled and not c.suppressed]

    def _free_components(self) -> list[str]:
        ids: list[str] = []
        for constraint in self._active_constraints():
            for cid in constraint.component_ids():
                comp = self.assembly.components.get(cid)
                if comp is None or comp.grounded or comp.suppressed:
                    continue
                if cid not in ids:
                    ids.append(cid)
        return ids

    # -- poses -------------------------------------------------------------
    def _poses(self, free_ids: list[str], deltas: np.ndarray, base: dict[str, Transform]) -> dict[str, Transform]:
        poses = {cid: base[cid].copy() for cid in free_ids}
        for index, cid in enumerate(free_ids):
            d = deltas[index * 6: index * 6 + 6]
            pose = base[cid]
            dq = Quat.from_axis_angle(d[3:6], np.linalg.norm(d[3:6])) if np.linalg.norm(d[3:6]) > 1e-14 else Quat.identity()
            poses[cid] = Transform(
                rotation=(dq * pose.rotation).normalized(),
                translation=pose.translation + d[0:3],
            )
        return poses

    def _pose_of(self, cid: str, poses: dict[str, Transform]) -> Transform | None:
        if cid in poses:
            return poses[cid]
        if cid not in self.assembly.components:
            return None
        # grounded or otherwise fixed components keep their world pose
        return self.assembly.world_transform(cid)

    def _residual_vector(self, constraints: list[AssemblyConstraint], poses: dict[str, Transform]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for constraint in constraints:
            pose_a = self._pose_of(constraint.a.component_id, poses)
            pose_b = self._pose_of(constraint.b.component_id, poses) if constraint.b else None
            if pose_a is None:
                continue
            chunks.append(constraint.residuals(pose_a, pose_b))
        return np.concatenate(chunks) if chunks else np.zeros(0)

    # -- solve -------------------------------------------------------------
    def solve(self, apply: bool = True, max_iter: int = 300,
              driver: tuple[str, Transform] | None = None) -> SolveResult:
        constraints = self._active_constraints()
        free_ids = self._free_components()
        diagnostics: list[Diagnostic] = []
        grounded = [c for c in self.assembly.components.values() if c.grounded and not c.suppressed]

        if driver is not None:
            driver_id, driver_pose = driver
            if driver_id not in free_ids:
                comp = self.assembly.components.get(driver_id)
                if comp is not None and not comp.grounded:
                    free_ids.append(driver_id)
            # temporarily pin the driven component to the requested pose
            saved = self.assembly.components[driver_id].transform.copy()
            self.assembly.set_world_transform(driver_id, driver_pose)
            self.assembly.components[driver_id].grounded = True
            try:
                result = self._solve_inner(constraints, free_ids, apply, max_iter)
            finally:
                self.assembly.components[driver_id].grounded = False
                self.assembly.components[driver_id].transform = saved
            return result

        if not constraints:
            diagnostics.append(
                Diagnostic("info", "no-constraints",
                           "No enabled constraints. Components are positioned by their placement transforms.")
            )
        if not grounded and constraints:
            diagnostics.append(
                Diagnostic("warning", "no-ground",
                           "No grounded component: the mate set can float. Ground a datum component "
                           "(e.g. the airframe body) for a stable solution.")
            )
        return self._solve_inner(constraints, free_ids, apply, max_iter, diagnostics)

    def _solve_inner(self, constraints: list[AssemblyConstraint], free_ids: list[str],
                     apply: bool, max_iter: int,
                     diagnostics: list[Diagnostic] | None = None) -> SolveResult:
        diagnostics = diagnostics if diagnostics is not None else []
        base = {cid: self.assembly.world_transform(cid) for cid in free_ids}
        if not free_ids or not constraints:
            per = {c.id: float(np.linalg.norm(self._residual_vector([c], {}))) for c in constraints}
            return SolveResult(
                converged=True, residual_norm=0.0, iterations=0, dof=6 * len(free_ids),
                equation_count=0, constraint_rank=0, per_constraint_residual=per,
                diagnostics=diagnostics,
            )

        x0 = np.zeros(len(free_ids) * 6)

        def fun(x: np.ndarray) -> np.ndarray:
            poses = self._poses(free_ids, x, base)
            return self._residual_vector(constraints, poses)

        solution = least_squares(
            fun, x0, method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=max_iter * len(free_ids),
            diff_step=self.linearisation,
        )
        residual = fun(solution.x)
        residual_norm = float(np.linalg.norm(residual)) if residual.size else 0.0
        converged = residual_norm <= max(self.tolerance * max(len(residual), 1), 1e-9) * 10.0

        # Jacobian and rank analysis
        J = getattr(solution, "jac", None)
        if J is None:
            J = _numeric_jacobian(fun, solution.x, self.linearisation)
        rank = int(np.linalg.matrix_rank(J, tol=1e-7)) if J.size else 0
        dof = max(6 * len(free_ids) - rank, 0)
        per_constraint: dict[str, float] = {}
        poses_now = self._poses(free_ids, solution.x, base)
        for constraint in constraints:
            pose_a = self._pose_of(constraint.a.component_id, poses_now)
            pose_b = self._pose_of(constraint.b.component_id, poses_now) if constraint.b else None
            r = constraint.residuals(pose_a, pose_b)
            per_constraint[constraint.id] = float(np.linalg.norm(r)) if r.size else 0.0

        # per-component DOF: how much null-space energy each component owns
        component_dof: dict[str, int] = {}
        if J.size and dof > 0:
            _, _, vt = np.linalg.svd(J)
            nullspace = vt[rank:, :] if vt.shape[0] > rank else np.zeros((0, J.shape[1]))
            for index, cid in enumerate(free_ids):
                block = nullspace[:, index * 6:(index + 1) * 6]
                energy = float(np.linalg.norm(block))
                component_dof[cid] = int(round(np.linalg.matrix_rank(block, tol=1e-6))) if block.size else 0
                if energy > 1e-8 and component_dof[cid] == 0:
                    component_dof[cid] = 1
        else:
            for cid in free_ids:
                component_dof[cid] = 0

        if not converged:
            worst = sorted(per_constraint.items(), key=lambda kv: -kv[1])[:3]
            names = []
            for cid, value in worst:
                constraint = next((c for c in constraints if c.id == cid), None)
                label = constraint.name or (constraint.type.label() if constraint else cid)
                names.append(f"{label} ({value:.3e})")
            diagnostics.append(
                Diagnostic("error", "constraint-conflict",
                           "Constraints cannot all be satisfied; largest errors: " + ", ".join(names)
                           + ". Check for conflicting mates or an invalid joint axis.")
            )
        else:
            equation_count = int(residual.size)
            if dof == 0:
                diagnostics.append(Diagnostic("info", "fully-constrained", "Assembly is fully constrained."))
            else:
                diagnostics.append(
                    Diagnostic("warning", "under-constrained",
                               f"Assembly has {dof} unconstrained degrees of freedom. "
                               "Add mates or ground the moving parts.")
                )
            if equation_count > rank and equation_count > 0:
                diagnostics.append(
                    Diagnostic("info", "redundant",
                               f"{equation_count - rank} dependent constraint equation(s) are redundant "
                               "but consistent.")
                )
            if not any(c.grounded for c in self.assembly.components.values()):
                diagnostics.append(
                    Diagnostic("warning", "no-ground",
                               "No grounded component; the mate set is free to float in space.")
                )

        # unconnected parts
        linked = {cid for c in constraints for cid in c.component_ids()}
        for comp in self.assembly.components.values():
            if comp.suppressed or comp.grounded or comp.is_subassembly and not comp.part_id:
                continue
            if comp.id not in linked:
                diagnostics.append(
                    Diagnostic("warning", "unconnected",
                               f"'{comp.name}' has no constraints; it will not move with the assembly.")
                )

        if apply:
            poses = self._poses(free_ids, solution.x, base)
            for cid, pose in poses.items():
                self.assembly.set_world_transform(cid, pose)

        return SolveResult(
            converged=converged,
            residual_norm=residual_norm,
            iterations=int(getattr(solution, "nfev", 0)),
            dof=dof,
            equation_count=int(residual.size),
            constraint_rank=rank,
            per_constraint_residual=per_constraint,
            component_dof=component_dof,
            diagnostics=diagnostics,
            free_components=free_ids,
        )

    # -- interactive driving ----------------------------------------------
    def drag(self, component_id: str, world_transform: Transform) -> SolveResult:
        """Move one component interactively; connected parts follow their mates."""
        return self.solve(apply=True, driver=(component_id, world_transform))


def _numeric_jacobian(fun, x: np.ndarray, step: float) -> np.ndarray:
    f0 = fun(x)
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        h = step * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += h
        J[:, i] = (fun(xp) - f0) / h
    return J


# --------------------------------------------------------------------------- #
# Convenience constructors used by the UI and the sample project
# --------------------------------------------------------------------------- #
def coincident(component_a: str, point_a, component_b: str, point_b, name: str = "") -> AssemblyConstraint:
    return AssemblyConstraint(
        id=f"con_{np.random.default_rng().integers(1e6)}",
        type=ConstraintType.COINCIDENT,
        a=EntityRef(component_a, EntityKind.POINT, point_a, label="point"),
        b=EntityRef(component_b, EntityKind.POINT, point_b, label="point"),
        name=name or "Coincident",
    )


def concentric(component_a: str, point_a, axis_a, component_b: str, point_b, axis_b,
               kind: ConstraintType = ConstraintType.CONCENTRIC, name: str = "") -> AssemblyConstraint:
    return AssemblyConstraint(
        id=f"con_{np.random.default_rng().integers(1e6)}",
        type=kind,
        a=EntityRef(component_a, EntityKind.AXIS, point_a, axis_a, "axis"),
        b=EntityRef(component_b, EntityKind.AXIS, point_b, axis_b, "axis"),
        name=name or kind.label(),
    )


def distance(component_a: str, point_a, component_b: str, point_b, value: float,
             name: str = "") -> AssemblyConstraint:
    return AssemblyConstraint(
        id=f"con_{np.random.default_rng().integers(1e6)}",
        type=ConstraintType.DISTANCE,
        a=EntityRef(component_a, EntityKind.POINT, point_a),
        b=EntityRef(component_b, EntityKind.POINT, point_b),
        value=value,
        name=name or "Distance",
    )


def angle(component_a: str, axis_a, component_b: str, axis_b, value_rad: float,
          name: str = "") -> AssemblyConstraint:
    return AssemblyConstraint(
        id=f"con_{np.random.default_rng().integers(1e6)}",
        type=ConstraintType.ANGLE,
        a=EntityRef(component_a, EntityKind.AXIS, (0, 0, 0), axis_a),
        b=EntityRef(component_b, EntityKind.AXIS, (0, 0, 0), axis_b),
        value=value_rad,
        name=name or "Angle",
    )


def grounded_entity(component_id: str) -> EntityRef:
    """World-anchored reference used by fixed-to-world mates."""
    return EntityRef("__world__", EntityKind.ORIGIN)


def fixed(component_id: str, target: Transform | None = None, name: str = "") -> AssemblyConstraint:
    return AssemblyConstraint(
        id=f"con_{np.random.default_rng().integers(1e6)}",
        type=ConstraintType.FIXED,
        a=EntityRef(component_id, EntityKind.ORIGIN),
        b=None,
        target=target,
        name=name or "Fixed",
    )
