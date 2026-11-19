"""Three-dimensional mathematics used by every subsystem.

Conventions (documented once here, relied on everywhere):

* World frame is right-handed, Z up, X forward (range), Y right.
* Internal units are SI: metre, kilogram, second, newton, radian.
* Quaternions are scalar-first ``(w, x, y, z)`` and rotate *body -> world*.
* Euler angles are aerospace intrinsic yaw-pitch-roll (Z-Y-X):
  ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
* Inertia tensors are expressed about a point, in a stated frame, in kg*m^2.

The module deliberately contains no physics: only algebra, so it can be unit
tested against closed-form results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-12
GRAVITY = 9.80665  # m/s^2, standard gravity (exact by definition)


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #
def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Return a float64 length-3 array."""
    return np.array([float(x), float(y), float(z)], dtype=float)


def as_vec3(value: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(3)
    return arr


def norm(v: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def unit(v: Sequence[float]) -> np.ndarray:
    arr = as_vec3(v)
    n = float(np.linalg.norm(arr))
    if n < EPS:
        raise ValueError("cannot normalise a zero-length vector")
    return arr / n


def skew(v: Sequence[float]) -> np.ndarray:
    """Cross-product matrix ``skew(v) @ u == cross(v, u)``."""
    x, y, z = as_vec3(v)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def rotation_matrix(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix about ``axis`` by ``angle_rad``."""
    a = unit(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    K = skew(a)
    return np.eye(3) * c + s * K + (1.0 - c) * np.outer(a, a)


def rot_x(a: float) -> np.ndarray:
    return rotation_matrix((1.0, 0.0, 0.0), a)


def rot_y(a: float) -> np.ndarray:
    return rotation_matrix((0.0, 1.0, 0.0), a)


def rot_z(a: float) -> np.ndarray:
    return rotation_matrix((0.0, 0.0, 1.0), a)


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


# --------------------------------------------------------------------------- #
# Quaternions
# --------------------------------------------------------------------------- #
@dataclass
class Quat:
    """Unit quaternion, scalar-first, rotating body vectors into world."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # -- construction ------------------------------------------------------
    @staticmethod
    def identity() -> "Quat":
        return Quat(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_axis_angle(axis: Sequence[float], angle_rad: float) -> "Quat":
        a = unit(axis)
        half = 0.5 * angle_rad
        s = math.sin(half)
        return Quat(math.cos(half), a[0] * s, a[1] * s, a[2] * s)

    @staticmethod
    def from_euler(yaw: float, pitch: float, roll: float) -> "Quat":
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        return Quat(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )

    @staticmethod
    def from_matrix(R: np.ndarray) -> "Quat":
        """Shepperd's method, numerically stable for all rotations."""
        m = np.asarray(R, dtype=float)
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        q = Quat(w, x, y, z)
        q.normalize_inplace()
        return q

    @staticmethod
    def from_two_vectors(a: Sequence[float], b: Sequence[float]) -> "Quat":
        """Shortest rotation taking direction ``a`` onto direction ``b``."""
        u, v = unit(a), unit(b)
        d = float(np.dot(u, v))
        if d < -1.0 + 1e-9:
            # 180 degree rotation about any axis perpendicular to u
            helper = vec3(1.0, 0.0, 0.0) if abs(u[0]) < 0.9 else vec3(0.0, 1.0, 0.0)
            return Quat.from_axis_angle(np.cross(u, helper), math.pi)
        c = np.cross(u, v)
        return Quat(1.0 + d, c[0], c[1], c[2]).normalized()

    # -- maths -------------------------------------------------------------
    @property
    def array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=float)

    @property
    def vec(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def norm(self) -> float:
        return math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Quat":
        n = self.norm()
        if n < EPS:
            return Quat.identity()
        return Quat(self.w / n, self.x / n, self.y / n, self.z / n)

    def normalize_inplace(self) -> None:
        n = self.norm()
        if n < EPS:
            self.w, self.x, self.y, self.z = 1.0, 0.0, 0.0, 0.0
            return
        self.w, self.x, self.y, self.z = self.w / n, self.x / n, self.y / n, self.z / n

    def conjugate(self) -> "Quat":
        return Quat(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> "Quat":
        n2 = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if n2 < EPS:
            return Quat.identity()
        c = self.conjugate()
        return Quat(c.w / n2, c.x / n2, c.y / n2, c.z / n2)

    def __mul__(self, other: "Quat") -> "Quat":
        if not isinstance(other, Quat):
            return NotImplemented
        a, b = self, other
        return Quat(
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        )

    def rotate(self, v: Sequence[float]) -> np.ndarray:
        """Rotate a body-frame vector into the world frame."""
        u = as_vec3(v)
        qv = Quat(0.0, u[0], u[1], u[2])
        r = (self.normalized() * qv) * self.normalized().conjugate()
        return np.array([r.x, r.y, r.z], dtype=float)

    def rotate_inverse(self, v: Sequence[float]) -> np.ndarray:
        """Rotate a world-frame vector into the body frame."""
        return self.inverse().rotate(v)

    def matrix(self) -> np.ndarray:
        w, x, y, z = self.normalized().array
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def to_euler(self) -> tuple[float, float, float]:
        """Return ``(yaw, pitch, roll)`` in radians."""
        w, x, y, z = self.normalized().array
        sinp = clamp(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = math.asin(sinp)
        if abs(sinp) > 1.0 - 1e-9:
            # gimbal lock: put all rotation in yaw
            yaw = 2.0 * math.atan2(x, w) * (1.0 if sinp > 0 else -1.0)
            roll = 0.0
            return yaw, pitch, roll
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        return yaw, pitch, roll

    def derivative(self, omega_body: Sequence[float]) -> "Quat":
        """Time derivative for body-frame angular velocity ``omega``."""
        qw = self * Quat(0.0, *as_vec3(omega_body))
        return Quat(0.5 * qw.w, 0.5 * qw.x, 0.5 * qw.y, 0.5 * qw.z)

    def slerp(self, other: "Quat", t: float) -> "Quat":
        a, b = self.normalized(), other.normalized()
        dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z
        if dot < 0.0:
            b = Quat(-b.w, -b.x, -b.y, -b.z)
            dot = -dot
        dot = clamp(dot, -1.0, 1.0)
        if dot > 0.9995:
            out = Quat(
                a.w + t * (b.w - a.w),
                a.x + t * (b.x - a.x),
                a.y + t * (b.y - a.y),
                a.z + t * (b.z - a.z),
            )
            return out.normalized()
        theta = math.acos(dot)
        sin_t = math.sin(theta)
        s0 = math.sin((1.0 - t) * theta) / sin_t
        s1 = math.sin(t * theta) / sin_t
        return Quat(
            s0 * a.w + s1 * b.w,
            s0 * a.x + s1 * b.x,
            s0 * a.y + s1 * b.y,
            s0 * a.z + s1 * b.z,
        ).normalized()

    def angle_to(self, other: "Quat") -> float:
        d = abs(self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z)
        return 2.0 * math.acos(clamp(d, -1.0, 1.0))

    def is_finite(self) -> bool:
        return all(math.isfinite(c) for c in (self.w, self.x, self.y, self.z))

    def to_list(self) -> list[float]:
        return [self.w, self.x, self.y, self.z]

    @staticmethod
    def from_list(values: Sequence[float]) -> "Quat":
        w, x, y, z = [float(c) for c in values]
        return Quat(w, x, y, z).normalized()


# --------------------------------------------------------------------------- #
# Rigid transform
# --------------------------------------------------------------------------- #
@dataclass
class Transform:
    """Rigid body placement: rotate then translate (``p_world = R p_local + t``)."""

    rotation: Quat = field(default_factory=Quat.identity)
    translation: np.ndarray = field(default_factory=lambda: vec3())

    @staticmethod
    def identity() -> "Transform":
        return Transform(Quat.identity(), vec3())

    @staticmethod
    def from_pos_euler(pos: Sequence[float], yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> "Transform":
        return Transform(Quat.from_euler(yaw, pitch, roll), as_vec3(pos))

    def copy(self) -> "Transform":
        return Transform(Quat(*self.rotation.array), self.translation.copy())

    def __matmul__(self, other: "Transform") -> "Transform":
        return Transform(self.rotation * other.rotation, self.rotation.rotate(other.translation) + self.translation)

    def inverse(self) -> "Transform":
        inv = self.rotation.inverse()
        return Transform(inv, -inv.rotate(self.translation))

    def apply_point(self, p: Sequence[float]) -> np.ndarray:
        return self.rotation.rotate(p) + self.translation

    def apply_vector(self, v: Sequence[float]) -> np.ndarray:
        return self.rotation.rotate(v)

    def matrix4(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation.matrix()
        m[:3, 3] = self.translation
        return m

    @staticmethod
    def from_matrix4(m: np.ndarray) -> "Transform":
        m = np.asarray(m, dtype=float)
        return Transform(Quat.from_matrix(m[:3, :3]), m[:3, 3].copy())

    def to_dict(self) -> dict:
        return {"quat": self.rotation.to_list(), "pos": self.translation.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "Transform":
        return Transform(Quat.from_list(d.get("quat", [1, 0, 0, 0])), as_vec3(d.get("pos", [0, 0, 0])))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        yaw, pitch, roll = self.rotation.to_euler()
        return (
            f"Transform(pos={np.round(self.translation, 5).tolist()}, "
            f"ypr_deg={[round(math.degrees(a), 3) for a in (yaw, pitch, roll)]})"
        )


# --------------------------------------------------------------------------- #
# Mass and inertia algebra
# --------------------------------------------------------------------------- #
@dataclass
class MassProperties:
    """Mass properties of one body or a whole assembly.

    ``inertia`` is about ``com`` and expressed in the frame named by ``frame``.
    ``provenance`` records how the numbers were obtained, which the UI and the
    reports surface so estimated values are never mistaken for measured ones.
    """

    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: vec3())
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    volume: float = 0.0
    surface_area: float = 0.0
    provenance: str = "unknown"
    frame: str = "world"
    unresolved: tuple[str, ...] = ()

    def inertia_at(self, point: Sequence[float]) -> np.ndarray:
        return parallel_axis(self.inertia, self.mass, self.com - as_vec3(point))

    def principal(self) -> tuple[np.ndarray, np.ndarray]:
        return principal_inertia(self.inertia)

    def is_valid(self) -> bool:
        return self.mass >= 0.0 and np.all(np.isfinite(self.inertia)) and np.all(np.isfinite(self.com))

    def to_dict(self) -> dict:
        return {
            "mass": self.mass,
            "com": self.com.tolist(),
            "inertia": np.asarray(self.inertia).tolist(),
            "volume": self.volume,
            "surface_area": self.surface_area,
            "provenance": self.provenance,
            "frame": self.frame,
            "unresolved": list(self.unresolved),
        }

    @staticmethod
    def from_dict(d: dict) -> "MassProperties":
        return MassProperties(
            mass=float(d.get("mass", 0.0)),
            com=as_vec3(d.get("com", [0, 0, 0])),
            inertia=np.asarray(d.get("inertia", np.zeros((3, 3))), dtype=float).reshape(3, 3),
            volume=float(d.get("volume", 0.0)),
            surface_area=float(d.get("surface_area", 0.0)),
            provenance=str(d.get("provenance", "unknown")),
            frame=str(d.get("frame", "world")),
            unresolved=tuple(d.get("unresolved", ())),
        )


def parallel_axis(inertia_com: np.ndarray, mass: float, offset: Sequence[float]) -> np.ndarray:
    """Steiner (parallel-axis) theorem: inertia about a point offset by ``offset``."""
    d = as_vec3(offset)
    return np.asarray(inertia_com, dtype=float) + mass * ((d @ d) * np.eye(3) - np.outer(d, d))


def rotate_inertia(inertia: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Re-express an inertia tensor under rotation ``R`` (``R I R^T``)."""
    return np.asarray(R, dtype=float) @ np.asarray(inertia, dtype=float) @ np.asarray(R, dtype=float).T


def principal_inertia(inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(eigenvalues, axes)`` sorted descending by eigenvalue magnitude."""
    vals, vecs = np.linalg.eigh(np.asarray(inertia, dtype=float))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def combine_mass_properties(parts: Iterable[MassProperties]) -> MassProperties:
    """Combine component properties into one set of assembly properties.

    Uses the parallel-axis theorem; each part's COM and inertia must already be
    expressed in the common (world) frame, which callers guarantee by composing
    the component placement transform first.
    """
    parts = list(parts)
    total_mass = float(sum(p.mass for p in parts))
    if not parts or total_mass <= 0.0:
        return MassProperties(mass=0.0, unresolved=("no mass in assembly",))
    com = np.zeros(3)
    for p in parts:
        com = com + p.mass * as_vec3(p.com)
    com = com / total_mass
    inertia = np.zeros((3, 3))
    for p in parts:
        inertia = inertia + parallel_axis(p.inertia, p.mass, as_vec3(p.com) - com)
    unresolved: list[str] = []
    provenance = "cad_derived"
    for p in parts:
        unresolved.extend(p.unresolved)
        if p.provenance != "cad_derived":
            provenance = "mixed"
    volume = float(sum(p.volume for p in parts))
    area = float(sum(p.surface_area for p in parts))
    return MassProperties(
        mass=total_mass,
        com=com,
        inertia=inertia,
        volume=volume,
        surface_area=area,
        provenance=provenance,
        unresolved=tuple(unresolved),
    )


def transform_mass_properties(mp: MassProperties, transform: Transform) -> MassProperties:
    """Move component properties from component frame into world frame."""
    R = transform.rotation.matrix()
    return MassProperties(
        mass=mp.mass,
        com=transform.apply_point(mp.com),
        inertia=rotate_inertia(mp.inertia, R),
        volume=mp.volume,
        surface_area=mp.surface_area,
        provenance=mp.provenance,
        frame="world",
        unresolved=mp.unresolved,
    )


# --------------------------------------------------------------------------- #
# Closed-form inertia solids (used by primitive components and by tests)
# --------------------------------------------------------------------------- #
def inertia_box(mass: float, sx: float, sy: float, sz: float) -> np.ndarray:
    return np.diag(
        [
            mass * (sy * sy + sz * sz) / 12.0,
            mass * (sx * sx + sz * sz) / 12.0,
            mass * (sx * sx + sy * sy) / 12.0,
        ]
    )


def inertia_cylinder(mass: float, radius: float, height: float, axis: int = 2) -> np.ndarray:
    """Solid cylinder; ``axis`` is 0, 1 or 2 for the symmetry axis."""
    axial = 0.5 * mass * radius * radius
    transverse = mass * (3.0 * radius * radius + height * height) / 12.0
    vals = [transverse, transverse, transverse]
    vals[axis] = axial
    return np.diag(vals)


def inertia_tube(mass: float, r_inner: float, r_outer: float, height: float, axis: int = 2) -> np.ndarray:
    axial = 0.5 * mass * (r_inner**2 + r_outer**2)
    transverse = mass * (3.0 * (r_inner**2 + r_outer**2) + height**2) / 12.0
    vals = [transverse, transverse, transverse]
    vals[axis] = axial
    return np.diag(vals)


def inertia_sphere(mass: float, radius: float) -> np.ndarray:
    i = 0.4 * mass * radius * radius
    return np.diag([i, i, i])


def inertia_point(mass: float, position: Sequence[float]) -> np.ndarray:
    """Point mass inertia about its own location (zero tensor)."""
    return np.zeros((3, 3))


def build_ray_plane_intersection(origin: Sequence[float], direction: Sequence[float], plane_point: Sequence[float], plane_normal: Sequence[float]) -> np.ndarray | None:
    o, d = as_vec3(origin), unit(direction)
    n = unit(plane_normal)
    denom = float(d @ n)
    if abs(denom) < 1e-9:
        return None
    t = float((as_vec3(plane_point) - o) @ n) / denom
    if t < 0:
        return None
    return o + d * t


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    u, v = unit(a), unit(b)
    return math.acos(clamp(float(u @ v), -1.0, 1.0))
