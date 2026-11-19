"""3D flight view (pyqtgraph OpenGL).

The rocket is drawn from the fixed geometry in :mod:`rocket_sim.config` — one body tube, an
ogive nose and four fixed fins — and moved with the attitude and position from the physics
simulation. Nothing about the drawing is animated independently of the simulation.
"""

from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - import guard for headless environments
    from PySide6 import QtGui
    import pyqtgraph.opengl as gl
    from pyqtgraph import Vector
    HAVE_QT = True
except Exception:  # pragma: no cover
    gl = None
    Vector = None
    QtGui = None
    HAVE_QT = False

from .config import MOTOR, ROCKET, MotorConfig, RocketConfig
from .physics import quat_matrix

BODY_COLOR = (0.93, 0.95, 0.98, 1.0)
NOSE_COLOR = (0.86, 0.28, 0.20, 1.0)
FIN_COLOR = (0.31, 0.52, 0.76, 1.0)
NOZZLE_COLOR = (0.22, 0.24, 0.28, 1.0)
PAD_COLOR = (0.30, 0.36, 0.44, 1.0)
TRAIL_COLOR = (1.0, 0.64, 0.28, 0.6)


def _ring(x: float, radius: float, segments: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    return np.stack([
        np.full(segments, x),
        radius * np.cos(angles),
        radius * np.sin(angles),
    ], axis=1)


def _tube(x_start: float, x_end: float, radius: float, segments: int = 28) -> tuple[np.ndarray, np.ndarray]:
    start = _ring(x_start, radius, segments)
    end = _ring(x_end, radius, segments)
    vertices = np.vstack([start, end])
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + j])
        faces.append([i, segments + j, segments + i])
    # closed ends so the body reads as a solid from any angle
    center_start = len(vertices)
    vertices = np.vstack([vertices, [[x_start, 0.0, 0.0]]])
    center_end = len(vertices)
    vertices = np.vstack([vertices, [[x_end, 0.0, 0.0]]])
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([center_start, j, i])
        faces.append([center_end, segments + i, segments + j])
    return vertices, np.array(faces, dtype=np.uint32)


def _cone(x_start: float, x_tip: float, radius: float, segments: int = 28,
          slices: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Ogive nose: radius falls as sqrt(1 - u^2) from shoulder to tip."""
    rings = []
    for index in range(slices):
        u = index / slices
        x = x_start + (x_tip - x_start) * u
        rings.append(_ring(x, radius * math.sqrt(max(0.0, 1.0 - u * u)), segments))
    vertices = np.vstack(rings)
    faces = []
    for layer in range(slices - 1):
        base = layer * segments
        nxt = base + segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([base + i, base + j, nxt + j])
            faces.append([base + i, nxt + j, nxt + i])
    tip_index = len(vertices)
    vertices = np.vstack([vertices, [[x_tip, 0.0, 0.0]]])
    last = (slices - 1) * segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([last + i, last + j, tip_index])
    # close the shoulder
    center = len(vertices)
    vertices = np.vstack([vertices, [[x_start, 0.0, 0.0]]])
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([center, j, i])
    return vertices, np.array(faces, dtype=np.uint32)


def _fin(x_trailing_root: float, root_chord: float, tip_chord: float, sweep: float,
         inner_radius: float, span: float, thickness: float,
         angle: float) -> tuple[np.ndarray, np.ndarray]:
    """One flat trapezoidal fin, extruded across its thickness, rotated about the axis."""
    x_le_root = x_trailing_root + root_chord
    x_le_tip = x_le_root - sweep
    x_te_tip = x_le_tip - tip_chord
    inner = inner_radius
    outer = inner_radius + span

    profile = [
        (x_trailing_root, inner),
        (x_le_root, inner),
        (x_le_tip, outer),
        (x_te_tip, outer),
    ]
    half = thickness / 2.0
    vertices = []
    for offset in (-half, half):
        for x, radius in profile:
            vertices.append([x, radius, offset])
    vertices = np.array(vertices, dtype=float)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],           # -thickness face
        [4, 6, 5], [4, 7, 6],           # +thickness face
        [0, 4, 5], [0, 5, 1],           # root edge
        [3, 2, 6], [3, 6, 7],           # tip edge
        [1, 5, 6], [1, 6, 2],           # leading edge
        [0, 3, 7], [0, 7, 4],           # trailing edge
    ], dtype=np.uint32)

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_a, -sin_a],
        [0.0, sin_a, cos_a],
    ])
    return vertices @ rotation.T, faces


def build_rocket_mesh(rocket: RocketConfig = ROCKET,
                      motor: MotorConfig = MOTOR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merged rocket mesh in body coordinates (origin at the centre of gravity)."""
    cg = rocket.cg_from_nose_tip
    x_tail = cg - rocket.body_length
    x_shoulder = x_tail + rocket.body_length - rocket.nose_length

    body_v, body_f = _tube(x_tail, x_shoulder, rocket.body_radius)
    nose_v, nose_f = _cone(x_shoulder, cg, rocket.body_radius)

    nozzle_x0 = x_tail + motor.aft_end_from_tail
    nozzle_v, nozzle_f = _tube(nozzle_x0 - motor.nozzle_length, nozzle_x0,
                               motor.nozzle_diameter / 2.0, segments=20)

    parts = [
        (body_v, body_f, BODY_COLOR),
        (nose_v, nose_f, NOSE_COLOR),
        (nozzle_v, nozzle_f, NOZZLE_COLOR),
    ]

    x_trailing = x_tail + rocket.fin_trailing_edge_from_tail
    for index in range(rocket.fin_count):
        angle = 2.0 * math.pi * index / rocket.fin_count + math.pi / 4.0
        fin_v, fin_f = _fin(x_trailing, rocket.fin_root_chord, rocket.fin_tip_chord,
                            rocket.fin_sweep, rocket.body_radius * 0.98,
                            rocket.fin_span, rocket.fin_thickness, angle)
        parts.append((fin_v, fin_f, FIN_COLOR))

    vertices = []
    faces = []
    colors = []
    offset = 0
    for part_v, part_f, color in parts:
        vertices.append(part_v)
        faces.append(part_f + offset)
        colors.append(np.tile(np.array(color), (len(part_v), 1)))
        offset += len(part_v)
    return (np.vstack(vertices), np.vstack(faces), np.vstack(colors))


def build_ground_plane(size: float = 600.0, z: float = -0.004) -> tuple[np.ndarray, np.ndarray]:
    """Solid ground so the rocket reads as resting on a surface, not floating."""
    half = size / 2.0
    vertices = np.array([
        [-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return vertices, faces


def build_pad_mesh(radius: float = 0.35, height: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """A short cylinder standing on the ground plane (axis along world +Z)."""
    segments = 32
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    bottom = np.stack([radius * np.cos(angles), radius * np.sin(angles),
                       np.zeros(segments)], axis=1)
    top = np.stack([radius * np.cos(angles), radius * np.sin(angles),
                    np.full(segments, height)], axis=1)
    vertices = np.vstack([bottom, top, [[0.0, 0.0, height]], [[0.0, 0.0, 0.0]]])
    center_top, center_bottom = 2 * segments, 2 * segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + j])
        faces.append([i, segments + j, segments + i])
        faces.append([center_top, segments + i, segments + j])
        faces.append([center_bottom, j, i])
    return vertices, np.array(faces, dtype=np.uint32)


if HAVE_QT:

    class RocketViewport(gl.GLViewWidget):
        """Ground, launch pad and the rocket, with orbit/zoom/follow camera controls."""

        def __init__(self, rocket: RocketConfig = ROCKET, motor: MotorConfig = MOTOR,
                     parent=None) -> None:
            super().__init__(parent)
            self.setBackgroundColor(QtGui.QColor(14, 17, 22))
            self.opts["fov"] = 45
            self.follow = False
            self._base_vertices, self._faces, self._colors = build_rocket_mesh(rocket, motor)

            self.rocket_item = gl.GLMeshItem(
                vertexes=self._base_vertices,
                faces=self._faces,
                vertexColors=self._colors,
                smooth=False,
                glOptions="opaque",
            )
            self.addItem(self.rocket_item)

            plane_vertices, plane_faces = build_ground_plane()
            self.ground_item = gl.GLMeshItem(
                vertexes=plane_vertices,
                faces=plane_faces,
                color=(0.115, 0.135, 0.160, 1.0),
                smooth=False,
            )
            self.addItem(self.ground_item)

            # A wide ground grid keeps the launch site visible far downrange.
            ground = gl.GLGridItem()
            ground.setSize(600.0, 600.0, 1.0)
            ground.setSpacing(20.0, 20.0, 20.0)
            ground.setColor((104, 118, 136, 120))
            self.addItem(ground)

            near = gl.GLGridItem()
            near.setSize(24.0, 24.0, 1.0)
            near.setSpacing(1.0, 1.0, 1.0)
            near.setColor((120, 136, 156, 150))
            self.addItem(near)

            pad_vertices, pad_faces = build_pad_mesh()
            self.pad_item = gl.GLMeshItem(
                vertexes=pad_vertices,
                faces=pad_faces,
                color=PAD_COLOR,
                smooth=False,
            )
            self.addItem(self.pad_item)

            self.trail_item = gl.GLLinePlotItem(
                pos=np.zeros((0, 3)), color=TRAIL_COLOR, width=1.5,
                antialias=True, mode="line_strip",
            )
            self.addItem(self.trail_item)

            self.reset_view()

        # --------------------------------------------------------------- drawing
        def update_rocket(self, position: np.ndarray, attitude: np.ndarray) -> None:
            rotation = quat_matrix(attitude)
            vertices = self._base_vertices @ rotation.T + np.asarray(position, dtype=float)
            self.rocket_item.setMeshData(
                vertexes=vertices,
                faces=self._faces,
                vertexColors=self._colors,
                smooth=False,
            )
            if self.follow:
                self.opts["center"] = Vector(float(position[0]), float(position[1]),
                                             float(position[2]) + 0.2)
            self.update()

        def update_trail(self, points: list[np.ndarray]) -> None:
            if not points:
                self.trail_item.setData(pos=np.zeros((0, 3)))
                return
            self.trail_item.setData(pos=np.vstack(points))

        def clear_trail(self) -> None:
            self.trail_item.setData(pos=np.zeros((0, 3)))

        # ---------------------------------------------------------------- camera
        def reset_view(self) -> None:
            self.follow = False
            self.setCameraPosition(pos=Vector(0.0, 0.0, 0.9), distance=5.5,
                                   elevation=10.0, azimuth=-55.0)

        def set_follow(self, enabled: bool) -> None:
            self.follow = enabled
            if not enabled:
                self.setCameraPosition(pos=Vector(0.0, 0.0, 1.0))

else:  # pragma: no cover - only imported when Qt is missing

    class RocketViewport:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PySide6 and pyqtgraph are required for the 3D view")
