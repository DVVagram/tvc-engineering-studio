"""3D viewport (pyqtgraph OpenGL).

Supports orbit/pan/zoom, orthographic and perspective projection, solid/wireframe/
transparent display, exploded view, section cut, ray picking against the displayed
triangles and playback of simulation poses. All geometry shown is tessellated from the CAD
parts; the viewport never invents geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.math3d import as_vec3, vec3
from .scene import SceneBuilder, SceneItem, parse_color, section_filter

try:  # pragma: no cover - import guard for headless test environments
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph.opengl as gl
    from pyqtgraph import Vector
    HAVE_QT = True
except Exception:  # pragma: no cover
    QtCore = QtGui = QtWidgets = None
    gl = None
    Vector = None
    HAVE_QT = False


@dataclass
class DisplayState:
    mode: str = "solid"           # solid | wireframe | transparent
    exploded: float = 0.0
    section_enabled: bool = False
    section_axis: int = 1
    section_offset: float = 0.0
    show_grid: bool = True
    show_edges: bool = True


if HAVE_QT:

    class Viewport(gl.GLViewWidget):
        """Interactive CAD viewport."""

        componentPicked = QtCore.Signal(str)
        cursorMoved = QtCore.Signal(float, float, float)

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.setBackgroundColor(QtGui.QColor(24, 26, 30))
            self.opts["fov"] = 40
            self.setCameraPosition(distance=0.9, elevation=18, azimuth=-55)
            self.state = DisplayState()
            self.scene_items: list[SceneItem] = []
            self._items: dict[str, object] = {}
            self._edges: list[object] = []
            self._grid = None
            self._axes = None
            self._build_grid()
            self._selection: str = ""
            self._project = None
            self._highlight_overlay = None
            self._last_mouse = None

        # ------------------------------------------------------------ scene
        def _build_grid(self) -> None:
            self._grid = gl.GLGridItem()
            self._grid.setSize(1.0, 1.0, 1.0)
            self._grid.setSpacing(0.05, 0.05, 0.05)
            self._grid.setColor((90, 96, 106, 90))
            self.addItem(self._grid)
            axis = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0.12, 0, 0], [0, 0, 0], [0, 0.12, 0], [0, 0, 0], [0, 0, 0.12]]),
                color=np.array([[0.9, 0.35, 0.35, 1], [0.9, 0.35, 0.35, 1],
                                [0.35, 0.9, 0.45, 1], [0.35, 0.9, 0.45, 1],
                                [0.4, 0.6, 0.95, 1], [0.4, 0.6, 0.95, 1]]),
                width=2, mode="lines", antialias=True)
            self.addItem(axis)
            self._axes = axis

        def set_project(self, project) -> None:
            """Load/reload geometry from a Project."""
            self._project = project
            self.rebuild()

        def rebuild(self) -> None:
            if self._project is None:
                return
            builder = SceneBuilder()
            self.scene_items = builder.build(self._project.assembly, self._project.parts,
                                       self._project.materials, self._project.fasteners)
            self.diagnostics = builder.diagnostics
            self._render()
            self.fit()

        def _clear_meshes(self) -> None:
            for item in list(self._items.values()):
                self.removeItem(item)
            self._items.clear()
            for edge in self._edges:
                self.removeItem(edge)
            self._edges.clear()

        def _display_vertices(self, item: SceneItem) -> np.ndarray:
            if self.state.exploded > 0 and item.exploded_vertices is not None:
                return item.exploded_vertices
            return item.vertices

        def _render(self) -> None:
            self._clear_meshes()
            keep = None
            if self.state.section_enabled:
                shifted = []
                for item in self.scene_items:
                    clone = SceneItem(**{**item.__dict__})
                    clone.vertices = self._display_vertices(item)
                    shifted.append(clone)
                keep = section_filter(shifted, self.state.section_axis, self.state.section_offset)
            for item in self.scene_items:
                verts = self._display_vertices(item)
                if verts.size == 0 or item.triangles.size == 0:
                    continue
                faces = item.triangles
                if keep is not None:
                    idx = keep.get(item.component_id, [])
                    if not idx:
                        continue
                    faces = faces[np.array(idx, dtype=int)]
                mode = self.state.mode
                alpha = 0.32 if mode == "transparent" else item.color[3]
                color = (item.color[0], item.color[1], item.color[2], alpha)
                highlight = item.component_id == self._selection
                if highlight:
                    color = (min(color[0] + 0.25, 1.0), min(color[1] + 0.2, 1.0),
                             min(color[2] + 0.05, 1.0), color[3])
                mesh = gl.MeshData(vertexes=verts, faces=faces)
                draw_faces = mode != "wireframe"
                draw_edges = self.state.show_edges or mode == "wireframe"
                item_gl = gl.GLMeshItem(
                    meshdata=mesh, smooth=False,
                    color=color if draw_faces else (0, 0, 0, 0),
                    drawFaces=draw_faces, drawEdges=draw_edges,
                    edgeColor=(0.10, 0.12, 0.14, 0.55 if mode != "wireframe" else 1.0),
                    shader="shaded" if draw_faces else "plain",
                )
                item_gl.setGLOptions("translucent" if mode == "transparent" else "opaque")
                self.addItem(item_gl)
                self._items[item.component_id] = item_gl
            if self._grid is not None:
                self._grid.setVisible(self.state.show_grid)

        # ------------------------------------------------------------- view
        def fit(self) -> None:
            if not self.scene_items:
                return
            lows, highs = [], []
            for item in self.scene_items:
                lo, hi = item.bounds()
                if np.allclose(lo, hi):
                    continue
                lows.append(lo)
                highs.append(hi)
            if not lows:
                return
            lo = np.min(np.array(lows), axis=0)
            hi = np.max(np.array(highs), axis=0)
            center = (lo + hi) / 2.0
            span = float(np.linalg.norm(hi - lo)) or 0.3
            self.setCameraPosition(pos=Vector(*center), distance=span * 1.6,
                                   elevation=18, azimuth=-55)
            self.opts["center"] = Vector(*center)

        def set_mode(self, mode: str) -> None:
            self.state.mode = mode
            self._render()

        def set_exploded(self, factor: float) -> None:
            self.state.exploded = float(max(factor, 0.0))
            self._render()

        def set_section(self, enabled: bool, axis: int = 1, offset: float = 0.0) -> None:
            self.state.section_enabled = bool(enabled)
            self.state.section_axis = int(axis)
            x, y, z = self.opts.get("center", Vector(0, 0, 0))
            span = max(0.05, self.opts.get("distance", 0.6) * 0.5)
            self.state.section_offset = float(offset * span + (x, y, z)[int(axis)])
            self._render()

        def set_orthographic(self, enabled: bool) -> None:
            self.opts["ortho"] = bool(enabled)
            self.update()

        def set_grid_visible(self, visible: bool) -> None:
            self.state.show_grid = bool(visible)
            self._grid.setVisible(self.state.show_grid)

        def highlight(self, component_id: str) -> None:
            if component_id == self._selection:
                return
            self._selection = component_id or ""
            self._render()

        def show_pose(self, poses: dict[str, object]) -> None:
            """Move items to simulation poses given world transforms per component."""
            for cid, transform in poses.items():
                gl_item = self._items.get(cid)
                scene_item = next((i for i in self.scene_items if i.component_id == cid), None)
                if gl_item is None or scene_item is None:
                    continue
                verts = transform.apply_points(scene_item.vertices) if hasattr(transform, "apply_points") \
                    else scene_item.vertices @ transform.rotation.matrix().T + transform.translation
                gl_item.setMeshData(meshdata=gl.MeshData(vertexes=verts, faces=scene_item.triangles),
                                    color=scene_item.color, smooth=False,
                                    drawFaces=self.state.mode != "wireframe",
                                    drawEdges=True)
                gl_item.resetTransform()

        # ---------------------------------------------------------- picking
        def _ray(self, x: float, y: float):
            width = max(self.width(), 1)
            height = max(self.height(), 1)
            ndc_x = 2.0 * x / width - 1.0
            ndc_y = 1.0 - 2.0 * y / height
            try:
                proj = self.projectionMatrix()
                view = self.viewMatrix()
                inv = np.linalg.inv((proj @ view).reshape(4, 4))
            except Exception:
                return None
            p0 = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0])
            p1 = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0])
            p0 = p0[:3] / p0[3]
            p1 = p1[:3] / p1[3]
            direction = p1 - p0
            norm = np.linalg.norm(direction)
            return p0, direction / norm if norm > 1e-12 else direction

        def pick(self, x: float, y: float) -> str:
            ray = self._ray(x, y)
            if ray is None:
                return ""
            origin, direction = ray
            best_t, best_id = np.inf, ""
            for item in self.scene_items:
                verts = self._display_vertices(item)
                if verts.size == 0:
                    continue
                tri = verts[item.triangles]
                v0 = tri[:, 0]
                e1 = tri[:, 1] - v0
                e2 = tri[:, 2] - v0
                h = np.cross(direction, e2)
                a = np.einsum("ij,ij->i", e1, h)
                mask = np.abs(a) > 1e-12
                if not np.any(mask):
                    continue
                f = np.zeros_like(a)
                f[mask] = 1.0 / a[mask]
                s = origin - v0
                u = f * np.einsum("ij,ij->i", s, h)
                q = np.cross(s, e1)
                v = f * np.einsum("j,ij->i", direction, q)
                t = f * np.einsum("ij,ij->i", e2, q)
                hit = mask & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
                if np.any(hit):
                    tmin = float(t[hit].min())
                    if tmin < best_t:
                        best_t, best_id = tmin, item.component_id
            return best_id

        # ------------------------------------------------------------ events
        def mousePressEvent(self, event) -> None:
            self._last_mouse = event.position()
            super().mousePressEvent(event)

        def mouseReleaseEvent(self, event) -> None:
            last = self._last_mouse
            self._last_mouse = None
            if last is not None and event.button() == QtCore.Qt.LeftButton:
                delta = event.position() - last
                if abs(delta.x()) < 4 and abs(delta.y()) < 4:
                    hit = self.pick(event.position().x(), event.position().y())
                    self.componentPicked.emit(hit)
            super().mouseReleaseEvent(event)

        def mouseMoveEvent(self, event) -> None:
            super().mouseMoveEvent(event)
            ray = self._ray(event.position().x(), event.position().y())
            if ray is not None:
                origin, direction = ray
                if abs(direction[2]) > 1e-9:
                    t = -origin[2] / direction[2]
                    p = origin + direction * t
                    self.cursorMoved.emit(float(p[0]), float(p[1]), float(p[2]))

        def keyPressEvent(self, event) -> None:
            key = event.key()
            if key == QtCore.Qt.Key_F:
                self.fit()
            elif key == QtCore.Qt.Key_W:
                self.set_mode("wireframe")
            elif key == QtCore.Qt.Key_S:
                self.set_mode("solid")
            elif key == QtCore.Qt.Key_T:
                self.set_mode("transparent")
            elif key == QtCore.Qt.Key_O:
                self.set_orthographic(not self.opts.get("ortho", False))
            else:
                super().keyPressEvent(event)
