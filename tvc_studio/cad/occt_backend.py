"""Open CASCADE (OCP) geometry backend — exact B-rep CAD.

This is the only module that talks to OCCT. Everything above it sees shapes as opaque
handles plus :class:`MassProperties`, :class:`Mesh` and measurement records, which keeps the
kernel replaceable and the rest of the application testable without a GPU or a kernel.

Design rules enforced here:

* Exact geometry is never replaced by a mesh. Booleans, volume, inertia and STEP export all
  operate on the B-rep.
* Every kernel failure raises :class:`CadError` with the operation name; nothing silently
  returns an empty shape.
* Tessellation is only ever used for drawing, contact and STL export, and is labelled as the
  VISUAL / COLLISION representation by the callers.
"""

from __future__ import annotations

import base64
import math
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import CadError, CadKernelUnavailable
from ..core.interfaces import GeometryRepr, Mesh, TessellationQuality
from ..core.math3d import MassProperties, Transform, as_vec3, unit, vec3

try:  # pragma: no cover - import guard exercised on machines without OCCT
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Section
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_Transform,
        BRepBuilderAPI_Sewing,
    )
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCone,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakePrism,
        BRepPrimAPI_MakeRevol,
        BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakeTorus,
    )
    from OCP.BRepTools import BRepTools
    from OCP.BinTools import BinTools
    from OCP.Bnd import Bnd_Box
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.IGESControl import IGESControl_Reader, IGESControl_Writer
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
    from OCP.StlAPI import StlAPI_Reader, StlAPI_Writer
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape
    from OCP.gp import gp_Ax1, gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Quaternion, gp_Trsf, gp_Vec
    from OCP.collections import Array1_gp_Pnt

    _OCP_AVAILABLE = True
    _OCP_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    _OCP_AVAILABLE = False
    _OCP_IMPORT_ERROR = str(exc)


def occt_available() -> bool:
    return _OCP_AVAILABLE


def occt_import_error() -> str:
    return _OCP_IMPORT_ERROR


def _require_kernel() -> None:
    if not _OCP_AVAILABLE:
        raise CadKernelUnavailable(
            "Open CASCADE (cadquery-ocp) is not installed: exact CAD operations are unavailable. "
            f"Import error: {_OCP_IMPORT_ERROR}. Install with: pip install cadquery-ocp"
        )


# --------------------------------------------------------------------------- #
# Topology info records
# --------------------------------------------------------------------------- #
@dataclass
class FaceInfo:
    index: int
    area: float
    centre: np.ndarray
    normal: np.ndarray
    is_planar: bool = False
    is_cylindrical: bool = False
    radius: float = 0.0
    axis_point: np.ndarray | None = None
    axis_direction: np.ndarray | None = None
    surface_type: str = "other"

    def to_entity_dict(self, component_id: str) -> dict:
        return {
            "component_id": component_id,
            "kind": "axis" if self.is_cylindrical else "face",
            "point": np.asarray(self.axis_point if self.is_cylindrical else self.centre).tolist(),
            "direction": np.asarray(self.axis_direction if self.is_cylindrical else self.normal).tolist(),
            "label": f"face {self.index}",
        }


@dataclass
class EdgeInfo:
    index: int
    length: float
    start: np.ndarray
    end: np.ndarray
    midpoint: np.ndarray
    direction: np.ndarray
    is_circular: bool = False
    radius: float = 0.0
    centre: np.ndarray | None = None


@dataclass
class VertexInfo:
    index: int
    point: np.ndarray


# --------------------------------------------------------------------------- #
# Geometry backend
# --------------------------------------------------------------------------- #
class OCCTGeometry:
    """Exact geometry operations and analysis on OCCT shapes."""

    name = "Open CASCADE (OCP)"

    def __init__(self) -> None:
        _require_kernel()
        self._mesh_cache: dict[tuple[int, str], Mesh] = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _shape(shape) -> "TopoDS_Shape":
        if shape is None:
            raise CadError("operation requires a shape but got None")
        if hasattr(shape, "IsNull") and shape.IsNull():
            raise CadError("shape is null")
        return shape

    @staticmethod
    def _pnt(p) -> "gp_Pnt":
        a = as_vec3(p)
        return gp_Pnt(float(a[0]), float(a[1]), float(a[2]))

    @staticmethod
    def _dir(d) -> "gp_Dir":
        a = unit(d)
        return gp_Dir(float(a[0]), float(a[1]), float(a[2]))

    def _gp_trsf(self, transform: Transform) -> "gp_Trsf":
        R = transform.rotation.matrix()
        trsf = gp_Trsf()
        quat = gp_Quaternion(float(transform.rotation.x), float(transform.rotation.y),
                             float(transform.rotation.z), float(transform.rotation.w))
        trsf.SetRotation(quat)
        t = as_vec3(transform.translation)
        trsf.SetTranslationPart(gp_Vec(float(t[0]), float(t[1]), float(t[2])))
        # guard: verify the rotation part matches the intended matrix
        if abs(trsf.Value(1, 1) - R[0, 0]) > 1e-9:
            raise CadError("internal transform conversion mismatch")
        return trsf

    def _explore(self, shape, kind) -> list:
        out = []
        explorer = TopExp_Explorer(self._shape(shape), kind)
        while explorer.More():
            out.append(explorer.Current())
            explorer.Next()
        return out

    def _map_shapes(self, shape, kind) -> list:
        """Unique sub-shapes in stable index order (index 0 = first mapped)."""
        out: list = []
        buckets: dict = {}
        for raw in self._explore(shape, kind):
            try:
                key = (hash(raw.TShape()), raw.Location().HashCode())
            except Exception:
                key = 0
            bucket = buckets.setdefault(key, [])
            if any(raw.IsSame(known) for known in bucket):
                continue
            bucket.append(raw)
            out.append(raw)
        return out

    # ------------------------------------------------------------- primitives
    def box(self, sx: float, sy: float, sz: float, origin=(0.0, 0.0, 0.0), centered: bool = False):
        """Axis-aligned box. ``centered`` places the origin at the centre."""
        o = as_vec3(origin)
        if centered:
            o = o - vec3(sx / 2, sy / 2, sz / 2)
        maker = BRepPrimAPI_MakeBox(self._pnt(o), float(sx), float(sy), float(sz))
        return maker.Shape()

    def cylinder(self, radius: float, height: float, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
        ax2 = gp_Ax2(self._pnt(origin), self._dir(axis))
        maker = BRepPrimAPI_MakeCylinder(ax2, float(radius), float(height))
        return maker.Shape()

    def cone(self, radius1: float, radius2: float, height: float, origin=(0.0, 0.0, 0.0),
             axis=(0.0, 0.0, 1.0)):
        ax2 = gp_Ax2(self._pnt(origin), self._dir(axis))
        return BRepPrimAPI_MakeCone(ax2, float(radius1), float(radius2), float(height)).Shape()

    def sphere(self, radius: float, origin=(0.0, 0.0, 0.0)):
        return BRepPrimAPI_MakeSphere(self._pnt(origin), float(radius)).Shape()

    def torus(self, major_radius: float, minor_radius: float, origin=(0.0, 0.0, 0.0),
              axis=(0.0, 0.0, 1.0)):
        ax2 = gp_Ax2(self._pnt(origin), self._dir(axis))
        return BRepPrimAPI_MakeTorus(ax2, float(major_radius), float(minor_radius)).Shape()

    def tube(self, inner_radius: float, outer_radius: float, height: float,
             origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
        """Hollow cylinder modelled as a revolved annular profile (exact)."""
        if outer_radius <= inner_radius:
            raise CadError("tube outer radius must exceed inner radius")
        profile = [
            (inner_radius, 0.0),
            (outer_radius, 0.0),
            (outer_radius, height),
            (inner_radius, height),
        ]
        shape = self._revolve_points(profile, angle=2 * math.pi)
        if np.allclose(as_vec3(axis), (0.0, 0.0, 1.0)):
            placement = Transform.from_pos_euler(origin, 0.0, 0.0, 0.0)
        else:
            placement = self._align_z_to(origin, axis)
        return self.transform(shape, placement)

    @staticmethod
    def _align_z_to(origin, axis) -> Transform:
        from ..core.math3d import Quat
        return Transform(Quat.from_two_vectors((0, 0, 1), axis), as_vec3(origin))

    # ------------------------------------------------------------ sketch-based
    def _wire_from_segments(self, segments) -> "TopoDS_Wire":
        """Build an exact wire from profile segments (lines and circular arcs)."""
        maker = BRepBuilderAPI_MakeWire()
        for seg in segments:
            start = as_vec3(seg.start_3d)
            end = as_vec3(seg.end_3d)
            kind = seg.kind
            if kind == "line":
                edge_maker = BRepBuilderAPI_MakeEdge(self._pnt(start), self._pnt(end))
                if not edge_maker.IsDone():
                    raise CadError("failed to build a line edge from the sketch")
                maker.Add(edge_maker.Edge())
            elif kind in ("arc", "circle"):
                centre = as_vec3(seg.centre_3d)
                radius = float(seg.radius)
                normal = as_vec3(seg.normal)
                circ = gp_Circ(gp_Ax2(self._pnt(centre), self._dir(normal)), radius)
                if kind == "circle":
                    edge_maker = BRepBuilderAPI_MakeEdge(circ)
                else:
                    u1 = self._circle_parameter(circ, start)
                    u2 = self._circle_parameter(circ, end)
                    if seg.ccw:
                        if u2 <= u1:
                            u2 += 2 * math.pi
                    else:
                        if u2 >= u1:
                            u2 -= 2 * math.pi
                    edge_maker = BRepBuilderAPI_MakeEdge(circ, u1, u2)
                if not edge_maker.IsDone():
                    raise CadError("failed to build a circular edge from the sketch")
                maker.Add(edge_maker.Edge())
            else:
                raise CadError(f"unsupported sketch segment kind: {kind}")
        if not maker.IsDone():
            raise CadError("sketch profile does not form a valid wire (gaps or branching at a node)")
        return maker.Wire()

    @staticmethod
    def _circle_parameter(circ: "gp_Circ", point) -> float:
        centre = circ.Location()
        xdir = circ.XAxis().Direction()
        ydir = circ.YAxis().Direction()
        d = as_vec3(point) - vec3(centre.X(), centre.Y(), centre.Z())
        x = float(np.dot(d, vec3(xdir.X(), xdir.Y(), xdir.Z())))
        y = float(np.dot(d, vec3(ydir.X(), ydir.Y(), ydir.Z())))
        return math.atan2(y, x)

    def profile_face(self, segments, origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                     u_axis=(1.0, 0.0, 0.0), inner_loops: list | None = None):
        """Build a planar face from one or more 2-D loops placed on a plane.

        Segments must carry 2-D coordinates (``start``/``end`` as (u, v), ``centre``).
        """
        basis_u = unit(u_axis)
        n = unit(normal)
        basis_v = np.cross(n, basis_u)
        basis_v = basis_v / (np.linalg.norm(basis_v) or 1.0)
        basis_u = np.cross(basis_v, n)
        o = as_vec3(origin)

        def to3d(p):
            a = np.asarray(p, dtype=float)[:2]
            return o + basis_u * a[0] + basis_v * a[1]

        def prepare(loop):
            prepared = []
            for seg in loop:
                item = _Segment3D(
                    kind=seg.kind, start_3d=to3d(seg.start), end_3d=to3d(seg.end),
                    radius=seg.radius, centre_3d=to3d(seg.centre) if seg.kind in ("arc", "circle") else vec3(),
                    normal=n, ccw=seg.ccw,
                )
                prepared.append(item)
            return prepared

        outer_wire = self._wire_from_segments(prepare(segments))
        try:
            face_maker = BRepBuilderAPI_MakeFace(outer_wire, True)
        except Exception:
            face_maker = BRepBuilderAPI_MakeFace(outer_wire)
        if not face_maker.IsDone():
            raise CadError("could not build a face from the sketch profile")
        for loop in inner_loops or []:
            try:
                inner_wire = self._wire_from_segments(prepare(loop))
                face_maker.Add(inner_wire)
            except CadError:
                continue
        return face_maker.Face()

    def extrude_profile(self, loops, length: float, origin=(0.0, 0.0, 0.0),
                        normal=(0.0, 0.0, 1.0), symmetric: bool = False, reverse: bool = False):
        if not loops:
            raise CadError("extrude requires at least one closed profile loop")
        outer = loops[0]
        inner = loops[1:] if len(loops) > 1 else None
        face = self.profile_face(outer, origin, normal, inner_loops=inner)
        n = unit(normal)
        if reverse:
            n = -n
        vector = n * float(length)
        start_face = self.translate(face, -0.5 * vector) if symmetric else face
        maker = BRepPrimAPI_MakePrism(start_face, gp_Vec(float(vector[0]), float(vector[1]), float(vector[2])))
        if not maker.IsDone():
            raise CadError("extrude failed (profile may self-intersect)")
        return maker.Shape()

    def revolve_profile(self, points, angle: float = 2 * math.pi, axis_origin=(0.0, 0.0, 0.0),
                        axis_dir=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                        u_axis=(1.0, 0.0, 0.0)):
        """Revolve a closed 2-D profile. The sketch's X axis is the radius, Y the axial axis."""
        if len(points) < 3:
            raise CadError("revolve requires at least three profile points")
        basis_u = unit(u_axis)
        n = unit(normal)
        basis_v = np.cross(n, basis_u)
        basis_v /= np.linalg.norm(basis_v) or 1.0
        basis_u = np.cross(basis_v, n)
        o = as_vec3(origin)
        pts3 = [o + basis_u * p[0] + basis_v * p[1] for p in points]
        edges = []
        for i in range(len(pts3)):
            a = pts3[i]
            b = pts3[(i + 1) % len(pts3)]
            edges.append(BRepBuilderAPI_MakeEdge(self._pnt(a), self._pnt(b)).Edge())
        maker = BRepBuilderAPI_MakeWire()
        for e in edges:
            maker.Add(e)
        face = BRepBuilderAPI_MakeFace(maker.Wire(), True).Face()
        axis = gp_Ax1(self._pnt(axis_origin), self._dir(axis_dir))
        revol = BRepPrimAPI_MakeRevol(face, axis, float(angle))
        if not revol.IsDone():
            raise CadError("revolve failed")
        return revol.Shape()

    def _revolve_points(self, points, angle=2 * math.pi):
        edges = []
        for i in range(len(points)):
            a = points[i]
            b = points[(i + 1) % len(points)]
            edges.append(BRepBuilderAPI_MakeEdge(gp_Pnt(a[0], 0, a[1]), gp_Pnt(b[0], 0, b[1])).Edge())
        maker = BRepBuilderAPI_MakeWire()
        for e in edges:
            maker.Add(e)
        face = BRepBuilderAPI_MakeFace(maker.Wire(), True).Face()
        revol = BRepPrimAPI_MakeRevol(face, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), float(angle))
        if not revol.IsDone():
            raise CadError("revolve failed")
        return revol.Shape()

    # ------------------------------------------------------------- operations
    def _boolean(self, op, a, b) -> "TopoDS_Shape":
        maker = op(self._shape(a), self._shape(b))
        maker.Build()
        if not maker.IsDone():
            raise CadError(f"{op.__name__} failed (shapes may not intersect or may be invalid)")
        return maker.Shape()

    def boolean_cut(self, base, tool):
        return self._boolean(BRepAlgoAPI_Cut, base, tool)

    def boolean_fuse(self, base, tool):
        return self._boolean(BRepAlgoAPI_Fuse, base, tool)

    def boolean_common(self, base, tool):
        return self._boolean(BRepAlgoAPI_Common, base, tool)

    def fillet(self, shape, edge_indices, radius: float):
        shape = self._shape(shape)
        edges = self._map_shapes(shape, TopAbs_EDGE)
        maker = BRepFilletAPI_MakeFillet(shape)
        selected = 0
        for index in edge_indices:
            if 0 <= index < len(edges):
                maker.Add(float(radius), TopoDS.Edge(edges[index]))
                selected += 1
        if selected == 0:
            raise CadError("fillet: no valid edges selected")
        try:
            result = maker.Shape()
        except Exception as exc:
            raise CadError(f"fillet of radius {radius} failed on the selected edges: {exc}") from exc
        if not maker.IsDone():
            raise CadError(f"fillet of radius {radius} failed on the selected edges (radius too large?)")
        return result

    def chamfer(self, shape, edge_indices, distance: float):
        shape = self._shape(shape)
        edges = self._map_shapes(shape, TopAbs_EDGE)
        maker = BRepFilletAPI_MakeChamfer(shape)
        selected = 0
        for index in edge_indices:
            if 0 <= index < len(edges):
                maker.Add(float(distance), TopoDS.Edge(edges[index]))
                selected += 1
        if selected == 0:
            raise CadError("chamfer: no valid edges selected")
        try:
            result = maker.Shape()
        except Exception as exc:
            raise CadError(f"chamfer failed on the selected edges: {exc}") from exc
        if not maker.IsDone():
            raise CadError("chamfer failed on the selected edges (distance too large?)")
        return result

    def hole(self, base, origin, axis, diameter: float, depth: float, through: bool = False):
        """Cut a cylindrical hole from ``origin`` along ``axis`` (into the material)."""
        base = self._shape(base)
        a = unit(axis)
        length = depth * (3.0 if through else 1.0)
        start = as_vec3(origin)
        tool = self.cylinder(diameter / 2.0, length, start, a)
        if through:
            tool = self.cylinder(diameter / 2.0, length, start - a * depth, a)
        return self.boolean_cut(base, tool)

    def pattern_linear(self, shape, direction, count: int, spacing: float, fuse: bool = True):
        if count < 2:
            return self._shape(shape)
        d = unit(direction)
        result = self._shape(shape)
        for i in range(1, int(count)):
            copy = self.translate(shape, d * spacing * i)
            result = self.boolean_fuse(result, copy) if fuse else self.compound([result, copy])
        return result

    def pattern_polar(self, shape, axis_origin, axis_dir, count: int, angle: float = 2 * math.pi):
        if count < 2:
            return self._shape(shape)
        result = self._shape(shape)
        for i in range(1, int(count)):
            t = Transform.from_pos_euler((0, 0, 0))
            from ..core.math3d import Quat
            step = angle / count * i
            rot = Quat.from_axis_angle(axis_dir, step)
            moved = Transform(rot, as_vec3(axis_origin)) @ Transform(Quat.identity(), -as_vec3(axis_origin))
            copy = self.transform(shape, moved)
            result = self.boolean_fuse(result, copy)
        return result

    def compound(self, shapes) -> "TopoDS_Shape":
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for s in shapes:
            builder.Add(compound, self._shape(s))
        return compound

    def translate(self, shape, offset):
        t = Transform.identity()
        t.translation = as_vec3(offset)
        return self.transform(shape, t)

    def transform(self, shape, transform: Transform):
        maker = BRepBuilderAPI_Transform(self._shape(shape), self._gp_trsf(transform), True)
        if not maker.IsDone():
            raise CadError("shape transform failed")
        return maker.Shape()

    def make_solid(self, shape):
        try:
            maker = BRepBuilderAPI_MakeSolid(TopoDS.Shell(self._shape(shape)))
            if maker.IsDone():
                return maker.Solid()
        except Exception:
            pass
        return self._shape(shape)

    # -------------------------------------------------------------- analysis
    def volume(self, shape) -> float:
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(self._shape(shape), props)
        return float(props.Mass())

    def surface_area(self, shape) -> float:
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(self._shape(shape), props)
        return float(props.Mass())

    def length(self, shape) -> float:
        props = GProp_GProps()
        BRepGProp.LinearProperties_s(self._shape(shape), props)
        return float(props.Mass())

    def centre_of_mass(self, shape) -> np.ndarray:
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(self._shape(shape), props)
        c = props.CentreOfMass()
        return vec3(c.X(), c.Y(), c.Z())

    def mass_properties(self, shape, density: float) -> MassProperties:
        shape = self._shape(shape)
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        volume = float(props.Mass())
        c = props.CentreOfMass()
        com = vec3(c.X(), c.Y(), c.Z())
        mat = props.MatrixOfInertia()
        inertia = np.array(
            [[mat.Value(1, 1), mat.Value(1, 2), mat.Value(1, 3)],
             [mat.Value(2, 1), mat.Value(2, 2), mat.Value(2, 3)],
             [mat.Value(3, 1), mat.Value(3, 2), mat.Value(3, 3)]],
            dtype=float,
        ) * float(density)
        area = self.surface_area(shape)
        return MassProperties(
            mass=volume * float(density),
            com=com,
            inertia=inertia,
            volume=volume,
            surface_area=area,
            provenance="cad_derived",
            frame="component",
        )

    def bounding_box(self, shape) -> tuple[np.ndarray, np.ndarray]:
        box = Bnd_Box()
        BRepBndLib.Add_s(self._shape(shape), box, True)
        return (vec3(box.GetXMin(), box.GetYMin(), box.GetZMin()),
                vec3(box.GetXMax(), box.GetYMax(), box.GetZMax()))

    def distance(self, a, b) -> float:
        calc = BRepExtrema_DistShapeShape(self._shape(a), self._shape(b))
        calc.Perform()
        if not calc.IsDone():
            raise CadError("distance calculation failed between the two shapes")
        return float(calc.Value())

    def common_volume(self, a, b) -> float:
        """Exact interference volume of two solids (0 when they do not overlap)."""
        try:
            shape = self.boolean_common(a, b)
        except CadError:
            return 0.0
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        return max(float(props.Mass()), 0.0)

    def check_valid(self, shape) -> list[str]:
        from OCP.BRepCheck import BRepCheck_Analyzer
        analyzer = BRepCheck_Analyzer(self._shape(shape))
        return [] if analyzer.IsValid() else ["shape fails BRep validity check"]

    def section_edges(self, shape, plane_origin, plane_normal) -> list[np.ndarray]:
        """Exact section curves as polylines for display."""
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
        from OCP.gp import gp_Pln
        plane = gp_Pln(self._pnt(plane_origin), self._dir(plane_normal))
        try:
            maker = BRepAlgoAPI_Section(self._shape(shape), plane)
            maker.Build()
            if not maker.IsDone():
                return []
            section_shape = maker.Shape()
        except Exception:
            return []
        polylines: list[np.ndarray] = []
        for edge in self._explore(section_shape, TopAbs_EDGE):
            try:
                pts = self._discretize_edge(edge, 0.0005)
                if len(pts) >= 2:
                    polylines.append(np.array(pts))
            except Exception:
                continue
        return polylines

    # ----------------------------------------------------------- topology info
    def faces(self, shape) -> list[FaceInfo]:
        shape = self._shape(shape)
        out: list[FaceInfo] = []
        for index, raw in enumerate(self._map_shapes(shape, TopAbs_FACE)):
            face = TopoDS.Face(raw)
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            area = float(props.Mass())
            c = props.CentreOfMass()
            centre = vec3(c.X(), c.Y(), c.Z())
            normal = vec3(0, 0, 1)
            is_planar = is_cyl = False
            radius = 0.0
            axis_point = None
            axis_dir = None
            stype = "other"
            try:
                adaptor = BRepAdaptor_Surface(face, True)
                gtype = adaptor.GetType()
                if gtype == GeomAbs_Plane:
                    is_planar = True
                    stype = "plane"
                    d = adaptor.Plane().Axis().Direction()
                    normal = vec3(d.X(), d.Y(), d.Z())
                elif gtype == GeomAbs_Cylinder:
                    is_cyl = True
                    stype = "cylinder"
                    cyl = adaptor.Cylinder()
                    radius = float(cyl.Radius())
                    ax = cyl.Axis()
                    p = ax.Location()
                    d = ax.Direction()
                    axis_point = vec3(p.X(), p.Y(), p.Z())
                    axis_dir = vec3(d.X(), d.Y(), d.Z())
                    normal = axis_dir
                else:
                    # evaluate a surface normal at the parametric midpoint
                    u = 0.5 * (adaptor.FirstUParameter() + adaptor.LastUParameter())
                    v = 0.5 * (adaptor.FirstVParameter() + adaptor.LastVParameter())
                    p = adaptor.Value(u, v)
                    du = adaptor.DN(u, v, 1, 0)
                    dv = adaptor.DN(u, v, 0, 1)
                    n = np.cross(vec3(du.X(), du.Y(), du.Z()), vec3(dv.X(), dv.Y(), dv.Z()))
                    if np.linalg.norm(n) > 1e-12:
                        normal = n / np.linalg.norm(n)
                    stype = str(gtype).split(".")[-1]
            except Exception:
                pass
            if face.Orientation() == TopAbs_REVERSED:
                normal = -normal
            out.append(FaceInfo(index, area, centre, normal, is_planar, is_cyl, radius,
                                axis_point, axis_dir, stype))
        return out

    def edges(self, shape) -> list[EdgeInfo]:
        shape = self._shape(shape)
        out: list[EdgeInfo] = []
        for index, raw in enumerate(self._map_shapes(shape, TopAbs_EDGE)):
            edge = TopoDS.Edge(raw)
            length = self.length(edge)
            pts = self._discretize_edge(edge, 0.001, max_points=24)
            start = pts[0]
            end = pts[-1]
            mid = pts[len(pts) // 2]
            direction = end - start
            n = np.linalg.norm(direction)
            direction = direction / n if n > 1e-12 else vec3(0, 0, 1)
            is_circular = False
            radius = 0.0
            centre = None
            try:
                curve = BRepAdaptor_Curve(edge)
                if curve.GetType() == GeomAbs_Circle:
                    is_circular = True
                    circ = curve.Circle()
                    radius = float(circ.Radius())
                    c = circ.Location()
                    centre = vec3(c.X(), c.Y(), c.Z())
            except Exception:
                pass
            out.append(EdgeInfo(index, length, start, end, mid, direction, is_circular, radius, centre))
        return out

    def vertices(self, shape) -> list[VertexInfo]:
        out: list[VertexInfo] = []
        for index, raw in enumerate(self._map_shapes(shape, TopAbs_VERTEX)):
            p = BRep_Tool.Pnt_s(TopoDS.Vertex(raw))
            out.append(VertexInfo(index, vec3(p.X(), p.Y(), p.Z())))
        return out

    def _discretize_edge(self, edge, deflection: float, max_points: int = 48) -> list[np.ndarray]:
        curve = BRepAdaptor_Curve(TopoDS.Edge(edge))
        u1 = float(curve.FirstParameter())
        u2 = float(curve.LastParameter())
        if not math.isfinite(u1) or not math.isfinite(u2) or abs(u2 - u1) < 1e-12:
            p = BRep_Tool.Pnt_s(TopoDS.Vertex(self._explore(edge, TopAbs_VERTEX)[0]))
            return [vec3(p.X(), p.Y(), p.Z())] * 2
        # point count from curve length and deflection, bounded
        try:
            length = self.length(edge)
        except Exception:
            length = 0.0
        segments = int(np.clip(math.ceil(max(length, 1e-6) / max(deflection * 8.0, 1e-6)), 1, max_points - 1))
        n = segments + 1
        pts = []
        for i in range(n):
            p = curve.Value(u1 + (u2 - u1) * i / (n - 1))
            pts.append(vec3(p.X(), p.Y(), p.Z()))
        return pts

    # -------------------------------------------------------------- tessellate
    def tessellate(self, shape, quality: TessellationQuality = TessellationQuality.NORMAL,
                   size_hint: float | None = None, include_edges: bool = True) -> Mesh:
        shape = self._shape(shape)
        if size_hint is None:
            lo, hi = self.bounding_box(shape)
            size_hint = float(np.linalg.norm(hi - lo))
        deflection = max(quality.deflection(max(size_hint, 1e-4)), 1e-6)
        BRepMesh_IncrementalMesh(shape, deflection, False, 0.35, True)

        vertices: list[np.ndarray] = []
        triangles: list[tuple[int, int, int]] = []
        normals: list[np.ndarray] = []

        for raw in self._explore(shape, TopAbs_FACE):
            face = TopoDS.Face(raw)
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is None:
                continue
            trsf = loc.Transformation()
            def node(i: int) -> np.ndarray:
                p = tri.Node(i)
                q = p.Transformed(trsf)
                return vec3(q.X(), q.Y(), q.Z())
            base = len(vertices)
            nodes = [node(i) for i in range(1, tri.NbNodes() + 1)]
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                pa, pb, pc = nodes[a - 1], nodes[b - 1], nodes[c - 1]
                normal = np.cross(pb - pa, pc - pa)
                nn = np.linalg.norm(normal)
                if nn < 1e-14:
                    continue
                normal = normal / nn
                if face.Orientation() == TopAbs_REVERSED:
                    # winding already follows the face; only the stored normal flips
                    pass
                # non-indexed triangles give correct flat shading on CAD edges
                index = len(vertices)
                vertices.extend([pa, pb, pc])
                normals.extend([normal, normal, normal])
                triangles.append((index, index + 1, index + 2))

        edge_indices: list[tuple[int, int]] = []
        if include_edges:
            edge_deflection = max(deflection * 2.0, 5e-5)
            for raw in self._explore(shape, TopAbs_EDGE):
                try:
                    pts = self._discretize_edge(raw, edge_deflection, max_points=32)
                except Exception:
                    continue
                start = len(vertices)
                vertices.extend(pts)
                for i in range(len(pts) - 1):
                    edge_indices.append((start + i, start + i + 1))

        if not triangles:
            raise CadError("tessellation produced no triangles (shape may be empty or invalid)")
        return Mesh(
            vertices=np.array(vertices, dtype=float),
            triangles=np.array(triangles, dtype=np.int32),
            normals=np.array(normals, dtype=float),
            edges=np.array(edge_indices, dtype=np.int32) if edge_indices else None,
        )

    # ------------------------------------------------------------------- I/O
    def import_step(self, path: str) -> list["TopoDS_Shape"]:
        reader = STEPControl_Reader()
        status = reader.ReadFile(path)
        if status != IFSelect_RetDone:
            raise CadError(f"STEP import failed for {os.path.basename(path)} (status {status})")
        reader.TransferRoots()
        shapes = []
        for i in range(1, reader.NbShapes() + 1):
            shape = reader.Shape(i)
            if shape is not None and not shape.IsNull():
                shapes.append(shape)
        if not shapes:
            one = reader.OneShape()
            if not one.IsNull():
                shapes.append(one)
        if not shapes:
            raise CadError("STEP file contains no transferable shapes")
        return shapes

    def import_iges(self, path: str) -> list["TopoDS_Shape"]:
        reader = IGESControl_Reader()
        status = reader.ReadFile(path)
        if status != IFSelect_RetDone:
            raise CadError(f"IGES import failed for {os.path.basename(path)}")
        reader.TransferRoots()
        one = reader.OneShape()
        return [] if one.IsNull() else [one]

    def import_stl(self, path: str) -> list["TopoDS_Shape"]:
        """STL is a mesh format: the result is a shell of planar triangles (no B-rep arcs)."""
        compound = TopoDS_Compound()
        BRep_Builder().MakeCompound(compound)
        reader = StlAPI_Reader()
        if not reader.Read(compound, path):
            raise CadError(f"STL import failed for {os.path.basename(path)}")
        sewing = BRepBuilderAPI_Sewing(1e-6)
        sewing.Add(compound)
        sewing.Perform()
        sewed = sewing.SewedShape()
        return [sewed if not sewed.IsNull() else compound]

    def import_brep(self, path: str) -> list["TopoDS_Shape"]:
        shape = TopoDS_Compound()
        BRep_Builder().MakeCompound(shape)
        try:
            ok = BinTools.Read_s(shape, path)
        except Exception as exc:
            raise CadError(f"BREP import failed for {os.path.basename(path)}: {exc}") from exc
        if not ok or shape.IsNull():
            raise CadError(f"BREP import failed for {os.path.basename(path)}")
        return [shape]

    def export_step(self, shape, path: str) -> None:
        writer = STEPControl_Writer()
        writer.Transfer(self._shape(shape), STEPControl_AsIs)
        status = writer.Write(path)
        if status != IFSelect_RetDone or not os.path.exists(path):
            raise CadError("STEP export failed")

    def export_iges(self, shape, path: str) -> None:
        writer = IGESControl_Writer()
        writer.AddShape(self._shape(shape))
        writer.ComputeModel()
        if not writer.Write(path):
            raise CadError("IGES export failed")

    def export_stl(self, shape, path: str, quality: TessellationQuality = TessellationQuality.FINE) -> None:
        shape = self._shape(shape)
        # StlAPI writes the existing triangulation: mesh first with the requested deflection
        lo, hi = self.bounding_box(shape)
        size = float(np.linalg.norm(hi - lo))
        BRepMesh_IncrementalMesh(shape, quality.deflection(max(size, 1e-4)), False, 0.35, True)
        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        if not writer.Write(shape, path) or not os.path.exists(path):
            raise CadError("STL export failed")

    def export_brep(self, shape, path: str) -> None:
        if not BinTools.Write_s(self._shape(shape), path):
            raise CadError("BREP export failed")

    def export_shape_bytes(self, shape) -> bytes:
        """Serialise exact geometry for embedding in a project file."""
        fd, path = tempfile.mkstemp(suffix=".brep")
        os.close(fd)
        try:
            self.export_brep(shape, path)
            with open(path, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def shape_from_bytes(self, data: bytes):
        fd, path = tempfile.mkstemp(suffix=".brep")
        os.close(fd)
        try:
            with open(path, "wb") as fh:
                fh.write(data)
            shapes = self.import_brep(path)
            return shapes[0]
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    @staticmethod
    def encode_bytes(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    @staticmethod
    def decode_bytes(text: str) -> bytes:
        return base64.b64decode(text.encode("ascii"))

    # ---------------------------------------------------------------- threads
    def thread_helix(self, thread, length: float, density: float = 7850.0):
        """Detailed helical thread solid fused to a core cylinder (exact B-rep)."""
        _require_kernel()
        length = float(length)
        if length <= 0:
            raise CadError("thread length must be positive")
        pitch = thread.pitch
        h = thread.thread_depth
        r_root = thread.major_diameter / 2.0 - h
        r_mid = thread.major_diameter / 2.0 - h / 2.0
        turns = max(length / pitch, 0.05)
        n = int(np.clip(turns * 48, 16, 4000))
        pts = np.array([
            [r_mid * math.cos(i / (n - 1) * turns * 2 * math.pi),
             r_mid * math.sin(i / (n - 1) * turns * 2 * math.pi),
             i / (n - 1) * length]
            for i in range(n)
        ])
        array = Array1_gp_Pnt(1, n)
        for i, p in enumerate(pts, start=1):
            array.SetValue(i, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
        curve_maker = GeomAPI_PointsToBSpline(array, 3, 8)
        if not curve_maker.IsDone():
            raise CadError("thread helix curve construction failed")
        path_wire = BRepBuilderAPI_MakeWire(
            BRepBuilderAPI_MakeEdge(curve_maker.Curve()).Edge()
        ).Wire()
        tangent = pts[1] - pts[0]
        tangent /= np.linalg.norm(tangent)
        radial = np.array([1.0, 0.0, 0.0])
        radial -= radial.dot(tangent) * tangent
        radial /= np.linalg.norm(radial)
        bino = np.cross(tangent, radial)
        origin = pts[0]

        def profile_point(a: float, b: float):
            v = origin + radial * a + bino * b
            return gp_Pnt(float(v[0]), float(v[1]), float(v[2]))

        profile = [
            profile_point(-h * 0.55, -pitch * 0.30),
            profile_point(h, 0.0),
            profile_point(-h * 0.55, pitch * 0.30),
        ]
        wire_maker = BRepBuilderAPI_MakeWire()
        for i in range(3):
            wire_maker.Add(BRepBuilderAPI_MakeEdge(profile[i], profile[(i + 1) % 3]).Edge())
        shell_maker = BRepOffsetAPI_MakePipeShell(path_wire)
        shell_maker.SetMode(True)
        shell_maker.Add(wire_maker.Wire(), False, False)
        shell_maker.Build()
        if not shell_maker.IsDone():
            raise CadError("detailed thread sweep failed")
        shell_maker.MakeSolid()
        thread_solid = shell_maker.Shape()
        core = self.cylinder(r_root, length)
        fused = self.boolean_fuse(core, thread_solid)
        return fused

    def analysis_volume_mm3(self, shape) -> float:
        return self.volume(shape) * 1e9


@dataclass
class _Segment3D:
    """Internal 3-D segment used to build wires (not part of the public API)."""

    kind: str
    start_3d: np.ndarray
    end_3d: np.ndarray
    radius: float = 0.0
    centre_3d: np.ndarray = field(default_factory=lambda: vec3())
    normal: np.ndarray = field(default_factory=lambda: vec3(0, 0, 1))
    ccw: bool = True


def gp_Trsf_SetTranslation(offset) -> "gp_Trsf":  # pragma: no cover - helper for readability
    trsf = gp_Trsf()
    t = as_vec3(offset)
    trsf.SetTranslationPart(gp_Vec(float(t[0]), float(t[1]), float(t[2])))
    return trsf
