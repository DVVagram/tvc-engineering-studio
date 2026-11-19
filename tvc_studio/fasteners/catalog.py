"""Fastener library.

Standards data (ISO 261/262 coarse and fine metric threads, ISO 68-1 60 degree profile,
UTS unified threads) is stored as data, not hard-coded into workflows, so custom
fasteners are first-class. Masses are computed from simplified envelopes and are tagged
ESTIMATED unless the user overrides them; real screws vary by head fillet and shank
undercut and the UI says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from ..core.interfaces import Provenance


class FastenerKind(str, Enum):
    MACHINE_SCREW = "machine_screw"
    SELF_TAPPING = "self_tapping"          # thread forming into plastic/wood
    SELF_THREADING = "self_threading"      # forms machine threads in metal/plastic
    SELF_DRILLING = "self_drilling"        # drills and taps in one operation
    BOLT = "bolt"
    SET_SCREW = "set_screw"
    NUT = "nut"
    WASHER = "washer"
    THREADED_INSERT = "threaded_insert"

    def label(self) -> str:
        return {
            "machine_screw": "Machine screw",
            "self_tapping": "Self-tapping screw",
            "self_threading": "Self-threading screw",
            "self_drilling": "Self-drilling screw",
            "bolt": "Bolt",
            "set_screw": "Set screw",
            "nut": "Nut",
            "washer": "Washer",
            "threaded_insert": "Threaded insert",
        }[self.value]


class HeadType(str, Enum):
    SOCKET_HEAD = "socket_head"
    BUTTON_HEAD = "button_head"
    FLAT_HEAD = "flat_head"
    PAN_HEAD = "pan_head"
    HEX_HEAD = "hex_head"
    COUNTERSUNK = "countersunk"
    CHEESE_HEAD = "cheese_head"
    NONE = "none"


class DriveType(str, Enum):
    HEX_SOCKET = "hex_socket"
    PHILLIPS = "phillips"
    POZIDRIV = "pozidriv"
    SLOT = "slot"
    TORX = "torx"
    HEX_EXTERNAL = "hex_external"
    NONE = "none"


class ThreadSeries(str, Enum):
    METRIC_COARSE = "metric_coarse"
    METRIC_FINE = "metric_fine"
    UNIFIED_COARSE = "unified_coarse"  # UNC
    UNIFIED_FINE = "unified_fine"      # UNF


@dataclass(frozen=True)
class ThreadSpec:
    """Thread geometry. Diameters/pitch in metres."""

    designation: str
    major_diameter: float
    pitch: float
    series: ThreadSeries = ThreadSeries.METRIC_COARSE
    thread_angle_deg: float = 60.0
    starts: int = 1

    # ---- ISO 68-1 derived geometry --------------------------------------
    @property
    def height_triangle(self) -> float:
        """H, the fundamental triangle height = (sqrt(3)/2) * pitch."""
        return (math.sqrt(3.0) / 2.0) * self.pitch

    @property
    def thread_depth(self) -> float:
        """h = 5H/8 for the ISO basic external profile."""
        return 0.625 * self.height_triangle

    @property
    def pitch_diameter(self) -> float:
        """d2 = d - 0.6495p (ISO metric)."""
        return self.major_diameter - 0.649519 * self.pitch

    @property
    def minor_diameter_external(self) -> float:
        """d3 = d - 1.2269p (external thread root)."""
        return self.major_diameter - 1.226869 * self.pitch

    @property
    def minor_diameter_internal(self) -> float:
        """D1 = d - 1.0825p (internal thread minor)."""
        return self.major_diameter - 1.082532 * self.pitch

    @property
    def lead(self) -> float:
        return self.pitch * self.starts

    @property
    def helix_angle(self) -> float:
        """lambda = atan(lead / (pi * d2))."""
        return math.atan2(self.lead, math.pi * self.pitch_diameter)

    @property
    def tensile_stress_area(self) -> float:
        """A_t = pi/4 * (d - 0.9382p)^2 (ISO 898-1)."""
        d_eff = self.major_diameter - 0.9382 * self.pitch
        return math.pi / 4.0 * d_eff**2

    def advance_per_radian(self) -> float:
        """Axial advance per radian of rotation: lead / (2 pi)."""
        return self.lead / (2.0 * math.pi)

    def turns_for_depth(self, depth: float) -> float:
        return depth / self.lead

    def rotation_for_depth(self, depth: float) -> float:
        """Rotation in radians required to advance ``depth``."""
        return depth / self.advance_per_radian()

    def is_metric(self) -> bool:
        return self.series in (ThreadSeries.METRIC_COARSE, ThreadSeries.METRIC_FINE)

    def to_dict(self) -> dict:
        return {
            "designation": self.designation,
            "major_diameter": self.major_diameter,
            "pitch": self.pitch,
            "series": self.series.value,
            "thread_angle_deg": self.thread_angle_deg,
            "starts": self.starts,
        }

    @staticmethod
    def from_dict(d: dict) -> "ThreadSpec":
        return ThreadSpec(
            designation=str(d.get("designation", "custom")),
            major_diameter=float(d["major_diameter"]),
            pitch=float(d["pitch"]),
            series=ThreadSeries(d.get("series", "metric_coarse")),
            thread_angle_deg=float(d.get("thread_angle_deg", 60.0)),
            starts=int(d.get("starts", 1)),
        )


# --------------------------------------------------------------------------- #
# Standard thread tables (ISO 261 / ANSI B1.1). Values in millimetres.
# --------------------------------------------------------------------------- #
_METRIC_COARSE_MM: dict[str, tuple[float, float]] = {
    "M1.6": (1.6, 0.35),
    "M2": (2.0, 0.40),
    "M2.5": (2.5, 0.45),
    "M3": (3.0, 0.50),
    "M3.5": (3.5, 0.60),
    "M4": (4.0, 0.70),
    "M5": (5.0, 0.80),
    "M6": (6.0, 1.00),
    "M8": (8.0, 1.25),
    "M10": (10.0, 1.50),
    "M12": (12.0, 1.75),
}

_METRIC_FINE_MM: dict[str, tuple[float, float]] = {
    "M2x0.25": (2.0, 0.25),
    "M2.5x0.35": (2.5, 0.35),
    "M3x0.35": (3.0, 0.35),
    "M4x0.5": (4.0, 0.50),
    "M5x0.5": (5.0, 0.50),
    "M6x0.75": (6.0, 0.75),
    "M8x1.0": (8.0, 1.00),
}

_UNIFIED_IN: dict[str, tuple[float, float, ThreadSeries]] = {
    "#0-80": (0.0600, 80.0, ThreadSeries.UNIFIED_FINE),
    "#2-56": (0.0860, 56.0, ThreadSeries.UNIFIED_COARSE),
    "#4-40": (0.1120, 40.0, ThreadSeries.UNIFIED_COARSE),
    "#6-32": (0.1380, 32.0, ThreadSeries.UNIFIED_COARSE),
    "#8-32": (0.1640, 32.0, ThreadSeries.UNIFIED_COARSE),
    "#10-24": (0.1900, 24.0, ThreadSeries.UNIFIED_COARSE),
    "#10-32": (0.1900, 32.0, ThreadSeries.UNIFIED_FINE),
    "1/4-20": (0.2500, 20.0, ThreadSeries.UNIFIED_COARSE),
    "5/16-18": (0.3125, 18.0, ThreadSeries.UNIFIED_COARSE),
}


def metric_thread(designation: str) -> ThreadSpec:
    key = designation.upper().replace(" ", "")
    if key in _METRIC_COARSE_MM:
        d, p = _METRIC_COARSE_MM[key]
        return ThreadSpec(key, d * 1e-3, p * 1e-3, ThreadSeries.METRIC_COARSE)
    if key in _METRIC_FINE_MM:
        d, p = _METRIC_FINE_MM[key]
        return ThreadSpec(key, d * 1e-3, p * 1e-3, ThreadSeries.METRIC_FINE)
    raise KeyError(f"unknown metric thread designation: {designation}")


def unified_thread(designation: str) -> ThreadSpec:
    if designation not in _UNIFIED_IN:
        raise KeyError(f"unknown unified thread designation: {designation}")
    d_in, tpi, series = _UNIFIED_IN[designation]
    d = d_in * 0.0254
    return ThreadSpec(designation, d, 0.0254 / tpi, series)


def all_thread_specs() -> list[ThreadSpec]:
    specs = [metric_thread(k) for k in _METRIC_COARSE_MM]
    specs += [metric_thread(k) for k in _METRIC_FINE_MM]
    specs += [unified_thread(k) for k in _UNIFIED_IN]
    return specs


@dataclass
class FastenerSpec:
    """One fastener definition; catalog entries and custom fasteners share this type."""

    id: str
    name: str
    kind: FastenerKind
    thread: ThreadSpec
    length: float                      # m, under-head length (or full length for bolts)
    head_type: HeadType = HeadType.SOCKET_HEAD
    drive_type: DriveType = DriveType.HEX_SOCKET
    material_id: str = "steel_1018"
    mass: float = 0.0                  # kg, material mass of screw only
    mass_provenance: Provenance = Provenance.ESTIMATED
    head_diameter: float = 0.0         # m
    head_height: float = 0.0           # m
    drive_size: float = 0.0            # m, hex socket across flats / driver size
    thread_length: float = 0.0         # m of threaded shank (0 = fully threaded)
    tip_angle_deg: float = 118.0       # drill point angle for self-drilling
    pilot_hole_diameter: float = 0.0   # m, recommended pilot for thread-forming screws
    standard: str = "custom"
    notes: str = ""

    @property
    def threaded_length(self) -> float:
        return self.thread_length if self.thread_length > 0 else self.length

    @property
    def unthreaded_length(self) -> float:
        return max(self.length - self.threaded_length, 0.0)

    def recommended_pilot(self) -> float:
        """Pilot diameter: user value, else 0.75-0.82 d for plastics (thread forming)."""
        if self.pilot_hole_diameter > 0:
            return self.pilot_hole_diameter
        if self.kind in (FastenerKind.SELF_TAPPING, FastenerKind.SELF_THREADING):
            return max(self.thread.major_diameter - 0.85 * self.thread.pitch, 0.25 * self.thread.major_diameter)
        return 0.0

    def is_thread_forming(self) -> bool:
        return self.kind in (
            FastenerKind.SELF_TAPPING,
            FastenerKind.SELF_THREADING,
            FastenerKind.SELF_DRILLING,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind.value,
            "thread": self.thread.to_dict(),
            "length": self.length,
            "head_type": self.head_type.value,
            "drive_type": self.drive_type.value,
            "material_id": self.material_id,
            "mass": self.mass,
            "mass_provenance": self.mass_provenance.value,
            "head_diameter": self.head_diameter,
            "head_height": self.head_height,
            "drive_size": self.drive_size,
            "thread_length": self.thread_length,
            "tip_angle_deg": self.tip_angle_deg,
            "pilot_hole_diameter": self.pilot_hole_diameter,
            "standard": self.standard,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "FastenerSpec":
        return FastenerSpec(
            id=str(d["id"]),
            name=str(d.get("name", d["id"])),
            kind=FastenerKind(d.get("kind", "machine_screw")),
            thread=ThreadSpec.from_dict(d.get("thread", {})),
            length=float(d.get("length", 0.01)),
            head_type=HeadType(d.get("head_type", "socket_head")),
            drive_type=DriveType(d.get("drive_type", "hex_socket")),
            material_id=str(d.get("material_id", "steel_1018")),
            mass=float(d.get("mass", 0.0)),
            mass_provenance=Provenance(d.get("mass_provenance", "estimated")),
            head_diameter=float(d.get("head_diameter", 0.0)),
            head_height=float(d.get("head_height", 0.0)),
            drive_size=float(d.get("drive_size", 0.0)),
            thread_length=float(d.get("thread_length", 0.0)),
            tip_angle_deg=float(d.get("tip_angle_deg", 118.0)),
            pilot_hole_diameter=float(d.get("pilot_hole_diameter", 0.0)),
            standard=str(d.get("standard", "custom")),
            notes=str(d.get("notes", "")),
        )


# Head envelope factors per ISO 4762 / ISO 7380 / ISO 7045 / ISO 10642 (approximate)
_HEAD_GEOMETRY: dict[HeadType, tuple[float, float]] = {  # (d_head/d, h_head/d)
    HeadType.SOCKET_HEAD: (1.50, 1.00),
    HeadType.BUTTON_HEAD: (1.83, 0.53),
    HeadType.PAN_HEAD: (1.75, 0.68),
    HeadType.FLAT_HEAD: (1.80, 0.50),
    HeadType.COUNTERSUNK: (2.00, 0.53),
    HeadType.CHEESE_HEAD: (1.75, 0.70),
    HeadType.HEX_HEAD: (1.75, 0.70),
    HeadType.NONE: (1.00, 0.0),
}

_DRIVE_SIZE: dict[tuple[DriveType, str], float] = {}


def _default_drive_size(thread: ThreadSpec, drive: DriveType) -> float:
    d = thread.major_diameter
    if drive == DriveType.HEX_SOCKET:
        return {1.6: 0.55e-3, 2.0: 0.7e-3, 2.5: 0.9e-3, 3.0: 1.1e-3, 4.0: 1.3e-3, 5.0: 2.0e-3,
                6.0: 2.5e-3, 8.0: 3.5e-3, 10.0: 4.0e-3}.get(round(d * 1e3, 2), d * 0.4)
    if drive == DriveType.TORX:
        return d * 0.42
    if drive in (DriveType.PHILLIPS, DriveType.POZIDRIV):
        return d * 0.55
    if drive == DriveType.SLOT:
        return d * 0.20
    if drive == DriveType.HEX_EXTERNAL:
        return d * 0.85
    return d * 0.4


class FastenerCatalog:
    """Library of standard plus custom fasteners."""

    def __init__(self, specs: list[FastenerSpec] | None = None, materials=None) -> None:
        self._specs: dict[str, FastenerSpec] = {}
        self.materials = materials
        for spec in specs or build_standard_catalog():
            self._specs[spec.id] = spec

    def get(self, fastener_id: str) -> FastenerSpec:
        if fastener_id not in self._specs:
            raise KeyError(f"fastener not in library: {fastener_id}")
        return self._specs[fastener_id]

    def all(self) -> list[FastenerSpec]:
        return sorted(self._specs.values(), key=lambda s: (s.kind.value, s.name))

    def by_kind(self, kind: FastenerKind) -> list[FastenerSpec]:
        return [s for s in self.all() if s.kind is kind]

    def add(self, spec: FastenerSpec) -> None:
        self._specs[spec.id] = spec

    def remove(self, fastener_id: str) -> bool:
        return self._specs.pop(fastener_id, None) is not None

    def matching_thread(self, thread: ThreadSpec) -> list[FastenerSpec]:
        return [s for s in self.all() if abs(s.thread.major_diameter - thread.major_diameter) < 1e-6
                and abs(s.thread.pitch - thread.pitch) < 1e-6]

    def to_dict(self) -> dict:
        return {"fasteners": [s.to_dict() for s in self._specs.values()]}

    def load_dict(self, d: dict) -> None:
        for entry in d.get("fasteners", []):
            try:
                self.add(FastenerSpec.from_dict(entry))
            except (KeyError, ValueError, TypeError):
                continue


def estimate_fastener_mass(spec: FastenerSpec, density: float) -> float:
    """Envelope-based mass estimate for a screw (head + shank, holes ignored)."""
    r = spec.thread.major_diameter / 2.0
    shank = math.pi * r * r * spec.length
    head_r = (spec.head_diameter / 2.0) if spec.head_diameter > 0 else spec.thread.major_diameter * 0.75
    head_h = spec.head_height if spec.head_height > 0 else spec.thread.major_diameter * 0.7
    head = math.pi * head_r * head_r * head_h
    if spec.head_type is HeadType.COUNTERSUNK:
        head *= 0.5  # conical head has half the envelope volume
    return density * (shank + head)


def make_spec(
    kind: FastenerKind,
    thread: ThreadSpec,
    length: float,
    head: HeadType = HeadType.SOCKET_HEAD,
    drive: DriveType = DriveType.HEX_SOCKET,
    material_id: str = "steel_1018",
    density: float = 7870.0,
    standard: str = "ISO",
) -> FastenerSpec:
    hr, hh = _HEAD_GEOMETRY.get(head, (1.5, 1.0))
    d = thread.major_diameter
    spec = FastenerSpec(
        id=f"{kind.value}:{thread.designation}x{int(round(length * 1e3))}:{head.value}",
        name=f"{thread.designation} x {length * 1e3:.0f} mm {kind.label().lower()}",
        kind=kind,
        thread=thread,
        length=length,
        head_type=head,
        drive_type=drive,
        material_id=material_id,
        head_diameter=hr * d,
        head_height=hh * d,
        drive_size=_default_drive_size(thread, drive),
        standard=standard,
        mass_provenance=Provenance.ESTIMATED,
    )
    spec.mass = estimate_fastener_mass(spec, density)
    return spec


def build_standard_catalog() -> list[FastenerSpec]:
    """Reasonable default library: M2-M5 machine screws, self-tappers, inserts, hardware."""
    specs: list[FastenerSpec] = []
    lengths_mm = {
        "M2": (6, 8, 10, 12, 16),
        "M2.5": (6, 8, 10, 12, 16, 20),
        "M3": (6, 8, 10, 12, 16, 20, 25),
        "M4": (8, 10, 12, 16, 20, 25, 30),
        "M5": (10, 12, 16, 20, 25, 30),
        "M6": (12, 16, 20, 25, 30),
    }
    for designation, lengths in lengths_mm.items():
        thread = metric_thread(designation)
        for L in lengths:
            specs.append(
                make_spec(
                    FastenerKind.MACHINE_SCREW,
                    thread,
                    L * 1e-3,
                    HeadType.SOCKET_HEAD,
                    DriveType.HEX_SOCKET,
                    "steel_1018",
                    7870.0,
                    "ISO 4762",
                )
            )
        # one button-head and one countersunk variant per size for variety
        specs.append(make_spec(FastenerKind.MACHINE_SCREW, thread, lengths[1] * 1e-3,
                               HeadType.BUTTON_HEAD, DriveType.HEX_SOCKET, "stainless_304", 8000.0, "ISO 7380"))
        specs.append(make_spec(FastenerKind.MACHINE_SCREW, thread, lengths[1] * 1e-3,
                               HeadType.COUNTERSUNK, DriveType.TORX, "stainless_304", 8000.0, "ISO 10642"))
        # thread-forming screws for printed airframes
        specs.append(
            make_spec(FastenerKind.SELF_TAPPING, thread, max(lengths[0], 8) * 1e-3,
                      HeadType.PAN_HEAD, DriveType.PHILLIPS, "steel_1018", 7870.0, "ISO 7049")
        )
        specs.append(
            make_spec(FastenerKind.SELF_THREADING, thread, max(lengths[1], 10) * 1e-3,
                      HeadType.BUTTON_HEAD, DriveType.HEX_SOCKET, "stainless_304", 8000.0, "custom")
        )
        specs.append(
            make_spec(FastenerKind.NUT, thread, 0.0, HeadType.HEX_HEAD, DriveType.HEX_EXTERNAL,
                      "steel_1018", 7870.0, "ISO 4032")
        )
        specs.append(
            make_spec(FastenerKind.WASHER, thread, 0.0, HeadType.NONE, DriveType.NONE,
                      "stainless_304", 8000.0, "ISO 7089")
        )
        specs.append(
            make_spec(FastenerKind.THREADED_INSERT, thread, 5 * 1e-3, HeadType.NONE, DriveType.NONE,
                      "brass", 8500.0, "heat-set insert")
        )

    # imperial hardware
    for designation in ("#4-40", "#6-32", "#8-32", "#10-24"):
        thread = unified_thread(designation)
        specs.append(make_spec(FastenerKind.MACHINE_SCREW, thread, 0.25 * 0.0254,
                               HeadType.SOCKET_HEAD, DriveType.HEX_SOCKET, "steel_1018", 7870.0, "ASME B18.3"))
        specs.append(make_spec(FastenerKind.NUT, thread, 0.0, HeadType.HEX_HEAD, DriveType.HEX_EXTERNAL,
                               "steel_1018", 7870.0, "ASME B18.2.2"))

    # fix nut/washer masses: they are rings, not rods
    for spec in specs:
        if spec.kind is FastenerKind.NUT:
            d = spec.thread.major_diameter
            spec.head_diameter = 1.6 * d
            spec.head_height = 0.8 * d
            area = math.pi / 4.0 * ((1.6 * d) ** 2 - (0.85 * d) ** 2)
            spec.mass = 7870.0 * area * 0.8 * d
        elif spec.kind is FastenerKind.WASHER:
            d = spec.thread.major_diameter
            spec.head_diameter = 2.0 * d
            spec.head_height = 0.2 * d
            area = math.pi / 4.0 * ((2.0 * d) ** 2 - (1.1 * d) ** 2)
            spec.mass = 8000.0 * area * 0.2 * d
        elif spec.kind is FastenerKind.THREADED_INSERT:
            d = spec.thread.major_diameter
            spec.head_diameter = 1.6 * d
            spec.head_height = 5e-3
            area = math.pi / 4.0 * ((1.6 * d) ** 2 - (0.9 * d) ** 2)
            spec.mass = 8500.0 * area * 5e-3

    return specs
