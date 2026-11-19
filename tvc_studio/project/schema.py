"""Project document: everything a saved engineering project contains.

The project owns the part repository, assembly, materials, fastener library, hardware
definitions, motor data, aerodynamics, flight configuration and stored results. It is
plain data with explicit schema versions so older projects can be migrated.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass, field

from ..assembly.constraints import AssemblyConstraint
from ..assembly.model import Assembly
from ..cad.document import PartRepository
from ..components.hardware import HardwareLibrary
from ..components.materials import MaterialLibrary
from ..electronics.flightcomputer import ControlGains
from ..fasteners.catalog import FastenerCatalog
from ..physics.aero import AeroConfig
from ..physics.flight import FlightConfig
from ..physics.propulsion import MotorLibrary, MotorDefinition


CURRENT_FORMAT_VERSION = 1


def _iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ElectricalConfig:
    """Which hardware the avionics uses and how it is wired to the TVC axes."""

    board_id: str = "pico2w"
    imu_id: str = "mpu6050"
    imu_component_id: str = ""
    pitch_servo_id: str = "mg90s"
    yaw_servo_id: str = "mg90s"
    pitch_channel: int = 0
    yaw_channel: int = 1
    supply_voltage: float = 5.0
    loop_rate_hz: float = 250.0
    telemetry_enabled: bool = True
    hil_port: str = "COM3"
    hil_baud: int = 921600

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @staticmethod
    def from_dict(d: dict) -> "ElectricalConfig":
        allowed = ElectricalConfig.__dataclass_fields__
        return ElectricalConfig(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class SimulationSettings:
    mech_dt: float = 0.001
    mech_iterations: int = 12
    gravity_enabled: bool = True
    contacts_enabled: bool = True
    record_every: int = 10
    collision_filters: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mech_dt": self.mech_dt, "mech_iterations": self.mech_iterations,
            "gravity_enabled": self.gravity_enabled, "contacts_enabled": self.contacts_enabled,
            "record_every": self.record_every,
            "collision_filters": [list(p) for p in self.collision_filters],
        }

    @staticmethod
    def from_dict(d: dict) -> "SimulationSettings":
        return SimulationSettings(
            mech_dt=float(d.get("mech_dt", 0.001)),
            mech_iterations=int(d.get("mech_iterations", 12)),
            gravity_enabled=bool(d.get("gravity_enabled", True)),
            contacts_enabled=bool(d.get("contacts_enabled", True)),
            record_every=int(d.get("record_every", 10)),
            collision_filters=[list(p) for p in d.get("collision_filters", [])],
        )


@dataclass
class Project:
    name: str = "Untitled project"
    id: str = field(default_factory=lambda: f"proj_{uuid.uuid4().hex[:10]}")
    format_version: int = CURRENT_FORMAT_VERSION
    created: str = field(default_factory=_iso_now)
    modified: str = field(default_factory=_iso_now)
    units: str = "metric"
    notes: str = ""
    parts: PartRepository = field(default_factory=PartRepository)
    assembly: Assembly = field(default_factory=Assembly)
    materials: MaterialLibrary = field(default_factory=MaterialLibrary)
    fasteners: FastenerCatalog = field(default_factory=FastenerCatalog)
    hardware: HardwareLibrary = field(default_factory=HardwareLibrary)
    motors: MotorLibrary = field(default_factory=MotorLibrary)
    active_motor_id: str = "demo_motor"
    aero: AeroConfig = field(default_factory=AeroConfig)
    flight: FlightConfig = field(default_factory=FlightConfig)
    control: ControlGains = field(default_factory=ControlGains)
    electrical: ElectricalConfig = field(default_factory=ElectricalConfig)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    results: list[dict] = field(default_factory=list)
    component_library: list[dict] = field(default_factory=list)
    path: str = ""

    def __post_init__(self) -> None:
        if self.parts.materials is None:
            self.parts.materials = self.materials
        if self.parts.backend is None:
            from ..cad import backend_available, get_backend
            if backend_available():
                self.parts.backend = get_backend()

    # -- helpers -----------------------------------------------------------
    def motor(self) -> MotorDefinition:
        return self.motors.get(self.active_motor_id)

    def touch(self) -> None:
        self.modified = _iso_now()

    def add_result(self, name: str, payload: dict, keep: int = 5) -> None:
        self.results.append({"name": name, "timestamp": _iso_now(), "data": payload})
        self.results = self.results[-keep:]

    def statistics(self) -> dict:
        return {
            "parts": len(self.parts.parts),
            "components": len(self.assembly.components),
            "constraints": len(self.assembly.constraints),
            "fasteners": len(self.assembly.fasteners),
            "materials": len(self.materials.all()),
            "motors": len(self.motors.all()),
            "results": len(self.results),
        }

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format_version": CURRENT_FORMAT_VERSION,
            "id": self.id,
            "name": self.name,
            "created": self.created,
            "modified": _iso_now(),
            "units": self.units,
            "notes": self.notes,
            "parts": self.parts.to_dict(),
            "assembly": self.assembly.to_dict(),
            "materials": self.materials.to_dict(),
            "fasteners": self.fasteners.to_dict(),
            "hardware": self.hardware.to_dict(),
            "motors": self.motors.to_dict(),
            "active_motor_id": self.active_motor_id,
            "aero": self.aero.to_dict(),
            "flight": self.flight.to_dict(),
            "control": self.control.__dict__,
            "electrical": self.electrical.to_dict(),
            "simulation": self.simulation.to_dict(),
            "results": self.results,
            "component_library": self.component_library,
        }

    @staticmethod
    def from_dict(d: dict, backend=None) -> "Project":
        d = migrate(d)
        project = Project(
            name=str(d.get("name", "Untitled project")),
            id=str(d.get("id", f"proj_{uuid.uuid4().hex[:10]}")),
            format_version=int(d.get("format_version", CURRENT_FORMAT_VERSION)),
            created=str(d.get("created", _iso_now())),
            modified=str(d.get("modified", _iso_now())),
            units=str(d.get("units", "metric")),
            notes=str(d.get("notes", "")),
        )
        project.materials = MaterialLibrary()
        project.materials.load_dict(d.get("materials", {}))
        project.fasteners = FastenerCatalog(materials=project.materials)
        project.fasteners.load_dict(d.get("fasteners", {}))
        project.hardware = HardwareLibrary()
        project.hardware.load_dict(d.get("hardware", {}))
        project.motors = MotorLibrary()
        project.motors.load_dict(d.get("motors", {}))
        project.active_motor_id = str(d.get("active_motor_id", "demo_motor"))
        project.parts = PartRepository(backend=backend, materials=project.materials)
        project.parts.load_dict(d.get("parts", {}))
        project.assembly = Assembly.from_dict(d.get("assembly", {}), AssemblyConstraint.from_dict)
        project.aero = AeroConfig.from_dict(d.get("aero", {}))
        project.flight = FlightConfig.from_dict(d.get("flight", {}))
        project.control = ControlGains(**{k: v for k, v in d.get("control", {}).items()
                                          if k in ControlGains.__dataclass_fields__})
        project.electrical = ElectricalConfig.from_dict(d.get("electrical", {}))
        project.simulation = SimulationSettings.from_dict(d.get("simulation", {}))
        project.results = list(d.get("results", []))
        project.component_library = list(d.get("component_library", []))
        if project.active_motor_id not in project.motors.motors and project.motors.motors:
            project.active_motor_id = next(iter(project.motors.motors))
        return project


def migrate(d: dict) -> dict:
    """Apply forward migrations to older project files, in place."""
    version = int(d.get("format_version", 1))
    if version > CURRENT_FORMAT_VERSION:
        from ..core.errors import ProjectFormatError
        raise ProjectFormatError(
            f"project format version {version} is newer than this application supports "
            f"({CURRENT_FORMAT_VERSION}); update the application"
        )
    if version < 1:  # pragma: no cover - placeholder for future migrations
        d["format_version"] = 1
    # example of a forward migration hook (kept honest: no silent data loss)
    if "electrical" not in d:
        d["electrical"] = ElectricalConfig().to_dict()
    if "simulation" not in d:
        d["simulation"] = SimulationSettings().to_dict()
    if "component_library" not in d:
        d["component_library"] = []
    d["format_version"] = CURRENT_FORMAT_VERSION
    return d
