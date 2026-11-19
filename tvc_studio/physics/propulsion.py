"""Solid-motor propulsion data.

Thrust curves are *data*. The application does not generate or optimise propellant
formulations; it imports manufacturer/measured curves (RASP ``.eng``, CSV) and computes
what follows from them: total impulse, burn time, mass flow and time-varying mass.

Provenance matters: a curve marked ESTIMATED is labelled as such everywhere it influences a
result. A synthetic curve ships only as an explicitly-named demo, never as a real motor.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import NotCharacterised
from ..core.interfaces import Provenance


@dataclass
class ThrustPoint:
    time: float
    thrust: float

    def to_dict(self) -> dict:
        return {"time": self.time, "thrust": self.thrust}


@dataclass
class MotorDefinition:
    id: str
    name: str
    manufacturer: str = ""
    designation: str = ""
    dry_mass: float = 0.0             # kg, motor hardware without propellant
    propellant_mass: float = 0.0      # kg
    total_mass: float = 0.0           # kg, launch mass of the motor if known
    thrust_curve: list[ThrustPoint] = field(default_factory=list)
    provenance: Provenance = Provenance.MANUFACTURER
    data_source: str = ""
    notes: str = ""
    delay_after_burn: float = 0.0
    diameter: float = 0.0
    length: float = 0.0
    certified_impulse: float | None = None
    metadata: dict = field(default_factory=dict)

    # -- derived -----------------------------------------------------------
    @property
    def burn_time(self) -> float:
        if not self.thrust_curve:
            return 0.0
        return self.thrust_curve[-1].time

    @property
    def total_impulse(self) -> float:
        if len(self.thrust_curve) < 2:
            return 0.0
        t = np.array([p.time for p in self.thrust_curve])
        f = np.array([p.thrust for p in self.thrust_curve])
        return float(np.trapezoid(f, t))

    @property
    def average_thrust(self) -> float:
        return self.total_impulse / self.burn_time if self.burn_time > 0 else 0.0

    @property
    def launch_mass(self) -> float:
        if self.total_mass > 0:
            return self.total_mass
        return self.dry_mass + self.propellant_mass

    @property
    def is_characterised(self) -> bool:
        return bool(self.thrust_curve) and self.provenance is not Provenance.UNKNOWN

    def thrust_at(self, time: float) -> float:
        if not self.thrust_curve or time < 0:
            return 0.0
        t = np.array([p.time for p in self.thrust_curve])
        f = np.array([p.thrust for p in self.thrust_curve])
        if time >= t[-1]:
            return 0.0
        return float(np.interp(time, t, f))

    def propellant_fraction(self, time: float) -> float:
        """Fraction of propellant remaining (1 = full tank)."""
        if self.burn_time <= 0 or self.propellant_mass <= 0:
            return 1.0
        if time <= 0:
            return 1.0
        if time >= self.burn_time:
            return 0.0
        # integrate thrust share consumed so far
        ts = np.array([p.time for p in self.thrust_curve])
        fs = np.array([p.thrust for p in self.thrust_curve])
        mask = ts <= time
        consumed = float(np.trapezoid(fs[mask], ts[mask])) if mask.sum() > 1 else 0.0
        current = self.thrust_at(time)
        # add partial segment
        idx = int(np.searchsorted(ts, time))
        if 0 < idx < len(ts):
            t0, f0 = ts[idx - 1], fs[idx - 1]
            consumed += 0.5 * (f0 + current) * (time - t0)
        total = self.total_impulse
        if total <= 0:
            return max(1.0 - time / self.burn_time, 0.0)
        return max(1.0 - consumed / total, 0.0)

    def mass_at(self, time: float) -> float:
        return self.dry_mass + self.propellant_mass * self.propellant_fraction(time)

    def mass_flow_at(self, time: float, dt: float = 1e-4) -> float:
        m0 = self.mass_at(max(time - dt, 0.0))
        m1 = self.mass_at(time + dt)
        return (m0 - m1) / (dt + min(max(time, 0.0), dt))

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.thrust_curve:
            problems.append("no thrust curve loaded")
            return problems
        if self.burn_time <= 0:
            problems.append("thrust curve has zero burn time")
        times = [p.time for p in self.thrust_curve]
        if any(b < a for a, b in zip(times, times[1:])):
            problems.append("thrust curve time stamps are not monotonic")
        if any(p.thrust < 0 for p in self.thrust_curve):
            problems.append("thrust curve contains negative thrust")
        if self.dry_mass <= 0:
            problems.append("motor dry mass is not set")
        if self.certified_impulse and abs(self.total_impulse - self.certified_impulse) > 0.05 * self.certified_impulse:
            problems.append(
                f"curve total impulse {self.total_impulse:.1f} N·s differs from the certified "
                f"{self.certified_impulse:.1f} N·s by more than 5%"
            )
        return problems

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "manufacturer": self.manufacturer,
            "designation": self.designation, "dry_mass": self.dry_mass,
            "propellant_mass": self.propellant_mass, "total_mass": self.total_mass,
            "thrust_curve": [p.to_dict() for p in self.thrust_curve],
            "provenance": self.provenance.value, "data_source": self.data_source,
            "notes": self.notes, "delay_after_burn": self.delay_after_burn,
            "diameter": self.diameter, "length": self.length,
            "certified_impulse": self.certified_impulse, "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(d: dict) -> "MotorDefinition":
        return MotorDefinition(
            id=str(d["id"]), name=str(d.get("name", d["id"])),
            manufacturer=str(d.get("manufacturer", "")),
            designation=str(d.get("designation", "")),
            dry_mass=float(d.get("dry_mass", 0.0)),
            propellant_mass=float(d.get("propellant_mass", 0.0)),
            total_mass=float(d.get("total_mass", 0.0)),
            thrust_curve=[ThrustPoint(float(p["time"]), float(p["thrust"])) for p in d.get("thrust_curve", [])],
            provenance=Provenance(d.get("provenance", "manufacturer")),
            data_source=str(d.get("data_source", "")),
            notes=str(d.get("notes", "")),
            delay_after_burn=float(d.get("delay_after_burn", 0.0)),
            diameter=float(d.get("diameter", 0.0)),
            length=float(d.get("length", 0.0)),
            certified_impulse=(None if d.get("certified_impulse") is None else float(d["certified_impulse"])),
            metadata=dict(d.get("metadata", {})),
        )

    # -- import/export -----------------------------------------------------
    @staticmethod
    def from_eng(text: str, name: str | None = None) -> "MotorDefinition":
        """Parse a RASP ``.eng`` thrust curve (RockSim/OpenRocket compatible)."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith(";")]
        if not lines:
            raise ValueError("empty .eng file")
        header = lines[0].split()
        if len(header) < 6:
            raise ValueError("unrecognised .eng header")
        designation = header[0]
        try:
            diameter_mm = float(header[1])
            length_mm = float(header[2])
            delays = header[3]
            prop_mass_g = float(header[4])
            total_mass_g = float(header[5])
        except ValueError as exc:
            raise ValueError(f"invalid .eng header: {exc}") from exc
        curve: list[ThrustPoint] = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                curve.append(ThrustPoint(float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if not curve:
            raise ValueError(".eng file has no thrust points")
        # ensure a zero-thrust tail at and after burnout
        if curve[-1].thrust != 0.0:
            curve.append(ThrustPoint(curve[-1].time + 0.001, 0.0))
        return MotorDefinition(
            id=designation.lower().replace(" ", "_"),
            name=name or designation,
            designation=designation,
            dry_mass=max(total_mass_g - prop_mass_g, 0.0) / 1000.0,
            propellant_mass=prop_mass_g / 1000.0,
            total_mass=total_mass_g / 1000.0,
            thrust_curve=curve,
            provenance=Provenance.MANUFACTURER,
            data_source="RASP .eng import",
            delay_after_burn=float(delays.split("-")[0]) if delays.split("-")[0].replace(".", "").isdigit() else 0.0,
            diameter=diameter_mm / 1000.0,
            length=length_mm / 1000.0,
        )

    @staticmethod
    def from_csv(text: str, name: str = "Imported motor", **kwargs) -> "MotorDefinition":
        """CSV with columns ``time,thrust`` (optional third column ``mass``)."""
        curve: list[ThrustPoint] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("time"):
                continue
            parts = re.split(r"[,\s;]+", line)
            if len(parts) < 2:
                continue
            try:
                curve.append(ThrustPoint(float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if not curve:
            raise ValueError("CSV has no time,thrust rows")
        return MotorDefinition(
            id=kwargs.pop("id", "imported_motor"), name=name, thrust_curve=curve,
            provenance=kwargs.pop("provenance", Provenance.MANUFACTURER), **kwargs,
        )

    def to_eng(self) -> str:
        total_g = self.launch_mass * 1000.0
        prop_g = self.propellant_mass * 1000.0
        designation = (self.designation or self.name or "MOTOR").replace(";", "_").replace(" ", "_")
        header = (f"{designation} {self.diameter*1000:.0f} {self.length*1000:.0f} "
                  f"P {prop_g:.3f} {total_g:.3f} {self.manufacturer or 'unknown'}")
        rows = [f"{p.time:.4f} {p.thrust:.4f}" for p in self.thrust_curve]
        return "\n".join([header] + rows)

    def plot_points(self, samples: int = 400) -> tuple[np.ndarray, np.ndarray]:
        if not self.thrust_curve:
            return np.array([]), np.array([])
        t = np.linspace(0.0, self.burn_time, samples)
        f = np.array([self.thrust_at(float(ti)) for ti in t])
        return t, f


def demo_motor() -> MotorDefinition:
    """Synthetic curve used only by the tutorial assembly.

    It is *not* a real motor and is named accordingly. It exists so the example project can
    demonstrate the pipeline before the user imports manufacturer data.
    """
    # smooth bell-ish curve, 1.2 s burn, peak 42 N, about 28 N·s total
    points = []
    for i in range(25):
        t = i * 1.2 / 24
        x = t / 1.2
        thrust = 42.0 * math.sin(math.pi * min(x * 1.08, 1.0)) ** 0.7 * (1.0 - 0.15 * x)
        if t >= 1.19:
            thrust = 0.0
        points.append(ThrustPoint(round(t, 4), round(max(thrust, 0.0), 3)))
    points[0] = ThrustPoint(0.0, 0.0)
    points[1] = ThrustPoint(0.05, 30.0)
    return MotorDefinition(
        id="demo_motor",
        name="DEMO curve — synthetic, not a real motor",
        manufacturer="TVC Engineering Studio example data",
        designation="DEMO-C1.2",
        dry_mass=0.045,
        propellant_mass=0.020,
        total_mass=0.065,
        thrust_curve=points,
        provenance=Provenance.ESTIMATED,
        data_source="generated demonstration data",
        notes=("Synthetic profile supplied with the tutorial project so the flight pipeline can be "
               "exercised. Replace with manufacturer or measured data before drawing conclusions."),
        diameter=0.029,
        length=0.070,
    )


class MotorLibrary:
    def __init__(self, motors: list[MotorDefinition] | None = None) -> None:
        self.motors: dict[str, MotorDefinition] = {m.id: m for m in (motors or [demo_motor()])}

    def add(self, motor: MotorDefinition) -> MotorDefinition:
        self.motors[motor.id] = motor
        return motor

    def get(self, motor_id: str) -> MotorDefinition:
        if motor_id not in self.motors:
            raise KeyError(f"unknown motor: {motor_id} (import a thrust curve first)")
        return self.motors[motor_id]

    def all(self) -> list[MotorDefinition]:
        return list(self.motors.values())

    def load_eng_file(self, path: str, name: str | None = None) -> MotorDefinition:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            motor = MotorDefinition.from_eng(fh.read(), name)
        motor.data_source = os.path.basename(path)
        return self.add(motor)

    def to_dict(self) -> dict:
        return {"motors": [m.to_dict() for m in self.motors.values()]}

    def load_dict(self, d: dict) -> None:
        for entry in d.get("motors", []):
            try:
                motor = MotorDefinition.from_dict(entry)
            except (KeyError, ValueError, TypeError):
                continue
            self.motors[motor.id] = motor


def require_characterised(motor: MotorDefinition) -> None:
    """Refuse to present a flight result based on a curve nobody supplied."""
    if not motor.is_characterised:
        raise NotCharacterised(
            f"motor '{motor.name}' has no characterised thrust curve; import manufacturer data first"
        )
