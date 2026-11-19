"""Fixed physical configuration of the rocket and its motor.

Every value in this module is a physical property of the hardware. They are frozen
dataclasses: nothing in the interface can change them, and the flight model reads them
directly. Only ``rocket_sim.motor.MotorPerformance`` (thrust, impulse, burn time) is
user-editable.

Coordinates: the body frame has its origin at the centre of gravity, +X pointing towards
the nose and +Z pointing up when the rocket stands on the pad. Distances along the rocket
are given from the nose tip in the documentation of each field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G0 = 9.80665  # m/s^2
AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m^3
ATMOSPHERE_SCALE_HEIGHT = 8500.0  # m


@dataclass(frozen=True)
class RocketConfig:
    """Fin-stabilised sounding rocket: one body tube, one nose cone, four fixed fins."""

    body_length: float = 1.20  # m, nose tip to tail
    body_diameter: float = 0.075  # m
    wall_thickness: float = 0.0015  # m, informational (mass is given directly)

    nose_length: float = 0.20  # m
    nose_shape: str = "ogive"

    fin_count: int = 4
    fin_root_chord: float = 0.16  # m
    fin_tip_chord: float = 0.06  # m
    fin_span: float = 0.09  # m, exposed span outside the body
    fin_sweep: float = 0.06  # m, tip leading edge behind root leading edge
    fin_thickness: float = 0.003  # m
    fin_trailing_edge_from_tail: float = 0.02  # m

    structural_mass: float = 0.615  # kg, everything except the motor
    cg_from_nose_tip: float = 0.62  # m
    cp_from_nose_tip: float = 0.82  # m

    drag_coefficient: float = 0.45  # subsonic, referenced to the body cross-section
    fin_normal_force_slope: float = 10.0  # per radian, referenced to the body area
    pitch_damping_coefficient: float = 6.0

    launch_tilt_deg: float = 4.0  # fixed launch-rail tilt; no wind is modelled

    @property
    def body_radius(self) -> float:
        return self.body_diameter / 2.0

    @property
    def reference_area(self) -> float:
        """Body cross-section, the reference area for drag and fin forces."""
        return math.pi * self.body_radius ** 2

    @property
    def static_margin(self) -> float:
        """Distance from centre of gravity to centre of pressure (positive = stable)."""
        return self.cp_from_nose_tip - self.cg_from_nose_tip

    @property
    def static_margin_calibers(self) -> float:
        return self.static_margin / self.body_diameter

    @property
    def center_of_gravity_from_nose_tip(self) -> float:
        return self.cg_from_nose_tip

    @property
    def tail_offset(self) -> float:
        """Distance from the centre of gravity down to the tail on the pad."""
        return self.body_length - self.cg_from_nose_tip

    @property
    def center_of_pressure_from_nose_tip(self) -> float:
        return self.cp_from_nose_tip

    @property
    def launch_tilt_radians(self) -> float:
        return math.radians(self.launch_tilt_deg)

    def transverse_inertia(self, motor_mass: float,
                           motor_offset: float = 0.518) -> float:
        """Pitch inertia about the centre of gravity, from the fixed geometry.

        Body tube and fin set are treated as thin shells at their measured offsets; the
        result is one scalar because the flight model assumes an axisymmetric body.
        """
        cg = self.cg_from_nose_tip
        tube_mass = 0.72 * self.structural_mass
        nose_mass = 0.18 * self.structural_mass
        fin_mass = 0.10 * self.structural_mass

        tube_center = cg - self.body_length / 2.0
        nose_center = cg - (self.nose_length / 2.0)
        fin_center = cg - (self.body_length - self.fin_trailing_edge_from_tail
                           - self.fin_root_chord / 2.0)

        tube = tube_mass * ((self.body_length ** 2) / 12.0 + self.body_radius ** 2 / 4.0)
        nose = nose_mass * ((self.nose_length ** 2) / 18.0 + (nose_center - cg) ** 2)
        fins = fin_mass * (fin_center - cg) ** 2
        # motor_offset is the fixed distance from the CG to the motor's centre.
        motor = motor_mass * motor_offset ** 2
        return tube + nose + fins + motor


@dataclass(frozen=True)
class MotorConfig:
    """Single fixed solid motor. Dimensions and masses never depend on performance."""

    diameter: float = 0.029  # m
    length: float = 0.10  # m
    casing_mass: float = 0.045  # kg, motor hardware after burnout
    propellant_mass: float = 0.030  # kg, always the same regardless of performance
    nozzle_diameter: float = 0.014  # m
    nozzle_length: float = 0.022  # m
    aft_end_from_tail: float = 0.012  # m, motor aft face inside the body tube

    @property
    def dry_mass(self) -> float:
        return self.casing_mass

    @property
    def initial_mass(self) -> float:
        return self.casing_mass + self.propellant_mass

    @property
    def diameter_mm(self) -> float:
        return self.diameter * 1000.0

    @property
    def length_mm(self) -> float:
        return self.length * 1000.0


ROCKET = RocketConfig()
MOTOR = MotorConfig()

#: Total launch mass with a full propellant grain, kg.
LAUNCH_MASS = ROCKET.structural_mass + MOTOR.initial_mass
#: Mass after burnout, kg.
DRY_MASS = ROCKET.structural_mass + MOTOR.dry_mass
#: Pitch inertia about the centre of gravity, kg m^2.
INERTIA = ROCKET.transverse_inertia(MOTOR.initial_mass)


def air_density(altitude_m: float) -> float:
    """Exponential atmosphere: dense enough for sounding rockets, cheap to integrate."""
    return AIR_DENSITY_SEA_LEVEL * math.exp(-max(altitude_m, 0.0) / ATMOSPHERE_SCALE_HEIGHT)


def speed_of_sound(altitude_m: float) -> float:
    """Approximate local speed of sound, used only to shape the drag rise near Mach 1."""
    temperature = max(216.65, 288.15 - 0.0065 * max(altitude_m, 0.0))
    return 20.05 * math.sqrt(temperature)
