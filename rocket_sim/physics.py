"""Rigid-body flight model for the fixed rocket.

The rocket is one rigid body with a fixed motor. Forces are thrust along the body axis,
gravity, drag opposite the air-relative velocity and a fin normal force applied at the
centre of pressure (which is what makes the rocket weathercock into its velocity vector).
There is no control system of any kind: the attitude is whatever those forces produce.

Integration is classical RK4 at a fixed 2 ms step. The rendered motion is never a canned
path — every frame advances this integrator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import (
    DRY_MASS,
    G0,
    INERTIA,
    LAUNCH_MASS,
    MOTOR,
    ROCKET,
    RocketConfig,
    MotorConfig,
    air_density,
    speed_of_sound,
)
from .motor import MotorPerformance, ThrustCurve

FIXED_DT = 0.002
MAX_FLIGHT_TIME = 300.0
TRAIL_INTERVAL = 0.005


# ---------------------------------------------------------------------- quaternions
def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.asarray(q, dtype=float) / norm


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix taking body coordinates into world coordinates."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_two_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-12:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, *axis])
    axis = np.cross(a, b)
    angle = math.acos(dot)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.array([math.cos(half), *(axis * math.sin(half))])


# --------------------------------------------------------------------------- state
@dataclass
class FlightState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    time: float = 0.0
    mass: float = LAUNCH_MASS
    thrust: float = 0.0
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "FlightState":
        return FlightState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            attitude=self.attitude.copy(),
            angular_velocity=self.angular_velocity.copy(),
            time=self.time,
            mass=self.mass,
            thrust=self.thrust,
            acceleration=self.acceleration.copy(),
        )

    @property
    def altitude(self) -> float:
        """Centre-of-gravity height; the interface reports tail height above the pad."""
        return float(self.position[2])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def vertical_speed(self) -> float:
        return float(self.velocity[2])

    @property
    def body_axis(self) -> np.ndarray:
        return quat_matrix(self.attitude)[:, 0]


@dataclass(frozen=True)
class FlightEvent:
    time: float
    name: str
    detail: str = ""


def _mach_drag_factor(mach: float) -> float:
    """Simple transonic drag rise; the airframe is subsonic on these motors."""
    if mach < 0.8:
        return 1.0
    if mach < 1.05:
        return 1.0 + 0.8 * (mach - 0.8) / 0.25
    if mach < 1.5:
        return 1.8 - 0.8 * (mach - 1.05) / 0.45
    return 1.0


class Simulation:
    """Fixed-timestep flight simulation of one rocket and one motor performance spec."""

    def __init__(
        self,
        performance: MotorPerformance | None = None,
        rocket: RocketConfig = ROCKET,
        motor: MotorConfig = MOTOR,
        dt: float = FIXED_DT,
    ) -> None:
        self.rocket = rocket
        self.motor = motor
        self.performance = (performance or MotorPerformance()).validate()
        self.curve: ThrustCurve = self.performance.curve()
        self.dt = dt
        self.state = FlightState()
        self.events: list[FlightEvent] = []
        self.trail: list[np.ndarray] = []
        self.max_altitude = 0.0
        self.max_speed = 0.0
        self.finished = False
        self.started = False
        self.on_pad = True
        self.status = "Ready"
        self._burnout_logged = False
        self._apogee_logged = False
        self._last_trail_time = -1.0
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        tilt = self.rocket.launch_tilt_radians
        axis = np.array([math.sin(tilt), 0.0, math.cos(tilt)])
        self.state = FlightState(
            position=np.array([0.0, 0.0, self.rocket.tail_offset]),
            attitude=quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), axis),
            mass=LAUNCH_MASS,
            thrust=self.curve.thrust_at(0.0),
        )
        self.events = []
        self.trail = [self.state.position.copy()]
        self.max_altitude = 0.0
        self.max_speed = 0.0
        self.finished = False
        self.started = False
        self.on_pad = True
        self.status = "Ready"
        self._burnout_logged = False
        self._apogee_logged = False
        self._last_trail_time = -1.0

    def start(self) -> None:
        self.started = True
        if self.status == "Ready":
            self.status = "On pad"

    def advance(self, duration: float) -> None:
        """Advance the simulation by ``duration`` seconds of flight time."""
        if self.finished:
            return
        self.start()
        target = self.state.time + max(0.0, float(duration))
        while not self.finished and self.state.time < target:
            self._step()
            if self.state.time >= MAX_FLIGHT_TIME:
                self.finished = True
                self.status = "Timed out"
                self._log(self.state.time, "Simulation stopped",
                          "Maximum simulated flight time reached")
        if self.on_pad and self.started:
            self.status = "On pad" if not self.finished else self.status

    def run_to_end(self, limit: float = MAX_FLIGHT_TIME) -> "Simulation":
        while not self.finished and self.state.time < limit:
            self._step()
        return self

    # --------------------------------------------------------------------- physics
    def _mass_at(self, time: float) -> float:
        """Propellant is consumed in proportion to the impulse actually delivered.

        A progressive curve therefore burns mass more slowly at ignition than a
        regressive one with the same total impulse, which is what the motor really does.
        """
        total = self.curve.total_impulse
        if total <= 0.0:
            return DRY_MASS
        burned = min(total, max(0.0, self.curve.impulse_delivered(time)))
        remaining = 1.0 - burned / total
        return (self.rocket.structural_mass + self.motor.casing_mass
                + self.motor.propellant_mass * remaining)

    def _derivatives(self, y: np.ndarray, time: float) -> np.ndarray:
        position = y[0:3]
        velocity = y[3:6]
        attitude = quat_normalize(y[6:10])
        omega = y[10:13]

        rotation = quat_matrix(attitude)
        axis = rotation[:, 0]
        mass = max(self._mass_at(time), 1e-9)

        speed = float(np.linalg.norm(velocity))
        # Atmosphere referenced to the launch site (the pad is at 0 m).
        altitude = float(position[2]) - self.rocket.tail_offset
        rho = air_density(altitude)
        dynamic_pressure = 0.5 * rho * speed * speed

        force = axis * self.curve.thrust_at(time) + np.array([0.0, 0.0, -mass * G0])
        torque = np.zeros(3)

        if speed > 1e-9:
            mach = speed / speed_of_sound(altitude)
            drag_coefficient = self.rocket.drag_coefficient * _mach_drag_factor(mach)
            drag = dynamic_pressure * drag_coefficient * self.rocket.reference_area
            force -= drag * (velocity / speed)

            lateral = velocity - float(np.dot(velocity, axis)) * axis
            lateral_speed = float(np.linalg.norm(lateral))
            if lateral_speed > 1e-9:
                # Slender-body slope at small angles, blending into flat-plate crossflow
                # at large angles of attack (the apogee pitch-over reaches both).
                sin_alpha = min(1.0, lateral_speed / max(speed, 1e-12))
                cos_alpha = math.sqrt(max(0.0, 1.0 - sin_alpha * sin_alpha))
                normal_coefficient = (
                    self.rocket.fin_normal_force_slope * sin_alpha * cos_alpha
                    + self.rocket.fin_crossflow_drag * sin_alpha * sin_alpha
                )
                normal = (dynamic_pressure
                          * self.rocket.reference_area
                          * normal_coefficient
                          * (-lateral / lateral_speed))
                force += normal
                lever_arm = -self.rocket.static_margin * axis  # centre of pressure
                torque += np.cross(lever_arm, normal)

                omega_perp = omega - float(np.dot(omega, axis)) * axis
                damping = (self.rocket.pitch_damping_coefficient
                           * dynamic_pressure
                           * self.rocket.reference_area
                           * self.rocket.static_margin ** 2)
                torque -= damping * omega_perp

        acceleration = force / mass
        # The pitch inertia falls as the propellant leaves the motor.
        inertia = max(self.rocket.transverse_inertia(mass - self.rocket.structural_mass),
                      0.01)
        angular_acceleration = torque / inertia
        attitude_rate = 0.5 * quat_mul(np.array([0.0, *omega]), attitude)
        return np.concatenate([velocity, acceleration, attitude_rate, angular_acceleration])

    def _step(self) -> None:
        if self.finished:
            return
        time = self.state.time
        thrust_now = self.curve.thrust_at(time)
        self.state.thrust = thrust_now

        if self.on_pad:
            vertical = thrust_now * math.cos(self.rocket.launch_tilt_radians)
            weight = self.state.mass * G0
            if vertical > weight:
                self.on_pad = False
                self.status = "Powered ascent"
                self._log(time, "Liftoff", f"{thrust_now:.1f} N thrust exceeds "
                                            f"{weight:.1f} N weight")
            else:
                if time >= self.curve.burn_time - 1e-9:
                    self.finished = True
                    self.status = "No liftoff"
                    self._log(time, "No liftoff",
                              f"Thrust never exceeded weight "
                              f"({self.performance.display_name})")
                self.state.time = time + self.dt
                self.state.mass = self._mass_at(self.state.time)
                return

        y = np.concatenate([
            self.state.position, self.state.velocity, self.state.attitude,
            self.state.angular_velocity,
        ])
        dt = self.dt
        k1 = self._derivatives(y, time)
        k2 = self._derivatives(y + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = self._derivatives(y + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = self._derivatives(y + dt * k3, time + dt)
        y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.all(np.isfinite(y_next)):
            self.finished = True
            self.status = "Integration failed"
            self._log(time, "Simulation stopped", "Non-finite state")
            return

        previous_vertical_speed = self.state.vertical_speed
        new_time = time + dt
        self.state = FlightState(
            position=y_next[0:3],
            velocity=y_next[3:6],
            attitude=quat_normalize(y_next[6:10]),
            angular_velocity=y_next[10:13],
            time=new_time,
            mass=self._mass_at(new_time),
            thrust=self.curve.thrust_at(new_time),
            acceleration=k1[3:6],
        )

        self.max_altitude = max(self.max_altitude, self.altitude)
        self.max_speed = max(self.max_speed, self.state.speed)

        if (new_time - self._last_trail_time) >= TRAIL_INTERVAL:
            self.trail.append(self.state.position.copy())
            self._last_trail_time = new_time
            if len(self.trail) > 6000:
                del self.trail[1]

        if not self._burnout_logged and new_time >= self.curve.burn_time:
            self._burnout_logged = True
            self._log(self.curve.burn_time, "Motor burnout",
                      f"{self.curve.total_impulse:.1f} N s burned")

        if self._burnout_logged and not self._apogee_logged and \
                previous_vertical_speed > 0.0 >= self.state.vertical_speed:
            self._apogee_logged = True
            self._log(new_time, "Apogee", f"{self.altitude:.1f} m")

        if self.altitude <= 0.0 and self.state.vertical_speed < 0.0:
            impact_speed = self.state.speed
            self.state.position[2] -= self.altitude  # rest the lowest end on the ground
            self.state.velocity[:] = 0.0
            self.state.angular_velocity[:] = 0.0
            self.finished = True
            self.status = "Landed"
            self._log(new_time, "Ground impact", f"{impact_speed:.1f} m/s")
            return

        if not self._burnout_logged:
            self.status = "Powered ascent"
        elif previous_vertical_speed > 0.0 or self.state.vertical_speed > 0.0:
            self.status = "Coast"
        else:
            self.status = "Descent"

    def _log(self, time: float, name: str, detail: str = "") -> None:
        self.events.append(FlightEvent(round(float(time), 3), name, detail))

    # ------------------------------------------------------------------- readouts
    @property
    def altitude(self) -> float:
        """Height of the lowest end above the pad, so a landed rocket reads 0 m."""
        axis_z = float(self.state.body_axis[2])
        tail = self.state.position[2] - self.rocket.tail_offset * axis_z
        nose = self.state.position[2] + self.rocket.cg_from_nose_tip * axis_z
        return min(tail, nose)

    @property
    def vertical_speed(self) -> float:
        return self.state.vertical_speed

    @property
    def speed(self) -> float:
        return self.state.speed

    @property
    def acceleration(self) -> float:
        return float(np.linalg.norm(self.state.acceleration))

    @property
    def time(self) -> float:
        return self.state.time

    @property
    def thrust(self) -> float:
        return self.state.thrust

    @property
    def motor_status(self) -> str:
        if not self.started:
            return "Idle"
        if self.on_pad:
            return "Ignition check"
        if self.curve.burn_time <= 0:
            return "No burn"
        if self.state.time < self.curve.burn_time:
            percent = 100.0 * self.state.time / self.curve.burn_time
            return f"Burning {percent:.0f} %"
        return "Burnout"

    @property
    def summary(self) -> dict[str, float]:
        return {
            "apogee_m": self.max_altitude,
            "max_speed_ms": self.max_speed,
            "burnout_s": self.curve.burn_time,
            "impact_s": self.state.time if self.status == "Landed" else 0.0,
        }
