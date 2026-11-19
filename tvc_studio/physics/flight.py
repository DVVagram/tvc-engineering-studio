"""Six-degree-of-freedom flight simulation.

State: world position, world velocity, quaternion attitude (body -> world), body angular
velocity. Forces come from the assembled rocket's own mass properties, the imported motor
curve, the aerodynamic model and the TVC deflection actually achieved by the servo plants.

The simulator reports states, margins and warnings; it does not decide whether a flight is
"successful".

Documented approximations:
* propellant is treated as a point mass at the motor's centre of mass while burning;
* structural flexibility and slosh are not modelled;
* the rail is an ideal straight constraint until rail departure;
* the inertial term from a changing inertia tensor is included with a first-order estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import SimulationError
from ..core.interfaces import Diagnostic, Provenance
from ..core.math3d import GRAVITY, MassProperties, Quat, Transform, as_vec3, parallel_axis, vec3
from ..electronics.flightcomputer import ControlGains, FlightPhase
from ..mech.servo import ServoPlant
from .aero import AeroConfig, AeroModel, StackElement, make_aero
from .atmosphere import Atmosphere
from .propulsion import MotorDefinition


@dataclass
class RocketProperties:
    """Physical description derived from the assembly (never hand-entered)."""

    name: str = "Rocket"
    mass_dry: float = 0.0
    com_dry: np.ndarray = field(default_factory=lambda: vec3())
    inertia_dry: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    motor_position: np.ndarray = field(default_factory=lambda: vec3())
    nozzle_position: np.ndarray = field(default_factory=lambda: vec3(0, 0, -0.02))
    propellant_position: np.ndarray | None = None
    tvc_position: np.ndarray = field(default_factory=lambda: vec3())
    imu_position: np.ndarray = field(default_factory=lambda: vec3())
    imu_orientation: Quat = field(default_factory=Quat.identity)
    reference_area: float = 0.0
    reference_length: float = 0.5
    max_diameter: float = 0.05
    cp_position: float = 0.0
    provenance: str = "cad_derived"
    notes: str = ""

    def mass_properties_at(self, propellant_mass: float) -> MassProperties:
        """Mass, COM and inertia with the remaining propellant as a point mass."""
        m_p = max(propellant_mass, 0.0)
        m = self.mass_dry + m_p
        if m <= 0:
            raise SimulationError("rocket has no mass: check the assembly and the motor definition")
        prop_pos = self.propellant_position if self.propellant_position is not None else self.motor_position
        com = (self.mass_dry * as_vec3(self.com_dry) + m_p * as_vec3(prop_pos)) / m
        inertia = parallel_axis(self.inertia_dry, self.mass_dry, as_vec3(self.com_dry) - com)
        if m_p > 0:
            inertia = inertia + m_p * ((lambda d: (d @ d) * np.eye(3) - np.outer(d, d))(
                as_vec3(prop_pos) - com))
        return MassProperties(mass=m, com=com, inertia=inertia, provenance=self.provenance, frame="body")

    def static_margin(self, propellant_mass: float) -> float:
        """Distance from CG to CP, positive when the CP is aft of the CG (stable).

        The body frame measures z from the tail, so a stable rocket has cp_position < com_z.
        """
        return float(self.mass_properties_at(propellant_mass).com[2]) - self.cp_position


@dataclass
class FlightConfig:
    dt: float = 0.001
    max_time: float = 120.0
    launch_elevation_deg: float = 90.0
    launch_azimuth_deg: float = 0.0
    rail_length: float = 1.0
    ground_altitude: float = 0.0
    tvc_enabled: bool = True
    max_gimbal: float = math.radians(8.0)
    servo_slew_limit: float = math.radians(600.0)  # rad/s, applied when servo dynamics are off
    parachute_area: float = 0.5
    parachute_cd: float = 1.5
    deploy_altitude: float = 150.0
    deploy_at_apogee: bool = True
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gust_std: float = 0.0
    record_interval: int = 10
    stop_at_landing: bool = True
    max_tilt_deg: float = 25.0
    launch_clearance_speed: float = 15.0

    def to_dict(self) -> dict:
        return {
            "dt": self.dt, "max_time": self.max_time,
            "launch_elevation_deg": self.launch_elevation_deg,
            "launch_azimuth_deg": self.launch_azimuth_deg, "rail_length": self.rail_length,
            "ground_altitude": self.ground_altitude, "tvc_enabled": self.tvc_enabled,
            "max_gimbal": self.max_gimbal, "servo_slew_limit": self.servo_slew_limit,
            "parachute_area": self.parachute_area, "parachute_cd": self.parachute_cd,
            "deploy_altitude": self.deploy_altitude, "deploy_at_apogee": self.deploy_at_apogee,
            "wind": list(self.wind), "gust_std": self.gust_std,
            "record_interval": self.record_interval, "stop_at_landing": self.stop_at_landing,
            "max_tilt_deg": self.max_tilt_deg, "launch_clearance_speed": self.launch_clearance_speed,
        }

    @staticmethod
    def from_dict(d: dict) -> "FlightConfig":
        allowed = FlightConfig.__dataclass_fields__
        kwargs = {}
        for k, v in d.items():
            if k not in allowed:
                continue
            kwargs[k] = tuple(v) if k == "wind" else v
        return FlightConfig(**kwargs)


@dataclass
class FlightState:
    time: float = 0.0
    position: np.ndarray = field(default_factory=lambda: vec3())
    velocity: np.ndarray = field(default_factory=lambda: vec3())
    attitude: Quat = field(default_factory=Quat.identity)
    omega: np.ndarray = field(default_factory=lambda: vec3())
    mass: float = 0.0
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    com: np.ndarray = field(default_factory=lambda: vec3())
    gimbal_pitch: float = 0.0
    gimbal_yaw: float = 0.0
    phase: FlightPhase = FlightPhase.PRELAUNCH
    thrust: float = 0.0
    alpha: float = 0.0
    mach: float = 0.0
    dynamic_pressure: float = 0.0
    altitude: float = 0.0
    speed: float = 0.0
    on_rail: bool = True
    parachute_deployed: bool = False

    def tilt_angle(self) -> float:
        body_axis_world = self.attitude.rotate(vec3(0, 0, 1))
        return math.acos(max(min(float(body_axis_world[2]), 1.0), -1.0))

    def to_dict(self) -> dict:
        yaw, pitch, roll = self.attitude.to_euler()
        return {
            "time": self.time,
            "x": float(self.position[0]), "y": float(self.position[1]), "z": float(self.position[2]),
            "vx": float(self.velocity[0]), "vy": float(self.velocity[1]), "vz": float(self.velocity[2]),
            "speed": self.speed,
            "altitude": self.altitude,
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "tilt_deg": math.degrees(self.tilt_angle()),
            "omega_x": float(self.omega[0]), "omega_y": float(self.omega[1]), "omega_z": float(self.omega[2]),
            "mass": self.mass, "thrust": self.thrust,
            "alpha": self.alpha, "mach": self.mach, "q": self.dynamic_pressure,
            "gimbal_pitch": self.gimbal_pitch, "gimbal_yaw": self.gimbal_yaw,
            "phase": self.phase.value, "parachute": self.parachute_deployed, "on_rail": self.on_rail,
        }


@dataclass
class FlightResult:
    telemetry: list[dict] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    motor: MotorDefinition | None = None

    def plot_series(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if not self.telemetry:
            return np.array([]), np.array([])
        t = np.array([row["time"] for row in self.telemetry])
        y = np.array([row.get(key, np.nan) for row in self.telemetry], dtype=float)
        return t, y


class FlightSimulator:
    """6-DOF plant, also usable as the SITL plant."""

    def __init__(self, rocket: RocketProperties, motor: MotorDefinition, aero_config: AeroConfig,
                 config: FlightConfig | None = None, stack: list[StackElement] | None = None,
                 pitch_servo: ServoPlant | None = None, yaw_servo: ServoPlant | None = None,
                 aero_model: AeroModel | None = None, control: ControlGains | None = None) -> None:
        self.rocket = rocket
        self.motor = motor
        self.config = config or FlightConfig()
        self.aero_config = aero_config
        self.aero = aero_model or make_aero(aero_config, stack)
        # The canopy pulls on the rocket at the forward end of the stack, not at the centre
        # of mass; that offset is what turns the body to hang under the chute.
        forward = max((e.start + e.length for e in (stack or [])),
                      default=float(rocket.reference_length))
        self.chute_attachment = vec3(0.0, 0.0, forward)
        self.atmosphere = Atmosphere(self.config.wind, self.config.gust_std)
        self.pitch_servo = pitch_servo
        self.yaw_servo = yaw_servo
        self.control = control
        self.state = FlightState()
        self.telemetry: list[dict] = []
        self.diagnostics: list[Diagnostic] = []
        self._cmd_pitch = 0.0
        self._cmd_yaw = 0.0
        self._previous_omega = vec3()
        self._previous_inertia: np.ndarray | None = None
        self._warned: set[str] = set()
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        cfg = self.config
        elevation = math.radians(cfg.launch_elevation_deg)
        azimuth = math.radians(cfg.launch_azimuth_deg)
        self.rail_direction = np.array([
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ])
        # body +Z along the rail, with a slightly pitched attitude for non-vertical launches
        self.rail_attitude = Quat.from_two_vectors(vec3(0, 0, 1), self.rail_direction)
        mp = self.rocket.mass_properties_at(self.motor.propellant_mass)
        self.state = FlightState(
            time=0.0,
            position=vec3(0, 0, cfg.ground_altitude),
            velocity=vec3(),
            attitude=self.rail_attitude,
            omega=vec3(),
            mass=mp.mass,
            inertia=mp.inertia,
            com=mp.com,
            phase=FlightPhase.PRELAUNCH,
            altitude=cfg.ground_altitude,
            on_rail=True,
        )
        self.telemetry.clear()
        self.diagnostics.clear()
        self._warned.clear()
        self._cmd_pitch = 0.0
        self._cmd_yaw = 0.0
        self._previous_omega = vec3()
        self._previous_inertia = mp.inertia.copy()
        self._rail_travel = 0.0
        if self.pitch_servo:
            self.pitch_servo.reset(0.0)
        if self.yaw_servo:
            self.yaw_servo.reset(0.0)
        for warning in self.aero_config.warnings():
            self.diagnostics.append(Diagnostic("warning", "aero", warning))
        for problem in self.motor.validate():
            self.diagnostics.append(Diagnostic("warning", "motor", problem))

    # ------------------------------------------------------------ interfaces
    def set_tvc(self, pitch: float, yaw: float) -> None:
        limit = self.config.max_gimbal
        self._cmd_pitch = float(np.clip(pitch, -limit, limit))
        self._cmd_yaw = float(np.clip(yaw, -limit, limit))
        if self.pitch_servo:
            self.pitch_servo.command(self._cmd_pitch)
        if self.yaw_servo:
            self.yaw_servo.command(self._cmd_yaw)

    @property
    def phase(self) -> FlightPhase:
        return self.state.phase

    def state_for_imu(self):
        """Return (body transform, world angular velocity, world linear acceleration)."""
        omega_world = self.state.attitude.rotate(self.state.omega)
        acceleration = self._last_acceleration if hasattr(self, "_last_acceleration") else vec3()
        transform = Transform(self.state.attitude, self.state.position)
        return transform, omega_world, acceleration

    def state_dict(self) -> dict:
        return self.state.to_dict()

    # -------------------------------------------------------------- dynamics
    def _mass_properties(self, t: float) -> MassProperties:
        remaining = self.motor.propellant_mass * self.motor.propellant_fraction(t)
        return self.rocket.mass_properties_at(remaining)

    def _autopilot_step(self) -> None:
        """Tilt/rate TVC control law, run at the plant rate.

        Uses true attitude and rates (an ideal sensor); the full sensor/estimator path is
        exercised by the SITL and HIL interfaces, not here. The sign convention follows the
        thrust vector geometry: positive pitch deflection produces a negative body-X moment.
        """
        gains = self.control
        state = self.state
        if gains is None:
            return
        if state.phase in (FlightPhase.PRELAUNCH, FlightPhase.LANDED) or state.on_rail:
            self.set_tvc(0.0, 0.0)
            return
        q_err = state.attitude.inverse() * self.rail_attitude
        v = np.array([q_err.x, q_err.y, q_err.z])
        w = q_err.w
        if w < 0.0:
            v, w = -v, -w
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            ex = ey = 0.0
        else:
            angle = 2.0 * math.atan2(norm, w)
            axis = v / norm
            ex, ey = float(axis[0]) * angle, float(axis[1]) * angle
        wx, wy = float(state.omega[0]), float(state.omega[1])
        # gimbal geometry: +pitch deflects thrust toward -Y and produces a -X moment. Tilting
        # back therefore needs a negative command, while rate damping needs a positive one.
        pitch = -gains.tilt_kp * ex + gains.rate_damping * wx
        yaw = -gains.tilt_kp * ey + gains.rate_damping * wy
        self.set_tvc(pitch, yaw)

    def _gimbal_angles(self, dt: float) -> tuple[float, float]:
        if not self.config.tvc_enabled:
            return 0.0, 0.0
        pitch = self._cmd_pitch
        yaw = self._cmd_yaw
        # The horn must hold the thrust-side load of the current deflection: a 1 mm nominal
        # offset between the thrust axis and the servo axis gives F * e * sin(deflection).
        # With zero deflection the load is zero, so a centred servo is not pushed off zero.
        thrust = abs(self.state.thrust)
        if self.pitch_servo is not None:
            load = thrust * 0.001 * math.sin(self.state.gimbal_pitch)
            pitch = self.pitch_servo.step(dt, load).output_angle
        if self.yaw_servo is not None:
            load = thrust * 0.001 * math.sin(self.state.gimbal_yaw)
            yaw = self.yaw_servo.step(dt, load).output_angle
        limit = self.config.max_gimbal
        return float(np.clip(pitch, -limit, limit)), float(np.clip(yaw, -limit, limit))

    def _forces(self, t: float, position, velocity, attitude: Quat, omega_body, gimbal, mass_props):
        atm = self.atmosphere.state(float(position[2]))
        wind = self.atmosphere.wind_at(float(position[2]), self.config.dt)
        rel_velocity = as_vec3(velocity) - wind
        density = atm.density
        aero = self.aero.forces(rel_velocity, attitude, float(position[2]), density,
                                atm.speed_of_sound, float(mass_props.com[2]), omega_body)
        force_body = aero.force_body.copy()
        moment_body = aero.moment_body.copy()

        thrust = self.motor.thrust_at(t)
        if thrust > 0:
            # gimbal deflection: pitch about body X, yaw about body Y
            cp, sp = math.cos(gimbal[0]), math.sin(gimbal[0])
            cy, sy = math.cos(gimbal[1]), math.sin(gimbal[1])
            thrust_dir_body = np.array([sy * cp, -sp, cp * cy])
            force_body = force_body + thrust_dir_body * thrust
            r_nozzle = as_vec3(self.rocket.nozzle_position) - as_vec3(mass_props.com)
            moment_body = moment_body + np.cross(r_nozzle, thrust_dir_body * thrust)
            # thrust misalignment from the CAD placement (documented, small)
            offset = as_vec3(self.rocket.motor_position) - as_vec3(self.rocket.nozzle_position)
            if np.linalg.norm(offset) > 1e-9:
                pass

        force_world = attitude.rotate(force_body)
        # Euler's equation below is written in the body frame, so the moment stays in the
        # body frame (it is *not* rotated with the attitude).
        moment = moment_body

        if self.state.parachute_deployed and float(velocity[2]) < 0:
            # Quasi-steady canopy drag at the riser attachment: the attachment point moves
            # with the body rate, so the same term supplies the restoring moment that makes
            # the vehicle hang under the canopy and the damping that settles the swing.
            r_chute_body = as_vec3(self.chute_attachment) - as_vec3(mass_props.com)
            r_chute_world = attitude.rotate(r_chute_body)
            omega_world = attitude.rotate(as_vec3(omega_body))
            v_site = as_vec3(rel_velocity) + np.cross(omega_world, r_chute_world)
            speed = float(np.linalg.norm(v_site))
            if speed > 1e-6:
                drag = 0.5 * density * speed * speed * self.config.parachute_cd * self.config.parachute_area
                force_chute = -(v_site / speed) * drag
                force_world = force_world + force_chute
                moment = moment + attitude.rotate_inverse(
                    np.cross(r_chute_world, force_chute))

        force_world = force_world + vec3(0, 0, -GRAVITY * mass_props.mass)
        return force_world, moment, aero, atm, thrust

    def _derivative(self, t, position, velocity, attitude, omega_body, mass_props, force_world, moment_body):
        mass = max(mass_props.mass, 1e-9)
        acceleration = force_world / mass
        I = np.asarray(mass_props.inertia, dtype=float)
        if self._previous_inertia is not None and self.config.dt > 0:
            I_dot = (I - self._previous_inertia) / self.config.dt
        else:
            I_dot = np.zeros((3, 3))
        try:
            I_inv = np.linalg.inv(I + np.eye(3) * 1e-12)
        except np.linalg.LinAlgError:
            raise SimulationError("inertia tensor is singular; check assembly mass properties")
        omega = as_vec3(omega_body)
        angular_acc = I_inv @ (as_vec3(moment_body) - np.cross(omega, I @ omega) - I_dot @ omega)
        dq = attitude.derivative(omega)
        return acceleration, angular_acc, dq

    def step(self, dt: float | None = None) -> FlightState:
        dt = float(dt if dt is not None else self.config.dt)
        state = self.state
        t = state.time
        if self.control is not None:
            self._autopilot_step()
        gimbal = self._gimbal_angles(dt)
        state.gimbal_pitch, state.gimbal_yaw = gimbal
        mp = self._mass_properties(t)
        force, moment, aero, atm, thrust = self._forces(t, state.position, state.velocity,
                                                        state.attitude, state.omega, gimbal, mp)

        # RK4 on translation and rotation
        pos0, vel0, q0, w0 = state.position.copy(), state.velocity.copy(), state.attitude, state.omega.copy()

        def eval_stage(pos, vel, q, w):
            a, alpha, dq = self._derivative(t, pos, vel, q, w, mp, force, moment)
            return a, alpha, dq

        a1, al1, dq1 = eval_stage(pos0, vel0, q0, w0)
        a2, al2, dq2 = eval_stage(pos0 + 0.5 * dt * vel0, vel0 + 0.5 * dt * a1,
                                  _advance(q0, dq1, 0.5 * dt), w0 + 0.5 * dt * al1)
        a3, al3, dq3 = eval_stage(pos0 + 0.5 * dt * (vel0 + 0.5 * dt * a1),
                                  vel0 + 0.5 * dt * a2,
                                  _advance(q0, dq2, 0.5 * dt), w0 + 0.5 * dt * al2)
        a4, al4, dq4 = eval_stage(pos0 + dt * (vel0 + 0.5 * dt * a2),
                                  vel0 + dt * a3,
                                  _advance(q0, dq3, dt), w0 + dt * al3)

        new_vel = vel0 + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        new_pos = pos0 + dt / 6.0 * (vel0 + 2 * (vel0 + 0.5 * dt * a1) + 2 * (vel0 + 0.5 * dt * a2) + (vel0 + dt * a3))
        new_w = w0 + dt / 6.0 * (al1 + 2 * al2 + 2 * al3 + al4)
        new_q = _advance(q0, dq1, dt / 6.0)
        new_q = _advance(new_q, dq2, dt / 3.0)
        new_q = _advance(new_q, dq3, dt / 3.0)
        new_q = _advance(new_q, dq4, dt / 6.0)

        # rail constraint
        on_rail = state.on_rail and state.phase in (FlightPhase.PRELAUNCH, FlightPhase.POWERED)
        if on_rail:
            along = float(np.dot(new_pos - vec3(0, 0, self.config.ground_altitude), self.rail_direction))
            lateral = (new_pos - vec3(0, 0, self.config.ground_altitude)) - along * self.rail_direction
            new_pos = new_pos - lateral
            v_along = float(np.dot(new_vel, self.rail_direction))
            new_vel = self.rail_direction * v_along
            # attitude locked to rail while sliding
            new_q = self.rail_attitude
            new_w = vec3()
            self._rail_travel = max(along, 0.0)
            if along >= self.config.rail_length:
                state.on_rail = False
                on_rail = False
                if v_along < self.config.launch_clearance_speed:
                    self._warn("rail", f"Rail departure speed is only {v_along:.1f} m/s: "
                                       "marginal aerodynamic control authority right after the rail.")
        else:
            state.on_rail = False

        # ground contact
        if new_pos[2] < self.config.ground_altitude:
            new_pos[2] = self.config.ground_altitude
            new_vel = vec3(new_vel[0] * 0.2, new_vel[1] * 0.2, 0.0)
            new_w = new_w * 0.2

        self._last_acceleration = (new_vel - vel0) / dt
        self._previous_omega = state.omega
        state.position = new_pos
        state.velocity = new_vel
        state.attitude = new_q.normalized()
        state.omega = new_w
        state.time = t + dt
        state.mass = mp.mass
        state.inertia = mp.inertia
        state.com = mp.com
        state.thrust = thrust
        state.alpha = aero.alpha
        state.mach = aero.mach
        state.dynamic_pressure = aero.dynamic_pressure
        state.altitude = float(new_pos[2])
        state.speed = float(np.linalg.norm(new_vel))
        self._previous_inertia = mp.inertia.copy()

        for warning in aero.warnings:
            key = warning[:40]
            if key not in self._warned:
                self._warned.add(key)
                self.diagnostics.append(Diagnostic("warning", "aero", warning))

        self._update_phase(dt, thrust, mp)

        if self.config.record_interval and int(round(state.time / dt)) % self.config.record_interval == 0:
            row = state.to_dict()
            row["stability_margin"] = self.rocket.cp_position - float(mp.com[2])
            self.telemetry.append(row)
        return state

    def _warn(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.diagnostics.append(Diagnostic("warning", key, message))

    def _update_phase(self, dt: float, thrust: float, mp: MassProperties) -> None:
        state = self.state
        cfg = self.config
        weight = mp.mass * GRAVITY
        if state.phase is FlightPhase.PRELAUNCH:
            if thrust > weight * 1.02 and self._rail_travel > 0.001:
                state.phase = FlightPhase.POWERED
                if self._is_estimated_motor():
                    self._warn("demo-motor",
                               "Flight uses the synthetic DEMO thrust curve; results are illustrative only.")
        elif state.phase is FlightPhase.POWERED:
            if thrust <= 0.0 or state.time > self.motor.burn_time:
                state.phase = FlightPhase.COAST
        elif state.phase is FlightPhase.COAST:
            if state.velocity[2] <= 0.0:
                state.phase = FlightPhase.DESCENT
                if cfg.deploy_at_apogee:
                    state.parachute_deployed = True
        elif state.phase is FlightPhase.DESCENT:
            if (not cfg.deploy_at_apogee and state.altitude <= cfg.ground_altitude + cfg.deploy_altitude):
                state.parachute_deployed = True
            if state.altitude <= cfg.ground_altitude + 1e-3 and state.speed < 0.5:
                state.phase = FlightPhase.LANDED
        tilt = math.degrees(state.tilt_angle())
        if tilt > cfg.max_tilt_deg and state.phase in (FlightPhase.POWERED, FlightPhase.COAST):
            self._warn("tilt", f"Vehicle tilt reached {tilt:.1f}° (limit {cfg.max_tilt_deg:.1f}°): "
                               "check TVC authority and static margin.")

    def _is_estimated_motor(self) -> bool:
        return self.motor.provenance is Provenance.ESTIMATED

    # ------------------------------------------------------------------- run
    def run(self, duration: float | None = None, stop_when=None) -> FlightResult:
        dt = self.config.dt
        max_time = duration if duration is not None else self.config.max_time
        steps = max(int(max_time / dt), 1)
        for _ in range(steps):
            self.step(dt)
            if self.state.phase is FlightPhase.LANDED and self.config.stop_at_landing:
                break
            if stop_when is not None and stop_when(self.state):
                break
        return self.result()

    def result(self) -> FlightResult:
        summary = self.summary()
        return FlightResult(telemetry=list(self.telemetry), diagnostics=list(self.diagnostics),
                            summary=summary, motor=self.motor)

    def summary(self) -> dict:
        rows = self.telemetry
        if not rows:
            return {"status": "no data"}
        altitudes = [r["altitude"] for r in rows]
        speeds = [r["speed"] for r in rows]
        qs = [r["q"] for r in rows]
        tilts = [r["tilt_deg"] for r in rows]
        apogee_index = int(np.argmax(altitudes))
        return {
            "apogee_m": float(max(altitudes)),
            "apogee_time_s": rows[apogee_index]["time"],
            "max_speed_ms": float(max(speeds)),
            "max_mach": float(max(r["mach"] for r in rows)),
            "max_dynamic_pressure_pa": float(max(qs)),
            "max_tilt_deg": float(max(tilts)),
            "burn_time_s": self.motor.burn_time,
            "total_impulse_ns": self.motor.total_impulse,
            "launch_mass_kg": self.motor.launch_mass + self.rocket.mass_dry,
            "dry_mass_kg": self.rocket.mass_dry,
            "static_margin_m": float(self.state.com[2]) - self.rocket.cp_position,
            "flight_time_s": rows[-1]["time"],
            "landed": self.state.phase is FlightPhase.LANDED,
            "motor_provenance": self.motor.provenance.value,
            "warnings": [d.message for d in self.diagnostics if d.severity == "warning"],
        }

    def telemetry_row(self) -> dict:
        row = self.state.to_dict()
        row["stability_margin"] = self.rocket.cp_position - float(self.state.com[2])
        return row


def _advance(q: Quat, dq: Quat, dt: float) -> Quat:
    return Quat(q.w + dq.w * dt, q.x + dq.x * dt, q.y + dq.y * dt, q.z + dq.z * dt).normalized()


def ballistic_altitude(v0: float, t: float, g: float = GRAVITY) -> float:
    """Closed form used by the gravity-only validation tests."""
    return v0 * t - 0.5 * g * t * t
