"""Flight computer simulation.

Models the parts of the avionics that change flight behaviour: loop timing, sensor
sampling, estimation, control law, servo commands, power draw and logging. It reads only
*simulated sensor samples*, never the plant's true state, so a control problem shows up the
way it would on hardware.

The controller implemented here is a tilt/rate TVC controller with a complementary attitude
filter. It is deliberately simple and fully parameterised; project-specific firmware is
expected to replace it through the SITL/HIL interfaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..components.hardware import BoardSpec, ServoSpec
from ..core.interfaces import Diagnostic
from ..core.math3d import Quat, as_vec3, vec3
from ..mech.servo import ServoPlant
from .imu import IMUSample, IMUModel


class FlightPhase(str, Enum):
    PRELAUNCH = "prelaunch"
    POWERED = "powered"
    COAST = "coast"
    DESCENT = "descent"
    LANDED = "landed"
    ABORT = "abort"


@dataclass
class PID:
    kp: float = 2.0
    ki: float = 0.0
    kd: float = 0.05
    integral_limit: float = 0.5
    output_limit: float = 1.0
    integral: float = 0.0
    previous_error: float = 0.0
    last_output: float = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_output = 0.0

    def update(self, error: float, dt: float) -> float:
        dt = max(dt, 1e-9)
        self.integral = float(np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit))
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_output = float(np.clip(out, -self.output_limit, self.output_limit))
        return self.last_output


@dataclass
class ControlGains:
    """TVC attitude controller gains and limits (editable in the Electronics workspace)."""

    tilt_kp: float = 2.4
    tilt_ki: float = 0.25
    tilt_kd: float = 0.10
    rate_damping: float = 0.25
    max_gimbal: float = math.radians(8.0)
    servo_channels: tuple[str, ...] = ("pitch", "yaw")
    servo_signs: tuple[float, float] = (-1.0, 1.0)
    servo_trim: tuple[float, float] = (0.0, 0.0)
    attitude_filter_gain: float = 0.4
    guidance_mode: str = "hold_vertical"   # hold_vertical | pitch_program | manual
    pitch_program_rate: float = 0.0        # rad/s of commanded tilt after liftoff
    deploy_altitude: float = 200.0         # m, recovery trigger (informational for FC)
    max_tilt_command: float = math.radians(10.0)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.max_gimbal <= 0:
            problems.append("maximum gimbal deflection must be positive")
        if self.tilt_kp < 0 or self.rate_damping < 0:
            problems.append("gains cannot be negative")
        return problems


@dataclass
class ServoChannel:
    name: str
    plant: ServoPlant
    direction: float = 1.0
    trim: float = 0.0
    min_angle: float = -math.pi / 2
    max_angle: float = math.pi / 2

    def command(self, angle: float) -> None:
        self.plant.command(float(np.clip(angle, self.min_angle, self.max_angle)))


@dataclass
class ControlFrame:
    time: float
    phase: str
    estimated_roll: float = 0.0
    estimated_pitch: float = 0.0
    estimated_yaw: float = 0.0
    tilt_error_x: float = 0.0
    tilt_error_y: float = 0.0
    gimbal_pitch: float = 0.0
    gimbal_yaw: float = 0.0
    servo_angles: dict[str, float] = field(default_factory=dict)
    servo_current: dict[str, float] = field(default_factory=dict)
    loop_time_us: float = 0.0
    missed_updates: int = 0
    supply_voltage: float = 5.0
    current_draw: float = 0.0
    accel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "time": self.time, "phase": self.phase,
            "roll": self.estimated_roll, "pitch": self.estimated_pitch, "yaw": self.estimated_yaw,
            "gimbal_pitch": self.gimbal_pitch, "gimbal_yaw": self.gimbal_yaw,
            "servo_angles": dict(self.servo_angles), "servo_current": dict(self.servo_current),
            "loop_time_us": self.loop_time_us, "missed_updates": self.missed_updates,
            "current_draw": self.current_draw,
        }


class AttitudeEstimator:
    """Complementary filter: gyro integration corrected by the accelerometer's up vector."""

    def __init__(self, gain: float = 0.4) -> None:
        self.quaternion = Quat.identity()
        self.gain = gain
        self.initialised = False

    def reset(self) -> None:
        self.quaternion = Quat.identity()
        self.initialised = False

    def update(self, sample: IMUSample, dt: float) -> Quat:
        accel = as_vec3(sample.accel)
        norm = float(np.linalg.norm(accel))
        # integrate gyro
        dq = self.quaternion.derivative(sample.gyro)
        q = Quat(self.quaternion.w + dq.w * dt, self.quaternion.x + dq.x * dt,
                 self.quaternion.y + dq.y * dt, self.quaternion.z + dq.z * dt).normalized()
        if norm > 1e-3:
            measured_up = accel / norm
            if not self.initialised:
                z = vec3(0, 0, 1)
                c = np.cross(z, measured_up)
                dot = float(np.clip(np.dot(z, measured_up), -1.0, 1.0))
                if np.linalg.norm(c) < 1e-9:
                    q = Quat.identity() if dot > 0 else Quat.from_axis_angle((1, 0, 0), math.pi)
                else:
                    q = Quat.from_axis_angle(c, math.acos(dot))
                self.initialised = True
            else:
                up_est = q.inverse().rotate(vec3(0, 0, 1))
                correction = np.cross(up_est, measured_up) * self.gain
                angle = float(np.linalg.norm(correction))
                if angle > 1e-9:
                    q = (Quat.from_axis_angle(correction / angle, angle) * q).normalized()
        self.quaternion = q
        return q


class FlightComputer:
    """One flight computer with a fixed-rate loop and servo outputs."""

    def __init__(self, board: BoardSpec, gains: ControlGains, imu: IMUModel,
                 channels: list[ServoChannel], supply_voltage: float = 5.0) -> None:
        self.board = board
        self.gains = gains
        self.imu = imu
        self.channels = {channel.name: channel for channel in channels}
        self.estimator = AttitudeEstimator(gains.attitude_filter_gain)
        self.pid_pitch = PID(gains.tilt_kp, gains.tilt_ki, gains.tilt_kd,
                             output_limit=gains.max_gimbal)
        self.pid_yaw = PID(gains.tilt_kp, gains.tilt_ki, gains.tilt_kd,
                           output_limit=gains.max_gimbal)
        self.phase = FlightPhase.PRELAUNCH
        self.time = 0.0
        self.loop_period = board.loop_period()
        self._next_loop_time = 0.0
        self.log: list[ControlFrame] = []
        self.supply_voltage = supply_voltage
        self.missed_updates = 0
        self.loop_overruns = 0
        self.processing_delay = 0.00035   # s, simulated compute time per loop
        self.diagnostics: list[Diagnostic] = []
        self.armed = False
        self._last_loop_wall = 0.0
        self._commanded_tilt = 0.0
        self._flight_time = 0.0

    # -- configuration -----------------------------------------------------
    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def set_phase(self, phase: FlightPhase) -> None:
        if phase is not self.phase:
            if phase is FlightPhase.POWERED:
                self._flight_time = 0.0
            self.phase = phase

    def reset(self) -> None:
        self.estimator.reset()
        self.pid_pitch.reset()
        self.pid_yaw.reset()
        self.time = 0.0
        self._next_loop_time = 0.0
        self.log.clear()
        self.missed_updates = 0
        self.phase = FlightPhase.PRELAUNCH
        self._flight_time = 0.0

    # -- loop --------------------------------------------------------------
    def due(self) -> bool:
        return self.time >= self._next_loop_time - 1e-12

    def update(self, sample: IMUSample, dt: float, phase: FlightPhase | None = None) -> ControlFrame | None:
        """Run one control loop if it is due. Returns the emitted frame."""
        if dt <= 0:
            return None
        self.time += dt
        if phase is not None:
            self.set_phase(phase)
        if not self.due():
            return None
        if self._next_loop_time > 0:
            self._flight_time += dt
        self._next_loop_time += self.loop_period
        # if the loop cannot keep up, count the slip instead of hiding it
        if self.time - self._next_loop_time > self.loop_period:
            missed = int((self.time - self._next_loop_time) / self.loop_period)
            self.missed_updates += max(missed, 0)
            self._next_loop_time = self.time + self.loop_period

        q_est = self.estimator.update(sample, self.loop_period)
        yaw, pitch, roll = q_est.to_euler()
        up_body = q_est.inverse().rotate(vec3(0, 0, 1))   # world up expressed in body

        # guidance demand (small-angle target tilt in body x/y)
        if self.gains.guidance_mode == "pitch_program" and self.phase in (FlightPhase.POWERED, FlightPhase.COAST):
            self._commanded_tilt += self.gains.pitch_program_rate * self.loop_period
            self._commanded_tilt = float(np.clip(self._commanded_tilt, -self.gains.max_tilt_command,
                                                 self.gains.max_tilt_command))
        target_x = math.sin(self._commanded_tilt)
        target_y = 0.0

        error_x = float(up_body[0] - target_x)
        error_y = float(up_body[1] - target_y)
        gimbal_pitch = self.pid_pitch.update(error_x, self.loop_period)
        gimbal_yaw = self.pid_yaw.update(error_y, self.loop_period)

        # rate damping from the gyro (body rates about x and y)
        gimbal_pitch -= self.gains.rate_damping * float(sample.gyro[1])
        gimbal_yaw += self.gains.rate_damping * float(sample.gyro[0])

        limit = self.gains.max_gimbal
        if self.phase is FlightPhase.PRELAUNCH:
            gimbal_pitch = gimbal_yaw = 0.0
        gimbal_pitch = float(np.clip(gimbal_pitch, -limit, limit))
        gimbal_yaw = float(np.clip(gimbal_yaw, -limit, limit))
        if not self.armed:
            gimbal_pitch = gimbal_yaw = 0.0

        # map to servos and step their plants
        angles: dict[str, float] = {}
        currents: dict[str, float] = {}
        for name, channel in self.channels.items():
            sign = 1.0
            trim = 0.0
            if name in self.gains.servo_channels:
                index = self.gains.servo_channels.index(name)
                sign = self.gains.servo_signs[index] if index < len(self.gains.servo_signs) else 1.0
                trim = self.gains.servo_trim[index] if index < len(self.gains.servo_trim) else 0.0
            command = sign * (gimbal_pitch if name == "pitch" else gimbal_yaw) + trim
            channel.command(command)
            state = channel.plant.step(self.loop_period, 0.0, self.supply_voltage)
            angles[name] = state.output_angle
            currents[name] = state.current

        current_draw = self.board.idle_current + sum(currents.values())
        frame = ControlFrame(
            time=self.time,
            phase=self.phase.value,
            estimated_roll=roll,
            estimated_pitch=pitch,
            estimated_yaw=yaw,
            tilt_error_x=error_x,
            tilt_error_y=error_y,
            gimbal_pitch=gimbal_pitch,
            gimbal_yaw=gimbal_yaw,
            servo_angles=angles,
            servo_current=currents,
            loop_time_us=self.processing_delay * 1e6,
            missed_updates=self.missed_updates,
            supply_voltage=self.supply_voltage,
            current_draw=current_draw,
            accel=tuple(float(x) for x in sample.accel),
            gyro=tuple(float(x) for x in sample.gyro),
        )
        self.log.append(frame)
        self._last_loop_wall = self.time
        return frame

    # -- reporting ---------------------------------------------------------
    def power_budget(self) -> dict:
        servo_current = sum(channel.plant.state.current for channel in self.channels.values())
        total = self.board.idle_current + servo_current
        return {
            "board_current_a": self.board.idle_current,
            "servo_current_a": servo_current,
            "total_current_a": total,
            "power_w": total * self.supply_voltage,
            "servo_rail_ok": servo_current <= self.board.max_servo_current,
        }

    def timing(self) -> dict:
        return {
            "loop_rate_hz": 1.0 / self.loop_period,
            "loops": len(self.log),
            "missed_updates": self.missed_updates,
            "loop_overruns": self.loop_overruns,
            "compute_time_us": self.processing_delay * 1e6,
        }

    def validate(self) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        out.extend(Diagnostic("error", "gain", p) for p in self.gains.validate())
        budget = self.power_budget()
        if not budget["servo_rail_ok"]:
            out.append(Diagnostic("warning", "servo-power",
                                  "Servo current exceeds the board's servo rail rating; add a separate BEC."))
        if self.missed_updates:
            out.append(Diagnostic("warning", "loop-overrun",
                                  f"{self.missed_updates} control loop(s) were missed at "
                                  f"{self.board.control_loop_hz:.0f} Hz."))
        return out

    def to_telemetry(self) -> dict:
        return {
            "frames": [f.to_dict() for f in self.log],
            "timing": self.timing(),
            "power": self.power_budget(),
            "phase": self.phase.value,
        }
