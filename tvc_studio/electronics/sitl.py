"""Software-in-the-loop harness.

Closes the loop between :class:`FlightComputer` and a flight plant: the plant advances,
the IMU produces samples at its own rate, the control loop runs at the board's rate, and
the resulting servo angles deflect the thrust vector. Nothing in the controller sees the
plant's true attitude.

The harness intentionally mirrors the HIL protocol: same channel semantics, same timing
statistics, so moving from SITL to a real Pico changes the transport rather than the
control logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.interfaces import Diagnostic
from .flightcomputer import ControlFrame, FlightComputer


@dataclass
class SITLResult:
    frames: list[ControlFrame] = field(default_factory=list)
    telemetry: list[dict] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    max_tilt_error: float = 0.0
    final_state: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "control_frames": len(self.frames),
            "samples": len(self.telemetry),
            "max_tilt_error_deg": np.degrees(self.max_tilt_error),
            "missed_updates": self.frames[-1].missed_updates if self.frames else 0,
            "final": self.final_state,
        }


class SoftwareInTheLoop:
    """Drives a plant with the simulated flight computer."""

    def __init__(self, plant, computer: FlightComputer, imu, dt: float = 0.001,
                 apply_commands=None) -> None:
        """
        ``plant`` must expose ``state_for_imu()`` returning ``(transform, omega, accel)``
        and ``set_tvc(pitch, yaw)`` / ``step(dt)``. The flight simulator in
        ``physics.flight`` implements this interface.
        """
        self.plant = plant
        self.computer = computer
        self.imu = imu
        self.dt = dt
        self.apply_commands = apply_commands
        self.last_commands: tuple[float, float] = (0.0, 0.0)

    def run(self, duration: float, stop_condition=None) -> SITLResult:
        result = SITLResult()
        steps = max(int(duration / self.dt), 1)
        self.computer.arm()
        for _ in range(steps):
            transform, omega, accel = self.plant.state_for_imu()
            sample = self.imu.measure(transform, omega, accel, self.dt)
            if sample is None:
                self.plant.step(self.dt)
                continue
            frame = self.computer.update(sample, self.dt, getattr(self.plant, "phase", None))
            if frame is not None:
                result.frames.append(frame)
                err = max(abs(frame.tilt_error_x), abs(frame.tilt_error_y))
                result.max_tilt_error = max(result.max_tilt_error, err)
                self._apply(frame)
            self.plant.step(self.dt)
            if hasattr(self.plant, "telemetry_row"):
                row = self.plant.telemetry_row()
                if row:
                    row["gimbal_pitch_cmd"] = self.last_commands[0]
                    row["gimbal_yaw_cmd"] = self.last_commands[1]
                    result.telemetry.append(row)
            if stop_condition is not None and stop_condition(self.plant):
                break
        result.diagnostics.extend(self.computer.validate())
        result.final_state = self.plant.state_dict()
        return result

    def _apply(self, frame: ControlFrame) -> None:
        angle_pitch = frame.servo_angles.get("pitch", frame.gimbal_pitch)
        angle_yaw = frame.servo_angles.get("yaw", frame.gimbal_yaw)
        self.last_commands = (angle_pitch, angle_yaw)
        if self.apply_commands is not None:
            self.apply_commands(angle_pitch, angle_yaw)
            return
        setter = getattr(self.plant, "set_tvc", None)
        if setter is not None:
            setter(angle_pitch, angle_yaw)


class HardwareLoopStats:
    """Transport statistics shared by SITL and HIL so the two are comparable."""

    def __init__(self, expected_period: float) -> None:
        self.expected_period = expected_period
        self.sent = 0
        self.received = 0
        self.missed = 0
        self.latencies: list[float] = []
        self.last_rx_time: float | None = None

    def record_sent(self) -> None:
        self.sent += 1

    def record_received(self, now: float) -> None:
        self.received += 1
        if self.last_rx_time is not None:
            delta = now - self.last_rx_time
            self.latencies.append(delta)
            if delta > 2.5 * self.expected_period:
                self.missed += int(delta / self.expected_period) - 1
        self.last_rx_time = now

    def summary(self) -> dict:
        if not self.latencies:
            return {"sent": self.sent, "received": self.received, "missed": self.missed}
        arr = np.array(self.latencies)
        return {
            "sent": self.sent,
            "received": self.received,
            "missed": self.missed,
            "mean_period_ms": float(arr.mean() * 1000.0),
            "jitter_ms": float(arr.std() * 1000.0),
            "max_gap_ms": float(arr.max() * 1000.0),
            "update_rate_hz": float(1.0 / max(arr.mean(), 1e-6)),
        }
