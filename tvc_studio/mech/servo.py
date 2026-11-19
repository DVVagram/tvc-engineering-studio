"""Servo plant model.

The servo is a two-inertia system: motor/gear-train angle and output angle, coupled through
a compliant, backlashed gear train, driven by a torque-limited motor with a rate limit from
the spec sheet. That structure is what makes the TVC loop behave like hardware:

* command latency and slew rate,
* gear backlash (output lags input inside the lash gap),
* deadband (small commands produce no motion),
* torque saturation at the voltage-scaled stall torque,
* stiffness of the output under load (the gimbal pushes back on the horn),
* travel stops, which stop motion and produce a stall torque.

It is not a gear-tooth contact simulation; that limitation is documented and the parameters
used are the ones visible in the UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..components.hardware import ServoSpec


def wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class ServoState:
    motor_angle: float = 0.0     # internal motor/gear angle, rad
    output_angle: float = 0.0    # horn angle, rad
    motor_speed: float = 0.0     # rad/s
    output_speed: float = 0.0    # rad/s
    command: float = 0.0         # commanded output angle, rad
    applied_command: float = 0.0  # after deadband/clamping
    torque: float = 0.0          # output torque delivered, N*m
    current: float = 0.0         # A
    saturated: bool = False
    at_stop: bool = False
    delay_remaining: float = 0.0
    lash_offset: float = 0.0     # current lash state (rad)

    def to_dict(self) -> dict:
        return {
            "motor_angle": self.motor_angle,
            "output_angle": self.output_angle,
            "motor_speed": self.motor_speed,
            "output_speed": self.output_speed,
            "command": self.command,
            "applied_command": self.applied_command,
            "torque": self.torque,
            "current": self.current,
            "saturated": self.saturated,
            "at_stop": self.at_stop,
            "lash_offset": self.lash_offset,
        }


class ServoPlant:
    """Integrates servo dynamics at a given timestep."""

    def __init__(self, spec: ServoSpec, initial_angle: float | None = None,
                 supply_voltage: float | None = None) -> None:
        self.spec = spec
        self.supply_voltage = supply_voltage if supply_voltage is not None else spec.rated_voltage
        self.state = ServoState()
        start = 0.0 if initial_angle is None else float(np.clip(initial_angle, spec.angle_min, spec.angle_max))
        self.state.motor_angle = start
        self.state.output_angle = start
        self.state.command = start
        self.state.applied_command = start
        self._delay_buffer: list[float] = []

    # -- properties --------------------------------------------------------
    @property
    def angle(self) -> float:
        return self.state.output_angle

    @property
    def velocity(self) -> float:
        return self.state.output_speed

    def limits(self) -> tuple[float, float]:
        return self.spec.angle_min, self.spec.angle_max

    def reset(self, angle: float = 0.0) -> None:
        self.state = ServoState(
            motor_angle=angle, output_angle=angle, command=angle, applied_command=angle
        )
        self._delay_buffer.clear()

    # -- control -----------------------------------------------------------
    def command(self, angle: float) -> None:
        """Set the commanded output angle (rad); clamped to travel in the dynamics."""
        self.state.command = float(angle)

    def step(self, dt: float, load_torque: float = 0.0,
             supply_voltage: float | None = None) -> ServoState:
        """Advance the servo by ``dt`` seconds under an external load torque.

        ``load_torque`` is the torque the mechanism applies to the horn (positive opposes
        positive motion). Returns the updated state.
        """
        spec = self.spec
        if dt <= 0:
            return self.state
        if supply_voltage is not None:
            self.supply_voltage = supply_voltage

        # command pipeline: latency then deadband (once per user step, not per substep)
        cmd = self.state.command
        if spec.response_delay > 0:
            self._delay_buffer.append(cmd)
            steps = max(int(round(spec.response_delay / dt)), 1)
            if len(self._delay_buffer) > steps:
                cmd = self._delay_buffer.pop(0)
            else:
                cmd = self._delay_buffer[0]
        cmd = float(np.clip(cmd, spec.angle_min, spec.angle_max))
        if abs(cmd - self.state.applied_command) <= spec.deadband:
            cmd = self.state.applied_command
        self.state.applied_command = cmd

        # The gear train is a stiff spring between two small inertias. A 1 kHz flight step
        # needs substeps short enough for the explicit coupling update to stay stable.
        stiffness = max(spec.gear_stiffness, 1e-6)
        inertia_out = max(spec.output_inertia, 1e-9)
        inertia_motor = max(spec.inertia_at_output(), 1e-12)
        omega = math.sqrt(stiffness * (1.0 / inertia_out + 1.0 / inertia_motor))
        stable = 0.25 / max(omega, 1e-9)
        count = int(min(max(math.ceil(dt / stable), 1), 200))
        sub = dt / count
        for _ in range(count):
            self._advance(sub, load_torque)
        return self.state

    def _advance(self, dt: float, load_torque: float) -> None:
        """One stable sub-step of the motor / gear-train / output dynamics."""
        spec = self.spec
        state = self.state
        scale = spec.voltage_scale(self.supply_voltage)
        stall_torque = spec.stall_torque * scale
        max_speed = spec.no_load_speed * scale
        inertia_motor = max(spec.inertia_at_output(), 1e-12)
        inertia_out = max(spec.output_inertia, 1e-9)

        # position loop: a full-scale error commands no-load speed, clipped to travel rate
        # with a ~20 ms closed-loop time constant so the horn tracks the command closely
        err = state.applied_command - state.motor_angle
        target_speed = float(np.clip(err / 0.020, -max_speed, max_speed))
        # motor drive torque: speed error times the torque available per unit speed. The
        # damping term is integrated implicitly below; on a 6e-7 kg*m^2 motor it is far too
        # stiff for an explicit update at flight timesteps.
        kp_torque = stall_torque / max(max_speed, 1e-9)

        # gear coupling with backlash and compliance (torque motor -> output)
        gap = spec.backlash
        delta = state.motor_angle - state.output_angle
        if gap > 0:
            if delta > gap / 2.0:
                coupling = spec.gear_stiffness * (delta - gap / 2.0)
            elif delta < -gap / 2.0:
                coupling = spec.gear_stiffness * (delta + gap / 2.0)
            else:
                coupling = 0.0
            state.lash_offset = float(np.clip(delta, -gap / 2.0, gap / 2.0))
        else:
            coupling = spec.gear_stiffness * delta
        coupling += spec.gear_damping * (state.motor_speed - state.output_speed)
        transmitted = float(np.clip(coupling, -stall_torque, stall_torque))
        state.saturated = abs(transmitted) >= stall_torque - 1e-9

        # friction in the gear train opposes the motor's motion
        friction = spec.gear_friction * (1.0 if state.motor_speed >= 0 else -1.0)
        if abs(state.motor_speed) < 1e-6:
            friction *= 0.5

        torque_ext = kp_torque * target_speed - transmitted / max(spec.efficiency, 1e-6) - friction
        state.motor_speed = ((state.motor_speed + dt * torque_ext / inertia_motor)
                             / (1.0 + dt * kp_torque / inertia_motor))
        state.motor_speed = float(np.clip(state.motor_speed, -max_speed * 1.05, max_speed * 1.05))
        state.motor_angle += state.motor_speed * dt

        delivered = transmitted * spec.efficiency
        state.output_speed += (delivered - load_torque) / inertia_out * dt
        state.output_speed = float(np.clip(state.output_speed, -max_speed * 1.5, max_speed * 1.5))
        state.output_angle += state.output_speed * dt

        # stops: kill motion beyond travel, hold angle, allow stall torque to build
        if state.output_angle < spec.angle_min:
            state.output_angle = spec.angle_min
            state.output_speed = 0.0
            state.at_stop = True
        elif state.output_angle > spec.angle_max:
            state.output_angle = spec.angle_max
            state.output_speed = 0.0
            state.at_stop = True
        else:
            state.at_stop = False

        # motor cannot over-wind far past the stops either (mechanical limit)
        state.motor_angle = float(np.clip(
            state.motor_angle,
            spec.angle_min - max(gap, 0.01),
            spec.angle_max + max(gap, 0.01),
        ))

        state.torque = delivered
        state.current = spec.idle_current + abs(transmitted) / max(spec.kt(), 1e-9)

    # -- diagnostics -------------------------------------------------------
    def lag(self) -> float:
        return self.state.applied_command - self.state.output_angle

    def power(self) -> float:
        return self.state.current * self.supply_voltage

    def summary(self) -> dict:
        return {
            "angle_deg": math.degrees(self.state.output_angle),
            "command_deg": math.degrees(self.state.applied_command),
            "error_deg": math.degrees(self.lag()),
            "speed_rpm": self.state.output_speed * 60.0 / (2.0 * math.pi),
            "torque_nm": self.state.torque,
            "current_a": self.state.current,
            "saturated": self.state.saturated,
            "at_stop": self.state.at_stop,
        }


@dataclass
class ServoTester:
    """Small helper that runs a step response for the UI and for tests."""

    plant: ServoPlant
    dt: float = 1.0 / 2000.0

    def step_response(self, target: float, duration: float = 0.5,
                      load_torque: float = 0.0) -> list[tuple[float, float, float]]:
        samples: list[tuple[float, float, float]] = []
        self.plant.command(target)
        t = 0.0
        while t < duration:
            state = self.plant.step(self.dt, load_torque)
            samples.append((t, state.output_angle, state.torque))
            t += self.dt
        return samples

    def slew_rate(self, duration: float = 0.2) -> float:
        """Measured maximum output speed for a large command, rad/s."""
        start = self.plant.state.output_angle
        self.plant.command(start + 1.0)
        max_speed = 0.0
        t = 0.0
        while t < duration:
            state = self.plant.step(self.dt)
            max_speed = max(max_speed, abs(state.output_speed))
            t += self.dt
        return max_speed
