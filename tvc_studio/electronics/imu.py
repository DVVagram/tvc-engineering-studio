"""IMU measurement synthesis.

The sensor reads what a real sensor at its own mounting pose would read: body-frame
angular rate and specific force, with configurable range, resolution, noise, bias, bias
drift, axis misalignment, sampling rate and latency.

This is a sensor model, not a state estimator: it does not filter, and it does not claim to
match a particular unit's calibration. Bias values default to the datasheet-typical
"uncalibrated" level, which is why flights show drift unless the firmware compensates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..components.hardware import SensorSpec
from ..core.math3d import GRAVITY, Quat, Transform, as_vec3, vec3


@dataclass
class IMUSample:
    time: float
    accel: np.ndarray          # m/s^2, sensor frame (specific force)
    gyro: np.ndarray           # rad/s, sensor frame
    accel_raw: np.ndarray      # integer counts as the ADC would report
    gyro_raw: np.ndarray
    temperature_c: float = 25.0
    clipped: bool = False

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "accel": self.accel.tolist(),
            "gyro": self.gyro.tolist(),
            "clipped": self.clipped,
            "temperature_c": self.temperature_c,
        }


class IMUModel:
    """Stateful simulated IMU attached to a mounting pose on a body."""

    def __init__(self, spec: SensorSpec, seed: int | None = 1234,
                 mounting: Transform | None = None) -> None:
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.mounting = mounting or spec.mounting_transform()
        self.gyro_bias = as_vec3(spec.gyro_bias).copy()
        self.accel_bias = as_vec3(spec.accel_bias).copy()
        self.time = 0.0
        self._sample_interval = 1.0 / max(spec.sample_rate, 1e-6)
        self._accumulator = 0.0
        self._delay_buffer: list[IMUSample] = []
        self.last_sample: IMUSample | None = None
        self._prev_omega: np.ndarray | None = None
        self._noise_bandwidth = math.sqrt(max(spec.sample_rate, 1.0))

    # -- mounting ----------------------------------------------------------
    def set_mounting(self, transform: Transform) -> None:
        self.mounting = transform

    def world_position(self, body_transform: Transform) -> np.ndarray:
        return (body_transform @ self.mounting).translation

    # -- measurement -------------------------------------------------------
    def measure(self, body_transform: Transform, angular_velocity_world: np.ndarray,
                linear_acceleration_world: np.ndarray, dt: float,
                body_velocity_world: np.ndarray | None = None) -> IMUSample | None:
        """Sample at the configured rate; returns None between samples."""
        self.time += dt
        self._accumulator += dt
        # integrate bias random walk at the real rate
        drift = self.spec.bias_drift * math.sqrt(max(dt, 0.0))
        self.gyro_bias = self.gyro_bias + self.rng.normal(0.0, drift, 3)
        if self._accumulator + 1e-12 < self._sample_interval:
            return None
        self._accumulator = 0.0

        R_mount = body_transform.rotation.matrix() @ self.mounting.rotation.matrix()
        omega_body = R_mount.T @ as_vec3(angular_velocity_world)
        # specific force includes gravity reaction and rotational acceleration terms
        specific_force_world = as_vec3(linear_acceleration_world) - vec3(0, 0, -GRAVITY)
        accel_body = R_mount.T @ specific_force_world

        # sensor noise (density integrated over the measurement bandwidth)
        accel_noise = self.rng.normal(0.0, self.spec.accel_noise_density * self._noise_bandwidth, 3)
        gyro_noise = self.rng.normal(0.0, self.spec.gyro_noise_density * self._noise_bandwidth, 3)

        # axis misalignment: small rotation error of the sensing axes
        if self.spec.axis_misalignment_deg:
            angle = math.radians(self.spec.axis_misalignment_deg)
            axis = self.rng.normal(size=3)
            if np.linalg.norm(axis) > 1e-9:
                M = Quat.from_axis_angle(axis, angle).matrix()
                accel_body = M @ accel_body
                gyro_body = M @ omega_body
            else:
                gyro_body = omega_body
        else:
            gyro_body = omega_body

        accel = accel_body + self.accel_bias + accel_noise
        gyro = gyro_body + self.gyro_bias + gyro_noise

        # range clipping and quantisation
        accel_full = self.spec.accel_range_g * GRAVITY
        gyro_full = math.radians(self.spec.gyro_range_dps)
        clipped = bool(np.any(np.abs(accel) > accel_full) or np.any(np.abs(gyro) > gyro_full))
        accel = np.clip(accel, -accel_full, accel_full)
        gyro = np.clip(gyro, -gyro_full, gyro_full)
        accel_res = self.spec.accel_resolution()
        gyro_res = self.spec.gyro_resolution()
        accel = np.round(accel / accel_res) * accel_res
        gyro = np.round(gyro / gyro_res) * gyro_res
        accel_raw = np.round(accel / accel_res).astype(np.int32)
        gyro_raw = np.round(gyro / gyro_res).astype(np.int32)

        sample = IMUSample(self.time, accel, gyro, accel_raw, gyro_raw, clipped=clipped)

        # latency: queue and release after the configured delay
        self._delay_buffer.append(sample)
        delay_steps = int(round(self.spec.delay / self._sample_interval))
        if len(self._delay_buffer) > delay_steps:
            self.last_sample = self._delay_buffer.pop(0)
            return self.last_sample
        return None

    # -- convenience -------------------------------------------------------
    def static_sample(self) -> IMUSample:
        """What the sensor reads when the body is level and at rest."""
        transform = Transform.identity()
        return self.measure(transform, vec3(), vec3(), self._sample_interval) or IMUSample(
            0.0, vec3(0, 0, GRAVITY), vec3(), np.zeros(3, dtype=int), np.zeros(3, dtype=int)
        )

    def bias_estimate(self) -> dict:
        return {"gyro_bias": self.gyro_bias.tolist(), "accel_bias": self.accel_bias.tolist()}


class TripleIMU:
    """Redundant IMU set (common on flight computers that vote two-of-three)."""

    def __init__(self, specs: list[SensorSpec]) -> None:
        self.units = [IMUModel(spec, seed=100 + i) for i, spec in enumerate(specs)]

    def measure(self, *args, **kwargs) -> list[IMUSample | None]:
        return [unit.measure(*args, **kwargs) for unit in self.units]
