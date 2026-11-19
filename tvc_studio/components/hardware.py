"""Configurable hardware definitions: servos, motors, sensors, flight computers.

Real hardware is modelled as data, not as hard-coded logic. The shipped defaults are the
specification's initial set (MG90S, MPU6050, Raspberry Pi Pico 2 W) plus a few common
alternatives; every field is editable and every definition can be saved to the component
library and reused across projects.

Datasheet values are tagged MANUFACTURER where they are published ratings. Where a value
is a plausible engineering estimate (e.g. gear-train stiffness of a hobby servo) it is
tagged ESTIMATED and the UI shows that.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.interfaces import Provenance
from ..core.math3d import Transform, as_vec3, vec3


# --------------------------------------------------------------------------- #
# Servo
# --------------------------------------------------------------------------- #
@dataclass
class ServoSpec:
    """A hobby/industrial servo with the parameters that actually drive the plant."""

    id: str
    name: str
    model: str = ""
    mass: float = 0.0134                  # kg
    dimensions: tuple[float, float, float] = (0.0228, 0.0122, 0.0285)  # m (x, y, z)
    body_offset: np.ndarray = field(default_factory=lambda: vec3(0, 0, 0))
    shaft_offset: np.ndarray = field(default_factory=lambda: vec3(0, 0, 0.0285 / 2))
    mount_hole_spacing: float = 0.028  # m, typical 28 mm for micro servos
    rated_voltage: float = 4.8          # V
    operating_voltage_min: float = 4.8
    operating_voltage_max: float = 6.0
    stall_torque: float = 1.8 * 0.0980665 * 0.01  # 1.8 kg*cm in N*m
    no_load_speed: float = math.radians(60.0) / 0.10  # 0.10 s/60 deg -> rad/s
    angle_min: float = math.radians(-90.0)
    angle_max: float = math.radians(90.0)
    response_delay: float = 0.006       # s, command-to-motion latency
    backlash: float = math.radians(0.6)  # rad, total gear lash
    deadband: float = math.radians(0.25)
    gear_friction: float = 0.004        # N*m, Coulomb friction in the train
    gear_stiffness: float = 2.5         # N*m/rad, output compliance
    gear_damping: float = 0.002         # N*m*s/rad
    efficiency: float = 0.75
    rotor_inertia: float = 6.0e-7       # kg*m^2 referred to the output
    horn_mass: float = 0.0015           # kg
    horn_length: float = 0.015          # m
    stall_current: float = 0.70         # A at stall, rated voltage
    idle_current: float = 0.006         # A
    torque_constant: float = 0.0        # N*m/A, derived when zero
    provenance: Provenance = Provenance.MANUFACTURER
    notes: str = ""

    # -- derived -----------------------------------------------------------
    def voltage_scale(self, voltage: float | None = None) -> float:
        """Torque/speed scale versus rated voltage (approximately linear, clamped)."""
        v = self.rated_voltage if voltage is None else voltage
        v = float(np.clip(v, self.operating_voltage_min * 0.8, self.operating_voltage_max * 1.2))
        return float(np.clip(v / self.rated_voltage, 0.0, 1.4))

    def torque_at(self, voltage: float | None = None) -> float:
        return self.stall_torque * self.voltage_scale(voltage)

    def speed_at(self, voltage: float | None = None) -> float:
        return self.no_load_speed * self.voltage_scale(voltage)

    def travel(self) -> float:
        return self.angle_max - self.angle_min

    def kt(self) -> float:
        """Torque constant N*m/A; from stall torque and stall current when not given."""
        if self.torque_constant > 0:
            return self.torque_constant
        return self.stall_torque / max(self.stall_current, 1e-6)

    def inertia_at_output(self) -> float:
        return self.rotor_inertia + self.horn_mass * (self.horn_length ** 2) / 3.0

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.stall_torque <= 0:
            problems.append("stall torque must be positive")
        if self.no_load_speed <= 0:
            problems.append("no-load speed must be positive")
        if self.angle_max <= self.angle_min:
            problems.append("angle range is empty")
        if self.rated_voltage <= 0:
            problems.append("rated voltage must be positive")
        if not 0 < self.efficiency <= 1:
            problems.append("efficiency must be in (0, 1]")
        return problems

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["body_offset"] = np.asarray(self.body_offset).tolist()
        d["shaft_offset"] = np.asarray(self.shaft_offset).tolist()
        d["dimensions"] = list(self.dimensions)
        d["provenance"] = self.provenance.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "ServoSpec":
        allowed = ServoSpec.__dataclass_fields__
        kwargs = {}
        for k, v in d.items():
            if k not in allowed:
                continue
            if k == "provenance":
                kwargs[k] = Provenance(v)
            elif k == "dimensions":
                kwargs[k] = tuple(float(x) for x in v)
            elif k in ("body_offset", "shaft_offset"):
                kwargs[k] = as_vec3(v)
            else:
                kwargs[k] = v
        return ServoSpec(**kwargs)


# --------------------------------------------------------------------------- #
# Electric motor / actuator
# --------------------------------------------------------------------------- #
@dataclass
class MotorSpec:
    id: str
    name: str
    kind: str = "brushless"       # brushless | brushed | stepper | linear
    mass: float = 0.030
    dimensions: tuple[float, float, float] = (0.028, 0.028, 0.030)
    rotor_inertia: float = 2.0e-6  # kg*m^2 at the output shaft
    max_torque: float = 0.10       # N*m continuous
    peak_torque: float = 0.25      # N*m intermittent
    no_load_speed: float = 1200.0  # rad/s at nominal voltage
    nominal_voltage: float = 11.1
    resistance: float = 0.09       # ohm (phase-to-phase)
    kv: float = 1400.0             # rpm/V
    gear_ratio: float = 1.0
    efficiency: float = 0.85
    max_current: float = 25.0
    response_delay: float = 0.002
    provenance: Provenance = Provenance.MANUFACTURER
    notes: str = ""

    def output_speed(self, voltage: float | None = None) -> float:
        v = self.nominal_voltage if voltage is None else voltage
        return self.no_load_speed * (v / self.nominal_voltage)

    def validate(self) -> list[str]:
        out: list[str] = []
        if self.max_torque <= 0 or self.peak_torque <= 0:
            out.append("motor torque ratings must be positive")
        if self.gear_ratio <= 0:
            out.append("gear ratio must be positive")
        return out

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["dimensions"] = list(self.dimensions)
        d["provenance"] = self.provenance.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "MotorSpec":
        allowed = MotorSpec.__dataclass_fields__
        kwargs = {}
        for k, v in d.items():
            if k not in allowed:
                continue
            if k == "provenance":
                kwargs[k] = Provenance(v)
            elif k == "dimensions":
                kwargs[k] = tuple(float(x) for x in v)
            else:
                kwargs[k] = v
        return MotorSpec(**kwargs)


# --------------------------------------------------------------------------- #
# Sensors
# --------------------------------------------------------------------------- #
@dataclass
class SensorSpec:
    """IMU-like sensor, characterised by what the firmware would actually read."""

    id: str
    name: str
    kind: str = "imu"              # imu | gyro | accelerometer | barometer | magnetometer
    mass: float = 0.001
    dimensions: tuple[float, float, float] = (0.020, 0.015, 0.003)
    mounting_offset: np.ndarray = field(default_factory=lambda: vec3())
    mounting_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    interface: str = "i2c"
    sample_rate: float = 1000.0     # Hz
    accel_range_g: float = 16.0
    gyro_range_dps: float = 1000.0
    accel_bits: int = 16
    gyro_bits: int = 16
    accel_noise_density: float = 300e-6 * 9.80665   # m/s^2/sqrt(Hz)
    gyro_noise_density: float = math.radians(0.005)  # rad/s/sqrt(Hz)
    accel_bias: np.ndarray = field(default_factory=lambda: vec3())
    gyro_bias: np.ndarray = field(default_factory=lambda: vec3())
    bias_drift: float = 1.0e-4       # rad/s per s (gyro random walk)
    delay: float = 0.002             # s, sample-to-available delay
    axis_misalignment_deg: float = 0.2
    provenance: Provenance = Provenance.MANUFACTURER
    notes: str = ""

    # -- derived -----------------------------------------------------------
    def accel_resolution(self) -> float:
        full_scale = self.accel_range_g * 9.80665 * 2.0
        return full_scale / (2 ** self.accel_bits)

    def gyro_resolution(self) -> float:
        full_scale = math.radians(self.gyro_range_dps) * 2.0
        return full_scale / (2 ** self.gyro_bits)

    def mounting_transform(self) -> Transform:
        from ..core.math3d import Quat
        yaw, pitch, roll = (math.radians(a) for a in self.mounting_euler_deg)
        return Transform(Quat.from_euler(yaw, pitch, roll), as_vec3(self.mounting_offset))

    def validate(self) -> list[str]:
        out: list[str] = []
        if self.sample_rate <= 0:
            out.append("sample rate must be positive")
        if self.accel_range_g <= 0 or self.gyro_range_dps <= 0:
            out.append("measurement ranges must be positive")
        if not 8 <= self.accel_bits <= 32:
            out.append("accelerometer resolution should be 8-32 bits")
        return out

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["dimensions"] = list(self.dimensions)
        d["mounting_offset"] = np.asarray(self.mounting_offset).tolist()
        d["mounting_euler_deg"] = list(self.mounting_euler_deg)
        d["accel_bias"] = np.asarray(self.accel_bias).tolist()
        d["gyro_bias"] = np.asarray(self.gyro_bias).tolist()
        d["provenance"] = self.provenance.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "SensorSpec":
        allowed = SensorSpec.__dataclass_fields__
        kwargs = {}
        for k, v in d.items():
            if k not in allowed:
                continue
            if k == "provenance":
                kwargs[k] = Provenance(v)
            elif k in ("dimensions", "mounting_euler_deg"):
                kwargs[k] = tuple(float(x) for x in v)
            elif k in ("mounting_offset", "accel_bias", "gyro_bias"):
                kwargs[k] = as_vec3(v)
            else:
                kwargs[k] = v
        return SensorSpec(**kwargs)


# --------------------------------------------------------------------------- #
# Flight computer / board
# --------------------------------------------------------------------------- #
@dataclass
class BoardSpec:
    id: str
    name: str
    mcu: str = "RP2350"
    cores: int = 2
    clock_hz: float = 150e6
    ram_bytes: int = 520 * 1024
    flash_bytes: int = 4 * 1024 * 1024
    logic_voltage: float = 3.3
    input_voltage_min: float = 1.8
    input_voltage_max: float = 5.5
    adc_bits: int = 12
    pwm_channels: int = 12
    uart_channels: int = 2
    i2c_channels: int = 2
    spi_channels: int = 2
    wifi: bool = True
    bluetooth: bool = True
    idle_current: float = 0.025     # A, board only
    max_servo_current: float = 1.5  # A through the 5 V rail (BEC dependent)
    control_loop_hz: float = 250.0  # default scheduler rate for SITL
    mass: float = 0.003
    dimensions: tuple[float, float, float] = (0.051, 0.021, 0.005)
    provenance: Provenance = Provenance.MANUFACTURER
    notes: str = ""

    def loop_period(self) -> float:
        return 1.0 / max(self.control_loop_hz, 1e-6)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["dimensions"] = list(self.dimensions)
        d["provenance"] = self.provenance.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "BoardSpec":
        allowed = BoardSpec.__dataclass_fields__
        kwargs = {}
        for k, v in d.items():
            if k not in allowed:
                continue
            if k == "provenance":
                kwargs[k] = Provenance(v)
            elif k == "dimensions":
                kwargs[k] = tuple(float(x) for x in v)
            else:
                kwargs[k] = v
        return BoardSpec(**kwargs)


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
def default_servos() -> list[ServoSpec]:
    return [
        ServoSpec(
            id="mg90s",
            name="MG90S micro servo",
            model="MG90S",
            mass=0.0134,
            dimensions=(0.0228, 0.0122, 0.0285),
            shaft_offset=vec3(0, -0.0061, 0.0142),
            mount_hole_spacing=0.028,
            rated_voltage=4.8,
            operating_voltage_min=4.8,
            operating_voltage_max=6.0,
            stall_torque=1.8 * 0.0980665 * 0.01,
            no_load_speed=math.radians(60.0) / 0.10,
            angle_min=math.radians(-90.0),
            angle_max=math.radians(90.0),
            response_delay=0.006,
            backlash=math.radians(0.6),
            deadband=math.radians(0.25),
            gear_friction=0.004,
            gear_stiffness=2.5,
            gear_damping=0.002,
            efficiency=0.75,
            rotor_inertia=6.0e-7,
            horn_mass=0.0015,
            horn_length=0.015,
            stall_current=0.70,
            idle_current=0.006,
            notes="Metal-gear micro servo. Ratings at 4.8 V; torque/speed scale with supply voltage.",
        ),
        ServoSpec(
            id="mg996r",
            name="MG996R standard servo",
            model="MG996R",
            mass=0.055,
            dimensions=(0.0407, 0.0197, 0.0429),
            shaft_offset=vec3(0, -0.00985, 0.0214),
            mount_hole_spacing=0.0495,
            rated_voltage=6.0,
            operating_voltage_min=4.8,
            operating_voltage_max=7.2,
            stall_torque=9.4 * 0.0980665 * 0.01,
            no_load_speed=math.radians(60.0) / 0.15,
            angle_min=math.radians(-90.0),
            angle_max=math.radians(90.0),
            response_delay=0.008,
            backlash=math.radians(0.8),
            deadband=math.radians(0.4),
            gear_friction=0.02,
            gear_stiffness=4.0,
            gear_damping=0.006,
            efficiency=0.72,
            rotor_inertia=5.0e-6,
            horn_mass=0.004,
            horn_length=0.025,
            stall_current=2.5,
            idle_current=0.02,
        ),
        ServoSpec(
            id="ds3218",
            name="DS3218 high-torque servo",
            model="DS3218",
            mass=0.060,
            dimensions=(0.040, 0.020, 0.040),
            shaft_offset=vec3(0, -0.010, 0.020),
            mount_hole_spacing=0.0495,
            rated_voltage=6.8,
            operating_voltage_min=4.8,
            operating_voltage_max=8.4,
            stall_torque=20.0 * 0.0980665 * 0.01,
            no_load_speed=math.radians(60.0) / 0.13,
            angle_min=math.radians(-90.0),
            angle_max=math.radians(90.0),
            response_delay=0.006,
            backlash=math.radians(0.5),
            deadband=math.radians(0.2),
            gear_friction=0.03,
            gear_stiffness=8.0,
            gear_damping=0.01,
            efficiency=0.75,
            rotor_inertia=8.0e-6,
            horn_mass=0.005,
            horn_length=0.025,
            stall_current=3.0,
            idle_current=0.02,
        ),
    ]


def default_motors() -> list[MotorSpec]:
    return [
        MotorSpec(id="motor_generic_brushless", name="Brushless outrunner 2207 2400KV",
                  mass=0.032, dimensions=(0.028, 0.028, 0.031),
                  rotor_inertia=1.8e-6, max_torque=0.18, peak_torque=0.45,
                  no_load_speed=1400.0, nominal_voltage=11.1, kv=2400.0,
                  gear_ratio=1.0, efficiency=0.85, max_current=30.0),
        MotorSpec(id="motor_geared_dc", name="Geared DC actuator (metal gearbox)",
                  kind="brushed", mass=0.045, dimensions=(0.025, 0.025, 0.055),
                  rotor_inertia=2.5e-5, max_torque=0.9, peak_torque=2.5,
                  no_load_speed=20.0, nominal_voltage=7.4, gear_ratio=48.0,
                  efficiency=0.65, max_current=3.0),
        MotorSpec(id="linear_actuator_small", name="Linear actuator (TVC link)",
                  kind="linear", mass=0.022, dimensions=(0.015, 0.015, 0.045),
                  max_torque=0.0, peak_torque=0.0, no_load_speed=0.06, nominal_voltage=6.0),
    ]


def default_sensors() -> list[SensorSpec]:
    return [
        SensorSpec(
            id="mpu6050",
            name="MPU6050 IMU",
            kind="imu",
            mass=0.0012,
            dimensions=(0.020, 0.015, 0.003),
            sample_rate=1000.0,
            accel_range_g=16.0,
            gyro_range_dps=1000.0,
            accel_noise_density=300e-6 * 9.80665,
            gyro_noise_density=math.radians(0.005),
            bias_drift=1.5e-4,
            delay=0.002,
            axis_misalignment_deg=0.2,
            notes="InvenSense MPU6050, configured ranges. Factory bias not calibrated in simulation.",
        ),
        SensorSpec(
            id="generic_gyro",
            name="Generic rate gyro",
            kind="gyro",
            mass=0.0008,
            dimensions=(0.012, 0.012, 0.003),
            gyro_range_dps=2000.0,
            gyro_noise_density=math.radians(0.01),
            delay=0.001,
        ),
        SensorSpec(
            id="generic_accel",
            name="Generic accelerometer",
            kind="accelerometer",
            mass=0.0008,
            dimensions=(0.012, 0.012, 0.003),
            accel_range_g=8.0,
            accel_noise_density=200e-6 * 9.80665,
            delay=0.001,
        ),
        SensorSpec(
            id="icm42688",
            name="ICM-42688-P IMU",
            kind="imu",
            mass=0.0009,
            dimensions=(0.014, 0.014, 0.003),
            accel_range_g=16.0,
            gyro_range_dps=2000.0,
            accel_noise_density=70e-6 * 9.80665,
            gyro_noise_density=math.radians(0.0035),
            sample_rate=8000.0,
            delay=0.0015,
        ),
    ]


def default_boards() -> list[BoardSpec]:
    return [
        BoardSpec(
            id="pico2w",
            name="Raspberry Pi Pico 2 W",
            mcu="RP2350",
            cores=2,
            clock_hz=150e6,
            ram_bytes=520 * 1024,
            flash_bytes=4 * 1024 * 1024,
            logic_voltage=3.3,
            input_voltage_min=1.8,
            input_voltage_max=5.5,
            adc_bits=12,
            pwm_channels=12,
            uart_channels=2,
            i2c_channels=2,
            spi_channels=2,
            wifi=True,
            bluetooth=True,
            idle_current=0.025,
            max_servo_current=1.5,
            control_loop_hz=250.0,
            mass=0.003,
            dimensions=(0.051, 0.021, 0.005),
            notes="Dual Cortex-M33 @150 MHz, 520 kB SRAM. Servo rail current limited by the 5 V supply.",
        ),
        BoardSpec(
            id="pico_w",
            name="Raspberry Pi Pico W",
            mcu="RP2040",
            cores=2,
            clock_hz=133e6,
            ram_bytes=264 * 1024,
            flash_bytes=2 * 1024 * 1024,
            wifi=True,
            control_loop_hz=200.0,
        ),
        BoardSpec(
            id="pico2",
            name="Raspberry Pi Pico 2",
            mcu="RP2350",
            clock_hz=150e6,
            wifi=False,
            bluetooth=False,
            control_loop_hz=250.0,
        ),
    ]


class HardwareLibrary:
    """Servo/motor/sensor/board registry with JSON persistence."""

    def __init__(self) -> None:
        self.servos: dict[str, ServoSpec] = {s.id: s for s in default_servos()}
        self.motors: dict[str, MotorSpec] = {m.id: m for m in default_motors()}
        self.sensors: dict[str, SensorSpec] = {s.id: s for s in default_sensors()}
        self.boards: dict[str, BoardSpec] = {b.id: b for b in default_boards()}

    # -- lookup ------------------------------------------------------------
    def servo(self, spec_id: str) -> ServoSpec:
        return self.servos.get(spec_id) or next(iter(self.servos.values()))

    def motor(self, spec_id: str) -> MotorSpec:
        return self.motors.get(spec_id) or next(iter(self.motors.values()))

    def sensor(self, spec_id: str) -> SensorSpec:
        return self.sensors.get(spec_id) or next(iter(self.sensors.values()))

    def board(self, spec_id: str) -> BoardSpec:
        return self.boards.get(spec_id) or next(iter(self.boards.values()))

    # -- mutation ----------------------------------------------------------
    def add(self, spec) -> None:
        if isinstance(spec, ServoSpec):
            self.servos[spec.id] = spec
        elif isinstance(spec, MotorSpec):
            self.motors[spec.id] = spec
        elif isinstance(spec, SensorSpec):
            self.sensors[spec.id] = spec
        elif isinstance(spec, BoardSpec):
            self.boards[spec.id] = spec
        else:
            raise TypeError(f"unsupported hardware type: {type(spec).__name__}")

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "servos": [s.to_dict() for s in self.servos.values()],
            "motors": [m.to_dict() for m in self.motors.values()],
            "sensors": [s.to_dict() for s in self.sensors.values()],
            "boards": [b.to_dict() for b in self.boards.values()],
        }

    def load_dict(self, d: dict) -> None:
        for entry in d.get("servos", []):
            try:
                spec = ServoSpec.from_dict(entry)
                self.servos[spec.id] = spec
            except (TypeError, ValueError):
                continue
        for entry in d.get("motors", []):
            try:
                spec = MotorSpec.from_dict(entry)
                self.motors[spec.id] = spec
            except (TypeError, ValueError):
                continue
        for entry in d.get("sensors", []):
            try:
                spec = SensorSpec.from_dict(entry)
                self.sensors[spec.id] = spec
            except (TypeError, ValueError):
                continue
        for entry in d.get("boards", []):
            try:
                spec = BoardSpec.from_dict(entry)
                self.boards[spec.id] = spec
            except (TypeError, ValueError):
                continue

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def load(self, path: str) -> None:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                self.load_dict(json.load(fh))


def clone(spec, **changes):
    return replace(spec, **changes)
