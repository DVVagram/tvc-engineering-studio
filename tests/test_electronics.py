import math

import numpy as np
import pytest

from tvc_studio.components.hardware import HardwareLibrary
from tvc_studio.core.math3d import Transform, vec3
from tvc_studio.electronics.flightcomputer import ControlGains, FlightComputer, ServoChannel
from tvc_studio.electronics.hil import LoopbackHIL
from tvc_studio.electronics.imu import IMUModel
from tvc_studio.mech.servo import ServoPlant, ServoTester


@pytest.fixture()
def hardware():
    return HardwareLibrary()


def test_imu_stationary_reads_gravity(hardware):
    imu = IMUModel(hardware.sensor("mpu6050"))
    # zero kinematic acceleration -> the sensor reads the gravity reaction (+1 g)
    samples = [imu.measure(Transform.identity(), vec3(), vec3(), 0.001)
               for _ in range(200)]
    samples = [s for s in samples if s is not None]
    mean = np.mean([s.accel for s in samples], axis=0)
    assert np.linalg.norm(mean) == pytest.approx(9.80665, rel=0.05)
    assert mean[0] == pytest.approx(0.0, abs=0.5)


def test_imu_accel_range_saturates(hardware):
    imu = IMUModel(hardware.sensor("mpu6050"))
    sample = None
    for _ in range(20):
        sample = imu.measure(Transform.identity(), vec3(), vec3(0, 0, 500.0), 0.001)
        if sample is not None:
            break
    assert sample is not None
    full_scale = hardware.sensor("mpu6050").accel_range_g * 9.80665
    assert abs(sample.accel[2]) <= full_scale + 1e-6


def test_servo_slew_rate_is_limited(hardware):
    plant = ServoPlant(hardware.servo("mg90s"))
    plant.command(math.radians(90))
    speeds = []
    for _ in range(200):
        state = plant.step(0.001)
        speeds.append(abs(state.output_speed))
    limit = hardware.servo("mg90s").no_load_speed * hardware.servo("mg90s").voltage_scale(5.0)
    assert max(speeds) <= limit * 1.2


def test_flight_computer_produces_control_frames(hardware):
    imu = IMUModel(hardware.sensor("mpu6050"))
    plant = ServoPlant(hardware.servo("mg90s"))
    computer = FlightComputer(hardware.board("pico2w"), ControlGains(), imu,
                              [ServoChannel("pitch", plant), ServoChannel("yaw", plant)])
    computer.arm()
    frames = 0
    for i in range(500):
        sample = imu.measure(Transform.identity(), vec3(), vec3(0, 0, 9.81), 0.001)
        if sample is None:
            continue
        frame = computer.update(sample, 0.001)
        if frame is not None:
            frames += 1
    assert frames >= 100   # 250 Hz loop over 0.5 s
    assert computer.timing()["loops"] >= 100


def test_hil_loopback_round_trip():
    link = LoopbackHIL(lambda accel, gyro, phase: (0.01, -0.02))
    for i in range(10):
        reply = link.exchange(i * 0.004, (0, 0, 9.81), (0, 0, 0))
        assert reply is not None
    report = link.report()
    assert report["stats"]["sent"] == 10
    assert report["stats"]["received"] == 10
