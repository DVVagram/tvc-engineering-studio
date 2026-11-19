"""Law-level checks of the flight model: gravity, drag, moments, mass and integration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rocket_sim.config import (
    DRY_MASS,
    G0,
    MOTOR,
    ROCKET,
    air_density,
    speed_of_sound,
)
from rocket_sim.motor import MotorPerformance
from rocket_sim.physics import Simulation, quat_from_two_vectors, quat_matrix, quat_normalize


def _state_vector(attitude: np.ndarray, velocity: np.ndarray, altitude: float = 100.0):
    position = np.array([0.0, 0.0, altitude])
    return np.concatenate([position, velocity, attitude, np.zeros(3)])


def test_a_stationary_rocket_after_burnout_accelerates_only_under_gravity() -> None:
    simulation = Simulation()
    derivatives = simulation._derivatives(
        _state_vector(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)), 10.0)
    assert derivatives[3:6] == pytest.approx([0.0, 0.0, -G0], abs=1e-12)


def test_drag_and_fin_side_force_follow_the_documented_coefficients() -> None:
    simulation = Simulation()
    speed = 40.0
    derivatives = simulation._derivatives(
        _state_vector(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, -speed])), 10.0)
    altitude = 100.0 - ROCKET.tail_offset
    dynamic_pressure = 0.5 * air_density(altitude) * speed ** 2
    drag = dynamic_pressure * ROCKET.drag_coefficient * ROCKET.reference_area
    # 90 degrees angle of attack: the slender-body slope vanishes, crossflow remains
    fin = dynamic_pressure * ROCKET.reference_area * ROCKET.fin_crossflow_drag
    # velocity points down, so both drag and the fin side force push up
    assert derivatives[5] == pytest.approx((drag + fin) / DRY_MASS - G0, rel=1e-9)
    assert derivatives[3] == pytest.approx(0.0, abs=1e-12)


def test_fins_produce_a_restoring_moment_towards_the_airflow() -> None:
    tilt = math.radians(12.0)
    attitude = quat_from_two_vectors([1.0, 0.0, 0.0],
                                     [math.sin(tilt), 0.0, math.cos(tilt)])
    derivatives = Simulation()._derivatives(
        _state_vector(attitude, np.array([0.0, 0.0, 60.0])), 10.0)
    # Rotating the nose from +X towards +Z is a rotation about -Y.
    assert derivatives[11] < 0.0


def test_propellant_is_burned_in_proportion_to_delivered_impulse() -> None:
    simulation = Simulation()
    curve = simulation.curve
    for target in (0.3, 0.9, 1.5):
        simulation.advance(target - simulation.time)
        fraction = curve.impulse_delivered(target) / curve.total_impulse
        expected = (ROCKET.structural_mass + MOTOR.casing_mass
                    + MOTOR.propellant_mass * (1.0 - fraction))
        assert simulation.state.mass == pytest.approx(expected, rel=1e-9)
    assert curve.impulse_delivered(curve.burn_time) == pytest.approx(curve.total_impulse)
    simulation.advance(2.0)
    assert simulation.state.mass == pytest.approx(DRY_MASS)


def test_mechanical_energy_never_increases_during_coast() -> None:
    simulation = Simulation()
    simulation.advance(2.0)  # past burnout
    energies = []
    while simulation.speed > 1.0 and not simulation.finished:
        energy = (0.5 * simulation.state.mass * simulation.speed ** 2
                  + simulation.state.mass * G0 * simulation.state.position[2])
        energies.append(energy)
        simulation.advance(0.05)
    assert len(energies) > 20
    assert max(np.diff(energies)) <= 1e-6


def test_attitude_stays_a_unit_quaternion_and_aerodynamics_do_not_spin_it_up() -> None:
    simulation = Simulation()
    simulation.run_to_end()
    assert np.linalg.norm(simulation.state.attitude) == pytest.approx(1.0, abs=1e-12)
    rates = []
    replay = Simulation()
    while not replay.finished:
        replay.advance(0.25)
        rates.append(float(np.linalg.norm(replay.state.angular_velocity)))
    assert max(rates) < math.pi  # no tumbling blow-up


def test_timestep_refinement_changes_the_apogee_barely() -> None:
    coarse = Simulation(dt=0.002).run_to_end()
    fine = Simulation(dt=0.0005).run_to_end()
    assert abs(coarse.max_altitude - fine.max_altitude) < 0.25
    assert abs(coarse.state.time - fine.state.time) < 0.05


def test_atmosphere_and_sound_speed_are_physical() -> None:
    assert air_density(0.0) == pytest.approx(1.225)
    assert air_density(8848.0) == pytest.approx(1.225 * math.exp(-8848.0 / 8500.0))
    assert 300.0 < speed_of_sound(0.0) < 350.0
    assert speed_of_sound(11000.0) < speed_of_sound(0.0)


def test_quaternion_helpers_are_consistent() -> None:
    attitude = quat_normalize(np.array([0.3, -0.4, 0.5, 0.7]))
    matrix = quat_matrix(attitude)
    assert matrix @ matrix.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_curve_shapes_deliver_impulse_at_different_rates() -> None:
    base = dict(average_thrust=30.0, maximum_thrust=40.0, total_impulse=60.0,
                burn_time=2.0)
    progressive = MotorPerformance("p", curve_shape="progressive", **base).curve()
    regressive = MotorPerformance("r", curve_shape="regressive", **base).curve()
    assert progressive.impulse_delivered(0.5) < regressive.impulse_delivered(0.5)
    assert progressive.impulse_delivered(2.0) == pytest.approx(60.0)
