"""Flight model: events, repeatability, reset and sensitivity to motor performance."""

from __future__ import annotations

import numpy as np
import pytest

from rocket_sim.config import DRY_MASS, G0, INERTIA, LAUNCH_MASS, ROCKET
from rocket_sim.motor import PRESETS, MotorPerformance
from rocket_sim.physics import Simulation, quat_matrix


def _fly(performance: MotorPerformance | None = None) -> Simulation:
    return Simulation(performance).run_to_end()


def test_standard_flight_reaches_apogee_and_lands() -> None:
    simulation = _fly()
    names = [event.name for event in simulation.events]
    assert names[0] == "Liftoff"
    assert "Motor burnout" in names
    assert "Apogee" in names
    assert names[-1] == "Ground impact"
    assert simulation.status == "Landed"
    assert simulation.max_altitude > 100.0
    assert simulation.state.time > 5.0


def test_events_happen_in_order() -> None:
    simulation = _fly()
    times = [event.time for event in simulation.events]
    assert times == sorted(times)
    burnout = next(event for event in simulation.events if event.name == "Motor burnout")
    apogee = next(event for event in simulation.events if event.name == "Apogee")
    impact = next(event for event in simulation.events if event.name == "Ground impact")
    assert burnout.time == pytest.approx(1.8, abs=0.01)
    assert burnout.time < apogee.time < impact.time


def test_ground_impact_ends_at_zero_altitude() -> None:
    simulation = _fly()
    assert simulation.altitude == pytest.approx(0.0, abs=1e-6)
    assert simulation.status == "Landed"
    assert simulation.state.speed == 0.0
    assert simulation.events[-1].name == "Ground impact"


def test_simulation_is_repeatable() -> None:
    first = _fly()
    second = _fly()
    assert first.max_altitude == pytest.approx(second.max_altitude, abs=1e-9)
    assert first.max_speed == pytest.approx(second.max_speed, abs=1e-9)
    assert first.state.time == pytest.approx(second.state.time, abs=1e-9)


def test_reset_restores_the_exact_initial_conditions() -> None:
    simulation = Simulation()
    initial_position = simulation.state.position.copy()
    initial_attitude = simulation.state.attitude.copy()
    simulation.advance(1.0)
    assert simulation.state.time > 0.0
    simulation.reset()
    assert simulation.state.time == 0.0
    assert np.array_equal(simulation.state.position, initial_position)
    assert np.array_equal(simulation.state.attitude, initial_attitude)
    assert simulation.state.velocity.tolist() == [0.0, 0.0, 0.0]
    assert simulation.events == []
    assert simulation.max_altitude == 0.0
    assert not simulation.finished


def test_mass_burns_down_to_the_dry_mass() -> None:
    simulation = Simulation()
    assert simulation.state.mass == pytest.approx(LAUNCH_MASS)
    simulation.advance(1.0)
    assert simulation.state.mass < LAUNCH_MASS
    simulation.advance(2.0)
    assert simulation.state.mass == pytest.approx(DRY_MASS)


def test_thrust_stops_at_burnout() -> None:
    simulation = Simulation()
    simulation.advance(1.0)
    assert simulation.thrust > 0.0
    simulation.advance(1.0)
    assert simulation.thrust == 0.0
    assert simulation.motor_status == "Burnout"


def test_more_impulse_gives_a_higher_flight() -> None:
    low = _fly(PRESETS["SR-30 — low power"])
    standard = _fly(PRESETS["SR-55 — standard"])
    high = _fly(PRESETS["SR-90 — high power"])
    assert low.max_altitude < standard.max_altitude < high.max_altitude
    assert standard.max_altitude - low.max_altitude > 50.0


def test_fins_keep_the_body_aligned_with_the_airflow() -> None:
    simulation = Simulation()
    simulation.advance(2.5)  # just after burnout
    axis = quat_matrix(simulation.state.attitude)[:, 0]
    velocity = simulation.state.velocity
    cosine = float(np.dot(axis, velocity / np.linalg.norm(velocity)))
    assert cosine > 0.97  # angle of attack below ~14 degrees


def test_a_motor_too_weak_to_lift_reports_no_liftoff() -> None:
    weak = MotorPerformance("weak", 1.0, 1.0, 1.8, 1.8, "neutral")
    simulation = _fly(weak)
    assert simulation.status == "No liftoff"
    assert simulation.events[-1].name == "No liftoff"
    assert simulation.max_altitude == 0.0


def test_unknown_control_systems_are_absent() -> None:
    # The flight model has no control input at all: nothing can steer the rocket.
    simulation = Simulation()
    assert not hasattr(simulation, "control")
    assert not hasattr(simulation, "pid")
    assert simulation.performance.measured is False
    assert simulation.rocket == ROCKET
    assert INERTIA > 0.0
    assert G0 > 0.0
