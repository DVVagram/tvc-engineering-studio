"""The physical configuration is frozen and internally consistent."""

from __future__ import annotations

import dataclasses
import math

import pytest

from rocket_sim.config import (
    DRY_MASS,
    G0,
    INERTIA,
    LAUNCH_MASS,
    MOTOR,
    ROCKET,
    RocketConfig,
    air_density,
    speed_of_sound,
)


def test_rocket_and_motor_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        ROCKET.body_length = 2.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        MOTOR.casing_mass = 0.2  # type: ignore[misc]


def test_geometry_is_plausible_and_stable() -> None:
    assert ROCKET.body_length > 0.5
    assert ROCKET.body_diameter > 0.0
    assert ROCKET.nose_length < ROCKET.body_length
    assert ROCKET.fin_count == 4
    assert ROCKET.static_margin > 0.05
    assert ROCKET.static_margin_calibers > 1.0
    assert ROCKET.reference_area == pytest.approx(math.pi * ROCKET.body_radius ** 2)


def test_masses_come_only_from_the_fixed_configuration() -> None:
    assert LAUNCH_MASS == pytest.approx(ROCKET.structural_mass + MOTOR.initial_mass)
    assert DRY_MASS == pytest.approx(ROCKET.structural_mass + MOTOR.casing_mass)
    assert MOTOR.initial_mass == pytest.approx(MOTOR.casing_mass + MOTOR.propellant_mass)
    assert INERTIA > 0.02


def test_motor_config_has_no_performance_fields() -> None:
    names = {field.name for field in dataclasses.fields(type(MOTOR))}
    for forbidden in ("thrust", "average_thrust", "maximum_thrust", "impulse", "burn_time"):
        assert forbidden not in names


def test_atmosphere_falls_off_with_altitude() -> None:
    assert air_density(0) == pytest.approx(1.225)
    assert air_density(5000) < air_density(0)
    assert air_density(-5) == pytest.approx(air_density(0))
    assert speed_of_sound(0) > 300


def test_rocket_config_is_constructed_once() -> None:
    # A second instance with the defaults must describe the same rocket.
    twin = RocketConfig()
    assert twin == ROCKET
    assert ROCKET.launch_tilt_radians == pytest.approx(math.radians(4.0))
    assert G0 > 9.7
