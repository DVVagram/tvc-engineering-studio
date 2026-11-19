"""Motor performance validation and the estimated thrust curves."""

from __future__ import annotations

import pytest

from rocket_sim.config import MOTOR
from rocket_sim.motor import (
    PRESETS,
    SHAPES,
    MotorPerformance,
    MotorSpecError,
    ThrustCurve,
)


def test_default_curve_matches_its_specification() -> None:
    performance = MotorPerformance()
    curve = performance.curve()
    assert isinstance(curve, ThrustCurve)
    assert curve.integrated_impulse() == pytest.approx(performance.total_impulse, rel=1e-6)
    assert curve.peak_thrust == pytest.approx(performance.maximum_thrust, rel=0.02)
    assert curve.burn_time == pytest.approx(performance.burn_time)
    assert curve.measured is False
    assert "estimated" in curve.source_label


@pytest.mark.parametrize("shape", SHAPES)
def test_every_shape_burns_the_stated_impulse(shape: str) -> None:
    average = 30.0
    burn = 2.0
    performance = MotorPerformance("shape test", average, 40.0, average * burn, burn, shape)
    if shape == "neutral":
        performance = MotorPerformance("shape test", average, average, average * burn,
                                       burn, shape)
    curve = performance.curve()
    assert curve.integrated_impulse() == pytest.approx(average * burn, rel=1e-6)
    if shape == "progressive":
        assert curve.thrusts[0] < curve.thrusts[-1]
    elif shape == "regressive":
        assert curve.thrusts[0] > curve.thrusts[-1]
    else:
        assert curve.thrusts.min() == pytest.approx(curve.thrusts.max())


def test_thrust_is_zero_outside_the_burn() -> None:
    curve = MotorPerformance().curve()
    assert curve.thrust_at(-0.1) == 0.0
    assert curve.thrust_at(curve.burn_time + 0.1) == 0.0
    assert curve.thrust_at(curve.burn_time / 2) > 0.0


@pytest.mark.parametrize(
    "performance, fragment",
    [
        (MotorPerformance("a", 40.0, 44.0, 54.9, 1.8, "progressive"), "Average thrust"),
        (MotorPerformance("b", 30.5, 90.0, 54.9, 1.8, "progressive"), "peak"),
        (MotorPerformance("c", 30.5, 44.0, 54.9, 1.8, "neutral"), "neutral"),
        (MotorPerformance("d", 30.5, 20.0, 54.9, 1.8, "progressive"), "Maximum thrust"),
        (MotorPerformance("e", 30.5, 44.0, 54.9, 0.0, "progressive"), "Burn time"),
        (MotorPerformance("f", 30.5, 44.0, -1.0, 1.8, "progressive"), "Total impulse"),
        (MotorPerformance("g", 30.5, 44.0, 54.9, 1.8, "wobbly"), "shape"),
    ],
)
def test_inconsistent_specifications_are_rejected(performance: MotorPerformance,
                                                  fragment: str) -> None:
    with pytest.raises(MotorSpecError) as raised:
        performance.validate()
    assert any(fragment.lower() in message.lower() for message in raised.value.errors)


def test_validation_never_rewrites_the_inputs() -> None:
    performance = MotorPerformance("a", 40.0, 44.0, 54.9, 1.8, "progressive")
    with pytest.raises(MotorSpecError):
        performance.validate()
    # The object still carries exactly what the user typed.
    assert performance.average_thrust == 40.0
    assert performance.total_impulse == 54.9


def test_explicit_impulse_fit_resolves_a_mismatch() -> None:
    performance = MotorPerformance("a", 31.0, 44.0, 50.0, 1.8, "progressive")
    fitted = performance.with_impulse_fitted()
    fitted.validate()
    assert fitted.total_impulse == pytest.approx(31.0 * 1.8, rel=1e-6)
    assert fitted.average_thrust == performance.average_thrust
    assert fitted.burn_time == performance.burn_time


def test_performance_changes_do_not_touch_the_motor_hardware() -> None:
    before = (MOTOR.diameter, MOTOR.length, MOTOR.casing_mass, MOTOR.propellant_mass)
    curve = PRESETS["SR-90 — high power"].curve()
    assert curve.total_impulse > PRESETS["SR-30 — low power"].curve().total_impulse
    after = (MOTOR.diameter, MOTOR.length, MOTOR.casing_mass, MOTOR.propellant_mass)
    assert before == after


def test_presets_are_valid() -> None:
    for name, preset in PRESETS.items():
        preset.validate()
        assert preset.thrust_to_weight > 1.0, name
