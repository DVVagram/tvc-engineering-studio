import math

import numpy as np
import pytest

from tvc_studio.assembly.massprops import AssemblyMassCalculator
from tvc_studio.core.math3d import Quat
from tvc_studio.physics.atmosphere import Atmosphere
from tvc_studio.physics.flight import FlightSimulator
from tvc_studio.physics.propulsion import MotorDefinition
from tvc_studio.physics.rocket import build_rocket


def test_isa_sea_level():
    atm = Atmosphere().state(0.0)
    assert atm.temperature == pytest.approx(288.15, rel=1e-6)
    assert atm.pressure == pytest.approx(101325.0, rel=1e-3)
    assert atm.density == pytest.approx(1.225, rel=1e-3)


def test_motor_eng_round_trip():
    text = """
; test motor
TEST 24 70 3 5 1.0 0.045 0.030
0 0
0.05 15
0.6 24
1.2 0
"""
    motor = MotorDefinition.from_eng(text, name="TEST")
    assert motor.total_impulse > 0
    assert motor.burn_time == pytest.approx(1.2, rel=0.05)
    assert motor.propellant_mass > 0
    reimported = MotorDefinition.from_eng(motor.to_eng())
    assert reimported.total_impulse == pytest.approx(motor.total_impulse, rel=0.02)


def test_demo_motor_is_labelled_estimated(sample_project):
    motor = sample_project.motor()
    assert motor.provenance.value in ("estimated", "manufacturer")


def test_rocket_static_margin_is_positive(sample_project):
    calc = AssemblyMassCalculator(sample_project.parts, sample_project.materials,
                                  sample_project.fasteners)
    rocket = build_rocket(sample_project.assembly, calc, sample_project.parts,
                          sample_project.motor(), sample_project.hardware, sample_project.aero)
    margin = rocket.properties.static_margin(0.0)
    assert margin > 0.005
    assert rocket.properties.cp_position < float(rocket.properties.com_dry[2])


def test_passive_rocket_is_stable(sample_project):
    from dataclasses import replace

    from tvc_studio.physics.propulsion import ThrustPoint

    calc = AssemblyMassCalculator(sample_project.parts, sample_project.materials,
                                  sample_project.fasteners)
    rocket = build_rocket(sample_project.assembly, calc, sample_project.parts,
                          sample_project.motor(), sample_project.hardware, sample_project.aero)
    coast = replace(sample_project.motor(), name="coast", designation="COAST",
                    propellant_mass=0.0, thrust_curve=[ThrustPoint(0.0, 0.0),
                                                       ThrustPoint(1.0, 0.0)])
    config = replace(sample_project.flight, tvc_enabled=False, rail_length=0.0,
                     wind=(0.0, 0.0, 0.0))
    sim = FlightSimulator(rocket.properties, coast, sample_project.aero, config, rocket.stack)
    sim.state.position = np.array([0.0, 0.0, 50.0])
    sim.state.velocity = np.array([0.0, 0.0, 40.0])
    sim.state.attitude = Quat.from_axis_angle((0, 1, 0), math.radians(2.0))
    sim.state.on_rail = False
    start = math.degrees(sim.state.tilt_angle())
    for _ in range(400):
        sim.step(0.001)
    end = math.degrees(sim.state.tilt_angle())
    assert end < start - 0.2


def test_guided_flight_reaches_plausible_apogee(sample_project):
    calc = AssemblyMassCalculator(sample_project.parts, sample_project.materials,
                                  sample_project.fasteners)
    rocket = build_rocket(sample_project.assembly, calc, sample_project.parts,
                          sample_project.motor(), sample_project.hardware, sample_project.aero)
    from dataclasses import replace
    config = replace(sample_project.flight, tvc_enabled=True)
    sim = FlightSimulator(rocket.properties, sample_project.motor(), sample_project.aero,
                          config, rocket.stack, control=sample_project.control)
    result = sim.run(20.0)
    assert 120.0 < result.summary["apogee_m"] < 400.0
    assert result.summary["max_tilt_deg"] < 45.0
    assert result.summary["landed"] in (True, False)
