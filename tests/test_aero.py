import math

import numpy as np
import pytest

from tvc_studio.physics.aero import (
    AeroConfig, AeroModelKind, StackElement, barrowman_cp, drag_equation,
)


def test_barrowman_cp_between_nose_and_tail():
    stack = [
        StackElement("nose", start=0.0, length=0.1, diameter=0.04),
        StackElement("body", start=0.1, length=0.4, diameter=0.04),
        StackElement("fin", start=0.45, length=0.05, diameter=0.04, fin_count=4,
                     fin_root_chord=0.05, fin_tip_chord=0.03, fin_span=0.04),
    ]
    result = barrowman_cp(stack)
    assert 0.0 < result.cp_position < 0.5
    assert result.cn_alpha > 0
    assert result.contributions


def test_fins_move_cp_aft():
    base = [
        StackElement("nose", start=0.0, length=0.1, diameter=0.04),
        StackElement("body", start=0.1, length=0.5, diameter=0.04),
    ]
    small = barrowman_cp(base + [StackElement(
        "fin", start=0.5, length=0.05, diameter=0.04, fin_count=4,
        fin_root_chord=0.05, fin_tip_chord=0.03, fin_span=0.02)])
    large = barrowman_cp(base + [StackElement(
        "fin", start=0.5, length=0.05, diameter=0.04, fin_count=4,
        fin_root_chord=0.05, fin_tip_chord=0.03, fin_span=0.08)])
    # more fin area pulls the overall CP towards the fin CP (aft of the nose)
    assert large.cp_position > small.cp_position


def test_drag_equation():
    force = drag_equation(1.225, 10.0, 0.5, 0.01)
    assert force == pytest.approx(0.5 * 1.225 * 100 * 0.5 * 0.01)


def test_static_margin_sign_convention():
    config = AeroConfig(model=AeroModelKind.COEFFICIENTS, cp_position=0.1,
                        cn_alpha=2.0, reference_area=0.001, reference_length=0.445)
    from tvc_studio.physics.aero import CoefficientAero
    model = CoefficientAero(config)
    velocity = np.array([0.0, 2.0, 50.0])
    from tvc_studio.core.math3d import Quat
    result = model.forces(velocity, Quat.from_axis_angle((0, 1, 0), math.radians(3.0)),
                          0.0, 1.225, 340.0, com_position=0.2)
    assert result.static_margin == pytest.approx(0.2 - 0.1)
    # a positive pitch tilt produces a negative (restoring) pitching moment
    assert result.moment_body[1] < 0
