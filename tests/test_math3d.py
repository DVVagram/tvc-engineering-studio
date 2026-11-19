import math

import numpy as np
import pytest

from tvc_studio.core.math3d import (
    MassProperties, Quat, Transform, combine_mass_properties, inertia_box, parallel_axis, vec3,
)


def test_quat_rotates_z_to_x_about_y():
    q = Quat.from_axis_angle((0, 1, 0), math.pi / 2)
    result = q.rotate(vec3(0, 0, 1))
    assert np.allclose(result, vec3(1, 0, 0), atol=1e-12)


def test_quat_inverse_and_compose():
    q = Quat.from_euler(0.3, -0.2, 0.7)
    assert np.allclose((q * q.inverse()).matrix(), np.eye(3), atol=1e-12)


def test_euler_round_trip():
    yaw, pitch, roll = 0.4, -0.3, 0.15
    q = Quat.from_euler(yaw, pitch, roll)
    y2, p2, r2 = q.to_euler()
    assert (y2, p2, r2) == pytest.approx((yaw, pitch, roll), abs=1e-9)


def test_transform_inverse_identity():
    t = Transform.from_pos_euler((0.1, -0.2, 0.3), 0.2, -0.4, 0.1)
    p = vec3(0.05, 0.02, -0.01)
    assert np.allclose(t.inverse().apply_point(t.apply_point(p)), p, atol=1e-12)


def test_parallel_axis_shift():
    inertia = np.diag([1e-4, 2e-4, 3e-4])
    shifted = parallel_axis(inertia, 2.0, (0.5, 0, 0))
    # Ixx stays, Iyy and Izz gain m*d^2
    assert shifted[1, 1] == pytest.approx(2e-4 + 2.0 * 0.25)
    assert shifted[0, 0] == pytest.approx(1e-4)


def test_combine_mass_properties_centre():
    a = MassProperties(mass=1.0, com=vec3(0, 0, 0), inertia=np.eye(3) * 0.01)
    b = MassProperties(mass=1.0, com=vec3(0.2, 0, 0), inertia=np.eye(3) * 0.01)
    combined = combine_mass_properties([a, b])
    assert combined.mass == pytest.approx(2.0)
    assert np.allclose(combined.com, vec3(0.1, 0, 0), atol=1e-12)


def test_inertia_box_analytic():
    mass, sx, sy, sz = 0.5, 0.1, 0.2, 0.3
    inertia = inertia_box(mass, sx, sy, sz)
    assert inertia[0, 0] == pytest.approx(mass / 12 * (sy**2 + sz**2))
    assert inertia[2, 2] == pytest.approx(mass / 12 * (sx**2 + sy**2))
