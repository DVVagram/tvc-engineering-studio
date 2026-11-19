import math

import numpy as np
import pytest

from tvc_studio.assembly.massprops import AssemblyMassCalculator
from tvc_studio.core.math3d import Transform
from tvc_studio.mech.build import build_world
from tvc_studio.mech.contact import ConvexProxy, sat_contact
from tvc_studio.mech.dynamics import Body, DynamicsWorld, Joint, JointKind


def test_isolated_revolute_joint_rotates_freely():
    world = DynamicsWorld("hinge")
    world.contacts_enabled = False
    world.gravity_enabled = False
    a = Body("a", "A", 1.0, np.diag([1e-3] * 3), position=(0, 0, 0), fixed=True)
    b = Body("b", "B", 0.02, np.diag([4.8e-6] * 3), position=(0, 0, 0))
    world.add_body(a)
    world.add_body(b)
    world.add_joint(Joint("j", "hinge", JointKind.REVOLUTE, "a", "b", anchor_a=(0, 0, 0),
                          anchor_b=(0, 0, 0), axis_a=(0, 1, 0), axis_b=(0, 1, 0)))
    b.angular_velocity = np.array([0.0, 1.0, 0.0])
    for _ in range(100):
        world.step(0.001)
    angle = world.joints["j"].coordinate(world.bodies)
    assert angle == pytest.approx(math.radians(5.4), rel=0.08)


def test_fixed_body_ignores_gravity():
    world = DynamicsWorld("static")
    world.gravity_enabled = True
    body = Body("b", "fixed", 1.0, np.eye(3) * 0.01, position=(0, 0, 1.0), fixed=True)
    world.add_body(body)
    for _ in range(100):
        world.step(0.001)
    assert np.allclose(body.position, (0, 0, 1.0))


def test_servo_hinge_tracks_command(sample_project):
    calc = AssemblyMassCalculator(sample_project.parts, sample_project.materials,
                                  sample_project.fasteners)
    build = build_world(sample_project.assembly, calc, sample_project.parts,
                        sample_project.hardware, backend=sample_project.parts.backend)
    world = build.world
    joint = world.joints["hinge_pitch"]
    for _ in range(900):
        world.set_joint_target("hinge_pitch", math.radians(5.0))
        world.step(0.001)
    angle = math.degrees(joint.coordinate(world.bodies))
    # backlash + deadband account for the small steady-state offset
    assert 3.8 < angle < 5.2
    assert abs(math.degrees(joint.coordinate(world.bodies))) < 10.0
    assert len(build.servos) == 2


def test_contact_detection_overlapping_boxes(backend):
    from tvc_studio.core.interfaces import TessellationQuality
    mesh = backend.tessellate(backend.box(0.02, 0.02, 0.02), TessellationQuality.COARSE)
    a = ConvexProxy.from_mesh(mesh)
    b = ConvexProxy.from_mesh(mesh)
    overlapping = sat_contact(a, Transform.from_pos_euler((0, 0, 0)), b,
                             Transform.from_pos_euler((0.005, 0, 0)))
    separated = sat_contact(a, Transform.from_pos_euler((0, 0, 0)), b,
                           Transform.from_pos_euler((0.1, 0, 0)))
    assert overlapping is not None
    assert separated is None
