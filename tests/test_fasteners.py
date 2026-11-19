import math

import pytest

from tvc_studio.components.materials import MaterialLibrary
from tvc_studio.fasteners.catalog import FastenerCatalog, FastenerKind, metric_thread
from tvc_studio.fasteners.install import HoleReference, InstalledFastener, ScrewInstaller
from tvc_studio.fasteners.thread import (
    EngagementMode, ThreadEngagementModel, clamp_force_from_torque, torque_from_clamp_force,
)


def test_catalogue_contains_standard_fasteners():
    catalog = FastenerCatalog(materials=MaterialLibrary())
    assert len(catalog.all()) > 10
    m3 = [s for s in catalog.all() if s.thread.designation == "M3"]
    assert m3
    assert any(s.kind is FastenerKind.MACHINE_SCREW for s in m3)
    assert any(s.is_thread_forming() for s in catalog.all())


def test_thread_engagement_increases_with_depth():
    thread = metric_thread("M3")
    model = ThreadEngagementModel()
    shallow = model.analyse(thread, 0.003, EngagementMode.PRE_THREADED, 60e6)
    deep = model.analyse(thread, 0.009, EngagementMode.PRE_THREADED, 60e6)
    assert deep.engagement_length > shallow.engagement_length
    assert deep.strip_torque_estimate > shallow.strip_torque_estimate


def test_clamp_force_torque_round_trip():
    thread = metric_thread("M3")
    torque = torque_from_clamp_force(thread, 350.0, friction=0.15)
    force = clamp_force_from_torque(thread, torque, friction=0.15)
    assert force == pytest.approx(350.0, rel=1e-6)


def test_pre_threaded_and_forming_modes():
    installer = ScrewInstaller()
    spec = FastenerCatalog(materials=MaterialLibrary()).all()[0]
    threaded = HoleReference(id="h1", component_id="c1", position=(0, 0, 0), axis=(0, 0, 1),
                             diameter=spec.thread.major_diameter, depth=0.008, pre_threaded=True,
                             material_id="pla", label="threaded")
    assert installer.mode_for(spec, threaded) is EngagementMode.PRE_THREADED
    clearance = HoleReference(id="h2", component_id="c1", position=(0, 0, 0), axis=(0, 0, 1),
                              diameter=0.0032, depth=0.008, pre_threaded=False,
                              material_id="pla", label="clearance")
    forming = installer.mode_for(spec, clearance)
    assert forming in (EngagementMode.THREAD_FORMING, EngagementMode.PRE_THREADED)


def test_install_depth_is_clamped_to_hole():
    catalog = FastenerCatalog(materials=MaterialLibrary())
    spec = next(s for s in catalog.all() if s.thread.designation == "M3")
    installer = ScrewInstaller()
    hole = HoleReference(id="h", component_id="c", position=(0, 0, 0), axis=(0, 0, 1),
                         diameter=0.0025, depth=0.004, pre_threaded=True, material_id="pla",
                         label="short hole")
    state = installer.install(installer.start(spec, hole), spec, hole)
    assert state.depth <= hole.depth + 1e-12
    assert state.installed
    assert state.engagement(spec, hole) <= hole.depth + 1e-12
