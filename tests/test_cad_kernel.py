import math
import os

import numpy as np
import pytest

from tvc_studio.core.interfaces import TessellationQuality


def test_box_volume_and_inertia(backend):
    shape = backend.box(0.02, 0.03, 0.04)
    mp = backend.mass_properties(shape, 1000.0)
    assert mp.mass == pytest.approx(0.02 * 0.03 * 0.04 * 1000.0, rel=1e-9)
    assert mp.inertia[0, 0] == pytest.approx(mp.mass / 12 * (0.03**2 + 0.04**2), rel=1e-6)


def test_boolean_cut_volume(backend):
    base = backend.box(0.04, 0.04, 0.04)
    tool = backend.cylinder(0.005, 0.08, origin=(0.02, 0.02, 0.0))
    cut = backend.boolean_cut(base, tool)
    volume = backend.mass_properties(cut, 1.0).mass
    expected = 0.04**3 - math.pi * 0.005**2 * 0.04
    assert volume == pytest.approx(expected, rel=1e-6)


def test_fillet_and_chamfer(backend):
    base = backend.box(0.03, 0.02, 0.01)
    all_edges = list(range(12))
    filleted = backend.fillet(base, all_edges, 0.002)
    chamfered = backend.chamfer(base, all_edges, 0.001)
    v0 = backend.mass_properties(base, 1.0).mass
    vf = backend.mass_properties(filleted, 1.0).mass
    vc = backend.mass_properties(chamfered, 1.0).mass
    assert 0 < vf < v0
    assert 0 < vc < v0


def test_extrude_profile(backend):
    from tvc_studio.cad.sketch import ProfileSegment
    profile = [
        ProfileSegment("line", (0.0, 0.0), (0.05, 0.0)),
        ProfileSegment("line", (0.05, 0.0), (0.05, 0.02)),
        ProfileSegment("line", (0.05, 0.02), (0.0, 0.02)),
        ProfileSegment("line", (0.0, 0.02), (0.0, 0.0)),
    ]
    shape = backend.extrude_profile([profile], 0.01)
    volume = backend.mass_properties(shape, 1.0).mass
    assert volume == pytest.approx(0.05 * 0.02 * 0.01, rel=1e-9)


def test_tessellation_is_non_empty(backend):
    mesh = backend.tessellate(backend.cylinder(0.01, 0.03), TessellationQuality.NORMAL)
    assert not mesh.is_empty
    assert mesh.triangle_count > 20
    lo, hi = mesh.bounds()
    assert hi[2] - lo[2] == pytest.approx(0.03, rel=1e-3)


def test_step_round_trip(tmp_path, backend):
    shape = backend.box(0.02, 0.03, 0.04)
    path = str(tmp_path / "part.step")
    backend.export_step(shape, path)
    imported = backend.import_step(path)[0]
    v0 = backend.mass_properties(shape, 1.0).mass
    v1 = backend.mass_properties(imported, 1.0).mass
    assert v1 == pytest.approx(v0, rel=1e-6)


def test_iges_round_trip(tmp_path, backend):
    shape = backend.cylinder(0.01, 0.03)
    path = str(tmp_path / "part.iges")
    backend.export_iges(shape, path)
    imported = backend.import_iges(path)[0]
    v0 = backend.mass_properties(shape, 1.0).mass
    v1 = backend.mass_properties(imported, 1.0).mass
    assert v1 == pytest.approx(v0, rel=1e-5)


def test_stl_round_trip(tmp_path, backend):
    shape = backend.box(0.02, 0.02, 0.02)
    path = str(tmp_path / "part.stl")
    backend.export_stl(shape, path)
    compound = backend.import_stl(path)[0]
    volume = backend.mass_properties(compound, 1.0).mass
    assert volume == pytest.approx(8e-6, rel=0.05)
