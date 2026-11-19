import math

import pytest

from tvc_studio.cad.document import PartRepository
from tvc_studio.cad.features import (
    BoxFeature, ConeFeature, CylinderFeature, HoleFeature, PatternPolarFeature, TubeFeature,
)


def test_part_rebuild_and_holes(backend):
    repo = PartRepository(backend=backend)
    part = repo.create("plate", [
        BoxFeature(0.05, 0.05, 0.005, name="plate"),
        HoleFeature(position=(0.02, 0.02, 0.005), axis=(0, 0, -1), diameter=0.0032,
                    depth=0.005, through=True, name="M3 clearance"),
    ], material_id="pla")
    result = part.rebuild(backend)
    assert result.success, result.diagnostics
    assert len(part.holes()) == 1
    volume = part.cad_mass_properties(backend, 1.0).mass
    expected = 0.05 * 0.05 * 0.005 - math.pi * 0.0016**2 * 0.005
    assert volume == pytest.approx(expected, rel=1e-5)


def test_tube_and_cone_volumes(backend):
    repo = PartRepository(backend=backend)
    tube = repo.create("tube", [TubeFeature(inner_radius=0.01, outer_radius=0.012, height=0.03)])
    tube.rebuild(backend)
    volume = tube.cad_mass_properties(backend, 1.0).mass
    assert volume == pytest.approx(math.pi * (0.012**2 - 0.01**2) * 0.03, rel=1e-6)

    cone = repo.create("cone", [ConeFeature(radius1=0.02, radius2=0.005, height=0.04)])
    cone.rebuild(backend)
    volume = cone.cad_mass_properties(backend, 1.0).mass
    expected = math.pi / 3 * 0.04 * (0.02**2 + 0.02 * 0.005 + 0.005**2)
    assert volume == pytest.approx(expected, rel=1e-6)


def test_polar_pattern_four_fins(backend):
    repo = PartRepository(backend=backend)
    part = repo.create("fins", [
        BoxFeature(0.02, 0.002, 0.03, origin=(0.01, -0.001, 0.0), name="fin"),
        PatternPolarFeature(count=4, axis_origin=(0, 0, 0.015), axis_dir=(0, 0, 1), name="pattern"),
    ])
    part.rebuild(backend)
    volume = part.cad_mass_properties(backend, 1.0).mass
    assert volume == pytest.approx(4 * 0.02 * 0.002 * 0.03, rel=1e-6)


def test_failed_feature_is_reported_not_hidden(backend):
    repo = PartRepository(backend=backend)
    part = repo.create("bad", [BoxFeature(0.01, 0.01, 0.01, operation="cut")])
    result = part.rebuild(backend)
    assert not result.success
    assert result.diagnostics
