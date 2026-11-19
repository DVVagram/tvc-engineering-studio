import math

import numpy as np
import pytest

from tvc_studio.assembly.constraints import AssemblyConstraint, AssemblySolver, ConstraintType, EntityKind, EntityRef
from tvc_studio.assembly.massprops import AssemblyMassCalculator
from tvc_studio.assembly.model import Assembly
from tvc_studio.cad.features import BoxFeature
from tvc_studio.components.materials import MaterialLibrary
from tvc_studio.components.properties import MassMode
from tvc_studio.fasteners.catalog import FastenerCatalog


@pytest.fixture()
def world(backend):
    from tvc_studio.cad.document import PartRepository
    repo = PartRepository(backend=backend)
    box = repo.create("box", [BoxFeature(0.02, 0.02, 0.02)], material_id="pla")
    box.rebuild(backend)
    assembly = Assembly("test")
    a = assembly.create_component(box.id, "A", position=(0, 0, 0))
    a.grounded = True
    b = assembly.create_component(box.id, "B", position=(0.05, 0.0, 0.0))
    return repo, assembly, a, b


def test_world_transform_composition(world):
    repo, assembly, a, b = world
    from tvc_studio.core.math3d import Transform
    b.parent_id = a.id
    b.transform = Transform.from_pos_euler((0.01, 0, 0), 0, 0, 0)
    assert np.allclose(assembly.world_transform(b.id).translation, (0.01, 0, 0))


def test_coincident_constraint_moves_free_component(world):
    repo, assembly, a, b = world
    assembly.constraints.append(AssemblyConstraint(
        id="c1", type=ConstraintType.COINCIDENT, name="coincident",
        a=EntityRef(a.id, EntityKind.POINT, (0, 0, 0)),
        b=EntityRef(b.id, EntityKind.POINT, (0, 0, 0)),
    ))
    result = AssemblySolver(assembly).solve()
    assert result.converged
    pa = assembly.world_transform(a.id).apply_point((0, 0, 0))
    pb = assembly.world_transform(b.id).apply_point((0, 0, 0))
    assert np.linalg.norm(pa - pb) < 1e-6


def test_mass_calculator_rows_sum_to_total(world):
    repo, assembly, a, b = world
    calc = AssemblyMassCalculator(repo, MaterialLibrary(), FastenerCatalog(materials=MaterialLibrary()))
    result = calc.calculate(assembly)
    total = sum(row.mass for row in result.rows)
    assert total == pytest.approx(result.mass, rel=1e-9)
    assert len(result.rows) == 2


def test_manual_mass_override_changes_total(world):
    repo, assembly, a, b = world
    calc = AssemblyMassCalculator(repo, MaterialLibrary(), FastenerCatalog(materials=MaterialLibrary()))
    before = calc.calculate(assembly)
    row_b = next(r for r in before.rows if r.component_id == b.id)
    b.physical.mass_mode = MassMode.MANUAL
    b.physical.manual_mass = 0.5
    b.physical.manual_com = assembly.world_transform(b.id).translation
    after = calc.calculate(assembly).mass
    assert after > before.mass + 0.4
    assert after == pytest.approx(before.mass - row_b.mass + 0.5, rel=1e-9)


def test_remove_component_recursive(world):
    repo, assembly, a, b = world
    c = assembly.create_component(repo.parts[list(repo.parts)[0]].id, "C", parent_id=b.id)
    removed = assembly.remove_component(b.id)
    assert set(removed) == {b.id, c.id}
    assert set(assembly.components) == {a.id}
