import json
import math
import os

import pytest

from tvc_studio.assembly.massprops import AssemblyMassCalculator
from tvc_studio.core.errors import ProjectFormatError
from tvc_studio.project.io import ComponentLibraryStore, JsonProjectStore
from tvc_studio.project.schema import CURRENT_FORMAT_VERSION, Project
from tvc_studio.project.undo import UndoStack


def test_project_round_trip(tmp_path, sample_project):
    store = JsonProjectStore(sample_project.parts.backend)
    path = str(tmp_path / "rocket.tvcproj")
    store.save(sample_project, path)
    loaded = store.load(path)
    assert loaded.name == sample_project.name
    assert len(loaded.parts.parts) == len(sample_project.parts.parts)
    assert len(loaded.assembly.components) == len(sample_project.assembly.components)
    assert len(loaded.assembly.fasteners) == len(sample_project.assembly.fasteners)
    calc = AssemblyMassCalculator(loaded.parts, loaded.materials, loaded.fasteners)
    mass = calc.calculate(loaded.assembly).mass
    calc0 = AssemblyMassCalculator(sample_project.parts, sample_project.materials,
                                   sample_project.fasteners)
    assert mass == pytest.approx(calc0.calculate(sample_project.assembly).mass, rel=1e-9)


def test_newer_format_rejected():
    payload = {"format_version": CURRENT_FORMAT_VERSION + 5, "assembly": {}}
    with pytest.raises(ProjectFormatError):
        Project.from_dict(payload)


def test_invalid_file_rejected(tmp_path):
    path = tmp_path / "not-a-project.json"
    path.write_text("{\"hello\": 1}", encoding="utf-8")
    with pytest.raises(ProjectFormatError):
        JsonProjectStore().load(str(path))


def test_component_library_store(tmp_path):
    store = ComponentLibraryStore(str(tmp_path / "library.json"))
    store.add({"id": "lib1", "kind": "servo", "name": "MG90S", "data": {"id": "mg90s"}})
    store.add({"id": "lib2", "kind": "motor", "name": "Demo", "data": {}})
    assert len(store.by_kind("servo")) == 1
    assert store.get("lib2")["kind"] == "motor"
    assert store.remove("lib1")
    reopened = ComponentLibraryStore(str(tmp_path / "library.json"))
    assert reopened.get("lib1") is None
    assert reopened.get("lib2") is not None


def test_undo_stack_restores_state(sample_project):
    stack = UndoStack()
    stack.begin("Rename", sample_project)
    original = sample_project.name
    sample_project.name = "Renamed"
    assert stack.commit(sample_project)
    assert stack.can_undo()
    restored = stack.undo()
    assert restored["name"] == original
    assert stack.redo()["name"] == "Renamed"


def test_undo_ignores_noop(sample_project):
    stack = UndoStack()
    stack.begin("Nothing", sample_project)
    assert not stack.commit(sample_project)
    assert not stack.can_undo()


def test_blank_project_has_libraries():
    from tvc_studio.project.sample import build_blank_project
    project = build_blank_project()
    assert project.materials.all()
    assert project.fasteners.all()
    assert project.hardware.servos
    assert project.parts.parts == {}
