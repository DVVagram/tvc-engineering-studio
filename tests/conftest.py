import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def backend():
    from tvc_studio.cad import get_backend
    return get_backend()


@pytest.fixture(scope="session")
def sample_project():
    from tvc_studio.project.sample import build_sample_project
    project = build_sample_project()
    project.parts.rebuild_all()
    return project
