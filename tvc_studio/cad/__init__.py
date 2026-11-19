"""CAD layer: exact geometry through Open CASCADE (OCP), features, sketches and parts."""

from __future__ import annotations

from ..core.errors import CadKernelUnavailable
from .occt_backend import OCCTGeometry, occt_available, occt_import_error

_BACKEND: OCCTGeometry | None = None


def backend_available() -> bool:
    return occt_available()


def get_backend() -> OCCTGeometry:
    """Process-wide OCCT backend (geometry is immutable and safe to share)."""
    global _BACKEND
    if _BACKEND is None:
        if not occt_available():
            raise CadKernelUnavailable(
                "Open CASCADE (cadquery-ocp) is not installed: exact CAD is unavailable. "
                f"Import error: {occt_import_error()}. Install with: pip install cadquery-ocp"
            )
        _BACKEND = OCCTGeometry()
    return _BACKEND
