"""Typed errors used across subsystem boundaries."""

from __future__ import annotations


class TVCStudioError(Exception):
    """Base class for recoverable application errors."""


class CadKernelUnavailable(TVCStudioError):
    """The OCCT/OCP kernel could not be imported.

    Geometry features report this instead of substituting mesh approximations,
    as required by the accuracy rules.
    """


class CadError(TVCStudioError):
    """Geometry operation failed inside the kernel."""


class SketchError(TVCStudioError):
    """Sketch is not solvable or has invalid topology."""


class AssemblyError(TVCStudioError):
    """Assembly structure operation failed."""


class ConstraintError(TVCStudioError):
    """Assembly constraint could not be created or solved."""


class MassPropertyError(TVCStudioError):
    """Mass properties could not be computed or are physically invalid."""


class FastenerError(TVCStudioError):
    """Fastener installation or definition invalid."""


class SimulationError(TVCStudioError):
    """Simulation could not run as configured."""


class ProjectFormatError(TVCStudioError):
    """Project file is corrupt or has an unsupported version."""


class HardwareError(TVCStudioError):
    """Hardware-in-the-loop transport problem."""


class UnsupportedCapability(TVCStudioError):
    """Requested capability is documented but not implemented in this build."""


class NotCharacterised(TVCStudioError):
    """A result would depend on uncharacterised physical data."""


class ValidationWarning(UserWarning):
    """Numerical result depends on assumptions that deserve a UI warning."""
