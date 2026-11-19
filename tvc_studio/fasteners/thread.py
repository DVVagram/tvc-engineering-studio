"""Threads: representation levels, engagement mechanics, installation estimates.

Two representation levels exist, per Part 5.3:

* **simplified** — a cylinder at major diameter plus pitch metadata. Cheap, used for
  normal assemblies, keeps installation kinematics and collision checks exact.
* **detailed** — an actual helical solid swept from the ISO 60 degree profile. Expensive,
  used for inspection, interference checks and high-resolution export.

Engagement mechanics is a reduced-order model. It predicts torque trends, strip risk and
pull-out estimates and is explicitly *not* a finite-element plasticity model. Every result
carries ``assumptions`` and ``Provenance.ESTIMATED``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..core.interfaces import Provenance
from .catalog import ThreadSpec


class ThreadRepresentation(str, Enum):
    SIMPLIFIED = "simplified"
    DETAILED = "detailed"


class EngagementMode(str, Enum):
    PRE_THREADED = "pre_threaded"     # engages a matching cut thread
    THREAD_FORMING = "thread_forming"  # displaces/cuts material in a pilot hole
    CUTTING = "cutting"              # self-drilling/tapping cuts its own material


@dataclass
class EngagementResult:
    """Outcome of an installation analysis, with its assumptions attached."""

    engagement_length: float
    turns_engaged: float
    thread_shear_area: float          # m^2, internal thread shear area
    stress_area: float                # m^2, screw tensile stress area
    clamp_force_estimate: float       # N
    form_torque: float                # N*m, 0 for pre-threaded
    seating_torque: float             # N*m
    strip_torque_estimate: float      # N*m
    pullout_force_estimate: float     # N
    strip_risk: float                 # 0..1 ratio of applied to strip torque
    engaged_thread_depth_fraction: float
    mode: EngagementMode
    assumptions: tuple[str, ...] = ()
    provenance: Provenance = Provenance.ESTIMATED
    warnings: tuple[str, ...] = ()


def thread_shear_area(thread: ThreadSpec, engagement_length: float, internal: bool = True) -> float:
    """Shear area of the engaged thread cylinder.

    Uses the standard approximation that the shear cylinder sits at the pitch diameter
    and that only half the axial length carries shear load through thread crowns.
    """
    d = thread.pitch_diameter
    return math.pi * d * max(engagement_length, 0.0) * 0.5


def bolt_stiffness(thread: ThreadSpec, grip_length: float, youngs_modulus: float,
                   head_diameter: float | None = None) -> float:
    """Axial stiffness of a bolt in a grip, N/m.

    Uses the Shigley conical frustum for the clamped members and the plain shank area for
    the bolt. Returns bolt stiffness only; joint stiffness is added by the caller.
    """
    A = thread.tensile_stress_area
    if grip_length <= 0 or youngs_modulus <= 0:
        return 0.0
    return youngs_modulus * A / grip_length


def clamp_force_from_torque(thread: ThreadSpec, torque: float, friction: float = 0.15) -> float:
    """Raise-load relation: T = F * (d2/2) * (tan(lambda) + mu) / (1 - mu*tan(lambda)).

    This is the standard thread-torque relation (Shigley eq. 8-2 without collar friction).
    Returns axial force in N.
    """
    d2 = thread.pitch_diameter / 2.0
    tan_l = math.tan(thread.helix_angle)
    denom = 1.0 - friction * tan_l
    if d2 <= 0 or abs(denom) < 1e-9:
        return 0.0
    return torque / (d2 * (tan_l + friction) / denom)


def torque_from_clamp_force(thread: ThreadSpec, force: float, friction: float = 0.15) -> float:
    d2 = thread.pitch_diameter / 2.0
    tan_l = math.tan(thread.helix_angle)
    denom = 1.0 - friction * tan_l
    if abs(denom) < 1e-9:
        return 0.0
    return force * d2 * (tan_l + friction) / denom


class ThreadEngagementModel:
    """Reduced-order model for thread forming, stripping and pull-out.

    Parameters that are material or process dependent are explicit and live in
    :class:`EngagementInputs`; nothing is hidden. Defaults are engineering estimates for
    printed PLA and are labelled as such.
    """

    def __init__(self, forming_friction: float = 0.35, shear_strength_factor: float = 0.6) -> None:
        #: friction during thread forming (higher than sliding: material is being worked)
        self.forming_friction = forming_friction
        #: tau_ult ~= shear_strength_factor * sigma_ult (von Mises estimate 0.577)
        self.shear_strength_factor = shear_strength_factor

    def analyse(
        self,
        thread: ThreadSpec,
        insertion_depth: float,
        mode: EngagementMode,
        target_ultimate_strength: float | None,
        target_shear_strength: float | None = None,
        pilot_diameter: float = 0.0,
        applied_torque: float = 0.0,
        layer_adhesion_factor: float = 1.0,
        youngs_modulus: float | None = None,
    ) -> EngagementResult:
        warnings: list[str] = []
        assumptions: list[str] = [
            "Reduced-order thread engagement model, not a finite-element plasticity solution.",
            "Thread shear area approximated by a cylinder at the pitch diameter.",
            "Torque from the standard raise-load relation with collar friction excluded.",
        ]

        engagement = max(insertion_depth, 0.0)
        turns = thread.turns_for_depth(engagement)

        if mode is EngagementMode.PRE_THREADED:
            engaged_depth_fraction = 1.0
        else:
            if pilot_diameter <= 0:
                raise ValueError("thread-forming installation requires a pilot hole diameter")
            interference = max(thread.major_diameter - pilot_diameter, 0.0) / 2.0
            engaged_depth_fraction = float(np.clip(interference / max(thread.thread_depth, 1e-9), 0.0, 1.5))
            if engaged_depth_fraction > 1.0:
                warnings.append("pilot hole smaller than the thread root; risk of splitting the boss")
            if engaged_depth_fraction < 0.35:
                warnings.append("very shallow engagement: the screw may strip out under load")

        eff_shear_strength = target_shear_strength
        if eff_shear_strength is None and target_ultimate_strength is not None:
            eff_shear_strength = target_ultimate_strength * self.shear_strength_factor

        A_shear = thread_shear_area(thread, engagement) * min(engaged_depth_fraction, 1.0)
        A_stress = thread.tensile_stress_area

        form_torque = 0.0
        if mode is not EngagementMode.PRE_THREADED and target_ultimate_strength is not None:
            # Forming torque scales with the material's yield stress, the interference
            # volume and the thread radius. 0.05 is the empirical friction/tooling factor.
            d = thread.major_diameter
            interference = max(thread.major_diameter - pilot_diameter, 0.0) / 2.0
            form_torque = 0.05 * target_ultimate_strength * d * engagement * interference / max(thread.pitch, 1e-9)
            assumptions.append("Forming torque uses an empirical 0.05 tooling factor; calibrate against your material.")

        # seating torque for pre-threaded joints from an assumed clamp (30% of proof load)
        f_proof = (A_stress * (target_ultimate_strength or 0.0)) * 0.6 if target_ultimate_strength else 0.0
        seating_torque = torque_from_clamp_force(thread, f_proof * 0.6, self.forming_friction) if f_proof else 0.0
        clamp_estimate = f_proof * 0.6

        if eff_shear_strength is not None:
            # internal thread strips along its shear cylinder
            strip_force = A_shear * eff_shear_strength * layer_adhesion_factor
            strip_torque = torque_from_clamp_force(thread, strip_force, self.forming_friction)
            pullout = A_shear * eff_shear_strength * layer_adhesion_factor
            if eff_shear_strength < 15e6:
                warnings.append("target material shear strength is low; threaded fasteners may pull out")
        else:
            strip_force = strip_torque = pullout = float("nan")
            warnings.append("target material strength not characterised: strip/pull-out not predicted")

        applied = applied_torque if applied_torque > 0 else (seating_torque if seating_torque > 0 else form_torque)
        if strip_torque and np.isfinite(strip_torque) and strip_torque > 0:
            risk = float(np.clip(applied / strip_torque, 0.0, 5.0))
            if risk > 1.0:
                warnings.append("applied installation torque exceeds the estimated strip torque")
        else:
            risk = float("nan")

        if youngs_modulus is not None and youngs_modulus < 1e9:
            assumptions.append("Elastomeric target: thread forming will relax over time (not modelled).")

        return EngagementResult(
            engagement_length=engagement,
            turns_engaged=turns,
            thread_shear_area=A_shear,
            stress_area=A_stress,
            clamp_force_estimate=clamp_estimate,
            form_torque=form_torque,
            seating_torque=seating_torque,
            strip_torque_estimate=strip_torque,
            pullout_force_estimate=pullout,
            strip_risk=risk,
            engaged_thread_depth_fraction=engaged_depth_fraction,
            mode=mode,
            assumptions=tuple(assumptions),
            provenance=Provenance.ESTIMATED,
            warnings=tuple(warnings),
        )


# --------------------------------------------------------------------------- #
# Geometry helpers (kernel-backed, imported lazily)
# --------------------------------------------------------------------------- #
def helical_sweep_points(thread: ThreadSpec, length: float, start_angle: float = 0.0,
                         points_per_turn: int = 48) -> np.ndarray:
    """Centreline helix of a thread, as an (n, 3) array.

    Axial axis is +Z. Used by the detailed-thread geometry builder and by tests that
    verify pitch/lead relationships.
    """
    turns = length / thread.lead
    n = max(int(turns * points_per_turn), 2)
    t = np.linspace(0.0, turns * 2.0 * math.pi, n)
    r = thread.pitch_diameter / 2.0
    return np.column_stack([
        r * np.cos(t + start_angle),
        r * np.sin(t + start_angle),
        t / (2.0 * math.pi) * thread.lead,
    ])


def thread_profile_points(thread: ThreadSpec, radial_offset: float = 0.0) -> np.ndarray:
    """Basic ISO 60 degree external thread profile in the axial-radial plane.

    Returns (n, 2) points (z, r) over one pitch, ready to be revolved/swept.
    """
    d = thread.major_diameter
    h = thread.thread_depth
    p = thread.pitch
    r_major = d / 2.0 + radial_offset
    r_minor = d / 2.0 - h + radial_offset
    # trapezoidal approximation of the trunc'd ISO profile
    return np.array([
        [0.0, r_minor],
        [0.125 * p, r_major],
        [0.375 * p, r_major],
        [0.5 * p, r_minor],
        [0.625 * p, r_minor],
        [0.875 * p, r_major],
        [p, r_major],
    ], dtype=float)


def detailed_thread_solid(thread: ThreadSpec, length: float, density: float):
    """Build an actual helical thread solid with OCCT.

    Returns an OCCT shape. Raises ``CadKernelUnavailable`` when the kernel is absent so no
    mesh is silently substituted.
    """
    from ..cad.occt_backend import OCCTGeometry  # local import keeps kernel optional

    return OCCTGeometry().thread_helix(thread, length, density)
