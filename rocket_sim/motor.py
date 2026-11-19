"""User-editable motor performance and the thrust curves generated from it.

Only performance lives here: average thrust, peak thrust, total impulse, burn time and the
shape of the curve. Physical hardware (diameter, length, casing and propellant mass) is
fixed in :mod:`rocket_sim.config` and is never derived from these numbers.

The curve is an estimate built from the four numbers the user supplies, and every object
built here is tagged ``measured=False`` so the interface can say so. Measured motor data
is not part of this first version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import G0, LAUNCH_MASS

SHAPES = ("neutral", "progressive", "regressive")
SHAPE_LABELS = {
    "neutral": "Neutral (flat)",
    "progressive": "Progressive (rising)",
    "regressive": "Regressive (falling)",
}

_IMPULSE_TOLERANCE = 0.01  # 1 % between average thrust x burn time and total impulse
_MIN_BURN_TIME = 0.02
_MAX_BURN_TIME = 30.0
_MAX_IMPULSE = 5000.0
_MAX_THRUST = 8000.0


class MotorSpecError(ValueError):
    """One or more motor performance inputs are not physically consistent."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(" ".join(errors))


def _is_positive_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0


@dataclass(frozen=True)
class ThrustCurve:
    """Sampled thrust history, normalised to the stated total impulse."""

    times: np.ndarray
    thrusts: np.ndarray
    cumulative_impulse: np.ndarray  # N s delivered from ignition up to each sample
    burn_time: float
    total_impulse: float
    mean_thrust: float
    peak_thrust: float
    shape: str
    measured: bool = False
    designation: str = field(default="")

    def thrust_at(self, time: float) -> float:
        if time < 0.0 or time > self.burn_time:
            return 0.0
        return float(np.interp(time, self.times, self.thrusts))

    def impulse_delivered(self, time: float) -> float:
        """Impulse burned by ``time``; drives propellant depletion."""
        if time <= 0.0:
            return 0.0
        if time >= self.burn_time:
            return self.total_impulse
        return float(np.interp(time, self.times, self.cumulative_impulse))

    def integrated_impulse(self) -> float:
        return float(np.sum(0.5 * (self.thrusts[1:] + self.thrusts[:-1])
                            * np.diff(self.times)))

    @property
    def source_label(self) -> str:
        return "estimated from stated performance" if not self.measured else "measured data"


@dataclass(frozen=True)
class MotorPerformance:
    """Performance specification of the fixed motor.

    ``average_thrust``, ``total_impulse`` and ``burn_time`` describe the same thing in two
    ways, so they must agree to within 1 %. Inconsistent combinations are rejected instead
    of being quietly rewritten.
    """

    designation: str = "SR-55"
    average_thrust: float = 30.5  # N
    maximum_thrust: float = 44.0  # N
    total_impulse: float = 54.9  # N s
    burn_time: float = 1.8  # s
    curve_shape: str = "progressive"
    measured: bool = False

    # ---------------------------------------------------------------- validation
    def errors(self) -> list[str]:
        problems: list[str] = []

        if not isinstance(self.curve_shape, str) or self.curve_shape not in SHAPES:
            problems.append(f"Unknown thrust-curve shape {self.curve_shape!r}; "
                            f"choose one of {', '.join(SHAPES)}.")
        if not _is_positive_finite(self.burn_time) or not (
                _MIN_BURN_TIME <= self.burn_time <= _MAX_BURN_TIME):
            problems.append(f"Burn time must be between {_MIN_BURN_TIME:.2f} s and "
                            f"{_MAX_BURN_TIME:.0f} s.")
        if not _is_positive_finite(self.total_impulse) or self.total_impulse > _MAX_IMPULSE:
            problems.append(f"Total impulse must be between 0 and {_MAX_IMPULSE:.0f} N s.")
        for label, value in (("Average thrust", self.average_thrust),
                             ("Maximum thrust", self.maximum_thrust)):
            if not _is_positive_finite(value) or value > _MAX_THRUST:
                problems.append(f"{label} must be between 0 and {_MAX_THRUST:.0f} N.")

        if problems:
            return problems

        implied = self.average_thrust * self.burn_time
        if abs(implied - self.total_impulse) > _IMPULSE_TOLERANCE * self.total_impulse:
            problems.append(
                f"Average thrust × burn time is {implied:.1f} N s, but the total impulse "
                f"is {self.total_impulse:.1f} N s. Adjust one of them (difference must be "
                f"under {_IMPULSE_TOLERANCE * 100:.0f} %)."
            )

        if self.maximum_thrust < self.average_thrust * (1.0 - 1e-6):
            problems.append(
                f"Maximum thrust ({self.maximum_thrust:.1f} N) cannot be lower than the "
                f"average thrust ({self.average_thrust:.1f} N)."
            )
        elif self.curve_shape == "neutral":
            if abs(self.maximum_thrust - self.average_thrust) > 0.02 * self.average_thrust:
                problems.append(
                    "A neutral curve burns at a constant thrust, so the maximum thrust must "
                    f"equal the average thrust ({self.average_thrust:.1f} N)."
                )
        elif self.maximum_thrust > 2.0 * self.average_thrust * (1.0 + 1e-9):
            problems.append(
                f"A single linear {self.curve_shape} ramp whose mean is "
                f"{self.average_thrust:.1f} N cannot peak above "
                f"{2.0 * self.average_thrust:.1f} N without starting below zero thrust. "
                "Lower the peak thrust or raise the average thrust."
            )

        return problems

    def validate(self) -> "MotorPerformance":
        problems = self.errors()
        if problems:
            raise MotorSpecError(problems)
        return self

    # ------------------------------------------------------------------- curve
    def curve(self, samples: int = 241) -> ThrustCurve:
        """Build the estimated thrust curve for these performance numbers."""
        self.validate()
        samples = max(3, int(samples))
        times = np.linspace(0.0, self.burn_time, samples)
        u = times / self.burn_time
        peak = float(self.maximum_thrust)

        if self.curve_shape == "neutral":
            thrusts = np.full_like(times, peak)
        elif self.curve_shape == "progressive":
            start = 2.0 * self.average_thrust - peak
            thrusts = start + (peak - start) * u
        else:  # regressive
            end = 2.0 * self.average_thrust - peak
            thrusts = peak + (end - peak) * u

        unit_impulse = float(np.sum(0.5 * (thrusts[1:] + thrusts[:-1]) * np.diff(times)))
        if unit_impulse <= 0.0:  # defensive: validation should prevent this
            raise MotorSpecError(["The requested curve has no positive thrust."])
        thrusts = thrusts * (self.total_impulse / unit_impulse)
        cumulative = np.concatenate([
            [0.0],
            np.cumsum(0.5 * (thrusts[1:] + thrusts[:-1]) * np.diff(times)),
        ])

        return ThrustCurve(
            times=times,
            thrusts=thrusts,
            cumulative_impulse=cumulative,
            burn_time=self.burn_time,
            total_impulse=float(np.sum(0.5 * (thrusts[1:] + thrusts[:-1])
                                       * np.diff(times))),
            mean_thrust=float(np.mean(thrusts)),
            peak_thrust=float(np.max(thrusts)),
            shape=self.curve_shape,
            measured=self.measured,
            designation=self.designation.strip(),
        )

    # ------------------------------------------------------------------ labels
    @property
    def display_name(self) -> str:
        return self.designation.strip() or "Untitled motor"

    @property
    def thrust_to_weight(self) -> float:
        """Average thrust divided by launch weight."""
        return self.average_thrust / (LAUNCH_MASS * G0)

    def with_impulse_fitted(self) -> "MotorPerformance":
        """Explicit user action: set total impulse to average thrust × burn time."""
        return MotorPerformance(
            designation=self.designation,
            average_thrust=self.average_thrust,
            maximum_thrust=self.maximum_thrust,
            total_impulse=round(self.average_thrust * self.burn_time, 3),
            burn_time=self.burn_time,
            curve_shape=self.curve_shape,
            measured=self.measured,
        )


DEFAULT_PERFORMANCE = MotorPerformance()

PRESETS: dict[str, MotorPerformance] = {
    "SR-30 — low power": MotorPerformance("SR-30", 18.0, 26.0, 32.4, 1.8, "progressive"),
    "SR-55 — standard": MotorPerformance("SR-55", 30.5, 44.0, 54.9, 1.8, "progressive"),
    "SR-90 — high power": MotorPerformance("SR-90", 45.0, 70.0, 81.0, 1.8, "progressive"),
    "SR-55 flat": MotorPerformance("SR-55F", 30.5, 30.5, 54.9, 1.8, "neutral"),
}
