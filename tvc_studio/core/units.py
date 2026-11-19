"""Units and formatting.

Internal storage is always SI (m, kg, s, N, rad, K, Pa). The UI converts for
display, and project files store SI so saved projects are portable and
comparable. Conversion helpers are explicit rather than implicit so no
calculation silently mixes millimetres with metres.
"""

from __future__ import annotations

import math

# ---- length --------------------------------------------------------------- #
M = 1.0
MM = 1.0e-3
CM = 1.0e-2
INCH = 0.0254
FT = 0.3048

# ---- mass ----------------------------------------------------------------- #
KG = 1.0
G = 1.0e-3
LB = 0.45359237
OZ = LB / 16.0

# ---- force / torque ------------------------------------------------------- #
N = 1.0
KGF = 9.80665
LBF = 4.4482216152605
NM = 1.0
NMM = 1.0e-3
IN_LBF = 0.1129848290276167
OZ_IN = IN_LBF / 16.0
KGF_CM = 0.0980665

# ---- pressure / density --------------------------------------------------- #
PA = 1.0
KPA = 1.0e3
MPA = 1.0e6
PSI = 6894.757293168
KG_M3 = 1.0
G_CM3 = 1000.0

# ---- angles --------------------------------------------------------------- #
DEG = math.pi / 180.0
RAD = 1.0

# ---- temperature ---------------------------------------------------------- #
KELVIN = 1.0
CELSIUS_OFFSET = 273.15


def deg2rad(value: float) -> float:
    return float(value) * DEG


def rad2deg(value: float) -> float:
    return float(value) / DEG


def celsius_to_kelvin(value: float) -> float:
    return float(value) + CELSIUS_OFFSET


def kelvin_to_celsius(value: float) -> float:
    return float(value) - CELSIUS_OFFSET


def mm(value: float) -> float:
    """Convert a millimetre value to SI metres."""
    return float(value) * MM


def inches(value: float) -> float:
    return float(value) * INCH


def grams(value: float) -> float:
    return float(value) * G


def fmt(value: float, unit: str, digits: int = 3) -> str:
    """Format a quantity with a fixed number of significant decimals."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f} {unit}".strip()


def fmt_length(meters: float, digits: int = 2) -> str:
    if not math.isfinite(meters):
        return "—"
    if abs(meters) < 1e-2:
        return f"{meters * 1000.0:.{digits}f} mm"
    return f"{meters:.{digits + 1}f} m"


def fmt_mass(kg: float, digits: int = 2) -> str:
    if not math.isfinite(kg):
        return "—"
    if abs(kg) < 1.0:
        return f"{kg * 1000.0:.{digits}f} g"
    return f"{kg:.{digits + 2}f} kg"


def fmt_angle(rad: float, digits: int = 2) -> str:
    return f"{rad2deg(rad):.{digits}f}°"


def fmt_torque(nm: float, digits: int = 3) -> str:
    if not math.isfinite(nm):
        return "—"
    if abs(nm) < 1.0:
        return f"{nm * 1000.0:.{digits}f} N·mm"
    return f"{nm:.{digits}f} N·m"


class UnitSystem:
    """Named display system; storage stays SI."""

    METRIC = "metric"
    IMPERIAL = "imperial"

    def __init__(self, name: str = METRIC) -> None:
        self.name = name

    @property
    def length_scale(self) -> float:
        return 1.0 if self.name == self.METRIC else 1.0 / INCH

    @property
    def length_label(self) -> str:
        return "m" if self.name == self.METRIC else "in"

    @property
    def mass_scale(self) -> float:
        return 1.0 if self.name == self.METRIC else 1.0 / LB

    @property
    def mass_label(self) -> str:
        return "kg" if self.name == self.METRIC else "lb"

    @property
    def force_label(self) -> str:
        return "N" if self.name == self.METRIC else "lbf"
