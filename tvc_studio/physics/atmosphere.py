"""International Standard Atmosphere (1976) with wind.

Valid to 86 km geometric altitude, layer by layer. Used for density, pressure,
temperature and speed of sound in the aerodynamic model. Wind is a configurable constant
plus optional gust, applied to the relative airspeed only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.math3d import as_vec3, vec3

R_AIR = 287.05287       # J/(kg K)
GAMMA = 1.4
G0 = 9.80665

# (base geopotential altitude, base temperature, lapse rate K/m)
_LAYERS = [
    (0.0, 288.15, -0.0065),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 0.001),
    (32000.0, 228.65, 0.0028),
    (47000.0, 270.65, 0.0),
    (51000.0, 270.65, -0.0028),
    (71000.0, 214.65, -0.002),
    (84852.0, 186.946, 0.0),
]


@dataclass
class AtmosphereState:
    altitude: float
    temperature: float
    pressure: float
    density: float
    speed_of_sound: float

    def to_dict(self) -> dict:
        return {
            "altitude": self.altitude,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "density": self.density,
            "speed_of_sound": self.speed_of_sound,
        }


def geopotential_to_geometric(h: float) -> float:
    r_e = 6356766.0
    return r_e * h / (r_e - h)


def isa(altitude: float) -> AtmosphereState:
    """ISA state at geometric altitude (metres). Below 0 m the sea-level state is held."""
    h = max(float(altitude), 0.0)
    h_geo = geopotential_to_geometric(h)
    h_geo = min(h_geo, _LAYERS[-1][0])
    base_alt, base_temp, lapse = _LAYERS[0]
    for alt, temp, rate in _LAYERS:
        if h_geo >= alt:
            base_alt, base_temp, lapse = alt, temp, rate
        else:
            break
    t = base_temp + lapse * (h_geo - base_alt)
    p0 = 101325.0
    # compute pressure at layer base by integrating from sea level
    p_base = p0
    for i in range(1, len(_LAYERS)):
        alt0, t0, rate0 = _LAYERS[i - 1]
        alt1 = _LAYERS[i][0]
        if base_alt <= alt0:
            break
        dh = min(base_alt, alt1) - alt0
        if abs(rate0) < 1e-9:
            p_base *= math.exp(-G0 * dh / (R_AIR * t0))
        else:
            t1 = t0 + rate0 * dh
            p_base *= (t1 / t0) ** (-G0 / (R_AIR * rate0))
    if abs(lapse) < 1e-9:
        p = p_base * math.exp(-G0 * (h_geo - base_alt) / (R_AIR * base_temp))
    else:
        p = p_base * (t / base_temp) ** (-G0 / (R_AIR * lapse))
    rho = p / (R_AIR * t)
    a = math.sqrt(GAMMA * R_AIR * t)
    return AtmosphereState(h, t, p, rho, a)


class Atmosphere:
    """Atmosphere plus wind field (constant wind with optional shear and gust)."""

    def __init__(self, wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 gust_std: float = 0.0, seed: int = 7) -> None:
        self.wind = as_vec3(wind)
        self.gust_std = float(gust_std)
        self.rng = np.random.default_rng(seed)
        self._gust = vec3()

    def state(self, altitude: float) -> AtmosphereState:
        return isa(altitude)

    def wind_at(self, altitude: float, dt: float = 0.0) -> np.ndarray:
        if self.gust_std <= 0 or dt <= 0:
            return self.wind.copy()
        # first-order Gauss-Markov gust
        alpha = math.exp(-dt / 1.5)
        self._gust = self._gust * alpha + self.rng.normal(0.0, self.gust_std * math.sqrt(1 - alpha ** 2), 3)
        return self.wind + self._gust

    def density(self, altitude: float) -> float:
        return isa(altitude).density

    def speed_of_sound(self, altitude: float) -> float:
        return isa(altitude).speed_of_sound


STANDARD_SEA_LEVEL = isa(0.0)
