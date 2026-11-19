"""Material library.

Values are *typical engineering values for the material class*, not guarantees for a
specific purchased or printed part. Every entry carries a :class:`Provenance` so the
application can tell the user when a strength or friction number that drives a result is
an estimate. Printed thermoplastics additionally carry a print-dependence note: real
parts vary with infill, wall count, orientation and extrusion temperature, which is why
mass can be overridden independently of geometry.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace

from ..core.interfaces import Provenance


@dataclass(frozen=True)
class Material:
    id: str
    name: str
    category: str
    density: float                      # kg/m^3
    youngs_modulus: float | None = None  # Pa
    poisson: float | None = None
    yield_strength: float | None = None  # Pa
    ultimate_strength: float | None = None  # Pa
    cte: float | None = None             # 1/K
    friction_static: float = 0.3
    friction_kinetic: float = 0.25
    damping_ratio: float = 0.02           # structural damping, used by joint damping defaults
    thermal_conductivity: float | None = None  # W/(m K)
    provenance: Provenance = Provenance.ESTIMATED
    print_dependent: bool = False
    notes: str = ""

    @property
    def shear_modulus(self) -> float | None:
        """G from E and nu: G = E / (2 (1 + nu))."""
        if self.youngs_modulus is None or self.poisson is None:
            return None
        denom = 2.0 * (1.0 + self.poisson)
        return None if abs(denom) < 1e-12 else self.youngs_modulus / denom

    def with_overrides(self, **kwargs) -> "Material":
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "density": self.density,
            "youngs_modulus": self.youngs_modulus,
            "poisson": self.poisson,
            "yield_strength": self.yield_strength,
            "ultimate_strength": self.ultimate_strength,
            "cte": self.cte,
            "friction_static": self.friction_static,
            "friction_kinetic": self.friction_kinetic,
            "damping_ratio": self.damping_ratio,
            "thermal_conductivity": self.thermal_conductivity,
            "provenance": self.provenance.value,
            "print_dependent": self.print_dependent,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "Material":
        return Material(
            id=str(d["id"]),
            name=str(d.get("name", d["id"])),
            category=str(d.get("category", "custom")),
            density=float(d.get("density", 1000.0)),
            youngs_modulus=_opt(d.get("youngs_modulus")),
            poisson=_opt(d.get("poisson")),
            yield_strength=_opt(d.get("yield_strength")),
            ultimate_strength=_opt(d.get("ultimate_strength")),
            cte=_opt(d.get("cte")),
            friction_static=float(d.get("friction_static", 0.3)),
            friction_kinetic=float(d.get("friction_kinetic", 0.25)),
            damping_ratio=float(d.get("damping_ratio", 0.02)),
            thermal_conductivity=_opt(d.get("thermal_conductivity")),
            provenance=Provenance(d.get("provenance", "estimated")),
            print_dependent=bool(d.get("print_dependent", False)),
            notes=str(d.get("notes", "")),
        )


def _opt(value):
    return None if value is None else float(value)


_GPA = 1.0e9
_MPA = 1.0e6


BUILTIN_MATERIALS: tuple[Material, ...] = (
    # ---- printed thermoplastics (mass/strength depend on process) -----------
    Material(
        id="pla",
        name="PLA (printed)",
        category="thermoplastic",
        density=1240.0,
        youngs_modulus=3.5 * _GPA,
        poisson=0.36,
        yield_strength=49.0 * _MPA,
        ultimate_strength=50.0 * _MPA,
        cte=70e-6,
        friction_static=0.4,
        friction_kinetic=0.35,
        damping_ratio=0.03,
        thermal_conductivity=0.13,
        provenance=Provenance.ESTIMATED,
        print_dependent=True,
        notes="Brittle; strength depends heavily on layer orientation and infill.",
    ),
    Material(
        id="petg",
        name="PETG (printed)",
        category="thermoplastic",
        density=1270.0,
        youngs_modulus=2.1 * _GPA,
        poisson=0.40,
        yield_strength=50.0 * _MPA,
        ultimate_strength=53.0 * _MPA,
        cte=68e-6,
        friction_static=0.35,
        friction_kinetic=0.30,
        damping_ratio=0.04,
        thermal_conductivity=0.19,
        provenance=Provenance.ESTIMATED,
        print_dependent=True,
        notes="Tougher than PLA; prone to stringing. Thread-forming screws may cold-flow.",
    ),
    Material(
        id="abs",
        name="ABS (printed)",
        category="thermoplastic",
        density=1040.0,
        youngs_modulus=2.3 * _GPA,
        poisson=0.35,
        yield_strength=40.0 * _MPA,
        ultimate_strength=42.0 * _MPA,
        cte=90e-6,
        friction_static=0.35,
        friction_kinetic=0.30,
        damping_ratio=0.04,
        thermal_conductivity=0.17,
        provenance=Provenance.ESTIMATED,
        print_dependent=True,
        notes="Layer adhesion is the limiting factor for fastener pull-out.",
    ),
    Material(
        id="asa",
        name="ASA (printed)",
        category="thermoplastic",
        density=1070.0,
        youngs_modulus=2.1 * _GPA,
        poisson=0.35,
        yield_strength=44.0 * _MPA,
        ultimate_strength=46.0 * _MPA,
        cte=90e-6,
        friction_static=0.35,
        friction_kinetic=0.30,
        damping_ratio=0.04,
        thermal_conductivity=0.17,
        provenance=Provenance.ESTIMATED,
        print_dependent=True,
        notes="UV-stable ABS variant; good for airframe exteriors.",
    ),
    Material(
        id="nylon_pa12",
        name="Nylon PA12 (printed)",
        category="thermoplastic",
        density=1010.0,
        youngs_modulus=1.7 * _GPA,
        poisson=0.40,
        yield_strength=48.0 * _MPA,
        ultimate_strength=50.0 * _MPA,
        cte=100e-6,
        friction_static=0.30,
        friction_kinetic=0.25,
        damping_ratio=0.06,
        thermal_conductivity=0.25,
        provenance=Provenance.ESTIMATED,
        print_dependent=True,
        notes="Tough; thread-forming friendly. Absorbs moisture, which changes stiffness.",
    ),
    Material(
        id="tpu_95a",
        name="TPU 95A",
        category="thermoplastic",
        density=1210.0,
        youngs_modulus=0.026 * _GPA,
        poisson=0.48,
        yield_strength=8.0 * _MPA,
        ultimate_strength=30.0 * _MPA,
        cte=150e-6,
        friction_static=0.6,
        friction_kinetic=0.55,
        damping_ratio=0.15,
        thermal_conductivity=0.20,
        provenance=Provenance.ESTIMATED,
        print_dependent=True,
        notes="Elastomeric; use for dampers and flexible mounts, not structural joints.",
    ),
    # ---- metals -----------------------------------------------------------
    Material(
        id="al_6061_t6",
        name="Aluminium 6061-T6",
        category="metal",
        density=2700.0,
        youngs_modulus=68.9 * _GPA,
        poisson=0.33,
        yield_strength=276.0 * _MPA,
        ultimate_strength=310.0 * _MPA,
        cte=23.6e-6,
        friction_static=0.42,
        friction_kinetic=0.35,
        damping_ratio=0.01,
        thermal_conductivity=167.0,
        provenance=Provenance.MANUFACTURER,
    ),
    Material(
        id="al_7075_t6",
        name="Aluminium 7075-T6",
        category="metal",
        density=2810.0,
        youngs_modulus=71.7 * _GPA,
        poisson=0.33,
        yield_strength=503.0 * _MPA,
        ultimate_strength=572.0 * _MPA,
        cte=23.4e-6,
        friction_static=0.42,
        friction_kinetic=0.35,
        damping_ratio=0.01,
        thermal_conductivity=130.0,
        provenance=Provenance.MANUFACTURER,
    ),
    Material(
        id="steel_1018",
        name="Steel 1018 (mild)",
        category="metal",
        density=7870.0,
        youngs_modulus=205.0 * _GPA,
        poisson=0.29,
        yield_strength=370.0 * _MPA,
        ultimate_strength=440.0 * _MPA,
        cte=11.7e-6,
        friction_static=0.55,
        friction_kinetic=0.45,
        damping_ratio=0.01,
        thermal_conductivity=51.9,
        provenance=Provenance.MANUFACTURER,
    ),
    Material(
        id="stainless_304",
        name="Stainless steel 304",
        category="metal",
        density=8000.0,
        youngs_modulus=193.0 * _GPA,
        poisson=0.29,
        yield_strength=215.0 * _MPA,
        ultimate_strength=505.0 * _MPA,
        cte=17.3e-6,
        friction_static=0.5,
        friction_kinetic=0.4,
        damping_ratio=0.01,
        thermal_conductivity=16.2,
        provenance=Provenance.MANUFACTURER,
    ),
    Material(
        id="brass",
        name="Brass (typical alloy)",
        category="metal",
        density=8500.0,
        youngs_modulus=100.0 * _GPA,
        poisson=0.34,
        yield_strength=200.0 * _MPA,
        ultimate_strength=340.0 * _MPA,
        cte=19.0e-6,
        friction_static=0.35,
        friction_kinetic=0.30,
        damping_ratio=0.01,
        thermal_conductivity=109.0,
        provenance=Provenance.ESTIMATED,
        notes="Threaded inserts are usually brass; engagement strength varies by insert design.",
    ),
    Material(
        id="ti_6al4v",
        name="Titanium Ti-6Al-4V",
        category="metal",
        density=4430.0,
        youngs_modulus=113.8 * _GPA,
        poisson=0.342,
        yield_strength=880.0 * _MPA,
        ultimate_strength=950.0 * _MPA,
        cte=8.6e-6,
        friction_static=0.36,
        friction_kinetic=0.30,
        damping_ratio=0.01,
        thermal_conductivity=6.7,
        provenance=Provenance.MANUFACTURER,
    ),
    Material(
        id="magnesium_az31",
        name="Magnesium AZ31",
        category="metal",
        density=1770.0,
        youngs_modulus=45.0 * _GPA,
        poisson=0.35,
        yield_strength=200.0 * _MPA,
        ultimate_strength=260.0 * _MPA,
        cte=26.0e-6,
        friction_static=0.35,
        friction_kinetic=0.30,
        damping_ratio=0.01,
        thermal_conductivity=96.0,
        provenance=Provenance.ESTIMATED,
    ),
    # ---- composites and laminates ----------------------------------------
    Material(
        id="cfrp_quasi_iso",
        name="Carbon fibre laminate (quasi-isotropic)",
        category="composite",
        density=1600.0,
        youngs_modulus=55.0 * _GPA,
        poisson=0.30,
        yield_strength=None,
        ultimate_strength=600.0 * _MPA,
        cte=2.0e-6,
        friction_static=0.4,
        friction_kinetic=0.35,
        damping_ratio=0.02,
        provenance=Provenance.ESTIMATED,
        notes="Lay-up dependent; treat as order-of-magnitude for stiffness.",
    ),
    Material(
        id="g10_fr4",
        name="G10 / FR4 laminate",
        category="composite",
        density=1850.0,
        youngs_modulus=18.0 * _GPA,
        poisson=0.28,
        yield_strength=None,
        ultimate_strength=280.0 * _MPA,
        cte=12.0e-6,
        friction_static=0.4,
        friction_kinetic=0.35,
        damping_ratio=0.02,
        provenance=Provenance.ESTIMATED,
        notes="Common fin material; excellent for threaded fasteners, brittle in bending.",
    ),
    Material(
        id="plywood_birch",
        name="Birch plywood",
        category="wood",
        density=680.0,
        youngs_modulus=9.0 * _GPA,
        poisson=0.30,
        yield_strength=None,
        ultimate_strength=60.0 * _MPA,
        cte=5.0e-6,
        friction_static=0.45,
        friction_kinetic=0.40,
        damping_ratio=0.03,
        provenance=Provenance.ESTIMATED,
        notes="Grain-dependent; centring ring and fin stock reference only.",
    ),
    # ---- printed/structural nylon with inserts ---------------------------
    Material(
        id="nylon_glass_filled",
        name="Nylon (glass-filled, printed)",
        category="thermoplastic",
        density=1350.0,
        youngs_modulus=6.0 * _GPA,
        poisson=0.35,
        yield_strength=75.0 * _MPA,
        ultimate_strength=80.0 * _MPA,
        cte=40e-6,
        friction_static=0.35,
        friction_kinetic=0.30,
        damping_ratio=0.04,
        thermal_conductivity=0.3,
        provenance=Provenance.ESTIMATED,
        print_dependent=True,
        notes="Abrasive to nozzles; good stiffness for gimbal mounts.",
    ),
    Material(
        id="aluminium_printed",
        name="Aluminium (metal-printed, DMLS)",
        category="metal",
        density=2670.0,
        youngs_modulus=70.0 * _GPA,
        poisson=0.33,
        yield_strength=230.0 * _MPA,
        ultimate_strength=330.0 * _MPA,
        cte=23.0e-6,
        friction_static=0.4,
        friction_kinetic=0.35,
        damping_ratio=0.01,
        provenance=Provenance.ESTIMATED,
        notes="Porosity-dependent; strength varies with build orientation.",
    ),
    # ---- generic fallback -------------------------------------------------
    Material(
        id="unknown",
        name="Uncharacterised material",
        category="custom",
        density=1000.0,
        provenance=Provenance.UNKNOWN,
        notes="Placeholder. Results using this material are flagged as not characterised.",
    ),
)


class MaterialLibrary:
    """Registry of built-in plus user materials, persisted as JSON."""

    def __init__(self, materials: list[Material] | None = None) -> None:
        self._materials: dict[str, Material] = {}
        for mat in materials or list(BUILTIN_MATERIALS):
            self._materials[mat.id] = mat

    # -- access ------------------------------------------------------------
    def get(self, material_id: str | None) -> Material:
        if material_id and material_id in self._materials:
            return self._materials[material_id]
        return self._materials["unknown"]

    def add(self, material: Material) -> None:
        if not material.id:
            raise ValueError("material requires an id")
        self._materials[material.id] = material

    def remove(self, material_id: str) -> bool:
        if material_id in {m.id for m in BUILTIN_MATERIALS}:
            return False
        return self._materials.pop(material_id, None) is not None

    def all(self) -> list[Material]:
        return sorted(self._materials.values(), key=lambda m: (m.category, m.name))

    def categories(self) -> list[str]:
        return sorted({m.category for m in self._materials.values()})

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {"materials": [m.to_dict() for m in self._materials.values()]}

    def load_dict(self, d: dict) -> None:
        for entry in d.get("materials", []):
            try:
                self.add(Material.from_dict(entry))
            except (KeyError, TypeError, ValueError):
                continue

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            self.load_dict(json.load(fh))


def default_library() -> MaterialLibrary:
    return MaterialLibrary()
