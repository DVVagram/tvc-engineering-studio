"""Engineering reports generated from the project.

Every report states its own provenance and assumptions; nothing here upgrades an estimate
to a measured value. Reports are plain text (with CSV/JSON export) so they can be pasted
into a design review or committed next to the project.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from dataclasses import dataclass, field

import numpy as np

from ..assembly.constraints import AssemblySolver
from ..assembly.massprops import AssemblyMassCalculator
from ..fasteners.install import ScrewInstaller
from ..fasteners.thread import EngagementMode
from ..mech.contact import clearance_report
from ..physics.aero import barrowman_cp
from ..physics.propulsion import MotorLibrary


@dataclass
class Report:
    title: str
    lines: list[str] = field(default_factory=list)
    tables: dict[str, list[dict]] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated: str = field(default_factory=lambda: _dt.datetime.now().strftime("%Y-%m-%d %H:%M"))

    def to_text(self) -> str:
        out = [self.title, "=" * len(self.title), f"Generated {self.generated}", ""]
        out.extend(self.lines)
        for name, rows in self.tables.items():
            out.append("")
            out.append(name)
            out.append("-" * len(name))
            if rows:
                headers = list(rows[0].keys())
                out.append(" | ".join(str(h) for h in headers))
                for row in rows:
                    out.append(" | ".join(_fmt(row.get(h)) for h in headers))
        if self.assumptions:
            out.append("")
            out.append("Assumptions")
            out.append("-----------")
            out.extend(f"* {a}" for a in self.assumptions)
        if self.warnings:
            out.append("")
            out.append("Warnings")
            out.append("--------")
            out.extend(f"! {w}" for w in self.warnings)
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "title": self.title, "generated": self.generated, "lines": self.lines,
            "tables": self.tables, "assumptions": self.assumptions, "warnings": self.warnings,
        }

    def save(self, path: str) -> str:
        if path.lower().endswith(".json"):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.to_text())
        return path


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if value == 0:
            return "0"
        if abs(value) >= 1e4 or abs(value) < 1e-3:
            return f"{value:.4g}"
        return f"{value:.6g}"
    return str(value)


def mass_report(project) -> Report:
    calc = AssemblyMassCalculator(project.parts, project.materials, project.fasteners)
    result = calc.calculate(project.assembly)
    report = Report("Mass properties")
    mp = result.properties
    report.lines = [
        f"Assembly: {project.assembly.name}",
        f"Total mass: {mp.mass*1000:.2f} g ({mp.mass:.5f} kg)",
        f"Centre of gravity: X={mp.com[0]*1000:+.2f}  Y={mp.com[1]*1000:+.2f}  Z={mp.com[2]*1000:+.2f} mm",
        f"Provenance: {mp.provenance}",
        "Principal inertia (kg*m^2): " + ", ".join(f"{v:.4e}" for v in np.linalg.eigvalsh(np.asarray(mp.inertia))),
    ]
    rows = [
        {"item": r.name, "kind": r.kind, "mass_g": round(r.mass * 1000, 3),
         "percent": round(r.percent, 2), "provenance": r.provenance,
         "cg_z_mm": round(float(r.com[2]) * 1000, 2)}
        for r in result.rows
    ]
    report.tables["Breakdown"] = rows
    report.assumptions = [
        "Masses are computed from the exact B-rep volume and the material density unless a manual override is set.",
        "Fastener masses use the catalogue estimate (head + threaded length), not a modelled solid.",
        "Components with a mass override are marked 'user_override' and are not CAD-derived.",
    ]
    report.warnings = [d for d in result.diagnostics]
    return report


def fastener_report(project) -> Report:
    installer = ScrewInstaller()
    report = Report("Fastener engagement")
    holes = {h.id: h for h in project.assembly.holes.values()}
    rows = []
    for state in project.assembly.fasteners.values():
        spec = project.fasteners.get(state.fastener_id)
        hole = holes.get(state.hole_id)
        if spec is None or hole is None:
            continue
        comp = project.assembly.components.get(hole.component_id)
        material = project.materials.get(hole.material_id or (comp.physical.material_id if comp else ""))
        strength = getattr(material, "ultimate_strength", None) if material else None
        modulus = getattr(material, "youngs_modulus", None) if material else None
        try:
            result = installer.inspect(state, spec, hole, strength, modulus)
        except Exception as exc:
            rows.append({"fastener": spec.name, "hole": hole.label, "error": str(exc)})
            continue
        passed = bool(state.installed) and not state.is_interfering(spec, hole)
        rows.append({
            "fastener": spec.name,
            "hole": hole.label,
            "mode": result.mode.value,
            "depth_mm": round(state.depth * 1000, 2),
            "engagement_mm": round(result.engagement_length * 1000, 2),
            "turns": round(result.turns_engaged, 2),
            "clamp_N": round(result.clamp_force_estimate, 1),
            "strip_torque_Nm": round(result.strip_torque_estimate, 3)
            if np.isfinite(result.strip_torque_estimate) else "n/a",
            "risk": round(result.strip_risk, 2) if np.isfinite(result.strip_risk) else "n/a",
            "seated": "yes" if passed else "no",
        })
    report.tables["Installed fasteners"] = rows
    report.lines = [
        f"{len(rows)} installed fastener(s) across {len(project.assembly.holes)} hole(s).",
        "Strip torque and clamp force are reduced-order estimates, not test data.",
    ]
    report.assumptions = [
        "Thread engagement is limited to the hole depth; heads are assumed seated.",
        "Target material strength comes from the material library (often a generic value).",
    ]
    risky = [r for r in rows if isinstance(r.get("risk"), (int, float)) and r["risk"] > 1.0]
    if risky:
        report.warnings.append(f"{len(risky)} fastener(s) exceed the estimated strip torque.")
    if any(r.get("seated") == "no" for r in rows):
        report.warnings.append("Some fasteners are not fully seated or interfere with the hole depth.")
    return report


def clearance_report_for_project(project, threshold: float = 0.001) -> Report:
    report = Report("Clearance and interference")
    shapes, transforms, names = {}, {}, {}
    for comp in project.assembly.active_components():
        if not comp.part_id:
            continue
        try:
            shapes[comp.id] = project.parts.shape(comp.part_id)
            transforms[comp.id] = project.assembly.world_transform(comp.id)
            names[comp.id] = comp.name
        except Exception as exc:
            report.warnings.append(f"'{comp.name}' could not be measured: {exc}")
    rows = clearance_report(project.parts.backend, shapes, transforms, threshold)
    table = []
    for row in rows:
        if "error" in row:
            table.append({"a": row["a"], "b": row["b"], "status": "error", "detail": row["error"]})
            continue
        table.append({
            "a": names.get(row["a"], row["a"]), "b": names.get(row["b"], row["b"]),
            "status": row["status"],
            "overlap_mm3": round(row["volume"] * 1e9, 3),
            "clearance_mm": round(row["distance"] * 1000, 3),
        })
    report.tables["Pairs"] = table
    interfering = [r for r in table if r["status"] == "interference"]
    report.lines = [
        f"Checked {len(table)} component pair(s) with exact B-rep booleans.",
        f"{len(interfering)} pair(s) overlap.",
    ]
    report.assumptions = [
        "Coincident mating faces report zero clearance and zero overlap; only volume overlap counts as interference.",
        "Fasteners are not included (they are catalogue representations).",
    ]
    if interfering:
        report.warnings.append("Overlapping solids found: check the assembly constraints and hole clearances.")
    return report


def joint_report(project) -> Report:
    solver = AssemblySolver(project.assembly)
    result = solver.solve(apply=False)
    report = Report("Assembly constraints")
    report.lines = [
        f"{len(project.assembly.constraints)} constraint(s), {len(project.assembly.components)} component(s).",
        result.summary(),
    ]
    report.tables["Constraints"] = [
        {
            "name": c.name or c.id,
            "type": c.type.value,
            "a": project.assembly.components[c.a.component_id].name
            if c.a.component_id in project.assembly.components else c.a.component_id,
            "b": project.assembly.components[c.b.component_id].name
            if c.b and c.b.component_id in project.assembly.components else "(world)",
            "enabled": "yes" if c.enabled and not c.suppressed else "no",
        }
        for c in project.assembly.constraints
    ]
    report.assumptions = [
        "Degrees of freedom come from the rank of the constraint Jacobian at the current pose.",
        "An under-constrained assembly is reported, not silently grounded.",
    ]
    report.warnings = [d.message for d in result.diagnostics]
    return report


def aero_report(project, rocket_build=None) -> Report:
    report = Report("Aerodynamics")
    cfg = project.aero
    lines = [
        f"Model: {cfg.model.value}",
        f"Reference area: {cfg.reference_area:.6f} m^2",
        f"Reference length: {cfg.reference_length:.4f} m",
        f"Cd0: {cfg.cd0:.3f}    Cn_alpha: {cfg.cn_alpha:.3f} /rad",
        f"Centre of pressure: {cfg.cp_position:.4f} m from the nose tip" if cfg.cp_position > 0
        else "Centre of pressure: geometry estimate (Barrowman) — no measured value",
        f"Coefficient uncertainty: +/-{cfg.uncertainty_percent:.0f}%",
    ]
    report.lines = lines
    report.assumptions = list(cfg.warnings()) if hasattr(cfg, "warnings") else []
    if rocket_build is not None:
        report.tables["Stack"] = [
            {"element": e.name, "kind": e.kind, "start_mm": round(e.start * 1000, 1),
             "length_mm": round(e.length * 1000, 1), "diameter_mm": round(e.diameter * 1000, 1),
             "fins": e.fin_count, "span_mm": round(e.fin_span * 1000, 1)}
            for e in rocket_build.stack
        ]
        estimate = barrowman_cp(rocket_build.stack)
        report.lines.append(f"Barrowman CP: {estimate.cp_position:.4f} m from the datum")
        report.tables["CP contributions"] = estimate.contributions
    return report


def flight_report(result, project=None) -> Report:
    report = Report("Flight simulation")
    summary = result.summary
    report.lines = [
        f"Apogee: {summary.get('apogee_m', float('nan')):.1f} m at t = {summary.get('apogee_time_s', float('nan')):.2f} s",
        f"Maximum speed: {summary.get('max_speed_ms', float('nan')):.1f} m/s (Mach {summary.get('max_mach', 0):.2f})",
        f"Maximum dynamic pressure: {summary.get('max_dynamic_pressure_pa', 0):.0f} Pa",
        f"Burn time: {summary.get('burn_time_s', 0):.2f} s, total impulse {summary.get('total_impulse_ns', 0):.1f} N*s",
        f"Launch mass: {summary.get('launch_mass_kg', 0)*1000:.0f} g, dry {summary.get('dry_mass_kg', 0)*1000:.0f} g",
        f"Static margin: {summary.get('static_margin_m', 0)*1000:.1f} mm",
        f"Landed: {'yes' if summary.get('landed') else 'no'} in {summary.get('flight_time_s', 0):.1f} s",
    ]
    report.assumptions = [
        "6-DOF rigid-body integration with a point-mass propellant; no structural flexibility.",
        "Aerodynamic coefficients are the project's configured values, not necessarily measured.",
        "The TVC autopilot uses true attitude/rates (no sensor/estimator in the loop); SITL/HIL exercise that path.",
    ]
    report.warnings = list(summary.get("warnings", [])) + [d.message for d in result.diagnostics]
    return report


def project_report(project) -> Report:
    report = Report(f"Project report: {project.name}")
    stats = project.statistics()
    report.lines = [
        f"Parts: {stats['parts']}   Components: {stats['components']}   Constraints: {stats['constraints']}",
        f"Fasteners: {stats['fasteners']}   Materials: {stats['materials']}   Motors: {stats['motors']}",
        f"Project file: {project.path or '(unsaved)'}",
    ]
    report.tables["Libraries"] = [
        {"kind": "material", "name": m.name, "provenance": str(m.provenance)}
        for m in project.materials.all()
    ][:40]
    report.assumptions = [
        "Generated from the currently loaded project; nothing is drawn from a network service.",
    ]
    return report
