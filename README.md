# TVC Engineering Studio

Integrated CAD → assembly → fastener → mechanical dynamics → electronics → propulsion →
aerodynamics → 6-DOF flight simulation desktop application for Thrust Vector Control rockets.

Native Windows desktop application (PySide6 + Open CASCADE). No browser, no cloud service,
no subscription, no telemetry. Projects are local files.

---

## Download and install (Windows 10/11, no Python needed)

1. Download `TVC-Engineering-Studio-0.1.0-setup.exe` from the
   [latest release](https://github.com/DVVagram/tvc-engineering-studio/releases/latest).
2. Run it. Windows SmartScreen will warn because the build is not code-signed:
   choose **More info → Run anyway**.
3. It installs for the current user (no administrator rights, no background services,
   no network access) and adds a Start-menu shortcut.
4. Remove it later from **Settings → Apps → TVC Engineering Studio** or the Start-menu
   entry's *Uninstall*.

Your projects are ordinary files you choose the location of; they are not stored in the
installation folder, so uninstalling never deletes them.

## What this build does

| # | Subsystem | State |
|---|-----------|-------|
| 1 | Exact B-rep CAD kernel (OCCT/OCP): primitives, booleans, fillet/chamfer, extrude/revolve, STEP/IGES/STL/BREP I/O, tessellation, measurements | Implemented |
| 2 | Parametric part documents with feature history, sketch + 2-D constraint solver, rebuild | Implemented |
| 3 | Assembly tree with subassemblies, placement, visibility/suppression, configurations | Implemented |
| 4 | Assembly constraint solver (mate types, DOF/over-constraint diagnostics) | Implemented |
| 5 | Component property system: mass modes, overrides, materials, parallel-axis assembly mass properties | Implemented |
| 6 | Fastener library, screw installation kinematics, simplified + helical threads, self-tapping model | Implemented |
| 7 | Mechanical dynamics: rigid bodies, joints, stops, friction, backlash, contact, interference | Implemented |
| 8 | Servo / motor / sensor hardware models with real dynamics | Implemented |
| 9 | Flight computer, IMU simulation, SITL loop, HIL serial protocol | Implemented |
| 10 | Propulsion (thrust-curve import with provenance), atmosphere, aero, 6-DOF flight with changing mass/inertia | Implemented |
| 11 | 3-D viewport: selection, exploded/section/transparent/wireframe, animation driven by simulation state | Implemented |
| 12 | Telemetry, engineering graphs, analysis reports, configuration comparison | Implemented |
| 13 | Automated validation test suite (see `docs/ACCURACY.md`) | Implemented |
| 14 | Windows packaging script (PyInstaller) + installer recipe | Scripted, not signed |
| 15 | Nonlinear FEA of thread forming / plastic deformation | **Not implemented** — documented model boundary |
| 16 | Geometry-derived aerodynamic coefficient prediction beyond Barrowman/CP estimate | **Partial** — documented as approximate |

Everything else in the specification that is not listed as implemented is either
*partial* or *planned*, and `docs/ROADMAP.md` tracks it per feature. Panels carry the
same status glyphs (`●` implemented, `◐` partial, `○` planned, `✕` unsupported) so the
UI never implies more capability than exists.

---

## Install from source (developers)

Requires Python 3.13 (3.11 or newer works) and about 1.5 GB of wheels.

```bat
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m tvc_studio
```

`python -m tvc_studio` starts the desktop application. First launch creates an empty
project with a starter rocket assembly you can explore.

Optional extras:

```bat
.venv\Scripts\python -m pip install -e ".[hardware]"   :: USB serial HIL with a real Pico
```

## Tests

```bat
.venv\Scripts\python -m pytest
```

The suite validates the mathematics that the application depends on — coordinate
transforms, parallel-axis inertia, sketch and assembly constraints, CAD volumes,
thread/insertion kinematics, servo response, sensor output, passive and guided flight,
aerodynamics, project round-trips and reports — against closed-form or independently
computed results rather than against saved output. `docs/ACCURACY.md` maps each claim to
the test that proves it.

## Project files

Projects are `.tvcproj` JSON documents (schema-versioned, migrated on load) that embed
part definitions, feature history, assembly hierarchy, constraints, installed fasteners,
materials, overrides, electronics, motor data, simulation settings and results.

Reusable hardware lives in a component library under `%APPDATA%\TVCStudio\library`
(configurable), and a bundled read-only library ships with the application.

## Packaging

```bat
packaging\build_windows.bat
```

Produces `dist\TVC Engineering Studio\TVC Engineering Studio.exe` (onedir, Qt and OCCT
DLLs collected) plus an Inno Setup script for the installer. The build is unsigned;
Windows SmartScreen will warn until you sign it.

## Documentation

* `docs/ARCHITECTURE.md` — subsystem map, interfaces, data flow, decisions.
* `docs/DEPENDENCIES.md` — evaluated libraries, licences, why each was chosen.
* `docs/ROADMAP.md` — phases and per-feature status.
* `docs/ACCURACY.md` — what is validated, what is approximate, what is not claimed.
* `docs/USER_GUIDE.md` — workspace-by-workspace guide.

## Engineering honesty rules baked into the code

* Mass properties are computed from exact B-rep geometry + material density and carry a
  `provenance` tag (`cad_derived`, `measured`, `manufacturer`, `estimated`, `user_override`).
  A manual override is labelled in the UI and in every report; it is never presented as derived.
* Estimated material strength, friction, thread engagement and aerodynamic coefficients are
  flagged as estimated wherever they influence a result.
* Settings that do not feed a calculation do not ship (part 4.4 / rule 10).
* Visualization is sampled from the simulation state, never animated independently (rule 11).

## Licence

MIT for this application's code. Open CASCADE (LGPL-2.1 with exception), Qt/PySide6 (LGPL-3),
NumPy/SciPy (BSD), pyqtgraph (MIT), PyOpenGL (BSD). See `docs/DEPENDENCIES.md`.
