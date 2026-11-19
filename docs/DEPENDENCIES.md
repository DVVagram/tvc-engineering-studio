# Dependency evaluation

Every dependency below was checked for a Windows wheel on the target interpreter
(Python 3.13 x64), licence compatibility and whether it duplicates functionality we
would otherwise have to write. Versions are pinned in `pyproject.toml` as ranges.

| Need | Chosen | Version | Licence | Why |
|------|--------|---------|---------|-----|
| Exact B-rep CAD kernel | `cadquery-ocp` (Open CASCADE 7.9 via OCP bindings) | 7.9.x | LGPL-2.1 + OCCT exception | Only mature open B-rep kernel with real STEP/IGES/STL, correct volume/inertia, fillets and booleans. Prebuilt Windows wheel; no compiler needed. |
| GUI | `PySide6` (Qt 6) | 6.12 | LGPL-3 | Native Windows look, dockable panels, mature model/view widgets, official Qt bindings. |
| Numerics | `numpy`, `scipy` | 2.x / 1.18 | BSD | Vector math, `least_squares` for constraint solving, linear algebra. |
| 3-D viewport | `pyqtgraph` + `PyOpenGL` | 0.14 / 3.1 | MIT / BSD | Qt-native OpenGL widget; integrates with the single application event loop; renders NumPy meshes directly. |
| 2-D graphs | `pyqtgraph` | 0.14 | MIT | Fast live plots for telemetry with no extra toolkit. |
| Tests | `pytest` | 9.x | MIT | Standard; fixtures for kernel-optional tests. |
| Serial HIL | `pyserial` (optional extra) | 3.5 | BSD | Talking to a real Pico 2 W over USB CDC. |

## Alternatives considered and rejected

**FreeCAD libraries (Part/PartDesign).** Excellent kernel use, but importing FreeCAD
requires the FreeCAD binary tree, its own Qt build and `pivy`; embedding it in a
standalone Qt application on Windows is brittle and drags a second Qt. We use the same
underlying kernel (OCCT) directly, which is what FreeCAD's Part module wraps. FreeCAD's
Assembly4/Sketcher concepts informed our constraint and sketch models.

**pythonocc-core.** Conda-only builds for Windows; our distribution target is a
pip-installable, PyInstaller-packable application. `cadquery-ocp` provides the same OCCT
via OCP wheels.

**CadQuery / build123d high-level API.** Great productivity layer, but version pins to
specific OCP builds and hides the B-rep entities (faces/edges/vertices) that selection,
measurement and constraint entities need. We call OCP directly and keep a thin, documented
`OCCTGeometry` façade. CadQuery remains a reference for correctness of common operations.

**MuJoCo.** Rejected as the primary dynamics engine: no Windows wheel for the target
interpreter at evaluation time, and it has no first-class screw joint, backlash gap,
gear-train compliance or servo model, all of which the TVC assembly requires. Our
sequential-impulse solver implements those and is validated against closed-form cases.
`mech.dynamics.DynamicsWorld` is behind an interface so a MuJoCo cross-check can be added.

**Panda3D.** No current Windows wheel for the target Python; separate render loop and
window, which fights Qt docking. `pyqtgraph.opengl` provides orbit/pan/zoom, ortho and
perspective, per-item shaders and picking inside the Qt event loop.

**OpenRocket / RocketPy source integration.** Their file formats and physical models are
used as *references for validation* (thrust curves in RASP `.eng` format, Barrowman CP
method, ISA atmosphere), not as runtime dependencies — both are heavy and neither models a
TVS mechanism. `python/rust` interop would not add accuracy to what we implement and test.

**Assimp/trimesh for collision.** `trimesh` was considered for mesh collision; our contact
needs are convex-part based and small (a handful of moving parts), so `mech.contact`
implements separating-axis tests and triangle-pair narrow phase directly, with no extra
dependency.

## Platform notes

* Python 3.13 x64 is the packaging target (all wheels available). 3.11/3.12 also work.
* Qt6 requires Windows 10 1809+; the packaged build ships the required VC++ runtime DLLs.
* OCCT DLLs are collected by the PyInstaller spec in `packaging/`.
* No internet access is required at runtime. Nothing in the app phones home.
