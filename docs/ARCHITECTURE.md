# Architecture

TVC Engineering Studio is one desktop process with fifteen modules that talk through
narrow interfaces. The CAD assembly is the single source of truth: every physical number
used by the mechanical, electronics and flight simulators is derived from the assembled
components and their configured properties, never from a hand-entered whole-rocket summary.

```
                         ┌───────────────────────────────────────────┐
                         │              UI (PySide6)                 │
                         │  CAD | Assembly | Fastener | Component    │
                         │  Mechanical | Electronics | Flight | Analysis │
                         └───────────────┬───────────────────────────┘
                                         │ commands / signals
      ┌──────────────────────────────────▼──────────────────────────────────┐
      │                     Application services layer                      │
      │  ProjectController · UndoStack · SelectionService · SignalBus        │
      └───────┬───────────────┬──────────────┬──────────────┬───────────────┘
              │               │              │              │
      ┌───────▼──────┐ ┌──────▼───────┐ ┌────▼───────┐ ┌────▼────────────┐
      │ CAD document │ │  Assembly    │ │ Fastener   │ │ Component       │
      │ + kernel     │ │  + solver    │ │ system     │ │ properties      │
      │ occt_backend │ │ constraints  │ │ thread/install │ materials    │
      └───────┬──────┘ └──────┬───────┘ └────┬───────┘ └────┬────────────┘
              │               │              │              │
              └───────────────┴──────┬───────┴──────────────┘
                                     │  assembled state + mass properties
                     ┌───────────────▼────────────────┐
                     │      Analysis / Simulation      │
                     │ mech.dynamics (SI solver)       │
                     │ electronics (FC + IMU + SITL)   │
                     │ physics (propulsion, aero, 6DOF)│
                     └───────────────┬─────────────────┘
                                     │ telemetry / state frames
                            ┌────────▼────────┐
                            │  viz viewport   │
                            └─────────────────┘
```

## Modules

| Module | Responsibility | Depends on |
|--------|----------------|------------|
| `core.math3d` | Vectors, quaternions, transforms, inertia algebra, mass-property composition | numpy |
| `core.units` | SI storage, display conversion, formatting | — |
| `core.interfaces` | Protocols, `Provenance`, `GeometryRepr`, `Mesh`, `Measurement`, `Diagnostic`, `SupportLevel` | core.math3d |
| `cad.occt_backend` | Exact B-rep geometry, booleans, fillets, STEP/IGES/STL/BREP, tessellation, mass properties, measurements | OCP (OCCT) |
| `cad.document` | Part documents, parametric feature history, rebuild pipeline, representation cache | occt_backend |
| `cad.features` | Feature definitions (extrude, revolve, hole, fillet, chamfer, pattern, boolean, import) | document |
| `cad.sketch` | 2-D sketch entities + dimensional/geometric constraint solver | scipy |
| `assembly.model` | Component instances, hierarchy, subassemblies, placement, configurations | core, cad |
| `assembly.constraints` | Mate types, numeric constraint solver, DOF/conflict diagnostics | scipy |
| `assembly.massprops` | Assembly COM/inertia with parallel-axis theorem, per-component breakdown | core.math3d, components |
| `components.materials` | Material library with provenance and print-process caveats | — |
| `components.properties` | Mass modes, overrides, extra mass, mechanical property block | — |
| `components.hardware` | Servo, motor, IMU/sensor, board definitions with real dynamic parameters | — |
| `fasteners.catalog` | Metric/imperial fastener standards, heads, drives, inserts, custom fasteners | — |
| `fasteners.thread` | Thread profile math, helix geometry, engagement/strip/pull-out estimates | occt_backend (detailed geometry) |
| `fasteners.install` | Installation kinematics, seating, torque model, interference checks | catalog, thread, cad |
| `mech.dynamics` | Rigid-body world with joints, limits, friction, backlash, contact | core.math3d |
| `mech.servo` | Servo plant: two-inertia gear model with drive, rate and stall limits, backlash, deadband, stops; stability sub-stepping | — |
| `mech.contact` | Mesh-based narrow-phase contact and static interference analysis | core.interfaces |
| `electronics.flightcomputer` | Control loop scheduling, sensor sampling, actuator output, logging, power | hardware |
| `electronics.imu` | MPU6050-class measurement synthesis at the sensor's mounting pose | core.math3d |
| `electronics.sitl` | Software-in-the-loop loop closure between FC and flight plant | flightcomputer, physics |
| `electronics.hil` | Framed serial protocol for a real Pico 2 W (COBS + CRC-16) | pyserial (optional) |
| `physics.atmosphere` | ISA atmosphere + wind | — |
| `physics.propulsion` | Thrust curves with provenance and mass flow | — |
| `physics.aero` | Coefficient model, Barrowman CP estimate, damping, uncertainty flags | core.math3d |
| `physics.flight` | 6-DOF RK4 (quaternion attitude, changing mass/inertia) | physics.*, assembly |
| `viz.*` | GL viewport, picking, section/explode, overlays, playback | pyqtgraph.opengl |
| `project.*` | Versioned project schema, JSON save/load, migration, library store | everything (serialization only) |
| `analysis.reports` | Engineering reports, comparisons, clearance/collision/fastener breakdowns | assembly, mech |
| `ui.*` | The eight workspaces and reusable panels | all of the above |

## Key decisions (and why)

1. **SI everywhere internally, units only at the edges.** Project files store metres,
   kilograms and radians. The UI converts on display. This removes an entire class of
   unit-mixing bugs from the solvers, and makes saved projects comparable.

2. **OCCT through `cadquery-ocp` as the exact kernel.** OCCT is the most mature open B-rep
   kernel with STEP/IGES/STL support and correct mass properties. `cadquery-ocp` ships
   OCCT as a wheel, so no conda or compiler toolchain is needed on Windows. We do not
   convert imports to meshes: exact B-rep is retained and meshes are cached per
   representation (`GeometryRepr.EXACT/VISUAL/COLLISION/SIMULATION`).

3. **Four geometry representations, never silently substituted.** Mass properties and
   booleans always use EXACT. Drawing uses VISUAL tessellation. Contact uses COLLISION.
   If the kernel is unavailable, features raise `CadKernelUnavailable` instead of quietly
   degrading to approximations.

4. **Our own rigid-body/joint solver rather than MuJoCo.** The mechanical model needs
   joint-level features MuJoCo does not express directly: screw joints coupling rotation to
   translation, backlash gaps, gear-train stiffness, servo torque/rate limits with deadband,
   and per-joint friction that is reported back to the engineer. A sequential-impulse
   solver in generalized joint coordinates gives those exactly and is validated against
   closed-form cases (pendulum period, energy conservation, sliding friction, pitch
   coupling). MuJoCo also lacks a Python 3.13 Windows wheel in the evaluated version.
   The `DynamicsBackend` interface leaves room to add a MuJoCo cross-check later.

5. **`pyqtgraph.opengl` for the viewport.** Panda3D has no current Windows wheel for the
   target Python and drags in its own windowing; a Qt-native GL viewport keeps one event
   loop, one widget hierarchy, and lets CAD tessellation render directly from NumPy arrays.
   Camera orbit/pan/zoom, orthographic/perspective, section clipping, wireframe and picking
   are implemented on top of it.

6. **JSON project format with explicit schema version and migration functions.** Human
   inspectable, diffable, no binary-lock-in, and each subsystem serializes its own state
   through a `to_dict`/`from_dict` pair. Geometry that cannot be rebuilt from features is
   embedded as BREP (exact) rather than mesh.

7. **Provenance is a first-class field.** `Provenance` travels with every mass property,
   material and thrust curve. Reports print it. This is how the application avoids
   presenting estimates as validated facts.

8. **No decorative settings.** Every property in the component editor is consumed by a
   solver, a diagnostic or a report; the property table in `docs/USER_GUIDE.md` names the
   consumer for each field.

## Data flow at simulation time

```
Assembly (components, constraints, fasteners)
   → assembly.massprops  → total mass, COM, inertia tensor (world frame)
   → physics.propulsion.mass_flow(t)  → mass/inertia time history
   → mech.dynamics        → TVC gimbal angle for commanded servo
   → electronics.imu      → gyro/accel at IMU mounting pose
   → electronics.flightcomputer → servo commands (rate-limited loop)
   → physics.aero + propulsion → forces/moments at COM
   → physics.flight (RK4)  → state
   → viz.viewport          → pose of every component (no independent animation)
```

## Threading model

Geometry rebuilds and long simulations run on a `QThreadPool` worker; results are applied
to the document on the GUI thread through queued signals. The viewport renders from an
immutable snapshot so a running simulation cannot tear the drawing.

## Failure behaviour

Solver diverges, mass is unknown, contact geometry missing, thrust curve unverified: each
produces a `Diagnostic` with severity and an actionable message, surfaced in the status
panel and in the report. Nothing is silently clamped without saying so.
