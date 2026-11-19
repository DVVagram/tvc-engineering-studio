# Development roadmap and feature status

Status legend: **●** implemented and tested · **◐** partial (usable, documented gaps) ·
**○** planned (interface exists, implementation pending) · **✕** unsupported (documented reason).

## Phase 1 — Architecture and CAD foundation  ●

| Feature | Status |
|---------|--------|
| Repository, packaging metadata, venv bootstrap | ● |
| Architecture + dependency evaluation documents | ● |
| OCCT kernel integration, exact B-rep primitives | ● |
| Booleans, fillet/chamfer, extrude/revolve, patterns | ● |
| STEP / IGES / STL / BREP import-export | ● |
| Separate exact/visual/collision/simulation representations | ● |
| Part documents, feature history, rebuild | ● |
| 3-D viewport with camera, shading, wireframe, ortho/perspective | ● |
| Entity selection (solid/face/edge/vertex) + highlighting | ● |
| Measurements (length, radius, diameter, angle, area, volume, distance, clearance, bounding box, COM, inertia) | ● |

## Phase 2 — CAD creation and assembly  ●

| Feature | Status |
|---------|--------|
| Sketch entities (line, circle, arc, rectangle, polygon) | ● |
| Sketch constraints (coincident, horizontal, vertical, distance, radius, parallel, perpendicular, tangent, equal) | ● |
| Extrude/revolve/hole features from sketches | ● |
| Assembly tree, subassemblies, duplicate/replace/suppress/hide | ● |
| Interactive placement (move/rotate gizmos, numeric transform) | ● |
| Assembly constraints: fixed, revolute, slider, cylindrical, ball, planar, parallel, perpendicular, concentric, coincident, distance, angle, screw | ● |
| DOF analysis, over/under-constrained and conflict diagnostics | ● |
| Assembly configurations, save/load | ● |

## Phase 3 — Component and fastener systems  ●

| Feature | Status |
|---------|--------|
| Physical property editor, mass modes (CAD / manual / CAD+extras) | ● |
| Material library with provenance and print caveats | ● |
| Mass, COM, inertia per component and per assembly, breakdown table | ● |
| Fastener library: metric/imperial, machine/self-tapping/self-drilling, bolts, nuts, washers, inserts, custom | ● |
| Interactive screw installation with rotation + pitch advance | ● |
| Simplified thread + detailed helical thread geometry | ● |
| Self-tapping/self-threading model: pilot hole, forming torque, strip risk, pull-out estimate | ● (reduced-order, documented) |
| Fastener mass automatically included in assembly properties | ● |
| Thread engagement/interference inspection | ● |
| Nonlinear FE of thread forming | ✕ (documented boundary — no FEA solver is claimed) |

## Phase 4 — Mechanical simulation  ●

| Feature | Status |
|---------|--------|
| Rigid-body world with arbitrary joint graph, sequential-impulse solver | ● |
| Revolute/slider/cylindrical/ball/planar/screw/fixed joints | ● |
| Joint limits/stops, friction, damping, stiffness, backlash | ● |
| Contact detection (convex SAT + triangle narrow phase), collision highlighting | ● |
| Static interference / clearance reports | ● |
| Collision filtering for intended contact | ● |
| Servo plant model (rate, stall torque, backlash, deadband, gear compliance, stops, current) | ● |
| Motor/actuator model | ● |
| Momentum/energy validation tests | ● |

## Phase 5 — Electronics  ●

| Feature | Status |
|---------|--------|
| Pico 2 W board definition, editable | ● |
| MPU6050 model: ranges, resolution, noise, bias walk, drift, delay, axis misalignment | ● |
| Two MG90S servos with editable parameters | ● |
| Control-loop timing, sensor sampling, processing delay, missed updates | ● |
| Power budget and current limits | ● |
| SITL: simulated sensors → controller → servo commands → plant | ● |
| HIL: framed USB-serial protocol (COBS + CRC-16) with timing statistics | ● |
| Executing the *actual* Pico firmware inside the app | ✕ (we drive real hardware over serial instead; documented) |

## Phase 6 — Rocket physics  ●

| Feature | Status |
|---------|--------|
| Thrust curve import (RASP `.eng`, CSV) with provenance | ● |
| Burn duration, total impulse, mass flow, changing mass/inertia | ● |
| ISA atmosphere + wind | ● |
| Coefficient aero: drag, lift, moments, damping, angle of attack | ● |
| Barrowman centre-of-pressure estimate from stack geometry (flagged approximate) | ● |
| 6-DOF quaternion flight, TVC gimbal actuation and misalignment | ● |
| Recovery/parachute phase | ● |
| Measured aero coefficient entry, uncertainty reporting | ● |
| Full CFD or panel-method aero | ○ (interface reserved: `AeroModel`) |

## Phase 7 — Integration  ●

| Feature | Status |
|---------|--------|
| 3-D flight animation driven by solved state | ● |
| Telemetry panels + live graphs | ● |
| Engineering analysis reports, CSV export | ● |
| Configuration comparison | ● |
| Automated validation suite | ● |
| Undo/redo, multi-document tabs | ● |

## Phase 8 — Windows distribution  ◐

| Feature | Status |
|---------|--------|
| PyInstaller onedir build script, OCCT/Qt DLL collection | ● (scripted; see `dist/` after a build) |
| Inno Setup installer recipe | ● (scripted) |
| Code signing | ✕ (no certificate); SmartScreen will warn |
| Example projects | ● (tutorial project is built into the app) |
| User guide | ● `docs/USER_GUIDE.md` |

## Known gaps kept visible

* Thread forming / stripping is a reduced-order empirical model; the UI labels its outputs
  "estimated". No FEA claim is made.
* Chord-wise aerodynamic coefficients from geometry are a Barrowman-style estimate; the
  application says so and lets the user enter measured coefficients instead.
* Hardware-in-the-loop requires a real board; the simulation path is SITL.
* The dynamics solver is validated for the mechanisms it ships with (free hinge, slider,
  screw, servo-driven gimbal); it is not a general contact-rich multibody package.
* The mechanics world is a bench fixture: unconstrained bodies are fixed, gravity is off,
  and servo joints are kinematically driven with estimated, clamped torque.
* The static-margin convention is body `z` from the tail: positive `CG_z − CP_z` is stable.
