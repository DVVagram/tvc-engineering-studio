# TVC Engineering Studio — user guide

This guide walks through the eight workspaces in the order a TVC rocket project normally
uses them. All numbers are SI internally (metres, kilograms, seconds, radians); the UI
shows millimetres, grams and degrees where that is the engineering convention.

## First start

* `python -m tvc_studio` opens the tutorial project (a small TVC rocket) on first launch.
  `File → New project` starts an empty project; `File → Open sample TVC rocket` reloads the
  tutorial.
* The **3D view** dock is always live. Drag with the left mouse button to orbit, use the
  wheel to zoom, `F` (or the toolbar **Fit**) frames the assembly.
* The status bar shows total mass and its provenance, plus the assembly solver's DOF
  summary. A red/amber DOF summary is a warning, not a decoration: it means the mate set is
  under- or over-constrained.
* `Help → Capability status` lists exactly what is implemented, partial and not
  implemented. The same list is in `README.md` and `docs/ROADMAP.md`.

## Projects and files

* `File → Save` writes a `.tvcproj` JSON document. It stores part feature history, exact
  BREP geometry (base64), assembly hierarchy, mates, fasteners, materials, overrides,
  motor curves, electronics and simulation settings.
* Saving is atomic (temporary file + replace). Loading rejects files written by a newer
  schema rather than guessing.
* Undo/redo (`Ctrl+Z` / `Ctrl+Y`) snapshots the whole project before each edit.
* The component library lives in `%APPDATA%\TVCStudio\library.json` and is shared between
  projects.

## 1. CAD workspace

1. **Parts** lists every part document. *New part* creates an empty document; double-click
   a part to edit it.
2. Add features from the feature list (box, cylinder, cone, tube, sphere, extruded
   sketch, revolve, hole, fillet, chamfer, polar/linear pattern). Each feature has an
   operation: `new`, `join`, `cut` or `intersect`. The first feature must be `new`;
   anything else is an error, not a silent new body.
3. Features rebuild in order. A failed feature is shown with its diagnostic and the part
   keeps its last good shape — it is never replaced by a mesh or a guess.
4. *Sketch* opens the 2-D solver. Draw lines/rectangles/circles/arcs, add constraints
   (coincident, horizontal, vertical, distance, radius, parallel, perpendicular, tangent,
   equal) and solve. The solver reports residual and remaining DOF.
5. **Import** accepts STEP, IGES, STL and BREP. STEP/IGES/BREP stay exact B-rep; STL is a
   triangle shell and is labelled as such. **Export** writes STEP or the embedded BREP.
6. Mass properties are computed from the exact B-rep and the part's material. Changing the
   material updates the assembly mass immediately.

## 2. Assembly workspace

1. Drag components from the part list (or use *Add component*) into the tree.
2. Parent-child transforms form subassemblies; the tree order is the transform order.
3. Set a component's placement numerically or with the move/rotate fields. Ground exactly
   one component per rigid group (the solver refuses to invent a ground).
4. Add mates in the constraint table: fixed, revolute, slider, cylindrical, ball, planar,
   parallel, perpendicular, concentric, coincident, distance, angle, screw. Each mate
   selects entities (point, axis, face, plane) on two components.
5. **Solve** runs the numerical constraint solver and reports residual, equation count,
   rank and remaining DOF. Selecting a row highlights the involved components in the
   viewport.
6. The component property editor sets material, mass mode (CAD / manual / CAD + extras
   such as glue or wiring), extra mass, COM and inertia overrides. Every override is
   labelled in the mass table and in reports.

## 3. Fasteners workspace

1. Holes come from the CAD hole features (or are defined on a component). Each hole carries
   position, axis, diameter, depth, material and whether it is pre-threaded.
2. Pick a catalogue fastener (machine screw, self-tapping/thread-forming, bolt, nut,
   washer, insert, custom) and a hole, then **Install**. The installer advances the screw by
   `pitch × turns`, clamps to the hole depth and records the engagement mode.
3. The engagement report gives engaged length, clamp-force estimate from torque, strip
   torque and pull-out estimate. These are reduced-order estimates and are labelled
   **estimated**; no FEA is performed.
4. Installed fasteners appear in the viewport as cylinders and in the assembly mass
   breakdown as their own rows.

## 4. Mechanics workspace

1. **Build mechanism** converts the assembly into a rigid-body world. Bodies with mates
   become participants; everything else is treated as rigidly mounted. Gravity is off:
   this is a bench fixture for checking mechanism motion, not a dropped assembly.
2. Joints come from assembly mates (revolute, slider, etc.). Servo-driven joints are
   kinematically driven: the servo's rate/torque limits shape the commanded motion and the
   required torque is estimated and clamped. They are not torque sources coupled back into
   the solver; the capabilities dialog says so.
3. Use the servo sliders to command angles. The viewport shows the solved pose after every
   step, so what you see is the simulated state.
4. Contact detection reports interference between components. Intended contacts must be
   filtered or they will be reported.

## 5. Electronics workspace

1. Choose the board (Pico 2 W), IMU (MPU6050) and servos (MG90S by default). Every
   parameter is editable: sample rate, ranges, resolution, noise density, bias drift,
   mounting rotation, delay, servo speed/torque/backlash/deadband, supply voltage.
2. **IMU test** produces samples from the model at rest and shows mean/σ for accel and
   gyro. A stationary sensor reads +1 g of specific force; that is the convention.
3. **SITL** closes the loop: simulated IMU → flight computer → servo plants → flight
   plant. The report lists frames, loop rate, missed deadlines and command statistics.
4. **HIL** speaks the framed serial protocol (COBS + CRC-16) to a real Pico over USB. Use
   *Loopback* to exercise the protocol without hardware; the UI labels loopback mode.

## 6. Propulsion workspace

1. **Import motor** reads a RASP `.eng` or CSV thrust curve. The curve is stored with its
   source and provenance; nothing is invented.
2. The motor table shows burn time, total impulse, peak thrust, propellant mass and the
   derived mass-flow curve. The tutorial motor is synthetic and named as such.
3. Enter nozzle exit diameter, gimbal limits and TVC servo mapping. The gimbal range is
   enforced by the flight controller and shown on the flight plots.
4. Atmosphere uses the ISA model with optional wind. Altitude density/pressure/sound speed
   are plotted so you can see the flight envelope.

## 7. Flight workspace

1. Set launch rail length, initial attitude, wind, parachute deployment altitude and drag,
   control gains and the autopilot mode.
2. **Run** integrates the 6-DOF equations with RK4 at the configured timestep. Mass and
   inertia change as propellant burns; thrust is applied at the nozzle; aero forces act at
   the centre of pressure; TVC commands pass through the servo plants.
3. The static-margin readout uses body `z` measured from the tail: `static_margin =
   CG_z − CP_z`. Positive means the CG is ahead of the CP (stable). The CP is a Barrowman
   estimate from the stack geometry and is labelled approximate.
4. Results: apogee, max speed, max dynamic pressure, tilt history, gimbal history, and
   landing state. The 3D view animates the recorded state; it never invents motion.
5. **Export** writes the telemetry CSV and a text/JSON report.

## 8. Analysis workspace

* Mass breakdown (component, fastener, subassembly, override flags), fastener engagement,
  clearance/interference pairs, joint DOF, aero summary and flight summary.
* Reports can be saved as text or JSON. Numbers carry their provenance; estimates are
  explicitly marked.
* Compare configurations by duplicating a project file and running the same reports.

## Units and conventions

| Quantity | Internal | UI |
|----------|----------|----|
| Length | m | mm |
| Mass | kg | g |
| Force | N | N |
| Torque | N·m | N·m / mN·m |
| Angle | rad | ° |
| Inertia | kg·m² | kg·m² |

* Body `+Z` points toward the nose; the origin is the tail datum unless a part says
  otherwise.
* Positive pitch is nose-up about body `+Y`; positive yaw is about body `+Z` per the
  right-hand rule.
* A positive static margin (`CG_z − CP_z`, z from the tail) is stable.

## Known limitations (also shown in Help → Capability status)

* No FEA, thermal, combustion or internal-ballistics solver.
* The mechanics world is a bench fixture: unconstrained bodies are fixed and gravity is
  off; servo joints are kinematically driven with estimated torque.
* Aerodynamics is coefficient-based with a Barrowman CP estimate; no CFD, transonic or
  supersonic corrections.
* Contact is rigid mesh contact; no compliance, wear or deformation.
* Section views cut triangles and do not cap the cut face.
* HIL requires a real Pico and firmware speaking the documented protocol; loopback only
  validates the protocol, not the firmware.

## Troubleshooting

* **"CAD backend unavailable"** — the `cadquery-ocp` wheel is missing or not supported on
  this Python/platform. Reinstall with `pip install -e ".[dev]"` on 64-bit Windows.
* **Blank 3D view** — check that OpenGL 2.1+ drivers are installed; the viewport falls back
  to wireframe when shaders are unavailable.
* **Project will not open** — the file was written by a newer schema. The error names the
  version; open it with the matching release.
* **HIL will not connect** — close the Arduino serial monitor, confirm the COM port and
  baud rate (115200 default), and check that the board is running the TVC firmware.
