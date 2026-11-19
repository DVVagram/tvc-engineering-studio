# Static Rocket Flight Simulator

A 3D flight simulator for **one fixed fin-stabilised rocket**. There is no thrust vectoring,
no control system and no rocket designer: the airframe and motor are constants, and the only
things you can change are the motor's *performance* numbers. Change them and watch the same
rocket fly differently.

## Run it

Windows:

```bat
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m rocket_sim
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m rocket_sim
```

The window opens with the rocket on the pad. Press **Start** and it flies: powered ascent,
burnout, coast, apogee, descent and impact, with the 3D view following the simulated motion.

## Interface

- **Large 3D viewport** — orbit with the left mouse button, zoom with the wheel, pan with the
  right mouse button. **Follow rocket** keeps the camera on the vehicle; **Reset camera**
  returns to the pad view.
- **Motor performance panel** — the only editable numbers in the application.
- **Flight data** — altitude, vertical and total velocity, acceleration, flight time, thrust,
  motor status, maximum altitude and maximum velocity, updated every frame.
- **Flight events** — liftoff, burnout, apogee and ground impact with their times.
- **Controls** — Start, Pause, Resume, Reset and Restart. Reset returns to the exact initial
  state, so the same motor flies the identical trajectory every time.

## The rocket and motor are fixed

All physical values live in `rocket_sim/config.py` as frozen dataclasses. Nothing in the
interface reads or writes them, and changing motor performance cannot change them.

| Rocket | Value |
|---|---|
| Body | 1.20 m long, 75 mm diameter |
| Nose | 200 mm ogive |
| Fins | 4 fixed trapezoidal fins, 160/60 mm chord, 90 mm span, 60 mm sweep |
| Structure mass | 615 g |
| Centre of gravity | 620 mm from the nose tip |
| Centre of pressure | 820 mm from the nose tip (static margin 2.7 calibers) |
| Launch tilt | 4° (fixed; no wind is modelled) |

| Motor | Value |
|---|---|
| Case | 29 mm diameter, 100 mm long |
| Casing mass | 45 g |
| Propellant mass | 30 g |
| Mount | Fixed, 12 mm ahead of the tail, one motor |

## The only editable values: motor performance

| Input | Meaning |
|---|---|
| Designation | Optional label shown with the run |
| Average thrust | N |
| Maximum thrust | N |
| Total impulse | N·s |
| Burn time | s |
| Curve shape | Neutral, progressive or regressive |

The thrust curve is **estimated** from these numbers and labelled as such in the interface —
it is not measured motor data. Average thrust × burn time must agree with the total impulse
to within 1 %, the peak must be at least the average, and a single linear ramp cannot peak
above twice its mean; anything inconsistent is rejected with an explanation rather than
silently corrected. The **Set impulse = average × burn time** button is the explicit way to
resolve a mismatch.

Changing any of these values leaves the motor's diameter, length, casing mass, propellant
mass and mounting position untouched, and the rocket's mass and geometry identical.

## Flight model

`rocket_sim/physics.py` integrates a single rigid body with classical RK4 at a fixed 2 ms
step. Forces and moments:

- thrust along the body axis from the sampled thrust curve, with propellant mass burned
  linearly over the burn time;
- gravity;
- drag from an exponential atmosphere (scale height 8.5 km) with a simple transonic drag rise;
- a fin normal force applied at the centre of pressure, which weathercocks the rocket into
  its velocity vector, plus pitch damping;
- a pad hold that releases the rocket only when thrust along the rail exceeds its weight.

There is no guidance, no autopilot, no gimbal and no parachute. A flight ends at ground
impact. `docs/PHYSICS.md` lists the equations and the assumptions behind the coefficients.

## Checks

```bash
python -m rocket_sim --self-test   # headless physics checks, prints a summary and exits
python -m pytest                   # 36 tests: config, motor validation, flight, UI smoke test
```

`--self-test` reports the fixed hardware, the thrust-curve integral, the complete flight
(apogee, maximum speed, impact time), repeatability, a low/high impulse comparison, and that
an inconsistent motor specification is rejected.

## Packaging

```bat
packaging\build_windows.bat        :: onedir build in dist\
packaging\build_macos.sh           :: .app bundle plus a ditto archive
```

`packaging/rocket_sim.spec` is the PyInstaller recipe and `packaging/installer.iss` the Inno
Setup 6 recipe for a per-user Windows installer. `.github/workflows/build.yml` runs the
self-test, the tests and the packaging step on Windows and macOS runners.

## Limitations

- One rocket, one motor, one stage; no recovery system, wind, or multi-stage modelling.
- Aerodynamics uses textbook estimates; there is no CFD, FEA or measured aero table.
- Estimated curves are linear profiles; measured motor files are not supported yet.
- Without a parachute the rocket arrives at the ground at roughly 50 m/s.

## Layout

```
rocket_sim/
  config.py     fixed rocket and motor geometry, atmosphere helpers
  motor.py      editable performance, validation, estimated thrust curves
  physics.py    rigid-body simulation, events, readouts
  viewport.py   pyqtgraph OpenGL scene: rocket mesh, ground, pad, trail, camera
  ui.py         window: viewport, motor panel, controls, live data
  app.py        entry point and --self-test
tests/          config, motor, flight and UI tests
packaging/      PyInstaller spec, build scripts, Inno Setup script
```

## Licence

MIT — see `LICENSE`.
