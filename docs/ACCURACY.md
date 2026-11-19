# Accuracy, validation and limitations

The application is only useful if its numbers are trustworthy. This document lists what
the automated test suite proves, how, and where the physics is deliberately approximate.

Run `python -m pytest` to reproduce every claim below (65 tests at the time of writing).
The suite tests *behaviour and closed-form results*, not saved output snapshots.

## What is validated against independent results

| Area | Test file | What is checked |
|------|-----------|-----------------|
| Quaternion algebra | `tests/test_math3d.py` | Rotation of basis vectors, inverse/composition identity, Euler round-trip, transform inverse |
| Inertia and parallel axis | `tests/test_math3d.py` | Offset inertia `I + m d²`, analytic box inertia, combined COM of two bodies |
| Sketch solver | `tests/test_sketch.py` | Rectangle/circle profiles close, a horizontal constraint moves a free point, sketch round-trip |
| Exact CAD kernel | `tests/test_cad_kernel.py` | Box/cylinder mass properties, boolean cut volume, fillet/chamfer reduce volume, extrude volume, tessellation bounds, STEP/IGES/STL round-trip volume |
| Feature history | `tests/test_features.py` | Hole volume from a plate, tube/cone analytic volumes, 4-fin polar pattern, failed feature is reported and not hidden |
| Fasteners | `tests/test_fasteners.py` | Catalogue contents, engagement/strip trend with depth, torque↔clamp-force round-trip, thread-forming vs pre-threaded mode, install depth clamp |
| Assembly model and masses | `tests/test_assembly.py` | Parent/child world transforms, coincident mate convergence, mass rows sum to total, manual override delta, recursive component removal |
| Mechanical dynamics | `tests/test_mech.py` | Free revolute joint rotates at its initial rate, fixed body ignores gravity, servo hinge tracks a 5° command, SAT contact detects overlap and separation |
| Servo/sensor/computer | `tests/test_electronics.py` | Stationary IMU reads 1 g, range saturation, servo slew limit, flight-computer loop produces frames, HIL loopback framing statistics |
| Atmosphere and propulsion | `tests/test_flight.py` | ISA sea-level values, `.eng` round-trip impulse, tutorial motor provenance, static margin sign, passive aero stability, guided flight apogee/tilt bounds |
| Aerodynamics | `tests/test_aero.py` | Barrowman CP inside the stack, fin area moves the CP, drag equation, restoring moment sign and static-margin definition |
| Project I/O | `tests/test_project.py` | Full round-trip equality of parts/components/fasteners/mass, newer-format rejection, invalid JSON rejection, component library persistence, undo/redo and no-op detection |
| Reports | `tests/test_reports.py` | Mass breakdown, installed-fastener rows, constraint DOF report, clearance pairs, JSON report output |

## Approximations and model boundaries the application declares

1. **Servo-driven joints are kinematically driven.** In the mechanical world the servo's
   rate limit, torque limit, deadband and backlash shape the *commanded* joint motion; the
   required torque is estimated and clamped rather than fed back as a torque source.
   Coupling a torque source into the 1 ms sequential-impulse solver produced unstable
   hinge behaviour, so the honest bench model is a position servo. The capability dialog
   and the mechanics workspace state this. Servo tracking accuracy is therefore bounded by
   the plant, not by a rigid-body torque balance.

   In the 6-DOF flight loop the same plant is stepped each 1 ms with the instantaneous
   thrust load, modelled as `F·e·sin δ` for a declared 1 mm nominal offset `e` between the
   thrust axis and the servo horn (zero load at zero deflection). The driven inertia is a
   declared assumption (`ServoSpec.output_inertia`, 2e-5 kg·m² for the default micro
   servo: horn plus linkage and gimbal or control surface); the horn alone is too light to
   integrate the gear-train stiffness at the flight timestep. Replace it with a measured
   value for the real installation. Stall torque is taken from the datasheet in kg·cm and
   converted once (1 kg·cm = 0.0980665 N·m).

2. **The mechanical world is a bench fixture.** Components that no mate references are
   treated as rigidly mounted and gravity is disabled. This makes a hinge or gimbal
   testable without a full vehicle model; it is not a free-falling assembly simulation.

3. **Thread forming and stripping.** A reduced-order empirical model
   (`fasteners.thread.ThreadEngagementModel`) combining a torque/clamp relationship with a
   material shear-strength strip criterion. It predicts trends and risk, not plastic
   deformation fields. Results are tagged `estimated`. No FEA is performed or claimed.

4. **Aerodynamics.** Coefficient-based: the user supplies or accepts drag/lift/moment
   derivatives. The geometry path computes a Barrowman-style centre of pressure from the
   assembled stack; fin CP uses the trapezoid centroid and sweep, and the fin normal-force
   slope uses the standard Barrowman semispan form. It is labelled approximate and does not
   replace wind-tunnel or CFD data. Full CFD is not implemented.

5. **Static margin convention.** Body `z` is measured from the tail datum. The application
   reports `static_margin = CG_z − CP_z`; positive means the CG is ahead of the CP and the
   vehicle is passively stable. The flight model's restoring moment is checked against this
   convention in `test_aero.py`.

6. **Thrust line and centre of mass.** Thrust acts at the nozzle's geometric centre. If the
   assembly's centre of mass is laterally offset (fins, avionics, wiring), the resulting
   trim torque is physical and the TVC loop must trim it; a passive rocket with a large
   offset will pitch. The passive-stability test isolates aerodynamics with a zero-thrust
   coast so this effect is not mistaken for an aero sign error.

7. **Contact.** Mesh-based, rigid, sequential impulses with restitution and Coulomb
   friction. No compliance, wear or deformation. Components intended to touch must be
   filtered or they will report contact — the UI says so.

8. **Mass properties of printed parts.** Density × CAD volume is exact for the modelled
   solid. Real prints have infill, so a printed part's mass differs unless the user
   overrides mass or models infill. Overrides and provenance are shown in the UI and in
   every report.

9. **Hardware-in-the-loop.** The serial protocol (COBS framing + CRC-16) and its timing
   statistics are real; the loopback test validates framing, not firmware behaviour. The
   application never claims simulated firmware equals hardware execution.

10. **Section views.** The viewport cuts triangles at a plane and does not generate a
    capping surface. It is a viewing aid, not a geometric section.

## What the application refuses to do

* It will not label a flight "successful" on its own. It reports states, margins and
  warnings; the engineer decides.
* It will not substitute a mesh for exact geometry in mass or boolean operations.
* It will not silently create a body from a `cut`/`join` feature when no body exists; the
  rebuild fails with a diagnostic.
* It will not let a visualization animate a component the solver did not move.
* It will not estimate a property and present it as measured.

## Numerical settings

* Flight integration: fixed-step RK4, default 1 ms (configurable). Servo and sensor models
  are sampled inside the flight loop.
* Servo plant: the gear coupling is sub-stepped so the explicit update stays stable at the
  flight timestep (about 8 sub-steps per 1 ms for the default MG90S); the drive torque is
  integrated implicitly. The output-angle curve therefore has no numerical oscillation of
  its own.
* Mechanical dynamics: sequential-impulse solver with a 1 ms default step; joints are
  solved before position integration; servo joints are driven kinematically.
* Constraint solving: SciPy `least_squares` with numerical Jacobians, residual tolerance
  1e-7 by default; rank/DOF diagnostics come from the linearised Jacobian.
* Tessellation: linear deflection derived from the bounding size (normal/fine/coarse).
