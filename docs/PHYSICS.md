# Flight model

One rigid body. Fixed geometry, fixed motor hardware, fixed launch tilt. The only variation
between runs is the motor's performance specification.

## State and integration

State vector: position `r` (world, z up), velocity `v`, attitude quaternion `q` (body → world,
body +X along the nose) and angular velocity `ω` in world coordinates. Classical RK4 with a
fixed 2 ms step; rendering only reads this state.

Mass changes during the burn in proportion to the impulse actually delivered, so a
progressive curve consumes the grain more slowly at ignition than a regressive one with
the same total impulse:

```
m(t) = structure + casing + propellant · (1 − I_delivered(t)/I_total)
m(t) = structure + casing                                     after burnout
```

## Forces

Thrust acts along the body axis and comes from the sampled curve:

```
F_thrust = T(t) · R(q)·[1,0,0]
```

Gravity: `F_g = m·(0, 0, −g)`, with `g = 9.80665 m/s²`.

Atmosphere, referenced to the launch site: `ρ(h) = 1.225·exp(−h/8500)`.

Drag, opposite the air-relative velocity:

```
D = ½·ρ·v²·C_d(M)·A          C_d = 0.45 below M 0.8, rising to 1.8·C_d near M 1.05
```

Fin normal force, applied at the centre of pressure, opposing the lateral airflow:

```
C_N(α) = 10·sin α·cos α + 1.2·sin²α        (slope per radian, then crossflow)
N = ½·ρ·v²·A·C_N(α)
τ_aero = (−d_cp · body_axis) × N,   d_cp = 0.20 m
```

The first term is the slender-body slope, valid at the few degrees of angle of attack seen
in ascent; the second is flat-plate crossflow, which keeps the force finite through the
large angles reached during the apogee pitch-over instead of over-predicting it.

Pitch damping, transverse component only:

```
τ_damp = −6 · ½·ρ·v²·A·d_cp² · ω_perp
```

The centre of pressure being behind the centre of gravity makes `τ_aero` restore the nose
towards the velocity vector; `ω` and `q` then follow from `ω̇ = τ/I(t)` and
`q̇ = ½·[0, ω]·q`. The pitch inertia comes from the fixed geometry (body tube, nose, fins
and motor at their offsets) and is recomputed as the motor empties, from about
`0.15 kg·m²` at liftoff to `0.13 kg·m²` at burnout.

## Ground and launch

The rocket is held on the pad until the thrust component along the rail exceeds its weight:

```
T(t)·cos(4°) > m·g
```

If that never happens the run ends with a **No liftoff** event. After release the rocket is
free; a run ends when the altitude returns to zero with a downward velocity (**Ground
impact**).

## Thrust curves

The user supplies total impulse `I`, burn time `t_b`, peak thrust `F_p`, mean thrust `F_a` and
a shape. Consistency requires `|F_a·t_b − I| ≤ 0.01·I` and `F_a ≤ F_p ≤ 2·F_a` (the upper
bound exists because a linear ramp with mean `F_a` and peak `F_p` starts at
`2·F_a − F_p`, which must stay positive).

- neutral: `T(t) = F_p`
- progressive: `T(u) = (2F_a − F_p) + (F_p − (2F_a − F_p))·u`
- regressive: `T(u) = F_p + ((2F_a − F_p) − F_p)·u`, with `u = t/t_b`

The sampled curve is normalised so its integral equals the stated total impulse. Curves are
tagged `measured=False`; the interface calls them estimates.

## Known simplifications

- Isotropic pitch inertia (the rocket is axisymmetric, so cross-coupling is negligible here).
- Constant drag coefficient with a hand-shaped transonic rise; no Reynolds or fin-fin
  interference effects.
- No wind, no turbulence, no launch-rail dynamics beyond the hold/release condition.
- No recovery system: the descent is a ballistic fall at roughly 50 m/s.

## Reference results (standard motor, `--self-test`)

| Quantity | Value |
|---|---|
| Apogee | 186.4 m |
| Maximum speed | 60.7 m/s |
| Burnout | 1.80 s |
| Impact | 13.73 s, 51 m/s |
| SR-30 preset apogee | 61.4 m |
| SR-90 preset apogee | 337.8 m |
