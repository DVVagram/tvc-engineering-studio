# Flight model

One rigid body. Fixed geometry, fixed motor hardware, fixed launch tilt. The only variation
between runs is the motor's performance specification.

## State and integration

State vector: position `r` (world, z up), velocity `v`, attitude quaternion `q` (body → world,
body +X along the nose) and angular velocity `ω` in world coordinates. Classical RK4 with a
fixed 2 ms step; rendering only reads this state.

Mass changes during the burn because the propellant grain depletes linearly:

```
m(t) = structure + casing + propellant · (1 − t/t_burn)      for t < t_burn
m(t) = structure + casing                                     afterwards
```

## Forces

Thrust acts along the body axis and comes from the sampled curve:

```
F_thrust = T(t) · R(q)·[1,0,0]
```

Gravity: `F_g = m·(0, 0, −g)`, with `g = 9.80665 m/s²`.

Atmosphere: `ρ(h) = 1.225·exp(−h/8500)`.

Drag, opposite the air-relative velocity:

```
D = ½·ρ·v²·C_d(M)·A          C_d = 0.45 below M 0.8, rising to 1.8·C_d near M 1.05
```

Fin normal force, applied at the centre of pressure, opposing the lateral airflow:

```
N = ½·ρ·v²·A·C_Nα·sin α,      C_Nα = 10 / rad
τ_aero = (−d_cp · body_axis) × N,   d_cp = 0.20 m
```

Pitch damping, transverse component only:

```
τ_damp = −6 · ½·ρ·v²·A·d_cp² · ω_perp
```

The centre of pressure being behind the centre of gravity makes `τ_aero` restore the nose
towards the velocity vector; `ω` and `q` then follow from `ω̇ = τ/I` and
`q̇ = ½·[0, ω]·q`. The pitch inertia is computed once from the fixed geometry (body tube,
nose, fins and motor at their offsets), giving approximately `0.15 kg·m²`.

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
| Apogee | 187 m |
| Maximum speed | 61 m/s |
| Burnout | 1.80 s |
| Impact | 13.8 s, 50 m/s |
| SR-30 preset apogee | 62 m |
| SR-90 preset apogee | 339 m |
