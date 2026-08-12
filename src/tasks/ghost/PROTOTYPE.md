# PROTOTYPE — TK3 physics-reference generator

This throwaway task answers one question:

> Can a privileged PPO policy turn one noisy 100 Hz TK3 motion into a
> torque-limited MuJoCo rollout with no hard joint-limit violation, deep ground
> penetration, or unsupported static floating while preserving its global
> trajectory?

The prototype deliberately handles one motion at a time.  It is a reference
generator, not a deployable policy and not a teacher-action distillation
pipeline.

## Paper facts used

OmniTrack Stage I uses full simulator-only state, no observation noise or
physical domain randomization, bounded noise on the reference command, relaxed
early termination, failure-driven segment sampling, real hardware torque
limits, and simulator rollout states as the Stage II reference
([paper §III-A](https://arxiv.org/html/2602.23832v1#S3.SS1),
[reward and randomization tables](https://arxiv.org/html/2602.23832v1#A0.T5)).

The paper does not publish its actor/critic split, network, PD gains, action
offset/scale, termination thresholds, adaptive-sampling update, or archive
format.  The choices below are therefore TK3 prototype decisions, not an exact
reproduction.

## Prototype decisions

- MuJoCo physics runs at 200 Hz and the policy at 100 Hz to match the local
  motion contract.
- Existing nominal TK3 Kp/Kd, armatures, friction loss, joint limits, and
  actuator effort limits are retained.  Ghost replaces the cylinder rails with
  separate left/right 10 mm sole-only convex hulls whose outlines are extracted
  from the lower visual foot meshes, with 2 mm edge chamfers and a firm,
  critically damped rubber contact (`solref=(0.015, 1.0)`, 3 mm impedance
  transition).
- Actor and critic receive the same privileged, error-centric observation.  It
  combines exact physical state/current residuals with reference previews at
  frame offsets `(0, 5, 10, 20)` (0–200 ms at 100 Hz).
- Each preview horizon contains joint pose/velocity, global-root correction,
  root twist, and four end-effector position residuals.  Current residuals
  retain all 14 key-body orientation/velocity errors.  Future indexing clips
  at the final frame and never wraps.
- Reference command noise uses the ranges published for OmniTrack PMG, is held
  per environment for 1.0 s, is shared across preview horizons, and linearly
  fades from scale 1 to 0 over common steps 960,000–1,200,000.  Rewards always
  use the clean reference; play and rollout disable command noise.
- Global root XY, root Z, and root orientation have separate rewards.  Body
  pose rewards are root-relative/heading-aligned, while low-weight joint
  rewards constrain the 29-DoF null-space left by the 14 tracked bodies.
- Contact observations contain only four allowed-end masks/log forces and
  aggregate non-end contacts.  Collision clearance/signed distance is not an
  observation or reward; the existing penetration field remains solely an
  output quality diagnostic required by the rollout schema.
- Early termination uses intentionally relaxed TK3-specific thresholds.
- A deterministic, unrandomized rollout writes the actual simulator state at
  policy boundaries.  It never joins a command wrap/reset to the output.

## Run

Train:

```bash
python scripts/train.py TK3-Ghost-Tracking \
  --motion-file datasets/<source>.npz \
  --env.scene.num-envs 8192
```

Generate and validate a physics reference:

```bash
python scripts/rollout_ghost.py TK3-Ghost-Tracking \
  --checkpoint-file logs/rsl_rl/tk3_ghost/<run>/model_<iteration>.pt \
  --motion-file datasets/<source>.npz \
  --output-file datasets/<source>_physical.npz
```

The rollout command also writes `<source>_physical.metrics.json`.  The output
NPZ remains compatible with the production tracking `MotionLoader`; diagnostic
fields are additive.

## Stage II handoff

The deployable actor should use only motion goals plus onboard proprioception
(`q`, `qdot`, root orientation, root angular velocity, previous action).  Its
critic may remain privileged.  Stage II should consume the generated contact
mask for desired-contact supervision and re-enable TK3-calibrated observation
noise, delay, friction/restitution, COM/joint-property randomization, and
external pushes.
