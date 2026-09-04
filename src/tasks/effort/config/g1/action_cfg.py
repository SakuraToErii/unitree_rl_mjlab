"""Absolute effort action constants for the Unitree G1."""

from mjlab.envs.mdp.actions import JointEffortActionCfg

from src.tasks.effort.zero_pd import EFFORT_ACTION_SCALE_FRACTION

# Same-direction peak torque (Y1) from Unitree's motor T-N bench:
# N7520-14.3 → 71, N7520-22.5 → 111, N5020-16 → 24.8, W4010-25 → 4.8.
# These are lookup values, not derived from URDF effort_limit (Y2-side).
EFFORT_ACTION_LIMIT: dict[str, float] = {
  r".*_hip_pitch_joint": 71.0,
  r".*_hip_yaw_joint": 71.0,
  "waist_yaw_joint": 71.0,
  r".*_hip_roll_joint": 111.0,
  r".*_knee_joint": 111.0,
  r".*_shoulder_.*": 24.8,
  r".*_elbow_joint": 24.8,
  r".*_wrist_roll.*": 24.8,
  r".*_ankle_.*": 24.8,
  "waist_roll_joint": 24.8,
  "waist_pitch_joint": 24.8,
  r".*_wrist_pitch.*": 4.8,
  r".*_wrist_yaw.*": 4.8,
}

EFFORT_ACTION_CLIP: dict[str, tuple[float, float]] = {
  joint_expr: (-limit, limit) for joint_expr, limit in EFFORT_ACTION_LIMIT.items()
}

EFFORT_ACTION_SCALE: dict[str, float] = {
  joint_expr: EFFORT_ACTION_SCALE_FRACTION * limit
  for joint_expr, limit in EFFORT_ACTION_LIMIT.items()
}


def g1_effort_action_cfg() -> JointEffortActionCfg:
  """Create ``tau = clip(action * fraction * Y1, ±Y1)`` for all 29 joints."""
  return JointEffortActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=EFFORT_ACTION_SCALE,
    offset=0.0,
    clip=EFFORT_ACTION_CLIP,
  )
