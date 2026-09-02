"""Unitree G1 configuration for feed-forward torque control."""

from dataclasses import replace

from mjlab.entity import EntityCfg

from src.assets.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from src.tasks.effort.zero_pd import with_zero_pd

EFFORT_STANDING_JOINT_POSITION: dict[str, float] = {
  "left_hip_pitch_joint": -0.059750,
  "right_hip_pitch_joint": -0.028836,
  "waist_yaw_joint": -0.002169,
  "left_hip_roll_joint": 0.001778,
  "right_hip_roll_joint": -0.003393,
  "waist_roll_joint": -0.001681,
  "left_hip_yaw_joint": 0.011778,
  "right_hip_yaw_joint": -0.003191,
  "waist_pitch_joint": -0.000188,
  "left_knee_joint": 0.175060,
  "right_knee_joint": 0.122524,
  "left_shoulder_pitch_joint": 0.312136,
  "right_shoulder_pitch_joint": 0.324009,
  "left_ankle_pitch_joint": -0.120162,
  "right_ankle_pitch_joint": -0.098493,
  "left_shoulder_roll_joint": 0.257946,
  "right_shoulder_roll_joint": -0.246549,
  "left_ankle_roll_joint": 0.005809,
  "right_ankle_roll_joint": 0.007654,
  "left_shoulder_yaw_joint": 0.003801,
  "right_shoulder_yaw_joint": -0.019234,
  "left_elbow_joint": 0.929968,
  "right_elbow_joint": 0.969214,
  "left_wrist_roll_joint": 0.152960,
  "right_wrist_roll_joint": -0.152853,
  "left_wrist_pitch_joint": -0.025037,
  "right_wrist_pitch_joint": -0.013271,
  "left_wrist_yaw_joint": 0.013372,
  "right_wrist_yaw_joint": 0.007627,
}

EFFORT_STANDING_ROOT_HEIGHT = 0.789733


def get_g1_effort_robot_cfg() -> EntityCfg:
  """Return a G1 config with zero-gain motors and the standing residual pose."""
  cfg = with_zero_pd(get_g1_robot_cfg())
  return replace(
    cfg,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, EFFORT_STANDING_ROOT_HEIGHT),
      joint_pos=EFFORT_STANDING_JOINT_POSITION.copy(),
      joint_vel={".*": 0.0},
    ),
  )
