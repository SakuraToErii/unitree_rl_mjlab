"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.

This is a re-implementation of BeyondMimic (https://beyondmimic.github.io/).

Based on https://github.com/HybridRobotics/whole_body_tracking
Commit: f8e20c880d9c8ec7172a13d3a88a65e3a5a88448
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

import src.tasks.ghost.mdp as mdp
from src.tasks.ghost.mdp import JointPositionLimitActionCfg, MotionCommandCfg


def make_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base tracking task configuration."""

  ##
  # Observations
  ##

  def privileged_terms() -> dict[str, ObservationTermCfg]:
    return {
      "joint_pos": ObservationTermCfg(func=mdp.exact_joint_pos),
      "joint_vel": ObservationTermCfg(func=mdp.exact_joint_vel),
      "projected_gravity": ObservationTermCfg(
        func=mdp.robot_projected_gravity, params={"command_name": "motion"}
      ),
      "root_height": ObservationTermCfg(
        func=mdp.robot_root_height, params={"command_name": "motion"}
      ),
      "root_velocity_h": ObservationTermCfg(
        func=mdp.robot_root_velocity_h, params={"command_name": "motion"}
      ),
      "current_motion_errors": ObservationTermCfg(
        func=mdp.current_motion_errors, params={"command_name": "motion"}
      ),
      "future_motion_goal": ObservationTermCfg(
        func=mdp.future_motion_goal, params={"command_name": "motion"}
      ),
      "compact_ground_contacts": ObservationTermCfg(
        func=mdp.compact_ground_contacts,
        params={
          "command_name": "motion",
          "sensor_name": "ground_contact",
          "allowed_body_names": (
            "ankle_roll_l_link",
            "ankle_roll_r_link",
            "wrist_pitch_l_link",
            "wrist_pitch_r_link",
          ),
          "force_scale": 100.0,
          "force_threshold": 1.0,
        },
      ),
      "actuator_force": ObservationTermCfg(
        func=mdp.normalized_actuator_force,
        params={"command_name": "motion"},
      ),
      "actions": ObservationTermCfg(func=mdp.last_action),
    }

  observations = {
    "actor": ObservationGroupCfg(
      terms=privileged_terms(),
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms=privileged_terms(),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionLimitActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={},
      velocity_range={},
      joint_position_range=(0.0, 0.0),
      # Override in robot cfg.
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  }

  ##
  # Events（物理域随机化）
  # Stage I / Ghost 原型刻意关闭摩擦、质心、推力等物理 DR；
  # 参考指令噪声与执行器延迟由 MotionCommandCfg 单独控制。
  ##

  events = {}

  ##
  # Rewards
  # 跟踪项均为 exp(-error^2 / std^2) 形高斯核；负权重项为正则/惩罚。
  # 身体位姿使用 root-relative/heading-aligned 目标，避免与根世界轨迹重复。
  ##

  rewards: dict[str, RewardTermCfg] = {
    # 根世界 XY 轨迹是独立强目标。
    "motion_global_root_xy": RewardTermCfg(
      func=mdp.motion_global_anchor_xy_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    # 根高度独立低权重，允许动力学修正参考中的浮空/穿地。
    "motion_global_root_z": RewardTermCfg(
      func=mdp.motion_global_anchor_z_error_exp,
      weight=0.25,
      params={"command_name": "motion", "std": 0.2},
    ),
    # 根姿态跟踪（世界系）
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    # 关键身体位置跟踪（root-relative/heading-aligned，干净参考）
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    # 关键身体姿态跟踪（heading-aligned，干净参考）
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    # 关键身体线速度跟踪
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    # 关键身体角速度跟踪
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    # 14 个 key body 无法唯一约束 29-DoF null-space；低权重 joint 项
    # 补足这一约束，目标仍是干净 reference，并非重复计算 keypoint 误差。
    "motion_joint_pos": RewardTermCfg(
      func=mdp.motion_joint_position_error_exp,
      weight=0.25,
      params={"command_name": "motion", "std": math.sqrt(0.1)},
    ),
    "motion_joint_vel": RewardTermCfg(
      func=mdp.motion_joint_velocity_error_exp,
      weight=0.1,
      params={"command_name": "motion", "std": math.sqrt(5.0)},
    ),
    # 动作变化率惩罚，抑制抖振
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    # 软关节限位惩罚
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # 非足/非手撑地接触惩罚（默认末端名按 TK3；机器人配置可覆盖）
    "undesired_ground_contacts": RewardTermCfg(
      func=mdp.undesired_ground_contact_cost,
      weight=-0.1,
      params={
        "sensor_name": "ground_contact",
        "allowed_body_names": (
          "ankle_roll_l_link",
          "ankle_roll_r_link",
          "wrist_pitch_l_link",
          "wrist_pitch_r_link",
        ),
        "force_threshold": 1.0,
      },
    ),
  }

  ##
  # Terminations（早停条件；Ghost 阈值相对宽松）
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # 根位置偏离参考过大
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos,
      params={"command_name": "motion", "threshold": 0.5},
    ),
    # 根姿态偏离参考过大
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_quat,
      params={
        "command_name": "motion",
        "threshold": 1.2,
      },
    ),
    # 末端身体位置偏离过大（body_names 按机器人覆盖）
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos,
      params={
        "command_name": "motion",
        "threshold": 0.5,
        "body_names": (),  # 按机器人设置。
      },
    ),
    # 状态出现 NaN/Inf
    "nonfinite_state": TerminationTermCfg(
      func=mdp.nonfinite_robot_state,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    # 硬关节限位违规
    "hard_joint_limit": TerminationTermCfg(
      func=mdp.hard_joint_limit_violation,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "tolerance": 1.0e-4,
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
