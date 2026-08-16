"""TK3 privileged physical-motion-generator environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg

import src.tasks.ghost.mdp as mdp
from src.assets.robots.tiangong3.tk3_constants_ghost import (
  TK3_ACTION_SCALE,
  TK3_NOMINAL_FOOT_GROUND_FRICTION,
  get_tk3_robot_cfg,
)
from src.tasks.ghost.mdp import (
  MotionCommandCfg,
  ReferenceJointPositionLimitActionCfg,
)
from src.tasks.ghost.tracking_env_cfg import make_tracking_env_cfg


_QREF_PROTOTYPE_VERTICAL_RELAXED_BODIES = (
  "ankle_roll_l_link",
  "ankle_roll_r_link",
  "wrist_roll_l_link",
  "wrist_roll_r_link",
)


def _tk3_base_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """创建 TK3 物理运动生成器的共享基础配置。"""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_tk3_robot_cfg()}
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
    ContactSensorCfg(
      name="ground_contact",
      primary=ContactMatch(mode="body", pattern=r".*", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    ),
    ContactSensorCfg(
      # 有符号距离不进入策略观测，仅用于惩罚明显穿地。
      name="ground_penetration",
      primary=ContactMatch(mode="body", pattern=r".*", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "dist"),
      reduce="mindist",
      num_slots=1,
    ),
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = TK3_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  # Supply the motion explicitly with --motion-file when training or playing.
  motion_cmd.motion_file = ""
  motion_cmd.anchor_body_name = "pelvis"
  motion_cmd.body_names = (
    "pelvis",
    "hip_roll_l_link",
    "knee_pitch_l_link",
    "ankle_roll_l_link",
    "hip_roll_r_link",
    "knee_pitch_r_link",
    "ankle_roll_r_link",
    "waist_pitch_link",
    "shoulder_roll_l_link",
    "elbow_pitch_l_link",
    "wrist_roll_l_link",
    "shoulder_roll_r_link",
    "elbow_pitch_r_link",
    "wrist_roll_r_link",
  )
  # Stage I uses nominal actuator timing without physical randomization.
  motion_cmd.actuator_command_lag_range = None
  # 100 Hz preview: now, 50 ms, and 100 ms. qdot already describes the
  # near-term trend, so a more highly correlated horizon is omitted.
  motion_cmd.preview_frame_offsets = (0, 5, 10)
  motion_cmd.preview_body_names = (
    "ankle_roll_l_link",
    "ankle_roll_r_link",
    "wrist_roll_l_link",
    "wrist_roll_r_link",
  )
  # Per-environment command perturbations are held for one second. For the
  # default 50k * 48-step run, fade them to zero by the halfway point.
  motion_cmd.command_noise_resample_time_s = 1.0
  motion_cmd.command_noise_anneal_start_step = 0
  motion_cmd.command_noise_anneal_end_step = 1_200_000
  # Small RSI on the birth root pose. Joints stay at the motion frame and
  # are clipped to 98% of hard limits in MotionCommand.
  motion_cmd.pose_range = {
    "x": (-0.03, 0.03),
    "y": (-0.03, 0.03),
    "z": (-0.01, 0.01),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
  }

  cfg.scene.terrain.collisions = (
    CollisionCfg(
      geom_names_expr=(r"^terrain$",),
      condim=3,
      priority=1,
      friction=(TK3_NOMINAL_FOOT_GROUND_FRICTION,),
    ),
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "ankle_roll_l_link",
    "ankle_roll_r_link",
    "wrist_roll_l_link",
    "wrist_roll_r_link",
  )

  cfg.viewer.body_name = "pelvis"
  cfg.viewer.distance = 3.5

  # The bundled TK3 reference motion is sampled at 100 Hz. With a 5 ms
  # simulation step, decimation=2 keeps one policy step per motion frame.
  cfg.decimation = 2
  cfg.episode_length_s = 20.0
  cfg.sim.nconmax = 70
  cfg.sim.contact_sensor_maxmatch = 500

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    # Playback uses nominal parameters and runs until the caller exits.
    # Disable randomization events, all terminations, RSI, and command noise.
    cfg.events.clear()
    cfg.terminations.clear()
    motion_cmd.actuator_command_lag_range = None

    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"
    motion_cmd.command_noise_enabled = False

  return cfg


def tk3_qref_residual_prototype_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """创建仅使用 q_ref 残差动作的 Stage-I Ghost 任务。"""
  cfg = _tk3_base_tracking_env_cfg(play=play)

  cfg.actions["joint_pos"] = ReferenceJointPositionLimitActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=TK3_ACTION_SCALE,
    command_name="motion",
    residual_clip=1.0,
    joint_limit_factor=0.98,
  )

  # q_ref 已经通过动作先验提供关节风格；若再奖励 q/qdot 精确跟踪，
  # 会重复强化参考动作并抑制策略对不合理 motion 的物理修正。
  cfg.rewards.pop("motion_joint_pos")
  cfg.rewards.pop("motion_joint_vel")

  # 放松四个接触末端的竖直位置跟踪，使策略能修正穿地或悬空；
  # 水平位置、姿态、速度及残差先验仍负责保持原动作风格。
  cfg.rewards["motion_body_pos"].params["vertical_relaxed_body_names"] = (
    _QREF_PROTOTYPE_VERTICAL_RELAXED_BODIES
  )

  # q_ref 本身的变化不会进入残差动作变化率，因此额外约束实际物理轨迹。
  cfg.rewards["joint_acceleration"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-2.5e-7,
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )
  cfg.rewards["deep_ground_penetration"] = RewardTermCfg(
    func=mdp.deep_ground_penetration_cost,
    weight=-0.1,
    params={
      "sensor_name": "ground_penetration",
      "tolerance": 0.002,
      "scale": 0.005,
    },
  )
  return cfg
