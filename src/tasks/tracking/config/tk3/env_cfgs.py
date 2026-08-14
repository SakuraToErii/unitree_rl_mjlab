"""TienKung 3 flat-terrain motion-tracking environment configuration."""

import math

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from src.assets.robots import TK3_ACTION_SCALE, get_tk3_robot_cfg
from src.assets.robots.tiangong3.tk3_constants import (
  FOOT_GEOM_PATTERN,
  TK3_COMMAND_DELAY_MAX_LAG,
  TK3_COMMAND_DELAY_MIN_LAG,
  TK3_NOMINAL_FOOT_GROUND_FRICTION,
)
from src.tasks.tracking import mdp
from src.tasks.tracking.mdp import MotionCommandCfg
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def _sample_uniform_mass_scale_as_alpha(
  lower: torch.Tensor,
  upper: torch.Tensor,
  shape: tuple[int, ...],
  device: str,
) -> torch.Tensor:
  """Sample pseudo-inertia alpha values that produce uniform mass scales."""
  # pseudo_inertia uses mass_scale = exp(2 * alpha).
  scale_lower = torch.exp(2.0 * lower)
  scale_upper = torch.exp(2.0 * upper)
  scale = (
    torch.rand(shape, device=device, dtype=lower.dtype)
    * (scale_upper - scale_lower)
    + scale_lower
  )
  return 0.5 * torch.log(scale)


_UNIFORM_MASS_SCALE_DISTRIBUTION = dr.Distribution(
  name="uniform_mass_scale",
  sample=_sample_uniform_mass_scale_as_alpha,
)


def tk3_flat_tracking_env_cfg(
  has_state_estimation: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the TienKung 3 BeyondMimic tracking configuration.

  Defaults to no state-estimation obs in the actor
  (``motion_anchor_pos_b``, ``base_lin_vel``) for sim-to-real deployability.
  """
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
  motion_cmd.actuator_command_lag_range = (
    TK3_COMMAND_DELAY_MIN_LAG,
    TK3_COMMAND_DELAY_MAX_LAG,
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = FOOT_GEOM_PATTERN

  cfg.scene.terrain.collisions = (
    CollisionCfg(
      geom_names_expr=(r"^terrain$",),
      condim=3,
      priority=1,
      friction=(TK3_NOMINAL_FOOT_GROUND_FRICTION,),
    ),
  )

  cfg.events["ground_friction"] = EventTermCfg(
    func=dr.geom_friction,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("terrain", geom_names=r"^terrain$"),
      "operation": "abs",
      "ranges": (0.25, 0.8),
      "shared_random": True,
    },
  )
  base_com_event = cfg.events.pop("base_com")
  base_com_event.params["asset_cfg"].body_names = ("pelvis",)
  cfg.events["randomize_rigid_body_mass_others"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "alpha_range": (
        0.5 * math.log(0.7),
        0.5 * math.log(1.3),
      ),
      "distribution": _UNIFORM_MASS_SCALE_DISTRIBUTION,
    },
  )
  # pseudo_inertia writes body_ipos from compile defaults. Reinsert base_com
  # afterward so its pelvis offset composes with the randomized mass/inertia.
  cfg.events["base_com"] = base_com_event
  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "scale",
      "ranges": (0.7, 1.3),
    },
  )
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
      "distribution": "uniform",
      "operation": "scale",
    },
  )
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      # Nominal frictionloss is 0.1, giving an effective 0.01-0.6 range.
      "ranges": (0.1, 6),
      "operation": "scale",
    },
  )
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "ankle_roll_l_link",
    "ankle_roll_r_link",
    "wrist_roll_l_link",
    "wrist_roll_r_link",
  )

  # TK3 使用比通用 tracking 任务更严格的全身跟踪核。
  cfg.rewards["motion_global_root_pos"].weight = 1.0
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.2
  cfg.rewards["motion_global_root_ori"].weight = 1.0
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.4
  cfg.rewards["motion_body_pos"].params["std"] = 0.2
  cfg.rewards["motion_body_lin_vel"].weight = 0.5
  cfg.rewards["motion_body_lin_vel"].params["std"] = 0.5
  cfg.rewards["motion_body_ang_vel"].weight = 0.5
  cfg.rewards["motion_body_ang_vel"].params["std"] = 3.14
  cfg.rewards["motion_root_lin_vel"] = RewardTermCfg(
    func=mdp.motion_global_body_linear_velocity_error_exp,
    weight=0.5,
    params={
      "command_name": "motion",
      "body_names": ("pelvis",),
      "std": math.sqrt(0.5),
    },
  )
  cfg.rewards["motion_root_ang_vel"] = RewardTermCfg(
    func=mdp.motion_global_body_angular_velocity_error_exp,
    weight=0.5,
    params={
      "command_name": "motion",
      "body_names": ("pelvis",),
      "std": 3.14,
    },
  )
  cfg.rewards["motion_joint_pos"] = RewardTermCfg(
    func=mdp.motion_joint_position_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": math.sqrt(0.1)},
  )
  cfg.rewards["motion_joint_vel"] = RewardTermCfg(
    func=mdp.motion_joint_velocity_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": math.sqrt(5.0)},
  )
  cfg.rewards["raw_action_torque_limit"] = RewardTermCfg(
    func=mdp.raw_action_torque_limit_penalty,
    weight=-1.0,
    params={
      "action_name": "joint_pos",
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      # soft_ratio 始终作用于运行时 actuator_forcerange。当前训练资产先将
      # effort_limit 乘以 0.85，因此训练中的触发点为缩放前上限的 80.75%。
      "soft_ratio": 0.95,
    },
  )
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-2.5e-7,
  )

  cfg.viewer.body_name = "pelvis"
  cfg.viewer.distance = 3.5

  # The bundled TK3 reference motion is sampled at 100 Hz. With a 5 ms
  # simulation step, decimation=2 keeps one policy step per motion frame.
  cfg.decimation = 2
  cfg.episode_length_s = 20.0
  # Leave headroom for simultaneous foot, terrain, and self contacts.
  cfg.sim.nconmax = 128
  # MujocoCfg default is 50; match Tiangong3-mjlab for denser CCD.
  cfg.sim.mujoco.ccd_iterations = 100

  if not has_state_estimation:
    actor_terms = {
      name: term
      for name, term in cfg.observations["actor"].terms.items()
      if name not in ("motion_anchor_pos_b", "base_lin_vel")
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    # Playback uses nominal parameters and runs until the caller exits.
    # Disable randomization events, all terminations, and actuator delay.
    cfg.events.clear()
    cfg.terminations.clear()
    motion_cmd.actuator_command_lag_range = None

    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"

  return cfg
