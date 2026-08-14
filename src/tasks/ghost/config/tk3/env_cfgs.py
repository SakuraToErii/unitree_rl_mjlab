"""TK3 privileged physical-motion-generator environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from src.assets.robots.tiangong3.tk3_constants_ghost import (
  TK3_ACTION_SCALE,
  TK3_NOMINAL_FOOT_GROUND_FRICTION,
  get_tk3_robot_cfg,
)
from src.tasks.ghost.mdp import MotionCommandCfg
from src.tasks.ghost.tracking_env_cfg import make_tracking_env_cfg


def tk3_flat_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the privileged TK3 physical-motion-generator task."""
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
      # Rollout-schema/quality diagnostic only: this signed distance is never
      # exposed to the policy and is not used by any reward.
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
  # default 50k * 24-step run, fade them to zero over the final 20%.
  motion_cmd.command_noise_resample_time_s = 1.0
  motion_cmd.command_noise_anneal_start_step = 960_000
  motion_cmd.command_noise_anneal_end_step = 1_200_000

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
  cfg.sim.nconmax = 128
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


def tk3_assault_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the clip-assault config with relaxed training boundaries."""
  cfg = tk3_flat_tracking_env_cfg(play=play)
  if not play:
    cfg.episode_length_s = 7.0
    del cfg.terminations["hard_joint_limit"]
  return cfg
