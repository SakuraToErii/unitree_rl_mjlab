"""Unitree G1 absolute-effort velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import src.tasks.effort.mdp as mdp
from src.tasks.effort.config.g1.action_cfg import g1_effort_action_cfg
from src.tasks.effort.config.g1.robot_cfg import (
  EFFORT_STANDING_ROOT_HEIGHT,
  get_g1_effort_robot_cfg,
)
from src.tasks.effort.velocity_env_cfg import (
  PostureGroups,
  disable_command_curriculum,
  enable_current_frame_obs,
  enable_mha_history,
  enable_partial_actor_obs,
  enable_speed_posture,
  make_effort_env_cfg,
)

# Every controlled joint except the four ankles, for ankle-blind actors.
_G1_NO_ANKLE_JOINT_NAMES = (
  ".*_hip_.*_joint",
  ".*_knee_joint",
  "waist_.*_joint",
  ".*_shoulder_.*_joint",
  ".*_elbow_joint",
  ".*_wrist_.*_joint",
)

# std in rad as (standing, walking, running). Sagittal legs open up the most
# with speed; frontal legs and waist stay tight; arms loosen for swing.
G1_SPEED_POSTURE_GROUPS: PostureGroups = {
  "legs_sagittal": (
    1.5,
    {
      r".*_hip_pitch_joint": (0.10, 0.50, 0.80),
      r".*_knee_joint": (0.10, 0.50, 0.80),
      r".*_ankle_pitch_joint": (0.10, 0.25, 0.40),
    },
  ),
  "legs_frontal": (
    1.5,
    {
      r".*_hip_roll_joint": (0.05, 0.15, 0.25),
      r".*_hip_yaw_joint": (0.05, 0.15, 0.25),
      r".*_ankle_roll_joint": (0.05, 0.10, 0.15),
    },
  ),
  "waist": (
    1.0,
    {
      r"waist_yaw_joint": (0.05, 0.15, 0.20),
      r"waist_(roll|pitch)_joint": (0.05, 0.08, 0.10),
    },
  ),
  "arms": (
    0.5,
    {
      r".*_shoulder_pitch_joint": (0.10, 0.30, 0.60),
      r".*_shoulder_(roll|yaw)_joint": (0.05, 0.10, 0.15),
      r".*_elbow_joint": (0.10, 0.30, 0.50),
      r".*_wrist_.*_joint": (0.10, 0.20, 0.30),
    },
  ),
}


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain absolute-effort configuration."""
  cfg = make_effort_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_g1_effort_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=site_name, entity="robot")
        for site_name in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  cfg.actions = {"joint_effort": g1_effort_action_cfg()}

  # import from isaac lab
  cfg.rewards["joint_deviation_arms"].params["asset_cfg"].joint_names = (
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*",
  )
  cfg.rewards["joint_deviation_waists"].params["asset_cfg"].joint_names = (
    "waist.*",
  )
  cfg.rewards["joint_deviation_legs"].params["asset_cfg"].joint_names = (
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
  )
  cfg.rewards["base_height"].params["target_height"] = EFFORT_STANDING_ROOT_HEIGHT

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain residual-effort configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-2.0, 2.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_g1_rough_mha_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the rough-terrain effort task with five-frame MHA history."""
  return enable_mha_history(unitree_g1_rough_env_cfg(play=play))


def unitree_g1_flat_mha_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat-terrain effort task with five-frame MHA history."""
  return enable_mha_history(unitree_g1_flat_env_cfg(play=play))


def unitree_g1_flat_partial_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat-terrain effort task with a position-centric actor."""
  return enable_partial_actor_obs(unitree_g1_flat_env_cfg(play=play))


def unitree_g1_flat_partial_phase1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Partial actor obs with both ankle encoders removed from joint_pos."""
  cfg = unitree_g1_flat_partial_env_cfg(play=play)
  cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_G1_NO_ANKLE_JOINT_NAMES
  )
  return cfg


def unitree_g1_flat_speed_posture_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat full-obs task with speed-blended posture groups in place of L1."""
  return enable_speed_posture(
    unitree_g1_flat_env_cfg(play=play), G1_SPEED_POSTURE_GROUPS
  )


def unitree_g1_flat_partial_phase1_speed_posture_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Flat partial phase-1 task with speed-blended posture groups."""
  return enable_speed_posture(
    unitree_g1_flat_partial_phase1_env_cfg(play=play), G1_SPEED_POSTURE_GROUPS
  )


def unitree_g1_flat_imu_no_ankle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Ankle-blind actor that keeps IMU and drops joint velocity and gait phase."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  actor_obs = cfg.observations["actor"]
  for term_name in ("joint_vel", "phase"):
    actor_obs.terms.pop(term_name, None)
  actor_obs.terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_G1_NO_ANKLE_JOINT_NAMES
  )
  return cfg


def unitree_g1_flat_imu_no_ankle_no_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """IMU ankle-blind MLP task with velocity commands at full range from step 0."""
  return disable_command_curriculum(unitree_g1_flat_imu_no_ankle_env_cfg(play=play))


def unitree_g1_flat_imu_no_ankle_h1_no_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Current-frame variant of the IMU ankle-blind no-curriculum task."""
  return enable_current_frame_obs(
    unitree_g1_flat_imu_no_ankle_no_curriculum_env_cfg(play=play)
  )


def unitree_g1_flat_imu_no_ankle_mha_no_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """TimeMHA variant of the IMU ankle-blind no-curriculum task."""
  return enable_mha_history(
    unitree_g1_flat_imu_no_ankle_no_curriculum_env_cfg(play=play)
  )


def unitree_g1_flat_partial_mha_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat partial-obs effort task with five-frame MHA history."""
  return enable_mha_history(unitree_g1_flat_partial_env_cfg(play=play))


def unitree_g1_flat_partial_phase1_mha_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the flat partial phase-1 effort task with five-frame MHA history."""
  return enable_mha_history(unitree_g1_flat_partial_phase1_env_cfg(play=play))
