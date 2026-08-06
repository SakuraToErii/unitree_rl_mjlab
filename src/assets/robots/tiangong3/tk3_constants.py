"""TienKung 3 29-DoF robot asset and actuator configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from src import SRC_PATH


# MJCF and assets.
TK3_XML: Path = (
  SRC_PATH / "assets" / "robots" / "tiangong3" / "xmls" / "tiangong3.xml"
)
TK3_MESH_DIR: Path = TK3_XML.parent.parent / "meshes"
TK3_BASE_HEIGHT = 0.95

assert TK3_XML.exists()
assert TK3_MESH_DIR.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  """Load meshes required when the robot spec is attached to a scene."""
  assets: dict[str, bytes] = {}
  update_assets(assets, TK3_MESH_DIR, meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """Return a fresh robot-only MuJoCo spec."""
  spec = mujoco.MjSpec.from_file(str(TK3_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


# Actuator configuration.
#
# Armatures are the effective joint armatures in the supplied xSIM MJCF.
# Effort limits follow its torque-control actuator ctrlrange values. The
# deployment position actuators are intentionally not retained in the training
# MJCF: MJLab creates exactly one position actuator for every policy joint.
NATURAL_FREQ = 5.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
FRICTIONLOSS = 0.1


def _position_actuator(
  target_names_expr: tuple[str, ...],
  *,
  armature: float,
  effort_limit: float,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=armature * NATURAL_FREQ**2,
    damping=2.0 * DAMPING_RATIO * armature * NATURAL_FREQ,
    effort_limit=effort_limit,
    armature=armature,
    frictionloss=FRICTIONLOSS,
  )


TK3_ACTUATOR_HIP_YAW = _position_actuator(
  (r"hip_yaw_[lr]_joint",), armature=0.18, effort_limit=142.0
)
TK3_ACTUATOR_HIP_PITCH_ROLL = _position_actuator(
  (r"hip_(pitch|roll)_[lr]_joint",), armature=0.24, effort_limit=200.0
)
TK3_ACTUATOR_KNEE = _position_actuator(
  (r"knee_pitch_[lr]_joint",), armature=0.37, effort_limit=330.0
)
TK3_ACTUATOR_ANKLE = _position_actuator(
  (r"ankle_(pitch|roll)_[lr]_joint",), armature=0.032, effort_limit=55.0
)
TK3_ACTUATOR_WAIST_YAW = _position_actuator(
  (r"waist_yaw_joint",), armature=0.17, effort_limit=91.0
)
TK3_ACTUATOR_WAIST_PITCH_ROLL = _position_actuator(
  (r"waist_(roll|pitch)_joint",), armature=0.17, effort_limit=150.0
)
TK3_ACTUATOR_SHOULDER_PITCH_ROLL = _position_actuator(
  (r"shoulder_(pitch|roll)_[lr]_joint",), armature=0.1, effort_limit=90.0
)
TK3_ACTUATOR_SHOULDER_YAW_ELBOW_PITCH = _position_actuator(
  (r"(shoulder_yaw|elbow_pitch)_[lr]_joint",),
  armature=0.1,
  effort_limit=50.0,
)
TK3_ACTUATOR_ELBOW_YAW = _position_actuator(
  (r"elbow_yaw_[lr]_joint",), armature=0.1, effort_limit=25.0
)
TK3_ACTUATOR_WRIST = _position_actuator(
  (r"wrist_(pitch|roll)_[lr]_joint",), armature=0.0236, effort_limit=25.0
)

TK3_ACTUATORS = (
  TK3_ACTUATOR_HIP_YAW,
  TK3_ACTUATOR_HIP_PITCH_ROLL,
  TK3_ACTUATOR_KNEE,
  TK3_ACTUATOR_ANKLE,
  TK3_ACTUATOR_WAIST_YAW,
  TK3_ACTUATOR_WAIST_PITCH_ROLL,
  TK3_ACTUATOR_SHOULDER_PITCH_ROLL,
  TK3_ACTUATOR_SHOULDER_YAW_ELBOW_PITCH,
  TK3_ACTUATOR_ELBOW_YAW,
  TK3_ACTUATOR_WRIST,
)


# Initial state.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, TK3_BASE_HEIGHT),
  joint_pos={
    r"hip_pitch_[lr]_joint": -0.13,
    r"hip_(roll|yaw)_[lr]_joint": 0.0,
    r"knee_pitch_[lr]_joint": 0.3,
    r"ankle_pitch_[lr]_joint": -0.17,
    r"ankle_roll_[lr]_joint": 0.0,
    r"waist_(yaw|roll|pitch)_joint": 0.0,
    r"shoulder_pitch_[lr]_joint": 0.2,
    r"elbow_pitch_[lr]_joint": -0.5,
    "shoulder_roll_l_joint": 0.1,
    "shoulder_roll_r_joint": -0.1,
    r"shoulder_yaw_[lr]_joint": 0.0,
    r"elbow_yaw_[lr]_joint": 0.0,
    r"wrist_(pitch|roll)_[lr]_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


# Collision configuration.
FOOT_GEOM_PATTERN = r"^foot_(left|right)_.*$"
COLLISION_GEOM_PATTERN = r".*_collision(?:_[0-9]+)?$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(COLLISION_GEOM_PATTERN, FOOT_GEOM_PATTERN),
  condim={FOOT_GEOM_PATTERN: 3, COLLISION_GEOM_PATTERN: 1},
  priority={FOOT_GEOM_PATTERN: 1},
  friction={FOOT_GEOM_PATTERN: (1.5,)},
)


TK3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=TK3_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_tk3_robot_cfg() -> EntityCfg:
  """Return a fresh TienKung 3 robot configuration."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=TK3_ARTICULATION,
  )


# One normalized policy-action unit requests one quarter of the maximum
# quasi-static joint deflection, matching the other velocity tasks.
TK3_ACTION_SCALE: dict[str, float] = {}
for actuator in TK3_ACTUATORS:
  effort = actuator.effort_limit
  assert effort is not None
  for name_expr in actuator.target_names_expr:
    TK3_ACTION_SCALE[name_expr] = 0.25 * effort / actuator.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_tk3_robot_cfg())
  viewer.launch(robot.spec.compile())
