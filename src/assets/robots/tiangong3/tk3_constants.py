"""TienKung 3 29-DoF robot asset and actuator configuration."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH

# MJCF and assets.
TK3_XML: Path = (
  SRC_PATH / "assets" / "robots" / "tiangong3" / "xmls" / "tiangong3.xml"
)
TK3_MESH_DIR: Path = TK3_XML.parent.parent / "meshes"
# Pelvis height for HOME joint angles so foot collision cylinders clear z=0.
# With z=0.95 the soles sit ~4.9 cm below the plane; 0.999 keeps the same pose.
TK3_BASE_HEIGHT = 0.998

assert TK3_XML.exists()
assert TK3_MESH_DIR.exists()


def get_spec() -> mujoco.MjSpec:
  """Return a fresh robot-only MuJoCo spec."""
  return mujoco.MjSpec.from_file(str(TK3_XML))


# Actuator configuration.
#
# Armatures are the effective joint armatures in the supplied xSIM MJCF.
# Effort limits follow its torque-control actuator ctrlrange values. The
# deployment position actuators are intentionally not retained in the training
# MJCF: MJLab creates exactly one position actuator for every policy joint.
NATURAL_FREQ = 8.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
FRICTIONLOSS = 0.1
TK3_COMMAND_DELAY_MIN_LAG = 0
TK3_COMMAND_DELAY_MAX_LAG = 4
# True: use the explicit Kp/Kd values below. False: use the original formula
# for every actuator group.
TK3_USE_EXPLICIT_PD_GAINS = True


def _position_actuator(
  target_names_expr: tuple[str, ...],
  *,
  armature: float,
  effort_limit: float,
  stiffness_override: float | None = None,
  damping_override: float | None = None,
) -> BuiltinPositionActuatorCfg:
  # Preserve the original armature-based PD calculation. TK3 actuator groups
  # may explicitly override either value without changing this default path.
  calculated_stiffness = armature * NATURAL_FREQ**2
  calculated_damping = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
  resolved_stiffness = calculated_stiffness
  resolved_damping = calculated_damping
  if TK3_USE_EXPLICIT_PD_GAINS and stiffness_override is not None:
    resolved_stiffness = stiffness_override
  if TK3_USE_EXPLICIT_PD_GAINS and damping_override is not None:
    resolved_damping = damping_override
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=resolved_stiffness,
    damping=resolved_damping,
    effort_limit=effort_limit,
    armature=armature,
    frictionloss=FRICTIONLOSS,
    delay_min_lag=TK3_COMMAND_DELAY_MIN_LAG,
    delay_max_lag=TK3_COMMAND_DELAY_MAX_LAG,
    # MotionCommand samples after reset and holds until the next motion resample.
    delay_hold_prob=1.0,
  )

ARMATURE_HIP_PITCH_ROLL = 0.24
ARMATURE_HIP_YAW = 0.18
ARMATURE_KNEE = 0.37
ARMATURE_ANKLE = 0.032
ARMATURE_WAIST = 0.17
ARMATURE_ARM = 0.1
ARMATURE_WRIST = 0.0236

TK3_ACTUATOR_HIP_PITCH_ROLL = _position_actuator(
  (r"hip_(pitch|roll)_[lr]_joint",),
  armature=ARMATURE_HIP_PITCH_ROLL,
  effort_limit=223.0,
  stiffness_override=900.0,
  damping_override=57.0,
)
TK3_ACTUATOR_HIP_YAW = _position_actuator(
  (r"hip_yaw_[lr]_joint",),
  armature=ARMATURE_HIP_YAW,
  effort_limit=142.0,
  stiffness_override=660.0,
  damping_override=42.0,
)
TK3_ACTUATOR_KNEE = _position_actuator(
  (r"knee_pitch_[lr]_joint",),
  armature=ARMATURE_KNEE,
  effort_limit=380.0,
  stiffness_override=1260.0,
  damping_override=80.0,
)
TK3_ACTUATOR_ANKLE = _position_actuator(
  (r"ankle_(pitch|roll)_[lr]_joint",),
  armature=ARMATURE_ANKLE,
  effort_limit=52.0,
  stiffness_override=55.0,
  damping_override=3.4,
)
TK3_ACTUATOR_WAIST_PITCH_ROLL = _position_actuator(
  (r"waist_(roll|pitch)_joint",),
  armature=ARMATURE_WAIST,
  effort_limit=142.0,
  stiffness_override=670.0,
  damping_override=41.0,
)
TK3_ACTUATOR_WAIST_YAW = _position_actuator(
  (r"waist_yaw_joint",),
  armature=ARMATURE_WAIST,
  effort_limit=86.0,
  stiffness_override=670.0,
  damping_override=41.0,
)
TK3_ACTUATOR_SHOULDER_PITCH_ROLL = _position_actuator(
  (r"shoulder_(pitch|roll)_[lr]_joint",),
  armature=ARMATURE_ARM,
  effort_limit=85.0,
  stiffness_override=100.0,
  damping_override=7.4,
)
TK3_ACTUATOR_SHOULDER_YAW_ELBOW_PITCH = _position_actuator(
  (r"(shoulder_yaw|elbow_pitch)_[lr]_joint",),
  armature=ARMATURE_ARM,
  effort_limit=47.0,
  stiffness_override=90.0,
  damping_override=5.9,
)
TK3_ACTUATOR_ELBOW_YAW = _position_actuator(
  (r"elbow_yaw_[lr]_joint",),
  armature=ARMATURE_ARM,
  effort_limit=24.0,
  stiffness_override=35.0,
  damping_override=2.4,
)
TK3_ACTUATOR_WRIST = _position_actuator(
  (r"wrist_(pitch|roll)_[lr]_joint",),
  armature=ARMATURE_WRIST,
  effort_limit=38.0,
  stiffness_override=35.0,
  damping_override=2.4,
)

TK3_ACTUATORS = (
  TK3_ACTUATOR_HIP_PITCH_ROLL,
  TK3_ACTUATOR_HIP_YAW,
  TK3_ACTUATOR_KNEE,
  TK3_ACTUATOR_ANKLE,
  TK3_ACTUATOR_WAIST_PITCH_ROLL,
  TK3_ACTUATOR_WAIST_YAW,
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
TK3_NOMINAL_FOOT_GROUND_FRICTION = 1.0

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(COLLISION_GEOM_PATTERN, FOOT_GEOM_PATTERN),
  condim={FOOT_GEOM_PATTERN: 3, COLLISION_GEOM_PATTERN: 1},
  priority={FOOT_GEOM_PATTERN: 2},
  # Foot priority makes this the effective foot-ground sliding coefficient.
  friction={FOOT_GEOM_PATTERN: (TK3_NOMINAL_FOOT_GROUND_FRICTION,)},
)


TK3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=TK3_ACTUATORS,
  soft_joint_pos_limit_factor=0.95,
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
    TK3_ACTION_SCALE[name_expr] = 0.25 #* effort / actuator.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_tk3_robot_cfg())
  viewer.launch(robot.spec.compile())
