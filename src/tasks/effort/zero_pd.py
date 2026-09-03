"""Convert position actuators to zero-gain motors for feed-forward torque."""

from collections.abc import Sequence
from dataclasses import replace

from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg


def zero_pd_actuators(
  actuators: tuple[BuiltinPositionActuatorCfg, ...],
) -> tuple[IdealPdActuatorCfg, ...]:
  """Return the same actuator groups with ``kp = kd = 0``."""
  converted: list[IdealPdActuatorCfg] = []
  for actuator in actuators:
    if actuator.effort_limit is None:
      raise ValueError(
        f"Actuator {actuator.target_names_expr} needs an effort_limit for 0pd control."
      )
    converted.append(
      IdealPdActuatorCfg(
        target_names_expr=actuator.target_names_expr,
        stiffness=0.0,
        damping=0.0,
        effort_limit=actuator.effort_limit,
        armature=actuator.armature,
        frictionloss=actuator.frictionloss,
      )
    )
  return tuple(converted)


def with_zero_pd(cfg: EntityCfg) -> EntityCfg:
  """Copy a robot config and replace position actuators with zero-gain motors."""
  assert cfg.articulation is not None
  position_actuators = tuple(
    actuator
    for actuator in cfg.articulation.actuators
    if isinstance(actuator, BuiltinPositionActuatorCfg)
  )
  if len(position_actuators) != len(cfg.articulation.actuators):
    raise TypeError("0pd conversion expects BuiltinPositionActuatorCfg groups.")
  return replace(
    cfg,
    articulation=EntityArticulationInfoCfg(
      actuators=zero_pd_actuators(position_actuators),
      soft_joint_pos_limit_factor=cfg.articulation.soft_joint_pos_limit_factor,
    ),
  )


def absolute_effort_action_cfg(
  actuators: Sequence[BuiltinPositionActuatorCfg],
) -> JointEffortActionCfg:
  """Build ``tau = clip(action * effort_limit, ±effort_limit)``."""
  scale: dict[str, float] = {}
  clip: dict[str, tuple[float, float]] = {}
  for actuator in actuators:
    limit = actuator.effort_limit
    assert limit is not None
    for expr in actuator.target_names_expr:
      scale[expr] = float(limit)
      clip[expr] = (-float(limit), float(limit))
  return JointEffortActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=scale,
    offset=0.0,
    clip=clip,
  )
