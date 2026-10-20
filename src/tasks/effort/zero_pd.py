"""Convert position actuators to zero-gain motors for feed-forward torque."""

from dataclasses import replace

from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# Policy action 1.0 maps to this fraction of each motor's peak torque.
# Clip stays at the full peak, so |action| = 1 / fraction saturates the motor.
EFFORT_ACTION_SCALE_FRACTION = 0.25


def zero_pd_actuators(
  actuators: tuple[BuiltinPositionActuatorCfg, ...],
) -> tuple[IdealPdActuatorCfg, ...]:
  """Return the same actuator groups with ``kp = kd = 0``.

  Physical joint overrides remain attached to each actuator group. In
  particular, ``viscous_damping`` is the passive joint damping term; it is
  distinct from the active PD ``damping`` gain set to zero here.
  """
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
        viscous_damping=actuator.viscous_damping,
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

