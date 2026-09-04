"""Convert position actuators to zero-gain motors for feed-forward torque."""

from collections.abc import Sequence
from dataclasses import replace

from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg

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


def absolute_effort_action_cfg(
  actuators: Sequence[BuiltinPositionActuatorCfg],
) -> JointEffortActionCfg:
  """Build ``tau = clip(action * fraction * effort_limit, ±effort_limit)``.

  ``fraction`` is ``EFFORT_ACTION_SCALE_FRACTION``. Clip stays at the peak, so
  ``|action| = 1 / fraction`` saturates the motor.
  """
  scale: dict[str, float] = {}
  clip: dict[str, tuple[float, float]] = {}
  for actuator in actuators:
    limit = actuator.effort_limit
    assert limit is not None
    for expr in actuator.target_names_expr:
      peak = float(limit)
      scale[expr] = EFFORT_ACTION_SCALE_FRACTION * peak
      clip[expr] = (-peak, peak)
  return JointEffortActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=scale,
    offset=0.0,
    clip=clip,
  )
