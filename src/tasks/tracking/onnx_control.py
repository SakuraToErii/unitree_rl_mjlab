"""Apply deployment control metadata from a tracking ONNX policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mjlab.actuator import BuiltinPositionActuator, XmlActuator
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_VECTOR_METADATA_KEYS = (
  "joint_stiffness",
  "joint_damping",
  "action_scale",
  "default_joint_pos",
  "joint_effort_limit",
)


@dataclass(frozen=True)
class OnnxControlOverlay:
  """Validated control parameters exported with a tracking ONNX policy.

  ``apply_action_cfg`` is the pre-construction phase: it configures action
  scaling before MJLab builds the action term. ``apply_runtime`` is the
  post-construction phase: it applies controller gains, effort limits, and the
  policy's default pose to the live environment.
  """

  source: Path
  joint_names: tuple[str, ...]
  joint_stiffness: tuple[float, ...] | None = None
  joint_damping: tuple[float, ...] | None = None
  action_scale: tuple[float, ...] | None = None
  default_joint_pos: tuple[float, ...] | None = None
  joint_effort_limit: tuple[float, ...] | None = None

  def __post_init__(self) -> None:
    if not self.joint_names:
      raise ValueError(f"{self.source}: ONNX joint_names metadata is empty.")
    _raise_for_duplicates(self.source, "joint_names", self.joint_names)

    expected = len(self.joint_names)
    for key in _VECTOR_METADATA_KEYS:
      values = getattr(self, key)
      if values is None:
        continue
      if len(values) != expected:
        raise ValueError(
          f"{self.source}: ONNX {key} has {len(values)} values; "
          f"expected {expected} from joint_names."
        )
      for index, value in enumerate(values):
        if not math.isfinite(value):
          raise ValueError(
            f"{self.source}: ONNX {key}[{index}] must be finite; got {value}."
          )

    if (self.joint_stiffness is None) != (self.joint_damping is None):
      raise ValueError(
        f"{self.source}: ONNX joint_stiffness and joint_damping must be "
        "provided together."
      )
    for key in ("joint_stiffness", "joint_damping"):
      values = getattr(self, key)
      if values is None:
        continue
      for index, value in enumerate(values):
        if value < 0.0:
          raise ValueError(
            f"{self.source}: ONNX {key}[{index}] must be non-negative; got {value}."
          )
    if self.action_scale is not None:
      for index, scale in enumerate(self.action_scale):
        if scale == 0.0:
          raise ValueError(
            f"{self.source}: ONNX action_scale[{index}] must be non-zero."
          )
    if self.joint_effort_limit is not None:
      for index, effort in enumerate(self.joint_effort_limit):
        if effort < 0.0:
          raise ValueError(
            f"{self.source}: ONNX joint_effort_limit[{index}] must be "
            f"non-negative; got {effort}."
          )

  @classmethod
  def from_onnx(cls, onnx_path: str | Path) -> OnnxControlOverlay:
    """Read and validate control metadata from an ONNX model."""
    source = Path(onnx_path).expanduser().resolve()
    if not source.is_file():
      raise FileNotFoundError(f"ONNX policy does not exist: {source}")

    import onnx

    model = onnx.load(str(source), load_external_data=False)
    metadata: dict[str, str] = {}
    for prop in model.metadata_props:
      if prop.key in metadata:
        raise ValueError(
          f"{source}: ONNX metadata contains duplicate key {prop.key!r}."
        )
      metadata[prop.key] = prop.value

    joint_names = _parse_joint_names(source, metadata.get("joint_names"))
    expected = len(joint_names)
    return cls(
      source=source,
      joint_names=joint_names,
      joint_stiffness=_parse_optional_vector(
        source, metadata, "joint_stiffness", expected
      ),
      joint_damping=_parse_optional_vector(source, metadata, "joint_damping", expected),
      action_scale=_parse_optional_vector(source, metadata, "action_scale", expected),
      default_joint_pos=_parse_optional_vector(
        source, metadata, "default_joint_pos", expected
      ),
      joint_effort_limit=_parse_optional_vector(
        source, metadata, "joint_effort_limit", expected
      ),
    )

  def apply_action_cfg(self, action_cfg: JointPositionActionCfg) -> tuple[str, ...]:
    """Apply action scale before constructing ``ManagerBasedRlEnv``."""
    if not isinstance(action_cfg, JointPositionActionCfg):
      raise TypeError(
        "ONNX action_scale requires a JointPositionActionCfg; got "
        f"{type(action_cfg).__name__}."
      )
    if self.action_scale is None:
      return ()
    action_cfg.scale = dict(zip(self.joint_names, self.action_scale, strict=True))
    return ("action_scale",)

  def apply_runtime(
    self,
    env: ManagerBasedRlEnv,
    *,
    action_name: str = "joint_pos",
  ) -> tuple[str, ...]:
    """Apply runtime parameters after constructing ``ManagerBasedRlEnv``.

    The action term must control exactly the joints declared by the ONNX. When
    action-scale metadata is present, this also verifies that the pre-build
    phase was applied (or that the existing configuration already matches).
    """
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, JointPositionAction):
      raise TypeError(
        f"Action term {action_name!r} must be a JointPositionAction; got "
        f"{type(action).__name__}."
      )

    target_names = tuple(action.target_names)
    _raise_for_duplicates(self.source, f"action term {action_name!r}", target_names)
    _validate_same_joints(self.source, self.joint_names, target_names)
    ordered_scale = _values_in_order(self.joint_names, self.action_scale, target_names)
    if ordered_scale is not None:
      _validate_action_scale(self.source, action.scale, ordered_scale)

    robot = env.scene[action.cfg.entity_name]
    target_ids = action.target_ids
    if target_ids.ndim != 1 or len(target_ids) != len(target_names):
      raise ValueError(
        f"{self.source}: action term {action_name!r} has invalid target_ids "
        f"shape {tuple(target_ids.shape)} for {len(target_names)} targets."
      )

    ordered_default = _values_in_order(
      self.joint_names, self.default_joint_pos, target_names
    )
    if ordered_default is not None:
      _validate_default_pose_targets(
        self.source,
        robot.data.default_joint_pos,
        action.offset,
        target_ids,
        len(target_names),
      )

    ctrl_ids: torch.Tensor | None = None
    if self.joint_stiffness is not None or self.joint_effort_limit is not None:
      ctrl_ids = _position_ctrl_ids(self.source, robot, self.joint_names, env.device)

    applied: list[str] = []
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    if self.joint_stiffness is not None:
      assert self.joint_damping is not None
      assert ctrl_ids is not None
      gainprm = env.sim.model.actuator_gainprm
      biasprm = env.sim.model.actuator_biasprm
      kp = torch.as_tensor(self.joint_stiffness, device=env.device, dtype=gainprm.dtype)
      kd = torch.as_tensor(self.joint_damping, device=env.device, dtype=biasprm.dtype)
      gainprm[env_ids[:, None], ctrl_ids, 0] = kp
      biasprm[env_ids[:, None], ctrl_ids, 1] = -kp
      biasprm[env_ids[:, None], ctrl_ids, 2] = -kd
      applied.append("Kp/Kd")

    if self.joint_effort_limit is not None:
      assert ctrl_ids is not None
      forcerange = env.sim.model.actuator_forcerange
      effort = torch.as_tensor(
        self.joint_effort_limit, device=env.device, dtype=forcerange.dtype
      )
      forcerange[env_ids[:, None], ctrl_ids, 0] = -effort
      forcerange[env_ids[:, None], ctrl_ids, 1] = effort
      applied.append("effort_limit")

    if ordered_default is not None:
      defaults = robot.data.default_joint_pos
      default_values = torch.as_tensor(
        ordered_default, device=defaults.device, dtype=defaults.dtype
      )
      defaults[:, target_ids] = default_values
      offset = action.offset
      assert isinstance(offset, torch.Tensor)
      offset_values = default_values.to(device=offset.device, dtype=offset.dtype)
      offset.copy_(_broadcast_vector(offset_values, offset.shape))
      applied.append("default_joint_pos")

    return tuple(applied)


def _parse_joint_names(source: Path, raw: str | None) -> tuple[str, ...]:
  if raw is None or not raw.strip():
    raise ValueError(f"{source}: ONNX has no joint_names metadata.")
  names = tuple(item.strip() for item in raw.split(","))
  empty = [index for index, name in enumerate(names) if not name]
  if empty:
    raise ValueError(
      f"{source}: ONNX joint_names contains empty entries at indices {empty}."
    )
  return names


def _parse_optional_vector(
  source: Path,
  metadata: dict[str, str],
  key: str,
  expected: int,
) -> tuple[float, ...] | None:
  raw = metadata.get(key)
  if raw is None or not raw.strip():
    return None
  parts = tuple(item.strip() for item in raw.split(","))
  empty = [index for index, item in enumerate(parts) if not item]
  if empty:
    raise ValueError(f"{source}: ONNX {key} contains empty entries at indices {empty}.")
  try:
    values = tuple(float(item) for item in parts)
  except ValueError as error:
    raise ValueError(
      f"{source}: ONNX {key} must contain comma-separated numbers; got {raw!r}."
    ) from error
  if len(values) == 1 and expected > 1:
    values *= expected
  if len(values) != expected:
    raise ValueError(
      f"{source}: ONNX {key} has {len(values)} values; "
      f"expected {expected} from joint_names."
    )
  return values


def _raise_for_duplicates(source: Path, label: str, names: tuple[str, ...]) -> None:
  seen: set[str] = set()
  duplicates: list[str] = []
  for name in names:
    if name in seen and name not in duplicates:
      duplicates.append(name)
    seen.add(name)
  if duplicates:
    raise ValueError(f"{source}: {label} contains duplicate names: {duplicates}.")


def _validate_same_joints(
  source: Path,
  onnx_names: tuple[str, ...],
  target_names: tuple[str, ...],
) -> None:
  onnx_set = set(onnx_names)
  target_set = set(target_names)
  missing_from_action = sorted(onnx_set - target_set)
  missing_from_onnx = sorted(target_set - onnx_set)
  if missing_from_action or missing_from_onnx:
    raise ValueError(
      f"{source}: ONNX joints and runtime action targets differ; "
      f"not controlled by the action={missing_from_action}, "
      f"absent from ONNX metadata={missing_from_onnx}."
    )


def _values_in_order(
  source_names: tuple[str, ...],
  values: tuple[float, ...] | None,
  target_names: tuple[str, ...],
) -> tuple[float, ...] | None:
  if values is None:
    return None
  by_name = dict(zip(source_names, values, strict=True))
  return tuple(by_name[name] for name in target_names)


def _validate_action_scale(
  source: Path,
  actual: torch.Tensor | float,
  expected_values: tuple[float, ...],
) -> None:
  if isinstance(actual, (float, int)):
    if all(
      math.isclose(float(actual), value, rel_tol=1.0e-6, abs_tol=1.0e-7)
      for value in expected_values
    ):
      return
    matches = False
  elif isinstance(actual, torch.Tensor):
    if actual.ndim == 0 or actual.shape[-1] != len(expected_values):
      matches = False
    else:
      expected = torch.as_tensor(
        expected_values, device=actual.device, dtype=actual.dtype
      )
      expected = _broadcast_vector(expected, actual.shape)
      matches = bool(torch.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-7))
  else:
    raise TypeError(
      f"{source}: runtime action scale has unsupported type {type(actual).__name__}."
    )
  if not matches:
    raise ValueError(
      f"{source}: runtime action scale does not match the ONNX metadata. "
      "Call apply_action_cfg() before constructing the environment."
    )


def _validate_default_pose_targets(
  source: Path,
  defaults: torch.Tensor,
  offset: torch.Tensor | float,
  target_ids: torch.Tensor,
  num_targets: int,
) -> None:
  if defaults.ndim != 2:
    raise ValueError(
      f"{source}: robot default_joint_pos must be rank 2; "
      f"got shape {tuple(defaults.shape)}."
    )
  if bool(torch.any(target_ids < 0)) or bool(
    torch.any(target_ids >= defaults.shape[1])
  ):
    raise ValueError(f"{source}: action target_ids are outside default_joint_pos.")
  if not isinstance(offset, torch.Tensor):
    raise TypeError(
      f"{source}: cannot apply default_joint_pos to an action with a scalar "
      "offset; use a default-offset JointPositionAction."
    )
  if offset.ndim == 0 or offset.shape[-1] != num_targets:
    raise ValueError(
      f"{source}: runtime action offset has shape {tuple(offset.shape)}; "
      f"expected a final dimension of {num_targets}."
    )


def _broadcast_vector(values: torch.Tensor, shape: torch.Size) -> torch.Tensor:
  view_shape = (1,) * (len(shape) - 1) + (len(values),)
  return values.reshape(view_shape).expand(shape)


def _position_ctrl_ids(
  source: Path,
  robot,
  joint_names: tuple[str, ...],
  device: str,
) -> torch.Tensor:
  requested = set(joint_names)
  ctrl_by_name: dict[str, int] = {}
  for actuator in robot.actuators:
    relevant = requested.intersection(actuator.target_names)
    if not relevant:
      continue
    supports_position_pd = isinstance(actuator, BuiltinPositionActuator) or (
      isinstance(actuator, XmlActuator) and actuator.command_field == "position"
    )
    if not supports_position_pd:
      raise TypeError(
        f"{source}: ONNX Kp/Kd and effort overlays require position "
        f"actuators; {type(actuator).__name__} controls {sorted(relevant)}."
      )
    ctrl_ids = actuator.global_ctrl_ids.detach().cpu().tolist()
    if len(ctrl_ids) != len(actuator.target_names):
      raise ValueError(
        f"{source}: actuator {type(actuator).__name__} exposes "
        f"{len(ctrl_ids)} controls for {len(actuator.target_names)} targets."
      )
    for name, ctrl_id in zip(actuator.target_names, ctrl_ids, strict=True):
      if name not in requested:
        continue
      if name in ctrl_by_name:
        raise ValueError(
          f"{source}: joint {name!r} maps to more than one position actuator."
        )
      ctrl_by_name[name] = int(ctrl_id)

  missing = [name for name in joint_names if name not in ctrl_by_name]
  if missing:
    raise ValueError(
      f"{source}: ONNX joints have no position actuator mapping: {missing}."
    )
  return torch.tensor(
    [ctrl_by_name[name] for name in joint_names],
    device=device,
    dtype=torch.long,
  )
