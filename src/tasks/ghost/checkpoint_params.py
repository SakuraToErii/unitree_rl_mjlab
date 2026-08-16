from __future__ import annotations

from pathlib import Path

import yaml


def read_run_action_params(
  checkpoint_path: Path,
) -> tuple[float, float | dict[str, float], float | None]:
  """读取本地 PyTorch checkpoint 同目录保存的动作语义。"""
  params_dir = checkpoint_path.parent / "params"
  agent_path = params_dir / "agent.yaml"
  env_path = params_dir / "env.yaml"
  missing = [path for path in (agent_path, env_path) if not path.is_file()]
  if missing:
    missing_text = ", ".join(str(path) for path in missing)
    raise FileNotFoundError(f"Missing saved run parameters: {missing_text}")

  # BaseLoader 只构造字符串、列表和字典，不实例化 YAML 中的 Python 对象。
  agent_params = yaml.load(agent_path.read_text(), Loader=yaml.BaseLoader)
  env_params = yaml.load(env_path.read_text(), Loader=yaml.BaseLoader)
  try:
    action_clip = float(agent_params["clip_actions"])
    joint_pos_params = env_params["actions"]["joint_pos"]
    raw_scale = joint_pos_params["scale"]
    residual_value = joint_pos_params.get("residual_clip")
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(
      f"Saved run parameters do not contain valid joint action settings in {params_dir}."
    ) from exc

  if isinstance(raw_scale, dict):
    action_scale: float | dict[str, float] = {
      str(name): float(value) for name, value in raw_scale.items()
    }
  else:
    action_scale = float(raw_scale)
  residual_clip = None if residual_value is None else float(residual_value)
  return action_clip, action_scale, residual_clip
