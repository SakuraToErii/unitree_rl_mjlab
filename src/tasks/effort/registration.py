"""Register effort tasks with an experiment directory matching their task ID.

Each registered observation, encoder, and curriculum variant gets its own
directory, so automatic checkpoint selection stays within that task's directory.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.effort.rl import EffortOnPolicyRunner


def register_effort_task(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  play_env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlOnPolicyRunnerCfg,
) -> None:
  """Register the task and derive its experiment name from the same identity."""
  rl_cfg.experiment_name = task_id.removeprefix("Unitree-").replace("-", "_").lower()
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_cfg,
    play_env_cfg=play_env_cfg,
    rl_cfg=rl_cfg,
    runner_cls=EffortOnPolicyRunner,
  )
