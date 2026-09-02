from mjlab.tasks.registry import register_mjlab_task

from src.tasks.effort.rl import EffortOnPolicyRunner

from .env_cfgs import (
  unitree_g1_23dof_flat_env_cfg,
  unitree_g1_23dof_flat_mha_env_cfg,
  unitree_g1_23dof_rough_env_cfg,
  unitree_g1_23dof_rough_mha_env_cfg,
)
from .rl_cfg import (
  unitree_g1_23dof_ppo_mha_runner_cfg,
  unitree_g1_23dof_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Effort-Rough",
  env_cfg=unitree_g1_23dof_rough_env_cfg(),
  play_env_cfg=unitree_g1_23dof_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Effort-Flat",
  env_cfg=unitree_g1_23dof_flat_env_cfg(),
  play_env_cfg=unitree_g1_23dof_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Effort-Rough-MHA",
  env_cfg=unitree_g1_23dof_rough_mha_env_cfg(),
  play_env_cfg=unitree_g1_23dof_rough_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_ppo_mha_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Effort-Flat-MHA",
  env_cfg=unitree_g1_23dof_flat_mha_env_cfg(),
  play_env_cfg=unitree_g1_23dof_flat_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_ppo_mha_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)
