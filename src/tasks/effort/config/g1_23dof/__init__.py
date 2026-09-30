from src.tasks.effort.registration import register_effort_task

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

register_effort_task(
  task_id="Unitree-G1-23Dof-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm",
  env_cfg=unitree_g1_23dof_rough_env_cfg(),
  play_env_cfg=unitree_g1_23dof_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-23Dof-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_23dof_flat_env_cfg(),
  play_env_cfg=unitree_g1_23dof_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-23Dof-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm",
  env_cfg=unitree_g1_23dof_rough_mha_env_cfg(),
  play_env_cfg=unitree_g1_23dof_rough_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_ppo_mha_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-23Dof-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_23dof_flat_mha_env_cfg(),
  play_env_cfg=unitree_g1_23dof_flat_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_ppo_mha_runner_cfg(),
)
