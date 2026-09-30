from src.tasks.effort.registration import register_effort_task

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_flat_mha_env_cfg,
  unitree_go2_rough_env_cfg,
  unitree_go2_rough_mha_env_cfg,
)
from .rl_cfg import unitree_go2_ppo_mha_runner_cfg, unitree_go2_ppo_runner_cfg

register_effort_task(
  task_id="Unitree-Go2-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-Go2-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-Go2-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm",
  env_cfg=unitree_go2_rough_mha_env_cfg(),
  play_env_cfg=unitree_go2_rough_mha_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_mha_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-Go2-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm",
  env_cfg=unitree_go2_flat_mha_env_cfg(),
  play_env_cfg=unitree_go2_flat_mha_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_mha_runner_cfg(),
)
