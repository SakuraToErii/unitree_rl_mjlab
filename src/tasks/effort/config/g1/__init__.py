from src.tasks.effort.registration import register_effort_task

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_flat_mha_env_cfg,
  unitree_g1_flat_partial_env_cfg,
  unitree_g1_flat_partial_mha_env_cfg,
  unitree_g1_flat_partial_no_ankle_group_time_env_cfg,
  unitree_g1_flat_partial_phase1_env_cfg,
  unitree_g1_flat_partial_phase1_no_curriculum_env_cfg,
  unitree_g1_flat_partial_phase1_group_time_env_cfg,
  unitree_g1_flat_partial_phase1_group_time_no_curriculum_env_cfg,
  unitree_g1_flat_partial_phase1_mha_env_cfg,
  unitree_g1_flat_partial_phase1_mha_no_curriculum_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_rough_mha_env_cfg,
)
from .rl_cfg import (
  unitree_g1_ppo_group_time_runner_cfg,
  unitree_g1_ppo_mha_runner_cfg,
  unitree_g1_ppo_runner_cfg,
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-MLP-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_partial_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-MLP-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_partial_phase1_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_phase1_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
)

_imu_mlp_no_curr_env_cfg = unitree_g1_flat_partial_phase1_no_curriculum_env_cfg()
register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-MLP-NoCurr-ObsNorm",
  env_cfg=_imu_mlp_no_curr_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_phase1_no_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm",
  env_cfg=unitree_g1_rough_mha_env_cfg(),
  play_env_cfg=unitree_g1_rough_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_partial_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_partial_phase1_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_phase1_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
)

_imu_mha_no_curr_env_cfg = unitree_g1_flat_partial_phase1_mha_no_curriculum_env_cfg()
register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-TimeMHA-NoCurr-ObsNorm",
  env_cfg=_imu_mha_no_curr_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_phase1_mha_no_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
)

_imu_group_time_env_cfg = unitree_g1_flat_partial_phase1_group_time_env_cfg()
register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-GroupTimeFusion-CmdCurr-ObsNorm",
  env_cfg=_imu_group_time_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_phase1_group_time_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_group_time_runner_cfg(_imu_group_time_env_cfg),
)

_imu_group_time_no_curr_env_cfg = (
  unitree_g1_flat_partial_phase1_group_time_no_curriculum_env_cfg()
)
register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-GroupTimeFusion-NoCurr-ObsNorm",
  env_cfg=_imu_group_time_no_curr_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_phase1_group_time_no_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_group_time_runner_cfg(_imu_group_time_no_curr_env_cfg),
)

_no_imu_group_time_env_cfg = unitree_g1_flat_partial_no_ankle_group_time_env_cfg()
register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-H5-GroupTimeFusion-CmdCurr-ObsNorm",
  env_cfg=_no_imu_group_time_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_no_ankle_group_time_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_group_time_runner_cfg(_no_imu_group_time_env_cfg),
)
