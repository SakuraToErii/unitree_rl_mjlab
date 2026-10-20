from src.tasks.effort.registration import register_effort_task

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_flat_mha_env_cfg,
  unitree_g1_flat_noankle_nojvel_nophase_h1_nocurr_env_cfg,
  unitree_g1_flat_noankle_nojvel_nophase_mha_nocurr_env_cfg,
  unitree_g1_flat_noankle_nojvel_nophase_nocurr_env_cfg,
  unitree_g1_flat_noimu_noankle_nojvel_nophase_env_cfg,
  unitree_g1_flat_noimu_noankle_nojvel_nophase_mha_env_cfg,
  unitree_g1_flat_noimu_nojvel_nophase_env_cfg,
  unitree_g1_flat_noimu_nojvel_nophase_mha_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_rough_mha_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_mha_runner_cfg, unitree_g1_ppo_runner_cfg

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-L1Pose-CmdCurr-TerrainCurr",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-L1Pose-CmdCurr",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-NoPhase-H5-MLP-L1Pose-CmdCurr",
  env_cfg=unitree_g1_flat_noimu_nojvel_nophase_env_cfg(),
  play_env_cfg=unitree_g1_flat_noimu_nojvel_nophase_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-NoPhase-H5-MLP-L1Pose-CmdCurr",
  env_cfg=unitree_g1_flat_noimu_noankle_nojvel_nophase_env_cfg(),
  play_env_cfg=unitree_g1_flat_noimu_noankle_nojvel_nophase_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-MLP-L1Pose-NoCurr",
  env_cfg=unitree_g1_flat_noankle_nojvel_nophase_nocurr_env_cfg(),
  play_env_cfg=unitree_g1_flat_noankle_nojvel_nophase_nocurr_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H1-MLP-L1Pose-NoCurr",
  env_cfg=unitree_g1_flat_noankle_nojvel_nophase_h1_nocurr_env_cfg(),
  play_env_cfg=unitree_g1_flat_noankle_nojvel_nophase_h1_nocurr_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-L1Pose-CmdCurr-TerrainCurr",
  env_cfg=unitree_g1_rough_mha_env_cfg(),
  play_env_cfg=unitree_g1_rough_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-L1Pose-CmdCurr",
  env_cfg=unitree_g1_flat_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-NoPhase-H5-TimeMHA-L1Pose-CmdCurr",
  env_cfg=unitree_g1_flat_noimu_nojvel_nophase_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_noimu_nojvel_nophase_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-NoPhase-H5-TimeMHA-L1Pose-CmdCurr",
  env_cfg=unitree_g1_flat_noimu_noankle_nojvel_nophase_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_noimu_noankle_nojvel_nophase_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
)

register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-TimeMHA-L1Pose-NoCurr",
  env_cfg=unitree_g1_flat_noankle_nojvel_nophase_mha_nocurr_env_cfg(),
  play_env_cfg=unitree_g1_flat_noankle_nojvel_nophase_mha_nocurr_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
)
