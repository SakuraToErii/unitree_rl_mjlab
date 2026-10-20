from mjlab.tasks.registry import register_mjlab_task

from src.tasks.effort.rl import EffortOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_flat_imu_no_ankle_h1_no_curriculum_env_cfg,
  unitree_g1_flat_imu_no_ankle_mha_no_curriculum_env_cfg,
  unitree_g1_flat_imu_no_ankle_no_curriculum_env_cfg,
  unitree_g1_flat_mha_env_cfg,
  unitree_g1_flat_partial_env_cfg,
  unitree_g1_flat_partial_mha_env_cfg,
  unitree_g1_flat_partial_phase1_env_cfg,
  unitree_g1_flat_partial_phase1_mha_env_cfg,
  unitree_g1_flat_partial_phase1_speed_posture_env_cfg,
  unitree_g1_flat_speed_posture_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_rough_mha_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_mha_runner_cfg, unitree_g1_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-MLP-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_partial_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-H5-MLP-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_partial_phase1_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_phase1_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

# Speed-blended posture variants; the matching ids without SpeedPose are the
# L1 controls.
register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-SpeedPose-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_speed_posture_env_cfg(),
  play_env_cfg=unitree_g1_flat_speed_posture_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-H5-MLP-SpeedPose-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_partial_phase1_speed_posture_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_phase1_speed_posture_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-MLP-NoCurr-ObsNorm",
  env_cfg=unitree_g1_flat_imu_no_ankle_no_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_flat_imu_no_ankle_no_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H1-MLP-NoCurr-ObsNorm",
  env_cfg=unitree_g1_flat_imu_no_ankle_h1_no_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_flat_imu_no_ankle_h1_no_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm",
  env_cfg=unitree_g1_rough_mha_env_cfg(),
  play_env_cfg=unitree_g1_rough_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_partial_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm",
  env_cfg=unitree_g1_flat_partial_phase1_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_phase1_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-TimeMHA-NoCurr-ObsNorm",
  env_cfg=unitree_g1_flat_imu_no_ankle_mha_no_curriculum_env_cfg(),
  play_env_cfg=unitree_g1_flat_imu_no_ankle_mha_no_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)
