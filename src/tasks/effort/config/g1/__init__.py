from src.tasks.effort.registration import register_effort_task

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_flat_mha_env_cfg,
  unitree_g1_flat_partial_env_cfg,
  unitree_g1_flat_partial_mha_env_cfg,
  unitree_g1_flat_partial_no_ankle_xattn_env_cfg,
  unitree_g1_flat_partial_phase1_env_cfg,
  unitree_g1_flat_partial_phase1_mha_env_cfg,
  unitree_g1_flat_partial_phase1_xattn_env_cfg,
  unitree_g1_flat_partial_phase1_xattn_no_curriculum_env_cfg,
  unitree_g1_flat_partial_xattn_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_rough_mha_env_cfg,
)
from .rl_cfg import (
  unitree_g1_ppo_cross_attn_runner_cfg,
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

_no_imu_joint_xattn_env_cfg = unitree_g1_flat_partial_xattn_env_cfg()
register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-JointXAttn-CmdCurr-ObsNorm",
  env_cfg=_no_imu_joint_xattn_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_xattn_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_cross_attn_runner_cfg(_no_imu_joint_xattn_env_cfg),
)

_no_imu_no_ankle_joint_xattn_env_cfg = unitree_g1_flat_partial_no_ankle_xattn_env_cfg()
register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-H5-JointXAttn-CmdCurr-ObsNorm",
  env_cfg=_no_imu_no_ankle_joint_xattn_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_no_ankle_xattn_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_cross_attn_runner_cfg(_no_imu_no_ankle_joint_xattn_env_cfg),
)

_imu_no_ankle_joint_xattn_env_cfg = unitree_g1_flat_partial_phase1_xattn_env_cfg()
register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-JointXAttn-CmdCurr-ObsNorm",
  env_cfg=_imu_no_ankle_joint_xattn_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_phase1_xattn_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_cross_attn_runner_cfg(_imu_no_ankle_joint_xattn_env_cfg),
)

_imu_no_ankle_joint_xattn_cmd_fixed_env_cfg = (
  unitree_g1_flat_partial_phase1_xattn_no_curriculum_env_cfg()
)
_imu_no_ankle_joint_xattn_cmd_fixed_rl_cfg = unitree_g1_ppo_cross_attn_runner_cfg(
  _imu_no_ankle_joint_xattn_cmd_fixed_env_cfg, use_bf16_autocast=False
)
register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-JointXAttn-CmdFixed-ObsNorm",
  env_cfg=_imu_no_ankle_joint_xattn_cmd_fixed_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_phase1_xattn_no_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=_imu_no_ankle_joint_xattn_cmd_fixed_rl_cfg,
)

# A separate task also gives playback the V2 head and isolates its checkpoints.
_imu_no_ankle_joint_shared_xattn_env_cfg = (
  unitree_g1_flat_partial_phase1_xattn_no_curriculum_env_cfg()
)
register_effort_task(
  task_id="Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-JointSharedXAttn-CmdFixed-ObsNorm",
  env_cfg=_imu_no_ankle_joint_shared_xattn_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_phase1_xattn_no_curriculum_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_cross_attn_runner_cfg(
    _imu_no_ankle_joint_shared_xattn_env_cfg,
    use_bf16_autocast=True,
    joint_shared_head=True,
  ),
)
