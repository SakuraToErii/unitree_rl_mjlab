from mjlab.tasks.registry import register_mjlab_task

from src.tasks.effort.rl import EffortOnPolicyRunner

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

register_mjlab_task(
  task_id="Unitree-G1-Effort-Rough",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-Flat",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-Flat-Partial",
  env_cfg=unitree_g1_flat_partial_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-Flat-Partial-Phase1",
  env_cfg=unitree_g1_flat_partial_phase1_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_phase1_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-Rough-MHA",
  env_cfg=unitree_g1_rough_mha_env_cfg(),
  play_env_cfg=unitree_g1_rough_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-Flat-MHA",
  env_cfg=unitree_g1_flat_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-Flat-Partial-MHA",
  env_cfg=unitree_g1_flat_partial_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Effort-Flat-Partial-Phase1-MHA",
  env_cfg=unitree_g1_flat_partial_phase1_mha_env_cfg(),
  play_env_cfg=unitree_g1_flat_partial_phase1_mha_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_mha_runner_cfg(),
  runner_cls=EffortOnPolicyRunner,
)

_partial_xattn_env_cfg = unitree_g1_flat_partial_xattn_env_cfg()
register_mjlab_task(
  task_id="Unitree-G1-Effort-Flat-Partial-XAttn",
  env_cfg=_partial_xattn_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_xattn_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_cross_attn_runner_cfg(_partial_xattn_env_cfg),
  runner_cls=EffortOnPolicyRunner,
)

_no_ankle_xattn_env_cfg = unitree_g1_flat_partial_no_ankle_xattn_env_cfg()
register_mjlab_task(
  task_id="Unitree-G1-Effort-Flat-Partial-NoAnkle-XAttn",
  env_cfg=_no_ankle_xattn_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_no_ankle_xattn_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_cross_attn_runner_cfg(_no_ankle_xattn_env_cfg),
  runner_cls=EffortOnPolicyRunner,
)

_phase1_xattn_env_cfg = unitree_g1_flat_partial_phase1_xattn_env_cfg()
register_mjlab_task(
  task_id="Unitree-G1-Effort-Flat-Partial-Phase1-XAttn",
  env_cfg=_phase1_xattn_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_phase1_xattn_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_cross_attn_runner_cfg(_phase1_xattn_env_cfg),
  runner_cls=EffortOnPolicyRunner,
)

_phase1_xattn_no_curriculum_env_cfg = (
  unitree_g1_flat_partial_phase1_xattn_no_curriculum_env_cfg()
)
_phase1_xattn_fp32_rl_cfg = unitree_g1_ppo_cross_attn_runner_cfg(
  _phase1_xattn_no_curriculum_env_cfg, use_bf16_autocast=False
)
_phase1_xattn_fp32_rl_cfg.experiment_name = "g1_effort_xattn_fp32_no_curriculum"
register_mjlab_task(
  task_id="Unitree-G1-Effort-Flat-Partial-Phase1-XAttn-NoCurriculum-FP32",
  env_cfg=_phase1_xattn_no_curriculum_env_cfg,
  play_env_cfg=unitree_g1_flat_partial_phase1_xattn_no_curriculum_env_cfg(
    play=True
  ),
  rl_cfg=_phase1_xattn_fp32_rl_cfg,
  runner_cls=EffortOnPolicyRunner,
)
