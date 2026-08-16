"""TienKung 3 motion-tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.ghost.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import tk3_qref_residual_prototype_env_cfg
from .rl_cfg import tk3_qref_residual_prototype_ppo_runner_cfg

register_mjlab_task(
  task_id="TK3-Ghost-Tracking-QRef-Prototype",
  env_cfg=tk3_qref_residual_prototype_env_cfg(),
  play_env_cfg=tk3_qref_residual_prototype_env_cfg(play=True),
  rl_cfg=tk3_qref_residual_prototype_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
