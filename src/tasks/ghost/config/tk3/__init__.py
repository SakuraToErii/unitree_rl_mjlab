"""TienKung 3 motion-tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.ghost.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import tk3_assault_tracking_env_cfg, tk3_flat_tracking_env_cfg
from .rl_cfg import tk3_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="TK3-Ghost-Tracking",
  env_cfg=tk3_flat_tracking_env_cfg(),
  play_env_cfg=tk3_flat_tracking_env_cfg(play=True),
  rl_cfg=tk3_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="TK3-Ghost-Tracking-Assault",
  env_cfg=tk3_assault_tracking_env_cfg(),
  play_env_cfg=tk3_assault_tracking_env_cfg(play=True),
  rl_cfg=tk3_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
