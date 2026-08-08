"""TienKung 3 motion-tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import tk3_flat_tracking_env_cfg
from .rl_cfg import tk3_tracking_ppo_runner_cfg


# Actor omits motion_anchor_pos_b / base_lin_vel (deployable, no state estimation).
register_mjlab_task(
  task_id="TK3-Tracking",
  env_cfg=tk3_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=tk3_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=tk3_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
