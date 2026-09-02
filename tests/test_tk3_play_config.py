"""Regression tests for TK3 train and play configuration boundaries."""

from __future__ import annotations

import unittest
from collections.abc import Callable

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.tracking.config.tk3.env_cfgs import (
  tk3_flat_tracking_env_cfg as tk3_tracking_env_cfg,
)

EnvCfgFactory = Callable[..., ManagerBasedRlEnvCfg]


class TestTk3PlayConfig(unittest.TestCase):
  """Verify the train/play boundary for production TK3 tracking."""

  def test_production_friction_and_contact_capacity_experiment(self) -> None:
    cfg = tk3_tracking_env_cfg()

    joint_friction = cfg.events["joint_friction"]
    self.assertEqual(joint_friction.params["ranges"], (0.1, 6))
    self.assertEqual(joint_friction.params["operation"], "scale")
    self.assertEqual(cfg.sim.nconmax, 70)
    self.assertEqual(cfg.sim.mujoco.ccd_iterations, 100)

  def test_training_preserves_termination_policy(self) -> None:
    """Training variants must retain their intended reset boundaries."""
    cfg = tk3_tracking_env_cfg()

    self.assertEqual(cfg.episode_length_s, 20.0)
    self.assertSetEqual(
      set(cfg.terminations),
      {"time_out", "anchor_pos", "anchor_ori", "ee_body_pos"},
    )

  def test_play_runs_without_terminations(self) -> None:
    """Playback must not reset before the caller finishes its rollout."""
    cfg = tk3_tracking_env_cfg(play=True)

    self.assertEqual(cfg.episode_length_s, 1_000_000_000)
    self.assertEqual(cfg.terminations, {})


if __name__ == "__main__":
  unittest.main()
