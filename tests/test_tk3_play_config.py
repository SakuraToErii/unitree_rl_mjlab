"""Regression tests for TK3 train and play configuration boundaries."""

from __future__ import annotations

import unittest
from collections.abc import Callable

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.ghost.config.tk3.env_cfgs import (
  tk3_flat_tracking_env_cfg as tk3_ghost_tracking_env_cfg,
)
from src.tasks.tracking.config.tk3.env_cfgs import (
  tk3_flat_tracking_env_cfg as tk3_tracking_env_cfg,
)

EnvCfgFactory = Callable[..., ManagerBasedRlEnvCfg]


class TestTk3PlayConfig(unittest.TestCase):
  """Verify the train/play boundary for production and Ghost TK3 tasks."""

  def test_training_preserves_termination_policy(self) -> None:
    """Training variants must retain their intended reset boundaries."""
    cases: tuple[tuple[str, EnvCfgFactory, float, set[str]], ...] = (
      (
        "production",
        tk3_tracking_env_cfg,
        20.0,
        {"time_out", "anchor_pos", "anchor_ori", "ee_body_pos"},
      ),
      (
        "ghost",
        tk3_ghost_tracking_env_cfg,
        20.0,
        {
          "time_out",
          "anchor_pos",
          "anchor_ori",
          "ee_body_pos",
          "nonfinite_state",
          "hard_joint_limit",
        },
      ),
    )

    for name, env_cfg_factory, episode_length_s, termination_names in cases:
      with self.subTest(name=name):
        cfg = env_cfg_factory()

        self.assertEqual(cfg.episode_length_s, episode_length_s)
        self.assertSetEqual(set(cfg.terminations), termination_names)

  def test_play_runs_without_terminations(self) -> None:
    """Playback must not reset before the caller finishes its rollout."""
    cases: tuple[tuple[str, EnvCfgFactory], ...] = (
      ("production", tk3_tracking_env_cfg),
      ("ghost", tk3_ghost_tracking_env_cfg),
    )

    for name, env_cfg_factory in cases:
      with self.subTest(name=name):
        cfg = env_cfg_factory(play=True)

        self.assertEqual(cfg.episode_length_s, 1_000_000_000)
        self.assertEqual(cfg.terminations, {})


if __name__ == "__main__":
  unittest.main()
