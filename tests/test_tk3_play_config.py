"""Regression tests for TK3 train and play configuration boundaries."""

from __future__ import annotations

import unittest
from collections.abc import Callable

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks

from src.tasks.ghost.config.tk3.env_cfgs import (
  tk3_qref_residual_prototype_env_cfg as tk3_ghost_tracking_env_cfg,
)
from src.tasks.ghost.mdp.actions import ReferenceJointPositionLimitActionCfg
from src.tasks.ghost.mdp.commands import MotionCommandCfg
from src.tasks.tracking.config.tk3.env_cfgs import (
  tk3_flat_tracking_env_cfg as tk3_tracking_env_cfg,
)

EnvCfgFactory = Callable[..., ManagerBasedRlEnvCfg]


class TestTk3PlayConfig(unittest.TestCase):
  """Verify the train/play boundary for production and Ghost TK3 tasks."""

  def test_only_qref_ghost_task_is_registered(self) -> None:
    ghost_tasks = [task for task in list_tasks() if task.startswith("TK3-Ghost")]

    self.assertEqual(ghost_tasks, ["TK3-Ghost-Tracking-QRef-Prototype"])

  def test_production_friction_and_contact_capacity_experiment(self) -> None:
    cfg = tk3_tracking_env_cfg()

    joint_friction = cfg.events["joint_friction"]
    self.assertEqual(joint_friction.params["ranges"], (0.1, 6))
    self.assertEqual(joint_friction.params["operation"], "scale")
    self.assertEqual(cfg.sim.nconmax, 70)
    self.assertEqual(cfg.sim.mujoco.ccd_iterations, 100)

    ghost_cfg = tk3_ghost_tracking_env_cfg()
    self.assertEqual(ghost_cfg.sim.nconmax, 70)
    self.assertEqual(ghost_cfg.sim.mujoco.ccd_iterations, 50)

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

  def test_ghost_training_anneals_command_noise_by_halfway(self) -> None:
    motion_cfg = tk3_ghost_tracking_env_cfg().commands["motion"]
    self.assertIsInstance(motion_cfg, MotionCommandCfg)
    self.assertEqual(motion_cfg.command_noise_anneal_start_step, 0)
    self.assertEqual(motion_cfg.command_noise_anneal_end_step, 1_200_000)

  def test_ghost_training_applies_small_spawn_pose_rsi(self) -> None:
    motion_cfg = tk3_ghost_tracking_env_cfg().commands["motion"]
    self.assertIsInstance(motion_cfg, MotionCommandCfg)
    self.assertEqual(
      motion_cfg.pose_range,
      {
        "x": (-0.03, 0.03),
        "y": (-0.03, 0.03),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
    )
    self.assertEqual(motion_cfg.joint_position_range, (0.0, 0.0))

  def test_ghost_uses_only_qref_residual_actions(self) -> None:
    cfg = tk3_ghost_tracking_env_cfg()

    self.assertIsInstance(
      cfg.actions["joint_pos"], ReferenceJointPositionLimitActionCfg
    )
    self.assertNotIn("motion_joint_pos", cfg.rewards)
    self.assertNotIn("motion_joint_vel", cfg.rewards)

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
        if name.startswith("ghost"):
          motion_cfg = cfg.commands["motion"]
          self.assertIsInstance(motion_cfg, MotionCommandCfg)
          self.assertEqual(motion_cfg.pose_range, {})
          self.assertEqual(motion_cfg.joint_position_range, (0.0, 0.0))


if __name__ == "__main__":
  unittest.main()
