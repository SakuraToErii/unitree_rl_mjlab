"""Contracts for residual-effort velocity tasks."""

from __future__ import annotations

import re
import unittest
from dataclasses import asdict
from unittest.mock import Mock

import torch
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.lab_api.string import resolve_matching_names_values
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

import src.tasks  # noqa: F401
from src.tasks.effort.config.g1.action_cfg import (
  EFFORT_ACTION_CLIP,
  EFFORT_ACTION_LIMIT,
  NOMINAL_TORQUE,
  RESIDUAL_ACTION_SCALE,
  g1_residual_effort_action_cfg,
)
from src.tasks.effort.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_flat_mha_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_rough_mha_env_cfg,
)
from src.tasks.effort.config.g1.rl_cfg import (
  unitree_g1_ppo_mha_runner_cfg,
  unitree_g1_ppo_runner_cfg,
)
from src.tasks.effort.config.g1.robot_cfg import (
  EFFORT_STANDING_JOINT_POSITION,
  EFFORT_STANDING_ROOT_HEIGHT,
  get_g1_effort_robot_cfg,
)
from src.tasks.effort.rl.models import ResidualMhaModel, ResidualMlpModel

G1_EFFORT_TASK_IDS = (
  "Unitree-G1-Effort-Rough",
  "Unitree-G1-Effort-Flat",
  "Unitree-G1-Effort-Rough-MHA",
  "Unitree-G1-Effort-Flat-MHA",
)

ALL_EFFORT_TASK_IDS = (
  *G1_EFFORT_TASK_IDS,
  "Unitree-G1-23Dof-Effort-Rough",
  "Unitree-G1-23Dof-Effort-Flat",
  "Unitree-G1-23Dof-Effort-Rough-MHA",
  "Unitree-G1-23Dof-Effort-Flat-MHA",
  "Unitree-Go2-Effort-Rough",
  "Unitree-Go2-Effort-Flat",
  "Unitree-Go2-Effort-Rough-MHA",
  "Unitree-Go2-Effort-Flat-MHA",
)

VELOCITY_TASK_IDS = (
  "Unitree-G1-Rough",
  "Unitree-G1-Flat",
  "Unitree-G1-23Dof-Rough",
  "Unitree-G1-23Dof-Flat",
  "Unitree-Go2-Rough",
  "Unitree-Go2-Flat",
)


def _last_linear(module: torch.nn.Module) -> torch.nn.Linear:
  layers = [layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)]
  assert layers
  return layers[-1]


class EffortTaskTest(unittest.TestCase):
  def test_effort_tasks_are_registered_without_colliding_with_velocity(self) -> None:
    task_ids = list_tasks()
    for task_id in ALL_EFFORT_TASK_IDS:
      self.assertIn(task_id, task_ids)
    for task_id in VELOCITY_TASK_IDS:
      self.assertIn(task_id, task_ids)
      cfg = load_env_cfg(task_id)
      self.assertIn("joint_pos", cfg.actions)
      self.assertIsInstance(cfg.actions["joint_pos"], JointPositionActionCfg)

  def test_g1_effort_robot_has_zero_gain_actuators(self) -> None:
    cfg = get_g1_effort_robot_cfg()
    self.assertEqual(cfg.init_state.pos, (0.0, 0.0, EFFORT_STANDING_ROOT_HEIGHT))
    self.assertEqual(cfg.init_state.joint_pos, EFFORT_STANDING_JOINT_POSITION)
    self.assertIsNotNone(cfg.articulation)
    assert cfg.articulation is not None
    self.assertEqual(len(cfg.articulation.actuators), 6)

    for actuator_cfg in cfg.articulation.actuators:
      self.assertIsInstance(actuator_cfg, IdealPdActuatorCfg)
      self.assertEqual(actuator_cfg.stiffness, 0.0)
      self.assertEqual(actuator_cfg.damping, 0.0)

    robot = Entity(cfg)
    controlled_joints = [
      joint_name for actuator in robot.actuators for joint_name in actuator.target_names
    ]
    self.assertEqual(len(controlled_joints), 29)
    self.assertEqual(len(set(controlled_joints)), 29)
    self.assertEqual(set(controlled_joints), set(robot.joint_names))

  def test_residual_effort_action_matches_motor_limits_and_nominal_torque(self) -> None:
    robot = Entity(get_g1_effort_robot_cfg())
    joint_names = list(robot.joint_names)
    action_cfg = g1_residual_effort_action_cfg()
    self.assertIsInstance(action_cfg, JointEffortActionCfg)
    self.assertEqual(action_cfg.scale, RESIDUAL_ACTION_SCALE)
    self.assertEqual(action_cfg.offset, NOMINAL_TORQUE)
    self.assertEqual(action_cfg.clip, EFFORT_ACTION_CLIP)
    self.assertEqual(set(NOMINAL_TORQUE), set(joint_names))

    indices, _, limits = resolve_matching_names_values(EFFORT_ACTION_LIMIT, joint_names)
    self.assertEqual(sorted(indices), list(range(29)))
    self.assertEqual(len(limits), 29)
    self.assertTrue(
      all(
        RESIDUAL_ACTION_SCALE[expr] == 0.4 * limit
        for expr, limit in EFFORT_ACTION_LIMIT.items()
      )
    )

    env = Mock(spec=ManagerBasedRlEnv)
    env.num_envs = 2
    env.device = "cpu"
    env.scene = {"robot": robot}
    action = action_cfg.build(env)
    self.assertEqual(action.target_names, joint_names)

    zero_action = torch.zeros(2, 29)
    action.process_actions(zero_action)
    expected_nominal = torch.tensor([NOMINAL_TORQUE[name] for name in joint_names])
    torch.testing.assert_close(action._processed_actions[0], expected_nominal)

    action.process_actions(torch.full((2, 29), 100.0))
    expected_upper = torch.tensor(
      [
        next(
          limit
          for joint_expr, limit in EFFORT_ACTION_LIMIT.items()
          if re.fullmatch(joint_expr, name)
        )
        for name in joint_names
      ]
    )
    torch.testing.assert_close(action._processed_actions[0], expected_upper)

  def test_mha_variants_only_add_observation_history(self) -> None:
    pairs = (
      (unitree_g1_rough_env_cfg(), unitree_g1_rough_mha_env_cfg()),
      (unitree_g1_flat_env_cfg(), unitree_g1_flat_mha_env_cfg()),
    )
    for base_cfg, mha_cfg in pairs:
      self.assertEqual(base_cfg.rewards, mha_cfg.rewards)
      self.assertEqual(base_cfg.curriculum, mha_cfg.curriculum)
      self.assertEqual(
        base_cfg.observations["actor"].terms, mha_cfg.observations["actor"].terms
      )
      self.assertEqual(
        base_cfg.observations["critic"].terms, mha_cfg.observations["critic"].terms
      )
      self.assertEqual(mha_cfg.observations["actor"].history_length, 5)
      self.assertFalse(mha_cfg.observations["actor"].flatten_history_dim)
      self.assertEqual(mha_cfg.observations["critic"].history_length, 5)
      self.assertTrue(mha_cfg.observations["critic"].flatten_history_dim)

  def test_all_effort_variants_use_joint_effort_actions(self) -> None:
    for task_id in ALL_EFFORT_TASK_IDS:
      cfg = load_env_cfg(task_id)
      self.assertEqual(list(cfg.actions), ["joint_effort"])
      self.assertIsInstance(cfg.actions["joint_effort"], JointEffortActionCfg)

  def test_ppo_configs_use_effort_initialization_and_mha_model(self) -> None:
    ppo_cfg = unitree_g1_ppo_runner_cfg()
    mha_cfg = unitree_g1_ppo_mha_runner_cfg()

    self.assertTrue(ppo_cfg.actor.class_name.endswith(":ResidualMlpModel"))
    self.assertTrue(mha_cfg.actor.class_name.endswith(":ResidualMhaModel"))
    self.assertIs(resolve_callable(ppo_cfg.actor.class_name), ResidualMlpModel)
    self.assertIs(resolve_callable(mha_cfg.actor.class_name), ResidualMhaModel)
    self.assertEqual(
      ppo_cfg.actor.distribution_cfg,
      {
        "class_name": "GaussianDistribution",
        "init_std": 0.3,
        "std_type": "log",
      },
    )
    self.assertEqual(mha_cfg.actor.distribution_cfg, ppo_cfg.actor.distribution_cfg)
    self.assertEqual(ppo_cfg.max_iterations, 50_000)
    self.assertEqual(mha_cfg.max_iterations, 10_000)

    serialized_mha_cfg = asdict(mha_cfg)
    self.assertEqual(serialized_mha_cfg["actor"]["history_length"], 5)
    self.assertEqual(serialized_mha_cfg["actor"]["num_heads"], 4)

  def test_residual_models_center_actor_output_and_export_mha(self) -> None:
    batch_size = 8
    output_dim = 29
    distribution_cfg = {
      "class_name": "GaussianDistribution",
      "init_std": 0.1,
      "std_type": "log",
    }

    mlp_obs = TensorDict({"actor": torch.zeros(batch_size, 64)}, batch_size=[batch_size])
    mlp = ResidualMlpModel(
      mlp_obs,
      {"actor": ["actor"]},
      "actor",
      output_dim,
      hidden_dims=(64, 32),
      distribution_cfg=distribution_cfg.copy(),
    )
    mlp_output_layer = _last_linear(mlp.mlp)
    torch.testing.assert_close(
      mlp_output_layer.weight.norm(dim=1),
      torch.full((output_dim,), 0.01),
      atol=1.0e-6,
      rtol=1.0e-5,
    )
    torch.testing.assert_close(
      mlp_output_layer.bias, torch.zeros_like(mlp_output_layer.bias)
    )

    mha_obs = TensorDict(
      {"actor": torch.zeros(batch_size, 5, 64)}, batch_size=[batch_size]
    )
    mha = ResidualMhaModel(
      mha_obs,
      {"actor": ["actor"]},
      "actor",
      output_dim,
      hidden_dims=(64, 32),
      distribution_cfg=distribution_cfg.copy(),
      encoder_hidden_dim=32,
      num_heads=4,
    )
    mha_output_layer = _last_linear(mha.mlp)
    torch.testing.assert_close(
      mha_output_layer.weight.norm(dim=1),
      torch.full((output_dim,), 0.01),
      atol=1.0e-6,
      rtol=1.0e-5,
    )
    torch.testing.assert_close(
      mha_output_layer.bias, torch.zeros_like(mha_output_layer.bias)
    )

    expected = mha(mha_obs)
    exported = mha.as_onnx()(mha_obs["actor"])
    self.assertEqual(expected.shape, (batch_size, output_dim))
    torch.testing.assert_close(exported, expected)


if __name__ == "__main__":
  unittest.main()
