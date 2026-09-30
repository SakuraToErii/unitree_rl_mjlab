"""Contracts for residual-effort velocity tasks."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import torch
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.string import resolve_matching_names_values
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

import src.tasks  # noqa: F401
import src.tasks.effort.mdp as mdp
from src.assets.robots.unitree_g1.g1_constants import (
  ARMATURE_4010,
  ARMATURE_5020,
  ARMATURE_7520_14,
  ARMATURE_7520_22,
)
from src.tasks.effort.config.g1.action_cfg import (
  EFFORT_ACTION_CLIP,
  EFFORT_ACTION_LIMIT,
  EFFORT_ACTION_SCALE,
  g1_effort_action_cfg,
)
from src.tasks.effort.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_flat_mha_env_cfg,
  unitree_g1_flat_partial_env_cfg,
  unitree_g1_flat_partial_mha_env_cfg,
  unitree_g1_flat_partial_phase1_env_cfg,
  unitree_g1_flat_partial_phase1_mha_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_rough_mha_env_cfg,
)
from src.tasks.effort.config.g1.rl_cfg import (
  unitree_g1_ppo_cross_attn_runner_cfg,
  unitree_g1_ppo_mha_runner_cfg,
  unitree_g1_ppo_runner_cfg,
)
from src.tasks.effort.config.g1_23dof.rl_cfg import (
  unitree_g1_23dof_ppo_mha_runner_cfg,
  unitree_g1_23dof_ppo_runner_cfg,
)
from src.tasks.effort.config.go2.rl_cfg import (
  unitree_go2_ppo_mha_runner_cfg,
  unitree_go2_ppo_runner_cfg,
)
from src.tasks.effort.config.g1.robot_cfg import (
  EFFORT_STANDING_JOINT_POSITION,
  EFFORT_STANDING_ROOT_HEIGHT,
  get_g1_effort_robot_cfg,
)
from src.tasks.effort.rl.bf16_autocast import Bf16AutocastModel, Bf16AutocastPPO
from src.tasks.effort.rl.joint_attention import (
  mean_joint_attention,
  plot_joint_attention,
)
from src.tasks.effort.rl.models import (
  ResidualCrossAttnModel,
  ResidualMhaModel,
  ResidualMlpModel,
)
from src.tasks.effort.rl.runner import get_effort_metadata
from src.tasks.effort.zero_pd import EFFORT_ACTION_SCALE_FRACTION

G1_EFFORT_TASK_IDS = (
  "Unitree-G1-Effort-Rough",
  "Unitree-G1-Effort-Flat",
  "Unitree-G1-Effort-Flat-Partial",
  "Unitree-G1-Effort-Flat-Partial-Phase1",
  "Unitree-G1-Effort-Flat-Partial-MHA",
  "Unitree-G1-Effort-Flat-Partial-Phase1-MHA",
  "Unitree-G1-Effort-Flat-Partial-XAttn",
  "Unitree-G1-Effort-Flat-Partial-NoAnkle-XAttn",
  "Unitree-G1-Effort-Flat-Partial-Phase1-XAttn",
  "Unitree-G1-Effort-Rough-MHA",
  "Unitree-G1-Effort-Flat-MHA",
)

_G1_ANKLE_JOINTS = (
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
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


class _FakeActionManager:
  def __init__(self, term, term_name="joint_effort"):
    self._term = term
    self.active_terms = [term_name]
    self.action_term_dim = [term.action_dim]
    self.prev_action = torch.zeros(term.num_envs, term.action_dim)

  def get_term(self, name):
    return self._term


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

  def test_g1_effort_applies_unitree_mujoco_joint_defaults(self) -> None:
    robot = Entity(get_g1_effort_robot_cfg())
    model = robot.spec.compile()

    for joint_name in robot.joint_names:
      joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
      dof_id = int(model.jnt_dofadr[joint_id])
      expected_frictionloss = (
        0.1
        if joint_name.endswith(("wrist_pitch_joint", "wrist_yaw_joint"))
        else 0.2
      )
      if joint_name.endswith(("wrist_pitch_joint", "wrist_yaw_joint")):
        expected_armature = ARMATURE_4010
      elif joint_name.endswith(("ankle_pitch_joint", "ankle_roll_joint")) or (
        joint_name in ("waist_pitch_joint", "waist_roll_joint")
      ):
        expected_armature = ARMATURE_5020 * 2
      elif joint_name.endswith(("hip_pitch_joint", "hip_yaw_joint")) or (
        joint_name == "waist_yaw_joint"
      ):
        expected_armature = ARMATURE_7520_14
      elif joint_name.endswith(("hip_roll_joint", "knee_joint")):
        expected_armature = ARMATURE_7520_22
      else:
        expected_armature = ARMATURE_5020
      self.assertAlmostEqual(float(model.dof_damping[dof_id]), 0.05)
      self.assertAlmostEqual(float(model.dof_armature[dof_id]), expected_armature)
      self.assertAlmostEqual(
        float(model.dof_frictionloss[dof_id]), expected_frictionloss
      )

    free_joint_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
    )
    free_dof_id = int(model.jnt_dofadr[free_joint_id])
    self.assertEqual(
      model.dof_damping[free_dof_id : free_dof_id + 6].tolist(),
      [0.0] * 6,
    )
    self.assertEqual(
      model.dof_armature[free_dof_id : free_dof_id + 6].tolist(),
      [0.0] * 6,
    )
    self.assertEqual(
      model.dof_frictionloss[free_dof_id : free_dof_id + 6].tolist(),
      [0.0] * 6,
    )

  def test_absolute_effort_action_scales_by_fraction_and_clips_at_y1(self) -> None:
    robot = Entity(get_g1_effort_robot_cfg())
    joint_names = list(robot.joint_names)
    action_cfg = g1_effort_action_cfg()
    self.assertIsInstance(action_cfg, JointEffortActionCfg)
    self.assertEqual(action_cfg.scale, EFFORT_ACTION_SCALE)
    self.assertEqual(action_cfg.offset, 0.0)
    self.assertEqual(action_cfg.clip, EFFORT_ACTION_CLIP)

    indices, _, limits = resolve_matching_names_values(EFFORT_ACTION_LIMIT, joint_names)
    self.assertEqual(sorted(indices), list(range(29)))
    self.assertEqual(len(limits), 29)

    env = Mock(spec=ManagerBasedRlEnv)
    env.num_envs = 2
    env.device = "cpu"
    env.scene = {"robot": robot}
    action = action_cfg.build(env)
    self.assertEqual(action.target_names, joint_names)
    expected_y1 = torch.tensor(
      [
        next(
          limit
          for joint_expr, limit in EFFORT_ACTION_LIMIT.items()
          if re.fullmatch(joint_expr, name)
        )
        for name in joint_names
      ]
    )

    action.process_actions(torch.zeros(2, 29))
    torch.testing.assert_close(action._processed_actions, torch.zeros(2, 29))

    action.process_actions(torch.ones(2, 29))
    torch.testing.assert_close(
      action._processed_actions[0], EFFORT_ACTION_SCALE_FRACTION * expected_y1
    )

    saturate = 1.0 / EFFORT_ACTION_SCALE_FRACTION
    action.process_actions(torch.full((2, 29), saturate))
    torch.testing.assert_close(action._processed_actions[0], expected_y1)

    action.process_actions(torch.full((2, 29), 100.0))
    torch.testing.assert_close(action._processed_actions[0], expected_y1)

  def test_effort_export_metadata_uses_torque_action_contract(self) -> None:
    env = ManagerBasedRlEnv(load_env_cfg("Unitree-G1-Effort-Flat"), device="cpu")
    try:
      with self.assertRaises(KeyError):
        get_base_metadata(env, "test-effort")

      metadata = get_effort_metadata(env, "test-effort")
      joint_names = list(env.scene["robot"].joint_names)
      self.assertEqual(metadata["run_path"], "test-effort")
      self.assertEqual(metadata["action_term"], "joint_effort")
      self.assertEqual(metadata["action_names"], joint_names)
      for actual, expected in zip(
        metadata["action_offset"],
        [0.0] * len(joint_names),
        strict=True,
      ):
        self.assertAlmostEqual(actual, expected, places=5)
      expected_scale = [
        next(
          EFFORT_ACTION_SCALE_FRACTION * limit
          for joint_expr, limit in EFFORT_ACTION_LIMIT.items()
          if re.fullmatch(joint_expr, name)
        )
        for name in joint_names
      ]
      for actual, expected in zip(
        metadata["action_scale"], expected_scale, strict=True
      ):
        self.assertAlmostEqual(actual, expected, places=5)
      expected_clip = [
        [
          -next(
            limit
            for joint_expr, limit in EFFORT_ACTION_LIMIT.items()
            if re.fullmatch(joint_expr, name)
          ),
          next(
            limit
            for joint_expr, limit in EFFORT_ACTION_LIMIT.items()
            if re.fullmatch(joint_expr, name)
          ),
        ]
        for name in joint_names
      ]
      for actual, expected in zip(
        metadata["action_clip"], expected_clip, strict=True
      ):
        for actual_bound, expected_bound in zip(actual, expected, strict=True):
          self.assertAlmostEqual(actual_bound, expected_bound, places=5)
      self.assertEqual(metadata["joint_stiffness"], [0.0] * len(joint_names))
      self.assertEqual(metadata["joint_damping"], [0.0] * len(joint_names))
      self.assertNotEqual(
        metadata["action_clip"][joint_names.index("left_ankle_pitch_joint")],
        [-50.0, 50.0],
      )
      self.assertEqual(len(metadata["joint_effort_limit"]), len(joint_names))
    finally:
      env.close()

  def test_g1_effort_rewards_use_walkable_pose_split(self) -> None:
    cfg = load_env_cfg("Unitree-G1-Effort-Flat")

    self.assertNotIn("pose", cfg.rewards)
    self.assertIs(cfg.rewards["stand_still"].func, mdp.stand_still)
    self.assertEqual(mdp.stand_still.__module__, "src.tasks.effort.mdp.rewards")
    self.assertEqual(cfg.rewards["stand_still"].weight, -1.0)
    self.assertEqual(cfg.rewards["stand_still"].params["command_threshold"], 0.1)
    self.assertNotIn("action_rate_l2", cfg.rewards)
    self.assertEqual(cfg.rewards["joint_deviation_arms"].weight, -0.3)
    self.assertEqual(
      cfg.rewards["joint_deviation_arms"].params["asset_cfg"].joint_names,
      (".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"),
    )
    self.assertEqual(
      cfg.rewards["joint_deviation_waists"].params["asset_cfg"].joint_names,
      ("waist.*",),
    )
    self.assertEqual(cfg.rewards["joint_deviation_waists"].weight, -1.0)
    self.assertEqual(
      cfg.rewards["joint_deviation_legs"].params["asset_cfg"].joint_names,
      (".*_hip_roll_joint", ".*_hip_yaw_joint"),
    )
    self.assertEqual(cfg.rewards["joint_deviation_legs"].weight, -1.0)
    self.assertEqual(cfg.rewards["body_orientation_l2"].weight, -4.0)
    self.assertEqual(cfg.rewards["body_ang_vel"].weight, -0.15)
    for term_name in (
      "joint_deviation_arms",
      "joint_deviation_waists",
      "joint_deviation_legs",
    ):
      joint_names = cfg.rewards[term_name].params["asset_cfg"].joint_names
      self.assertFalse(
        any("hip_pitch" in name or "knee" in name for name in joint_names)
      )
    self.assertEqual(cfg.rewards["base_height"].weight, -10.0)
    self.assertEqual(
      cfg.rewards["base_height"].params["target_height"],
      EFFORT_STANDING_ROOT_HEIGHT,
    )
    self.assertEqual(cfg.rewards["base_height"].params["deadzone"], 0.02)
    self.assertIn("effort_action_rate_l2", cfg.rewards)
    self.assertEqual(
      cfg.actions["joint_effort"].offset,
      0.0,
    )
    self.assertEqual(cfg.actions["joint_effort"].scale, EFFORT_ACTION_SCALE)
    self.assertIn("nan_detection", cfg.terminations)
    self.assertEqual(cfg.rewards["foot_gait"].weight, 5.0)
    self.assertEqual(cfg.rewards["foot_gait"].params["period"], 0.6)
    self.assertEqual(cfg.rewards["feet_air_time"].weight, 2.0)
    self.assertEqual(cfg.rewards["feet_air_time"].params["threshold"], 0.30)
    self.assertEqual(
      cfg.rewards["feet_air_time"].params["sensor_name"],
      "feet_ground_contact",
    )
    self.assertEqual(cfg.rewards["feet_air_time"].params["command_name"], "twist")

  def test_other_effort_variants_configure_new_rewards(self) -> None:
    g1_23dof_cfg = load_env_cfg("Unitree-G1-23Dof-Effort-Flat")
    go2_cfg = load_env_cfg("Unitree-Go2-Effort-Flat")

    for cfg in (g1_23dof_cfg, go2_cfg):
      self.assertNotIn("pose", cfg.rewards)
      self.assertIs(cfg.rewards["stand_still"].func, mdp.stand_still)
      self.assertIn("joint_deviation_arms", cfg.rewards)
      self.assertIn("joint_deviation_waists", cfg.rewards)
      self.assertIn("joint_deviation_legs", cfg.rewards)
      self.assertEqual(cfg.rewards["stand_still"].weight, -1.0)
      self.assertEqual(cfg.rewards["body_orientation_l2"].weight, -4.0)
      self.assertEqual(cfg.rewards["body_ang_vel"].weight, -0.15)
      self.assertEqual(cfg.rewards["base_height"].weight, -10.0)
      self.assertEqual(cfg.rewards["base_height"].params["deadzone"], 0.02)
      self.assertNotIn("action_rate_l2", cfg.rewards)
      self.assertIn("effort_action_rate_l2", cfg.rewards)

    self.assertEqual(g1_23dof_cfg.rewards["joint_deviation_arms"].weight, -0.3)
    self.assertEqual(
      g1_23dof_cfg.rewards["joint_deviation_legs"].params["asset_cfg"].joint_names,
      (".*_hip_roll_joint", ".*_hip_yaw_joint"),
    )
    self.assertEqual(
      g1_23dof_cfg.rewards["base_height"].params["target_height"],
      0.79,
    )

    self.assertEqual(go2_cfg.rewards["joint_deviation_arms"].weight, 0.0)
    self.assertEqual(go2_cfg.rewards["joint_deviation_waists"].weight, 0.0)
    self.assertEqual(
      go2_cfg.rewards["joint_deviation_legs"].params["asset_cfg"].joint_names,
      (".*_hip_joint",),
    )
    self.assertEqual(
      go2_cfg.rewards["base_height"].params["target_height"],
      0.32,
    )
    self.assertEqual(
      go2_cfg.rewards["foot_gait"].params["offset"],
      [0.0, 0.5, 0.5, 0.0],
    )
    self.assertIn("illegal_contact", go2_cfg.terminations)

  def test_effort_action_rate_uses_processed_torque_fraction(self) -> None:
    robot = Entity(get_g1_effort_robot_cfg())
    env = Mock(spec=ManagerBasedRlEnv)
    env.num_envs = 2
    env.device = "cpu"
    env.scene = {"robot": robot}
    action = g1_effort_action_cfg().build(env)
    action_manager = _FakeActionManager(action)
    env.action_manager = action_manager

    action.process_actions(torch.zeros(2, action.action_dim))
    torch.testing.assert_close(mdp.effort_action_rate_l2(env), torch.zeros(2))

    current_raw = torch.zeros(2, action.action_dim)
    current_raw[:, 0] = 1.0
    action_manager.prev_action = torch.zeros_like(current_raw)
    action.process_actions(current_raw)
    torch.testing.assert_close(
      mdp.effort_action_rate_l2(env),
      torch.full((2,), EFFORT_ACTION_SCALE_FRACTION**2),
      atol=1.0e-6,
      rtol=0.0,
    )

    ankle_index = action.target_names.index("left_ankle_pitch_joint")
    current_raw.zero_()
    current_raw[:, ankle_index] = 100.0
    action_manager.prev_action = torch.zeros_like(current_raw)
    action.process_actions(current_raw)
    torch.testing.assert_close(
      mdp.effort_action_rate_l2(env), torch.ones(2), atol=1.0e-6, rtol=0.0
    )

  def test_effort_rewards_use_torque_and_pose_data(self) -> None:
    robot = Entity(get_g1_effort_robot_cfg())
    action_env = Mock(spec=ManagerBasedRlEnv)
    action_env.num_envs = 2
    action_env.device = "cpu"
    action_env.scene = {"robot": robot}
    action = g1_effort_action_cfg().build(action_env)
    action_manager = _FakeActionManager(action)
    action_env.action_manager = action_manager

    raw_action = torch.zeros(2, action.action_dim)
    hip_pitch_index = action.target_names.index("left_hip_pitch_joint")
    raw_action[:, hip_pitch_index] = 1.0
    action.process_actions(raw_action)

    asset = SimpleNamespace(
      data=SimpleNamespace(
        joint_vel=torch.zeros(2, action.action_dim),
        joint_pos=torch.zeros(2, action.action_dim),
        default_joint_pos=torch.zeros(2, action.action_dim),
        root_link_pos_w=torch.tensor(
          [[0.0, 0.0, 0.789733], [0.0, 0.0, 0.789733]]
        ),
      )
    )
    action_env.scene = {"robot": asset}

    torch.testing.assert_close(
      mdp.energy(action_env), torch.zeros(2)
    )
    asset.data.joint_vel[:, hip_pitch_index] = 1.0
    hip_peak = EFFORT_ACTION_LIMIT[r".*_hip_pitch_joint"]
    torch.testing.assert_close(
      mdp.energy(action_env),
      torch.full((2,), EFFORT_ACTION_SCALE_FRACTION * hip_peak),
    )

    joint_cfg = mdp.SceneEntityCfg("robot", joint_ids=[0, 1])
    torch.testing.assert_close(
      mdp.joint_deviation_l1(action_env, asset_cfg=joint_cfg), torch.zeros(2)
    )
    asset.data.joint_pos[:, 0] = 0.25
    torch.testing.assert_close(
      mdp.joint_deviation_l1(action_env, asset_cfg=joint_cfg),
      torch.full((2,), 0.25),
    )

    torch.testing.assert_close(
      mdp.base_height_l2(
        action_env, target_height=0.789733
      ),
      torch.zeros(2),
    )
    torch.testing.assert_close(
      mdp.base_height_l2(action_env, target_height=0.7),
      torch.full((2,), (0.789733 - 0.7) ** 2),
    )
    asset.data.root_link_pos_w[:, 2] = 0.80
    torch.testing.assert_close(
      mdp.base_height_l2(
        action_env, target_height=0.78, deadzone=0.02
      ),
      torch.zeros(2),
    )
    asset.data.root_link_pos_w[:, 2] = 0.82
    torch.testing.assert_close(
      mdp.base_height_l2(
        action_env, target_height=0.78, deadzone=0.02
      ),
      torch.full((2,), 0.02**2),
    )

  def test_mha_variants_only_add_observation_history(self) -> None:
    pairs = (
      (unitree_g1_rough_env_cfg(), unitree_g1_rough_mha_env_cfg()),
      (unitree_g1_flat_env_cfg(), unitree_g1_flat_mha_env_cfg()),
      (unitree_g1_flat_partial_env_cfg(), unitree_g1_flat_partial_mha_env_cfg()),
      (
        unitree_g1_flat_partial_phase1_env_cfg(),
        unitree_g1_flat_partial_phase1_mha_env_cfg(),
      ),
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
      self.assertEqual(base_cfg.observations["actor"].history_length, 5)
      self.assertTrue(base_cfg.observations["actor"].flatten_history_dim)
      self.assertEqual(base_cfg.observations["critic"].history_length, 5)
      self.assertTrue(base_cfg.observations["critic"].flatten_history_dim)
      self.assertEqual(mha_cfg.observations["actor"].history_length, 5)
      self.assertFalse(mha_cfg.observations["actor"].flatten_history_dim)
      self.assertEqual(mha_cfg.observations["critic"].history_length, 5)
      self.assertTrue(mha_cfg.observations["critic"].flatten_history_dim)

  def test_flat_partial_strips_actor_and_keeps_privileged_critic(self) -> None:
    cfg = load_env_cfg("Unitree-G1-Effort-Flat-Partial")
    play_cfg = unitree_g1_flat_partial_env_cfg(play=True)

    self.assertEqual(
      set(cfg.observations["actor"].terms),
      {"joint_pos", "command", "actions"},
    )
    self.assertEqual(
      set(play_cfg.observations["actor"].terms),
      {"joint_pos", "command", "actions"},
    )
    self.assertFalse(play_cfg.observations["actor"].enable_corruption)

    critic_terms = cfg.observations["critic"].terms
    for term_name in (
      "base_ang_vel",
      "projected_gravity",
      "joint_vel",
      "phase",
      "foot_contact",
      "foot_contact_forces",
      "base_lin_vel",
    ):
      self.assertIn(term_name, critic_terms)
    self.assertNotIn("height_scan", critic_terms)
    self.assertEqual(cfg.observations["actor"].history_length, 5)
    self.assertTrue(cfg.observations["actor"].flatten_history_dim)

    mha_cfg = load_env_cfg("Unitree-G1-Effort-Flat-Partial-MHA")
    self.assertEqual(
      set(mha_cfg.observations["actor"].terms),
      {"joint_pos", "command", "actions"},
    )
    self.assertFalse(mha_cfg.observations["actor"].flatten_history_dim)
    self.assertTrue(
      load_rl_cfg("Unitree-G1-Effort-Flat-Partial-MHA").actor.class_name.endswith(
        ":ResidualMhaModel"
      )
    )

  def test_flat_partial_phase1_keeps_imu_drops_ankles_and_command_curriculum(
    self,
  ) -> None:
    cfg = load_env_cfg("Unitree-G1-Effort-Flat-Partial-Phase1")
    play_cfg = unitree_g1_flat_partial_phase1_env_cfg(play=True)
    mha_cfg = load_env_cfg("Unitree-G1-Effort-Flat-Partial-Phase1-MHA")

    self.assertEqual(
      set(cfg.observations["actor"].terms),
      {
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "command",
        "actions",
      },
    )
    self.assertEqual(
      set(play_cfg.observations["actor"].terms),
      {
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "command",
        "actions",
      },
    )
    self.assertNotIn("joint_vel", cfg.observations["actor"].terms)
    self.assertNotIn("phase", cfg.observations["actor"].terms)
    self.assertFalse(play_cfg.observations["actor"].enable_corruption)
    self.assertNotIn("asset_cfg", cfg.observations["critic"].terms["joint_pos"].params)
    self.assertNotIn("asset_cfg", cfg.observations["actor"].terms["actions"].params)
    flat_cfg = unitree_g1_flat_env_cfg()
    self.assertEqual(cfg.curriculum, flat_cfg.curriculum)
    self.assertIn("command_vel", cfg.curriculum)
    self.assertEqual(play_cfg.curriculum, {})
    self.assertEqual(mha_cfg.curriculum, cfg.curriculum)
    stages = cfg.curriculum["command_vel"].params["velocity_stages"]
    self.assertEqual(stages[1]["step"], 20000 * 24)
    self.assertEqual(stages[2]["step"], 30000 * 24)

    robot = Entity(get_g1_effort_robot_cfg())
    keep_names = cfg.observations["actor"].terms["joint_pos"].params[
      "asset_cfg"
    ].joint_names
    _, observed_joints = robot.find_joints(keep_names)
    dropped = [name for name in robot.joint_names if name not in observed_joints]
    self.assertEqual(dropped, list(_G1_ANKLE_JOINTS))
    self.assertEqual(len(observed_joints), 25)
    self.assertEqual(len(robot.joint_names), 29)

    action_cfg = cfg.actions["joint_effort"]
    self.assertEqual(action_cfg.actuator_names, (".*",))

    twist_cmd = cfg.commands["twist"]
    self.assertEqual(twist_cmd.ranges.lin_vel_x, (-2.0, 2.0))
    self.assertEqual(twist_cmd.ranges.lin_vel_y, (-1.0, 1.0))
    self.assertEqual(twist_cmd.ranges.ang_vel_z, (-1.0, 1.0))

    self.assertEqual(
      set(mha_cfg.observations["actor"].terms),
      {
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "command",
        "actions",
      },
    )
    self.assertFalse(mha_cfg.observations["actor"].flatten_history_dim)
    self.assertTrue(
      load_rl_cfg("Unitree-G1-Effort-Flat-Partial-Phase1").actor.class_name.endswith(
        ":ResidualMlpModel"
      )
    )
    self.assertTrue(
      load_rl_cfg("Unitree-G1-Effort-Flat-Partial-Phase1-MHA").actor.class_name.endswith(
        ":ResidualMhaModel"
      )
    )

  def test_all_effort_variants_use_joint_effort_actions(self) -> None:
    for task_id in ALL_EFFORT_TASK_IDS:
      cfg = load_env_cfg(task_id)
      self.assertEqual(list(cfg.actions), ["joint_effort"])
      action_cfg = cfg.actions["joint_effort"]
      self.assertIsInstance(action_cfg, JointEffortActionCfg)
      self.assertEqual(action_cfg.offset, 0.0)
      self.assertIsInstance(action_cfg.scale, dict)
      self.assertIsInstance(action_cfg.clip, dict)
      self.assertEqual(set(action_cfg.scale), set(action_cfg.clip))
      for expr, scale in action_cfg.scale.items():
        lo, hi = action_cfg.clip[expr]
        self.assertEqual(lo, -hi)
        self.assertAlmostEqual(scale, EFFORT_ACTION_SCALE_FRACTION * hi)
      self.assertIn("nan_detection", cfg.terminations)

  def test_ppo_configs_use_effort_initialization_and_mha_model(self) -> None:
    ppo_cfg = unitree_g1_ppo_runner_cfg()
    mha_cfg = unitree_g1_ppo_mha_runner_cfg()
    expected_distribution = {
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
      "std_range": (1e-3, 2.0),
    }

    self.assertTrue(ppo_cfg.actor.class_name.endswith(":ResidualMlpModel"))
    self.assertTrue(mha_cfg.actor.class_name.endswith(":ResidualMhaModel"))
    self.assertIs(resolve_callable(ppo_cfg.actor.class_name), ResidualMlpModel)
    self.assertIs(resolve_callable(mha_cfg.actor.class_name), ResidualMhaModel)
    self.assertEqual(ppo_cfg.actor.distribution_cfg, expected_distribution)
    self.assertEqual(mha_cfg.actor.distribution_cfg, expected_distribution)
    self.assertTrue(ppo_cfg.actor.obs_normalization)
    self.assertTrue(mha_cfg.actor.obs_normalization)
    self.assertTrue(ppo_cfg.critic.obs_normalization)
    self.assertTrue(mha_cfg.critic.obs_normalization)
    self.assertIsNone(ppo_cfg.clip_actions)
    self.assertIsNone(mha_cfg.clip_actions)
    self.assertEqual(ppo_cfg.algorithm.entropy_coef, 0.001)
    self.assertEqual(mha_cfg.algorithm.entropy_coef, 0.001)
    self.assertEqual(ppo_cfg.max_iterations, 50_000)
    self.assertEqual(mha_cfg.max_iterations, 10_000)

    other_runners = (
      unitree_g1_23dof_ppo_runner_cfg(),
      unitree_g1_23dof_ppo_mha_runner_cfg(),
      unitree_go2_ppo_runner_cfg(),
      unitree_go2_ppo_mha_runner_cfg(),
    )
    for other_cfg in other_runners:
      self.assertEqual(other_cfg.actor.distribution_cfg, expected_distribution)
      self.assertTrue(other_cfg.actor.obs_normalization)
      self.assertTrue(other_cfg.critic.obs_normalization)
      self.assertIsNone(other_cfg.clip_actions)
      self.assertEqual(other_cfg.algorithm.entropy_coef, 0.001)

    serialized_mha_cfg = asdict(mha_cfg)
    self.assertEqual(serialized_mha_cfg["actor"]["history_length"], 5)
    self.assertEqual(serialized_mha_cfg["actor"]["num_heads"], 4)

    for task_id in ALL_EFFORT_TASK_IDS:
      registered_cfg = load_rl_cfg(task_id)
      self.assertTrue(
        registered_cfg.actor.obs_normalization,
        task_id,
      )
      self.assertTrue(
        registered_cfg.critic.obs_normalization,
        task_id,
      )

  def test_residual_models_center_actor_output_and_export_mha(self) -> None:
    batch_size = 8
    output_dim = 29
    distribution_cfg = {
      "class_name": "GaussianDistribution",
      "init_std": 1,
      "std_type": "scalar",
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

  def test_cross_attention_keeps_a_token_for_every_controlled_joint(self) -> None:
    model, obs = _toy_cross_attn()
    weights = model.attention_weights(obs)

    self.assertEqual(model.joint_names, ("knee", "hip", "ankle"))
    self.assertEqual(model.token_names, ("knee", "hip", "ankle"))
    self.assertEqual(tuple(weights.shape), (2, 2, 3, 3))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 2, 3))
    torch.testing.assert_close(
      model.cross_attention.missing, torch.tensor([0.0, 0.0, 1.0])
    )
    # Current-frame embedding plus one flattened output slot per joint.
    self.assertEqual(tuple(model.get_latent(obs).shape), (2, 8 + 3 * 2))

  def test_joints_without_position_read_zero_and_keep_their_torque(self) -> None:
    model, obs = _toy_cross_attn()
    positions, torques = model.cross_attention.slicer(obs["actor"])

    torch.testing.assert_close(positions[..., 0], obs["actor"][..., 3])
    torch.testing.assert_close(positions[..., 1], obs["actor"][..., 4])
    torch.testing.assert_close(positions[..., 2], torch.zeros(2, 5))
    torch.testing.assert_close(torques, obs["actor"][..., 5:8])

  def test_positions_follow_the_action_order_of_the_joints(self) -> None:
    obs = TensorDict({"actor": torch.randn(2, 5, 8)}, batch_size=[2])
    model = ResidualCrossAttnModel(
      obs,
      {"actor": ["actor"]},
      "actor",
      4,
      hidden_dims=(16, 16),
      obs_normalization=False,
      num_heads=2,
      encoder_hidden_dim=8,
      joint_embedding_dim=4,
      joint_output_dim=2,
      term_dims=(("command", 3), ("joint_pos", 2), ("actions", 3)),
      joint_pos_names=("hip", "knee"),
      action_names=("knee", "hip", "ankle"),
    )
    positions, _ = model.cross_attention.slicer(obs["actor"])

    torch.testing.assert_close(positions[..., 0], obs["actor"][..., 4])
    torch.testing.assert_close(positions[..., 1], obs["actor"][..., 3])
    torch.testing.assert_close(positions[..., 2], torch.zeros(2, 5))

  def test_past_torque_of_every_joint_reaches_the_actor(self) -> None:
    model, obs = _toy_cross_attn()
    actions = model(obs)
    weights = model.attention_weights(obs)

    # Columns 5-7 are the knee, hip, and ankle torques; the ankle has no position.
    for column in (5, 6, 7):
      changed = _clone_actor_obs(obs)
      changed["actor"][:, :-1, column] += 1.5
      self.assertFalse(torch.allclose(model(changed), actions))
      self.assertFalse(torch.allclose(model.attention_weights(changed), weights))

  def test_command_reaches_the_actor_only_through_the_current_frame(self) -> None:
    model, obs = _toy_cross_attn()
    actions = model(obs)

    past_command = _clone_actor_obs(obs)
    past_command["actor"][:, :-1, 0:3] += 1.5
    torch.testing.assert_close(model(past_command), actions)

    current_command = _clone_actor_obs(obs)
    current_command["actor"][:, -1, 0:3] += 1.5
    self.assertFalse(torch.allclose(model(current_command), actions))

  def test_cross_attention_keeps_past_joint_samples_in_time_order(self) -> None:
    model, obs = _toy_cross_attn()
    obs["actor"].zero_()
    for frame, value in enumerate((1.0, 2.0, 3.0, 4.0)):
      obs["actor"][:, frame, 3] = value
    ordered = model.attention_weights(obs)

    reversed_history = _clone_actor_obs(obs)
    reversed_history["actor"][:, 0, 3] = 4.0
    reversed_history["actor"][:, 3, 3] = 1.0
    self.assertFalse(
      torch.allclose(model.attention_weights(reversed_history), ordered)
    )

  def test_identical_joint_histories_receive_different_attention(self) -> None:
    model, obs = _toy_cross_attn()
    obs["actor"].zero_()
    for frame, value in enumerate((1.0, 2.0, 3.0, 4.0)):
      obs["actor"][:, frame, 3:8] = value

    weights = model.attention_weights(obs)
    self.assertFalse(torch.allclose(weights[..., 0], weights[..., 1]))

  def test_current_state_of_a_joint_only_moves_its_own_query(self) -> None:
    model, obs = _toy_cross_attn()
    weights = model.attention_weights(obs)

    # Keys come from past frames, so the current frame can only move queries.
    # Columns: 3 knee position, 4 hip position, 5-7 knee, hip, ankle torque.
    for column, joint in ((3, 0), (5, 0), (4, 1), (6, 1), (7, 2)):
      changed = _clone_actor_obs(obs)
      changed["actor"][:, -1, column] += 1.5
      moved = model.attention_weights(changed)
      others = [index for index in range(3) if index != joint]
      self.assertFalse(torch.allclose(moved[:, :, joint], weights[:, :, joint]))
      torch.testing.assert_close(moved[:, :, others], weights[:, :, others])

  def test_imu_history_is_one_token_and_current_imu_reaches_every_query(
    self,
  ) -> None:
    torch.manual_seed(0)
    batch_size = 2
    obs = TensorDict(
      {"actor": torch.randn(batch_size, 5, 14)}, batch_size=[batch_size]
    )
    model = ResidualCrossAttnModel(
      obs,
      {"actor": ["actor"]},
      "actor",
      4,
      hidden_dims=(16, 16),
      obs_normalization=False,
      history_length=5,
      num_heads=2,
      encoder_hidden_dim=8,
      joint_embedding_dim=4,
      joint_output_dim=2,
      term_dims=(
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("command", 3),
        ("joint_pos", 2),
        ("actions", 3),
      ),
      joint_pos_names=("knee", "hip"),
      action_names=("knee", "hip", "ankle"),
    )
    model.eval()
    self.assertEqual(model.token_names, ("knee", "hip", "ankle", "imu"))
    attention = model.cross_attention
    self.assertEqual(attention.history_encoder[0].in_features, 2 * 4 + 4 + 1)
    self.assertEqual(attention.imu_encoder[0].in_features, 4 * 6)
    self.assertEqual(attention.context_encoder[0].in_features, 3 + 6)

    weights = model.attention_weights(obs)
    self.assertEqual(tuple(weights.shape), (batch_size, 2, 3, 4))
    past_imu = _clone_actor_obs(obs)
    past_imu["actor"][:, :-1, 0:6] += 1.5
    self.assertFalse(torch.allclose(model.attention_weights(past_imu), weights))
    current_imu = _clone_actor_obs(obs)
    current_imu["actor"][:, -1, 0:6] += 1.5
    self.assertFalse(torch.allclose(model.attention_weights(current_imu), weights))

  def test_cross_attention_export_matches_actions_and_centers_output(self) -> None:
    model, obs = _toy_cross_attn()
    output_layer = _last_linear(model.mlp)
    torch.testing.assert_close(
      output_layer.weight.norm(dim=1),
      torch.full((4,), 0.01),
      atol=1.0e-6,
      rtol=1.0e-5,
    )
    torch.testing.assert_close(model.as_onnx()(obs["actor"]), model(obs))

  def test_cross_attn_runner_defaults_to_fp32_and_can_opt_into_bf16(self) -> None:
    for task_id in (
      "Unitree-G1-Effort-Flat-Partial-XAttn",
      "Unitree-G1-Effort-Flat-Partial-NoAnkle-XAttn",
      "Unitree-G1-Effort-Flat-Partial-Phase1-XAttn",
      "Unitree-G1-Effort-Flat-Partial-Phase1-XAttn-NoCurriculum-FP32",
    ):
      self.assertEqual(load_rl_cfg(task_id).algorithm.class_name, "PPO", task_id)
    self.assertEqual(
      load_rl_cfg("Unitree-G1-Effort-Flat-Partial-MHA").algorithm.class_name,
      "PPO",
    )

    bf16 = unitree_g1_ppo_cross_attn_runner_cfg(
      load_env_cfg("Unitree-G1-Effort-Flat-Partial-XAttn"), use_bf16_autocast=True
    )
    self.assertEqual(
      bf16.algorithm.class_name,
      "src.tasks.effort.rl.bf16_autocast:Bf16AutocastPPO",
    )
    self.assertIs(resolve_callable(bf16.algorithm.class_name), Bf16AutocastPPO)

  def test_bf16_autocast_wrapper_keeps_fp32_outputs_and_matches_on_cpu(
    self,
  ) -> None:
    model, obs = _toy_cross_attn()
    wrapped = Bf16AutocastModel(model)
    for param in wrapped.parameters():
      self.assertEqual(param.dtype, torch.float32)

    baseline = model(obs)
    wrapped_out = wrapped(obs)
    self.assertEqual(wrapped_out.dtype, torch.float32)
    torch.testing.assert_close(wrapped_out, baseline)

    stochastic_model = ResidualCrossAttnModel(
      obs,
      {"actor": ["actor"]},
      "actor",
      4,
      hidden_dims=(16, 16),
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      history_length=5,
      num_heads=2,
      encoder_hidden_dim=8,
      joint_embedding_dim=4,
      term_dims=(("command", 3), ("joint_pos", 2), ("actions", 3)),
      joint_pos_names=("knee", "hip"),
      action_names=("knee", "hip", "ankle"),
    )
    stochastic_wrapped = Bf16AutocastModel(stochastic_model)
    stochastic = stochastic_wrapped(obs, stochastic_output=True)
    self.assertEqual(stochastic.dtype, torch.float32)
    log_prob = stochastic_wrapped.get_output_log_prob(stochastic)
    self.assertEqual(log_prob.dtype, torch.float32)

    critic = ResidualMlpModel(
      TensorDict({"critic": torch.randn(2, 8)}, batch_size=[2]),
      {"critic": ["critic"]},
      "critic",
      1,
      hidden_dims=(16, 16),
      obs_normalization=False,
    )
    critic_obs = TensorDict({"critic": torch.randn(2, 8)}, batch_size=[2])
    wrapped_critic = Bf16AutocastModel(critic)
    values = wrapped_critic(critic_obs)
    self.assertEqual(values.dtype, torch.float32)
    torch.testing.assert_close(values, critic(critic_obs))

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
      cuda_model = ResidualCrossAttnModel(
        obs,
        {"actor": ["actor"]},
        "actor",
        4,
        hidden_dims=(16, 16),
        obs_normalization=False,
        history_length=5,
        num_heads=2,
        encoder_hidden_dim=8,
        joint_embedding_dim=4,
        term_dims=(("command", 3), ("joint_pos", 2), ("actions", 3)),
        joint_pos_names=("knee", "hip"),
        action_names=("knee", "hip", "ankle"),
      ).to("cuda")
      cuda_obs = TensorDict(
        {"actor": obs["actor"].to("cuda")}, batch_size=obs.batch_size
      )
      cuda_wrapped = Bf16AutocastModel(cuda_model)
      for param in cuda_wrapped.parameters():
        self.assertEqual(param.dtype, torch.float32)
      cuda_actions = cuda_wrapped(cuda_obs)
      self.assertEqual(cuda_actions.dtype, torch.float32)
      self.assertEqual(cuda_actions.device.type, "cuda")

  def test_cross_attn_tasks_follow_parent_observations_and_joint_names(
    self,
  ) -> None:
    robot = Entity(get_g1_effort_robot_cfg())
    all_joints = tuple(robot.joint_names)
    ankle_free = tuple(name for name in all_joints if name not in _G1_ANKLE_JOINTS)
    cases = (
      (
        "Unitree-G1-Effort-Flat-Partial-XAttn",
        "Unitree-G1-Effort-Flat-Partial",
        ("command", "joint_pos", "actions"),
        (("command", 3), ("joint_pos", 29), ("actions", 29)),
        all_joints,
      ),
      (
        "Unitree-G1-Effort-Flat-Partial-NoAnkle-XAttn",
        "Unitree-G1-Effort-Flat-Partial",
        ("command", "joint_pos", "actions"),
        (("command", 3), ("joint_pos", 25), ("actions", 29)),
        ankle_free,
      ),
      (
        "Unitree-G1-Effort-Flat-Partial-Phase1-XAttn",
        "Unitree-G1-Effort-Flat-Partial-Phase1",
        (
          "base_ang_vel",
          "projected_gravity",
          "command",
          "joint_pos",
          "actions",
        ),
        (
          ("base_ang_vel", 3),
          ("projected_gravity", 3),
          ("command", 3),
          ("joint_pos", 25),
          ("actions", 29),
        ),
        ankle_free,
      ),
    )

    for task_id, parent_id, term_names, term_dims, joint_names in cases:
      cfg = load_env_cfg(task_id)
      parent = load_env_cfg(parent_id)
      play_cfg = load_env_cfg(task_id, play=True)
      rl_cfg = load_rl_cfg(task_id)

      self.assertEqual(tuple(cfg.observations["actor"].terms), term_names)
      self.assertEqual(cfg.rewards, parent.rewards)
      self.assertEqual(cfg.commands, parent.commands)
      self.assertEqual(cfg.curriculum, parent.curriculum)
      self.assertEqual(
        cfg.observations["critic"].terms, parent.observations["critic"].terms
      )
      self.assertEqual(cfg.observations["actor"].history_length, 5)
      self.assertFalse(cfg.observations["actor"].flatten_history_dim)
      self.assertTrue(cfg.observations["critic"].flatten_history_dim)
      self.assertFalse(play_cfg.observations["actor"].enable_corruption)
      self.assertFalse(play_cfg.observations["actor"].flatten_history_dim)

      asset_cfg = cfg.observations["actor"].terms["joint_pos"].params.get(
        "asset_cfg"
      )
      if asset_cfg is None:
        resolved = all_joints
      else:
        _, resolved_names = robot.find_joints(asset_cfg.joint_names)
        resolved = tuple(resolved_names)
      self.assertEqual(resolved, joint_names)
      self.assertNotIn(
        "asset_cfg", cfg.observations["critic"].terms["joint_pos"].params
      )

      self.assertTrue(rl_cfg.actor.class_name.endswith(":ResidualCrossAttnModel"))
      self.assertIs(resolve_callable(rl_cfg.actor.class_name), ResidualCrossAttnModel)
      self.assertEqual(rl_cfg.experiment_name, "g1_effort_xattn")
      self.assertEqual(rl_cfg.max_iterations, 10_000)
      self.assertEqual(rl_cfg.actor.joint_embedding_dim, 16)
      self.assertEqual(rl_cfg.actor.encoder_hidden_dim, 128)
      self.assertEqual(rl_cfg.actor.joint_output_dim, 32)
      self.assertEqual(
        tuple(tuple(item) for item in rl_cfg.actor.term_dims), term_dims
      )
      self.assertEqual(tuple(rl_cfg.actor.joint_pos_names), joint_names)
      self.assertEqual(tuple(rl_cfg.actor.action_names), all_joints)

    self.assertTrue(
      load_rl_cfg("Unitree-G1-Effort-Flat-Partial-MHA").actor.class_name.endswith(
        ":ResidualMhaModel"
      )
    )
    self.assertTrue(
      load_rl_cfg(
        "Unitree-G1-Effort-Flat-Partial-Phase1-MHA"
      ).actor.class_name.endswith(":ResidualMhaModel")
    )
    self.assertTrue(
      load_rl_cfg("Unitree-G1-Effort-Flat-Partial-Phase1").actor.class_name.endswith(
        ":ResidualMlpModel"
      )
    )

  def test_registered_cross_attn_actor_emits_effort_and_named_weights(self) -> None:
    all_joints = tuple(Entity(get_g1_effort_robot_cfg()).joint_names)
    expected = {
      "Unitree-G1-Effort-Flat-Partial-XAttn": (29, ()),
      "Unitree-G1-Effort-Flat-Partial-NoAnkle-XAttn": (29, _G1_ANKLE_JOINTS),
      "Unitree-G1-Effort-Flat-Partial-Phase1-XAttn": (30, _G1_ANKLE_JOINTS),
    }
    for task_id, (num_tokens, missing_joints) in expected.items():
      rl_cfg = load_rl_cfg(task_id)
      width = sum(int(dim) for _, dim in rl_cfg.actor.term_dims)
      batch_size = 2
      obs = TensorDict(
        {"actor": torch.randn(batch_size, 5, width)}, batch_size=[batch_size]
      )
      model = resolve_callable(rl_cfg.actor.class_name)(
        obs,
        {"actor": ["actor"]},
        "actor",
        29,
        hidden_dims=rl_cfg.actor.hidden_dims,
        activation=rl_cfg.actor.activation,
        obs_normalization=True,
        distribution_cfg=dict(rl_cfg.actor.distribution_cfg),
        history_length=rl_cfg.actor.history_length,
        num_heads=rl_cfg.actor.num_heads,
        encoder_hidden_dim=rl_cfg.actor.encoder_hidden_dim,
        joint_embedding_dim=rl_cfg.actor.joint_embedding_dim,
        joint_output_dim=rl_cfg.actor.joint_output_dim,
        term_dims=rl_cfg.actor.term_dims,
        joint_pos_names=rl_cfg.actor.joint_pos_names,
        action_names=rl_cfg.actor.action_names,
      )
      model.update_normalization(obs)
      actions = model(obs)
      weights = model.attention_weights(obs)
      self.assertEqual(tuple(actions.shape), (batch_size, 29))
      self.assertEqual(tuple(weights.shape), (batch_size, 4, 29, num_tokens))
      self.assertEqual(model.joint_names, all_joints)
      self.assertEqual(model.token_names[:29], all_joints)
      self.assertEqual(model.token_names[29:], ("imu",) if num_tokens > 29 else ())
      flagged = tuple(
        name
        for name, flag in zip(model.joint_names, model.cross_attention.missing.tolist())
        if flag
      )
      self.assertEqual(flagged, missing_joints)
      torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(batch_size, 4, 29), atol=1e-5, rtol=1e-5
      )

  def test_joint_attention_plot_averages_batch_and_heads_into_a_heatmap(self) -> None:
    weights = torch.tensor(
      [[[0.2, 0.8], [0.6, 0.4]], [[0.0, 1.0], [0.2, 0.8]]]
    )  # [heads, joints, tokens]
    expected = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
    torch.testing.assert_close(mean_joint_attention(weights), expected)
    torch.testing.assert_close(
      mean_joint_attention(torch.stack((weights, weights))), expected
    )

    names = ("left_knee_joint", "right_knee_joint")
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "attention.png"
      plot_joint_attention(weights, names, names, path)
      self.assertGreater(path.stat().st_size, 0)
      with self.assertRaises(ValueError):
        plot_joint_attention(weights, names[:1], names, path)


def _clone_actor_obs(obs: TensorDict) -> TensorDict:
  return TensorDict(
    {"actor": obs["actor"].clone()}, batch_size=obs.batch_size
  )


def _toy_cross_attn() -> tuple[ResidualCrossAttnModel, TensorDict]:
  """Three controlled joints, two of them with position encoders.

  Flat frame layout is command (3), joint_pos (knee, hip), actions
  (knee, hip, ankle). History length is 5, so the past axis is the first 4.
  """
  torch.manual_seed(0)
  batch_size = 2
  obs = TensorDict(
    {"actor": torch.randn(batch_size, 5, 8)}, batch_size=[batch_size]
  )
  model = ResidualCrossAttnModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    4,
    hidden_dims=(16, 16),
    obs_normalization=False,
    history_length=5,
    num_heads=2,
    encoder_hidden_dim=8,
    joint_embedding_dim=4,
    joint_output_dim=2,
    term_dims=(("command", 3), ("joint_pos", 2), ("actions", 3)),
    joint_pos_names=("knee", "hip"),
    action_names=("knee", "hip", "ankle"),
  )
  model.eval()
  return model, obs



if __name__ == "__main__":
  unittest.main()
