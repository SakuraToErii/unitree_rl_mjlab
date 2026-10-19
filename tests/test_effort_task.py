"""Contracts for residual-effort velocity tasks."""

from __future__ import annotations

import re
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
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
from src.tasks.effort.rl.joint_attention import (
  mean_joint_attention,
  plot_group_time_attention,
  plot_joint_attention,
)
from src.tasks.effort.rl.models import (
  ResidualGroupTimeModel,
  ResidualMhaModel,
  ResidualMlpModel,
)
from src.tasks.effort.rl.runner import (
  EffortOnPolicyRunner,
  append_training_metrics,
  get_effort_metadata,
  is_metrics_iteration,
  metrics_log_hook,
)
from src.tasks.effort.zero_pd import EFFORT_ACTION_SCALE_FRACTION

G1_EFFORT_TASK_IDS = (
  "Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-MLP-CmdCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-MLP-CmdCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-MLP-NoCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H1-MLP-NoCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-TimeMHA-NoCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-GroupTimeFusion-CmdCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-GroupTimeFusion-NoCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-H5-GroupTimeFusion-CmdCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm",
  "Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm",
)

_G1_ANKLE_JOINTS = (
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)

ALL_EFFORT_TASK_IDS = (
  *G1_EFFORT_TASK_IDS,
  "Unitree-G1-23Dof-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm",
  "Unitree-G1-23Dof-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm",
  "Unitree-G1-23Dof-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm",
  "Unitree-G1-23Dof-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm",
  "Unitree-Go2-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm",
  "Unitree-Go2-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm",
  "Unitree-Go2-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm",
  "Unitree-Go2-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm",
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
    experiment_names = [
      load_rl_cfg(task_id).experiment_name for task_id in ALL_EFFORT_TASK_IDS
    ]
    self.assertEqual(len(set(experiment_names)), len(ALL_EFFORT_TASK_IDS))
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
    env = ManagerBasedRlEnv(load_env_cfg("Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm"), device="cpu")
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
    cfg = load_env_cfg("Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm")

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
    g1_23dof_cfg = load_env_cfg("Unitree-G1-23Dof-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm")
    go2_cfg = load_env_cfg("Unitree-Go2-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm")

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
    cfg = load_env_cfg("Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-MLP-CmdCurr-ObsNorm")
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

    mha_cfg = load_env_cfg("Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm")
    self.assertEqual(
      set(mha_cfg.observations["actor"].terms),
      {"joint_pos", "command", "actions"},
    )
    self.assertFalse(mha_cfg.observations["actor"].flatten_history_dim)
    self.assertTrue(
      load_rl_cfg("Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm").actor.class_name.endswith(
        ":ResidualMhaModel"
      )
    )

  def test_flat_partial_phase1_keeps_imu_drops_ankles_and_command_curriculum(
    self,
  ) -> None:
    cfg = load_env_cfg("Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-MLP-CmdCurr-ObsNorm")
    play_cfg = unitree_g1_flat_partial_phase1_env_cfg(play=True)
    mha_cfg = load_env_cfg("Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm")

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
      load_rl_cfg("Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-MLP-CmdCurr-ObsNorm").actor.class_name.endswith(
        ":ResidualMlpModel"
      )
    )
    self.assertTrue(
      load_rl_cfg("Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm").actor.class_name.endswith(
        ":ResidualMhaModel"
      )
    )
    no_curr_mlp = load_env_cfg(
      "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-MLP-NoCurr-ObsNorm"
    )
    self.assertNotIn("command_vel", no_curr_mlp.curriculum)
    self.assertNotIn("terrain_levels", no_curr_mlp.curriculum)
    self.assertEqual(no_curr_mlp.commands["twist"].ranges.lin_vel_x, (-2.0, 2.0))
    self.assertEqual(no_curr_mlp.commands["twist"].ranges.lin_vel_y, (-1.0, 1.0))
    self.assertEqual(no_curr_mlp.commands["twist"].ranges.ang_vel_z, (-1.0, 1.0))
    self.assertEqual(
      set(no_curr_mlp.observations["actor"].terms),
      set(cfg.observations["actor"].terms),
    )
    self.assertEqual(no_curr_mlp.observations["actor"].history_length, 5)
    self.assertTrue(no_curr_mlp.observations["actor"].flatten_history_dim)
    self.assertTrue(
      load_rl_cfg(
        "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-MLP-NoCurr-ObsNorm"
      ).actor.class_name.endswith(":ResidualMlpModel")
    )
    h1_no_curr_mlp = load_env_cfg(
      "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H1-MLP-NoCurr-ObsNorm"
    )
    self.assertNotIn("command_vel", h1_no_curr_mlp.curriculum)
    self.assertNotIn("terrain_levels", h1_no_curr_mlp.curriculum)
    self.assertEqual(h1_no_curr_mlp.commands["twist"].ranges.lin_vel_x, (-2.0, 2.0))
    self.assertEqual(h1_no_curr_mlp.commands["twist"].ranges.lin_vel_y, (-1.0, 1.0))
    self.assertEqual(h1_no_curr_mlp.commands["twist"].ranges.ang_vel_z, (-1.0, 1.0))
    self.assertEqual(
      set(h1_no_curr_mlp.observations["actor"].terms),
      set(cfg.observations["actor"].terms),
    )
    self.assertEqual(
      set(h1_no_curr_mlp.observations["critic"].terms),
      set(cfg.observations["critic"].terms),
    )
    self.assertEqual(h1_no_curr_mlp.observations["actor"].history_length, 1)
    self.assertEqual(h1_no_curr_mlp.observations["critic"].history_length, 1)
    self.assertTrue(h1_no_curr_mlp.observations["actor"].flatten_history_dim)
    self.assertTrue(h1_no_curr_mlp.observations["critic"].flatten_history_dim)
    self.assertTrue(
      load_rl_cfg(
        "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H1-MLP-NoCurr-ObsNorm"
      ).actor.class_name.endswith(":ResidualMlpModel")
    )
    no_curr_mha = load_env_cfg(
      "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-TimeMHA-NoCurr-ObsNorm"
    )
    self.assertNotIn("command_vel", no_curr_mha.curriculum)
    self.assertNotIn("terrain_levels", no_curr_mha.curriculum)
    self.assertEqual(no_curr_mha.commands["twist"].ranges.lin_vel_x, (-2.0, 2.0))
    self.assertEqual(no_curr_mha.commands["twist"].ranges.lin_vel_y, (-1.0, 1.0))
    self.assertEqual(no_curr_mha.commands["twist"].ranges.ang_vel_z, (-1.0, 1.0))
    self.assertEqual(
      set(no_curr_mha.observations["actor"].terms),
      set(mha_cfg.observations["actor"].terms),
    )
    self.assertEqual(no_curr_mha.observations["actor"].history_length, 5)
    self.assertFalse(no_curr_mha.observations["actor"].flatten_history_dim)
    self.assertTrue(
      load_rl_cfg(
        "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-TimeMHA-NoCurr-ObsNorm"
      ).actor.class_name.endswith(":ResidualMhaModel")
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

  def test_group_time_pairs_position_with_applied_torque_and_masks_q(self) -> None:
    torch.manual_seed(0)
    batch_size = 2
    history = 5
    # command 3, joint_pos (knee, hip), actions (knee, hip, ankle)
    obs = TensorDict(
      {"actor": torch.randn(batch_size, history, 8)}, batch_size=[batch_size]
    )
    obs["actor"][..., 3] = 4.0
    obs["actor"][..., 7] = 3.0
    model = _toy_group_time(obs)

    positions, torques = model.core.slicer(obs["actor"])
    torch.testing.assert_close(positions[..., 0], obs["actor"][..., 3])
    torch.testing.assert_close(positions[..., 2], torch.zeros(batch_size, history))
    torch.testing.assert_close(torques, obs["actor"][..., 5:8])

    model.update_normalization(obs)
    self.assertGreater(float(model.core.joint_pos_norm.count[0]), 0.0)
    self.assertEqual(float(model.core.joint_pos_norm.count[2]), 0.0)
    self.assertEqual(float(model.core.joint_torque_norm.count[2]), 0.0)
    captured: dict[str, torch.Tensor] = {}

    def _capture_global(_module, args, _output):
      captured["features"] = args[0].detach()

    handle = model.core.global_encoder.register_forward_hook(_capture_global)
    latent = model.get_latent(obs)
    handle.remove()
    joints = len(model.joint_names)
    global_features = captured["features"]
    self.assertEqual(tuple(global_features.shape), (batch_size, 3 * joints))
    torch.testing.assert_close(global_features[:, 2], torch.zeros(batch_size))
    torch.testing.assert_close(global_features[:, joints + 2], torch.full((batch_size,), 3.0 / 1.01))
    torch.testing.assert_close(global_features[:, 2 * joints + 2], torch.zeros(batch_size))
    bypass = _summary_bypass(model)
    ankle_q = latent[:, bypass + 2]
    ankle_valid = latent[:, bypass + joints + 2]
    ankle_torque = latent[:, bypass + 2 * joints + 2]
    torch.testing.assert_close(ankle_q, torch.zeros(batch_size))
    torch.testing.assert_close(ankle_valid, torch.zeros(batch_size))
    torch.testing.assert_close(latent[:, bypass + joints], torch.ones(batch_size))
    torch.testing.assert_close(ankle_torque, torch.full((batch_size,), 3.0 / 1.01))
    self.assertEqual(
      [name for name, _param in model.named_parameters() if "positional_embedding" in name],
      ["core.positional_embedding"],
    )

    output_layer = _last_linear(model.mlp)
    torch.testing.assert_close(
      output_layer.weight.norm(dim=1),
      torch.full((3,), 0.01),
      atol=1.0e-6,
      rtol=1.0e-5,
    )
    _assert_group_time_export(self, model, obs)

  def test_group_time_fusion_is_the_registered_five_frame_actor(self) -> None:
    task_id = "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-GroupTimeFusion-CmdCurr-ObsNorm"
    no_imu_task_id = (
      "Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-H5-GroupTimeFusion-CmdCurr-ObsNorm"
    )
    registered = set(list_tasks())
    for removed in (
      "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H8-GroupTime-CmdCurr-ObsNorm",
      "Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-H8-GroupTime-CmdCurr-ObsNorm",
      "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H8-GroupTimeFusion-CmdCurr-ObsNorm",
    ):
      self.assertNotIn(removed, registered)

    actor_cfg = load_rl_cfg(task_id).actor
    env_cfg = load_env_cfg(task_id)
    no_imu_env = load_env_cfg(no_imu_task_id)
    self.assertTrue(actor_cfg.class_name.endswith(":ResidualGroupTimeModel"))
    self.assertFalse(hasattr(actor_cfg, "cross_joint_fusion"))
    self.assertEqual(actor_cfg.history_length, 5)
    self.assertEqual(actor_cfg.hidden_dims, (320, 160))
    self.assertEqual(actor_cfg.encoder_hidden_dim, 64)
    self.assertEqual(actor_cfg.joint_attention_dim, 32)
    self.assertEqual(actor_cfg.joint_attention_heads, 4)
    self.assertEqual(actor_cfg.joint_attention_output_dim, 64)
    self.assertEqual(env_cfg.observations["actor"].history_length, 5)
    self.assertEqual(env_cfg.observations["critic"].history_length, 5)
    self.assertFalse(env_cfg.observations["actor"].flatten_history_dim)
    self.assertIn("base_ang_vel", env_cfg.observations["actor"].terms)
    self.assertEqual(no_imu_env.observations["actor"].history_length, 5)
    self.assertNotIn("base_ang_vel", no_imu_env.observations["actor"].terms)
    self.assertIn("command_vel", env_cfg.curriculum)

    no_curr_id = (
      "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-GroupTimeFusion-NoCurr-ObsNorm"
    )
    no_curr = load_env_cfg(no_curr_id)
    self.assertNotIn("command_vel", no_curr.curriculum)
    self.assertNotIn("terrain_levels", no_curr.curriculum)
    self.assertEqual(no_curr.commands["twist"].ranges.lin_vel_x, (-2.0, 2.0))
    self.assertEqual(no_curr.commands["twist"].ranges.lin_vel_y, (-1.0, 1.0))
    self.assertEqual(no_curr.commands["twist"].ranges.ang_vel_z, (-1.0, 1.0))
    self.assertEqual(
      set(no_curr.observations["actor"].terms),
      set(env_cfg.observations["actor"].terms),
    )
    self.assertEqual(no_curr.observations["actor"].history_length, 5)
    self.assertTrue(
      load_rl_cfg(no_curr_id).actor.class_name.endswith(":ResidualGroupTimeModel")
    )

    feature_dim = sum(width for _name, width in actor_cfg.term_dims)
    obs = TensorDict(
      {"actor": torch.zeros(2, actor_cfg.history_length, feature_dim)},
      batch_size=[2],
    )
    model = ResidualGroupTimeModel(
      obs,
      {"actor": ["actor"]},
      "actor",
      len(actor_cfg.action_names),
      hidden_dims=actor_cfg.hidden_dims,
      activation=actor_cfg.activation,
      distribution_cfg=dict(actor_cfg.distribution_cfg),
      history_length=actor_cfg.history_length,
      num_heads=actor_cfg.num_heads,
      encoder_hidden_dim=actor_cfg.encoder_hidden_dim,
      joint_attention_dim=actor_cfg.joint_attention_dim,
      joint_attention_heads=actor_cfg.joint_attention_heads,
      joint_attention_output_dim=actor_cfg.joint_attention_output_dim,
      term_dims=actor_cfg.term_dims,
      joint_pos_names=actor_cfg.joint_pos_names,
      action_names=actor_cfg.action_names,
    )
    self.assertEqual(model.joint_names, tuple(actor_cfg.action_names))
    self.assertEqual(model.core.joint_id.num_embeddings, 29)
    self.assertEqual(model.core.joint_id.embedding_dim, 16)
    self.assertEqual(int(model.core.valid_joint.sum()), 25)
    for name in _G1_ANKLE_JOINTS:
      index = model.joint_names.index(name)
      self.assertEqual(float(model.core.valid_joint[index]), 0.0)
    self.assertEqual(model.core.global_encoder.projection.out_features, 32)
    self.assertEqual(model.core.global_encoder.projection.in_features, 3 * 29 + 6)
    queries = model.core.conditional_queries(obs["actor"])
    self.assertEqual(tuple(queries.shape), (2, 29, 32))
    self.assertFalse(torch.allclose(queries[:, 0], queries[:, 1]))
    self.assertIsNotNone(model.core.fusion)
    output = model(obs)
    self.assertTrue(torch.isfinite(output).all())

  def test_joint_time_attention_stays_32d_and_lifts_after_the_residual(self) -> None:
    torch.manual_seed(0)
    batch_size = 2
    history = 5
    obs = TensorDict({"actor": torch.randn(batch_size, history, 14)}, batch_size=[batch_size])
    model = ResidualGroupTimeModel(
      obs,
      {"actor": ["actor"]},
      "actor",
      3,
      hidden_dims=(16,),
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      history_length=history,
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
    joint = model.core.joint_encoder
    imu = model.core.imu_encoder
    self.assertIsNotNone(imu)
    assert imu is not None
    self.assertIsNot(joint.history_encoder, joint.query_encoder)
    self.assertEqual(joint.history_encoder.projection.in_features, 3)
    self.assertEqual(joint.history_encoder.projection.out_features, 32)
    self.assertEqual(joint.query_encoder.projection.in_features, 3 + 16 + 32 + 3)
    self.assertEqual(joint.query_encoder.projection.out_features, 32)
    self.assertEqual(joint.attention.embed_dim, 32)
    self.assertEqual(joint.attention.num_heads, 4)
    self.assertEqual(joint.query_norm.normalized_shape, (32,))
    self.assertEqual(joint.memory_norm.normalized_shape, (32,))
    self.assertEqual(joint.output_norm.normalized_shape, (32,))
    self.assertEqual(joint.output_projection.in_features, 32)
    self.assertEqual(joint.output_projection.out_features, 64)
    self.assertTrue(joint.need_weights)
    self.assertEqual(imu.projection.in_features, 7)
    self.assertEqual(imu.projection.out_features, 64)
    self.assertEqual(imu.attention.embed_dim, 64)
    self.assertEqual(imu.attention.num_heads, 4)
    self.assertIsInstance(imu.output_projection, torch.nn.Identity)
    self.assertTrue(imu.need_weights)
    self.assertEqual(model.core.fusion.attention.embed_dim, 64)
    self.assertEqual(model.core.fusion.attention.num_heads, 4)
    self.assertEqual(tuple(model.core.joint_positional_embedding.shape), (1, history, 32))
    self.assertEqual(tuple(model.core.imu_positional_embedding.shape), (1, history, 64))
    self.assertEqual(model.mlp[0].in_features, 3 * 64 + 64 + 3 * 3 + 3)

    seen: dict[str, object] = {}
    joint_attention = joint.attention.forward
    imu_attention = imu.attention.forward
    fusion_attention = model.core.fusion.attention.forward

    def _record_joint(query, key, value, need_weights=True, attn_mask=None, **kwargs):
      seen["query"] = tuple(query.shape)
      seen["key"] = tuple(key.shape)
      seen["need_weights"] = need_weights
      seen["attn_mask"] = attn_mask
      return joint_attention(
        query, key, value, need_weights=need_weights, attn_mask=attn_mask, **kwargs
      )

    def _record_imu(query, key, value, need_weights=True, attn_mask=None, **kwargs):
      seen["imu_query"] = tuple(query.shape)
      seen["imu_key"] = tuple(key.shape)
      seen["imu_need_weights"] = need_weights
      return imu_attention(
        query, key, value, need_weights=need_weights, attn_mask=attn_mask, **kwargs
      )

    def _record_fusion(query, key, value, need_weights=True, attn_mask=None, **kwargs):
      seen["fusion_query"] = tuple(query.shape)
      seen["fusion_need_weights"] = need_weights
      return fusion_attention(
        query, key, value, need_weights=need_weights, attn_mask=attn_mask, **kwargs
      )

    def _record_norm(_module, _inputs, output):
      seen["norm_out"] = output.detach()

    def _record_lift(_module, inputs, output):
      seen["lift_in"] = inputs[0].detach()
      seen["lift_out"] = output.detach()

    joint.attention.forward = _record_joint
    imu.attention.forward = _record_imu
    model.core.fusion.attention.forward = _record_fusion
    norm_handle = joint.output_norm.register_forward_hook(_record_norm)
    lift_handle = joint.output_projection.register_forward_hook(_record_lift)
    latent = model.get_latent(obs)
    self.assertEqual(seen["query"], (batch_size * 3, 1, 32))
    self.assertEqual(seen["key"], (batch_size * 3, history, 32))
    self.assertTrue(seen["need_weights"])
    self.assertIsNone(seen["attn_mask"])
    self.assertEqual(seen["imu_query"], (batch_size, 1, 64))
    self.assertEqual(seen["imu_key"], (batch_size, history, 64))
    self.assertTrue(seen["imu_need_weights"])
    self.assertEqual(seen["fusion_query"], (batch_size, 4, 64))
    self.assertFalse(seen["fusion_need_weights"])
    norm_out = seen["norm_out"]
    lift_in = seen["lift_in"]
    lift_out = seen["lift_out"]
    assert isinstance(norm_out, torch.Tensor)
    assert isinstance(lift_in, torch.Tensor)
    assert isinstance(lift_out, torch.Tensor)
    torch.testing.assert_close(lift_in, norm_out.squeeze(1))
    self.assertEqual(lift_in.shape[-1], 32)
    self.assertEqual(lift_out.shape[-1], 64)
    self.assertEqual(tuple(latent.shape), (batch_size, model.mlp[0].in_features))
    output = model(obs)
    self.assertEqual(tuple(output.shape), (batch_size, 3))
    output.square().mean().backward()
    projection_grad = joint.output_projection.weight.grad
    fusion_grad = model.core.fusion.attention.in_proj_weight.grad
    self.assertIsNotNone(projection_grad)
    self.assertIsNotNone(fusion_grad)
    assert projection_grad is not None
    assert fusion_grad is not None
    self.assertTrue(torch.isfinite(output).all())
    self.assertTrue(torch.isfinite(projection_grad).all())
    self.assertTrue(torch.isfinite(fusion_grad).all())
    self.assertGreater(float(fusion_grad.abs().sum()), 0.0)
    joint.attention.forward = joint_attention
    imu.attention.forward = imu_attention
    model.core.fusion.attention.forward = fusion_attention
    norm_handle.remove()
    lift_handle.remove()
    _assert_group_time_export(self, model, obs, onnx_file=True)

  def test_conditional_query_uses_joint_id_global_state_and_command(self) -> None:
    torch.manual_seed(0)
    batch_size = 2
    history = 5
    obs = TensorDict(
      {"actor": torch.randn(batch_size, history, 8)}, batch_size=[batch_size]
    )
    model = _toy_group_time(obs)
    sequence = obs["actor"]
    local_dim = model.core.joint_encoder.local_dim
    id_dim = model.core.JOINT_ID_DIM
    global_dim = model.core.GLOBAL_DIM
    command_dim = int(model.core.command_in_context.numel())

    captured: dict[str, torch.Tensor] = {}

    def _capture_global(_module, _args, output):
      captured["global"] = output.detach()

    def _capture_query(_module, args, _output):
      captured.setdefault("query", args[0].detach())

    global_handle = model.core.global_encoder.register_forward_hook(_capture_global)
    query_handle = model.core.joint_encoder.query_encoder.register_forward_hook(_capture_query)
    with torch.no_grad():
      inputs = model.core.query_inputs(sequence)
      base_queries = model.core.conditional_queries(sequence)
      base_memory = model.core.history_memory(sequence)
    global_handle.remove()
    query_handle.remove()

    self.assertEqual(
      tuple(inputs.shape),
      (batch_size, 3, local_dim + id_dim + global_dim + command_dim),
    )
    torch.testing.assert_close(captured["query"], inputs.reshape(batch_size * 3, -1))
    torch.testing.assert_close(inputs[0, :, local_dim : local_dim + id_dim], model.core.joint_id.weight)
    torch.testing.assert_close(
      inputs[:, 0, local_dim + id_dim : local_dim + id_dim + global_dim],
      captured["global"],
    )
    positions, torques = model.core.slicer(sequence)
    valid = model.core.valid_joint.view(1, 1, -1)
    local = torch.stack(
      (
        model.core.joint_pos_norm(positions, valid)[:, -1],
        model.core.joint_torque_norm(torques, torch.ones_like(model.core.valid_joint))[:, -1],
        valid.expand(batch_size, history, -1)[:, -1],
      ),
      dim=-1,
    )
    torch.testing.assert_close(inputs[..., :local_dim], local)
    torch.testing.assert_close(inputs[0, 0, -command_dim:], sequence[0, -1, :command_dim] / 1.01)

    matched = torch.zeros(batch_size, history, 8)
    matched[..., 3:5] = 0.4
    matched[..., 5:8] = -0.2
    with torch.no_grad():
      matched_inputs = model.core.query_inputs(matched)
      matched_queries = model.core.conditional_queries(matched)
    torch.testing.assert_close(matched_inputs[:, 0, :local_dim], matched_inputs[:, 1, :local_dim])
    self.assertFalse(
      torch.allclose(
        matched_inputs[:, 0, local_dim : local_dim + id_dim],
        matched_inputs[:, 1, local_dim : local_dim + id_dim],
      )
    )
    self.assertFalse(torch.allclose(matched_queries[:, 0], matched_queries[:, 1]))

    with torch.no_grad():
      command_obs = sequence.clone()
      command_obs[:, -1, 0] += 1.5
      command_queries = model.core.conditional_queries(command_obs)
      command_memory = model.core.history_memory(command_obs)
      past_command = sequence.clone()
      past_command[:, 0, 0] += 1.5
      past_command_queries = model.core.conditional_queries(past_command)
      other_joint = sequence.clone()
      other_joint[:, -1, 4] += 1.5
      other_queries = model.core.conditional_queries(other_joint)
      other_memory = model.core.history_memory(other_joint)
      past_other = sequence.clone()
      past_other[:, 0, 4] += 1.5
      past_other_queries = model.core.conditional_queries(past_other)
      past_other_memory = model.core.history_memory(past_other)
    self.assertFalse(torch.allclose(base_queries[:, 0], command_queries[:, 0]))
    torch.testing.assert_close(base_memory, command_memory)
    torch.testing.assert_close(past_command_queries, base_queries)
    self.assertFalse(torch.allclose(base_queries[:, 0], other_queries[:, 0]))
    torch.testing.assert_close(base_memory[:, 0], other_memory[:, 0])
    torch.testing.assert_close(past_other_queries, base_queries)
    torch.testing.assert_close(past_other_memory[:, 0], base_memory[:, 0])
    self.assertFalse(torch.allclose(past_other_memory[:, 1], base_memory[:, 1]))

    output = model(obs)
    output.square().mean().backward()
    id_grad = model.core.joint_id.weight.grad
    query_grad = model.core.joint_encoder.query_encoder.projection.weight.grad
    self.assertIsNotNone(id_grad)
    self.assertIsNotNone(query_grad)
    assert id_grad is not None
    assert query_grad is not None
    self.assertTrue(torch.isfinite(output).all())
    self.assertTrue(torch.isfinite(id_grad).all())
    self.assertGreater(float(id_grad.abs().sum()), 0.0)
    command_columns = query_grad[:, local_dim + id_dim + global_dim :]
    global_columns = query_grad[:, local_dim + id_dim : local_dim + id_dim + global_dim]
    id_columns = query_grad[:, local_dim : local_dim + id_dim]
    self.assertTrue(torch.isfinite(query_grad).all())
    self.assertGreater(float(command_columns.abs().sum()), 0.0)
    self.assertGreater(float(global_columns.abs().sum()), 0.0)
    self.assertGreater(float(id_columns.abs().sum()), 0.0)
    for name, param in model.core.global_encoder.named_parameters():
      self.assertIsNotNone(param.grad, name)
      assert param.grad is not None
      self.assertTrue(torch.isfinite(param.grad).all(), name)
      self.assertGreater(float(param.grad.abs().sum()), 0.0, name)

  def test_summary_fusion_carries_one_joint_into_the_others(self) -> None:
    torch.manual_seed(0)
    obs = TensorDict({"actor": torch.zeros(2, 5, 8)}, batch_size=[2])
    model = _toy_group_time(obs)
    fusion = model.core.fusion
    width = fusion.pre_norm.normalized_shape[0]
    joints = torch.randn(4, 3, width)
    imu = torch.randn(4, width)
    base_joints, base_imu = fusion(joints, imu)
    changed = joints.clone()
    # A uniform shift is removed by the pre-norm, so move one feature.
    changed[:, 0, 0] = changed[:, 0, 0] + 3.0
    other_joints, other_imu = fusion(changed, imu)
    self.assertFalse(torch.allclose(base_joints[:, 1], other_joints[:, 1]))
    self.assertFalse(torch.allclose(base_joints[:, 2], other_joints[:, 2]))
    self.assertFalse(torch.allclose(base_imu, other_imu))
    self.assertTrue(torch.isfinite(other_joints).all())
    self.assertTrue(torch.isfinite(other_imu).all())

  def test_group_time_attention_maps_are_softmaxes_over_frames_or_tokens(self) -> None:
    torch.manual_seed(0)
    batch_size = 2
    history = 5
    obs = TensorDict({"actor": torch.randn(batch_size, history, 8)}, batch_size=[batch_size])
    model = _toy_group_time(obs)
    imu_obs = TensorDict(
      {"actor": torch.randn(batch_size, history, 14)}, batch_size=[batch_size]
    )
    imu = ResidualGroupTimeModel(
      imu_obs,
      {"actor": ["actor"]},
      "actor",
      3,
      hidden_dims=(16,),
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      history_length=history,
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

    time_map = model.attention_weights(obs)
    self.assertEqual(tuple(time_map["time"].shape), (batch_size, 4, 3, history))
    self.assertEqual(time_map["time_rows"], ("knee", "hip", "ankle"))
    self.assertEqual(time_map["time_columns"], ("t-4", "t-3", "t-2", "t-1", "t"))
    self.assertNotIn("imu_time", time_map)
    self.assertEqual(tuple(time_map["fusion"].shape), (batch_size, 4, 4, 4))
    self.assertEqual(time_map["fusion_names"], ("knee", "hip", "ankle", "imu"))
    torch.testing.assert_close(
      time_map["time"].sum(dim=-1), torch.ones(batch_size, 4, 3), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
      time_map["fusion"].sum(dim=-1), torch.ones(batch_size, 4, 4), atol=1e-5, rtol=1e-5
    )

    with torch.no_grad():
      current_imu = imu_obs["actor"].clone()
      current_imu[:, -1, 0] += 1.5
      past_imu = imu_obs["actor"].clone()
      past_imu[:, 0, 0] += 1.5
      base_queries = imu.core.conditional_queries(imu_obs["actor"])
      current_queries = imu.core.conditional_queries(current_imu)
      past_queries = imu.core.conditional_queries(past_imu)
      base_memory = imu.core.history_memory(imu_obs["actor"])
      current_memory = imu.core.history_memory(current_imu)
    self.assertFalse(torch.allclose(base_queries, current_queries))
    torch.testing.assert_close(base_memory, current_memory)
    torch.testing.assert_close(past_queries, base_queries)

    imu_map = imu.attention_weights(imu_obs)
    self.assertEqual(tuple(imu_map["imu_time"].shape), (batch_size, 4, 1, history))
    self.assertIn("fusion", imu_map)
    torch.testing.assert_close(
      imu_map["imu_time"].sum(dim=-1), torch.ones(batch_size, 4, 1), atol=1e-5, rtol=1e-5
    )
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "attention.png"
      plot_group_time_attention(imu_map, path)
      self.assertGreater(path.stat().st_size, 0)
      self.assertGreater(path.with_name("attention_imu.png").stat().st_size, 0)
      self.assertGreater(path.with_name("attention_fusion.png").stat().st_size, 0)

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


def _toy_group_time(obs: TensorDict) -> ResidualGroupTimeModel:
  return ResidualGroupTimeModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    3,
    hidden_dims=(16,),
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    history_length=int(obs["actor"].shape[1]),
    num_heads=4,
    encoder_hidden_dim=32,
    joint_attention_dim=32,
    joint_attention_output_dim=32,
    term_dims=(("command", 3), ("joint_pos", 2), ("actions", 3)),
    joint_pos_names=("knee", "hip"),
    action_names=("knee", "hip", "ankle"),
  )


def _summary_bypass(model: ResidualGroupTimeModel) -> int:
  joints = len(model.joint_names)
  imu_width = model.encoder_hidden_dim if model.has_imu else model.joint_attention_output_dim
  return joints * model.joint_attention_output_dim + imu_width


def _assert_group_time_export(
  test: unittest.TestCase,
  model: ResidualGroupTimeModel,
  obs: TensorDict,
  *,
  onnx_file: bool = False,
) -> None:
  expected = model(obs)
  torch.testing.assert_close(model.as_onnx()(obs["actor"]), expected)
  scripted = torch.jit.script(model.as_jit())
  torch.testing.assert_close(scripted(obs["actor"]), expected)
  if not onnx_file:
    return
  import onnxruntime as ort

  onnx_model = model.as_onnx()
  onnx_model.eval()
  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "policy.onnx"
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),
      path,
      export_params=True,
      opset_version=18,
      input_names=onnx_model.input_names,
      output_names=onnx_model.output_names,
      dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # The training exporter traces a batch of one, with static axes.
    single = obs["actor"][:1].detach().cpu().contiguous()
    exported = session.run(None, {"obs": single.numpy()})[0]
  single_obs = TensorDict({"actor": obs["actor"][:1]}, batch_size=[1])
  torch.testing.assert_close(torch.from_numpy(exported), model(single_obs).detach())


def _clone_actor_obs(obs: TensorDict) -> TensorDict:
  return TensorDict(
    {"actor": obs["actor"].clone()}, batch_size=obs.batch_size
  )


class TrainingMetricsFileTest(unittest.TestCase):
  def test_max_iterations_are_split_into_two_hundred_points(self) -> None:
    points = [step for step in range(50_000) if is_metrics_iteration(step, 50_000)]
    self.assertEqual(len(points), 200)
    self.assertEqual(points[0], 249)
    self.assertEqual(points[-1], 49_999)
    self.assertEqual(points[1] - points[0], 250)

    uneven = [step for step in range(10_001) if is_metrics_iteration(step, 10_001)]
    self.assertEqual(len(uneven), 200)
    self.assertEqual(uneven[-1], 10_000)
    self.assertEqual(
      [step for step in range(4) if is_metrics_iteration(step, 4)],
      [0, 1, 2, 3],
    )

  def test_log_hook_appends_only_sampled_iterations(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      logger = SimpleNamespace(
        disable_logs=False,
        log_dir=tmp,
        lenbuffer=[8.0],
        rewbuffer=[3.0],
        log=lambda **_kwargs: None,
      )
      runner = SimpleNamespace(
        logger=logger,
        current_learning_iteration=0,
      )
      runner._record_training_metrics = lambda: (  # type: ignore[attr-defined]
        EffortOnPolicyRunner._record_training_metrics(runner)  # type: ignore[arg-type]
      )
      hooked = metrics_log_hook(runner, logger.log, 50_000)  # type: ignore[arg-type]
      hooked(it=248)
      hooked(it=249)
      hooked(it=250)
      lines = (Path(tmp) / "training_metrics.csv").read_text(encoding="utf-8").splitlines()

    self.assertEqual(
      lines,
      [
        "iteration,episode_length,cumulative_reward",
        "249,8.000000,3.000000",
      ],
    )

  def test_checkpoint_appends_iteration_length_and_return(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      logger = SimpleNamespace(
        disable_logs=False,
        log_dir=tmp,
        lenbuffer=[10.0, 30.0],
        rewbuffer=[-2.0, 4.0],
      )
      runner = SimpleNamespace(
        logger=logger,
        current_learning_iteration=1000,
      )
      EffortOnPolicyRunner._record_training_metrics(runner)  # type: ignore[arg-type]
      runner.current_learning_iteration = 1000
      EffortOnPolicyRunner._record_training_metrics(runner)  # type: ignore[arg-type]
      runner.current_learning_iteration = 2000
      logger.lenbuffer = []
      logger.rewbuffer = []
      EffortOnPolicyRunner._record_training_metrics(runner)  # type: ignore[arg-type]

      lines = (Path(tmp) / "training_metrics.csv").read_text(encoding="utf-8").splitlines()

    self.assertEqual(
      lines,
      [
        "iteration,episode_length,cumulative_reward",
        "1000,20.000000,1.000000",
        "2000,,",
      ],
    )

  def test_append_training_metrics_starts_a_new_file(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      append_training_metrics(tmp, 0, 12.5, -3.25)
      text = (Path(tmp) / "training_metrics.csv").read_text(encoding="utf-8")
    self.assertTrue(text.startswith("iteration,episode_length,cumulative_reward\n"))
    self.assertIn("0,12.500000,-3.250000", text)


if __name__ == "__main__":
  unittest.main()
