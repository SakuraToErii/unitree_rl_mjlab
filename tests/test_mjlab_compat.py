from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mjlab.tasks  # noqa: F401
import numpy as np
import torch
from mjlab.sensor import TerrainHeightSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import src.tasks  # noqa: F401
from src.assets import robots
from src.assets.robots.tiangong3.tk3_constants import (
  TK3_ARTICULATION,
  TK3_COMMAND_DELAY_MAX_LAG,
  TK3_COMMAND_DELAY_MIN_LAG,
)
from src.assets.robots.unitree_g1.g1_23dof_constants import G1_23DOF_ARTICULATION
from src.assets.robots.unitree_g1.g1_constants import G1_ARTICULATION
from src.assets.robots.unitree_go2.go2_constants import GO2_ARTICULATION
from src.tasks.tracking.mdp.commands import MotionLoader
from src.tasks.tracking.mdp.events import randomize_actuator_command_lag
from src.tasks.tracking.mdp.metrics import (
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from src.tasks.velocity.mdp.curriculums import terrain_levels_vel

LOCAL_TASKS = {
  "Unitree-G1-Rough",
  "Unitree-G1-Flat",
  "Unitree-G1-23Dof-Rough",
  "Unitree-G1-23Dof-Flat",
  "Unitree-Go2-Rough",
  "Unitree-Go2-Flat",
  "Unitree-G1-Tracking",
  "Unitree-G1-Tracking-No-State-Estimation",
  "Unitree-G1-23Dof-Tracking",
  "Unitree-G1-23Dof-Tracking-No-State-Estimation",
  "TK3-Tracking",
}

REMOVED_TASK_MARKERS = ("R1", "H1_2", "H2", "A2", "AS2")


class Mjlab153CompatibilityTest(unittest.TestCase):
  def test_local_tasks_register_and_load(self) -> None:
    task_ids = set(list_tasks())
    self.assertTrue(LOCAL_TASKS.issubset(task_ids))
    for task_id in LOCAL_TASKS:
      load_env_cfg(task_id)
      load_rl_cfg(task_id)

    local_unitree_tasks = {task_id for task_id in task_ids if "Unitree-" in task_id}
    for marker in REMOVED_TASK_MARKERS:
      self.assertFalse(
        any(marker in task_id for task_id in local_unitree_tasks),
        f"Removed robot task still registered: {marker}",
      )

  def test_retained_robot_specs_compile(self) -> None:
    getters = (
      robots.get_g1_robot_cfg,
      robots.get_g1_23dof_robot_cfg,
      robots.get_go2_robot_cfg,
      robots.get_tk3_robot_cfg,
    )
    for get_robot_cfg in getters:
      model = get_robot_cfg().spec_fn().compile()
      self.assertGreater(model.nbody, 1)
      self.assertGreater(model.njnt, 0)

  def test_velocity_height_sensors_are_terrain_relative(self) -> None:
    for task_id in (
      "Unitree-G1-Rough",
      "Unitree-G1-23Dof-Rough",
      "Unitree-Go2-Rough",
    ):
      cfg = load_env_cfg(task_id)
      sensors = {sensor.name: sensor for sensor in cfg.scene.sensors or ()}
      terrain_scan = sensors["terrain_scan"]
      self.assertEqual(terrain_scan.include_geom_groups, (0,))
      foot_height_scan = sensors["foot_height_scan"]
      self.assertIsInstance(foot_height_scan, TerrainHeightSensorCfg)
      self.assertEqual(foot_height_scan.include_geom_groups, (0,))
      self.assertGreater(len(foot_height_scan.frame), 0)
      actor_joint_pos = cfg.observations["actor"].terms["joint_pos"]
      critic_joint_pos = cfg.observations["critic"].terms["joint_pos"]
      self.assertTrue(actor_joint_pos.params["biased"])
      self.assertNotIn("biased", critic_joint_pos.params)

  def test_initial_reset_does_not_advance_terrain_level(self) -> None:
    class Terrain:
      def __init__(self) -> None:
        self.cfg = SimpleNamespace(
          terrain_generator=SimpleNamespace(
            size=(4.0, 4.0),
            sub_terrains={"rough": object()},
          )
        )
        self.terrain_levels = torch.zeros(2, dtype=torch.long)
        self.terrain_origins = torch.zeros((2, 1, 3))
        self.terrain_types = torch.zeros(2, dtype=torch.long)
        self.update_args: tuple[torch.Tensor, torch.Tensor] | None = None

      def update_env_origins(
        self,
        env_ids: torch.Tensor,
        move_up: torch.Tensor,
        move_down: torch.Tensor,
      ) -> None:
        del env_ids
        self.update_args = (move_up, move_down)

    class Scene(dict):
      pass

    terrain = Terrain()
    asset = SimpleNamespace(
      data=SimpleNamespace(
        root_link_pos_w=torch.tensor(((10.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
      )
    )
    scene = Scene(robot=asset)
    scene.terrain = terrain
    scene.env_origins = torch.zeros((2, 3))
    env = SimpleNamespace(
      scene=scene,
      command_manager=SimpleNamespace(
        get_command=lambda _name: torch.zeros((2, 3))
      ),
      max_episode_length_s=20.0,
      common_step_counter=0,
    )

    result = terrain_levels_vel(env, torch.tensor((0, 1)), "twist")
    assert terrain.update_args is not None
    move_up, move_down = terrain.update_args
    self.assertFalse(move_up.any())
    self.assertFalse(move_down.any())
    self.assertEqual(set(result), {"mean", "max", "rough"})

  def test_tracking_metrics_match_mjlab_153_semantics(self) -> None:
    command = SimpleNamespace(
      body_pos_w=torch.tensor((((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),)),
      body_pos_relative_w=torch.zeros((1, 2, 3)),
      robot_body_pos_w=torch.zeros((1, 2, 3)),
      joint_vel=torch.tensor(((1.0, 3.0),)),
      robot_joint_vel=torch.zeros((1, 2)),
      cfg=SimpleNamespace(body_names=("body_a", "body_b")),
      num_envs=1,
      device=torch.device("cpu"),
    )

    torch.testing.assert_close(compute_mpkpe(command), torch.tensor((1.0,)))
    torch.testing.assert_close(
      compute_root_relative_mpkpe(command), torch.tensor((0.0,))
    )
    torch.testing.assert_close(
      compute_joint_velocity_error(command), torch.tensor((5.0**0.5,))
    )
    with self.assertRaises(ValueError):
      compute_ee_position_error(command, ("missing_body",))

  def test_removed_robots_are_not_exported(self) -> None:
    for name in (
      "get_r1_robot_cfg",
      "get_h1_2_robot_cfg",
      "get_h2_robot_cfg",
      "get_a2_robot_cfg",
      "get_as2_robot_cfg",
    ):
      self.assertFalse(hasattr(robots, name))

  def test_actuator_frictionloss_is_explicit(self) -> None:
    for articulation in (
      G1_ARTICULATION,
      G1_23DOF_ARTICULATION,
      GO2_ARTICULATION,
      TK3_ARTICULATION,
    ):
      for actuator in articulation.actuators:
        self.assertIsNotNone(actuator.frictionloss)

  def test_tk3_actuator_command_lag_is_sampled_per_episode(self) -> None:
    for actuator in TK3_ARTICULATION.actuators:
      self.assertEqual(actuator.delay_min_lag, TK3_COMMAND_DELAY_MIN_LAG)
      self.assertEqual(actuator.delay_max_lag, TK3_COMMAND_DELAY_MAX_LAG)
      self.assertEqual(actuator.delay_hold_prob, 1.0)

    cfg = load_env_cfg("TK3-Tracking")
    delay_event = cfg.events["actuator_command_delay"]
    self.assertEqual(delay_event.mode, "reset")
    self.assertIs(delay_event.func, randomize_actuator_command_lag)
    self.assertEqual(
      delay_event.params["lag_range"],
      (TK3_COMMAND_DELAY_MIN_LAG, TK3_COMMAND_DELAY_MAX_LAG),
    )

    class FakeActuator:
      def __init__(self, has_delay: bool) -> None:
        self.has_delay = has_delay
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

      def set_lags(self, lags: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.calls.append((lags.clone(), env_ids.clone()))

    delayed_a = FakeActuator(has_delay=True)
    delayed_b = FakeActuator(has_delay=True)
    direct = FakeActuator(has_delay=False)
    env = SimpleNamespace(
      num_envs=4,
      device=torch.device("cpu"),
      scene={
        "robot": SimpleNamespace(actuators=[delayed_a, delayed_b, direct]),
      },
    )
    env_ids = torch.tensor((1, 3))
    randomize_actuator_command_lag(
      env,
      env_ids,
      lag_range=(TK3_COMMAND_DELAY_MIN_LAG, TK3_COMMAND_DELAY_MAX_LAG),
    )

    self.assertEqual(len(delayed_a.calls), 1)
    self.assertEqual(len(delayed_b.calls), 1)
    self.assertEqual(len(direct.calls), 0)
    sampled_lags, sampled_env_ids = delayed_a.calls[0]
    torch.testing.assert_close(sampled_lags, delayed_b.calls[0][0])
    torch.testing.assert_close(sampled_env_ids, env_ids)
    self.assertTrue(torch.all(sampled_lags >= TK3_COMMAND_DELAY_MIN_LAG))
    self.assertTrue(torch.all(sampled_lags <= TK3_COMMAND_DELAY_MAX_LAG))

  def test_visualizer_imports_with_mjviser(self) -> None:
    importlib.import_module("scripts.visualize_terrain")

  def test_motion_loader_supports_named_and_legacy_layouts(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      arrays = self._motion_arrays()

      named_path = root / "named.npz"
      np.savez(
        named_path,
        **arrays,
        joint_names=np.array(("j2", "j1", "j3")),
        body_names=np.array(("body_b", "body_a", "body_c")),
      )
      named = MotionLoader(
        str(named_path),
        torch.tensor((0, 1)),
        joint_names=("j1", "j3"),
        body_names=("body_c", "body_a"),
      )
      np.testing.assert_allclose(named.joint_pos.numpy(), arrays["joint_pos"][:, (1, 2)])
      np.testing.assert_allclose(
        named.body_pos_w.numpy(), arrays["body_pos_w"][:, (2, 1)]
      )

      legacy_path = root / "legacy.npz"
      np.savez(legacy_path, **arrays)
      legacy = MotionLoader(str(legacy_path), torch.tensor((1, 0)))
      np.testing.assert_allclose(legacy.joint_pos.numpy(), arrays["joint_pos"])
      np.testing.assert_allclose(
        legacy.body_pos_w.numpy(), arrays["body_pos_w"][:, (1, 0)]
      )

  @staticmethod
  def _motion_arrays() -> dict[str, np.ndarray]:
    frame_count, joint_count, body_count = 2, 3, 3
    return {
      "joint_pos": np.arange(frame_count * joint_count, dtype=np.float32).reshape(
        frame_count, joint_count
      ),
      "joint_vel": np.zeros((frame_count, joint_count), dtype=np.float32),
      "body_pos_w": np.arange(
        frame_count * body_count * 3, dtype=np.float32
      ).reshape(frame_count, body_count, 3),
      "body_quat_w": np.tile(
        np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        (frame_count, body_count, 1),
      ),
      "body_lin_vel_w": np.zeros((frame_count, body_count, 3), dtype=np.float32),
      "body_ang_vel_w": np.zeros((frame_count, body_count, 3), dtype=np.float32),
    }


if __name__ == "__main__":
  unittest.main()
