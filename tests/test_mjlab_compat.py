from __future__ import annotations

import importlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mjlab.tasks  # noqa: F401
import mujoco
import numpy as np
import torch
from mjlab.actuator import BuiltinPositionActuator
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
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
from src.tasks.tracking.mdp.commands import (
  MotionCommand,
  MotionCommandCfg,
  MotionLoader,
)
from src.tasks.tracking.mdp.metrics import (
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from src.tasks.tracking.mdp.rewards import raw_action_torque_limit_penalty
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

  def test_tk3_urdf_preserves_fixed_links_and_visuals(self) -> None:
    urdf_path = (
      Path(__file__).parents[1]
      / "src/assets/robots/tiangong3/urdf/tiangong3.urdf"
    )
    link_count = len(ET.parse(urdf_path).getroot().findall("link"))
    model = mujoco.MjSpec.from_file(str(urdf_path)).compile()

    self.assertEqual(model.nbody - 1, link_count)
    self.assertEqual((model.geom_group == 1).sum(), link_count)

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

  def test_raw_action_torque_limit_penalty_uses_soft_limit(self) -> None:
    actuator = BuiltinPositionActuator.__new__(BuiltinPositionActuator)
    actuator.cfg = SimpleNamespace(transmission_type=TransmissionType.JOINT)
    actuator._target_ids = torch.tensor((0, 1))
    actuator._global_ctrl_ids = torch.tensor((0, 1))

    action_term = JointPositionAction.__new__(JointPositionAction)
    action_term._target_ids = torch.tensor((0, 1))
    action_term._raw_actions = torch.tensor(((0.5, 1.0), (0.0, 0.0)))
    action_term._scale = 1.0
    action_term._offset = 0.0

    gainprm = torch.zeros((2, 2, 10))
    gainprm[..., 0] = 10.0
    biasprm = torch.zeros((2, 2, 10))
    biasprm[..., 1] = -10.0
    asset = SimpleNamespace(
      num_joints=2,
      joint_names=("joint_0", "joint_1"),
      actuators=[actuator],
      data=SimpleNamespace(
        encoder_bias=torch.zeros((2, 2)),
        joint_pos=torch.zeros((2, 2)),
        joint_vel=torch.zeros((2, 2)),
      ),
    )
    env = SimpleNamespace(
      num_envs=2,
      device=torch.device("cpu"),
      scene={"robot": asset},
      action_manager=SimpleNamespace(get_term=lambda _name: action_term),
      sim=SimpleNamespace(
        model=SimpleNamespace(
          actuator_gainprm=gainprm,
          actuator_biasprm=biasprm,
          actuator_forcerange=torch.tensor(((-5.0, 5.0), (-5.0, 5.0))),
        ),
        expanded_fields={"actuator_gainprm", "actuator_biasprm"},
      ),
    )
    asset_cfg = SceneEntityCfg("robot", joint_ids=[0, 1])
    term_cfg = RewardTermCfg(
      func=raw_action_torque_limit_penalty,
      weight=-2.0,
      params={
        "action_name": "joint_pos",
        "asset_cfg": asset_cfg,
        "soft_ratio": 0.8,
      },
    )

    penalty = raw_action_torque_limit_penalty(term_cfg, env)
    value = penalty(
      env,
      action_name="joint_pos",
      asset_cfg=asset_cfg,
      soft_ratio=0.8,
    )
    torch.testing.assert_close(value, torch.tensor((2.3125, 0.0)))

    tk3_cfg = load_env_cfg("TK3-Tracking")
    self.assertEqual(
      tk3_cfg.rewards["raw_action_torque_limit"].params["soft_ratio"],
      0.8,
    )

  def test_tk3_actuator_command_lag_is_sampled_after_motion_resample(self) -> None:
    for actuator in TK3_ARTICULATION.actuators:
      self.assertEqual(actuator.delay_min_lag, TK3_COMMAND_DELAY_MIN_LAG)
      self.assertEqual(actuator.delay_max_lag, TK3_COMMAND_DELAY_MAX_LAG)
      self.assertEqual(actuator.delay_hold_prob, 1.0)

    cfg = load_env_cfg("TK3-Tracking")
    self.assertNotIn("actuator_command_delay", cfg.events)
    motion_cfg = cfg.commands["motion"]
    self.assertIsInstance(motion_cfg, MotionCommandCfg)
    self.assertEqual(
      motion_cfg.actuator_command_lag_range,
      (TK3_COMMAND_DELAY_MIN_LAG, TK3_COMMAND_DELAY_MAX_LAG),
    )
    play_cfg = load_env_cfg("TK3-Tracking", play=True)
    play_motion_cfg = play_cfg.commands["motion"]
    self.assertIsInstance(play_motion_cfg, MotionCommandCfg)
    self.assertIsNone(play_motion_cfg.actuator_command_lag_range)

    class FakeActuator:
      def __init__(self, has_delay: bool) -> None:
        self.has_delay = has_delay
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.current_lags = torch.zeros(4, dtype=torch.long)

      def reset(self, env_ids: torch.Tensor) -> None:
        self.current_lags[env_ids] = 0

      def set_lags(self, lags: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.calls.append((lags.clone(), env_ids.clone()))
        self.current_lags[env_ids] = lags

    class FakeRobot:
      def __init__(self) -> None:
        self.actuators = [
          FakeActuator(has_delay=True),
          FakeActuator(has_delay=True),
          FakeActuator(has_delay=False),
        ]
        self.reset_calls: list[torch.Tensor] = []

      def reset(self, env_ids: torch.Tensor) -> None:
        self.reset_calls.append(env_ids.clone())
        for actuator in self.actuators:
          actuator.reset(env_ids)

    robot = FakeRobot()
    env_ids = torch.tensor((1, 3))
    command = SimpleNamespace(
      robot=robot,
      cfg=SimpleNamespace(
        actuator_command_lag_range=(3, 3),
      ),
    )

    MotionCommand._reset_robot_and_randomize_actuator_command_lag(
      command, env_ids
    )
    self.assertEqual(len(robot.reset_calls), 1)
    delayed_a, delayed_b, direct = robot.actuators
    self.assertEqual(len(delayed_a.calls), 1)
    self.assertEqual(len(delayed_b.calls), 1)
    self.assertEqual(len(direct.calls), 0)
    sampled_lags, sampled_env_ids = delayed_a.calls[0]
    torch.testing.assert_close(sampled_lags, delayed_b.calls[0][0])
    torch.testing.assert_close(sampled_env_ids, env_ids)
    torch.testing.assert_close(delayed_a.current_lags[env_ids], torch.tensor((3, 3)))

    # Reaching the end of the reference motion calls the same reset path and
    # intentionally samples a new lag for the next reference segment.
    command.cfg.actuator_command_lag_range = (1, 1)
    MotionCommand._reset_robot_and_randomize_actuator_command_lag(
      command, env_ids
    )
    self.assertEqual(len(robot.reset_calls), 2)
    torch.testing.assert_close(delayed_a.current_lags[env_ids], torch.tensor((1, 1)))

    command.cfg.actuator_command_lag_range = None
    MotionCommand._reset_robot_and_randomize_actuator_command_lag(
      command, env_ids
    )
    self.assertEqual(len(delayed_a.calls), 2)
    torch.testing.assert_close(delayed_a.current_lags[env_ids], torch.tensor((0, 0)))

  def test_tk3_base_com_applies_after_pseudo_inertia(self) -> None:
    cfg = load_env_cfg("TK3-Tracking")
    startup_names = [
      name for name, term_cfg in cfg.events.items() if term_cfg.mode == "startup"
    ]
    self.assertLess(
      startup_names.index("randomize_rigid_body_mass_others"),
      startup_names.index("base_com"),
    )
    self.assertEqual(
      cfg.events["base_com"].params["asset_cfg"].body_names,
      ("pelvis",),
    )

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
