"""Regression tests for selectable TK3 foot collision geometry."""

from __future__ import annotations

import importlib.util
import pickle
import sys
import unittest
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import EntityCfg
from mjlab.entity.entity import Entity

from src.assets.robots.tiangong3.tk3_constants import (
  TK3_ARTICULATION,
  get_tk3_robot_cfg,
)
from src.assets.robots.tiangong3.tk3_constants import get_spec as get_tk3_spec
from src.assets.robots.tiangong3.tk3_constants_ghost import (
  TK3_ARTICULATION as TK3_GHOST_ARTICULATION,
)
from src.assets.robots.tiangong3.tk3_constants_ghost import (
  get_spec as get_tk3_ghost_spec,
)
from src.assets.robots.tiangong3.tk3_constants_ghost import (
  get_tk3_robot_cfg as get_tk3_ghost_robot_cfg,
)
from src.assets.robots.tiangong3.tk3_selection import select_tk3_robot_cfg
from src.assets.robots.tiangong3.tk3_spec import (
  TK3_BASE_HEIGHT,
  TK3_CONVEX_SOLE_SOLIMP,
  TK3_CONVEX_SOLE_SOLREF,
)

_SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"


def _load_script_config(script_name: str, config_name: str) -> type:
  module_name = f"_tk3_{script_name}_config"
  spec = importlib.util.spec_from_file_location(
    module_name, _SCRIPTS_DIR / f"{script_name}.py"
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return getattr(module, config_name)


PlayConfig = _load_script_config("play", "PlayConfig")
TrainConfig = _load_script_config("train", "TrainConfig")

RobotCfgFactory = Callable[..., EntityCfg]
ROBOT_CFG_FACTORIES = (get_tk3_robot_cfg, get_tk3_ghost_robot_cfg)

XML_FOOT_GEOMS = {
  f"foot_{side}_{part}_{position}"
  for side in ("left", "right")
  for part, position in (
    ("front", "outer"),
    ("front", "inner"),
    ("strip", "outer"),
    ("strip", "inner"),
  )
}
SOLE_FOOT_GEOMS = {"foot_left_sole", "foot_right_sole"}
SOLE_MESHES = {"tk3_left_sole_collision", "tk3_right_sole_collision"}
XML_FOOT_SOLREF = (0.02, 1.0)
XML_FOOT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)


def _foot_geom_names(model: mujoco.MjModel) -> set[str]:
  return {
    name
    for geom_id in range(model.ngeom)
    if (name := model.geom(geom_id).name).startswith("foot_")
  }


def _mesh_names(model: mujoco.MjModel) -> set[str]:
  return {model.mesh(mesh_id).name for mesh_id in range(model.nmesh)}


def _compile_home_with_ground(
  robot_cfg: EntityCfg,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
  spec = Entity(robot_cfg).spec
  spec.worldbody.add_geom(
    name="ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(1.0, 1.0, 0.1),
    contype=1,
    conaffinity=1,
  )
  model = spec.compile()
  data = mujoco.MjData(model)
  data.qpos[:] = model.key_qpos[0]
  mujoco.mj_forward(model, data)
  return model, data


def _geom_minimum_z(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  geom_id: int,
) -> float:
  geom_type = model.geom_type[geom_id]
  if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
    mesh_id = model.geom_dataid[geom_id]
    vertex_start = model.mesh_vertadr[mesh_id]
    vertex_end = vertex_start + model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[vertex_start:vertex_end]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    world_vertices = data.geom_xpos[geom_id] + vertices @ rotation.T
    return float(world_vertices[:, 2].min())

  if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    axis = rotation[:, 2]
    radius, half_length = model.geom_size[geom_id, :2]
    radial_z = radius * np.sqrt(max(0.0, 1.0 - axis[2] ** 2))
    return float(
      data.geom_xpos[geom_id, 2]
      - abs(axis[2]) * half_length
      - radial_z
    )

  raise AssertionError(f"Unsupported TK3 foot geom type: {geom_type}")


class Tk3FootConfigurationTest(unittest.TestCase):
  def test_script_configs_preserve_registered_foot_mode_by_default(self) -> None:
    train_foot = next(field for field in fields(TrainConfig) if field.name == "foot")
    play_foot = next(field for field in fields(PlayConfig) if field.name == "foot")

    self.assertIsNone(train_foot.default)
    self.assertIsNone(play_foot.default)

  def test_no_argument_factories_preserve_legacy_foot_modes(self) -> None:
    production_model = Entity(get_tk3_robot_cfg()).spec.compile()
    ghost_model = Entity(get_tk3_ghost_robot_cfg()).spec.compile()

    self.assertEqual(_foot_geom_names(production_model), XML_FOOT_GEOMS)
    self.assertEqual(_foot_geom_names(ghost_model), SOLE_FOOT_GEOMS)

  def test_public_spec_builders_preserve_legacy_defaults_and_switching(self) -> None:
    self.assertEqual(_foot_geom_names(get_tk3_spec().compile()), XML_FOOT_GEOMS)
    self.assertEqual(
      _foot_geom_names(get_tk3_ghost_spec().compile()), SOLE_FOOT_GEOMS
    )
    self.assertEqual(
      _foot_geom_names(get_tk3_spec(convex_sole=True).compile()),
      SOLE_FOOT_GEOMS,
    )
    self.assertEqual(
      _foot_geom_names(get_tk3_ghost_spec(convex_sole=False).compile()),
      XML_FOOT_GEOMS,
    )

  def test_foot_modes_build_expected_geometry(self) -> None:
    for robot_cfg_factory in ROBOT_CFG_FACTORIES:
      with self.subTest(robot_cfg_factory=robot_cfg_factory.__module__):
        xml_model = Entity(
          robot_cfg_factory(convex_sole=False)
        ).spec.compile()
        sole_model = Entity(
          robot_cfg_factory(convex_sole=True)
        ).spec.compile()

        self.assertEqual(_foot_geom_names(xml_model), XML_FOOT_GEOMS)
        self.assertEqual(_foot_geom_names(sole_model), SOLE_FOOT_GEOMS)
        self.assertTrue(SOLE_MESHES.isdisjoint(_mesh_names(xml_model)))
        self.assertTrue(SOLE_MESHES <= _mesh_names(sole_model))

  def test_sole_contact_parameters_do_not_override_xml_feet(self) -> None:
    for robot_cfg_factory in ROBOT_CFG_FACTORIES:
      with self.subTest(robot_cfg_factory=robot_cfg_factory.__module__):
        xml_model = Entity(
          robot_cfg_factory(convex_sole=False)
        ).spec.compile()
        sole_model = Entity(
          robot_cfg_factory(convex_sole=True)
        ).spec.compile()

        for geom_name in XML_FOOT_GEOMS:
          geom_id = xml_model.geom(geom_name).id
          np.testing.assert_allclose(
            xml_model.geom_solref[geom_id], XML_FOOT_SOLREF
          )
          np.testing.assert_allclose(
            xml_model.geom_solimp[geom_id], XML_FOOT_SOLIMP
          )
        for geom_name in SOLE_FOOT_GEOMS:
          geom_id = sole_model.geom(geom_name).id
          np.testing.assert_allclose(
            sole_model.geom_solref[geom_id], TK3_CONVEX_SOLE_SOLREF
          )
          np.testing.assert_allclose(
            sole_model.geom_solimp[geom_id], TK3_CONVEX_SOLE_SOLIMP
          )

  def test_home_pose_clears_ground(self) -> None:
    for robot_cfg_factory in ROBOT_CFG_FACTORIES:
      for convex_sole, maximum_clearance in ((False, 0.002), (True, 0.001)):
        with self.subTest(
          robot_cfg_factory=robot_cfg_factory.__module__,
          convex_sole=convex_sole,
        ):
          model, data = _compile_home_with_ground(
            robot_cfg_factory(convex_sole=convex_sole)
          )
          foot_geom_ids = tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom(geom_id).name.startswith("foot_")
          )
          clearance = min(
            _geom_minimum_z(model, data, geom_id)
            for geom_id in foot_geom_ids
          )

          self.assertAlmostEqual(model.key_qpos[0, 2], TK3_BASE_HEIGHT)
          self.assertGreaterEqual(clearance, 0.0)
          self.assertLess(clearance, maximum_clearance)

          ground_id = model.geom("ground").id
          ground_contacts = (
            contact
            for contact in data.contact
            if ground_id in (contact.geom1, contact.geom2)
          )
          self.assertTrue(
            all(contact.dist >= 0.0 for contact in ground_contacts)
          )

  def test_robot_cfg_is_picklable_and_builds_fresh_specs(self) -> None:
    for robot_cfg_factory in ROBOT_CFG_FACTORIES:
      for convex_sole in (False, True):
        with self.subTest(
          robot_cfg_factory=robot_cfg_factory.__module__,
          convex_sole=convex_sole,
        ):
          restored_cfg = pickle.loads(
            pickle.dumps(robot_cfg_factory(convex_sole=convex_sole))
          )
          first_spec = restored_cfg.spec_fn()
          second_spec = restored_cfg.spec_fn()

          self.assertIsNot(first_spec, second_spec)
          first_spec.worldbody.add_geom(
            name="first_spec_only",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.01, 0.0, 0.0),
          )
          self.assertEqual(
            first_spec.compile().ngeom,
            second_spec.compile().ngeom + 1,
          )

  def test_task_selector_preserves_variant_and_non_tk3_default(self) -> None:
    production = select_tk3_robot_cfg("TK3-Tracking", foot="sole")
    ghost = select_tk3_robot_cfg("TK3-Ghost-Tracking", foot="xml")

    self.assertIsNone(select_tk3_robot_cfg("TK3-Tracking", foot=None))
    self.assertIsNone(select_tk3_robot_cfg("TK3-Ghost-Tracking", foot=None))
    self.assertIsNone(select_tk3_robot_cfg("Unitree-G1-Flat", foot=None))
    self.assertIsNotNone(production)
    assert production is not None
    self.assertIs(production.articulation, TK3_ARTICULATION)
    self.assertEqual(
      _foot_geom_names(Entity(production).spec.compile()),
      SOLE_FOOT_GEOMS,
    )
    self.assertIsNotNone(ghost)
    assert ghost is not None
    self.assertIs(ghost.articulation, TK3_GHOST_ARTICULATION)
    self.assertEqual(
      _foot_geom_names(Entity(ghost).spec.compile()),
      XML_FOOT_GEOMS,
    )
    for foot in ("xml", "sole"):
      with self.subTest(foot=foot), self.assertRaisesRegex(
        ValueError, "only supported for TK3"
      ):
        select_tk3_robot_cfg("Unitree-G1-Flat", foot=foot)


if __name__ == "__main__":
  unittest.main()
