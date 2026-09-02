"""Regression tests for TK3 XML foot collision geometry."""

from __future__ import annotations

import pickle
import unittest

import mujoco
import numpy as np
from mjlab.entity import EntityCfg
from mjlab.entity.entity import Entity

from src.assets.robots.tiangong3.tk3_constants import (
  TK3_BASE_HEIGHT,
  get_tk3_robot_cfg,
)
from src.assets.robots.tiangong3.tk3_constants import get_spec as get_tk3_spec

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
MUJOCO_DEFAULT_SOLREF = (0.02, 1.0)
MUJOCO_DEFAULT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)


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
  if geom_type != mujoco.mjtGeom.mjGEOM_CYLINDER:
    raise AssertionError(f"Unsupported TK3 foot geom type: {geom_type}")

  rotation = data.geom_xmat[geom_id].reshape(3, 3)
  axis = rotation[:, 2]
  radius, half_length = model.geom_size[geom_id, :2]
  radial_z = radius * np.sqrt(max(0.0, 1.0 - axis[2] ** 2))
  return float(
    data.geom_xpos[geom_id, 2]
    - abs(axis[2]) * half_length
    - radial_z
  )


class Tk3FootConfigurationTest(unittest.TestCase):
  def test_robot_uses_xml_cylinder_feet(self) -> None:
    model = Entity(get_tk3_robot_cfg()).spec.compile()

    self.assertEqual(_foot_geom_names(model), XML_FOOT_GEOMS)
    self.assertTrue(SOLE_FOOT_GEOMS.isdisjoint(_foot_geom_names(model)))
    self.assertTrue(SOLE_MESHES.isdisjoint(_mesh_names(model)))

  def test_spec_builder_uses_xml_feet(self) -> None:
    self.assertEqual(_foot_geom_names(get_tk3_spec().compile()), XML_FOOT_GEOMS)

  def test_foot_contact_parameters_use_mujoco_defaults(self) -> None:
    model = Entity(get_tk3_robot_cfg()).spec.compile()

    for geom_name in XML_FOOT_GEOMS:
      geom_id = model.geom(geom_name).id
      np.testing.assert_allclose(
        model.geom_solref[geom_id], MUJOCO_DEFAULT_SOLREF
      )
      np.testing.assert_allclose(
        model.geom_solimp[geom_id], MUJOCO_DEFAULT_SOLIMP
      )

  def test_home_pose_clears_ground(self) -> None:
    model, data = _compile_home_with_ground(get_tk3_robot_cfg())
    foot_geom_ids = tuple(
      geom_id
      for geom_id in range(model.ngeom)
      if model.geom(geom_id).name.startswith("foot_")
    )
    clearance = min(
      _geom_minimum_z(model, data, geom_id) for geom_id in foot_geom_ids
    )

    self.assertAlmostEqual(model.key_qpos[0, 2], TK3_BASE_HEIGHT)
    self.assertGreaterEqual(clearance, 0.0)
    self.assertLess(clearance, 0.002)

    ground_id = model.geom("ground").id
    ground_contacts = (
      contact
      for contact in data.contact
      if ground_id in (contact.geom1, contact.geom2)
    )
    self.assertTrue(all(contact.dist >= 0.0 for contact in ground_contacts))

  def test_robot_cfg_is_picklable_and_builds_fresh_specs(self) -> None:
    restored_cfg = pickle.loads(pickle.dumps(get_tk3_robot_cfg()))
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


if __name__ == "__main__":
  unittest.main()
