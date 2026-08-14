"""MuJoCo specification builder for the TienKung 3 robot."""

from pathlib import Path

import mujoco

from src import SRC_PATH

TK3_XML: Path = (
  SRC_PATH / "assets" / "robots" / "tiangong3" / "xmls" / "tiangong3.xml"
)
TK3_MESH_DIR: Path = TK3_XML.parent.parent / "meshes"
TK3_SPEC_CONFIG: Path = Path(__file__)

# The compiled HOME pose has 0.65 mm of clearance at the lowest convex-sole
# vertex.  Smaller values penetrate the ground because the right sole is
# slightly lower than the left sole in the nominal joint pose.
TK3_BASE_HEIGHT = 1.0
TK3_NOMINAL_FOOT_GROUND_FRICTION = 1.0

assert TK3_XML.exists()
assert TK3_MESH_DIR.exists()

# Convex outlines extracted from the lower 10.5 mm of each visual foot mesh,
# then simplified to a 2.5 mm contour tolerance. Coordinates are in the
# corresponding ankle-roll body frame.
_SOLE_CONTOURS = {
  "left": (
    (-0.071965, -0.041471),
    (0.086721, -0.055401),
    (0.137475, -0.045930),
    (0.152276, -0.039385),
    (0.174362, -0.016440),
    (0.177666, 0.008754),
    (0.165341, 0.028921),
    (0.144226, 0.043401),
    (0.087627, 0.055387),
    (-0.063796, 0.045695),
    (-0.085748, 0.034165),
    (-0.098498, 0.010363),
    (-0.098935, -0.013036),
    (-0.087035, -0.032691),
  ),
  "right": (
    (-0.069665, -0.042151),
    (0.085894, -0.055633),
    (0.128261, -0.048972),
    (0.145510, -0.042607),
    (0.164472, -0.029437),
    (0.174290, -0.016319),
    (0.175848, 0.013808),
    (0.166686, 0.027070),
    (0.144226, 0.043397),
    (0.084354, 0.055620),
    (-0.070622, 0.042058),
    (-0.085097, 0.034857),
    (-0.098817, 0.011586),
    (-0.095766, -0.018474),
    (-0.082391, -0.036664),
  ),
}
_SOLE_BOTTOM_Z = -0.058
_SOLE_TOP_Z = -0.048
_SOLE_BEVEL = 0.002


def _cross_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
  return a[0] * b[1] - a[1] * b[0]


def _inset_convex_contour(
  contour: tuple[tuple[float, float], ...],
  inset: float,
) -> tuple[tuple[float, float], ...]:
  """Offset a counter-clockwise convex contour inward by ``inset``."""
  result: list[tuple[float, float]] = []
  count = len(contour)
  for index in range(count):
    previous = contour[index - 1]
    current = contour[index]
    following = contour[(index + 1) % count]
    previous_edge = (
      current[0] - previous[0],
      current[1] - previous[1],
    )
    next_edge = (
      following[0] - current[0],
      following[1] - current[1],
    )
    previous_length = (
      previous_edge[0] ** 2 + previous_edge[1] ** 2
    ) ** 0.5
    next_length = (next_edge[0] ** 2 + next_edge[1] ** 2) ** 0.5
    previous_normal = (
      -previous_edge[1] / previous_length,
      previous_edge[0] / previous_length,
    )
    next_normal = (
      -next_edge[1] / next_length,
      next_edge[0] / next_length,
    )
    previous_offset = (
      previous[0] + inset * previous_normal[0],
      previous[1] + inset * previous_normal[1],
    )
    next_offset = (
      current[0] + inset * next_normal[0],
      current[1] + inset * next_normal[1],
    )
    offset_delta = (
      next_offset[0] - previous_offset[0],
      next_offset[1] - previous_offset[1],
    )
    denominator = _cross_2d(previous_edge, next_edge)
    if abs(denominator) < 1.0e-12:
      result.append(
        (
          current[0] + inset * next_normal[0],
          current[1] + inset * next_normal[1],
        )
      )
      continue
    distance = _cross_2d(offset_delta, next_edge) / denominator
    result.append(
      (
        previous_offset[0] + distance * previous_edge[0],
        previous_offset[1] + distance * previous_edge[1],
      )
    )
  return tuple(result)


def _beveled_sole_vertices(
  contour: tuple[tuple[float, float], ...],
) -> list[float]:
  """Build four convex rings for a 10 mm sole with 2 mm edge bevels."""
  inset = _inset_convex_contour(contour, _SOLE_BEVEL)
  lower_bevel_z = _SOLE_BOTTOM_Z + _SOLE_BEVEL
  upper_bevel_z = _SOLE_TOP_Z - _SOLE_BEVEL
  rings = (
    (inset, _SOLE_BOTTOM_Z),
    (contour, lower_bevel_z),
    (contour, upper_bevel_z),
    (inset, _SOLE_TOP_Z),
  )
  return [
    coordinate
    for ring, z_position in rings
    for x_position, y_position in ring
    for coordinate in (x_position, y_position, z_position)
  ]


def _replace_xml_feet_with_convex_soles(spec: mujoco.MjSpec) -> None:
  for side in ("left", "right"):
    mesh_name = f"tk3_{side}_sole_collision"
    spec.add_mesh(
      name=mesh_name,
      uservert=_beveled_sole_vertices(_SOLE_CONTOURS[side]),
      maxhullvert=64,
    )
    body = spec.body(f"ankle_roll_{side[0]}_link")
    for geom in list(body.geoms):
      if geom.name.startswith(f"foot_{side}_"):
        spec.delete(geom)
    body.add_geom(
      name=f"foot_{side}_sole",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh_name,
      contype=1,
      conaffinity=1,
      condim=3,
      priority=2,
      friction=(TK3_NOMINAL_FOOT_GROUND_FRICTION, 0.005, 0.0001),
      density=0.0,
      group=3,
      rgba=(0.15, 0.15, 0.15, 1.0),
    )


def get_spec(*, convex_sole: bool = True) -> mujoco.MjSpec:
  """Return a fresh TK3 spec with either convex or original XML feet."""
  spec = mujoco.MjSpec.from_file(str(TK3_XML))
  if convex_sole:
    _replace_xml_feet_with_convex_soles(spec)
  return spec
