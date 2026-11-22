"""Construction of the SO-101 simulation scene."""

from pathlib import Path

import mujoco


TABLE_SURFACE_HEIGHT = 0.75
TABLE_THICKNESS = 0.05
ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = ROOT / "assets" / "so101" / "scene.xml"


def build_scene() -> mujoco.MjModel:
    """Build and compile the robot, table, floor, lighting, and cameras."""
    spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    world = spec.worldbody
    spec.body("base").pos[2] = TABLE_SURFACE_HEIGHT
    world.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, TABLE_SURFACE_HEIGHT - TABLE_THICKNESS / 2),
        size=(0.6, 0.45, TABLE_THICKNESS / 2), rgba=(0.42, 0.27, 0.16, 1),
    )
    world.add_light(name="key_light", pos=(0.0, -1.0, 2.5), dir=(0.0, 0.2, -1.0))
    world.add_camera(name="front", pos=(1.4, -1.8, 1.25), xyaxes=(0.79, 0.61, 0, -0.23, 0.30, 0.93), fovy=45)
    world.add_camera(name="top", pos=(0.0, 0.0, 2.5), xyaxes=(1, 0, 0, 0, 1, 0), fovy=50)
    return spec.compile()
