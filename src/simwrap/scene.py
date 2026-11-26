"""Construction of the SO-101 simulation scene."""

from pathlib import Path

import mujoco


TABLE_SURFACE_HEIGHT = 0.75
TABLE_THICKNESS = 0.05
ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = ROOT / "assets" / "so101" / "scene.xml"
OBJECT_POOL_COUNTS = {"box": 10, "cylinder": 10, "sphere": 10}
OBJECT_POOL_SIZES = {
    "box": (0.035, 0.035, 0.035),
    "cylinder": (0.03, 0.04),
    "sphere": (0.035,),
}
OBJECT_POOL_COLORS = {
    "box": (0.85, 0.25, 0.2, 1.0),
    "cylinder": (0.2, 0.65, 0.3, 1.0),
    "sphere": (0.2, 0.4, 0.9, 1.0),
}


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
    spec.body("gripper").add_camera(name="wrist", pos=(0.0, 0.025, 0.075), fovy=55)
    for kind, count in OBJECT_POOL_COUNTS.items():
        for index in range(count):
            name = f"pool_{kind}_{index}"
            body = world.add_body(name=name, pos=(20.0 + index * 0.3, 20.0 + list(OBJECT_POOL_COUNTS).index(kind) * 2.0, -10.0))
            body.add_freejoint(name=f"{name}_free")
            geom_type = {
                "box": mujoco.mjtGeom.mjGEOM_BOX,
                "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            }[kind]
            body.add_geom(
                name=f"{name}_geom", type=geom_type, size=OBJECT_POOL_SIZES[kind],
                rgba=OBJECT_POOL_COLORS[kind], contype=1, conaffinity=1,
            )
    return spec.compile()
