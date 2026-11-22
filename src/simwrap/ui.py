"""Viser rendering and controls for a running SO-101 simulation."""

import threading
import time
from typing import Any

import mujoco
import numpy as np
import viser

from .sim import Sim


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply MuJoCo-order (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw))


class SimulationUI:
    """Create static geometry nodes and update their poses at about 30 Hz."""

    def __init__(self, sim: Sim, host: str = "0.0.0.0", port: int = 8080):
        """Start a viser server and build scene nodes from the simulation."""
        self.sim = sim
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")
        self.server.initial_camera.position = (1.1, -1.35, 1.35)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.78)
        self._nodes: list[tuple[Any, int, np.ndarray, np.ndarray]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._paused = False
        self._status = self.server.gui.add_markdown("Physics: running")
        self._pause_button = self.server.gui.add_button("Pause physics")
        self.server.gui.add_button("Reset").on_click(lambda _: self._reset())
        self._pause_button.on_click(lambda _: self._toggle_pause())
        for geom in sim.get_geom_visual_data():
            self._add_geom(geom)
        self.server.scene.add_grid("grid", width=2.0, height=2.0, position=(0, 0, 0))
        self.update_once()

    def _add_geom(self, geom: dict[str, object]) -> None:
        """Create a single viser node for one MuJoCo geom."""
        kind = int(geom["type"])
        size = np.asarray(geom["size"])
        pos = np.asarray(geom["pos"])
        quat = np.asarray(geom["quat"])
        rgba = np.asarray(geom["rgba"])
        color = tuple(np.clip(np.round(rgba[:3] * 255), 0, 255).astype(int))
        opacity = float(np.clip(rgba[3], 0, 1))
        name = f"geom/{geom['id']}_{geom['name']}"
        options = {"color": color, "opacity": opacity, "position": pos, "wxyz": quat}
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            node = self.server.scene.add_box(name, dimensions=2 * size, **options)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            node = self.server.scene.add_icosphere(name, radius=float(size[0]), **options)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            cylinder_quat = _quat_multiply(quat, np.array((np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0)))
            node = self.server.scene.add_cylinder(
                name, radius=float(size[0]), height=2*float(size[1]),
                color=color, opacity=opacity, position=pos, wxyz=cylinder_quat,
            )
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            vertices, faces = self._capsule_mesh(float(size[0]), float(size[1]))
            node = self.server.scene.add_mesh_simple(name, vertices, faces, color=color, opacity=opacity,
                                                      position=pos, wxyz=quat, side="double")
        elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
            sx, sy = size[:2]
            if sx <= 0: sx = 5.0
            if sy <= 0: sy = 5.0
            vertices = np.array(((-sx,-sy,0), (sx,-sy,0), (sx,sy,0), (-sx,sy,0)), dtype=float)
            faces = np.array(((0,1,2),(0,2,3)), dtype=np.uint32)
            node = self.server.scene.add_mesh_simple(name, vertices, faces, color=color, opacity=opacity,
                                                      position=pos, wxyz=quat, side="double")
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            # MuJoCo's compiled mesh vertices are already in model units;
            # geom_size describes its bounds and is not an additional scale.
            vertices = np.asarray(geom["vertices"])
            faces = np.asarray(geom["faces"], dtype=np.uint32)
            node = self.server.scene.add_mesh_simple(name, vertices, faces, color=color, opacity=opacity,
                                                      position=pos, wxyz=quat, side="double")
        else:
            return
        self._nodes.append((node, int(geom["body"]), pos, quat))

    @staticmethod
    def _capsule_mesh(radius: float, half_length: float) -> tuple[np.ndarray, np.ndarray]:
        """Build a z-aligned capsule mesh whose cylindrical half-length is given."""
        rings: list[tuple[float, float]] = []
        for i in range(9):
            angle = -np.pi / 2 + (np.pi / 2) * i / 8
            rings.append((radius * np.cos(angle), -half_length + radius * np.sin(angle)))
        for i in range(9):
            angle = (np.pi / 2) * i / 8
            rings.append((radius * np.cos(angle), half_length + radius * np.sin(angle)))
        sides = 16
        vertices = np.array([(r*np.cos(2*np.pi*j/sides), r*np.sin(2*np.pi*j/sides), z)
                             for r, z in rings for j in range(sides)])
        faces = []
        for i in range(len(rings)-1):
            for j in range(sides):
                a, b = i*sides+j, i*sides+(j+1)%sides
                faces.extend(((a,b,b+sides),(a,b+sides,a+sides)))
        return vertices, np.asarray(faces, dtype=np.uint32)

    def update_once(self) -> None:
        """Update node world poses using a consistent locked Sim snapshot."""
        body_pos, body_quat = self.sim.get_body_poses()
        for node, body_id, local_pos, local_quat in self._nodes:
            bp, bq = body_pos[body_id], body_quat[body_id]
            rotated = np.empty(3)
            mujoco.mju_rotVecQuat(rotated, local_pos, bq)
            node.position = bp + rotated
            node.wxyz = _quat_multiply(bq, local_quat)

    def start(self) -> None:
        """Start the 30 Hz scene update loop."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._render_loop, name="viser-render", daemon=True)
        self._thread.start()

    def _render_loop(self) -> None:
        while not self._stop_event.is_set():
            self.update_once()
            self._stop_event.wait(1 / 30)

    def _toggle_pause(self) -> None:
        if self._paused:
            self.sim.start()
            self._paused = False
            self._pause_button.label = "Pause physics"
            self._status.content = "Physics: running"
        else:
            self.sim.stop()
            self._paused = True
            self._pause_button.label = "Resume physics"
            self._status.content = "Physics: paused"

    def _reset(self) -> None:
        self.sim.reset()
        self._status.content = "Physics: paused" if self._paused else "Physics: running (reset)"

    def close(self) -> None:
        """Stop UI updates and close the server."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.server.stop()
