"""Thread-safe fixed-rate MuJoCo simulation wrapper."""

import threading
import time
from typing import Optional

import mujoco
import numpy as np

from .scene import build_scene


class Sim:
    """Own a MuJoCo model and data, serialized through a single lock."""

    def __init__(self, timestep: float = 0.002, model: Optional[mujoco.MjModel] = None):
        """Create simulation state with the requested physics timestep."""
        self.model = model if model is not None else build_scene()
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._initial_qpos = self.data.qpos.copy()

    def start(self) -> None:
        """Start the fixed-rate physics thread if it is not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._step_loop, name="mujoco-physics", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop and join the physics thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _step_loop(self) -> None:
        dt = self.model.opt.timestep
        deadline = time.perf_counter()
        while not self._stop_event.is_set():
            with self.lock:
                mujoco.mj_step(self.model, self.data)
            deadline += dt
            self._stop_event.wait(max(0.0, deadline - time.perf_counter()))

    def step(self, count: int = 1) -> None:
        """Advance physics synchronously by a number of steps."""
        with self.lock:
            for _ in range(count):
                mujoco.mj_step(self.model, self.data)

    def get_arm_state(self) -> np.ndarray:
        """Return actuator-controlled joint positions as a copy."""
        with self.lock:
            addresses = [self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]] for i in range(self.model.nu)]
            return self.data.qpos[addresses].copy()

    def get_ctrl_ranges(self) -> np.ndarray:
        """Return actuator control ranges as a copy."""
        with self.lock:
            return self.model.actuator_ctrlrange.copy()

    def get_actuator_count(self) -> int:
        """Return the number of actuators."""
        with self.lock:
            return int(self.model.nu)

    def set_ctrl(self, targets: np.ndarray) -> None:
        """Set actuator targets, clipping them to configured control ranges."""
        values = np.asarray(targets, dtype=float)
        with self.lock:
            if values.shape != (self.model.nu,):
                raise ValueError(f"Expected {self.model.nu} actuator targets, got {values.shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError("Actuator targets must be finite")
            if np.any(self.model.actuator_ctrllimited):
                for i in range(self.model.nu):
                    if self.model.actuator_ctrllimited[i]:
                        values[i] = np.clip(values[i], *self.model.actuator_ctrlrange[i])
            self.data.ctrl[:] = values

    def reset(self) -> None:
        """Restore the initial state and clear controls and velocities."""
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self._initial_qpos
            self.data.ctrl[:] = 0
            mujoco.mj_forward(self.model, self.data)

    def get_body_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of all body positions and quaternions."""
        with self.lock:
            return self.data.xpos.copy(), self.data.xquat.copy()

    def get_geom_visual_data(self) -> list[dict[str, object]]:
        """Return renderable geom descriptors and mesh data as owned copies."""
        with self.lock:
            result: list[dict[str, object]] = []
            for geom_id in range(self.model.ngeom):
                group = int(self.model.geom_group[geom_id])
                # This SO-101 asset assigns visual geoms to group 2 and
                # collision geoms to group 3. Keep standard scene geoms too.
                if group == 3:
                    continue
                geom_type = int(self.model.geom_type[geom_id])
                item: dict[str, object] = {
                    "id": geom_id,
                    "name": self.model.geom(geom_id).name or f"geom_{geom_id}",
                    "body": int(self.model.geom_bodyid[geom_id]),
                    "type": geom_type,
                    "size": self.model.geom_size[geom_id].copy(),
                    "pos": self.model.geom_pos[geom_id].copy(),
                    "quat": self.model.geom_quat[geom_id].copy(),
                    "rgba": self.model.geom_rgba[geom_id].copy(),
                }
                if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh_id = int(self.model.geom_dataid[geom_id])
                    vert_start = int(self.model.mesh_vertadr[mesh_id])
                    vert_count = int(self.model.mesh_vertnum[mesh_id])
                    face_start = int(self.model.mesh_faceadr[mesh_id])
                    face_count = int(self.model.mesh_facenum[mesh_id])
                    item["vertices"] = self.model.mesh_vert[vert_start:vert_start + vert_count].copy()
                    item["faces"] = self.model.mesh_face[face_start:face_start + face_count].copy()
                result.append(item)
            return result

    def close(self) -> None:
        """Stop the physics thread."""
        self.stop()
