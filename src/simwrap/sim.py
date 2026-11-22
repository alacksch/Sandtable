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
        self._camera_renderer = None
        self._app_state = None
        # Compile the pool with collision pairs available, then park it inactive.
        # MuJoCo may omit candidate collision pairs for geoms disabled at compile time.
        for geom_id in range(self.model.ngeom):
            if (self.model.geom(geom_id).name or "").startswith("pool_"):
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0

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

    def create_renderer(self, width: int, height: int) -> mujoco.Renderer:
        """Create a renderer while protecting the model from concurrent access."""
        with self.lock:
            return mujoco.Renderer(self.model, width=width, height=height)

    def render_camera(self, renderer: mujoco.Renderer, camera: str) -> np.ndarray:
        """Render one named camera while physics state is locked."""
        with self.lock:
            renderer.update_scene(self.data, camera=camera)
            return renderer.render().copy()

    def set_camera_renderer(self, camera_renderer: object) -> None:
        """Register the camera source used by get_observation."""
        self._camera_renderer = camera_renderer

    def set_app_state(self, app_state: object) -> None:
        """Register the thread-safe UI state used for observation prompts."""
        self._app_state = app_state

    def get_observation(self) -> dict[str, object]:
        """Return rendered camera images, arm joint positions, and a prompt."""
        renderer = self._camera_renderer
        images = {} if renderer is None else renderer.get_images()
        state = self._app_state
        prompt = "" if state is None else state.snapshot()[0]
        return {"images": images, "joint_state": self.get_arm_state(), "prompt": prompt}

    def apply_action(self, joint_targets: np.ndarray) -> None:
        """Apply joint targets through the existing validated actuator interface."""
        self.set_ctrl(joint_targets)

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

    def get_object_pose(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one body's world pose while holding the simulation lock."""
        with self.lock:
            return self.data.xpos[body_id].copy(), self.data.xquat[body_id].copy()

    def move_object_slot(self, joint_id: int, geom_id: int, pos: np.ndarray,
                         quat: np.ndarray, commit: bool = True) -> None:
        """Commit a freejoint pose, reject interpenetration, and clear velocity.

        Placement is checked with MuJoCo's generated contacts after a temporary
        forward pass. Contacts with the table/floor are allowed; contacts with
        another pool object or the robot arm reject the placement.
        """
        position = np.asarray(pos, dtype=float)
        rotation = np.asarray(quat, dtype=float)
        if position.shape != (3,) or rotation.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("Object pose must contain finite position (3) and quaternion (4)")
        norm = np.linalg.norm(rotation)
        if norm == 0:
            raise ValueError("Quaternion must be nonzero")
        rotation /= norm
        with self.lock:
            qadr = int(self.model.jnt_qposadr[joint_id])
            vadr = int(self.model.jnt_dofadr[joint_id])
            old_pose = self.data.qpos[qadr:qadr + 7].copy()
            self.data.qpos[qadr:qadr + 3] = position
            self.data.qpos[qadr + 3:qadr + 7] = rotation
            mujoco.mj_forward(self.model, self.data)
            rejected = False
            for contact in self.data.contact[:self.data.ncon]:
                pair = (int(contact.geom1), int(contact.geom2))
                if geom_id not in pair:
                    continue
                other = pair[1] if pair[0] == geom_id else pair[0]
                body_name = self.model.body(int(self.model.geom_bodyid[other])).name or ""
                geom_name = self.model.geom(other).name or ""
                if body_name.startswith("pool_") or ("arm" in body_name.lower()) or ("robot" in body_name.lower()):
                    rejected = True
                    break
                # Robot asset names are not consistently prefixed; permit only
                # the explicit table and floor for supporting contacts.
                if geom_name not in {"table", "floor"} and body_name not in {"world", "ground"}:
                    rejected = True
                    break
            if rejected:
                self.data.qpos[qadr:qadr + 7] = old_pose
                mujoco.mj_forward(self.model, self.data)
                raise ValueError("Placement overlaps another object or the robot")
            if not commit:
                self.data.qpos[qadr:qadr + 7] = old_pose
                mujoco.mj_forward(self.model, self.data)
                return
            self.data.qvel[vadr:vadr + 6] = 0
            mujoco.mj_forward(self.model, self.data)

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
                    "body_name": self.model.body(int(self.model.geom_bodyid[geom_id])).name or "world",
                    "type": geom_type,
                    "size": self.model.geom_size[geom_id].copy(),
                    "pos": self.model.geom_pos[geom_id].copy(),
                    "quat": self.model.geom_quat[geom_id].copy(),
                    "rgba": self.model.geom_rgba[geom_id].copy(),
                    "is_pool": bool((self.model.geom(geom_id).name or "").startswith("pool_")),
                }
                if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh_id = int(self.model.geom_dataid[geom_id])
                    item["mesh_id"] = mesh_id
                    vert_start = int(self.model.mesh_vertadr[mesh_id])
                    vert_count = int(self.model.mesh_vertnum[mesh_id])
                    face_start = int(self.model.mesh_faceadr[mesh_id])
                    face_count = int(self.model.mesh_facenum[mesh_id])
                    item["vertices"] = self.model.mesh_vert[vert_start:vert_start + vert_count].copy()
                    item["faces"] = self.model.mesh_face[face_start:face_start + face_count].copy()
                result.append(item)
            return result

    def get_body_count(self) -> int:
        """Return the compiled body count under the simulation lock."""
        with self.lock:
            return int(self.model.nbody)

    def get_object_pool(self) -> list[dict[str, object]]:
        """Return pool slot identifiers and default geometry properties."""
        with self.lock:
            slots = []
            for body_id in range(self.model.nbody):
                body_name = self.model.body(body_id).name or ""
                if not body_name.startswith("pool_"):
                    continue
                geom_id = next((i for i in range(self.model.ngeom)
                                if int(self.model.geom_bodyid[i]) == body_id), None)
                joint_id = next((i for i in range(self.model.njnt)
                                 if int(self.model.jnt_bodyid[i]) == body_id), None)
                if geom_id is None or joint_id is None:
                    continue
                slots.append({"name": body_name, "kind": body_name.split("_")[1],
                              "body_id": body_id, "joint_id": joint_id, "geom_id": geom_id,
                              "size": self.model.geom_size[geom_id].copy(),
                              "rgba": self.model.geom_rgba[geom_id].copy()})
            return slots

    def activate_object_slot(self, joint_id: int, geom_id: int, pos: np.ndarray,
                             quat: np.ndarray, size: np.ndarray | None = None,
                             color: np.ndarray | None = None) -> None:
        """Set an object freejoint pose and enable its collisions atomically."""
        position = np.asarray(pos, dtype=float)
        rotation = np.asarray(quat, dtype=float)
        if position.shape != (3,) or rotation.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("Object pose must contain finite position (3) and quaternion (4)")
        norm = np.linalg.norm(rotation)
        if norm == 0:
            raise ValueError("Quaternion must be nonzero")
        rotation = rotation / norm
        with self.lock:
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError("Object pool joint must be a freejoint")
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            qvel_adr = int(self.model.jnt_dofadr[joint_id])
            self.data.qpos[qpos_adr:qpos_adr + 3] = position
            self.data.qpos[qpos_adr + 3:qpos_adr + 7] = rotation
            self.data.qvel[qvel_adr:qvel_adr + 6] = 0
            if size is not None:
                self.model.geom_size[geom_id] = np.asarray(size, dtype=float)
            if color is not None:
                self.model.geom_rgba[geom_id] = np.asarray(color, dtype=float)
            self.model.geom_contype[geom_id] = 1
            self.model.geom_conaffinity[geom_id] = 1
            mujoco.mj_forward(self.model, self.data)

    def deactivate_object_slot(self, joint_id: int, geom_id: int, parked_pos: np.ndarray) -> None:
        """Park an object slot and disable its collisions atomically."""
        with self.lock:
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            qvel_adr = int(self.model.jnt_dofadr[joint_id])
            self.data.qpos[qpos_adr:qpos_adr + 3] = parked_pos
            self.data.qpos[qpos_adr + 3:qpos_adr + 7] = (1, 0, 0, 0)
            self.data.qvel[qvel_adr:qvel_adr + 6] = 0
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
            mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        """Stop the physics thread."""
        self.stop()
