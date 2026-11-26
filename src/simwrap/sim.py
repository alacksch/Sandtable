"""Thread-safe fixed-rate MuJoCo simulation wrapper."""

import threading
import time
from typing import Optional

import mujoco
import numpy as np

from .scene import build_scene
from .policy import PolicyObservation, validate_action_targets


class Sim:
    """Own a MuJoCo model and data, serialized through a single lock."""

    def __init__(self, timestep: float = 0.002, model: Optional[mujoco.MjModel] = None):
        """Create simulation state with the requested physics timestep."""
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError("Simulation timestep must be positive and finite")
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._worker_error: Exception | None = None
        self._camera_renderer = None
        self._app_state = None
        with self._lock:
            self._model = model if model is not None else build_scene()
            self._model.opt.timestep = timestep
            self._data = mujoco.MjData(self._model)
            self._initial_qpos = self._data.qpos.copy()
            self._timestep = float(timestep)
            # Compile the pool with collision pairs available, then park it inactive.
            # MuJoCo may omit candidate collision pairs for geoms disabled at compile time.
            for geom_id in range(self._model.ngeom):
                if (self._model.geom(geom_id).name or "").startswith("pool_"):
                    self._model.geom_contype[geom_id] = 0
                    self._model.geom_conaffinity[geom_id] = 0

    def start(self) -> None:
        """Start the fixed-rate physics thread if it is not already running."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Simulation is closed")
            if self._thread is not None and self._thread.is_alive():
                return
            self._worker_error = None
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._step_loop, name="mujoco-physics", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop and join the physics thread."""
        with self._lifecycle_lock:
            self._stop_event.set()
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join()
            if self._thread is thread:
                self._thread = None

    def _step_loop(self) -> None:
        deadline = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    mujoco.mj_step(self._model, self._data)
                deadline += self._timestep
                now = time.perf_counter()
                if now - deadline > self._timestep:
                    # Drop missed wall-clock deadlines rather than busy-catching up.
                    deadline = now
                self._stop_event.wait(max(0.0, deadline - time.perf_counter()))
        except Exception as exc:
            self._worker_error = exc
            self._stop_event.set()

    def get_worker_error(self) -> str | None:
        """Return the last physics worker error, if the worker stopped unexpectedly."""
        with self._lifecycle_lock:
            return None if self._worker_error is None else repr(self._worker_error)

    def get_state_snapshot(self) -> dict[str, np.ndarray]:
        """Return owned copies of core state arrays for diagnostics and tests."""
        with self._lock:
            return {"qpos": self._data.qpos.copy(), "qvel": self._data.qvel.copy(),
                    "xpos": self._data.xpos.copy(), "ctrl": self._data.ctrl.copy()}

    def get_simulation_time(self) -> float:
        """Return the current simulation time in seconds."""
        with self._lock:
            return float(self._data.time)

    def step(self, count: int = 1) -> None:
        """Advance physics synchronously by a number of steps."""
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("Step count must be a non-negative integer")
        with self._lock:
            for _ in range(count):
                mujoco.mj_step(self._model, self._data)

    def get_arm_state(self) -> np.ndarray:
        """Return actuator-controlled joint positions as a copy."""
        return self.get_arm_state_snapshot()[0]

    def get_arm_state_snapshot(self) -> tuple[np.ndarray, float]:
        """Return ordered actuator joint positions and simulation time atomically."""
        with self._lock:
            addresses = [self._model.jnt_qposadr[self._model.actuator_trnid[i, 0]] for i in range(self._model.nu)]
            return self._data.qpos[addresses].copy(), float(self._data.time)

    def get_arm_control_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return actual joint positions and commanded targets as owned copies."""
        with self._lock:
            addresses = [self._model.jnt_qposadr[self._model.actuator_trnid[i, 0]]
                         for i in range(self._model.nu)]
            return self._data.qpos[addresses].copy(), self._data.ctrl.copy()

    def get_actuator_info(self) -> list[dict[str, object]]:
        """Return actuator order, joint names, units, and control limits."""
        with self._lock:
            info = []
            for actuator_id in range(self._model.nu):
                joint_id = int(self._model.actuator_trnid[actuator_id, 0])
                joint_type = self._model.jnt_type[joint_id]
                unit = "rad" if joint_type == mujoco.mjtJoint.mjJNT_HINGE else (
                    "m" if joint_type == mujoco.mjtJoint.mjJNT_SLIDE else "unknown")
                info.append({
                    "index": actuator_id,
                    "actuator_name": self._model.actuator(actuator_id).name or f"actuator_{actuator_id}",
                    "joint_name": self._model.joint(joint_id).name or f"joint_{joint_id}",
                    "joint_type": int(joint_type),
                    "unit": unit,
                    "limited": bool(self._model.actuator_ctrllimited[actuator_id]),
                    "control_range": self._model.actuator_ctrlrange[actuator_id].copy(),
                })
            return info

    def create_renderer(self, width: int, height: int) -> mujoco.Renderer:
        """Create a renderer while protecting the model from concurrent access."""
        with self._lock:
            return mujoco.Renderer(self._model, width=width, height=height)

    def render_camera(self, renderer: mujoco.Renderer, camera: str) -> np.ndarray:
        """Prepare one named view under lock, then draw its owned scene snapshot."""
        with self._lock:
            renderer.update_scene(self._data, camera=camera)
        # MuJoCo's RGB draw consumes the renderer's prepared scene and GL context,
        # not MjModel/MjData. The camera worker exclusively owns this renderer.
        return renderer.render().copy()

    def render_cameras(self, renderer: mujoco.Renderer, cameras: tuple[str, ...]) -> dict[str, np.ndarray]:
        """Render each named view from a synchronized, individually prepared snapshot."""
        return {name: self.render_camera(renderer, name) for name in cameras}

    def set_camera_renderer(self, camera_renderer: object) -> None:
        """Register the camera source used by get_observation."""
        self._camera_renderer = camera_renderer

    def set_app_state(self, app_state: object) -> None:
        """Register the thread-safe UI state used for observation prompts."""
        self._app_state = app_state

    def get_observation(self) -> PolicyObservation:
        """Return named RGB camera images, ordered joint state, and prompt metadata."""
        renderer = self._camera_renderer
        images = {} if renderer is None else renderer.get_images()
        state = self._app_state
        prompt = "" if state is None else state.snapshot()[0]
        actuator_info = self.get_actuator_info()
        joint_state, simulation_time = self.get_arm_state_snapshot()
        return {"images": images, "joint_state": joint_state, "prompt": prompt,
                "joint_names": [str(item["joint_name"]) for item in actuator_info],
                "joint_units": [str(item["unit"]) for item in actuator_info],
                "simulation_time": simulation_time}

    def apply_action(self, joint_targets: np.ndarray, *, observation_time: float | None = None,
                     max_observation_age: float = 0.25) -> None:
        """Apply valid policy targets, optionally rejecting stale observations."""
        if not np.isfinite(max_observation_age) or max_observation_age < 0:
            raise ValueError("Maximum observation age must be finite and non-negative")
        values = validate_action_targets(joint_targets, self.get_actuator_info())
        with self._lock:
            if values.shape != (self._model.nu,):
                raise ValueError("Actuator count changed while applying policy targets")
            if observation_time is not None:
                if not np.isfinite(observation_time):
                    raise ValueError("Observation time must be finite")
                age = float(self._data.time) - float(observation_time)
                if age < 0 or age > max_observation_age:
                    raise ValueError(f"Policy observation is stale (age {age:.3f}s)")
            self._data.ctrl[:] = values

    def hold_arm_position(self) -> None:
        """Set actuator targets to the current arm pose as a policy safe stop."""
        with self._lock:
            addresses = [self._model.jnt_qposadr[self._model.actuator_trnid[i, 0]]
                         for i in range(self._model.nu)]
            self._data.ctrl[:] = self._data.qpos[addresses]

    def get_ctrl_ranges(self) -> np.ndarray:
        """Return actuator control ranges as a copy."""
        with self._lock:
            return self._model.actuator_ctrlrange.copy()

    def get_actuator_count(self) -> int:
        """Return the number of actuators."""
        with self._lock:
            return int(self._model.nu)

    def set_ctrl(self, targets: np.ndarray) -> None:
        """Set actuator targets, clipping them to configured control ranges."""
        values = np.array(targets, dtype=float, copy=True)
        with self._lock:
            if values.shape != (self._model.nu,):
                raise ValueError(f"Expected {self._model.nu} actuator targets, got {values.shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError("Actuator targets must be finite")
            if np.any(self._model.actuator_ctrllimited):
                for i in range(self._model.nu):
                    if self._model.actuator_ctrllimited[i]:
                        values[i] = np.clip(values[i], *self._model.actuator_ctrlrange[i])
            self._data.ctrl[:] = values

    def reset(self) -> None:
        """Restore the initial state and clear controls and velocities."""
        with self._lock:
            mujoco.mj_resetData(self._model, self._data)
            self._data.qpos[:] = self._initial_qpos
            self._data.ctrl[:] = 0
            mujoco.mj_forward(self._model, self._data)

    def get_body_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of all body positions and quaternions."""
        with self._lock:
            return self._data.xpos.copy(), self._data.xquat.copy()

    def get_object_pose(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one body's world pose while holding the simulation lock."""
        with self._lock:
            return self._data.xpos[body_id].copy(), self._data.xquat[body_id].copy()

    def move_object_slot(self, joint_id: int, geom_id: int, pos: np.ndarray,
                         quat: np.ndarray, commit: bool = True) -> None:
        """Commit a freejoint pose, reject interpenetration, and clear velocity.

        Placement is checked with MuJoCo's generated contacts after a temporary
        forward pass. Contacts with the table/floor are allowed; contacts with
        another pool object or the robot arm reject the placement.
        """
        position = np.array(pos, dtype=float, copy=True)
        rotation = np.array(quat, dtype=float, copy=True)
        if position.shape != (3,) or rotation.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("Object pose must contain finite position (3) and quaternion (4)")
        norm = np.linalg.norm(rotation)
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError("Quaternion norm must be finite and nonzero")
        rotation /= norm
        with self._lock:
            self._validate_slot_ids_locked(joint_id, geom_id)
            qadr = int(self._model.jnt_qposadr[joint_id])
            vadr = int(self._model.jnt_dofadr[joint_id])
            old_pose = self._data.qpos[qadr:qadr + 7].copy()
            old_velocity = self._data.qvel[vadr:vadr + 6].copy()
            old_contype = int(self._model.geom_contype[geom_id])
            old_conaffinity = int(self._model.geom_conaffinity[geom_id])
            try:
                self._validate_placement_bounds_locked(position, rotation, geom_id)
                if not commit:
                    self._model.geom_contype[geom_id] = 1
                    self._model.geom_conaffinity[geom_id] = 1
                self._data.qpos[qadr:qadr + 3] = position
                self._data.qpos[qadr + 3:qadr + 7] = rotation
                mujoco.mj_forward(self._model, self._data)
                if self._has_forbidden_contact_locked(geom_id):
                    raise ValueError("Placement overlaps another object or the robot")
                if commit:
                    self._data.qvel[vadr:vadr + 6] = 0
                    mujoco.mj_forward(self._model, self._data)
            except Exception:
                self._data.qpos[qadr:qadr + 7] = old_pose
                self._data.qvel[vadr:vadr + 6] = old_velocity
                self._model.geom_contype[geom_id] = old_contype
                self._model.geom_conaffinity[geom_id] = old_conaffinity
                mujoco.mj_forward(self._model, self._data)
                raise
            if not commit:
                self._data.qpos[qadr:qadr + 7] = old_pose
                self._data.qvel[vadr:vadr + 6] = old_velocity
                self._model.geom_contype[geom_id] = old_contype
                self._model.geom_conaffinity[geom_id] = old_conaffinity
                mujoco.mj_forward(self._model, self._data)

    def _validate_slot_ids_locked(self, joint_id: int, geom_id: int) -> None:
        """Validate that the given joint and geom belong to one free object slot."""
        if not (0 <= joint_id < self._model.njnt and 0 <= geom_id < self._model.ngeom):
            raise ValueError("Object slot ids are outside the compiled model")
        if self._model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Object pool joint must be a freejoint")
        if self._model.jnt_bodyid[joint_id] != self._model.geom_bodyid[geom_id]:
            raise ValueError("Object slot joint and geom must belong to the same body")
        body_name = self._model.body(int(self._model.jnt_bodyid[joint_id])).name or ""
        if not body_name.startswith("pool_"):
            raise ValueError("Object slot must belong to a preallocated pool body")

    def _has_forbidden_contact_locked(self, geom_id: int) -> bool:
        """Return whether an object contact is with anything except table/floor."""
        for contact in self._data.contact[:self._data.ncon]:
            pair = (int(contact.geom1), int(contact.geom2))
            if geom_id not in pair:
                continue
            other = pair[1] if pair[0] == geom_id else pair[0]
            other_name = self._model.geom(other).name or ""
            if other_name not in {"table", "floor"}:
                return True
        return False

    def _validate_placement_bounds_locked(self, position: np.ndarray, rotation: np.ndarray,
                                          geom_id: int, dimensions: np.ndarray | None = None) -> None:
        """Reject object poses whose rotated shape does not fit on the table."""
        table_id = next((i for i in range(self._model.ngeom)
                         if self._model.geom(i).name == "table"), None)
        if table_id is None:
            raise RuntimeError("Scene has no table geom for placement validation")
        table_center = self._model.geom_pos[table_id]
        table_size = self._model.geom_size[table_id]
        sizes = self._model.geom_size[geom_id] if dimensions is None else dimensions
        geom_type = self._model.geom_type[geom_id]
        w, x, y, z = rotation
        matrix = np.array((
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        ))
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(matrix) @ sizes[:3]
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = sizes[0] * np.sqrt(np.sum(matrix[:, :2] ** 2, axis=1)) + sizes[1] * np.abs(matrix[:, 2])
        else:
            extent = np.full(3, sizes[0])
        table_top = table_center[2] + table_size[2]
        if (position[0] - extent[0] < table_center[0] - table_size[0] - 1e-6 or
                position[0] + extent[0] > table_center[0] + table_size[0] + 1e-6 or
                position[1] - extent[1] < table_center[1] - table_size[1] - 1e-6 or
                position[1] + extent[1] > table_center[1] + table_size[1] + 1e-6 or
                position[2] - extent[2] < table_top - 0.005):
            raise ValueError("Placement is outside the table bounds")

    def get_geom_visual_data(self, *, include_mesh_data: bool = True) -> list[dict[str, object]]:
        """Return renderable geom descriptors, optionally including copied mesh arrays."""
        with self._lock:
            result: list[dict[str, object]] = []
            for geom_id in range(self._model.ngeom):
                group = int(self._model.geom_group[geom_id])
                # This SO-101 asset assigns visual geoms to group 2 and
                # collision geoms to group 3. Keep standard scene geoms too.
                if group == 3:
                    continue
                geom_type = int(self._model.geom_type[geom_id])
                item: dict[str, object] = {
                    "id": geom_id,
                    "name": self._model.geom(geom_id).name or f"geom_{geom_id}",
                    "body": int(self._model.geom_bodyid[geom_id]),
                    "body_name": self._model.body(int(self._model.geom_bodyid[geom_id])).name or "world",
                    "type": geom_type,
                    "size": self._model.geom_size[geom_id].copy(),
                    "pos": self._model.geom_pos[geom_id].copy(),
                    "quat": self._model.geom_quat[geom_id].copy(),
                    "rgba": self._model.geom_rgba[geom_id].copy(),
                    "is_pool": bool((self._model.geom(geom_id).name or "").startswith("pool_")),
                }
                if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh_id = int(self._model.geom_dataid[geom_id])
                    item["mesh_id"] = mesh_id
                    if include_mesh_data:
                        vert_start = int(self._model.mesh_vertadr[mesh_id])
                        vert_count = int(self._model.mesh_vertnum[mesh_id])
                        face_start = int(self._model.mesh_faceadr[mesh_id])
                        face_count = int(self._model.mesh_facenum[mesh_id])
                        item["vertices"] = self._model.mesh_vert[vert_start:vert_start + vert_count].copy()
                        item["faces"] = self._model.mesh_face[face_start:face_start + face_count].copy()
                result.append(item)
            return result

    def get_mesh_data(self, mesh_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return one compiled mesh's local vertices and faces as owned copies."""
        with self._lock:
            if not (0 <= mesh_id < self._model.nmesh):
                return None
            vert_start = int(self._model.mesh_vertadr[mesh_id])
            vert_count = int(self._model.mesh_vertnum[mesh_id])
            face_start = int(self._model.mesh_faceadr[mesh_id])
            face_count = int(self._model.mesh_facenum[mesh_id])
            return (self._model.mesh_vert[vert_start:vert_start + vert_count].copy(),
                    self._model.mesh_face[face_start:face_start + face_count].copy())

    def get_body_count(self) -> int:
        """Return the compiled body count under the simulation lock."""
        with self._lock:
            return int(self._model.nbody)

    def get_camera_names(self) -> tuple[str, ...]:
        """Return camera names without exposing the compiled model."""
        with self._lock:
            return tuple(self._model.camera(i).name or f"camera_{i}" for i in range(self._model.ncam))

    def get_object_pool(self) -> list[dict[str, object]]:
        """Return pool slot identifiers and default geometry properties."""
        with self._lock:
            slots = []
            for body_id in range(self._model.nbody):
                body_name = self._model.body(body_id).name or ""
                if not body_name.startswith("pool_"):
                    continue
                geom_id = next((i for i in range(self._model.ngeom)
                                if int(self._model.geom_bodyid[i]) == body_id), None)
                joint_id = next((i for i in range(self._model.njnt)
                                 if int(self._model.jnt_bodyid[i]) == body_id), None)
                if geom_id is None or joint_id is None:
                    continue
                kind = body_name.split("_")[1]
                size_length = 3 if kind == "box" else 2 if kind == "cylinder" else 1
                slots.append({"name": body_name, "kind": kind,
                              "body_id": body_id, "joint_id": joint_id, "geom_id": geom_id,
                              "size": self._model.geom_size[geom_id, :size_length].copy(),
                              "rgba": self._model.geom_rgba[geom_id].copy()})
            return slots

    def activate_object_slot(self, joint_id: int, geom_id: int, pos: np.ndarray,
                             quat: np.ndarray, size: np.ndarray | None = None,
                             color: np.ndarray | None = None,
                             validate_placement: bool = True) -> None:
        """Validate and activate one pool object atomically under the Sim lock."""
        position = np.array(pos, dtype=float, copy=True)
        rotation = np.array(quat, dtype=float, copy=True)
        if position.shape != (3,) or rotation.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("Object pose must contain finite position (3) and quaternion (4)")
        norm = np.linalg.norm(rotation)
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError("Quaternion norm must be finite and nonzero")
        rotation = rotation / norm
        dimensions = None if size is None else np.array(size, dtype=float, copy=True)
        rgba = None if color is None else np.array(color, dtype=float, copy=True)
        with self._lock:
            self._validate_slot_ids_locked(joint_id, geom_id)
            expected_size = 3 if self._model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX else (
                2 if self._model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER else 1)
            if dimensions is not None and (dimensions.shape != (expected_size,) or
                    not np.isfinite(dimensions).all() or np.any(dimensions <= 0) or
                    np.any(dimensions > 0.12)):
                raise ValueError(f"Size must contain {expected_size} positive finite values no larger than 0.12 m")
            if rgba is not None and (rgba.shape != (4,) or not np.isfinite(rgba).all() or
                    np.any((rgba < 0) | (rgba > 1))):
                raise ValueError("Color must contain four values between 0 and 1")
            qpos_adr = int(self._model.jnt_qposadr[joint_id])
            qvel_adr = int(self._model.jnt_dofadr[joint_id])
            old_pose = self._data.qpos[qpos_adr:qpos_adr + 7].copy()
            old_velocity = self._data.qvel[qvel_adr:qvel_adr + 6].copy()
            old_size = self._model.geom_size[geom_id].copy()
            old_rgba = self._model.geom_rgba[geom_id].copy()
            old_contype = int(self._model.geom_contype[geom_id])
            old_conaffinity = int(self._model.geom_conaffinity[geom_id])
            try:
                if validate_placement:
                    self._validate_placement_bounds_locked(position, rotation, geom_id, dimensions)
                self._data.qpos[qpos_adr:qpos_adr + 3] = position
                self._data.qpos[qpos_adr + 3:qpos_adr + 7] = rotation
                self._data.qvel[qvel_adr:qvel_adr + 6] = 0
                if dimensions is not None:
                    self._model.geom_size[geom_id, :expected_size] = dimensions
                if rgba is not None:
                    self._model.geom_rgba[geom_id] = rgba
                self._model.geom_contype[geom_id] = 1
                self._model.geom_conaffinity[geom_id] = 1
                mujoco.mj_forward(self._model, self._data)
                if validate_placement and self._has_forbidden_contact_locked(geom_id):
                    raise ValueError("Placement overlaps another object or the robot")
            except Exception:
                self._data.qpos[qpos_adr:qpos_adr + 7] = old_pose
                self._data.qvel[qvel_adr:qvel_adr + 6] = old_velocity
                self._model.geom_size[geom_id] = old_size
                self._model.geom_rgba[geom_id] = old_rgba
                self._model.geom_contype[geom_id] = old_contype
                self._model.geom_conaffinity[geom_id] = old_conaffinity
                mujoco.mj_forward(self._model, self._data)
                raise

    def deactivate_object_slot(self, joint_id: int, geom_id: int, parked_pos: np.ndarray) -> None:
        """Park an object slot and disable its collisions atomically."""
        with self._lock:
            qpos_adr = int(self._model.jnt_qposadr[joint_id])
            qvel_adr = int(self._model.jnt_dofadr[joint_id])
            self._data.qpos[qpos_adr:qpos_adr + 3] = parked_pos
            self._data.qpos[qpos_adr + 3:qpos_adr + 7] = (1, 0, 0, 0)
            self._data.qvel[qvel_adr:qvel_adr + 6] = 0
            self._model.geom_contype[geom_id] = 0
            self._model.geom_conaffinity[geom_id] = 0
            mujoco.mj_forward(self._model, self._data)

    def update_object_geom(self, geom_id: int, size: np.ndarray | None = None,
                           color: np.ndarray | None = None) -> None:
        """Update supported object visual dimensions and RGBA under the Sim lock."""
        dimensions = None if size is None else np.array(size, dtype=float, copy=True)
        rgba = None if color is None else np.array(color, dtype=float, copy=True)
        with self._lock:
            if not (0 <= geom_id < self._model.ngeom):
                raise ValueError("Object geom id is outside the compiled model")
            body_name = self._model.body(int(self._model.geom_bodyid[geom_id])).name or ""
            if not body_name.startswith("pool_"):
                raise ValueError("Visual updates are limited to pooled object geoms")
            expected = 3 if self._model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX else (
                2 if self._model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER else 1)
            if dimensions is not None:
                if (dimensions.shape != (expected,) or not np.isfinite(dimensions).all() or
                        np.any(dimensions <= 0) or np.any(dimensions > 0.12)):
                    raise ValueError(f"Size must contain {expected} positive finite values no larger than 0.12 m")
            if rgba is not None:
                if rgba.shape != (4,) or not np.isfinite(rgba).all() or np.any((rgba < 0) | (rgba > 1)):
                    raise ValueError("Color must contain four values between 0 and 1")
            old_size = self._model.geom_size[geom_id].copy()
            old_rgba = self._model.geom_rgba[geom_id].copy()
            try:
                if dimensions is not None:
                    self._model.geom_size[geom_id, :expected] = dimensions
                    mujoco.mj_forward(self._model, self._data)
                    if self._model.geom_contype[geom_id]:
                        body_id = int(self._model.geom_bodyid[geom_id])
                        self._validate_placement_bounds_locked(self._data.xpos[body_id], self._data.xquat[body_id], geom_id)
                        if self._has_forbidden_contact_locked(geom_id):
                            raise ValueError("Size change overlaps another object or the robot")
                if rgba is not None:
                    self._model.geom_rgba[geom_id] = rgba
            except Exception:
                self._model.geom_size[geom_id] = old_size
                self._model.geom_rgba[geom_id] = old_rgba
                mujoco.mj_forward(self._model, self._data)
                raise

    def close(self) -> None:
        """Stop the physics thread and permanently close this simulation."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._stop_event.set()
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join()
            self._thread = None
            self._closed = True
