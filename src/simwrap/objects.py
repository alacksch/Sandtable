"""Management of precompiled scene object slots."""

import threading
from collections import deque
from typing import Any

import numpy as np

from .sim import Sim


class ObjectManager:
    """Allocate and control scene objects without direct model access."""

    def __init__(self, sim: Sim):
        """Build a name-to-slot registry over the compiled scene pool."""
        self.sim = sim
        self._lock = threading.RLock()
        self._slots = sim.get_object_pool()
        self._active: dict[str, dict[str, Any]] = {}
        self._free = list(self._slots)
        self._counter = 0
        self._recovery_sequence = 0
        self._recovery_events: deque[tuple[int, list[str]]] = deque(maxlen=100)

    def spawn(self, kind: str, pos: np.ndarray, quat: np.ndarray | None = None,
              size: np.ndarray | None = None, color: np.ndarray | None = None) -> str:
        """Activate an available slot and return its unique object name."""
        with self._lock:
            return self._spawn_locked(kind, pos, quat, size, color)

    def _spawn_locked(self, kind: str, pos: np.ndarray, quat: np.ndarray | None = None,
                      size: np.ndarray | None = None, color: np.ndarray | None = None,
                      name: str | None = None, validate_placement: bool = True) -> str:
        """Activate a slot; caller must hold the registry lock."""
        if kind not in {"box", "cylinder", "sphere"}:
            raise ValueError(f"Unsupported object kind: {kind}")
        slot_index = next((i for i, slot in enumerate(self._free) if slot["kind"] == kind), None)
        if slot_index is None:
            capacity = sum(s["kind"] == kind for s in self._slots)
            raise RuntimeError(f"Object pool exhausted for {kind} (capacity: {capacity})")
        if name is None:
            while f"{kind}_{self._counter}" in self._active:
                self._counter += 1
            name = f"{kind}_{self._counter}"
        if name in self._active:
            raise ValueError(f"Object name {name!r} is already active")
        slot = self._free.pop(slot_index)
        try:
            active_size = np.asarray(slot["size"] if size is None else size, dtype=float).copy()
            active_color = np.asarray(slot["rgba"] if color is None else color, dtype=float).copy()
            self.sim.activate_object_slot(
                int(slot["joint_id"]), int(slot["geom_id"]), np.asarray(pos),
                np.asarray((1, 0, 0, 0) if quat is None else quat),
                active_size, active_color,
                validate_placement=validate_placement,
            )
        except Exception:
            self._free.insert(slot_index, slot)
            raise
        self._counter += 1
        slot["size"] = active_size
        slot["rgba"] = active_color
        self._active[name] = slot
        return name

    def get_session_objects(self) -> list[dict[str, Any]]:
        """Return active object descriptors with copied world poses for session files."""
        with self._lock:
            positions, quaternions = self.sim.get_body_poses()
            result = []
            for name, slot in self._active.items():
                body_id = int(slot["body_id"])
                position, quaternion = positions[body_id], quaternions[body_id]
                result.append({"name": name, "kind": str(slot["kind"]),
                               "position": position, "quaternion": quaternion,
                               "size": np.asarray(slot["size"]).copy(),
                               "color": np.asarray(slot["rgba"]).copy()})
            return result

    def replace_scene(self, descriptors: list[dict[str, Any]]) -> list[str]:
        """Replace active objects with a validated scene, restoring the old scene on failure."""
        if not isinstance(descriptors, list):
            raise ValueError("Session objects must be a list")
        with self._lock:
            capacities = {kind: sum(s["kind"] == kind for s in self._slots)
                          for kind in ("box", "cylinder", "sphere")}
            counts = {kind: 0 for kind in capacities}
            seen: set[str] = set()
            normalized: list[dict[str, Any]] = []
            for item in descriptors:
                if not isinstance(item, dict):
                    raise ValueError("Every session object must be an object")
                kind = item.get("kind")
                name = item.get("name")
                if not isinstance(kind, str) or kind not in capacities:
                    raise ValueError(f"Unsupported object kind: {kind}")
                if not isinstance(name, str) or not name or len(name) > 80 or name in seen:
                    raise ValueError("Object names must be unique, non-empty strings of at most 80 characters")
                seen.add(name)
                counts[kind] += 1
                if counts[kind] > capacities[kind]:
                    raise ValueError(f"Session exceeds {kind} pool capacity ({capacities[kind]})")
                try:
                    normalized.append({
                        "name": name, "kind": kind,
                        "position": np.asarray(item["position"], dtype=float),
                        "quaternion": np.asarray(item["quaternion"], dtype=float),
                        "size": np.asarray(item["size"], dtype=float),
                        "color": np.asarray(item["color"], dtype=float),
                    })
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Session object {name!r} has invalid or missing fields") from exc

            previous = self.get_session_objects()
            try:
                self.clear_all()
                for item in normalized:
                    self._spawn_locked(item["kind"], item["position"], item["quaternion"],
                                       item["size"], item["color"], item["name"])
            except Exception as load_error:
                try:
                    self.clear_all()
                    for item in previous:
                        self._spawn_locked(item["kind"], item["position"], item["quaternion"],
                                           item["size"], item["color"], item["name"],
                                           validate_placement=False)
                except Exception as restore_error:
                    raise RuntimeError("Session load failed and previous scene could not be restored") from restore_error
                raise load_error
            return list(self._active)

    def delete(self, name: str) -> None:
        """Deactivate a named object and return its slot to the pool."""
        with self._lock:
            if name not in self._active:
                raise KeyError(f"No active object named {name!r}")
            slot = self._active[name]
            slot_number = sum(s["kind"] == slot["kind"] for s in self._slots[:self._slots.index(slot)])
            parked = np.array((20.0 + slot_number * 0.3, 20.0 + ("box", "cylinder", "sphere").index(slot["kind"]) * 2.0, -10.0))
            self.sim.deactivate_object_slot(int(slot["joint_id"]), int(slot["geom_id"]), parked)
            del self._active[name]
            self._free.append(slot)

    def list_active(self) -> list[str]:
        """Return active object names in creation order."""
        with self._lock:
            return list(self._active)

    def active_slot_names(self) -> set[str]:
        """Return pool slot names currently in use for renderer visibility."""
        with self._lock:
            return {str(slot["name"]) for slot in self._active.values()}

    def move(self, name: str, pos: np.ndarray, quat: np.ndarray) -> None:
        """Commit a placed pose, clearing velocity and forwarding the model."""
        with self._lock:
            if name not in self._active:
                raise KeyError(f"No active object named {name!r}")
            slot = self._active[name]
            self.sim.move_object_slot(int(slot["joint_id"]), int(slot["geom_id"]), pos, quat)

    def check_placement(self, pos: np.ndarray, quat: np.ndarray, *, kind: str | None = None,
                        name: str | None = None) -> None:
        """Validate a candidate pose without committing it."""
        with self._lock:
            if name is not None:
                slot = self._active.get(name)
                if slot is None:
                    raise KeyError(f"No active object named {name!r}")
            else:
                slot = next((s for s in self._free if s["kind"] == kind), None)
                if slot is None:
                    capacity = sum(s["kind"] == kind for s in self._slots)
                    raise RuntimeError(f"Object pool exhausted for {kind} (capacity: {capacity})")
            self.sim.move_object_slot(int(slot["joint_id"]), int(slot["geom_id"]), pos, quat, commit=False)

    def list_objects(self) -> list[dict[str, Any]]:
        """Return active object descriptors without reading MuJoCo data directly."""
        with self._lock:
            return [{"name": name, "kind": str(slot["kind"]), "slot": str(slot["name"]),
                     "size": np.asarray(slot["size"]).copy(), "color": np.asarray(slot["rgba"]).copy()}
                    for name, slot in self._active.items()]

    def update_visual(self, name: str, size: np.ndarray | None = None,
                      color: np.ndarray | None = None) -> None:
        """Update an active object's supported size or color through Sim."""
        with self._lock:
            if name not in self._active:
                raise KeyError(f"No active object named {name!r}")
            slot = self._active[name]
            new_size = None if size is None else np.asarray(size, dtype=float)
            new_color = None if color is None else np.asarray(color, dtype=float)
            self.sim.update_object_geom(int(slot["geom_id"]), new_size, new_color)
            if new_size is not None:
                slot["size"] = new_size.copy()
            if new_color is not None:
                slot["rgba"] = new_color.copy()

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the current world pose for an active object."""
        with self._lock:
            if name not in self._active:
                raise KeyError(f"No active object named {name!r}")
            slot = self._active[name]
            return self.sim.get_object_pose(int(slot["body_id"]))

    def half_height(self, name: str) -> float:
        """Return the object's vertical half extent for upright placement."""
        with self._lock:
            size = np.asarray(self._active[name]["size"])
            kind = self._active[name]["kind"]
            return float(size[2] if kind == "box" else size[1] if kind == "cylinder" else size[0])

    def clear_all(self) -> None:
        """Deactivate all active objects."""
        with self._lock:
            for name in list(self._active):
                self.delete(name)

    def recover_fallen(self, threshold_z: float = 0.25) -> list[str]:
        """Return pooled objects below the recovery plane to their inactive slots."""
        try:
            recovery_height = float(threshold_z)
        except (TypeError, ValueError) as exc:
            raise ValueError("Recovery height must be finite") from exc
        if not np.isfinite(recovery_height):
            raise ValueError("Recovery height must be finite")
        with self._lock:
            positions, _ = self.sim.get_body_poses()
            fallen = [name for name, slot in self._active.items()
                      if positions[int(slot["body_id"]), 2] < recovery_height]
            for name in fallen:
                self.delete(name)
            if fallen:
                self._recovery_sequence += 1
                self._recovery_events.append((self._recovery_sequence, fallen.copy()))
            return fallen

    def get_recovery_cursor(self) -> int:
        """Return the latest recovery event sequence for a new stream client."""
        with self._lock:
            return self._recovery_sequence

    def get_recovery_events_after(self, sequence: int) -> list[tuple[int, list[str]]]:
        """Return recovery notices newer than a client's last acknowledged sequence."""
        with self._lock:
            return [(event_id, names.copy()) for event_id, names in self._recovery_events
                    if event_id > sequence]
