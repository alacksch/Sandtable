"""Management of precompiled scene object slots."""

from typing import Any

import numpy as np

from .sim import Sim


class ObjectManager:
    """Allocate and control scene objects without direct model access."""

    def __init__(self, sim: Sim):
        """Build a name-to-slot registry over the compiled scene pool."""
        self.sim = sim
        self._slots = sim.get_object_pool()
        self._active: dict[str, dict[str, Any]] = {}
        self._free = list(self._slots)
        self._counter = 0

    def spawn(self, kind: str, pos: np.ndarray, quat: np.ndarray | None = None,
              size: np.ndarray | None = None, color: np.ndarray | None = None) -> str:
        """Activate an available slot and return its unique object name."""
        if kind not in {"box", "cylinder", "sphere"}:
            raise ValueError(f"Unsupported object kind: {kind}")
        slot_index = next((i for i, slot in enumerate(self._free) if slot["kind"] == kind), None)
        if slot_index is None:
            raise RuntimeError(f"Object pool exhausted for {kind} (capacity: {sum(s['kind'] == kind for s in self._slots)})")
        slot = self._free.pop(slot_index)
        name = f"{kind}_{self._counter}"
        self._counter += 1
        self.sim.activate_object_slot(
            int(slot["joint_id"]), int(slot["geom_id"]), np.asarray(pos),
            np.asarray((1, 0, 0, 0) if quat is None else quat),
            slot["size"] if size is None else size,
            slot["rgba"] if color is None else color,
        )
        self._active[name] = slot
        return name

    def delete(self, name: str) -> None:
        """Deactivate a named object and return its slot to the pool."""
        if name not in self._active:
            raise KeyError(f"No active object named {name!r}")
        slot = self._active.pop(name)
        slot_number = sum(s["kind"] == slot["kind"] for s in self._slots[:self._slots.index(slot)])
        parked = np.array((20.0 + slot_number * 0.3, 20.0 + ("box", "cylinder", "sphere").index(slot["kind"]) * 2.0, -10.0))
        self.sim.deactivate_object_slot(int(slot["joint_id"]), int(slot["geom_id"]), parked)
        self._free.append(slot)

    def list_active(self) -> list[str]:
        """Return active object names in creation order."""
        return list(self._active)

    def active_slot_names(self) -> set[str]:
        """Return pool slot names currently in use for renderer visibility."""
        return {str(slot["name"]) for slot in self._active.values()}

    def move(self, name: str, pos: np.ndarray, quat: np.ndarray) -> None:
        """Commit a placed pose, clearing velocity and forwarding the model."""
        if name not in self._active:
            raise KeyError(f"No active object named {name!r}")
        slot = self._active[name]
        self.sim.move_object_slot(int(slot["joint_id"]), int(slot["geom_id"]), pos, quat)

    def check_placement(self, pos: np.ndarray, quat: np.ndarray, *, kind: str | None = None,
                        name: str | None = None) -> None:
        """Validate a candidate pose without committing it."""
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

    def get_object(self, name: str) -> dict[str, Any]:
        """Return an active object's metadata and current world pose."""
        if name not in self._active:
            raise KeyError(f"No active object named {name!r}")
        slot = self._active[name]
        pos, quat = self.pose(name)
        return {"name": name, "kind": slot["kind"], "pos": pos, "quat": quat,
                "size": np.asarray(slot["size"]).copy(), "color": np.asarray(slot["rgba"]).copy()}

    def list_objects(self) -> list[dict[str, Any]]:
        """Return active object descriptors without reading MuJoCo data directly."""
        return [{"name": name, "kind": str(slot["kind"]), "slot": str(slot["name"]),
                 "size": np.asarray(slot["size"]).copy(), "color": np.asarray(slot["rgba"]).copy()}
                for name, slot in self._active.items()]

    def update_visual(self, name: str, size: np.ndarray | None = None,
                      color: np.ndarray | None = None) -> None:
        """Update an active object's supported size or color through Sim."""
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
        if name not in self._active:
            raise KeyError(f"No active object named {name!r}")
        slot = self._active[name]
        return self.sim.get_object_pose(int(slot["body_id"]))

    def half_height(self, name: str) -> float:
        """Return the object's vertical half extent for upright placement."""
        size = np.asarray(self._active[name]["size"])
        kind = self._active[name]["kind"]
        return float(size[2] if kind == "box" else size[1] if kind == "cylinder" else size[0])

    def clear_all(self) -> None:
        """Deactivate all active objects."""
        for name in list(self._active):
            self.delete(name)
