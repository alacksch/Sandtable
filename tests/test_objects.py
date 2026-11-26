"""Object pool lifecycle and stability checks."""

import numpy as np
import pytest

from simwrap.objects import ObjectManager
from simwrap.sim import Sim


def test_spawn_delete_cycles_reuse_slots() -> None:
    sim = Sim()
    objects = ObjectManager(sim)
    try:
        first = objects.spawn("box", np.array((0.0, -0.3, 1.2)))
        assert objects.list_active() == [first]
        objects.delete(first)
        second = objects.spawn("box", np.array((0.1, -0.3, 1.2)))
        assert second != first
        assert objects.list_active() == [second]
        objects.clear_all()
        assert objects.list_active() == []
    finally:
        sim.close()


def test_pool_exhaustion_is_clear_and_recoverable() -> None:
    sim = Sim()
    objects = ObjectManager(sim)
    try:
        names = [objects.spawn("sphere", np.array((-0.36 + i * 0.08, -0.3, 1.2))) for i in range(10)]
        with pytest.raises(RuntimeError, match="pool exhausted for sphere"):
            objects.spawn("sphere", np.array((0.0, -0.3, 1.2)))
        objects.delete(names[0])
        assert objects.spawn("sphere", np.array((-0.36, -0.3, 1.2)))
    finally:
        sim.close()


def test_objects_settle_without_nans() -> None:
    sim = Sim()
    objects = ObjectManager(sim)
    try:
        objects.spawn("box", np.array((-0.18, -0.3, 1.0)))
        objects.spawn("cylinder", np.array((0.0, -0.3, 1.0)))
        objects.spawn("sphere", np.array((0.18, -0.3, 1.0)))
        sim.step(3000)
        slots = sim.get_object_pool()
        active_slots = objects.active_slot_names()
        state = sim.get_state_snapshot()
        assert np.isfinite(state["qpos"]).all()
        assert np.isfinite(state["qvel"]).all()
        assert np.isfinite(state["xpos"]).all()
        for slot in slots:
            if slot["name"] in active_slots:
                z = state["xpos"][int(slot["body_id"]), 2]
                assert 0.74 < z < 0.9
    finally:
        sim.close()


def test_check_placement_does_not_commit_candidate_pose() -> None:
    sim = Sim()
    objects = ObjectManager(sim)
    try:
        sim.reset()
        before_pos, before_quat = sim.get_body_poses()
        objects.check_placement(np.array((0.0, -0.4, 0.785)), np.array((1.0, 0.0, 0.0, 0.0)), kind="box")
        after_pos, after_quat = sim.get_body_poses()
        assert np.allclose(before_pos, after_pos)
        assert np.allclose(before_quat, after_quat)
    finally:
        sim.close()


def test_interleaved_object_selection_move_sequence() -> None:
    """Moving earlier and later spawns remains name-addressed and independent."""
    sim = Sim()
    objects = ObjectManager(sim)
    try:
        a = objects.spawn("box", np.array((-0.4, -0.32, 0.785)))
        b = objects.spawn("cylinder", np.array((0.0, -0.32, 0.79)))
        c = objects.spawn("sphere", np.array((0.4, -0.32, 0.785)))
        pose_a1 = np.array((-0.3, -0.32, 0.785))
        pose_b = np.array((0.1, -0.32, 0.79))
        pose_a2 = np.array((-0.25, -0.32, 0.785))
        pose_c = np.array((0.45, -0.32, 0.785))
        identity = np.array((1.0, 0.0, 0.0, 0.0))
        objects.move(a, pose_a1, identity)
        objects.move(b, pose_b, identity)
        objects.move(a, pose_a2, identity)
        objects.delete(b)
        objects.move(c, pose_c, identity)
        assert np.allclose(objects.pose(a)[0], pose_a2)
        assert np.allclose(objects.pose(c)[0], pose_c)
        assert objects.list_active() == [a, c]
    finally:
        sim.close()


def test_failed_spawn_keeps_pool_slot_available() -> None:
    """An overlapping spawn is rejected without consuming its pool slot."""
    sim = Sim()
    objects = ObjectManager(sim)
    try:
        first = objects.spawn("sphere", np.array((0.0, -0.3, 1.2)))
        with pytest.raises(ValueError, match="overlaps"):
            objects.spawn("sphere", np.array((0.0, -0.3, 1.2)))
        assert objects.list_active() == [first]
        second = objects.spawn("sphere", np.array((0.15, -0.3, 1.2)))
        assert objects.list_active() == [first, second]
    finally:
        sim.close()


def test_spawn_outside_table_is_rejected_by_backend() -> None:
    """The backend enforces table bounds even if a client skips its preview check."""
    sim = Sim()
    objects = ObjectManager(sim)
    try:
        with pytest.raises(ValueError, match="table bounds"):
            objects.spawn("box", np.array((0.59, -0.3, 0.785)))
        name = objects.spawn("box", np.array((0.0, -0.3, 0.785)))
        assert objects.list_active() == [name]
    finally:
        sim.close()


def test_rejected_size_update_restores_geometry_and_registry() -> None:
    """A resize that causes overlap is rolled back in both Sim and object metadata."""
    sim = Sim()
    objects = ObjectManager(sim)
    try:
        first = objects.spawn("box", np.array((-0.1, -0.3, 0.785)))
        second = objects.spawn("box", np.array((0.0, -0.3, 0.785)))
        with pytest.raises(ValueError, match="overlaps"):
            objects.update_visual(first, size=np.array((0.08, 0.035, 0.035)))
        descriptor = next(item for item in objects.list_objects() if item["name"] == first)
        assert np.allclose(descriptor["size"], (0.035, 0.035, 0.035))
        assert np.allclose(objects.pose(first)[0], (-0.1, -0.3, 0.785))
        assert second in objects.list_active()
    finally:
        sim.close()


def test_active_object_visual_properties_are_updated() -> None:
    sim = Sim()
    objects = ObjectManager(sim)
    try:
        name = objects.spawn("box", np.array((0.0, -0.4, 0.785)))
        size = np.array((0.05, 0.045, 0.04))
        color = np.array((0.1, 0.3, 0.6, 1.0))
        objects.update_visual(name, size=size, color=color)
        descriptor = objects.list_objects()[0]
        assert descriptor["name"] == name
        assert np.allclose(descriptor["size"], size)
        assert np.allclose(descriptor["color"], color)
    finally:
        sim.close()


def test_session_replacement_preserves_names_and_rolls_back_invalid_scene() -> None:
    """Session load preserves stable names and restores the prior scene after rejection."""
    sim = Sim()
    objects = ObjectManager(sim)
    try:
        original = objects.spawn("box", np.array((-0.25, -0.3, 0.785)))
        snapshot = objects.get_session_objects()[0]
        descriptors = [{"name": "saved_sphere", "kind": "sphere",
                        "position": np.array((0.25, -0.3, 0.785)),
                        "quaternion": np.array((1.0, 0.0, 0.0, 0.0)),
                        "size": np.array((0.04,)), "color": np.array((0.2, 0.4, 0.8, 1.0))}]
        assert objects.replace_scene(descriptors) == ["saved_sphere"]
        saved = objects.get_session_objects()[0]
        assert saved["name"] == "saved_sphere"
        assert saved["kind"] == "sphere"
        assert np.allclose(saved["position"], descriptors[0]["position"])
        assert np.allclose(saved["size"], descriptors[0]["size"])

        overlap = [{"name": "sphere_a", "kind": "sphere", "position": (0, -0.3, 0.85),
                   "quaternion": (1, 0, 0, 0), "size": (0.035,), "color": (1, 0, 0, 1)},
                   {"name": "sphere_b", "kind": "sphere", "position": (0, -0.3, 0.85),
                    "quaternion": (1, 0, 0, 0), "size": (0.035,), "color": (0, 1, 0, 1)}]
        with pytest.raises(ValueError, match="overlaps"):
            objects.replace_scene(overlap)
        restored = objects.get_session_objects()
        assert [item["name"] for item in restored] == ["saved_sphere"]
        assert np.allclose(restored[0]["position"], descriptors[0]["position"])
        assert original not in objects.list_active()
        assert len(objects.list_active()) == 1
        assert snapshot["kind"] == "box"
    finally:
        sim.close()


def test_fallen_objects_return_to_pool_and_publish_recovery_notice() -> None:
    """Recovery deactivates a fallen slot and retains a sequenced notice for clients."""
    sim = Sim()
    objects = ObjectManager(sim)
    try:
        name = objects.spawn("sphere", np.array((0.0, -0.3, 0.785)))
        assert objects.recover_fallen(threshold_z=0.8) == [name]
        assert objects.list_active() == []
        assert objects.get_recovery_cursor() == 1
        assert objects.get_recovery_events_after(0) == [(1, [name])]
        assert objects.get_recovery_events_after(1) == []
        assert objects.spawn("sphere", np.array((0.15, -0.3, 0.785)))
    finally:
        sim.close()
