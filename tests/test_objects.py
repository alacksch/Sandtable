"""Object pool lifecycle and stability checks."""

import numpy as np
import pytest

from simwrap.objects import ObjectManager
from simwrap.sim import Sim


def test_spawn_delete_cycles_reuse_slots() -> None:
    sim = Sim()
    objects = ObjectManager(sim)
    try:
        first = objects.spawn("box", np.array((0.0, 0.0, 1.0)))
        assert objects.list_active() == [first]
        objects.delete(first)
        second = objects.spawn("box", np.array((0.1, 0.0, 1.0)))
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
        names = [objects.spawn("sphere", np.array((0.0, 0.0, 1.0))) for _ in range(10)]
        with pytest.raises(RuntimeError, match="pool exhausted for sphere"):
            objects.spawn("sphere", np.array((0.0, 0.0, 1.0)))
        objects.delete(names[0])
        assert objects.spawn("sphere", np.array((0.0, 0.0, 1.0)))
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
        with sim.lock:
            assert np.isfinite(sim.data.qpos).all()
            assert np.isfinite(sim.data.qvel).all()
            assert np.isfinite(sim.data.xpos).all()
            for slot in slots:
                if slot["name"] in active_slots:
                    z = sim.data.xpos[int(slot["body_id"]), 2]
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
