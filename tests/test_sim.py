"""Basic simulation stability and reset checks."""

import numpy as np
import pytest
import time
import threading

from simwrap.sim import Sim


def test_load_step_and_reset() -> None:
    """The scene loads, remains finite through 1000 steps, and resets."""
    sim = Sim()
    try:
        initial = sim.get_state_snapshot()["qpos"]
        sim.set_ctrl(np.zeros(sim.get_actuator_count()))
        sim.step(1000)
        state = sim.get_state_snapshot()
        assert np.isfinite(state["qpos"]).all()
        assert np.isfinite(state["qvel"]).all()
        assert np.isfinite(state["xpos"]).all()
        moved = np.ones(sim.get_actuator_count()) * 0.1
        sim.set_ctrl(moved)
        sim.step(50)
        sim.reset()
        assert np.isfinite(sim.get_arm_state()).all()
        state = sim.get_state_snapshot()
        assert np.allclose(state["qpos"], initial)
        assert np.allclose(state["qvel"], 0)
    finally:
        sim.close()


def test_physics_lifecycle_can_pause_resume_and_close() -> None:
    """The physics worker stops cleanly, restarts once, and cannot restart after close."""
    sim = Sim()
    try:
        sim.start()
        sim.start()
        time.sleep(0.025)
        sim.stop()
        paused_time = sim.get_simulation_time()
        time.sleep(0.01)
        assert sim.get_simulation_time() == paused_time

        sim.start()
        time.sleep(0.025)
        assert sim.get_simulation_time() > paused_time
        sim.close()
        sim.close()
        with pytest.raises(RuntimeError, match="closed"):
            sim.start()
    finally:
        sim.close()


def test_rgb_render_draw_does_not_hold_simulation_lock() -> None:
    """Rendering uses a prepared scene after releasing the model/data lock."""
    class LockProbeRenderer:
        def update_scene(self, data, camera):
            assert camera in {"front", "top", "wrist"}

        def render(self):
            acquired = threading.Event()

            def read_simulation_time():
                sim.get_simulation_time()
                acquired.set()

            worker = threading.Thread(target=read_simulation_time, daemon=True)
            worker.start()
            worker.join(timeout=0.5)
            assert acquired.is_set()
            return np.zeros((8, 8, 3), dtype=np.uint8)

    sim = Sim()
    try:
        renderer = LockProbeRenderer()
        images = sim.render_cameras(renderer, ("front", "top", "wrist"))
        assert set(images) == {"front", "top", "wrist"}
        assert all(image.shape == (8, 8, 3) for image in images.values())
    finally:
        sim.close()
