"""Basic simulation stability and reset checks."""

import numpy as np

from simwrap.sim import Sim


def test_load_step_and_reset() -> None:
    """The scene loads, remains finite through 1000 steps, and resets."""
    sim = Sim()
    try:
        sim.set_ctrl(np.zeros(sim.model.nu))
        sim.step(1000)
        with sim.lock:
            assert np.isfinite(sim.data.qpos).all()
            assert np.isfinite(sim.data.qvel).all()
            assert np.isfinite(sim.data.xpos).all()
        moved = np.ones(sim.model.nu) * 0.1
        sim.set_ctrl(moved)
        sim.step(50)
        sim.reset()
        assert np.isfinite(sim.get_arm_state()).all()
        with sim.lock:
            assert np.allclose(sim.data.qpos, sim._initial_qpos)
            assert np.allclose(sim.data.qvel, 0)
    finally:
        sim.close()
