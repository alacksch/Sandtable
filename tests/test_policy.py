"""Model-independent observation and action boundary checks."""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simwrap.policy import HoldPositionPolicy, PolicyObservation, validate_action_targets
from simwrap.app_state import AppState
from simwrap.policy_runner import (
    PolicyRunner,
    checkpoint_action_to_simulator_radians,
    map_checkpoint_camera_images,
    policy_observation_age_limit,
    simulator_state_to_checkpoint_degrees,
    validate_smolvla_prompt,
)
from simwrap.sim import Sim


EXPECTED_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def test_observation_exposes_named_joint_order_and_explicit_units() -> None:
    """Base observations name their six simulator joints and retain camera keys as-is."""
    sim = Sim()
    try:
        observation = sim.get_observation()
        assert observation["images"] == {}
        assert sim.get_camera_names() == ("front", "top", "wrist")
        assert observation["joint_names"] == EXPECTED_JOINTS
        assert observation["joint_units"] == ["rad"] * 6
        assert observation["joint_state"].shape == (6,)
        assert observation["simulation_time"] == 0.0
    finally:
        sim.close()


def test_hold_policy_returns_an_owned_copy_in_joint_order() -> None:
    """The offline policy returns the observed state without mutating it."""
    state = np.array((0.1, -0.2, 0.3, 0.4, -0.5, 0.0))
    observation: PolicyObservation = {
        "images": {"top": np.zeros((12, 16, 3), dtype=np.uint8)},
        "joint_state": state,
        "prompt": "hold still",
        "joint_names": EXPECTED_JOINTS,
        "joint_units": ["rad"] * 6,
        "simulation_time": 1.25,
    }
    targets = HoldPositionPolicy().act(observation)
    assert np.array_equal(targets, state)
    assert targets is not state
    targets[0] = 9
    assert state[0] == 0.1


def test_policy_action_validation_rejects_bad_shape_nonfinite_and_range() -> None:
    """Policy outputs are rejected rather than clipped into actuator bounds."""
    sim = Sim()
    try:
        info = sim.get_actuator_info()
        midpoint = np.mean(sim.get_ctrl_ranges(), axis=1)
        validated = validate_action_targets(midpoint, info)
        assert np.allclose(validated, midpoint)
        with pytest.raises(ValueError, match="Expected 6"):
            sim.apply_action(np.zeros(5))
        with pytest.raises(ValueError, match="finite"):
            sim.apply_action(np.full(6, np.nan))
        out_of_range = sim.get_ctrl_ranges()[:, 1].copy()
        out_of_range[0] += 1
        with pytest.raises(ValueError, match="outside"):
            sim.apply_action(out_of_range)
        assert np.allclose(sim.get_state_snapshot()["ctrl"], 0)
        sim.apply_action(midpoint)
        assert np.allclose(sim.get_state_snapshot()["ctrl"], midpoint)
    finally:
        sim.close()


def test_policy_rejects_stale_observation_and_holds_current_pose_on_stop() -> None:
    """Old worker results cannot overwrite the current target; stop holds the arm."""
    sim = Sim()
    try:
        sim.step(100)
        state, observed_at = sim.get_arm_state_snapshot()
        sim.apply_action(state, observation_time=observed_at)
        sim.step(200)
        before_stop = sim.get_arm_state()
        with pytest.raises(ValueError, match="stale"):
            sim.apply_action(np.zeros(6), observation_time=observed_at)
        sim.hold_arm_position()
        assert np.allclose(sim.get_state_snapshot()["ctrl"], before_stop)
    finally:
        sim.close()


def test_checkpoint_joint_conversion_is_explicit_degrees_to_radians() -> None:
    """The checkpoint's degree convention is adapted at the policy boundary."""
    radians = np.deg2rad(np.array([0, 30, -45, 90, 180, 10], dtype=float))
    assert np.allclose(simulator_state_to_checkpoint_degrees(radians), [0, 30, -45, 90, 180, 10])
    assert np.allclose(checkpoint_action_to_simulator_radians([0, 30, -45, 90, 180, 10]), radians)
    with pytest.raises(ValueError, match="six"):
        checkpoint_action_to_simulator_radians(np.zeros(5))


def test_smolvla_prompt_gate_checks_the_trained_object_and_action() -> None:
    """A fetch cue alone must not route an untrained box task into the policy."""
    validate_smolvla_prompt("Fetch the ball and put it into my hand")
    validate_smolvla_prompt("Sort the red lego bricks onto the red plate")
    with pytest.raises(ValueError, match="no box task"):
        validate_smolvla_prompt("fetch the box")
    with pytest.raises(ValueError, match="no box task"):
        validate_smolvla_prompt("CARRY THE BOX")
    with pytest.raises(ValueError, match="does not match"):
        validate_smolvla_prompt("fetch the desk")
    with pytest.raises(ValueError, match="Enter a task prompt"):
        validate_smolvla_prompt("  ")


def test_checkpoint_camera_map_uses_the_two_training_views_without_stretching() -> None:
    """The desk view maps to front, and the untrained third camera is omitted."""
    wrist = np.zeros((360, 480, 3), dtype=np.uint8)
    front = np.ones((360, 480, 3), dtype=np.uint8)
    top = np.full((360, 480, 3), 2, dtype=np.uint8)
    mapped = map_checkpoint_camera_images({"wrist": wrist, "front": front, "top": top})
    assert set(mapped) == {"wrist_left", "desk_view"}
    assert mapped["wrist_left"] is wrist
    assert mapped["desk_view"] is front
    assert mapped["desk_view"].shape == (360, 480, 3)
    with pytest.raises(ValueError, match="front"):
        map_checkpoint_camera_images({"wrist": wrist, "top": top})


def test_policy_freshness_tracks_inference_latency_with_a_hard_cap() -> None:
    """Slow real-camera inference gets measured allowance without unbounded staleness."""
    assert policy_observation_age_limit(1.4) == 2.5
    assert policy_observation_age_limit(4.074) == pytest.approx(4.824)
    assert policy_observation_age_limit(10.0) == 8.0
    with pytest.raises(ValueError, match="Inference duration"):
        policy_observation_age_limit(float("nan"))


def test_policy_runner_accepts_measured_inference_over_250ms() -> None:
    """A slow policy within its measured allowance applies without a stale error."""
    completed = threading.Event()

    class SlowPolicy:
        def act(self, observation: PolicyObservation) -> np.ndarray:
            time.sleep(2.8)
            completed.set()
            return np.full(6, 0.05)

    sim = Sim()
    state = AppState()
    sim.set_app_state(state)
    runner = PolicyRunner(sim, object(), state, policy_factory=SlowPolicy)
    sim.start()
    try:
        assert runner.start() == "loading"
        assert completed.wait(timeout=6)
        deadline = time.perf_counter() + 1
        while time.perf_counter() < deadline and not np.allclose(sim.get_arm_control_state()[1], 0.05):
            time.sleep(0.01)
        assert np.allclose(sim.get_arm_control_state()[1], 0.05)
    finally:
        runner.stop(join_timeout=0.2)
        sim.close()
    assert state.policy_status() == "stopped"
