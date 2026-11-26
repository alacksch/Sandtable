"""Typed, model-independent observation and action boundaries for policies."""

from typing import Protocol, TypedDict

import numpy as np


class PolicyObservation(TypedDict):
    """Simulator observation plus explicit joint and clock metadata."""

    images: dict[str, np.ndarray]
    joint_state: np.ndarray
    prompt: str
    joint_names: list[str]
    joint_units: list[str]
    simulation_time: float


class Policy(Protocol):
    """Pure inference interface; implementations must not access MuJoCo state."""

    def act(self, observation: PolicyObservation) -> np.ndarray:
        """Return one actuator target per joint in the documented joint order."""
        ...


class HoldPositionPolicy:
    """Deterministic offline policy that returns the observed joint position."""

    def act(self, observation: PolicyObservation) -> np.ndarray:
        """Return an owned copy of the current joint state as hold targets."""
        state = np.asarray(observation["joint_state"], dtype=float)
        names = observation["joint_names"]
        if state.ndim != 1 or len(names) != state.size or not np.isfinite(state).all():
            raise ValueError("Observation joint state must be finite and match its named joints")
        return state.copy()


def validate_action_targets(targets: np.ndarray, actuator_info: list[dict[str, object]]) -> np.ndarray:
    """Validate and copy policy targets using the simulator's actuator metadata."""
    values = np.array(targets, dtype=float, copy=True)
    if values.shape != (len(actuator_info),):
        raise ValueError(f"Expected {len(actuator_info)} targets, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Policy targets must be finite")
    for index, actuator in enumerate(actuator_info):
        if not actuator["limited"]:
            continue
        limits = np.asarray(actuator["control_range"], dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] > limits[1]:
            raise ValueError(f"Actuator {actuator['actuator_name']} has invalid control limits")
        if values[index] < limits[0] or values[index] > limits[1]:
            raise ValueError(f"Target for {actuator['joint_name']} is outside its control range")
    return values
