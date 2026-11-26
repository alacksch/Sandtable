"""Optional, isolated LeRobot model runtime for simulated arm control."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .app_state import AppState
from .policy import Policy, PolicyObservation
from .sim import Sim

logger = logging.getLogger(__name__)
MODEL_ID = "Elbarbary/so101_smolvla"
EXPECTED_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
POLICY_MIN_INFERENCE_AGE_SECONDS = 2.5
POLICY_INFERENCE_AGE_MARGIN_SECONDS = 0.75
POLICY_MAX_INFERENCE_AGE_SECONDS = 8.0


def policy_observation_age_limit(inference_seconds: float) -> float:
    """Allow measured inference latency plus margin, with a hard stale-data cap."""
    if not np.isfinite(inference_seconds) or inference_seconds < 0:
        raise ValueError("Inference duration must be finite and non-negative")
    return min(
        POLICY_MAX_INFERENCE_AGE_SECONDS,
        max(POLICY_MIN_INFERENCE_AGE_SECONDS, inference_seconds + POLICY_INFERENCE_AGE_MARGIN_SECONDS),
    )


def simulator_state_to_checkpoint_degrees(state_radians: np.ndarray) -> np.ndarray:
    """Convert the six ordered MuJoCo hinge positions to checkpoint degrees."""
    values = np.asarray(state_radians, dtype=np.float32)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Expected six finite simulator joint positions")
    return np.rad2deg(values)


def checkpoint_action_to_simulator_radians(action_degrees: np.ndarray) -> np.ndarray:
    """Convert the six normalized, unnormalized checkpoint targets to radians."""
    values = np.asarray(action_degrees, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Expected six finite checkpoint action targets")
    return np.deg2rad(values)


class LeRobotSmolVLA:
    """Adapt the verified checkpoint schema to Sandtable radians and RGB frames."""

    def __init__(self, *, device: str | None = None, cache_dir: str | Path | None = None):
        cache = Path(cache_dir or os.environ.get("SANDTABLE_MODEL_CACHE", ".model-cache")).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_CACHE", str(cache))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla import SmolVLAPolicy
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "SmolVLA dependencies are missing from this server process. Install with "
                "`uv pip install --python .venv-policy\\Scripts\\python.exe -e \".[smolvla]\"`, "
                "then launch the app using `.venv-policy\\Scripts\\python.exe main.py`."
            ) from exc

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s", MODEL_ID, self.device)
        checkpoint = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=str(cache),
            allow_patterns=[
                "config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
                "policy_preprocessor_step_5_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            ],
        )
        self.model = SmolVLAPolicy.from_pretrained(checkpoint, device=self.device)
        self.model.eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def act(self, observation: PolicyObservation) -> np.ndarray:
        """Infer one chunked degree target, converted to simulator radians."""
        if observation["joint_names"] != EXPECTED_JOINTS:
            raise ValueError(f"Checkpoint joint order mismatch: {observation['joint_names']}")
        if set(("top", "wrist")) - observation["images"].keys():
            raise ValueError("SmolVLA requires rendered top and wrist camera frames")
        torch = self._torch
        state_deg = simulator_state_to_checkpoint_degrees(observation["joint_state"])
        batch: dict[str, object] = {
            "observation.state": torch.from_numpy(state_deg).unsqueeze(0),
            "task": [observation["prompt"]],
        }
        for name in ("top", "wrist"):
            rgb = np.asarray(observation["images"][name])
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"Camera {name} must be an HWC uint8 RGB image")
            image = ImageResize.resize_640x480(rgb)
            batch[f"observation.images.{name}"] = (
                torch.from_numpy(image).to(torch.float32).div_(255).permute(2, 0, 1).unsqueeze(0)
            )
        prepared = self.preprocess(batch)
        with torch.inference_mode():
            action = self.postprocess(self.model.select_action(prepared))
        values = action.detach().to("cpu").numpy().reshape(-1)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"Model returned an invalid action vector: {values.shape}")
        # The checkpoint's SO-100 dataset stores all six position targets in degrees.
        return checkpoint_action_to_simulator_radians(values)

    def reset(self) -> None:
        """Clear LeRobot's queued chunk when a policy run starts again."""
        self.model.reset()


class ImageResize:
    """Small import-on-demand image utility for the checkpoint's 640x480 input."""

    @staticmethod
    def resize_640x480(rgb: np.ndarray) -> np.ndarray:
        """Resize RGB without changing channel order or returning shared storage."""
        from PIL import Image

        return np.asarray(Image.fromarray(rgb).resize((640, 480), Image.Resampling.BILINEAR), dtype=np.uint8).copy()


class PolicyRunner:
    """Run one policy on a worker thread; physics and the ASGI loop stay independent."""

    def __init__(self, sim: Sim, cameras: object, state: AppState,
                 policy_factory: Callable[[], Policy] = LeRobotSmolVLA):
        self.sim = sim
        self.cameras = cameras
        self.state = state
        self.policy_factory = policy_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._policy: Policy | None = None

    def start(self) -> str:
        """Start or report the current policy worker without blocking on model load."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return "already_running"
            self._stop.clear()
            self.state.set_policy(running=True, status="loading")
            self._thread = threading.Thread(target=self._run, name="smolvla-inference", daemon=True)
            self._thread.start()
        return "loading"

    def stop(self, *, join_timeout: float = 0.0) -> None:
        """Request a safe stop and hold the current arm position immediately."""
        self._stop.set()
        self.sim.hold_arm_position()
        self.state.set_policy(running=False, status="stopping")
        thread = self._thread
        if thread is not None and thread is not threading.current_thread() and join_timeout > 0:
            thread.join(join_timeout)
        if thread is None or not thread.is_alive():
            self.state.set_policy(running=False, status="stopped")

    def _run(self) -> None:
        try:
            if self._policy is None:
                self._policy = self.policy_factory()
            if self._stop.is_set():
                self.state.set_policy(running=False, status="stopped")
                return
            reset = getattr(self._policy, "reset", None)
            if reset is not None:
                reset()
            device = getattr(self._policy, "device", None)
            self.state.set_policy(running=True, status="running" if device is None else f"running · {device}")
            while not self._stop.is_set():
                observation = self.sim.get_observation()
                # Camera observations are owned copies from CameraRenderer. A model with
                # an empty_camera_0 feature gets that zero view from its saved processor.
                inference_started = time.perf_counter()
                targets = self._policy.act(observation)
                inference_seconds = time.perf_counter() - inference_started
                if self._stop.is_set():
                    break
                self.sim.apply_action(
                    targets,
                    observation_time=observation["simulation_time"],
                    max_observation_age=policy_observation_age_limit(inference_seconds),
                )
                # Bound policy action updates independently of the fixed-rate physics loop.
                self._stop.wait(0.1)
            self.sim.hold_arm_position()
            self.state.set_policy(running=False, status="stopped")
        except Exception as exc:
            logger.exception("SmolVLA worker failed")
            self.sim.hold_arm_position()
            self.state.set_policy(running=False, status=f"error: {exc}")

