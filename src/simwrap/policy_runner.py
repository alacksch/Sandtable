"""Optional, isolated LeRobot model runtime for simulated arm control."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .app_state import AppState
from .policy import Policy, PolicyObservation
from .sim import Sim

logger = logging.getLogger(__name__)
MODEL_ID = "Elbarbary/so101-smolvla-v4-30k"
EXPECTED_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
POLICY_MIN_INFERENCE_AGE_SECONDS = 2.5
POLICY_INFERENCE_AGE_MARGIN_SECONDS = 0.75
POLICY_MAX_INFERENCE_AGE_SECONDS = 8.0
SUPPORTED_TASK_CUES = {
    "Lego sorting": (
        {"lego", "legos", "brick", "bricks"},
        {"sort", "sorting", "put", "place", "move", "pick", "take", "drop", "match"},
    ),
    "desk cleanup": (
        {"desk", "workspace", "bowl", "marker", "keys", "object", "objects", "item", "items"},
        {"clean", "cleanup", "clear", "tidy", "put", "place", "move", "pick", "collect", "sweep",
         "transfer", "grab", "take", "drop", "remove"},
    ),
    "dice throwing": (
        {"dice", "die", "cup"},
        {"roll", "throw", "toss", "pour", "flip", "tip", "empty", "cast", "release", "pick", "grasp"},
    ),
    "fetching a ball": (
        {"ball", "balls"},
        {"fetch", "bring", "retrieve", "put", "carry", "set", "pass", "find", "grab", "pick", "place", "drop"},
    ),
}

CHECKPOINT_CAMERA_MAP = {
    "wrist_left": "wrist",
    "desk_view": "front",
}


def validate_smolvla_prompt(prompt: str) -> None:
    """Reject prompts whose object and action do not match a trained task family."""
    tokens = set(re.findall(r"[a-z]+", prompt.casefold()))
    if not tokens:
        raise ValueError("Enter a task prompt before running the policy")
    if any(objects & tokens and actions & tokens for objects, actions in SUPPORTED_TASK_CUES.values()):
        return
    supported = ", ".join(SUPPORTED_TASK_CUES)
    if "box" in tokens or "boxes" in tokens:
        raise ValueError(
            "This checkpoint has no box task; its fetching demonstrations target a ball. "
            f"Supported task families: {supported}."
        )
    raise ValueError(f"This checkpoint supports {supported}; the prompt does not match one of those tasks.")


def map_checkpoint_camera_images(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map the checkpoint's two trained camera views to Sandtable's RGB feeds."""
    missing = set(CHECKPOINT_CAMERA_MAP.values()) - images.keys()
    if missing:
        raise ValueError(f"SmolVLA requires rendered camera frames: {', '.join(sorted(missing))}")
    mapped: dict[str, np.ndarray] = {}
    for checkpoint_name, source_name in CHECKPOINT_CAMERA_MAP.items():
        rgb = np.asarray(images[source_name])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Camera {source_name} must be an HWC uint8 RGB image")
        mapped[checkpoint_name] = rgb
    return mapped


def clamp_policy_targets(targets: np.ndarray, actuator_info: list[dict[str, object]]) -> np.ndarray:
    """Clamp finite policy predictions to the simulator's configured actuator ranges."""
    values = np.array(targets, dtype=float, copy=True)
    if values.shape != (len(actuator_info),):
        raise ValueError(f"Expected {len(actuator_info)} policy targets, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Policy targets must be finite")
    for index, actuator in enumerate(actuator_info):
        if actuator["limited"]:
            limits = np.asarray(actuator["control_range"], dtype=float)
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] > limits[1]:
                raise ValueError(f"Actuator {actuator['actuator_name']} has invalid control limits")
            values[index] = np.clip(values[index], limits[0], limits[1])
    return values


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
    """Adapt the checkpoint schema to Sandtable observations and actions."""

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
        config_path = Path(checkpoint) / "config.json"
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        # This checkpoint was exported with newer LeRobot-only null metadata fields.
        # Remove those optional fields so the project's pinned LeRobot 0.6.1 can parse it.
        changed = False
        for key in ("dtype", "optimizer_fused"):
            if config_data.get(key) is None and key in config_data:
                del config_data[key]
                changed = True
        if changed:
            config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
        self.model = SmolVLAPolicy.from_pretrained(checkpoint, device=self.device)
        self.model.to(self.device)
        self.model.eval()
        tokenizer_path = snapshot_download(
            repo_id=self.model.config.vlm_model_name,
            cache_dir=str(cache),
            local_files_only=True,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
            ],
        )
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
                "tokenizer_processor": {"tokenizer_name": tokenizer_path},
            },
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def act(self, observation: PolicyObservation) -> np.ndarray:
        """Infer one chunked degree target, converted to simulator radians."""
        if observation["joint_names"] != EXPECTED_JOINTS:
            raise ValueError(f"Checkpoint joint order mismatch: {observation['joint_names']}")
        torch = self._torch
        state_deg = simulator_state_to_checkpoint_degrees(observation["joint_state"])
        batch: dict[str, object] = {
            "observation.state": torch.from_numpy(state_deg).unsqueeze(0),
            "task": [observation["prompt"]],
        }
        # The training dataset has wrist_left and desk_view only. Map its fixed,
        # oblique desk view to Sandtable's closest available camera; omit camera3.
        # SmolVLA's own image path resizes with padding, preserving the 4:3 camera view.
        for source_key, rgb in map_checkpoint_camera_images(observation["images"]).items():
            batch[f"observation.images.{source_key}"] = (
                torch.from_numpy(rgb).to(torch.float32).div_(255).permute(2, 0, 1).unsqueeze(0)
            )
        prepared = self.preprocess(batch)
        with torch.inference_mode():
            action = self.postprocess(self.model.select_action(prepared))
        values = action.detach().to("cpu").numpy().reshape(-1)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"Model returned an invalid action vector: {values.shape}")
        # Preserve the existing degree-to-radian adapter assumption; the new model's
        # metadata does not declare action joint order or units.
        return checkpoint_action_to_simulator_radians(values)

    def reset(self) -> None:
        """Clear LeRobot's queued chunk when a policy run starts again."""
        self.model.reset()


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
        self._log_lock = threading.Lock()
        self._log_path = Path(__file__).resolve().parents[2] / "logs" / "policy-diagnostics.jsonl"

    def get_log_path(self) -> Path:
        """Return the persistent policy diagnostics file path."""
        return self._log_path

    def _log(self, event: str, **details: object) -> None:
        """Record a bounded, JSON-safe policy diagnostic event."""
        entry = {"time": time.strftime("%H:%M:%S"), "event": event, **details}
        with self._log_lock:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                if self._log_path.stat().st_size > 2_000_000:
                    contents = self._log_path.read_bytes()[-1_000_000:]
                    first_line = contents.find(b"\n")
                    self._log_path.write_bytes(contents[first_line + 1:] if first_line >= 0 else contents)
            except OSError:
                logger.exception("Could not persist policy diagnostic log")
        logger.info("Policy diagnostic: %s", entry)

    def start(self) -> str:
        """Start or report the current policy worker without blocking on model load."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return "already_running"
            if self.policy_factory is LeRobotSmolVLA:
                prompt, _ = self.state.snapshot()
                validate_smolvla_prompt(prompt)
            self._stop.clear()
            self.state.set_policy(running=True, status="loading")
            self._log("start_requested")
            self._thread = threading.Thread(target=self._run, name="smolvla-inference", daemon=True)
            self._thread.start()
        return "loading"

    def stop(self, *, join_timeout: float = 0.0) -> None:
        """Request a safe stop and hold the current arm position immediately."""
        self._stop.set()
        self.sim.hold_arm_position()
        self.state.set_policy(running=False, status="stopping")
        self._log("stop_requested")
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
            self._log("started", device=device or "unknown")
            while not self._stop.is_set():
                observation = self.sim.get_observation()
                # Camera observations are owned copies from CameraRenderer and mapped
                # by the checkpoint adapter before they enter the saved processor.
                inference_started = time.perf_counter()
                targets = self._policy.act(observation)
                inference_seconds = time.perf_counter() - inference_started
                predicted_targets = np.asarray(targets, dtype=float)
                applied_targets = clamp_policy_targets(predicted_targets, self.sim.get_actuator_info())
                was_clamped = not np.array_equal(predicted_targets, applied_targets)
                joint_state, _ = self.sim.get_arm_state_snapshot()
                observed_state = np.asarray(observation["joint_state"], dtype=float)
                images = observation.get("images", {})
                self._log(
                    "inference",
                    simulation_time=round(float(observation["simulation_time"]), 3),
                    inference_seconds=round(inference_seconds, 3),
                    prompt=observation.get("prompt", ""),
                    cameras={name: list(image.shape) for name, image in images.items()},
                    observed_joint_state=np.round(observed_state, 4).tolist(),
                    current_joint_state=np.round(joint_state, 4).tolist(),
                    targets=np.round(predicted_targets, 4).tolist(),
                    applied_targets=np.round(applied_targets, 4).tolist(),
                    target_clamped=was_clamped,
                    target_delta=np.round(applied_targets - observed_state, 4).tolist(),
                )
                if self._stop.is_set():
                    break
                self.sim.apply_action(
                    applied_targets,
                    observation_time=observation["simulation_time"],
                    max_observation_age=policy_observation_age_limit(inference_seconds),
                )
                self._log("action_applied", simulation_time=round(self.sim.get_simulation_time(), 3))
                # Bound policy action updates independently of the fixed-rate physics loop.
                self._stop.wait(0.1)
            self.sim.hold_arm_position()
            self.state.set_policy(running=False, status="stopped")
            self._log("stopped")
        except Exception as exc:
            logger.exception("SmolVLA worker failed")
            self._log("error", message=str(exc), error_type=type(exc).__name__)
            self.sim.hold_arm_position()
            self.state.set_policy(running=False, status=f"error: {exc}")

