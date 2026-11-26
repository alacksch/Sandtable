"""Stress the web API, websocket, camera renderer, and physics worker together."""

import argparse
import json
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simwrap.server import create_app
from simwrap.sim import Sim


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile, or zero when there are no samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def run_soak(duration: float) -> dict[str, object]:
    """Exercise concurrent physics, named camera rendering, API and websocket work."""
    sim = Sim()
    sim.start()
    app = create_app(sim)
    cameras = app.state.cameras
    http_seconds: deque[float] = deque(maxlen=10_000)
    http_by_route: dict[str, deque[float]] = {}
    ack_seconds: deque[float] = deque(maxlen=10_000)
    pose_frame_times: deque[float] = deque(maxlen=10_000)
    physics_ratio: deque[float] = deque(maxlen=10_000)
    request_count = 0
    command_count = 0
    camera_responses = 0
    camera_generations: dict[str, int] = {}
    worker_error: str | None = None
    result: dict[str, object] = {}
    try:
        with TestClient(app) as client:
            camera_deadline = time.perf_counter() + 15
            while any(cameras.get_image(name) is None for name in cameras.cameras):
                if cameras.get_error() or time.perf_counter() >= camera_deadline:
                    raise RuntimeError(f"Camera renderer did not produce frames: {cameras.get_error()}")
                time.sleep(0.05)
            scene_response = client.get("/api/scene")
            if scene_response.status_code != 200:
                raise RuntimeError(f"Scene endpoint returned {scene_response.status_code}")

            with client.websocket_connect("/ws") as websocket:
                sequence = 0

                def send_command(command_name: str, **payload: object) -> object:
                    nonlocal sequence, command_count
                    sequence += 1
                    request_id = f"soak-{sequence}"
                    started = time.perf_counter()
                    websocket.send_text(json.dumps({"id": request_id, "command": command_name, **payload}))
                    while True:
                        message = websocket.receive()
                        if "bytes" in message:
                            pose_frame_times.append(time.perf_counter())
                            continue
                        response = json.loads(message["text"])
                        if response.get("id") == request_id and response.get("type") in {"ack", "error"}:
                            ack_seconds.append(time.perf_counter() - started)
                            command_count += 1
                            if response["type"] != "ack":
                                raise RuntimeError(f"WebSocket {command_name} failed: {response}")
                            return response.get("result")

                object_name = send_command("spawn", kind="box", pos=(0.0, -0.3, 0.785))
                started_wall = time.perf_counter()
                deadline = started_wall + duration
                previous_wall = started_wall
                previous_sim = sim.get_simulation_time()
                index = 0
                while time.perf_counter() < deadline:
                    x = 0.015 if index % 2 else -0.015
                    send_command("move", name=object_name, pos=(x, -0.3, 0.785), quat=(1, 0, 0, 0))
                    index += 1
                    if index % 8 == 0:
                        for path in ("/api/scene", "/api/camera/front.jpg", "/api/camera/top.jpg"):
                            request_started = time.perf_counter()
                            response = client.get(path)
                            elapsed = time.perf_counter() - request_started
                            http_seconds.append(elapsed)
                            http_by_route.setdefault(path, deque(maxlen=10_000)).append(elapsed)
                            request_count += 1
                            if response.status_code != 200:
                                raise RuntimeError(f"{path} returned {response.status_code}")
                            if path.endswith(".jpg"):
                                camera_responses += 1
                                name = path.rsplit("/", 1)[1].split(".", 1)[0]
                                frame = cameras.get_image(name)
                                if frame is not None:
                                    camera_generations[name] = frame[0]
                    now = time.perf_counter()
                    current_sim = sim.get_simulation_time()
                    physics_ratio.append((current_sim - previous_sim) / (now - previous_wall))
                    previous_wall, previous_sim = now, current_sim
                send_command("delete", name=object_name)
                final_state = sim.get_state_snapshot()
                worker_error = sim.get_worker_error()
                camera_error = cameras.get_error()
                frame_samples = list(pose_frame_times)
                frame_intervals_ms = [(later - earlier) * 1000
                                      for earlier, later in zip(frame_samples, frame_samples[1:])]
                result = {
                    "duration_seconds": duration,
                    "websocket_commands": command_count,
                    "pose_frames_received": len(pose_frame_times),
                    "pose_frame_interval_ms": {
                        "p50": percentile(frame_intervals_ms, 0.50),
                        "p95": percentile(frame_intervals_ms, 0.95),
                        "min": min(frame_intervals_ms, default=0.0),
                    },
                    "http_requests": request_count,
                    "camera_responses": camera_responses,
                    "camera_generations": camera_generations,
                    "websocket_ack_latency_ms": {
                        "p50": percentile([value * 1000 for value in ack_seconds], 0.50),
                        "p95": percentile([value * 1000 for value in ack_seconds], 0.95),
                        "max": max((value * 1000 for value in ack_seconds), default=0.0),
                    },
                    "http_latency_ms": {
                        "p50": percentile([value * 1000 for value in http_seconds], 0.50),
                        "p95": percentile([value * 1000 for value in http_seconds], 0.95),
                        "max": max((value * 1000 for value in http_seconds), default=0.0),
                    },
                    "http_latency_by_route_ms": {
                        path: {
                            "p50": percentile([value * 1000 for value in values], 0.50),
                            "p95": percentile([value * 1000 for value in values], 0.95),
                            "max": max((value * 1000 for value in values), default=0.0),
                        }
                        for path, values in http_by_route.items()
                    },
                    "physics_time_ratio": {
                        "mean": statistics.fmean(physics_ratio) if physics_ratio else 0.0,
                        "min": min(physics_ratio, default=0.0),
                    },
                    "finite_state": all(np.isfinite(value).all() for value in final_state.values()),
                    "physics_worker_error": worker_error,
                    "camera_worker_error": camera_error,
                }
    finally:
        sim.close()
    result["workers_stopped"] = not any(
        thread.name in {"mujoco-physics", "mujoco-cameras"} and thread.is_alive()
        for thread in threading.enumerate()
    )
    return result


def main() -> None:
    """Parse options, run the bounded server soak, and report failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0, help="Soak duration in seconds (default: 30)")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.seconds > 1800:
        parser.error("--seconds must be positive and no more than 1800")
    result = run_soak(args.seconds)
    print(json.dumps(result, indent=2))
    if (not result["finite_state"] or result["physics_worker_error"] or
            result["camera_worker_error"] or not result["workers_stopped"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
