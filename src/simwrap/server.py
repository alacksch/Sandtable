"""FastAPI transport for the thread-safe MuJoCo simulation."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image

from .app_state import AppState
from .camera import CameraRenderer
from .objects import ObjectManager
from .policy import Policy
from .policy_runner import PolicyRunner
from .scene import OBJECT_POOL_COUNTS
from .sim import Sim

logger = logging.getLogger(__name__)


def _jpeg_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB camera frame as JPEG."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=82)
    return buffer.getvalue()


def _stl_bytes(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Encode mesh triangles as binary little-endian STL using vectorized arrays."""
    triangles = np.asarray(vertices[faces], dtype="<f4")
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    np.divide(normals, lengths[:, None], out=normals, where=lengths[:, None] > 0)
    records = np.zeros(len(faces), dtype=np.dtype([
        ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2"),
    ]))
    records["normal"] = normals
    records["vertices"] = triangles
    return b"\0" * 80 + np.asarray(len(faces), dtype="<u4").tobytes() + records.tobytes()


async def _mjpeg_frames(cameras: CameraRenderer, name: str,
                        encoded_frame: Callable[[str], bytes | None] | None = None) -> AsyncIterator[bytes]:
    """Yield the latest named camera frame as an MJPEG multipart part."""
    while True:
        frame = encoded_frame(name) if encoded_frame is not None else None
        if encoded_frame is None:
            image = cameras.get_images().get(name)
            frame = None if image is None else _jpeg_bytes(image)
        if frame is None:
            await asyncio.sleep(0.1)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
               + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
               + frame + b"\r\n")
        await asyncio.sleep(0.1)


def create_app(sim: Sim | None = None, objects: ObjectManager | None = None,
               cameras: CameraRenderer | None = None, app_state: AppState | None = None,
               policy_factory: Callable[[], Policy] | None = None) -> FastAPI:
    """Create API routes around an existing Sim and its managers."""
    simulation = sim or Sim()
    manager = objects or ObjectManager(simulation)
    state = app_state or AppState()
    simulation.set_app_state(state)
    camera_renderer = cameras
    if camera_renderer is None:
        camera_renderer = CameraRenderer(simulation)
    policy_runner = PolicyRunner(simulation, camera_renderer, state,
                                  **({} if policy_factory is None else {"policy_factory": policy_factory}))
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Run object recovery and release API-owned background resources."""
        async def recover_fallen_objects() -> None:
            """Return objects that fell off the table without blocking API I/O."""
            while True:
                await asyncio.sleep(0.25)
                try:
                    await asyncio.to_thread(manager.recover_fallen)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Fallen-object recovery pass failed")

        recovery_task = asyncio.create_task(recover_fallen_objects(), name="sandtable-object-recovery")
        try:
            yield
        finally:
            recovery_task.cancel()
            with suppress(asyncio.CancelledError):
                await recovery_task
            policy_runner.stop(join_timeout=0.5)
            if cameras is None:
                camera_renderer.close()

    app = FastAPI(title="Sandtable API", lifespan=lifespan)
    app.state.sim = simulation
    app.state.objects = manager
    app.state.cameras = camera_renderer
    app.state.app_state = state
    app.state.policy_runner = policy_runner
    app.state.paused = False
    jpeg_lock = threading.Lock()
    jpeg_cache: dict[str, tuple[int, bytes]] = {}
    mesh_lock = threading.Lock()
    mesh_cache: dict[int, bytes] = {}

    @app.get("/api/policy/logs/file")
    def policy_logs_file() -> FileResponse:
        """Download the diagnostics file written directly by the policy worker."""
        path = policy_runner.get_log_path()
        if not path.is_file():
            raise HTTPException(status_code=404, detail="No policy diagnostics have been recorded yet")
        return FileResponse(path, media_type="application/x-ndjson",
                            filename="policy-diagnostics.jsonl")

    def encoded_camera_frame(name: str) -> bytes | None:
        """JPEG-encode only once per rendered frame, shared across feed clients."""
        get_image = getattr(camera_renderer, "get_image", None)
        if get_image is None:
            image = camera_renderer.get_images().get(name)
            return None if image is None else _jpeg_bytes(image)
        snapshot = get_image(name)
        if snapshot is None:
            return None
        generation, image = snapshot
        with jpeg_lock:
            cached = jpeg_cache.get(name)
            if cached is not None and cached[0] == generation:
                return cached[1]
            encoded = _jpeg_bytes(image)
            jpeg_cache[name] = (generation, encoded)
            return encoded

    @app.get("/api/scene")
    def scene() -> dict[str, Any]:
        """Describe renderable geoms, body frame order, and pool capacity."""
        geoms = simulation.get_geom_visual_data(include_mesh_data=False)
        pool = simulation.get_object_pool()
        active_slots = manager.active_slot_names()
        return {
            "geoms": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in geom.items() if k not in {"vertices", "faces"}}
                      for geom in geoms],
            "body_order": list(range(simulation.get_body_count())),
            "pool": {kind: {"capacity": count,
                            "names": [str(s["name"]) for s in pool if s["kind"] == kind],
                            "active": [str(s["name"]) for s in pool if s["name"] in active_slots and s["kind"] == kind]}
                     for kind, count in OBJECT_POOL_COUNTS.items()},
            "collision_geom_rule": "Skip geoms with geom_group == 3 (the SO-101 asset collision-only group).",
        }

    @app.get("/api/session")
    def get_session() -> dict[str, Any]:
        """Return a versioned snapshot of active objects and the task prompt."""
        prompt, _ = state.snapshot()
        objects_snapshot = [{key: (value.tolist() if isinstance(value, np.ndarray) else value)
                             for key, value in item.items()}
                            for item in manager.get_session_objects()]
        return {"version": 1, "prompt": prompt, "objects": objects_snapshot}

    @app.get("/api/arm")
    def arm() -> dict[str, Any]:
        """Return actuator names, radians-based target limits, and live arm state."""
        joint_state, joint_targets = simulation.get_arm_control_state()
        actuators = simulation.get_actuator_info()
        return {"actuators": [{**item, "control_range": item["control_range"].tolist()}
                              for item in actuators],
                "joint_state": joint_state.tolist(), "joint_targets": joint_targets.tolist()}

    @app.put("/api/session")
    def put_session(payload: dict[str, Any]) -> dict[str, Any]:
        """Replace active objects from a versioned session, keeping rollback on failure."""
        if type(payload.get("version")) is not int or payload["version"] != 1:
            raise HTTPException(status_code=422, detail="Unsupported session version")
        prompt = payload.get("prompt", "")
        if not isinstance(prompt, str) or len(prompt) > 2000:
            raise HTTPException(status_code=422, detail="Prompt must be a string of at most 2000 characters")
        objects_payload = payload.get("objects")
        if not isinstance(objects_payload, list):
            raise HTTPException(status_code=422, detail="Session objects must be a list")
        if len(objects_payload) > sum(OBJECT_POOL_COUNTS.values()):
            raise HTTPException(status_code=422, detail="Session exceeds total object pool capacity")
        try:
            names = manager.replace_scene(objects_payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        state.set_prompt(prompt)
        return {"version": 1, "active_objects": names}

    @app.get("/api/mesh/{mesh_id}")
    def mesh(mesh_id: int) -> Response:
        """Return a mesh as binary little-endian STL."""
        with mesh_lock:
            cached = mesh_cache.get(mesh_id)
            if cached is not None:
                return Response(cached, media_type="model/stl")
            mesh_data = simulation.get_mesh_data(mesh_id)
            if mesh_data is None:
                return Response(status_code=404)
            body = _stl_bytes(*mesh_data)
            mesh_cache[mesh_id] = body
        if not body:
            return Response(status_code=404)
        return Response(body, media_type="model/stl")

    @app.get("/api/camera/{name}.jpg")
    def camera(name: str) -> Response:
        """Return the latest named camera image as JPEG."""
        if name not in camera_renderer.cameras:
            return Response(status_code=404)
        encoded = encoded_camera_frame(name)
        if encoded is None:
            return Response(status_code=503)
        return Response(encoded, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/camera/{name}.mjpg")
    def camera_stream(name: str) -> Response:
        """Stream a continuously refreshed camera feed using MJPEG."""
        if name not in camera_renderer.cameras:
            return Response(status_code=404)
        return StreamingResponse(
            _mjpeg_frames(camera_renderer, name, encoded_camera_frame),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        last_state: str | None = None
        last_recovery_sequence = manager.get_recovery_cursor()
        loop = asyncio.get_running_loop()
        frame_period = 1 / 30
        next_frame = loop.time() + frame_period
        command_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=64)

        async def receive_commands() -> None:
            """Keep one uncancelled websocket read active and bound queued commands."""
            while True:
                await command_queue.put(await websocket.receive_text())

        receiver = asyncio.create_task(receive_commands(), name="sandtable-websocket-receiver")
        try:
            while True:
                # A burst of commands must not increase the pose stream above its fixed cadence.
                if receiver.done():
                    receiver.result()
                timeout = max(0.0, next_frame - loop.time())
                if timeout == 0:
                    try:
                        raw = command_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        raw = None
                else:
                    try:
                        raw = await asyncio.wait_for(command_queue.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        raw = None
                if raw is not None:
                    try:
                        command = json.loads(raw)
                    except json.JSONDecodeError:
                        await websocket.send_json({"type": "error", "id": None,
                                                   "code": "invalid_json", "message": "Command must be valid JSON"})
                    else:
                        await _handle_command(websocket, command, app)
                now = loop.time()
                if now < next_frame:
                    continue
                positions, quaternions = simulation.get_body_poses()
                frame = np.concatenate((positions, quaternions), axis=1).astype("<f4", copy=False)
                await websocket.send_bytes(frame.tobytes())
                prompt, running = state.snapshot()
                active = manager.list_active()
                status = "paused" if app.state.paused else "running"
                object_slots = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in item.items()} for item in manager.list_objects()]
                serialized = json.dumps({"type": "state", "active_objects": active,
                                         "object_slots": object_slots, "prompt": prompt,
                                         "running": running, "status": status,
                                         "policy_status": state.policy_status()})
                if serialized != last_state:
                    await websocket.send_text(serialized)
                    last_state = serialized
                joint_state, joint_targets = simulation.get_arm_control_state()
                await websocket.send_json({"type": "arm", "joint_state": joint_state.tolist(),
                                           "joint_targets": joint_targets.tolist()})
                for sequence, names in manager.get_recovery_events_after(last_recovery_sequence):
                    await websocket.send_json({"type": "notice", "code": "objects_recovered",
                                               "sequence": sequence, "names": names})
                    last_recovery_sequence = sequence
                next_frame += frame_period
                now = loop.time()
                if now >= next_frame:
                    next_frame = now + frame_period
        except WebSocketDisconnect:
            return
        finally:
            receiver.cancel()
            with suppress(asyncio.CancelledError, WebSocketDisconnect):
                await receiver

    dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if dist.is_dir():
        @app.get("/{asset_path:path}")
        def static_files(asset_path: str) -> FileResponse:
            candidate = (dist / asset_path).resolve()
            if not candidate.is_relative_to(dist.resolve()) or not candidate.is_file():
                candidate = dist / "index.html"
            return FileResponse(candidate)

    return app


async def _handle_command(websocket: WebSocket, command: dict[str, Any], app: FastAPI) -> None:
    """Dispatch one JSON command and return a structured ack or error."""
    if not isinstance(command, dict):
        await websocket.send_json({"type": "error", "id": None,
                                   "code": "invalid_command", "message": "Command must be a JSON object"})
        return
    objects: ObjectManager = app.state.objects
    sim: Sim = app.state.sim
    state: AppState = app.state.app_state
    name = command.get("command") or command.get("type")
    request_id = command.get("id")
    try:
        if name == "spawn":
            result = objects.spawn(str(command["kind"]), np.asarray(command["pos"], dtype=float),
                                   None if command.get("quat") is None else np.asarray(command["quat"], dtype=float))
        elif name == "move":
            objects.move(str(command["name"]), np.asarray(command["pos"], dtype=float),
                         np.asarray(command["quat"], dtype=float))
            result = command["name"]
        elif name == "update_object":
            objects.update_visual(str(command["name"]),
                                  None if command.get("size") is None else np.asarray(command["size"], dtype=float),
                                  None if command.get("color") is None else np.asarray(command["color"], dtype=float))
            result = command["name"]
        elif name == "delete":
            objects.delete(str(command["name"]))
            result = command["name"]
        elif name == "clear_all":
            objects.clear_all()
            result = None
        elif name == "reset":
            app.state.policy_runner.stop()
            objects.clear_all()
            sim.reset()
            result = None
        elif name == "pause":
            app.state.policy_runner.stop()
            sim.stop()
            app.state.paused = True
            result = None
        elif name == "resume":
            sim.start()
            app.state.paused = False
            result = None
        elif name == "set_prompt":
            state.set_prompt(str(command.get("text", "")))
            result = None
        elif name == "set_arm":
            app.state.policy_runner.stop()
            sim.apply_action(np.asarray(command["targets"], dtype=float))
            result = sim.get_arm_control_state()[1].tolist()
        elif name == "hold_arm":
            app.state.policy_runner.stop()
            sim.hold_arm_position()
            result = sim.get_arm_control_state()[1].tolist()
        elif name == "run":
            result = app.state.policy_runner.start()
        elif name == "stop":
            app.state.policy_runner.stop()
            result = "stopping"
        elif name == "check_placement":
            objects.check_placement(np.asarray(command["pos"], dtype=float),
                                    np.asarray(command.get("quat", (1, 0, 0, 0)), dtype=float),
                                    kind=command.get("kind"), name=command.get("name"))
            result = True
        else:
            raise ValueError(f"Unknown command: {name}")
        await websocket.send_json({"type": "ack", "id": request_id, "command": name, "result": result})
    except RuntimeError as exc:
        code = "pool_exhausted" if "pool exhausted" in str(exc).lower() else "command_failed"
        await websocket.send_json({"type": "error", "id": request_id, "code": code, "message": str(exc)})
    except KeyError as exc:
        await websocket.send_json({"type": "error", "id": request_id, "code": "unknown_object", "message": str(exc)})
    except (ValueError, TypeError, IndexError) as exc:
        code = "placement_rejected" if name in {"move", "check_placement", "spawn"} else "invalid_command"
        await websocket.send_json({"type": "error", "id": request_id, "code": code, "message": str(exc)})
