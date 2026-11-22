"""FastAPI transport for the thread-safe MuJoCo simulation."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from .app_state import AppState
from .camera import CameraRenderer
from .objects import ObjectManager
from .scene import OBJECT_POOL_COUNTS
from .sim import Sim


def create_app(sim: Sim | None = None, objects: ObjectManager | None = None,
               cameras: CameraRenderer | None = None, app_state: AppState | None = None) -> FastAPI:
    """Create API routes around an existing Sim and its managers."""
    simulation = sim or Sim()
    manager = objects or ObjectManager(simulation)
    state = app_state or AppState()
    simulation.set_app_state(state)
    camera_renderer = cameras
    if camera_renderer is None:
        camera_renderer = CameraRenderer(simulation)
    app = FastAPI(title="Sandtable API")
    app.state.sim = simulation
    app.state.objects = manager
    app.state.cameras = camera_renderer
    app.state.app_state = state
    app.state.paused = False
    app.state.owns_camera = cameras is None

    @app.get("/api/scene")
    def scene() -> dict[str, Any]:
        """Describe renderable geoms, body frame order, and pool capacity."""
        geoms = simulation.get_geom_visual_data()
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

    @app.get("/api/mesh/{mesh_id}")
    def mesh(mesh_id: int) -> Response:
        """Return a mesh as binary little-endian STL."""
        geoms = simulation.get_geom_visual_data()
        item = next((g for g in geoms if g.get("mesh_id") == mesh_id), None)
        if item is None:
            return Response(status_code=404)
        vertices = np.asarray(item["vertices"], dtype=np.float32)
        faces = np.asarray(item["faces"], dtype=np.uint32)
        stream = io.BytesIO()
        stream.write(b"\0" * 80)
        stream.write(np.uint32(len(faces)).tobytes())
        for face in faces:
            triangle = vertices[face]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            norm = float(np.linalg.norm(normal))
            if norm:
                normal /= norm
            stream.write(np.asarray((*normal, *triangle.flatten()), dtype="<f4").tobytes())
            stream.write(b"\0\0")
        return Response(stream.getvalue(), media_type="model/stl")

    @app.get("/api/camera/{name}.jpg")
    def camera(name: str) -> Response:
        """Return the latest named camera image as JPEG."""
        if name not in camera_renderer.cameras:
            return Response(status_code=404)
        image = camera_renderer.get_images().get(name)
        if image is None:
            return Response(status_code=503)
        from PIL import Image
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=82)
        return Response(buffer.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        last_state: str | None = None
        try:
            while True:
                # Separate sender cadence from command reads, so frames stay near 30 Hz.
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=1 / 30)
                except asyncio.TimeoutError:
                    raw = None
                if raw is not None:
                    await _handle_command(websocket, json.loads(raw), app)
                positions, quaternions = simulation.get_body_poses()
                frame = np.concatenate((positions, quaternions), axis=1).astype("<f4", copy=False)
                await websocket.send_bytes(frame.tobytes())
                prompt, running = state.snapshot()
                active = manager.list_active()
                status = "paused" if app.state.paused else "running"
                serialized = json.dumps({"type": "state", "active_objects": active,
                                         "prompt": prompt, "running": running, "status": status})
                if serialized != last_state:
                    await websocket.send_text(serialized)
                    last_state = serialized
        except WebSocketDisconnect:
            return

    dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if dist.is_dir():
        @app.get("/{asset_path:path}")
        def static_files(asset_path: str) -> FileResponse:
            candidate = (dist / asset_path).resolve()
            if not candidate.is_relative_to(dist.resolve()) or not candidate.is_file():
                candidate = dist / "index.html"
            return FileResponse(candidate)

    @app.on_event("shutdown")
    def shutdown() -> None:
        if app.state.owns_camera:
            camera_renderer.close()

    return app


async def _handle_command(websocket: WebSocket, command: dict[str, Any], app: FastAPI) -> None:
    """Dispatch one JSON command and return a structured ack or error."""
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
        elif name == "delete":
            objects.delete(str(command["name"]))
            result = command["name"]
        elif name == "clear_all":
            objects.clear_all()
            result = None
        elif name == "reset":
            objects.clear_all()
            sim.reset()
            result = None
        elif name == "pause":
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
        elif name == "run":
            state.set_running(True)
            result = None
        elif name == "stop":
            state.set_running(False)
            result = None
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
