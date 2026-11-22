"""HTTP and websocket contract checks for the web transport."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simwrap.server import _mjpeg_frames, create_app
from simwrap.sim import Sim


class EmptyCameras:
    """Camera stub that avoids creating a GL renderer in API tests."""
    cameras = ("front", "top")

    def get_images(self):
        return {name: np.zeros((16, 16, 3), dtype=np.uint8) for name in self.cameras}


def test_scene_shape_and_websocket_pose_frame() -> None:
    sim = Sim()
    app = create_app(sim, cameras=EmptyCameras())
    try:
        with TestClient(app) as client:
            scene = client.get("/api/scene").json()
            assert {"geoms", "body_order", "pool", "collision_geom_rule"} <= scene.keys()
            assert all({"body_name", "type", "size", "pos", "quat", "rgba"} <= g.keys()
                       for g in scene["geoms"])
            feed = client.get("/api/camera/front.jpg")
            assert feed.status_code == 200
            assert feed.headers["content-type"] == "image/jpeg"
            with client.websocket_connect("/ws") as ws:
                frame = ws.receive_bytes()
                assert len(frame) == len(scene["body_order"]) * 7 * 4
    finally:
        sim.close()


def test_mjpeg_camera_stream_yields_live_frame_parts() -> None:
    async def read_first_part() -> bytes:
        stream = _mjpeg_frames(EmptyCameras(), "front")
        first = await anext(stream)
        await stream.aclose()
        return first

    part = asyncio.run(read_first_part())
    assert part.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
    assert b"\xff\xd8" in part and b"\xff\xd9" in part


def test_websocket_spawn_move_delete_and_pool_exhaustion() -> None:
    sim = Sim()
    app = create_app(sim, cameras=EmptyCameras())
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                def send(command):
                    ws.send_text(json.dumps(command))
                    while True:
                        message = ws.receive()
                        if "bytes" in message:
                            continue
                        response = json.loads(message["text"])
                        if response.get("type") in {"ack", "error"}:
                            return response

                response = send({"command": "spawn", "kind": "box", "pos": [0.0, 0.0, 1.2]})
                assert response["type"] == "ack"
                name = response["result"]
                assert send({"command": "move", "name": name, "pos": [0.15, 0.1, 1.2],
                             "quat": [1, 0, 0, 0]})["type"] == "ack"
                assert send({"command": "delete", "name": name})["type"] == "ack"
                for i in range(10):
                    assert send({"command": "spawn", "kind": "sphere",
                                 "pos": [i * 0.08, -0.3, 1.2]})["type"] == "ack"
                exhausted = send({"command": "spawn", "kind": "sphere", "pos": [0, -0.3, 1.2]})
                assert exhausted["type"] == "error"
                assert exhausted["code"] == "pool_exhausted"
    finally:
        sim.close()
