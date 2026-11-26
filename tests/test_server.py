"""HTTP and websocket contract checks for the web transport."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simwrap.objects import ObjectManager
from simwrap.server import _mjpeg_frames, create_app
from simwrap.policy import HoldPositionPolicy
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
            mesh_id = next(g["mesh_id"] for g in scene["geoms"] if "mesh_id" in g)
            mesh = client.get(f"/api/mesh/{mesh_id}")
            assert mesh.status_code == 200
            assert mesh.headers["content-type"] == "model/stl"
            assert len(mesh.content) >= 84
            assert client.get("/api/mesh/99999").status_code == 404
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


def test_session_round_trip_and_invalid_load_preserves_current_scene() -> None:
    sim = Sim()
    app = create_app(sim, cameras=EmptyCameras())
    try:
        with TestClient(app) as client:
            original = {"version": 1, "prompt": "keep the blue block", "objects": [
                {"name": "blue_box", "kind": "box", "position": [-0.2, -0.3, 0.785],
                 "quaternion": [1, 0, 0, 0], "size": [0.04, 0.035, 0.03],
                 "color": [0.1, 0.3, 0.8, 1]},
            ]}
            loaded = client.put("/api/session", json=original)
            assert loaded.status_code == 200
            assert loaded.json()["active_objects"] == ["blue_box"]
            saved = client.get("/api/session").json()
            assert saved["version"] == 1
            assert saved["prompt"] == original["prompt"]
            assert saved["objects"][0]["name"] == "blue_box"
            assert np.allclose(saved["objects"][0]["position"], original["objects"][0]["position"])

            invalid = {"version": 1, "prompt": "replacement", "objects": [
                {"name": "sphere_a", "kind": "sphere", "position": [0, -0.3, 0.85],
                 "quaternion": [1, 0, 0, 0], "size": [0.035], "color": [1, 0, 0, 1]},
                {"name": "sphere_b", "kind": "sphere", "position": [0, -0.3, 0.85],
                 "quaternion": [1, 0, 0, 0], "size": [0.035], "color": [0, 1, 0, 1]},
            ]}
            rejected = client.put("/api/session", json=invalid)
            assert rejected.status_code == 422
            restored = client.get("/api/session").json()
            assert restored["prompt"] == original["prompt"]
            assert [item["name"] for item in restored["objects"]] == ["blue_box"]
    finally:
        sim.close()


def test_arm_endpoint_and_websocket_commands_expose_safe_joint_controls() -> None:
    """Manual controls use named actuator limits and the strict action validator."""
    sim = Sim()
    app = create_app(sim, cameras=EmptyCameras())
    try:
        with TestClient(app) as client:
            arm = client.get("/api/arm")
            assert arm.status_code == 200
            snapshot = arm.json()
            assert [item["joint_name"] for item in snapshot["actuators"]] == [
                "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
            ]
            targets = [sum(item["control_range"]) / 2 for item in snapshot["actuators"]]
            with client.websocket_connect("/ws") as ws:
                ws.receive_bytes()
                ws.send_text(json.dumps({"id": "arm-1", "command": "set_arm", "targets": targets}))
                while True:
                    message = ws.receive()
                    if "bytes" in message:
                        continue
                    response = json.loads(message["text"])
                    if response.get("id") == "arm-1":
                        assert response["type"] == "ack"
                        assert np.allclose(response["result"], targets)
                        break
                assert np.allclose(sim.get_state_snapshot()["ctrl"], targets)
                ws.send_text(json.dumps({"id": "arm-bad", "command": "set_arm", "targets": [0.0] * 5}))
                while True:
                    message = ws.receive()
                    if "bytes" in message:
                        continue
                    response = json.loads(message["text"])
                    if response.get("id") == "arm-bad":
                        assert response["type"] == "error"
                        break
                ws.send_text(json.dumps({"id": "arm-hold", "command": "hold_arm"}))
                while True:
                    message = ws.receive()
                    if "bytes" in message:
                        continue
                    response = json.loads(message["text"])
                    if response.get("id") == "arm-hold":
                        assert response["type"] == "ack"
                        assert np.allclose(response["result"], sim.get_arm_state())
                        break
    finally:
        sim.close()


def test_websocket_reports_invalid_json_and_non_object_commands() -> None:
    """Malformed messages return structured errors without dropping the socket."""
    sim = Sim()
    app = create_app(sim, cameras=EmptyCameras())
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                for payload, expected_code in (("{", "invalid_json"), ("[]", "invalid_command")):
                    ws.send_text(payload)
                    while True:
                        message = ws.receive()
                        if "bytes" in message:
                            continue
                        response = json.loads(message["text"])
                        if response.get("type") == "error":
                            assert response["code"] == expected_code
                            break
    finally:
        sim.close()


def test_reset_stops_task_request_and_preserves_pause_and_prompt() -> None:
    """Reset clears objects and run state while keeping the pause flag and prompt."""
    sim = Sim()
    app = create_app(sim, cameras=EmptyCameras(), policy_factory=HoldPositionPolicy)
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

                assert send({"command": "set_prompt", "text": "reset the scene"})["type"] == "ack"
                assert send({"command": "run"})["type"] == "ack"
                assert send({"command": "pause"})["type"] == "ack"
                assert send({"command": "spawn", "kind": "box", "pos": [0.0, -0.3, 0.785]})["type"] == "ack"
                assert send({"command": "reset"})["type"] == "ack"
                while True:
                    message = ws.receive()
                    if "bytes" in message:
                        continue
                    state = json.loads(message["text"])
                    if state.get("type") == "state" and state.get("active_objects") == []:
                        assert state["running"] is False
                        assert state["status"] == "paused"
                        assert state["prompt"] == "reset the scene"
                        break
    finally:
        sim.close()


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
                                 "pos": [-0.36 + i * 0.08, -0.3, 1.2]})["type"] == "ack"
                exhausted = send({"command": "spawn", "kind": "sphere", "pos": [0, -0.3, 1.2]})
                assert exhausted["type"] == "error"
                assert exhausted["code"] == "pool_exhausted"
    finally:
        sim.close()


def test_websocket_broadcasts_fallen_object_recovery_notice() -> None:
    """Connected clients receive names of objects returned to the tray."""
    sim = Sim()
    objects = ObjectManager(sim)
    app = create_app(sim, objects=objects, cameras=EmptyCameras())
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                name = objects.spawn("sphere", np.array((0.0, -0.3, 0.785)))
                assert objects.recover_fallen(threshold_z=0.8) == [name]
                while True:
                    message = ws.receive()
                    if "bytes" in message:
                        continue
                    payload = json.loads(message["text"])
                    if payload.get("type") == "notice":
                        assert payload["code"] == "objects_recovered"
                        assert payload["names"] == [name]
                        break
    finally:
        sim.close()


def test_websocket_pose_rate_is_not_accelerated_by_command_bursts() -> None:
    """Rapid commands are acknowledged without emitting one pose frame per command."""
    sim = Sim()
    app = create_app(sim, cameras=EmptyCameras())
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                while True:
                    message = ws.receive()
                    if "text" in message and json.loads(message["text"]).get("type") == "state":
                        break
                request_ids = [f"burst-{index}" for index in range(30)]
                for index, request_id in enumerate(request_ids):
                    ws.send_text(json.dumps({"id": request_id, "command": "set_prompt", "text": str(index)}))
                remaining = set(request_ids)
                pose_frames = 0
                while remaining:
                    message = ws.receive()
                    if "bytes" in message:
                        pose_frames += 1
                        continue
                    if "text" in message:
                        response = json.loads(message["text"])
                        if response.get("type") == "ack" and response.get("id") in remaining:
                            remaining.remove(response["id"])
                assert pose_frames <= 5
    finally:
        sim.close()
