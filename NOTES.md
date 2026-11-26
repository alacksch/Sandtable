# Web UI migration notes

## U1 backend API

- The checkout uses the `src/simwrap/` package layout; implementation stays there rather than moving modules to the planned `src/sandtable/` path.
- Collision-only geoms are skipped when `geom_group == 3`. That is the explicit convention used by the SO-101 asset; groups 0, 1, and 2 remain renderable.
- Pose websocket frames contain seven little-endian float32 values per body, position (xyz) then MuJoCo quaternion (wxyz), in ascending compiled body id order.
- Meshes are served as binary STL. Placement checks temporarily evaluate contact at the candidate pose and restore the prior qpos before returning.
- FastAPI TestClient coverage verifies spawn, move, delete, pool exhaustion, scene JSON, websocket pose cadence/notice/reset behavior, malformed command handling, session round-trip/rollback, and camera image output. Full suite: `pytest -q` (26 passed; one Starlette/httpx deprecation warning; see `ARCHITECTURE_AUDIT.md`).

## U2 frontend shell

- Added the Vite + React + TypeScript strict-mode app, with a 3D scene, body pose mutation from binary websocket frames, prompt controls, toolbar, shape tray, and camera panel.
- The renderer uses MuJoCo geom type values verified from installed MuJoCo 4.0.1; mesh geoms load the API's binary STL through Three.js `STLLoader`.
- `@types/three` is pinned to `^0.181.0`, matching the published DefinitelyTyped release for Three 0.181.x. `web/package-lock.json` locks the installed frontend dependency tree.
- Frontend dependencies installed and `npm run build` passed. Production UI loaded at `http://localhost:8081`; `/api/scene` returned 49 geoms and 38 bodies. Vite dev proxy returned HTTP 200 for `/` and `/api/scene` on port 5173.

## U3 tray placement

- Tray items can be dragged or selected before clicking the table. The preview raycasts against the table plane, places the shape on the surface, checks the table bounds, and asks the backend to validate contacts before spawning.
- Placement checks temporarily enable the free slot geom's collisions under the Sim lock, then restore its pose and collision masks. A focused test verifies the check does not change body poses.
- Visual overlap feedback is based on the websocket `check_placement` response. The backend validates again immediately before spawn. Production UI smoke check spawned and deleted a test box, returning the tray to 0/10.

## U4 select and move

- Selection uses one `selectedName` state. Active object names map to fixed pool slots from websocket state, so clicks and gizmos address each object by its name after any spawn order.
- Added selection outlines, a ghost-only transform target, translation and Z-axis rotation modes, table snap, editable position/yaw, color and size controls, and Delete/Escape handling. New `update_object` commands change pool geom size/color only through ObjectManager and Sim methods.
- The official Drei TransformControls docs/source confirm the `object` ref, `mode`, axis visibility, and mouse/object change callbacks used here. See [TransformControls docs](https://drei.docs.pmnd.rs/gizmos/transform-controls) and [Drei implementation](https://github.com/pmndrs/drei/blob/master/src/core/TransformControls.tsx). A non-null `Group` ref is required by the installed Drei type definitions.
- The scripted interleaved move test (spawn A/B/C, move A/B/A, delete B, move C) passes with final pose assertions. Full backend suite: 10 passed.

## Current direction and cleanup

- The duplicate legacy `AppState` implementation was removed; viser and web now share `simwrap.app_state.AppState`. FastAPI camera cleanup uses a lifespan handler.
- The camera panel now reads `/api/camera/{name}.mjpg`, a continuous 10 Hz MJPEG stream. The `.jpg` endpoint remains for one-off snapshots.
- The `IN VIEW` panel lists active object bounds intersecting the Three.js camera frustum and selects an object when its row is clicked. Its React state changes only when the visible-name set changes.
- Future policy control should use `Sim.get_observation()` and `Sim.apply_action()` through the existing application state boundary. The current `run` command only updates status.
- `Sim` keeps MuJoCo model/data private, protects access with one lock, serializes lifecycle operations, reports worker errors, and rolls back failed placement/activation/geometry changes. `CameraRenderer` captures synchronized named frames on its own worker and exposes copied frames.
- Web transport rejects malformed websocket messages, serves only the requested mesh as binary STL, and reuses encoded camera frames per published generation. Browser requests have timeouts, reconnect backoff, and disconnect cleanup; rendering failures inside the Canvas are isolated behind a recovery view.
- Browser smoke checks at 1280×720 and 390×844 confirmed scene rendering, placement, selection, inspector, and no console errors. At 390×844, session controls did not overlap or cause horizontal overflow; Save downloaded a JSON file and Load restored it. The smoke test exported `C:\Users\Haitham\Downloads\sandtable-session.json`. This does not substitute for an Electron check or longer soak. See `ARCHITECTURE_AUDIT.md` for ownership, wire contracts, model constraints, and remaining gates.
- Versioned scene sessions are available at `GET /api/session` and `PUT /api/session`, with Save/Load controls in the browser. They preserve prompt and named object pose/size/color; robot joint state and object velocities are deliberately not captured. Loads validate through `ObjectManager`/`Sim` and restore the prior object scene if a candidate placement fails. Automated coverage includes a round trip and rejected-load rollback.
- `scripts/soak_sim.py --seconds 30` runs concurrent object-edit and pool-reuse workers against the fixed-rate simulation, reports bounded latency/physics metrics, checks state finiteness, and confirms worker shutdown. A 5-second local run completed 8,065 object operations: p50 1.11 ms, p95 2.37 ms, max 33.60 ms; physics simulation time advanced at 0.991× wall time on average. This does not include API or camera contention and is a short smoke soak, not a long-run acceptance result.
- `scripts/soak_server.py --seconds 30` stresses HTTP scene/camera routes, WebSocket commands/pose frames, real camera rendering, and physics together. The 5-second smoke run completed 1,138 WebSocket commands and 426 HTTP requests; ack p95 3.13 ms, HTTP p95 12.45 ms, mean physics-time ratio 0.942×, finite state, and clean shutdown. A separate integrated 30-second run is recorded below.
- Web app recovery returns active objects whose center falls below z=0.25 m to the tray every 250 ms and sends connected clients a notice. Reset clears objects and resets the robot, stops the task-request flag, and preserves the prompt and paused/running simulation state.
- The completed 30-second integrated soak processed 6,637 WebSocket commands, 2,487 HTTP requests, and 899 pose frames; ack p95 3.49 ms, HTTP p95 12.63 ms, physics-time ratio 0.968×, finite state, no worker errors, and clean shutdown.
- `POLICY_CONTRACT.md` documents the `PolicyObservation` and action range contract. `HoldPositionPolicy` remains a deterministic fixture for tests.
- The current object pool still spawns primitives. The renderer already handles mesh geoms; a future downloaded-object catalog/import step should provide stable object names and mesh assets through the existing scene/mesh API. Online asset import and model integration are not part of this cleanup.

### Manual web UI checklist

1. Start the backend and confirm the arm/table render, front/top/wrist feeds load, and Run/Stop plus Pause/Resume update status.
2. Drag each tray shape onto the table; verify the preview follows the pointer, turns red over collisions/outside the table, and drop outside cancels. Repeat with click-card then click-table.
3. Select A, B, then A again; move each with the gizmo and inspector, rotate yaw, edit size/color, and confirm other objects keep simulating.
4. Delete a selected object with the button and Delete key; press Escape or click empty space to deselect. Exhaust a shape pool and confirm the card disables.
5. Confirm `IN VIEW` changes as objects enter and leave the camera frustum, and that front/top/wrist feeds keep refreshing while the camera panel is open.
6. Save a session, clear/reset the scene, load the saved JSON, and verify object names, positions, sizes, colors, and prompt return. Try a JSON file with overlapping objects and confirm it is rejected while the prior scene remains intact.

### Arm controls and SmolVLA pilot

- Added `/api/arm`, direct six-joint sliders (radians in the UI/API), live joint/target telemetry, and a hold-current command. Pause, reset, direct arm commands, and Stop halt policy control before changing targets.
- Added a wrist-mounted MuJoCo camera and feed. The model adapter supplies real `top` and `wrist` RGB frames at 640×480, lets the saved SmolVLA config create its masked `empty_camera_0`, maps the checkpoint's six degree-valued joint states/actions to the simulator's ordered radians, and consumes the checkpoint's queued action chunk before requesting the next one.
- Run/Stop now uses a dedicated daemon worker. Checkpoint and processor files load lazily from `Elbarbary/so101_smolvla` into `.model-cache`; the base simulator remains free of ML dependencies. The optional environment is `.venv-policy` and may launch the app with `.venv-policy\Scripts\python.exe main.py --ui web --port 8081`.
- Downloaded the 1,197,789,224-byte checkpoint and processor files to ignored `.model-cache`. The optional environment uses PyTorch 2.11.0+cu130 and detected the GTX 1650. Live websocket Runs with rendered cameras loaded the model on CUDA and changed simulated arm targets; Stop returned policy status to `stopped`. A 4.074 s camera inference exposed that a fixed 2.5 s freshness limit was too tight. The worker now allows measured inference duration plus 750 ms, bounded by an 8 s hard cap, and holds/stops beyond the cap. This is simulation-only and does not establish task success or real-robot safety.
- Latest checks: `pytest -q` passed (30 tests, one Starlette/httpx deprecation warning); `npm run build` passed (Vite reports the existing >500 kB JS chunk advisory). A worker regression test simulates 2.8 s inference and confirms the action applies under the measured-age allowance.

### Commands

- Install backend dependencies: `python -m pip install -e .`
- Install the optional model environment: `uv venv .venv-policy --python python`; `uv pip install --python .venv-policy\Scripts\python.exe -e ".[smolvla]"`
- Run the production web UI: `python main.py --ui web --port 8081` (8080 is reserved on this machine).
- Run the Vite development UI: start the backend above, then from `web/` run `set VITE_API_PROXY=http://localhost:8081 && npm run dev` in Command Prompt.
- Build static frontend files (in `web/`): `npm run build`; then run the backend from the repository root. FastAPI serves `web/dist` when present.
- Run the bounded simulation/object soak from the repository root: `python scripts/soak_sim.py --seconds 30`.
