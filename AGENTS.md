# Project context: Sandtable

A MuJoCo-based simulation wrapper with a browser UI, built around an SO-101 robot arm
that will later be driven by an ML policy taking text prompts. The ML model is NOT part
of the current work; leave clean interfaces only.

## Stack
- Backend: Python 3.10+, `mujoco` (3.2+ for MjSpec), `numpy`, `fastapi`, `uvicorn`
  (websockets), `pytest`
- Frontend: `web/` with Vite + React + TypeScript + three.js, using
  `@react-three/fiber` and `@react-three/drei`
- Legacy: the viser UI (`ui.py`) stays working behind `--ui viser` until the web UI
  fully replaces it. Do not extend it.

## Repo layout
- `src/sandtable/`: `sim.py`, `scene.py`, `objects.py`, `camera.py`, `controller.py`,
  `server.py`, `ui.py` (legacy viser)
- `web/`: frontend
- `main.py`: entry point (`--ui web|viser`)
- `assets/so101/`: robot assets
- `scripts/`, `tests/`

## Backend architecture rules (always follow)
1. `Sim` owns MjModel/MjData. ALL reads/writes of model/data happen inside Sim methods
   under one `threading.Lock`. Nothing else touches model/data directly, including the
   server, ObjectManager internals, and camera code (render under the lock or on a
   data copy).
2. Physics steps in a fixed-rate background thread.
3. MuJoCo models are compiled/static. Objects use a pre-allocated pool (each with a
   freejoint, parked off-table with collisions disabled: contype/conaffinity = 0), with
   an optional recompile-with-state-transfer fallback via MjSpec (transfer by joint
   name, never by index).
4. Move/commit: write freejoint qpos, zero qvel, call mj_forward, all under the lock.
5. Object registry maps name -> body/joint/geom ids. Never hardcode ids.
6. Future policy interface (stubs only for now):
   `get_observation() -> {"images": {...}, "joint_state": np.ndarray, "prompt": str}`
   `apply_action(joint_targets: np.ndarray)`
7. `AppState` (prompt: str, running: bool) is small and thread-safe.

## Server rules (`server.py`)
- FastAPI wrapping the existing Sim, ObjectManager, CameraRenderer, AppState. Do not
  change their lock discipline.
- `GET /api/scene`: one-time JSON of every renderable geom (body name, geom type, size,
  local pos/quat, rgba, mesh id) plus the object pool registry. Collision-only geoms are
  skipped; document the exact rule used.
- `GET /api/mesh/{id}`: binary mesh data (GLB or STL).
- `WebSocket /ws`:
  - server -> client at ~30 Hz: binary float32 frame of all body poses in a fixed order
    matching /api/scene; plus small JSON messages on state change (active objects,
    prompt/running state, status).
  - client -> server JSON commands: `spawn{kind,pos,quat?}`, `move{name,pos,quat}`,
    `delete{name}`, `clear_all`, `reset`, `pause`, `resume`, `set_prompt{text}`, `run`,
    `stop`, optional `check_placement{...}`.
  - Every command returns an ack or a structured error (e.g. pool exhausted,
    placement rejected).
- `GET /api/camera/{name}.jpg` (or MJPEG) for the front/top feeds.
- Commands call ObjectManager/Sim methods only.

## Frontend rules (`web/`)
- Z-up to match MuJoCo. Fetch /api/scene once, build geometry once, update poses from
  binary websocket frames by mutating three.js objects directly in `useFrame`. No React
  re-render per frame.
- Layout (dark theme, rounded panels, subtle blur, no default browser styling):
  - Full-screen 3D canvas.
  - Floating prompt bar, horizontally centered at the top, with Run/Stop buttons and a
    status dot.
  - Packet Tracer-style object tray docked at the bottom: horizontal strip of draggable
    item cards (box, cylinder, sphere) showing pool usage (e.g. 7/10), disabled when
    exhausted.
  - Top-left toolbar: Reset, Pause/Resume.
  - Collapsible camera panel (front/top feeds) at bottom-right above the tray.
  - Right-side inspector appears only when an object is selected.
- Placement: drag a card onto the table to spawn (translucent preview follows the cursor,
  snapped to the table surface, red when overlapping). Click-card-then-click-table also
  works. Dropping outside the table cancels. The backend is the authority on validity.
- **Selection is single-source-of-truth state**: `selectedName: string | null`.
  - Click any object to select it (raycast on pointer-up; ignore if the pointer was
    dragged/orbited). Click empty space or press Esc to deselect.
  - Selection must work for every object at all times, including ones spawned later,
    independent of spawn order or any previously attached gizmo. Never bind
    handlers/gizmos to "the last created object".
  - The gizmo attaches to a ghost at the selected object's pose (translate with
    snap-to-table toggle, plus yaw). Only the ghost moves while dragging; on release send
    `move`. If the backend rejects it, snap the ghost back and show a toast.
  - Inspector: name, kind, editable position/yaw, color, size (if supported), Delete.
    Delete key also deletes.
- Objects keep simulating physically while edited; moving one must not disturb the arm or
  others.

## Conventions
- Type hints, small functions, docstrings on public methods; TypeScript strict mode.
- Verify MuJoCo, FastAPI, three.js, and drei APIs against installed versions (introspect
  or check docs) instead of relying on memory. If an API differs, adapt and say so.
- Keep each task minimal. Do not build features from later stages.
- Always finish a stage by running the tests and stating how to verify manually.
- Stop after the requested stage and report: what was done, deviations, known issues.

## Stage status
- Done: foundation, scene and Sim, viser UI, camera panel, object pool, drag-and-drop (viser).
- Current: web UI rewrite (U1 backend API, U2 frontend shell, U3 tray placement, U4
  select/move anytime).
- Not started: hardening (scene save/load, soak test), mesh import, policy integration.