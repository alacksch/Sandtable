# Web UI migration notes

## U1 backend API

- The checkout uses the `src/simwrap/` package layout; implementation stays there rather than moving modules to the planned `src/sandtable/` path.
- Collision-only geoms are skipped when `geom_group == 3`. That is the explicit convention used by the SO-101 asset; groups 0, 1, and 2 remain renderable.
- Pose websocket frames contain seven little-endian float32 values per body, position (xyz) then MuJoCo quaternion (wxyz), in ascending compiled body id order.
- Meshes are served as binary STL. Placement checks temporarily evaluate contact at the candidate pose and restore the prior qpos before returning.
- FastAPI TestClient coverage was added, but this environment does not have FastAPI installed and package installation is blocked by DNS/network access to PyPI. The existing Sim/ObjectManager suite passed with `PYTHONPATH=src pytest -q tests/test_sim.py tests/test_objects.py` (4 passed). Full U1 API tests remain unverified here.

## U2 frontend shell

- Added the Vite + React + TypeScript strict-mode app, with a 3D scene, body pose mutation from binary websocket frames, prompt controls, toolbar, shape tray, and camera panel.
- The renderer uses MuJoCo geom type values verified from installed MuJoCo 4.0.1; mesh geoms load the API's binary STL through Three.js `STLLoader`.
- `npm` is absent from PATH and network access is blocked, so dependency installation and `npm run build` could not run. Package versions are declared in `web/package.json`; verify/build with npm after it is available.

## U3 tray placement

- Tray items can be dragged or selected before clicking the table. The preview raycasts against the table plane, places the shape on the surface, checks the table bounds, and asks the backend to validate contacts before spawning.
- Placement checks temporarily enable the free slot geom's collisions under the Sim lock, then restore its pose and collision masks. A focused test verifies the check does not change body poses.
- Visual overlap feedback is based on the websocket `check_placement` response. The backend still validates again immediately before spawn. Frontend execution/build remains unverified because npm is unavailable.

## U4 select and move

- Selection uses one `selectedName` state. Active object names map to fixed pool slots from websocket state, so clicks and gizmos address each object by its name after any spawn order.
- Added selection outlines, a ghost-only transform target, translation and Z-axis rotation modes, table snap, editable position/yaw, color and size controls, and Delete/Escape handling. New `update_object` commands change pool geom size/color only through ObjectManager and Sim methods.
- The official Drei TransformControls docs/source confirm the `object` ref, `mode`, axis visibility, and mouse/object change callbacks used here. See [TransformControls docs](https://drei.docs.pmnd.rs/gizmos/transform-controls) and [Drei implementation](https://github.com/pmndrs/drei/blob/master/src/core/TransformControls.tsx). The declared frontend is Drei 10.7.6; npm is unavailable here, so TypeScript/build verification remains pending.
- The scripted interleaved move test (spawn A/B/C, move A/B/A, delete B, move C) passes with final pose assertions. The local backend tests pass (7 passed).

### Manual web UI checklist

1. Start the backend and confirm the arm/table render, front/top feeds load, and Run/Stop plus Pause/Resume update status.
2. Drag each tray shape onto the table; verify the preview follows the pointer, turns red over collisions/outside the table, and drop outside cancels. Repeat with click-card then click-table.
3. Select A, B, then A again; move each with the gizmo and inspector, rotate yaw, edit size/color, and confirm other objects keep simulating.
4. Delete a selected object with the button and Delete key; press Escape or click empty space to deselect. Exhaust a shape pool and confirm the card disables.

### Commands

- Install backend dependencies: `python -m pip install -e .`
- Run the production web UI: `python main.py --ui web`
- Run the Vite development UI (in `web/`): `npm install` then `npm run dev`; start the backend separately with `python main.py --ui web`.
- Build static frontend files (in `web/`): `npm install` then `npm run build`; then run `python main.py --ui web` from the repository root. FastAPI serves `web/dist` when present.
