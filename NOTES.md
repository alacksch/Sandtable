# Web UI migration notes

## U1 backend API

- The checkout uses the `src/simwrap/` package layout; implementation stays there rather than moving modules to the planned `src/sandtable/` path.
- Collision-only geoms are skipped when `geom_group == 3`. That is the explicit convention used by the SO-101 asset; groups 0, 1, and 2 remain renderable.
- Pose websocket frames contain seven little-endian float32 values per body, position (xyz) then MuJoCo quaternion (wxyz), in ascending compiled body id order.
- Meshes are served as binary STL. Placement checks temporarily evaluate contact at the candidate pose and restore the prior qpos before returning.
- FastAPI TestClient coverage verifies spawn, move, delete, pool exhaustion, scene JSON, and websocket pose frame length. Full suite: `pytest -q` (9 passed).

## U2 frontend shell

- Added the Vite + React + TypeScript strict-mode app, with a 3D scene, body pose mutation from binary websocket frames, prompt controls, toolbar, shape tray, and camera panel.
- The renderer uses MuJoCo geom type values verified from installed MuJoCo 4.0.1; mesh geoms load the API's binary STL through Three.js `STLLoader`.
- `@types/three` is pinned to `^0.181.0`, matching the published DefinitelyTyped release for Three 0.181.x. `web/package-lock.json` locks the installed frontend dependency tree.
- Frontend dependencies installed and `npm run build` passed. Production UI loaded at `http://localhost:8081`; `/api/scene` returned 49 geoms and 38 bodies, and both camera feeds loaded. Vite dev proxy returned HTTP 200 for `/` and `/api/scene` on port 5173.

## U3 tray placement

- Tray items can be dragged or selected before clicking the table. The preview raycasts against the table plane, places the shape on the surface, checks the table bounds, and asks the backend to validate contacts before spawning.
- Placement checks temporarily enable the free slot geom's collisions under the Sim lock, then restore its pose and collision masks. A focused test verifies the check does not change body poses.
- Visual overlap feedback is based on the websocket `check_placement` response. The backend validates again immediately before spawn. Production UI smoke check spawned and deleted a test box, returning the tray to 0/10.

## U4 select and move

- Selection uses one `selectedName` state. Active object names map to fixed pool slots from websocket state, so clicks and gizmos address each object by its name after any spawn order.
- Added selection outlines, a ghost-only transform target, translation and Z-axis rotation modes, table snap, editable position/yaw, color and size controls, and Delete/Escape handling. New `update_object` commands change pool geom size/color only through ObjectManager and Sim methods.
- The official Drei TransformControls docs/source confirm the `object` ref, `mode`, axis visibility, and mouse/object change callbacks used here. See [TransformControls docs](https://drei.docs.pmnd.rs/gizmos/transform-controls) and [Drei implementation](https://github.com/pmndrs/drei/blob/master/src/core/TransformControls.tsx). A non-null `Group` ref is required by the installed Drei type definitions.
- The scripted interleaved move test (spawn A/B/C, move A/B/A, delete B, move C) passes with final pose assertions. Full backend suite: 9 passed.

### Manual web UI checklist

1. Start the backend and confirm the arm/table render, front/top feeds load, and Run/Stop plus Pause/Resume update status.
2. Drag each tray shape onto the table; verify the preview follows the pointer, turns red over collisions/outside the table, and drop outside cancels. Repeat with click-card then click-table.
3. Select A, B, then A again; move each with the gizmo and inspector, rotate yaw, edit size/color, and confirm other objects keep simulating.
4. Delete a selected object with the button and Delete key; press Escape or click empty space to deselect. Exhaust a shape pool and confirm the card disables.

### Commands

- Install backend dependencies: `python -m pip install -e .`
- Run the production web UI: `python main.py --ui web --port 8081` (8080 is reserved on this machine).
- Run the Vite development UI: start the backend above, then from `web/` run `set VITE_API_PROXY=http://localhost:8081 && npm run dev` in Command Prompt.
- Build static frontend files (in `web/`): `npm run build`; then run the backend from the repository root. FastAPI serves `web/dist` when present.
