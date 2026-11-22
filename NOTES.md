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
