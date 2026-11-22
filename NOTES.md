# Web UI migration notes

## U1 backend API

- The checkout uses the `src/simwrap/` package layout; implementation stays there rather than moving modules to the planned `src/sandtable/` path.
- Collision-only geoms are skipped when `geom_group == 3`. That is the explicit convention used by the SO-101 asset; groups 0, 1, and 2 remain renderable.
- Pose websocket frames contain seven little-endian float32 values per body, position (xyz) then MuJoCo quaternion (wxyz), in ascending compiled body id order.
- Meshes are served as binary STL. Placement checks temporarily evaluate contact at the candidate pose and restore the prior qpos before returning.
- FastAPI TestClient coverage was added, but this environment does not have FastAPI installed and package installation is blocked by DNS/network access to PyPI. The existing Sim/ObjectManager suite passed with `PYTHONPATH=src pytest -q tests/test_sim.py tests/test_objects.py` (4 passed). Full U1 API tests remain unverified here.
