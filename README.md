# Sandtable

Sandtable is a MuJoCo workbench for the SO-101 robot arm. It includes a browser-based 3D interface, live simulator cameras, a pool of editable objects, and a legacy Viser interface.

## Requirements

- Python 3.10 or newer
- Node.js and npm (only needed to develop or build the web interface)
- MuJoCo-compatible graphics support for rendering

## Install and run

From the repository root, install the Python package and start the web app:

```powershell
python -m pip install -e .
python main.py --ui web --port 8080
```

Open <http://localhost:8080>. The server binds to loopback by default. To use the legacy Viser interface instead, run:

```powershell
python main.py --ui viser --port 8080
```

To use the Vite development server, install frontend dependencies and run the backend and Vite in separate terminals:

```powershell
cd web
npm ci
$env:VITE_API_PROXY = "http://localhost:8080"
npm run dev
```

The Vite server prints its local URL when it starts. Create a production frontend bundle with `npm run build` from `web/`; the backend serves `web/dist` when that directory exists.

## Optional policy runtime

The base simulator does not require ML packages. An optional SmolVLA environment can be installed with `uv`:

```powershell
uv venv .venv-policy --python python
uv pip install --python .venv-policy\Scripts\python.exe -e ".[smolvla]"
```

The optional runtime downloads model files into `.model-cache` on first use. Check `POLICY_CONTRACT.md` for the observation/action contract and integration constraints. Policy execution is simulation-only.

## Project structure

```text
main.py                 CLI entry point
src/simwrap/            Simulation, scene, object, camera, API, and policy code
web/                    Vite + React + TypeScript interface
assets/so101/           SO-101 MuJoCo model and mesh assets
scripts/                Inspection and soak utilities
tests/                  Backend tests
```

The `Sim` class owns MuJoCo model/data and serializes access with one lock. A fixed-rate worker advances physics. `ObjectManager` edits a preallocated object pool through `Sim` methods; the FastAPI server exposes scene metadata, meshes, cameras, sessions, and websocket controls. The frontend builds static geometry from `/api/scene` and applies pose frames directly to Three.js objects.

## Useful commands

```powershell
pytest -q
python scripts/soak_sim.py --seconds 30
python scripts/soak_server.py --seconds 30
```

Run `npm run build` from `web/` to type-check and build the frontend.

## Current limits

- The browser UI is the primary interface; Viser remains available for compatibility.
- Saved sessions include the prompt and active object poses, sizes, and colors; they do not capture robot joint state or object velocities.
- Model inference and hardware control are outside the simulator's safety guarantees. The optional policy path is intended for simulation experiments only.
- See [NOTES.md](NOTES.md), [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md), and [POLICY_CONTRACT.md](POLICY_CONTRACT.md) for implementation history, reliability findings, and policy details.
