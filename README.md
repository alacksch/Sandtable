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

## Attach a model

The simulator keeps model inference behind a policy adapter and worker. The model receives observations from `Sim.get_observation()` and returns six joint position targets through `Sim.apply_action()`. Keep model loading and inference out of the physics thread; the existing policy worker runs inference separately and the simulator validates target shape, finiteness, range, and observation age.

### Elbarbary SmolVLA checkpoint

The current optional adapter targets `Elbarbary/so101_smolvla`. It loads the checkpoint and processor files lazily into `.model-cache` when Run is pressed. Install its optional dependencies and launch the app with that environment:

```powershell
uv venv .venv-policy --python python
uv pip install --python .venv-policy\Scripts\python.exe -e ".[smolvla]"
.venv-policy\Scripts\python.exe main.py --ui web --port 8080
```

Open the web UI, enter a task prompt, and press **Run**. Press **Stop** to stop inference and hold the arm's current pose. The first Run downloads the required checkpoint and processor artifacts into `.model-cache`; later runs reuse the cache. To change the Hugging Face model ID or adapt a different checkpoint, update `MODEL_ID` and the loading/preprocessing logic in `src/simwrap/policy_runner.py` to match that model's actual config and processor. The adapter currently expects six specific joints, `top` and `wrist` images, and checkpoint actions in degrees; it converts those to the simulator's ordered radians.

### Integrate another model

1. Install the model's runtime dependencies in an isolated environment. Keep large weights and private credentials out of source control; use `.model-cache` or a configured cache directory for downloaded artifacts.
2. Implement the `Policy` interface in `src/simwrap/policy.py`. Its `act(observation)` method should preprocess the prompt, images, and joint state as required by the model, run inference, and return a NumPy vector of six target positions.
3. Add a loader/adapter in `src/simwrap/policy_runner.py` (or a separate module). Make joint order, units, image names/shape/color order, normalization, action chunking, and checkpoint version explicit. Convert outputs to actuator order and radians before returning them.
4. Connect the adapter to the existing policy worker lifecycle: lazy load, Run/Stop, pause/reset handling, and safe hold on errors or shutdown. Do not call MuJoCo directly from the model; use the `Sim` API.
5. Add tests for observation preprocessing, output conversion, invalid/non-finite/out-of-range actions, and stop/error behavior. Then run the browser with the target model in simulation and inspect the arm response before using it for tasks.

`Sim.get_observation()` supplies owned RGB frames (`front`, `top`, `wrist`), six joint positions in radians, joint names/units, simulation time, and the prompt. `Sim.apply_action()` accepts only six finite, in-range position targets in actuator order and can reject stale observations. A model that expects a different number of joints, cameras, units, or action type needs a deliberate mapping/adapter; do not assume matching names mean matching calibration. See [POLICY_CONTRACT.md](POLICY_CONTRACT.md) for the current contract and checkpoint-specific details. Policy execution is for simulation experiments and does not establish real-robot safety.

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
