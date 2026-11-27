# Sandtable

Sandtable is a browser workbench for simulating an SO-101 robot arm in MuJoCo. You can watch the arm move, place and edit objects on the table, view live cameras, and experiment with a text-prompt policy.

## Run it

You need Python 3.10 or newer. From the project folder, create an environment and install the project:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python main.py --ui web --port 8080
```

Open <http://localhost:8080>. The web interface is the default. The older Viser interface is still available with:

```powershell
python main.py --ui viser --port 8080
```

To work on the frontend, install Node.js, then run the backend in one terminal. In a second terminal:

```powershell
cd web
npm ci
$env:VITE_API_PROXY = "http://localhost:8080"
npm run dev
```

`npm ci` installs the locked frontend dependencies. Build the production frontend from `web/` with `npm run build`.

## What happens when it runs

```text
Browser (React + Three.js)  ⇄  FastAPI server  ⇄  Sim (MuJoCo)
                                   │                  │
                         cameras, objects,       fixed-rate physics
                         sessions, commands       and robot state
```

- **Sim** owns MuJoCo’s model and data. A background worker advances physics at a fixed rate, and Sim’s lock protects all model/data access.
- **ObjectManager** controls a preallocated pool of boxes, cylinders, and spheres through Sim methods. The browser can place, select, move, resize, recolor, and delete them.
- **FastAPI** provides scene geometry, mesh data, camera images, session save/load, and a WebSocket for commands and live body poses.
- **The browser** builds the scene once, then updates Three.js poses from WebSocket frames. It also provides the prompt bar, object tray, inspector, and camera panel.
- **The policy worker** runs optional model inference outside the physics thread. Sim validates joint targets before applying them.

## Try a model

The optional adapter supports Elbarbary’s `Elbarbary/so101-smolvla-v4-30k` checkpoint. Install its dependencies in an isolated environment and start Sandtable with that Python:

```powershell
uv venv .venv-policy --python python
uv pip install --python .venv-policy\Scripts\python.exe -e ".[smolvla]"
.venv-policy\Scripts\python.exe main.py --ui web --port 8080
```

In the browser, enter a prompt and press **Run**. The first run downloads model files to `.model-cache`; press **Stop** to stop inference and hold the arm’s current pose. The adapter maps the checkpoint’s six joint targets from degrees to the simulator’s actuator order in radians.

To connect another model, implement the `Policy` interface in `src/simwrap/policy.py` and an adapter/loader alongside `src/simwrap/policy_runner.py`. Convert its inputs and outputs explicitly: camera names and image format, joint order, units, normalization, and action shape must match the model. Keep inference in the policy worker and send targets through `Sim.apply_action()`; don’t access MuJoCo data from model code. See [POLICY_CONTRACT.md](POLICY_CONTRACT.md) for details.

## Project map

```text
main.py                 Start the web or Viser interface
src/simwrap/            Simulation, scene, objects, cameras, API, policy
web/                    React + TypeScript + Three.js frontend
assets/so101/           SO-101 model and mesh assets
scripts/                Model inspection and soak utilities
tests/                  Backend tests
```

## Useful checks

```powershell
pytest -q
python scripts/soak_sim.py --seconds 30
python scripts/soak_server.py --seconds 30
```

## Git setup

The repository ignores virtual environments, downloaded model files, Python caches, and generated frontend output. Keep credentials and machine-specific settings in ignored `.env` files; use `web/.env.example` as the frontend template. `web/package-lock.json` is committed so frontend installs are reproducible.

From `web/`, `npm run build` type-checks and builds the frontend.

## Current limits

- The optional model path is for simulation experiments; it does not establish real-robot safety.
- Saved sessions include the prompt and active object poses, sizes, and colors. They do not include robot joint state or object velocities.
- `NOTES.md` and `ARCHITECTURE_AUDIT.md` describe implementation history and known reliability gaps.
