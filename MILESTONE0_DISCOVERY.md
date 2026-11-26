# Milestone 0: discovery and architecture note

Date: 2026-11-25

## Checkout state

- Existing implementation is in `src/simwrap/`, with the web frontend in `web/` and the Viser UI retained. The checkout is `main` at `1f7e935` (`feat: stream cameras and list in-view objects`), seven commits ahead of `origin/main`.
- `git status` initially showed the user-provided `PROJECT_RESTART_BRIEF.md` as untracked. No implementation files were changed for this discovery milestone.
- The supplied `AGENTS.md` stage status is stale: U1–U4 and additional camera/in-view work are present, as `NOTES.md` also records. Continue from this checkout; do not restart or overwrite it without a separately agreed migration plan.

## Asset and robot contract

- `assets/so101/README.md` says the MJCF/URDF files were generated using onshape-to-robot from an Onshape CAD model and modified for relative mesh paths; base collision meshes were removed. `scene.xml` includes `so101_new_calib.xml`.
- The repository has a root Apache-2.0 license, but the robot asset README does not state an asset-specific license or attribution terms. Treat the asset licensing/provenance as unresolved; the project license alone does not establish the CAD/mesh license.
- The compiled current scene reports six hinge joints and six position actuators in this order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. Joint positions and actuator control ranges are radians, with ranges sourced from `so101_new_calib.xml`. The first five use angular hinge coordinates; the gripper also remains an angular hinge, not LeRobot's documented linear 0=closed, 100=open convention.
- The compiled scene has only `front` and `top` cameras. `assets/so101/so101_new_calib_camera.xml` and the camera URDF show a possible wrist-mounted camera asset, but the current scene does not include it. A checkpoint feature named `empty_camera_0` also has no defined source; do not fill it by duplicating or relabeling another camera.

## Environment inspected

- Python 3.13.17; MuJoCo 4.0.1; NumPy 2.5.4; FastAPI 0.165.0; Uvicorn 0.54.0.
- Node v26.13.1; npm 11.20.0. `npm ls --depth=0` reports React 19.3.1, Three 0.181.2, R3F 9.8.2, Drei 10.7.9, TypeScript 5.9.3, and Vite 7.3.8. `pyproject.toml` and the frontend manifest specify lower-bound/range dependencies; there is no Python lock file. `web/package-lock.json` is present.
- These are the versions in the current environment, not a claim of compatibility across all supported installations.

## Checkpoint metadata reviewed

Public Hugging Face files for `Elbarbary/so101_smolvla` were inspected through the model repository, without downloading checkpoint weights. The model repository reports 33.4 GB total; `model.safetensors` is 1.2 GB. Its small metadata files show:

- `config.json`: 6-wide `observation.state` and `action`; `observation.images.top`, `.wrist`, and `.empty_camera_0` each declared as `(3, 480, 640)`; visual normalization is identity while state/action use mean/std; image resize-with-padding is configured to 512x512; one empty camera; action chunk and action-step counts are 50.
- `policy_preprocessor.json` repeats the camera/state/action feature shapes, applies a tokenizer to task text, and points to a small safetensors file for normalization state. `policy_postprocessor.json` points to a small safetensors unnormalizer state.
- The actual normalization tensors/values, checkpoint revision pin, and supported installed LeRobot API have not been inspected. The model card says robot type `so100`, training dataset `lerobot/svla_so100_pickplace`, and provides no evaluation results. The actual six feature names/order, joint calibration, units, gripper meaning and action semantics still require verification from the dataset/checkpoint normalization metadata and supported LeRobot version before adapter implementation.

## Architecture direction

Preserve the current runnable layers and avoid a wholesale restart. Keep `Sim` as the sole synchronized owner of MuJoCo model/data; object and camera operations continue through Sim methods. Keep FastAPI as transport over those interfaces, and the browser as a renderer/command client. If hardening or policy work proceeds, make lifecycle/state contracts explicit and add a separate `policy` boundary: named copied image observations and timestamped robot state in, validated/rate-limited joint targets out. Run inference in a worker isolated from physics and the web event loop. Keep LeRobot and inference dependencies optional. Require an explicit source for every checkpoint camera input, including the wrist and empty camera, and a tested mapping for joint units/order and gripper conversion before enabling policy control.

## Risks and remaining discovery

1. Asset licensing/attribution needs confirmation from the asset's upstream source.
2. Current front/top views do not satisfy the checkpoint's top/wrist/empty-camera schema. The wrist asset may help with one view; the empty-camera feature remains unexplained.
3. Checkpoint state/action are only documented as six-dimensional vectors. Their order, units, meaning, normalization values, chunk behavior, and SO-100/SO-101 compatibility are not proven by the model card/config shapes.
4. Existing architecture/hardening status should be audited before milestone 7; the migration notes do not document a full thread-safety audit, save/load, fall recovery, soak, or shutdown validation.
5. The environment uses newer package versions than the project's minimum declarations. Reproducibility and supported-version bounds need attention before policy dependencies are introduced.

## UI follow-up reported 2026-11-25

The initial report described the UI as Electron-hosted, but the user clarified there is no Electron app yet and asked to defer it until last. The reported click-to-black-screen/crash and non-responsive sizing are therefore browser UI follow-ups. They are not recorded as resolved: the click crash has not been reproduced in a live browser, and no runtime stack was captured. The frontend now has a responsive layout across narrow widths/heights, a Web Crypto request-ID fallback, a visible runtime error overlay for uncaught errors/rejected promises, and a React/Three Canvas recovery boundary. These improve resilience but do not prove the click crash is fixed. Reproduce the issue in the browser and capture its console/runtime stack before declaring it resolved.

## Verification for this milestone

The current model compiled successfully during introspection and reported `nq=216`, `nu=6`, `ncam=2`. The sandboxed full test run stalls while the Windows asyncio Proactor creates its local socketpair for Starlette `TestClient`; focused object/simulation tests pass in the sandbox. The latest full `pytest -q` run passed all 17 tests in 4.78s (one Starlette/httpx deprecation warning). `npm run build` passes (591 modules transformed); Vite emits a large-chunk warning (1,217.85 kB minified JS, 342.00 kB gzip).

Manual follow-up: start the web UI using the command in `NOTES.md`; confirm the front/top feeds, scene rendering, object tray and controls. No policy compatibility or inference claim is made by this discovery milestone.
