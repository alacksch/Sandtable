# Sandtable delivery plan

This plan continues from the current `src/simwrap/` implementation. “From scratch” means the product and eventual Electron shell are being built as a new app; it does not mean discarding the existing simulator and browser UI. Keep the simulator usable without ML dependencies or model downloads. Electron is the final packaging milestone and stays out of scope until the web application passes its reliability, latency, and interaction gates.

## Product boundaries

- First deliver a dependable browser workbench: MuJoCo simulation, object editing, camera feeds, and task prompt.
- Keep the policy behind a typed observation/action adapter. The ordinary simulator must never install or load LeRobot, PyTorch, or checkpoint weights.
- Keep simulation stepping independent of API, rendering, and inference workloads. Bound queues and action rates; report overload instead of allowing unbounded latency.
- Preserve the checkpoint facts already discovered: `Elbarbary/so101_smolvla` declares 6 state values, 6 action values, `top`, `wrist`, and `empty_camera_0` images, mean/std state/action normalization, identity visual normalization, and 512×512 resize-with-padding. Its repository is 33.4 GB and the model file is 1.2 GB. The model card calls it `so100`; do not infer SO-101 compatibility, action order/units, gripper conversion, or camera equivalence from names/shapes alone.

## Milestones and gates

### 0. Checkout and baseline — complete

- Preserve the current implementation and record actual repository/package/model facts.
- Identify any failing or stalled automated checks before making broad changes.
- Gate: clean baseline report and architecture inventory recorded in `PROJECT_RESTART_BRIEF.md` and `ARCHITECTURE_AUDIT.md`; no destructive restart or asset replacement.

### 1. Simulation reliability

- Audit `Sim` lock ownership, lifecycle (start/pause/resume/reset/close), fixed-rate stepping, and thread joining.
- Ensure reset and failed placement restore all state, and camera rendering returns owned frames without starving physics.
- Gate: focused lifecycle, isolation, failure-restoration, and shutdown tests pass; physics cadence remains bounded under API/camera load.

### 2. API and streaming reliability

- Audit validation, structured errors, ack timeouts, WebSocket disconnect/reconnect behavior, and bounded work per client.
- Make stream format/order explicit and test command acknowledgements and frame dimensions. Keep cameras at a measured, bounded rate and avoid per-client duplicate renders where safe.
- Gate: API/websocket tests pass repeatedly; no stuck requests, unbounded pending work, or leaked camera/physics threads.

### 3. Browser UI interaction and design

- Finish responsive layouts, connection/loading/error states, selection/edit behavior, tray placement, and camera panel at desktop, tablet, and narrow-window sizes.
- Keep per-frame pose updates in Three.js refs; avoid React state updates at simulation frame rate. Keep expensive geometry/material creation stable and measure render/update costs before optimizing.
- Gate: production build passes; manual interaction checklist passes at representative sizes; all click, drag, keyboard, and command failures remain visible without a blank screen.

### 4. Scene and session hardening — complete

- Implemented versioned JSON session GET/PUT and browser Save/Load controls. Sessions preserve prompt and active object names, poses, sizes, and colors. Loads validate backend placement and roll back the prior object set on rejection; tests cover round-trip and rollback.
- Browser smoke check exercised Save then Load at 390×844; pool state restored and Reset cleared the test scene, with no console errors or horizontal overflow.
- Defined reset behavior: reset clears pooled objects, resets the robot, stops the task-request flag, and preserves prompt/pause state. Web lifespan now returns objects below z=0.25 m to their slots every 250 ms and sends a visible notice to connected clients.
- Pose streaming now has a fixed 30 Hz scheduler independent of command bursts. RGB rendering prepares the scene under the Sim lock and performs GL draw outside it after confirming the installed MuJoCo 4.0.1 renderer implementation does not read MjData in the RGB draw path.
- Added `scripts/soak_sim.py` with bounded metric storage, concurrent edit/pool-reuse workers, finite-state/error checks, and shutdown verification. A 5-second smoke run completed 8,065 operations (p50 1.11 ms, p95 2.37 ms, max 33.60 ms) with mean physics-time ratio 0.991× and no worker errors. It does not exercise API/camera load and is not a long soak.
- Added `scripts/soak_server.py` for joint API/WebSocket/camera/physics contention. A 5-second smoke run completed 1,138 WebSocket commands, 426 HTTP requests, and 150 pose frames; ack p95 3.13 ms, HTTP p95 12.45 ms, mean physics-time ratio 0.942×, finite state, no worker errors, and clean shutdown.
- Completed the 30-second integrated soak: 6,637 WebSocket commands, 2,487 HTTP requests, 899 pose frames, ack p95 3.49 ms, HTTP p95 12.63 ms, mean physics-time ratio 0.968×, finite state, no worker errors, and clean shutdown.
- Gate: round-trip and invalid-file tests; interleaved object edits; a documented soak run with bounded memory, threads, and latency.

### 5. Policy contract and offline/mock validation — in progress

- Added typed `PolicyObservation`/`Policy` interfaces, a deterministic hold-position mock, named actuator/joint metadata, explicit radians/order/ranges, strict action validation, optional stale-observation rejection, and a hold-current-pose safe-stop primitive. See `POLICY_CONTRACT.md`.
- No inference worker or real checkpoint adapter is connected. Cancellation and controller lifecycle integration remain before inference.
- Gate: mock-policy tests validate joint order, owned output, range and stale-action rejection, and safe stop with no model installed; cancellation awaits the future worker.

### 6. Checkpoint compatibility study

- Pin the intended LeRobot version and inspect the exact checkpoint config, preprocessing/postprocessing, normalization tensors, and training feature metadata. Read only small metadata/stat files first; do not fetch the 1.2 GB weights unless explicitly planned.
- Prove camera sourcing for `top`, `wrist`, and `empty_camera_0`; add a wrist view only with a documented pose and purpose. Never duplicate an image under a different camera name.
- Verify SO-100/SO-101 joint conventions, calibration, gripper mapping, normalization, and action semantics against the actual dataset/checkpoint metadata.
- Gate: written compatibility table and unit-tested adapter transforms; unresolved dimensions block real inference.

### 7. Optional controlled policy runtime

- Add LeRobot/model dependencies only as an optional extra/environment. Load and run policy inference in a cancellable worker with bounded latest-observation/action queues, rate limits, diagnostics, and a safe stop.
- Gate: mock policy first; then separately approved, sized model download and staged simulation validation. No claim of robot readiness from simulation alone.

### 8. Electron packaging — deliberately last

- Only after milestones 1–7 meet their gates, add a minimal Electron host around the proven web app. Keep simulation service lifecycle explicit, secure the renderer boundary, and package the Python runtime predictably.
- Gate: packaged app smoke/soak tests, resizing and click tests, crash diagnostics, clean shutdown, and repeatable installation. Electron is not part of the current implementation phase.

## Current known blockers and follow-up

- The user reports a black screen/crash after clicking in the UI and non-responsive element sizing. Responsive CSS and a runtime error panel/request-ID fallback have been added. On 2026-11-25, a 1280×720 live browser check exercised tray click-to-place, object selection/inspector, Pause/Resume, and Reset; the scene remained visible and browser logs had no warnings or errors. The reported crash was not reproduced in the browser; Electron behavior remains unverified because no Electron app exists. Earlier 390×844 smoke checks found no horizontal overflow. Complete the tablet/narrow-size interaction pass before closing this gate.
- The sandboxed full `pytest -vv -x` run stalls as Windows asyncio creates a local socketpair for Starlette `TestClient`; the exact first affected test is `tests/test_server.py::test_scene_shape_and_websocket_pose_frame`. The object and simulation test files pass inside the sandbox. The latest full suite passes outside the sandbox: 26 passed (one Starlette/httpx deprecation warning). Treat the sandbox stall as a local test-environment restriction, not a reproduced application failure.
- Current UI build passes but the minified JavaScript bundle is 1,217.85 kB (342.00 kB gzip), triggering Vite's 500 kB warning. Measure startup and interaction costs before deciding whether code splitting is worthwhile.
- Asset-specific license terms, model normalization tensors, camera source mapping, and action semantics remain unresolved as recorded in `MILESTONE0_DISCOVERY.md`.

## Working cadence

Work one milestone at a time, keep the app runnable, change the smallest surface that addresses a demonstrated issue, and record evidence. For each milestone report changed files, checks and exact outcomes, known limits, and concrete manual steps. Do not add Electron or load the VLA checkpoint while browser interaction and simulator/API reliability remain unverified.
