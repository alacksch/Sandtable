# Sandtable architecture and reliability audit

This records the current implementation after hardening, so follow-up work can target observed gaps without restarting the project or duplicating working layers. Electron and VLA inference remain out of scope until the browser app and simulator gates pass.

## Runtime ownership

- `main.py` selects the web server by default; Viser is loaded only when explicitly requested. The web server binds to loopback.
- `Sim` in `src/simwrap/sim.py` owns MuJoCo model/data privately and serializes access with one lock. The physics worker advances at the model timestep and drops missed deadlines rather than catching up in a burst. Lifecycle operations are serialized; worker errors can be inspected; close is idempotent.
- `ObjectManager` owns the name-to-preallocated-slot registry and calls `Sim` methods for pose, activation, deactivation, and geometry edits. It does not read MuJoCo arrays. Spawn rolls back slot allocation if validation fails.
- `CameraRenderer` owns its renderer on one worker. `Sim` prepares each RGB view from MjData under the lock, then draws the renderer-owned scene outside the lock; this lets physics and API reads proceed during the GL draw. Each view is an individually synchronized snapshot, frames are copied before publishing, and worker errors/lifecycle are tracked.
- `AppState` is the small synchronized prompt/run/status boundary. `run` currently updates application state only; no policy is loaded or executed.
- The offline policy boundary is in `src/simwrap/policy.py`: typed observation/policy interfaces, a hold-position mock, and strict target validation. `Sim.apply_action()` rejects invalid outputs and optionally stale observations; `hold_arm_position()` holds the current pose for safe stop; legacy `set_ctrl()` retains clipping behavior. See `POLICY_CONTRACT.md` for the camera/joint/checkpoint gaps.

## Transport and wire contracts

- FastAPI routes are assembled in `src/simwrap/server.py`; scene and mesh handlers call Sim/ObjectManager APIs rather than accessing MuJoCo internals.
- `/api/scene` reports renderable geoms and body order. The current asset rule omits `geom_group == 3` (collision-only); groups 0–2 render. Mesh geometry is served on demand as binary STL, with only the requested mesh copied and encoded.
- WebSocket pose packets currently contain one body pose per compiled body, ascending body ID: seven little-endian float32 values (position xyz, quaternion wxyz). JSON commands get ack/error responses; malformed JSON and non-object payloads are rejected without dropping the socket.
- Camera snapshots and MJPEG streams reuse JPEG bytes for a published renderer generation so each client does not independently encode the same frame.
- Versioned `/api/session` GET/PUT snapshots the prompt and active object names, world poses, sizes, and colors. Loads are pool-bounded and run through normal placement validation; a failed candidate restores the old active set. Robot joint positions and object velocities are outside session version 1.
- Web app lifespan checks for pooled object centers below z=0.25 m every 250 ms, returns fallen slots to the tray, and emits sequenced WebSocket notices. Reset clears pooled objects and resets the robot, stops the task-request flag, and preserves prompt and pause state.
- Pose frames are scheduled on a fixed ~30 Hz cadence; command bursts are acknowledged independently and cannot cause one frame per command.
- A persistent WebSocket receiver feeds a bounded 64-command queue, avoiding cancel/recreate receive races that could drop commands at pose-frame deadlines.

## Browser rendering and interaction

- `web/src/App.tsx` fetches scene metadata once, builds static geometry, then updates body transforms from websocket frames in the Three.js render loop. Object selection is keyed by stable active names and editing commits through commands.
- WebSocket requests have IDs, bounded timeouts, disconnect cleanup, reconnect backoff, and malformed-pose checks. Prompt writes are debounced. Canvas errors are isolated from the rest of the interface, which shows a recovery panel rather than taking down the controls. The prompt bar labels policy inference as offline; Run currently records a task request only.
- Responsive layout is in `web/src/style.css`; preview validation colors mutate material state rather than triggering a React update each render frame. Repeated mesh references share a loader promise.
- The browser-reported black-screen-on-click issue was not reproduced in a 2026-11-25 live browser pass: tray click-to-place, object selection/inspector, Pause/Resume, and Reset worked at 1280×720; the rendered scene stayed visible and the browser reported no warnings or errors. Error boundaries improve recovery for caught React/Three rendering exceptions, but do not establish Electron reliability. Earlier 390×844 smoke checks found no horizontal overflow; a tablet/narrow interaction pass remains.

## VLA boundary and constraints

- The simulator exposes policy-shaped `get_observation()` and `apply_action()` stubs only. No ML framework, checkpoint, model weights, inference worker, or model download is part of the base app.
- The target checkpoint brief lists six state/action values plus `top`, `wrist`, and `empty_camera_0` images. Current simulated cameras are front/top, and SO-100/SO-101 joint order, units, gripper mapping, image preprocessing, and checkpoint normalization still need verification against actual checkpoint metadata before an adapter can be implemented.
- The checkpoint repository is large; metadata should be inspected selectively before any weight download. Inference must eventually use a bounded worker/queue and must never block the physics thread or API event loop.

## Remaining gates

1. Reproduce and diagnose click-triggered browser failure with console/runtime diagnostics; run the responsive interaction checklist at desktop, tablet, and narrow sizes.
2. Integrate the safe-stop primitive with a cancellable policy-worker lifecycle; stale-action rejection and hold-current-pose behavior are covered, but no inference worker is connected.
3. Verify every target checkpoint camera/joint/normalization/action mapping before implementing a real adapter.
4. Only after those gates, package the web UI with Electron as the final milestone.

## Current verification

- Full backend suite: 26 passed; one upstream Starlette/httpx deprecation warning.
- Frontend production build: passed; output includes a 1,217.85 kB minified JS bundle (342.00 kB gzip), triggering Vite's 500 kB chunk warning.
- Browser smoke check: at 1280×720 and 390×844, the scene rendered, tray placement and object-list selection worked, and the inspector appeared; no browser console errors were observed. The 2026-11-25 desktop check also verified pause/resume/reset and direct object selection with no browser warnings or errors. Save then Load restored the exported session at 390×844, and Reset cleared the smoke scene. The narrow layout had no horizontal overflow. This did not exercise Electron, which does not exist yet, or a long soak.
- Session tests: full suite now covers named-object round trips and failed-load rollback to the prior scene.
- Simulation/object soak smoke: `python scripts/soak_sim.py --seconds 5` completed 8,065 object operations, with p50 1.11 ms / p95 2.37 ms / max 33.60 ms, mean physics-time ratio 0.991×, finite state, no worker errors, and clean worker shutdown. Camera/API contention and longer-duration stability remain unmeasured.
- Integrated 30-second soak: 6,637 WebSocket commands, 2,487 HTTP requests (including 1,658 camera snapshots), and 899 received pose frames. Ack p95 was 3.49 ms; HTTP p95 12.63 ms; mean physics-time ratio 0.968×. State remained finite, both worker errors were null, and shutdown left no workers. This is a bounded stress check, not an overnight soak.
- WebSocket burst regression test confirms a burst of 30 commands does not produce one pose frame per command. RGB renderer lock-release test confirms another thread can read simulation time during the draw.
- Offline policy tests verify joint order, radians, owned hold output, rejection of invalid/range-exceeding/stale targets, and hold-current-pose stop behavior.
