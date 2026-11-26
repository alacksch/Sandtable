# Sandtable — Project Summary and Restart Prompt

This document is the handoff brief for restarting Sandtable cleanly. It records the product goal, the current checkout, the model constraints, an optimal implementation direction, and a complete prompt that can be given to a coding agent.

## 1. Project in one paragraph

Sandtable is a local, browser-based MuJoCo workbench for simulating an SO-101 robot arm and arranging objects on a table. The user provides a natural-language task, sees the robot and scene through a 3D UI and simulated cameras, and can later run a vision-language-action policy. The current Hugging Face policy is `Elbarbary/so101_smolvla`; real policy execution is a later integration stage, after the simulation, camera observations, robot state/action mapping, and safety controls have been validated. The project should keep the simulator usable without downloading or loading the model.

## 2. Model facts and implications

Model card: <https://huggingface.co/Elbarbary/so101_smolvla>

The model card identifies this as a 0.5B-parameter SmolVLA checkpoint, Apache-2.0, fine-tuned from `lerobot/smolvla_base`. The card labels its robot type `so100` even though the repository/checkpoint name says SO-101. It lists input state `observation.state` with shape `(6,)`, image inputs `observation.images.top`, `observation.images.wrist`, and `observation.images.empty_camera_0`, each `(3, 480, 640)`, and output `action` with shape `(6,)`. The training dataset is `lerobot/svla_so100_pickplace`, 50 episodes, described as “Pick up the cube and place it in the box.” The model card reports no evaluation results.

Treat the listed schema as a starting point to verify against the checkpoint's actual LeRobot config and feature/action normalization metadata before integration. Do not assume the simulator's front/top cameras satisfy the checkpoint schema: there is a wrist camera and a third empty-camera input in the card, and camera keys, image dimensions, color layout, and preprocessing must match. Do not assume six raw MuJoCo joint values and six MuJoCo actuator targets have the same units, order, or normalization as LeRobot state/actions. Verify SO-100 versus SO-101 joint names, calibration, gripper convention, and action semantics. The provided SO-101 asset README says the gripper is not yet mapped to LeRobot's linear convention (`0` closed, `100` open). The exact checkpoint may have additional metadata or differences from the model card; inspect it before designing the adapter.

The most useful separation is a simulator-facing adapter contract and a LeRobot-specific policy adapter, with explicit observation and action transforms. Keep policy inference isolated from the fixed-rate physics thread. Add inference dependencies only in a dedicated optional extra/environment so the basic simulator remains light and easy to install. Use an explicit Run/Stop lifecycle, bounded action updates, model-load errors in the UI, and a safe stop behavior. Do not claim successful policy control until a staged validation confirms schema compatibility and motion behavior.

## 3. Current checkout: what exists

Repository package path is `src/simwrap/` (not the originally planned `src/sandtable/`). The checkout has a single `main` branch at commit `1f7e935` (`feat: stream cameras and list in-view objects`), with prior commits for U1 backend API, U2 frontend shell, U3 tray placement, and U4 selection/move. At inspection, `git status --short` was empty. The supplied `AGENTS.md` stage status is stale: the web migration is implemented through U4, and camera streaming plus in-view object listing are also present.

Implemented pieces:

- MuJoCo SO-101 scene from `assets/so101/scene.xml`, table, floor, lighting, front/top cameras, and a pool of ten each of boxes, cylinders, and spheres.
- `Sim` owns model/data, uses a lock, provides fixed-rate stepping, reset, pose/state access, object-slot operations, camera rendering, and policy-shaped stubs (`get_observation`, `apply_action`).
- `ObjectManager` maps names to pool slots and handles spawn, delete, move, placement validation, listing, and visual properties through `Sim` methods.
- FastAPI app has scene and mesh APIs, websocket pose frames and commands, camera snapshots and MJPEG feeds, state updates, and serves production frontend assets when built.
- Vite/React/TypeScript/Three.js frontend renders MuJoCo scene geometry with Z-up, streams body poses, provides prompt/run/stop and pause/reset controls, drag/click tray placement with backend validation, selection and ghost-based editing, inspector controls, and camera feeds. It also has an in-view object list.
- Legacy Viser UI remains available through `--ui viser`; web UI is selected with `--ui web`.
- Existing tests cover simulation basics, object pool lifecycle, placement non-commit, interleaved name-based editing, visual property updates, API scene/frame/camera behavior, websocket commands, and pool exhaustion.

Important known limits from the current notes and files:

- `run` only changes application status; it does not load or execute a policy.
- Current camera pair is front/top; the model card lists top/wrist/empty-camera inputs. No wrist camera or model schema adapter is established.
- Current pool supports only primitive objects, not asset import/catalog or persistent scene save/load.
- Hardening work (full thread-safety audit, long soak, reset semantics, fall recovery) is not documented as complete. The legacy `AGENTS.md` also asks for JSON scene save/load and graceful shutdown validation.
- The code uses `mujoco>=3.2`, FastAPI, Viser, NumPy, Uvicorn, and a React 19 / Three 0.181 / R3F 9 / Drei 10 frontend. Record exact tested versions in a fresh setup; do not blindly carry forward old pins.
- In the current implementation collision-only render geometry is skipped when `geom_group == 3`; this is a project-specific asset convention and must stay documented if retained.
- Websocket body pose frames use seven little-endian float32 values per body (position xyz then MuJoCo quaternion wxyz) ordered by body ID. This wire format must be explicit/versioned in a rewrite.

## 4. Restart strategy

Restart from a clean, documented architecture rather than copying implementation details wholesale. Preserve the functional requirements below, but first inspect the actual SO-101 asset and LeRobot checkpoint metadata. Build the non-ML simulator and UI as independently testable layers; integrate the policy only after the camera/state/action contract is proven. Prefer a small number of deliberate milestones and a runnable application at each milestone.

Suggested module boundaries:

- `simulation/`: `Sim` is the only owner of `MjModel`/`MjData`; all reads and writes go through methods protected by one lock. Physics has one fixed-rate owner thread. Pausing, reset, snapshot and close are explicit lifecycle operations.
- `scene/`: scene construction and stable configuration/constants; robot asset files remain local and licensed/readme information is preserved.
- `objects/`: registry and object operations call only `Sim` APIs. Prefer a compiled preallocated pool for runtime add/remove. Never store naked body/joint/geom IDs in the HTTP layer.
- `cameras/`: named camera capture and lifecycle. Rendering is serialized safely with simulation access and has a clear image format/resolution contract. Add or simulate a wrist-mounted view only when its camera pose and observation purpose are explicit.
- `api/`: FastAPI transport, schemas, validation, structured errors, and application assembly. It must not reach into MuJoCo model/data.
- `web/`: strict TypeScript React/Vite app, Three.js renderer. Build geometry once from scene metadata and mutate Three.js object transforms from binary pose packets without React renders per physics frame.
- `policy/`: typed observation/action contract, mock policy for integration tests, then optional LeRobot adapter and model metadata validation. Policy code never owns MuJoCo objects.
- `tests/`: pure schema/unit tests, API tests with camera stubs, simulation/object tests, and opt-in/manual browser and model smoke procedures.

## 5. Complete restart prompt

Copy the prompt below into the new project/task. It is intentionally self-contained.

```text
You are building Sandtable from a clean start. Read this entire specification before changing files. Work in small, reviewable milestones; keep the app runnable at the end of every milestone. Do not add features from later milestones early. At the end of each milestone, run the relevant automated checks and give concise manual verification steps. Do not claim unperformed tests or model validation.

PRODUCT
Sandtable is a local browser-based MuJoCo workbench for an SO-101 robot arm. A user can inspect the robot, arrange objects on a table, view simulated cameras, and enter a natural-language task. A vision-language-action (VLA) policy will eventually run the task. The basic simulator and UI must work without downloading or loading any ML model. Keep the VLA behind a clean interface; do not train a model as part of this work.

TARGET MODEL
Hugging Face checkpoint: https://huggingface.co/Elbarbary/so101_smolvla
The model card currently describes a 0.5B SmolVLA checkpoint, Apache-2.0, based on lerobot/smolvla_base. Although the repo is named so101_smolvla, its card says robot type so100. The card lists observation.state (6,), observation.images.top (3,480,640), observation.images.wrist (3,480,640), observation.images.empty_camera_0 (3,480,640), and action (6,). The training dataset is lerobot/svla_so100_pickplace (50 episodes) for “Pick up the cube and place it in the box”; the card provides no evaluation results.
Before implementing the real policy adapter, inspect the exact checkpoint configuration, normalization stats, feature names, expected image preprocessing, joint ordering, units, and action semantics using the supported LeRobot version. Confirm SO-100/SO-101 compatibility; do not infer it from the repository name. The SO-101 assets' README notes that the gripper convention is not yet aligned with LeRobot's linear 0=closed, 100=open representation. Confirm all six state/action dimensions against the actual MuJoCo actuators and calibration. Front/top cameras alone do not satisfy the listed top/wrist/empty-camera inputs. Specify how each required camera is obtained; never silently duplicate or mislabel camera images. Keep policy dependencies optional and isolated from the base simulator install.

TECHNICAL BASELINE
- Python 3.10+ backend: MuJoCo with MjSpec where useful, NumPy, FastAPI, Uvicorn/WebSockets, pytest.
- Frontend: Vite, React, strict TypeScript, Three.js, @react-three/fiber, @react-three/drei.
- Robot assets: official SO-101 MJCF and meshes under assets/so101 with asset license/readme retained.
- The legacy Viser UI may be retained temporarily behind a CLI choice if it is useful, but do not add features to it. Prefer the web UI as the primary interface.
- Verify APIs against the versions installed in the new environment. Pin or lock reproducible dependencies after verifying them. Do not import MuJoCo renderer objects unsafely across threads.

ARCHITECTURE RULES
1. Sim exclusively owns MjModel and MjData. All model/data reads and writes are inside Sim methods and protected by one threading lock. Server, object manager, policy adapter, and camera/UI layers never touch model/data directly.
2. Physics advances on one fixed-rate background thread. Define clean start/pause/resume/reset/close behavior and join threads at shutdown.
3. Prefer a compiled static model and preallocated object pools. Each dynamic object slot uses a freejoint, starts parked out of view with collisions disabled, and is activated/deactivated through Sim methods. If runtime recompilation is truly necessary, use MjSpec and transfer state by joint name, never by index.
4. Committing a moved object writes freejoint qpos, clears qvel, and calls mj_forward under the Sim lock. Candidate placement validation must restore the previous state even on error.
5. Maintain registries mapping stable object names to slot metadata. Never hardcode MuJoCo IDs in API or frontend code.
6. Keep a small thread-safe AppState for prompt, run state, and user-visible status. Policy lifecycle/state belongs behind a separate controller interface.
7. Policy boundary (initially a mock/stub): observation has named camera images, joint state, and prompt; action is a validated vector of joint targets. Define units, order, range, normalization and timestamps explicitly. Policy inference must not block physics or the web event loop.
8. Treat every incoming command as untrusted input: validate command names, array lengths, finite numbers, ranges, object names, and pose/quaternion validity. Return structured acknowledgements/errors. The backend is authoritative for placement validity.

BACKEND/API REQUIREMENTS
- GET /api/scene returns stable, one-time scene metadata: renderable geoms with body name, geom type, dimensions, local position/quaternion, color, mesh reference; body order; object pool information; coordinate convention; collision-geom filtering rule. Document and test the exact filtering rule.
- GET /api/mesh/{id} returns binary mesh data with a documented format/content type.
- WebSocket /ws streams body poses at approximately 30 Hz in a fixed order consistent with /api/scene. Specify a versioned binary format, endianness, field order, and quaternion convention. Send small JSON state messages only when state changes. Commands include spawn, move, delete, clear_all, reset, pause, resume, set_prompt, run, stop, and check_placement. Every command returns an ack or structured error.
- Camera endpoints provide front/top feeds as snapshots or MJPEG. Add wrist camera only after the model camera requirements and camera placement are established.
- Camera rendering and policy observations must obey Sim's lock/lifecycle rules and return owned image copies.
- Keep reset semantics explicit: robot reset and object clear/restore must be predictable and reflected in the UI.

FRONTEND REQUIREMENTS
- Z-up coordinate system matching MuJoCo. Fetch scene metadata once, construct geometry once, then apply incoming body poses by mutating Three.js objects in the render loop. Do not trigger React state updates at the simulation frame rate.
- Full-window 3D canvas with a polished dark interface, restrained rounded translucent panels, and no default browser styling.
- Centered top prompt bar with editable task, Run/Stop, connection and policy/simulation status.
- Upper-left Reset and Pause/Resume controls.
- Bottom horizontal draggable object tray (box/cylinder/sphere) with capacity counts and exhausted-item disabling.
- Bottom-right collapsible camera feeds; inspector on right only when an object is selected.
- Dragging a tray card shows a translucent placement preview that tracks the pointer and snaps to table height; red preview for invalid/out-of-bounds/overlapping placement. Click-card then click-table is also supported. Dropping outside cancels. Backend validates again on spawn.
- Selection is one source of truth: selectedName: string|null. Click object selects; click empty space/Escape deselects; distinguish clicks from orbit/drag. Every object remains selectable independent of spawn order and gizmo history.
- Edit selected objects via a ghost transform target: only the ghost moves during manipulation; commit move on release. Support translation, yaw, table snapping, position/yaw inspector, color/size when supported, and delete button/Delete key. Rejection restores the ghost and shows an error toast. Other physics continues while editing.
- Provide clear connection, pool exhaustion, validation, server, and policy-load errors.

MILESTONES (stop at the milestone explicitly assigned)
0. Discovery/design: inspect asset provenance/license, exact joint and actuator names/order/ranges, camera feasibility, installed tool versions, and checkpoint metadata/schema availability. Write a short architecture and risk note. Do not download model weights unless necessary and explicitly sized/controlled.
1. Foundation: clean package layout, reproducible setup/entry point, scene compilation, robot/table/floor/light/cameras, Sim lock and fixed-step lifecycle, basic stability/reset tests.
2. Observation plumbing: named camera rendering and copied image observations; documented coordinate and robot state conventions; mock observation consumer. Confirm rendering thread behavior.
3. Object system: preallocated primitive pool, registry, spawn/move/delete/list, collision validation, stable physical settling, tests.
4. Web transport: FastAPI lifecycle and schemas, scene/mesh endpoints, versioned websocket frames and commands, camera feeds, structured errors, API tests.
5. Web shell: Three.js scene from metadata, pose mutation, controls, tray, prompt bar and camera panel.
6. Placement and editing: preview/backend validation, single-source selection, ghost gizmo and inspector; verify interleaved edits for multiple objects.
7. Hardening: thread-safety audit, save/load scene JSON, clear reset behavior, object fall handling, shutdown, longer soak test, README and manual verification guide.
8. Model compatibility (offline/mock first): implement and test observation/action transforms from verified checkpoint metadata, including camera names, image processing, joint mapping, gripper mapping, units and normalization. Add optional dependency extra and model-load diagnostics.
9. Controlled policy integration: load the selected checkpoint through its supported LeRobot API, run inference in an isolated worker, validate and rate-limit actions, add safe stop behavior. First test against mock policy and constrained simulated motion; record exact compatibility/evaluation limits. Never imply real-robot readiness from simulation-only tests.

QUALITY BAR
- Type hints and docstrings for public Python methods; strict TypeScript.
- No silent coordinate frame, quaternion, camera key, unit, or joint-order assumptions.
- Tests should assert meaningful behavior, including that failed/candidate operations do not corrupt state. Include API tests with renderer stubs.
- Keep documentation honest about what is tested and what remains a limitation.
- Finish each assigned milestone with changed files/features, checks run and exact outcomes, deviations/known issues, and concrete manual verification steps. Stop at the requested milestone.

BEGIN by inspecting the repository and current task instructions. Do not overwrite existing work until a restart/migration plan is explicit. The first implementation task is milestone 0 unless the user names a later milestone.
```

## 6. Source references

- Model card and schema: <https://huggingface.co/Elbarbary/so101_smolvla>
- SmolVLA paper linked by the model card: <https://arxiv.org/abs/2506.01844>
- LeRobot project/docs linked by the model card: <https://github.com/huggingface/lerobot>
- Local robot asset calibration and gripper notes: `assets/so101/README.md`
- Existing migration notes: `NOTES.md`
- Existing project rules (some stage-status text is stale): `AGENTS.md`
