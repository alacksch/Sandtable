# Policy contract and SmolVLA adapter

The simulator-facing contract stays model-independent. The optional model adapter and worker are described below and remain isolated from the base simulator dependencies.

## Current observation

`Sim.get_observation()` returns:

- `images`: owned `uint8` camera arrays keyed by the actual simulator camera names: `front`, `top`, and `wrist`; frames are HWC as returned by the MuJoCo RGB renderer. Missing frames are omitted. No camera is renamed or duplicated.
- `joint_state`: six actuator-controlled joint positions, in actuator order, in radians.
- `joint_names`: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
- `joint_units`: one `rad` unit for each current hinge joint.
- `simulation_time`: MuJoCo simulation clock in seconds, sampled when the joint state is read.
- `prompt`: current task text.

The images and joint state are safe owned copies, but camera capture and state sampling are not an atomic multimodal snapshot. A future runtime must use timestamps or a coordinated capture path if the policy requires tighter synchronization.

## Current action

An action is one finite float vector with exactly six values, in the actuator order above. Values are target positions in radians and must lie within each MuJoCo actuator's configured control range. `Sim.apply_action()` rejects wrong shapes, non-finite values, and out-of-range values; when passed an observation timestamp, it also rejects future or stale observations (default maximum age 250 ms of simulation time). `Sim.hold_arm_position()` targets the current controlled joint positions as a safe-stop primitive. Legacy `Sim.set_ctrl()` retains its clipping behavior for the existing non-policy controller path.

`get_actuator_info()` exposes actuator/joint names, units, limited flags, and control ranges through copies. `validate_action_targets()` can validate outputs before a controller forwards them.

## Offline mock

`HoldPositionPolicy` returns a copy of the observed joint state. It is deterministic and does not access MuJoCo; it is a test fixture, not a task-performing policy.

## Target checkpoint gap

The downloaded checkpoint metadata declares six state/action values, `top`, `wrist`, and `empty_camera_0` visual features at `(3, 480, 640)`, state/action mean/std normalization, and resize-with-padding to 512×512. The simulator provides real `top` and `wrist` RGB camera inputs; LeRobot's saved SmolVLA config represents the missing third view with a masked empty image. The checkpoint and dataset use the six SO-100 joint positions in degrees; Sandtable maps the verified actuator order to radians at the adapter boundary. The gripper remains a simulated hinge joint, and this mapping does not imply calibrated equivalence to hardware.

`LeRobotSmolVLA` loads lazily from `Elbarbary/so101_smolvla`; the 1,197,789,224-byte checkpoint, config, processor state, and Hub metadata are stored in ignored `.model-cache`. It uses LeRobot's saved preprocess/postprocess pipelines, resizes RGB frames to 640×480, normalizes pixels to `[0, 1]`, and consumes LeRobot's queued action chunk one action at a time. A dedicated worker holds the arm on Stop, pause, reset, manual target, worker failure, and application shutdown. The Run state reports model loading/inference errors to the UI. The optional `.venv-policy` environment is needed for the Run command to import LeRobot.

Automated tests cover the model-independent contract, action validation, degree/radian conversion, and the slow-inference worker path. A real checkpoint inference returned finite, actuator-range-safe targets on the NVIDIA GTX 1650. A live websocket run with rendered simulator cameras changed joint targets, and Stop held the pose and reported `stopped`. A 4.074-second camera inference exposed the prior fixed 2.5-second freshness limit. The worker now allows measured inference duration plus 750 ms, with an 8-second hard cap; a worker regression test confirms an action after 2.8 seconds is accepted. The integration is simulation-only and has not been evaluated for task success or hardware safety.

Before treating the output as useful policy control, run the pinned LeRobot pipeline against this checkpoint, verify the downloaded processor tensors, observe action latency/freshness on the target GPU, and validate constrained simulated behavior for each task. Never treat the current adapter as real-robot readiness.
