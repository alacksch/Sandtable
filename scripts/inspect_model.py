"""Report joints, actuators, and cameras in the official SO-101 scene."""

from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "so101" / "scene.xml"
JOINT_TYPES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}


def object_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    """Get an object's MuJoCo name, falling back to its numeric id."""
    return mujoco.mj_id2name(model, obj_type, obj_id) or f"<unnamed:{obj_id}>"


def is_position_actuator(model: mujoco.MjModel, actuator_id: int) -> bool:
    """Identify compiled MuJoCo position actuators from their affine bias."""
    if model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_AFFINE:
        return False
    kp = model.actuator_gainprm[actuator_id, 0]
    return kp != 0 and abs(model.actuator_biasprm[actuator_id, 1] + kp) < 1e-6


def main() -> None:
    """Load the model and print its joints, actuators, and cameras."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    print(f"Model: {MODEL_PATH}")
    print("\nJoints (model order):")
    for joint_id in range(model.njnt):
        name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = JOINT_TYPES.get(model.jnt_type[joint_id], "unknown")
        joint_range = (
            model.jnt_range[joint_id].tolist()
            if model.jnt_limited[joint_id]
            else "unlimited"
        )
        print(
            f"  {joint_id}: {name} | {joint_type}"
            f" | qpos={model.jnt_qposadr[joint_id]} | range={joint_range}"
        )

    print("\nActuators:")
    for actuator_id in range(model.nu):
        name = object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        actuator_type = "position" if is_position_actuator(model, actuator_id) else "general"
        ctrlrange = (
            model.actuator_ctrlrange[actuator_id].tolist()
            if model.actuator_ctrllimited[actuator_id]
            else "unlimited"
        )
        print(
            f"  {actuator_id}: {name} | {actuator_type}"
            f" | ctrlrange={ctrlrange}"
            f" | gainprm={model.actuator_gainprm[actuator_id].tolist()}"
            f" | biasprm={model.actuator_biasprm[actuator_id].tolist()}"
        )

    print("\nCameras:")
    if model.ncam == 0:
        print("  (none defined in this scene)")
    for camera_id in range(model.ncam):
        name = object_name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        print(
            f"  {camera_id}: {name} | pos={model.cam_pos[camera_id].tolist()}"
            f" | fovy={model.cam_fovy[camera_id]}"
        )


if __name__ == "__main__":
    main()
